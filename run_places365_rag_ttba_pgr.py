
import argparse
import csv
import glob
import json
import os
import random

import lpips
import numpy as np
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim_fn,
)

from models import builder
from attacks.rag_ttba_pgr import RAGTtBAPGRAttack


device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Reproducibility
# ======================================================================
def set_seed(seed=992):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Places365 victim model
# ======================================================================
def load_places365_model(
    model_name,
    device,
):
    if model_name == "resnet50":
        model = torch_models.resnet50(
            weights=None
        )
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            365,
        )
        url = (
            "http://places2.csail.mit.edu/"
            "models_places365/"
            "resnet50_places365.pth.tar"
        )

    elif model_name == "densenet161":
        model = torch_models.densenet161(
            weights=None
        )
        model.classifier = torch.nn.Linear(
            model.classifier.in_features,
            365,
        )
        url = (
            "http://places2.csail.mit.edu/"
            "models_places365/"
            "densenet161_places365.pth.tar"
        )

    else:
        raise ValueError(
            f"Unknown architecture: "
            f"{model_name}"
        )

    checkpoint = (
        torch.hub.load_state_dict_from_url(
            url,
            map_location=device,
        )
    )

    state_dict = checkpoint[
        "state_dict"
    ]

    clean_state = {}

    for k, v in state_dict.items():
        name = k.replace(
            "module.",
            "",
        )
        name = name.replace(
            "norm.1.",
            "norm1.",
        )
        name = name.replace(
            "norm.2.",
            "norm2.",
        )
        name = name.replace(
            "conv.1.",
            "conv1.",
        )
        name = name.replace(
            "conv.2.",
            "conv2.",
        )
        clean_state[name] = v

    model.load_state_dict(
        clean_state
    )

    return model.to(
        device
    ).eval()


# ======================================================================
# RAG wrapper
# ======================================================================
class LatentModelWrapper(
    torch.nn.Module
):
    """
    Phi(z) =
        clip(
            x_orig
            + decoder(z)
            - decoder(z_source),
            0, 1
        )
    """

    def __init__(
        self,
        decoder,
        classifier,
        mean,
        std,
    ):
        super().__init__()

        self.decoder = decoder
        self.classifier = classifier

        self.register_buffer(
            "mean",
            torch.tensor(
                mean
            ).view(
                1, 3, 1, 1
            ),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std
            ).view(
                1, 3, 1, 1
            ),
        )

        self.x_orig = None
        self.x_rec_0 = None

    def set_reference_images(
        self,
        x_orig,
        x_rec_0,
    ):
        self.x_orig = (
            x_orig.detach()
        )
        self.x_rec_0 = (
            x_rec_0.detach()
        )

    def get_pixel_image(
        self,
        z,
    ):
        delta = (
            self.decoder(z)
            - self.x_rec_0
        )

        return torch.clamp(
            self.x_orig + delta,
            0.0,
            1.0,
        )

    def forward(
        self,
        z,
    ):
        x = self.get_pixel_image(
            z
        )

        return self.classifier(
            (x - self.mean)
            / self.std
        )


# ======================================================================
# Strict hard-label targeted seed construction
# ======================================================================
def build_target_seed_hardlabel(
    encoder,
    decoder,
    wrapper,
    x_source,
    x_target,
    target_label,
    max_queries=80,
):
    """
    No victim-logit optimization.
    No victim gradient.
    Top-1 hard labels only.
    """
    queries = 0

    with torch.no_grad():
        z_source = encoder(
            x_source
        )
        x_rec_source = decoder(
            z_source
        )

        pseudo_target = torch.clamp(
            x_target
            - (
                x_source
                - x_rec_source
            ),
            0.0,
            1.0,
        )

        z_target = encoder(
            pseudo_target
        )

    def label_of(z):
        nonlocal queries

        if queries >= max_queries:
            return None

        with torch.no_grad():
            label = wrapper(
                z
            ).argmax(
                1
            ).item()

        queries += 1
        return label

    if (
        label_of(z_target)
        == target_label
    ):
        return (
            z_target.detach(),
            queries,
        )

    delta = (
        z_target - z_source
    )

    for factor in [
        1.10,
        1.25,
        1.50,
        2.0,
        2.5,
        3.0,
        4.0,
    ]:
        if queries >= max_queries:
            break

        cand = (
            z_source
            + factor * delta
        )

        if (
            label_of(cand)
            == target_label
        ):
            return (
                cand.detach(),
                queries,
            )

    for j in range(1, 73):
        if queries >= max_queries:
            break

        scale = 0.005 * j

        cand = (
            z_target
            + scale
            * torch.randn_like(
                z_target
            )
        )

        if (
            label_of(cand)
            == target_label
        ):
            return (
                cand.detach(),
                queries,
            )

    return None, queries


# ======================================================================
# Reuse / extend exact split
# ======================================================================
def _load_split(
    path,
    mode,
):
    pairs = []

    if not os.path.exists(path):
        return pairs

    with open(
        path,
        "r",
    ) as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if not row:
                continue

            if mode == "targeted":
                if len(row) >= 4:
                    pairs.append(
                        (
                            row[0],
                            int(row[1]),
                            row[2],
                            int(row[3]),
                        )
                    )

            else:
                if len(row) >= 2:
                    pairs.append(
                        (
                            row[0],
                            int(row[1]),
                            None,
                            None,
                        )
                    )

    return pairs


def prepare_strict_splits(
    args,
    arch,
    image_dir,
    img2label,
    classifier,
    mean_t,
    std_t,
):
    split_dir = os.path.join(
        "splits",
        "places365",
    )

    os.makedirs(
        split_dir,
        exist_ok=True,
    )

    # Use the same split already used by the previous validated runs.
    split_file = os.path.join(
        split_dir,
        f"{arch}_{args.mode}.csv",
    )

    pairs = _load_split(
        split_file,
        args.mode,
    )

    if len(pairs) >= args.num_samples:
        print(
            f"[*] Loading split from "
            f"{split_file} "
            f"({len(pairs)} available)"
        )

        return pairs[
            : args.num_samples
        ]

    print(
        f"[*] Existing split has "
        f"{len(pairs)} samples; "
        f"extending to "
        f"{args.num_samples}."
    )

    files = sorted(
        glob.glob(
            os.path.join(
                image_dir,
                "*.JPEG",
            )
        )
    )

    rng = random.Random(
        args.seed
    )
    rng.shuffle(
        files
    )

    existing = {
        row[0]
        for row in pairs
    }

    resize = transforms.Resize(
        (224, 224)
    )
    to_tensor = (
        transforms.ToTensor()
    )

    def load_img(path):
        return to_tensor(
            resize(
                Image.open(
                    path
                ).convert(
                    "RGB"
                )
            )
        ).unsqueeze(
            0
        ).to(
            device
        )

    for source_path in files:
        if (
            len(pairs)
            >= args.num_samples
        ):
            break

        source_name = (
            os.path.basename(
                source_path
            )
        )

        if source_name in existing:
            continue

        source_label = (
            img2label.get(
                source_name
            )
        )

        if source_label is None:
            continue

        x_source = load_img(
            source_path
        )

        with torch.no_grad():
            pred = classifier(
                (
                    x_source
                    - mean_t
                )
                / std_t
            ).argmax(
                1
            ).item()

        if pred != source_label:
            continue

        if args.mode == "non_targeted":
            pairs.append(
                (
                    source_name,
                    source_label,
                    None,
                    None,
                )
            )

            existing.add(
                source_name
            )
            continue

        targets = list(
            files
        )
        rng.shuffle(
            targets
        )

        for target_path in targets:
            target_name = (
                os.path.basename(
                    target_path
                )
            )

            target_label = (
                img2label.get(
                    target_name
                )
            )

            if (
                target_label is None
                or target_label
                == source_label
            ):
                continue

            x_target = load_img(
                target_path
            )

            with torch.no_grad():
                pred_target = classifier(
                    (
                        x_target
                        - mean_t
                    )
                    / std_t
                ).argmax(
                    1
                ).item()

            if (
                pred_target
                == target_label
            ):
                pairs.append(
                    (
                        source_name,
                        source_label,
                        target_name,
                        target_label,
                    )
                )

                existing.add(
                    source_name
                )
                break

    if len(pairs) < args.num_samples:
        raise RuntimeError(
            f"Only {len(pairs)} "
            f"valid samples found; "
            f"requested "
            f"{args.num_samples}."
        )

    with open(
        split_file,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        if args.mode == "targeted":
            writer.writerow(
                [
                    "source_image",
                    "source_label",
                    "target_image",
                    "target_label",
                ]
            )

            for row in pairs:
                writer.writerow(
                    [
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                    ]
                )

        else:
            writer.writerow(
                [
                    "source_image",
                    "source_label",
                ]
            )

            for row in pairs:
                writer.writerow(
                    [
                        row[0],
                        row[1],
                    ]
                )

    return pairs[
        : args.num_samples
    ]


# ======================================================================
# Save numeric outputs only
# ======================================================================
def save_method_outputs(
    out_dir,
    rows,
    checkpoints,
    thresholds,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    n = len(rows)
    c = len(checkpoints)

    def stack_float(key):
        if n == 0:
            return np.empty(
                (0, c),
                dtype=np.float64,
            )

        return np.stack(
            [
                row[key]
                for row in rows
            ],
            axis=0,
        ).astype(
            np.float64
        )

    def stack_int(key):
        if n == 0:
            return np.empty(
                (0, c),
                dtype=np.int64,
            )

        return np.stack(
            [
                row[key]
                for row in rows
            ],
            axis=0,
        ).astype(
            np.int64
        )

    arrays = {
        "checkpoints": np.asarray(
            checkpoints,
            dtype=np.int64,
        ),
        "state_query": stack_int(
            "state_query"
        ),
        "l2_lat": stack_float(
            "l2_lat"
        ),
        "l2_pix": stack_float(
            "l2_pix"
        ),
        "ssim": stack_float(
            "ssim"
        ),
        "lpips": stack_float(
            "lpips"
        ),
        "asr": stack_float(
            "asr"
        ),
        "best_query": np.asarray(
            [
                row["best_query"]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "budget_consumed": np.asarray(
            [
                row["budget_consumed"]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "final_l2_lat": np.asarray(
            [
                row["final_l2_lat"]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_l2_pix": np.asarray(
            [
                row["final_l2_pix"]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_ssim": np.asarray(
            [
                row["final_ssim"]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_lpips": np.asarray(
            [
                row["final_lpips"]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "q_init": np.asarray(
            [
                row["q_init"]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_normal": np.asarray(
            [
                row["q_normal"]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_bridge": np.asarray(
            [
                row["q_bridge"]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_boundary": np.asarray(
            [
                row["q_boundary"]
                for row in rows
            ],
            dtype=np.int64,
        ),
    }

    for threshold in thresholds:
        safe = str(
            threshold
        ).replace(
            ".",
            "p",
        )

        arrays[
            f"q_to_l2_{safe}"
        ] = np.asarray(
            [
                row["thresholds"][
                    float(threshold)
                ]
                for row in rows
            ],
            dtype=np.int64,
        )

    np.savez(
        os.path.join(
            out_dir,
            "metrics_numeric.npz",
        ),
        **arrays,
    )

    with open(
        os.path.join(
            out_dir,
            "checkpoint_summary.csv",
        ),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "sample_index",
                "checkpoint",
                "state_query",
                "l2_lat",
                "l2_pix",
                "ssim",
                "lpips",
                "asr",
            ]
        )

        for i, row in enumerate(
            rows
        ):
            for j, cp in enumerate(
                checkpoints
            ):
                writer.writerow(
                    [
                        i,
                        cp,
                        int(
                            row[
                                "state_query"
                            ][j]
                        ),
                        float(
                            row[
                                "l2_lat"
                            ][j]
                        ),
                        float(
                            row[
                                "l2_pix"
                            ][j]
                        ),
                        float(
                            row[
                                "ssim"
                            ][j]
                        ),
                        float(
                            row[
                                "lpips"
                            ][j]
                        ),
                        float(
                            row[
                                "asr"
                            ][j]
                        ),
                    ]
                )


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "targeted",
            "non_targeted",
        ],
        required=True,
    )

    parser.add_argument(
        "--target_archs",
        nargs="+",
        default=[
            "resnet50"
        ],
    )

    parser.add_argument(
        "--attack_methods",
        nargs="+",
        default=[
            "RAG_TTBA_BASE",
            "RAG_TTBA_PGR_GATE",
            "RAG_TTBA_PGR_ADAPT",
            "RAG_TTBA_PGR",
        ],
    )

    parser.add_argument(
        "--ae_checkpoint",
        required=True,
    )

    parser.add_argument(
        "--base_dir",
        default="./data/Places365",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_queries",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--query_checkpoints",
        nargs="+",
        type=int,
        default=[
            500,
            1000,
            2000,
        ],
    )

    parser.add_argument(
        "--l2_thresholds",
        nargs="+",
        type=float,
        default=[
            10.0,
            5.0,
            3.0,
            2.5,
            2.0,
        ],
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--initial_query",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--ttba_bridge_tol",
        type=float,
        default=0.001,
    )

    # PGR Step 1
    parser.add_argument(
        "--pgr_progress_threshold",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--pgr_max_reuse_streak",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--pgr_accept_eps",
        type=float,
        default=1e-8,
    )

    # PGR Step 2
    parser.add_argument(
        "--pgr_refresh_min_ratio",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--pgr_refresh_power",
        type=float,
        default=1.0,
    )

    # PGR Step 3
    parser.add_argument(
        "--pgr_history_alpha_max",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--update_query_reserve",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--target_seed_max_queries",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=992,
    )

    parser.add_argument(
        "--verbose",
        choices=[
            "Yes",
            "No",
        ],
        default="No",
    )

    args = parser.parse_args()

    bad = sorted(
        set(
            args.attack_methods
        )
        - RAGTtBAPGRAttack.SUPPORTED
    )

    if bad:
        raise ValueError(
            f"Unsupported methods: {bad}. "
            f"Supported="
            f"{sorted(RAGTtBAPGRAttack.SUPPORTED)}"
        )

    set_seed(
        args.seed
    )

    image_dir = os.path.join(
        args.base_dir,
        "images",
    )

    labels_path = os.path.join(
        args.base_dir,
        "adv_test_labels.txt",
    )

    img2label = {}

    with open(
        labels_path,
        "r",
    ) as f:
        for line in f:
            parts = (
                line.strip().split()
            )

            if len(parts) >= 2:
                img2label[
                    parts[0]
                ] = int(
                    parts[1]
                )

    mean = [
        0.485,
        0.456,
        0.406,
    ]

    std = [
        0.229,
        0.224,
        0.225,
    ]

    mean_t = torch.tensor(
        mean
    ).view(
        1, 3, 1, 1
    ).to(
        device
    )

    std_t = torch.tensor(
        std
    ).view(
        1, 3, 1, 1
    ).to(
        device
    )

    class MockArgs:
        arch = "vgg16"
        parallel = 0
        gpu = 0

    ae_model = (
        builder.BuildAutoEncoder(
            MockArgs()
        )
    )

    checkpoint = torch.load(
        args.ae_checkpoint,
        map_location=device,
    )

    state_dict = (
        checkpoint[
            "state_dict"
        ]
        if "state_dict"
        in checkpoint
        else checkpoint
    )

    aligned = {
        (
            "module."
            + k.replace(
                "module.",
                "",
            )
        ): v
        for k, v
        in state_dict.items()
    }

    ae_model.load_state_dict(
        aligned
    )

    ae_model = ae_model.to(
        device
    ).eval()

    if isinstance(
        ae_model,
        torch.nn.DataParallel,
    ):
        encoder = (
            ae_model.module.encoder
        )
        decoder = (
            ae_model.module.decoder
        )
    else:
        encoder = (
            ae_model.encoder
        )
        decoder = (
            ae_model.decoder
        )

    loss_fn_vgg = lpips.LPIPS(
        net="vgg"
    ).to(
        device
    )

    resize = transforms.Resize(
        (224, 224)
    )

    to_tensor = (
        transforms.ToTensor()
    )

    def load_rgb(path):
        return to_tensor(
            resize(
                Image.open(
                    path
                ).convert(
                    "RGB"
                )
            )
        ).unsqueeze(
            0
        ).to(
            device
        )

    for arch in args.target_archs:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "RAG-TtBA PGR ABLATION — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )
        print(
            "=" * 80
        )

        classifier = (
            load_places365_model(
                arch,
                device,
            )
        )

        wrapper = (
            LatentModelWrapper(
                decoder,
                classifier,
                mean,
                std,
            )
            .to(device)
            .eval()
        )

        pairs = prepare_strict_splits(
            args,
            arch,
            image_dir,
            img2label,
            classifier,
            mean_t,
            std_t,
        )

        rows_by_method = {
            method: []
            for method
            in args.attack_methods
        }

        for sample_index, pair in enumerate(
            pairs
        ):
            (
                source_name,
                source_label,
                target_name,
                target_label,
            ) = pair

            x_source = load_rgb(
                os.path.join(
                    image_dir,
                    source_name,
                )
            )

            with torch.no_grad():
                z_source = encoder(
                    x_source
                )

                x_rec_source = decoder(
                    z_source
                )

                wrapper.set_reference_images(
                    x_source,
                    x_rec_source,
                )

            target_seed = None
            target_seed_queries = 0

            if args.mode == "targeted":
                x_target = load_rgb(
                    os.path.join(
                        image_dir,
                        target_name,
                    )
                )

                set_seed(
                    args.seed
                    + sample_index
                )

                (
                    target_seed,
                    target_seed_queries,
                ) = build_target_seed_hardlabel(
                    encoder=encoder,
                    decoder=decoder,
                    wrapper=wrapper,
                    x_source=x_source,
                    x_target=x_target,
                    target_label=target_label,
                    max_queries=args.target_seed_max_queries,
                )

                if target_seed is None:
                    print(
                        f"[!] Skip "
                        f"{source_name}: "
                        f"hard-label target "
                        f"seed failed."
                    )
                    continue

            print(
                f"\n"
                f"[{sample_index + 1}/"
                f"{len(pairs)}] "
                f"{source_name}"
            )

            for method in args.attack_methods:
                print(
                    f"\n --- {method} ---"
                )

                # Same initial randomness for each method on the same image.
                set_seed(
                    args.seed
                    + sample_index
                )

                attack = (
                    RAGTtBAPGRAttack(
                        model=wrapper,
                        z_source=z_source,
                        x_orig=x_source,
                        source_label=source_label,
                        loss_fn_vgg=loss_fn_vgg,
                        ssim_fn=ssim_fn,
                        attack_method=method,
                        target_seed=target_seed,
                        target_label=target_label,
                        iteration=args.iterations,
                        initial_query=args.initial_query,
                        tol=args.tol,
                        sigma=args.sigma,
                        seed=(
                            args.seed
                            + sample_index
                        ),
                        query_checkpoints=args.query_checkpoints,
                        max_queries=args.max_queries,
                        verbose=args.verbose,
                        initial_query_offset=target_seed_queries,
                        ttba_bridge_tol=args.ttba_bridge_tol,
                        update_query_reserve=args.update_query_reserve,
                        pgr_progress_threshold=args.pgr_progress_threshold,
                        pgr_max_reuse_streak=args.pgr_max_reuse_streak,
                        pgr_accept_eps=args.pgr_accept_eps,
                        pgr_refresh_min_ratio=args.pgr_refresh_min_ratio,
                        pgr_refresh_power=args.pgr_refresh_power,
                        pgr_history_alpha_max=args.pgr_history_alpha_max,
                        l2_thresholds=args.l2_thresholds,
                    )
                )

                result = attack.attack()

                if (
                    not result["success"]
                    or result["final"]
                    is None
                ):
                    print(
                        "   [FAIL] "
                        "attack initialization "
                        "failed."
                    )
                    continue

                final = result[
                    "final"
                ]

                qb = result[
                    "query_breakdown"
                ]

                cp = result[
                    "checkpoints"
                ]

                print(
                    f"   BEST Q="
                    f"{final['query']} "
                    f"(used="
                    f"{final['budget_consumed']}) "
                    f"PixL2="
                    f"{final['l2_pix']:.4f} "
                    f"LatL2="
                    f"{final['l2_lat']:.4f} "
                    f"SSIM="
                    f"{final['ssim']:.5f} "
                    f"LPIPS="
                    f"{final['lpips']:.5f}"
                )

                print(
                    "   Q: "
                    f"init={qb['init']} "
                    f"normal={qb['normal']} "
                    f"bridge={qb['bridge']} "
                    f"boundary={qb['boundary']}"
                )

                rows_by_method[
                    method
                ].append(
                    {
                        "state_query": cp[
                            "state_query"
                        ],
                        "l2_lat": cp[
                            "l2_lat"
                        ],
                        "l2_pix": cp[
                            "l2_pix"
                        ],
                        "ssim": cp[
                            "ssim"
                        ],
                        "lpips": cp[
                            "lpips"
                        ],
                        "asr": cp[
                            "asr"
                        ],
                        "best_query": int(
                            final[
                                "query"
                            ]
                        ),
                        "budget_consumed": int(
                            final[
                                "budget_consumed"
                            ]
                        ),
                        "final_l2_lat": float(
                            final[
                                "l2_lat"
                            ]
                        ),
                        "final_l2_pix": float(
                            final[
                                "l2_pix"
                            ]
                        ),
                        "final_ssim": float(
                            final[
                                "ssim"
                            ]
                        ),
                        "final_lpips": float(
                            final[
                                "lpips"
                            ]
                        ),
                        "q_init": int(
                            qb["init"]
                        ),
                        "q_normal": int(
                            qb["normal"]
                        ),
                        "q_bridge": int(
                            qb["bridge"]
                        ),
                        "q_boundary": int(
                            qb["boundary"]
                        ),
                        "thresholds": result[
                            "thresholds"
                        ],
                        "diagnostics": result[
                            "diagnostics"
                        ],
                    }
                )

        root = os.path.join(
            "Results",
            "RAG_TTBA_PGR",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        summary_path = os.path.join(
            root,
            "final_summary.csv",
        )

        threshold_path = os.path.join(
            root,
            "threshold_summary.csv",
        )

        with open(
            summary_path,
            "w",
            newline="",
        ) as fs, open(
            threshold_path,
            "w",
            newline="",
        ) as ft:
            sw = csv.writer(
                fs
            )

            tw = csv.writer(
                ft
            )

            sw.writerow(
                [
                    "method",
                    "n",
                    "median_best_query",
                    "median_budget_consumed",
                    "median_final_l2_lat",
                    "median_final_l2_pix",
                    "mean_final_ssim",
                    "mean_final_lpips",
                    "sum_q_init",
                    "sum_q_normal",
                    "sum_q_bridge",
                    "sum_q_boundary",
                ]
            )

            tw.writerow(
                [
                    "method",
                    "l2_threshold",
                    "hit_count",
                    "hit_rate",
                    "median_query_among_hits",
                ]
            )

            print(
                "\n"
                + "=" * 80
            )
            print(
                "PGR ABLATION SUMMARY"
            )
            print(
                "=" * 80
            )

            for method in args.attack_methods:
                rows = rows_by_method[
                    method
                ]

                if not rows:
                    continue

                method_dir = (
                    os.path.join(
                        root,
                        method,
                    )
                )

                save_method_outputs(
                    method_dir,
                    rows,
                    args.query_checkpoints,
                    args.l2_thresholds,
                )

                with open(
                    os.path.join(
                        method_dir,
                        "diagnostics.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        [
                            row[
                                "diagnostics"
                            ]
                            for row
                            in rows
                        ],
                        f,
                        indent=2,
                    )

                pix = np.asarray(
                    [
                        row[
                            "final_l2_pix"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                lat = np.asarray(
                    [
                        row[
                            "final_l2_lat"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                ssim_arr = np.asarray(
                    [
                        row[
                            "final_ssim"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                lpips_arr = np.asarray(
                    [
                        row[
                            "final_lpips"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                best_q = np.asarray(
                    [
                        row[
                            "best_query"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                consumed = np.asarray(
                    [
                        row[
                            "budget_consumed"
                        ]
                        for row
                        in rows
                    ],
                    dtype=float,
                )

                q_init = int(
                    sum(
                        row[
                            "q_init"
                        ]
                        for row
                        in rows
                    )
                )

                q_normal = int(
                    sum(
                        row[
                            "q_normal"
                        ]
                        for row
                        in rows
                    )
                )

                q_bridge = int(
                    sum(
                        row[
                            "q_bridge"
                        ]
                        for row
                        in rows
                    )
                )

                q_boundary = int(
                    sum(
                        row[
                            "q_boundary"
                        ]
                        for row
                        in rows
                    )
                )

                print(
                    f"{method:<24} "
                    f"PixL2med="
                    f"{np.median(pix):.4f} "
                    f"LatL2med="
                    f"{np.median(lat):.4f} "
                    f"SSIM="
                    f"{np.mean(ssim_arr):.5f} "
                    f"LPIPS="
                    f"{np.mean(lpips_arr):.5f} "
                    f"Qnormal="
                    f"{q_normal}"
                )

                sw.writerow(
                    [
                        method,
                        len(rows),
                        float(
                            np.median(
                                best_q
                            )
                        ),
                        float(
                            np.median(
                                consumed
                            )
                        ),
                        float(
                            np.median(
                                lat
                            )
                        ),
                        float(
                            np.median(
                                pix
                            )
                        ),
                        float(
                            np.mean(
                                ssim_arr
                            )
                        ),
                        float(
                            np.mean(
                                lpips_arr
                            )
                        ),
                        q_init,
                        q_normal,
                        q_bridge,
                        q_boundary,
                    ]
                )

                for threshold in args.l2_thresholds:
                    hits = [
                        row[
                            "thresholds"
                        ][
                            float(
                                threshold
                            )
                        ]
                        for row
                        in rows
                    ]

                    valid = [
                        q
                        for q in hits
                        if q >= 0
                    ]

                    hit_count = len(
                        valid
                    )

                    hit_rate = (
                        hit_count
                        / len(rows)
                        if rows
                        else 0.0
                    )

                    median_query = (
                        float(
                            np.median(
                                valid
                            )
                        )
                        if valid
                        else -1.0
                    )

                    tw.writerow(
                        [
                            method,
                            float(
                                threshold
                            ),
                            hit_count,
                            hit_rate,
                            median_query,
                        ]
                    )

        print(
            f"\n[*] Summary: "
            f"{summary_path}"
        )

        print(
            f"[*] Thresholds: "
            f"{threshold_path}"
        )

        del wrapper
        del classifier

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
