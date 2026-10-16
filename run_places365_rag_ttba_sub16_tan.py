
import argparse
import csv
import json
import os

import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim_fn,
)

from models import builder

from attacks.rag_ttba_sub16_tan import (
    RAGTtBASub16TanAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["targeted", "non_targeted"],
        required=True,
    )

    parser.add_argument(
        "--target_archs",
        nargs="+",
        default=["resnet50"],
    )

    parser.add_argument(
        "--attack_methods",
        nargs="+",
        default=[
            "RAG_TTBA_BASE",
            "RAG_TTBA_SUB16_TAN",
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
        default=[500, 1000, 2000],
    )

    parser.add_argument(
        "--l2_thresholds",
        nargs="+",
        type=float,
        default=[10.0, 5.0, 3.0, 2.5, 2.0],
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

    parser.add_argument(
        "--update_query_reserve",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--subspace_dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--subspace_basis_seed",
        type=int,
        default=4992,
    )

    parser.add_argument(
        "--subspace_probe_seed",
        type=int,
        default=5992,
    )

    parser.add_argument(
        "--tangent_eps",
        type=float,
        default=1e-10,
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
        choices=["Yes", "No"],
        default="Yes",
    )

    args = parser.parse_args()

    unsupported = sorted(
        set(args.attack_methods)
        - RAGTtBASub16TanAttack.SUPPORTED
    )

    if unsupported:
        raise ValueError(
            f"Unsupported attack methods: {unsupported}. "
            f"Supported={sorted(RAGTtBASub16TanAttack.SUPPORTED)}"
        )

    if args.subspace_dim < 2:
        raise ValueError(
            "--subspace_dim must be >= 2."
        )

    set_seed(args.seed)

    image_dir = os.path.join(
        args.base_dir,
        "images",
    )

    labels_path = os.path.join(
        args.base_dir,
        "adv_test_labels.txt",
    )

    img2label = {}

    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img2label[parts[0]] = int(parts[1])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean_t = torch.tensor(
        mean
    ).view(
        1, 3, 1, 1
    ).to(device)

    std_t = torch.tensor(
        std
    ).view(
        1, 3, 1, 1
    ).to(device)

    class MockArgs:
        arch = "vgg16"
        parallel = 0
        gpu = 0

    ae_model = builder.BuildAutoEncoder(
        MockArgs()
    )

    checkpoint = torch.load(
        args.ae_checkpoint,
        map_location=device,
    )

    state_dict = (
        checkpoint["state_dict"]
        if "state_dict" in checkpoint
        else checkpoint
    )

    aligned = {
        (
            "module."
            + key.replace(
                "module.",
                "",
            )
        ): value
        for key, value in state_dict.items()
    }

    ae_model.load_state_dict(aligned)

    ae_model = ae_model.to(
        device
    ).eval()

    if isinstance(
        ae_model,
        torch.nn.DataParallel,
    ):
        encoder = ae_model.module.encoder
        decoder = ae_model.module.decoder
    else:
        encoder = ae_model.encoder
        decoder = ae_model.decoder

    loss_fn_vgg = lpips.LPIPS(
        net="vgg"
    ).to(device)

    resize = transforms.Resize(
        (224, 224)
    )

    to_tensor = transforms.ToTensor()

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
        ).to(device)

    for arch in args.target_archs:
        print()
        print("=" * 104)
        print(
            "RAG-TtBA SUB16 EXACT-TANGENT MATCHED-BUDGET TEST — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )
        print("=" * 104)

        classifier = load_places365_model(
            arch,
            device,
        )

        wrapper = LatentModelWrapper(
            decoder,
            classifier,
            mean,
            std,
        ).to(
            device
        ).eval()

        pairs = prepare_strict_splits(
            args,
            arch,
            image_dir,
            img2label,
            classifier,
            mean_t,
            std_t,
        )

        results_by_method = {
            method: []
            for method in args.attack_methods
        }

        for sample_index, pair in enumerate(pairs):
            (
                source_name,
                source_label,
                target_name,
                target_label,
            ) = pair

            print()
            print(
                f"[{sample_index + 1}/{len(pairs)}] "
                f"{source_name}"
            )

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
                    args.seed + sample_index
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
                        "[!] Hard-label target seed failed; "
                        "skipping sample."
                    )
                    continue

            for method in args.attack_methods:
                print()
                print(
                    f" --- {method} ---"
                )

                # Critical fairness reset.
                set_seed(
                    args.seed + sample_index
                )

                attack = RAGTtBASub16TanAttack(
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
                        args.seed + sample_index
                    ),
                    query_checkpoints=args.query_checkpoints,
                    max_queries=args.max_queries,
                    verbose=args.verbose,
                    initial_query_offset=target_seed_queries,
                    ttba_bridge_tol=args.ttba_bridge_tol,
                    update_query_reserve=args.update_query_reserve,
                    l2_thresholds=args.l2_thresholds,
                    subspace_dim=args.subspace_dim,
                    subspace_basis_seed=(
                        args.subspace_basis_seed
                        + sample_index
                    ),
                    subspace_probe_seed=(
                        args.subspace_probe_seed
                        + 10000 * sample_index
                    ),
                    tangent_eps=args.tangent_eps,
                )

                result = attack.attack()

                if (
                    not result["success"]
                    or result["final"] is None
                ):
                    print(
                        "   [FAIL] attack initialization/optimization failed."
                    )
                    continue

                final = result["final"]
                qb = result["query_breakdown"]

                tan_summary = result.get(
                    "tangent_projection_summary",
                    {},
                )

                sub_summary = result.get(
                    "subspace_normal_summary",
                    {},
                )

                print(
                    f"   BEST Q={final['query']} "
                    f"(used={final['budget_consumed']}) "
                    f"PixL2={final['l2_pix']:.4f} "
                    f"LatL2={final['l2_lat']:.4f} "
                    f"SSIM={final['ssim']:.5f} "
                    f"LPIPS={final['lpips']:.5f}"
                )

                print(
                    "   Q: "
                    f"init={qb.get('init', 0)} "
                    f"normal={qb.get('normal', 0)} "
                    f"bridge={qb.get('bridge', 0)} "
                    f"boundary={qb.get('boundary', 0)}"
                )

                if (
                    method
                    == RAGTtBASub16TanAttack.METHOD_TAN
                ):
                    print(
                        "   TAN summary: "
                        f"updates={tan_summary.get('projection_updates', 0)} "
                        f"meanRawAlpha="
                        f"{tan_summary.get('mean_raw_alpha')} "
                        f"medianRawAlpha="
                        f"{tan_summary.get('median_raw_alpha')} "
                        f"meanRawPerp="
                        f"{tan_summary.get('mean_raw_perp_norm')} "
                        f"max|tanDotV|="
                        f"{tan_summary.get('max_abs_tan_dot_v')}"
                    )

                record = {
                    "sample_index": int(sample_index),
                    "source_name": source_name,
                    "source_label": int(source_label),
                    "method": method,
                    "final": final,
                    "query_breakdown": dict(qb),
                    "checkpoints": result["checkpoints"],
                    "thresholds": result["thresholds"],
                    "diagnostics": result.get(
                        "diagnostics",
                        [],
                    ),
                    "subspace_normal_diagnostics":
                        result.get(
                            "subspace_normal_diagnostics",
                            [],
                        ),
                    "subspace_normal_summary":
                        sub_summary,
                    "tangent_projection_diagnostics":
                        result.get(
                            "tangent_projection_diagnostics",
                            [],
                        ),
                    "tangent_projection_summary":
                        tan_summary,
                }

                results_by_method[
                    method
                ].append(
                    record
                )

        root = os.path.join(
            "Results",
            "RAG_TTBA_SUB16_TAN",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        final_summary_rows = []
        checkpoint_rows_all = []
        threshold_rows = []

        print()
        print("=" * 104)
        print(
            "RAG-TtBA SUB16 EXACT-TANGENT MATCHED-BUDGET SUMMARY"
        )
        print("=" * 104)

        for method in args.attack_methods:
            records = results_by_method[
                method
            ]

            if not records:
                continue

            method_dir = os.path.join(
                root,
                method,
            )

            os.makedirs(
                method_dir,
                exist_ok=True,
            )

            with open(
                os.path.join(
                    method_dir,
                    "results.json",
                ),
                "w",
            ) as f:
                json.dump(
                    records,
                    f,
                    indent=2,
                )

            pix = np.asarray(
                [
                    row["final"]["l2_pix"]
                    for row in records
                ],
                dtype=float,
            )

            lat = np.asarray(
                [
                    row["final"]["l2_lat"]
                    for row in records
                ],
                dtype=float,
            )

            ssim_arr = np.asarray(
                [
                    row["final"]["ssim"]
                    for row in records
                ],
                dtype=float,
            )

            lpips_arr = np.asarray(
                [
                    row["final"]["lpips"]
                    for row in records
                ],
                dtype=float,
            )

            q_normal = int(
                sum(
                    row[
                        "query_breakdown"
                    ].get(
                        "normal",
                        0,
                    )
                    for row in records
                )
            )

            q_bridge = int(
                sum(
                    row[
                        "query_breakdown"
                    ].get(
                        "bridge",
                        0,
                    )
                    for row in records
                )
            )

            q_boundary = int(
                sum(
                    row[
                        "query_breakdown"
                    ].get(
                        "boundary",
                        0,
                    )
                    for row in records
                )
            )

            q_init = int(
                sum(
                    row[
                        "query_breakdown"
                    ].get(
                        "init",
                        0,
                    )
                    for row in records
                )
            )

            tan_alpha_values = []

            if (
                method
                == RAGTtBASub16TanAttack.METHOD_TAN
            ):
                for row in records:
                    value = row[
                        "tangent_projection_summary"
                    ].get(
                        "mean_raw_alpha"
                    )

                    if (
                        value is not None
                        and np.isfinite(
                            value
                        )
                    ):
                        tan_alpha_values.append(
                            float(value)
                        )

            mean_raw_alpha = (
                None
                if not tan_alpha_values
                else float(
                    np.mean(
                        tan_alpha_values
                    )
                )
            )

            print(
                f"{method:<22} "
                f"PixL2med={np.median(pix):.4f} "
                f"LatL2med={np.median(lat):.4f} "
                f"SSIM={np.mean(ssim_arr):.5f} "
                f"LPIPS={np.mean(lpips_arr):.5f} "
                f"Qnormal={q_normal} "
                f"Qbridge={q_bridge} "
                f"Qboundary={q_boundary}"
                + (
                    ""
                    if mean_raw_alpha is None
                    else (
                        f" meanRawAlpha="
                        f"{mean_raw_alpha:.4f}"
                    )
                )
            )

            final_summary_rows.append(
                {
                    "method": method,
                    "n": int(len(records)),
                    "median_final_l2_pix":
                        float(np.median(pix)),
                    "median_final_l2_lat":
                        float(np.median(lat)),
                    "mean_final_ssim":
                        float(np.mean(ssim_arr)),
                    "mean_final_lpips":
                        float(np.mean(lpips_arr)),
                    "sum_q_init": int(q_init),
                    "sum_q_normal": int(q_normal),
                    "sum_q_bridge": int(q_bridge),
                    "sum_q_boundary": int(q_boundary),
                    "mean_raw_alpha_before_tangent_projection":
                        mean_raw_alpha,
                }
            )

            for row in records:
                cp = row["checkpoints"]

                for j, checkpoint in enumerate(
                    args.query_checkpoints
                ):
                    checkpoint_rows_all.append(
                        {
                            "method": method,
                            "sample_index":
                                row["sample_index"],
                            "source_name":
                                row["source_name"],
                            "checkpoint":
                                int(checkpoint),
                            "state_query":
                                int(
                                    cp["state_query"][j]
                                ),
                            "l2_lat":
                                float(
                                    cp["l2_lat"][j]
                                ),
                            "l2_pix":
                                float(
                                    cp["l2_pix"][j]
                                ),
                            "ssim":
                                float(
                                    cp["ssim"][j]
                                ),
                            "lpips":
                                float(
                                    cp["lpips"][j]
                                ),
                            "asr":
                                float(
                                    cp["asr"][j]
                                ),
                        }
                    )

            for threshold in args.l2_thresholds:
                hits = [
                    row[
                        "thresholds"
                    ][
                        float(threshold)
                    ]
                    for row in records
                ]

                valid_hits = [
                    int(q)
                    for q in hits
                    if int(q) >= 0
                ]

                threshold_rows.append(
                    {
                        "method": method,
                        "l2_threshold":
                            float(threshold),
                        "hit_count":
                            int(len(valid_hits)),
                        "hit_rate":
                            float(
                                len(valid_hits)
                                / len(records)
                            ),
                        "median_query_among_hits":
                            (
                                -1.0
                                if not valid_hits
                                else float(
                                    np.median(
                                        valid_hits
                                    )
                                )
                            ),
                    }
                )

        write_csv(
            os.path.join(
                root,
                "final_summary.csv",
            ),
            final_summary_rows,
        )

        write_csv(
            os.path.join(
                root,
                "checkpoint_summary.csv",
            ),
            checkpoint_rows_all,
        )

        write_csv(
            os.path.join(
                root,
                "threshold_summary.csv",
            ),
            threshold_rows,
        )

        print()
        print(
            f"[*] Results root: {root}"
        )

        del wrapper
        del classifier

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
