
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

from attacks.rag_ttba_qsfd_compare import (
    RAGTtBAQSFDCompareAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if torch.is_tensor(obj):
        value = obj.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            json_safe(v)
            for v in obj
        ]

    return obj


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


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
            "RAG_QSFD",
            "RAG_TTBA_BASE",
            "RAG_TTBA_QSFD",
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

    parser.add_argument(
        "--update_query_reserve",
        type=int,
        default=18,
    )

    # QS-FD
    parser.add_argument(
        "--qsfd_k",
        type=int,
        default=3,
        help=(
            "Fixed number of QS-FD candidate rays for the first mechanism test."
        ),
    )

    parser.add_argument(
        "--qsfd_sector_deg",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--qsfd_coarse_steps",
        type=int,
        default=6,
        help=(
            "Coarse radial binary-search steps per candidate before choosing "
            "the winner. Only the winner gets full refinement."
        ),
    )

    parser.add_argument(
        "--qsfd_expand_factors",
        nargs="+",
        type=float,
        default=[
            1.0,
            1.25,
            1.50,
        ],
    )

    parser.add_argument(
        "--qsfd_combo_span",
        type=float,
        default=0.25,
        help=(
            "Local TtBA-k search span. 0.25 means approximately "
            "k_app*(1 +/- 0.25), clipped to [0,L]."
        ),
    )

    parser.add_argument(
        "--qsfd_k_mode",
        choices=[
            "fixed",
            "adaptive",
        ],
        default="fixed",
        help=(
            "Use fixed for the first fair comparison. "
            "Adaptive mode is provided for later experiments."
        ),
    )

    parser.add_argument(
        "--qsfd_k_early",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--qsfd_k_late",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--qsfd_switch_ssim",
        type=float,
        default=0.995,
    )

    parser.add_argument(
        "--qsfd_accept_eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--qsfd_min_post_normal_reserve",
        type=int,
        default=12,
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
        default="Yes",
    )

    args = parser.parse_args()

    unsupported = sorted(
        set(
            args.attack_methods
        )
        - RAGTtBAQSFDCompareAttack.SUPPORTED_COMPARE
    )

    if unsupported:
        raise ValueError(
            f"Unsupported methods: {unsupported}. "
            f"Supported="
            f"{sorted(RAGTtBAQSFDCompareAttack.SUPPORTED_COMPARE)}"
        )

    if args.qsfd_k < 2:
        raise ValueError(
            "--qsfd_k must be >= 2."
        )

    if args.qsfd_coarse_steps < 1:
        raise ValueError(
            "--qsfd_coarse_steps must be >= 1."
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

            if len(
                parts
            ) >= 2:
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
        1,
        3,
        1,
        1,
    ).to(
        device
    )

    std_t = torch.tensor(
        std
    ).view(
        1,
        3,
        1,
        1,
    ).to(
        device
    )

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
            + key.replace(
                "module.",
                "",
            )
        ): value
        for key, value
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
        (
            224,
            224,
        )
    )

    to_tensor = transforms.ToTensor()

    def load_rgb(
        path,
    ):
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
        print()
        print(
            "=" * 110
        )
        print(
            "QS-FD vs TtBA vs TtBA+QS-FD — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )
        print(
            "=" * 110
        )

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

        for sample_index, pair in enumerate(
            pairs
        ):
            (
                source_name,
                source_label,
                target_name,
                target_label,
            ) = pair

            print()
            print(
                f"[{sample_index + 1}/"
                f"{len(pairs)}] "
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
                    max_queries=
                        args.target_seed_max_queries,
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

                # Fair deterministic reset before EVERY method.
                set_seed(
                    args.seed
                    + sample_index
                )

                attack = RAGTtBAQSFDCompareAttack(
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
                    query_checkpoints=
                        args.query_checkpoints,
                    max_queries=args.max_queries,
                    verbose=args.verbose,
                    initial_query_offset=
                        target_seed_queries,
                    ttba_bridge_tol=
                        args.ttba_bridge_tol,
                    update_query_reserve=
                        args.update_query_reserve,
                    l2_thresholds=
                        args.l2_thresholds,
                    qsfd_k=args.qsfd_k,
                    qsfd_sector_deg=
                        args.qsfd_sector_deg,
                    qsfd_coarse_steps=
                        args.qsfd_coarse_steps,
                    qsfd_expand_factors=
                        args.qsfd_expand_factors,
                    qsfd_accept_eps=
                        args.qsfd_accept_eps,
                    qsfd_combo_span=
                        args.qsfd_combo_span,
                    qsfd_k_mode=
                        args.qsfd_k_mode,
                    qsfd_k_early=
                        args.qsfd_k_early,
                    qsfd_k_late=
                        args.qsfd_k_late,
                    qsfd_switch_ssim=
                        args.qsfd_switch_ssim,
                    qsfd_min_post_normal_reserve=
                        args.qsfd_min_post_normal_reserve,
                )

                result = attack.attack()

                if (
                    not result[
                        "success"
                    ]
                    or result[
                        "final"
                    ] is None
                ):
                    print(
                        "   [FAIL] attack failed."
                    )
                    continue

                final = result[
                    "final"
                ]

                qb = result[
                    "query_breakdown"
                ]

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
                    f"qsfd={qb.get('qsfd', 0)} "
                    f"boundary={qb.get('boundary', 0)}"
                )

                record = {
                    "sample_index": int(
                        sample_index
                    ),
                    "source_name":
                        source_name,
                    "source_label":
                        int(
                            source_label
                        ),
                    "method":
                        method,
                    "final":
                        final,
                    "query_breakdown":
                        dict(
                            qb
                        ),
                    "checkpoints":
                        result[
                            "checkpoints"
                        ],
                    "thresholds":
                        result[
                            "thresholds"
                        ],
                    "diagnostics":
                        result.get(
                            "diagnostics",
                            [],
                        ),
                    "qsfd_diagnostics":
                        result.get(
                            "qsfd_diagnostics",
                            [],
                        ),
                }

                results_by_method[
                    method
                ].append(
                    record
                )

        root = os.path.join(
            "Results",
            "RAG_TTBA_QSFD_COMPARE",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        final_summary_rows = []
        checkpoint_rows = []
        threshold_rows = []

        print()
        print(
            "=" * 110
        )
        print(
            "FINAL COMPARISON — QS-FD vs TtBA vs TtBA+QS-FD"
        )
        print(
            "=" * 110
        )

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
                    json_safe(
                        records
                    ),
                    f,
                    indent=2,
                )

            pix = np.asarray(
                [
                    row[
                        "final"
                    ][
                        "l2_pix"
                    ]
                    for row in records
                ],
                dtype=float,
            )

            lat = np.asarray(
                [
                    row[
                        "final"
                    ][
                        "l2_lat"
                    ]
                    for row in records
                ],
                dtype=float,
            )

            ssim_values = np.asarray(
                [
                    row[
                        "final"
                    ][
                        "ssim"
                    ]
                    for row in records
                ],
                dtype=float,
            )

            lpips_values = np.asarray(
                [
                    row[
                        "final"
                    ][
                        "lpips"
                    ]
                    for row in records
                ],
                dtype=float,
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

            q_qsfd = int(
                sum(
                    row[
                        "query_breakdown"
                    ].get(
                        "qsfd",
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

            print(
                f"{method:<18} "
                f"PixL2med={np.median(pix):.4f} "
                f"LatL2med={np.median(lat):.4f} "
                f"SSIM={np.mean(ssim_values):.5f} "
                f"LPIPS={np.mean(lpips_values):.5f} "
                f"Qnormal={q_normal} "
                f"Qbridge={q_bridge} "
                f"Qqsfd={q_qsfd} "
                f"Qboundary={q_boundary}"
            )

            final_summary_rows.append(
                {
                    "method":
                        method,
                    "n":
                        int(
                            len(
                                records
                            )
                        ),
                    "median_final_l2_pix":
                        float(
                            np.median(
                                pix
                            )
                        ),
                    "median_final_l2_lat":
                        float(
                            np.median(
                                lat
                            )
                        ),
                    "mean_final_ssim":
                        float(
                            np.mean(
                                ssim_values
                            )
                        ),
                    "mean_final_lpips":
                        float(
                            np.mean(
                                lpips_values
                            )
                        ),
                    "sum_q_init":
                        q_init,
                    "sum_q_normal":
                        q_normal,
                    "sum_q_bridge":
                        q_bridge,
                    "sum_q_qsfd":
                        q_qsfd,
                    "sum_q_boundary":
                        q_boundary,
                }
            )

            for row in records:
                cp = row[
                    "checkpoints"
                ]

                for j, checkpoint in enumerate(
                    args.query_checkpoints
                ):
                    checkpoint_rows.append(
                        {
                            "method":
                                method,
                            "sample_index":
                                int(
                                    row[
                                        "sample_index"
                                    ]
                                ),
                            "source_name":
                                row[
                                    "source_name"
                                ],
                            "checkpoint":
                                int(
                                    checkpoint
                                ),
                            "state_query":
                                int(
                                    cp[
                                        "state_query"
                                    ][
                                        j
                                    ]
                                ),
                            "l2_lat":
                                float(
                                    cp[
                                        "l2_lat"
                                    ][
                                        j
                                    ]
                                ),
                            "l2_pix":
                                float(
                                    cp[
                                        "l2_pix"
                                    ][
                                        j
                                    ]
                                ),
                            "ssim":
                                float(
                                    cp[
                                        "ssim"
                                    ][
                                        j
                                    ]
                                ),
                            "lpips":
                                float(
                                    cp[
                                        "lpips"
                                    ][
                                        j
                                    ]
                                ),
                            "asr":
                                float(
                                    cp[
                                        "asr"
                                    ][
                                        j
                                    ]
                                ),
                        }
                    )

            for threshold in args.l2_thresholds:
                hits = [
                    int(
                        row[
                            "thresholds"
                        ][
                            float(
                                threshold
                            )
                        ]
                    )
                    for row in records
                ]

                valid_hits = [
                    q
                    for q in hits
                    if q >= 0
                ]

                threshold_rows.append(
                    {
                        "method":
                            method,
                        "l2_threshold":
                            float(
                                threshold
                            ),
                        "hit_count":
                            int(
                                len(
                                    valid_hits
                                )
                            ),
                        "hit_rate":
                            float(
                                len(
                                    valid_hits
                                )
                                / len(
                                    records
                                )
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
            checkpoint_rows,
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
