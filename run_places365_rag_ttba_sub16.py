
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

from attacks.rag_ttba_sub16 import (
    RAGTtBASubspaceAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def _write_csv(
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


def save_numeric_npz(
    out_dir,
    rows,
    checkpoints,
    thresholds,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    if not rows:
        return

    def stack_float(
        key,
    ):
        return np.stack(
            [
                row[
                    key
                ]
                for row in rows
            ],
            axis=0,
        ).astype(
            np.float64
        )

    def stack_int(
        key,
    ):
        return np.stack(
            [
                row[
                    key
                ]
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
                row[
                    "best_query"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "budget_consumed": np.asarray(
            [
                row[
                    "budget_consumed"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "final_l2_lat": np.asarray(
            [
                row[
                    "final_l2_lat"
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_l2_pix": np.asarray(
            [
                row[
                    "final_l2_pix"
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_ssim": np.asarray(
            [
                row[
                    "final_ssim"
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "final_lpips": np.asarray(
            [
                row[
                    "final_lpips"
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "q_init": np.asarray(
            [
                row[
                    "q_init"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_normal": np.asarray(
            [
                row[
                    "q_normal"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_bridge": np.asarray(
            [
                row[
                    "q_bridge"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "q_boundary": np.asarray(
            [
                row[
                    "q_boundary"
                ]
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
                row[
                    "thresholds"
                ][
                    float(
                        threshold
                    )
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
            "RAG_TTBA_SUB16",
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

    bad = sorted(
        set(
            args.attack_methods
        )
        - RAGTtBASubspaceAttack.SUPPORTED
    )

    if bad:
        raise ValueError(
            f"Unsupported attack methods: {bad}. "
            f"Supported={sorted(RAGTtBASubspaceAttack.SUPPORTED)}"
        )

    if args.subspace_dim < 2:
        raise ValueError(
            "--subspace_dim must be >= 2."
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
                    parts[
                        0
                    ]
                ] = int(
                    parts[
                        1
                    ]
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
            "=" * 100
        )
        print(
            "RAG-TtBA SUBSPACE-NORMAL MATCHED-BUDGET TEST — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )
        print(
            "=" * 100
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

        rows_by_method = {
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

                # Exact per-sample reset before every method.
                set_seed(
                    args.seed
                    + sample_index
                )

                attack = RAGTtBASubspaceAttack(
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
                    l2_thresholds=args.l2_thresholds,
                    subspace_dim=args.subspace_dim,
                    subspace_basis_seed=(
                        args.subspace_basis_seed
                        + sample_index
                    ),
                    subspace_probe_seed=(
                        args.subspace_probe_seed
                        + 10_000
                        * sample_index
                    ),
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
                        "   [FAIL] Attack initialization failed."
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
                    == RAGTtBASubspaceAttack.METHOD_SUB16
                ):
                    print(
                        "   SUB16 summary: "
                        f"updates="
                        f"{sub_summary.get('normal_updates', 0)} "
                        f"meanSplitCos="
                        f"{sub_summary.get('mean_split_half_cos')} "
                        f"medianSplitCos="
                        f"{sub_summary.get('median_split_half_cos')} "
                        f"meanPerp="
                        f"{sub_summary.get('mean_n_perp_norm')}"
                    )

                row = {
                    "source_name": source_name,
                    "source_label": int(
                        source_label
                    ),
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
                        qb.get(
                            "init",
                            0,
                        )
                    ),
                    "q_normal": int(
                        qb.get(
                            "normal",
                            0,
                        )
                    ),
                    "q_bridge": int(
                        qb.get(
                            "bridge",
                            0,
                        )
                    ),
                    "q_boundary": int(
                        qb.get(
                            "boundary",
                            0,
                        )
                    ),
                    "thresholds": result[
                        "thresholds"
                    ],
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
                }

                rows_by_method[
                    method
                ].append(
                    row
                )

        root = os.path.join(
            "Results",
            "RAG_TTBA_SUB16",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        summary_rows = []
        threshold_rows = []

        print()
        print(
            "=" * 100
        )
        print(
            "RAG-TtBA SUB16 MATCHED-BUDGET SUMMARY"
        )
        print(
            "=" * 100
        )

        for method in args.attack_methods:
            rows = rows_by_method[
                method
            ]

            if not rows:
                continue

            method_dir = os.path.join(
                root,
                method,
            )

            os.makedirs(
                method_dir,
                exist_ok=True,
            )

            save_numeric_npz(
                method_dir,
                rows,
                args.query_checkpoints,
                args.l2_thresholds,
            )

            checkpoint_rows = []

            for i, row in enumerate(
                rows
            ):
                for j, checkpoint in enumerate(
                    args.query_checkpoints
                ):
                    checkpoint_rows.append(
                        {
                            "sample_index": int(
                                i
                            ),
                            "source_name":
                                row[
                                    "source_name"
                                ],
                            "checkpoint": int(
                                checkpoint
                            ),
                            "state_query": int(
                                row[
                                    "state_query"
                                ][
                                    j
                                ]
                            ),
                            "l2_lat": float(
                                row[
                                    "l2_lat"
                                ][
                                    j
                                ]
                            ),
                            "l2_pix": float(
                                row[
                                    "l2_pix"
                                ][
                                    j
                                ]
                            ),
                            "ssim": float(
                                row[
                                    "ssim"
                                ][
                                    j
                                ]
                            ),
                            "lpips": float(
                                row[
                                    "lpips"
                                ][
                                    j
                                ]
                            ),
                            "asr": float(
                                row[
                                    "asr"
                                ][
                                    j
                                ]
                            ),
                        }
                    )

            _write_csv(
                os.path.join(
                    method_dir,
                    "checkpoint_summary.csv",
                ),
                checkpoint_rows,
            )

            with open(
                os.path.join(
                    method_dir,
                    "attack_diagnostics.json",
                ),
                "w",
            ) as f:
                json.dump(
                    [
                        row[
                            "diagnostics"
                        ]
                        for row in rows
                    ],
                    f,
                    indent=2,
                )

            with open(
                os.path.join(
                    method_dir,
                    "subspace_normal_diagnostics.json",
                ),
                "w",
            ) as f:
                json.dump(
                    [
                        row[
                            "subspace_normal_diagnostics"
                        ]
                        for row in rows
                    ],
                    f,
                    indent=2,
                )

            pix = np.asarray(
                [
                    row[
                        "final_l2_pix"
                    ]
                    for row in rows
                ],
                dtype=float,
            )

            lat = np.asarray(
                [
                    row[
                        "final_l2_lat"
                    ]
                    for row in rows
                ],
                dtype=float,
            )

            ssim_arr = np.asarray(
                [
                    row[
                        "final_ssim"
                    ]
                    for row in rows
                ],
                dtype=float,
            )

            lpips_arr = np.asarray(
                [
                    row[
                        "final_lpips"
                    ]
                    for row in rows
                ],
                dtype=float,
            )

            q_init = int(
                sum(
                    row[
                        "q_init"
                    ]
                    for row in rows
                )
            )

            q_normal = int(
                sum(
                    row[
                        "q_normal"
                    ]
                    for row in rows
                )
            )

            q_bridge = int(
                sum(
                    row[
                        "q_bridge"
                    ]
                    for row in rows
                )
            )

            q_boundary = int(
                sum(
                    row[
                        "q_boundary"
                    ]
                    for row in rows
                )
            )

            split_values = []

            if (
                method
                == RAGTtBASubspaceAttack.METHOD_SUB16
            ):
                for row in rows:
                    value = row[
                        "subspace_normal_summary"
                    ].get(
                        "mean_split_half_cos"
                    )

                    if (
                        value is not None
                        and np.isfinite(
                            value
                        )
                    ):
                        split_values.append(
                            float(
                                value
                            )
                        )

            mean_split = (
                None
                if not split_values
                else float(
                    np.mean(
                        split_values
                    )
                )
            )

            print(
                f"{method:<18} "
                f"PixL2med={np.median(pix):.4f} "
                f"LatL2med={np.median(lat):.4f} "
                f"SSIM={np.mean(ssim_arr):.5f} "
                f"LPIPS={np.mean(lpips_arr):.5f} "
                f"Qnormal={q_normal} "
                f"Qbridge={q_bridge} "
                f"Qboundary={q_boundary}"
                + (
                    ""
                    if mean_split is None
                    else f" meanSplitCos={mean_split:.4f}"
                )
            )

            summary_rows.append(
                {
                    "method": method,
                    "n": int(
                        len(
                            rows
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
                                ssim_arr
                            )
                        ),
                    "mean_final_lpips":
                        float(
                            np.mean(
                                lpips_arr
                            )
                        ),
                    "sum_q_init":
                        int(
                            q_init
                        ),
                    "sum_q_normal":
                        int(
                            q_normal
                        ),
                    "sum_q_bridge":
                        int(
                            q_bridge
                        ),
                    "sum_q_boundary":
                        int(
                            q_boundary
                        ),
                    "mean_subspace_split_half_cos":
                        mean_split,
                }
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
                    for row in rows
                ]

                valid_hits = [
                    q
                    for q in hits
                    if q >= 0
                ]

                threshold_rows.append(
                    {
                        "method": method,
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
                                    rows
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

        summary_path = os.path.join(
            root,
            "final_summary.csv",
        )

        threshold_path = os.path.join(
            root,
            "threshold_summary.csv",
        )

        _write_csv(
            summary_path,
            summary_rows,
        )

        _write_csv(
            threshold_path,
            threshold_rows,
        )

        print()
        print(
            f"[*] Summary: {summary_path}"
        )

        print(
            f"[*] Thresholds: {threshold_path}"
        )

        del wrapper
        del classifier

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
