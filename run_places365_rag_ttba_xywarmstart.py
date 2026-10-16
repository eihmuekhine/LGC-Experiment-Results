
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

from attacks.rag_ttba_xywarmstart import (
    RAGTtBAXYWarmStartAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def parse_xy_pairs(
    flat_values,
):
    if len(
        flat_values
    ) % 2 != 0:
        raise ValueError(
            "XY list must contain an even number of values: "
            "x1 y1 x2 y2 ..."
        )

    return [
        (
            float(
                flat_values[
                    i
                ]
            ),
            float(
                flat_values[
                    i + 1
                ]
            ),
        )
        for i in range(
            0,
            len(
                flat_values
            ),
            2,
        )
    ]


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

    n = len(
        rows
    )

    c = len(
        checkpoints
    )

    def stack_float(
        key,
    ):
        if n == 0:
            return np.empty(
                (
                    0,
                    c,
                ),
                dtype=np.float64,
            )

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
        if n == 0:
            return np.empty(
                (
                    0,
                    c,
                ),
                dtype=np.int64,
            )

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
        "xy_target_updates": np.asarray(
            [
                row[
                    "xy_target_updates"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "xy_accepted_updates": np.asarray(
            [
                row[
                    "xy_accepted_updates"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "xy_attempted_updates": np.asarray(
            [
                row[
                    "xy_attempted_updates"
                ]
                for row in rows
            ],
            dtype=np.int64,
        ),
        "xy_aborted": np.asarray(
            [
                row[
                    "xy_aborted"
                ]
                for row in rows
            ],
            dtype=np.int64,
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
        "q_xy": np.asarray(
            [
                row[
                    "q_xy"
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
                            ][
                                j
                            ]
                        ),
                        float(
                            row[
                                "l2_lat"
                            ][
                                j
                            ]
                        ),
                        float(
                            row[
                                "l2_pix"
                            ][
                                j
                            ]
                        ),
                        float(
                            row[
                                "ssim"
                            ][
                                j
                            ]
                        ),
                        float(
                            row[
                                "lpips"
                            ][
                                j
                            ]
                        ),
                        float(
                            row[
                                "asr"
                            ][
                                j
                            ]
                        ),
                    ]
                )


def main(
):
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
            "RAG_XY_WS1",
            "RAG_XY_WS2",
            "RAG_XY_WS3",
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
        "--xy_primary_points",
        nargs="+",
        type=float,
        default=[
            0.90, 0.10,
            0.82, 0.20,
            0.70, 0.30,
            0.58, 0.48,
        ],
    )

    parser.add_argument(
        "--xy_rescue_points",
        nargs="+",
        type=float,
        default=[
            0.96, 0.05,
            0.92, 0.15,
            0.86, 0.28,
            0.75, 0.42,
        ],
    )

    parser.add_argument(
        "--xy_accept_eps",
        type=float,
        default=1e-8,
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
        - RAGTtBAXYWarmStartAttack.SUPPORTED_WARMSTART
    )

    if bad:
        raise ValueError(
            f"Unsupported methods: {bad}. "
            f"Supported="
            f"{sorted(RAGTtBAXYWarmStartAttack.SUPPORTED_WARMSTART)}"
        )

    primary_points = parse_xy_pairs(
        args.xy_primary_points
    )

    rescue_points = parse_xy_pairs(
        args.xy_rescue_points
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
            parts = line.strip().split()

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
        encoder = ae_model.module.encoder
        decoder = ae_model.module.decoder
    else:
        encoder = ae_model.encoder
        decoder = ae_model.decoder

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
        print(
            "\n"
            + "=" * 92
        )

        print(
            "RAG XY-WARMSTART ABLATION — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )

        print(
            "=" * 92
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

            with torch.no_grad(
            ):
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
                        f"[!] Skip {source_name}: "
                        "hard-label target seed failed."
                    )
                    continue

            print(
                f"\n[{sample_index + 1}/"
                f"{len(pairs)}] "
                f"{source_name}"
            )

            for method in args.attack_methods:
                print(
                    f"\n --- {method} ---"
                )

                set_seed(
                    args.seed
                    + sample_index
                )

                attack = RAGTtBAXYWarmStartAttack(
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
                    xy_primary_points=primary_points,
                    xy_rescue_points=rescue_points,
                    xy_accept_eps=args.xy_accept_eps,
                    l2_thresholds=args.l2_thresholds,
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
                        "   [FAIL] attack initialization failed."
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
                    f"   BEST Q={final['query']} "
                    f"(used={final['budget_consumed']}) "
                    f"PixL2={final['l2_pix']:.4f} "
                    f"LatL2={final['l2_lat']:.4f} "
                    f"SSIM={final['ssim']:.5f} "
                    f"LPIPS={final['lpips']:.5f}"
                )

                print(
                    "   Warm-start: "
                    f"target={final['xy_target_updates']} "
                    f"accepted={final['xy_accepted_updates']} "
                    f"attempted={final['xy_attempted_updates']} "
                    f"aborted={final['xy_aborted']}"
                )

                print(
                    "   Q: "
                    f"init={qb.get('init', 0)} "
                    f"normal={qb.get('normal', 0)} "
                    f"bridge={qb.get('bridge', 0)} "
                    f"xy={qb.get('xy', 0)} "
                    f"boundary={qb.get('boundary', 0)}"
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
                        "xy_target_updates": int(
                            final[
                                "xy_target_updates"
                            ]
                        ),
                        "xy_accepted_updates": int(
                            final[
                                "xy_accepted_updates"
                            ]
                        ),
                        "xy_attempted_updates": int(
                            final[
                                "xy_attempted_updates"
                            ]
                        ),
                        "xy_aborted": int(
                            final[
                                "xy_aborted"
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
                        "q_xy": int(
                            qb.get(
                                "xy",
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
                        "diagnostics": result[
                            "diagnostics"
                        ],
                    }
                )

        root = os.path.join(
            "Results",
            "RAG_TTBA_XYWARMSTART",
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
                    "xy_target_updates",
                    "median_xy_accepted",
                    "median_xy_attempted",
                    "abort_count",
                    "median_best_query",
                    "median_budget_consumed",
                    "median_final_l2_lat",
                    "median_final_l2_pix",
                    "mean_final_ssim",
                    "mean_final_lpips",
                    "sum_q_init",
                    "sum_q_normal",
                    "sum_q_bridge",
                    "sum_q_xy",
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
                + "=" * 92
            )

            print(
                "XY-WARMSTART ABLATION SUMMARY"
            )

            print(
                "=" * 92
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

                best_q = np.asarray(
                    [
                        row[
                            "best_query"
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                consumed = np.asarray(
                    [
                        row[
                            "budget_consumed"
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                xy_target = int(
                    rows[
                        0
                    ][
                        "xy_target_updates"
                    ]
                )

                xy_acc = np.asarray(
                    [
                        row[
                            "xy_accepted_updates"
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                xy_attempted = np.asarray(
                    [
                        row[
                            "xy_attempted_updates"
                        ]
                        for row in rows
                    ],
                    dtype=float,
                )

                abort_count = int(
                    sum(
                        row[
                            "xy_aborted"
                        ]
                        for row in rows
                    )
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

                q_xy = int(
                    sum(
                        row[
                            "q_xy"
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

                print(
                    f"{method:<18} "
                    f"XYtarget={xy_target} "
                    f"XYacc={np.median(xy_acc):.1f} "
                    f"PixL2med={np.median(pix):.4f} "
                    f"LatL2med={np.median(lat):.4f} "
                    f"SSIM={np.mean(ssim_arr):.5f} "
                    f"LPIPS={np.mean(lpips_arr):.5f} "
                    f"Qxy={q_xy} "
                    f"Qbridge={q_bridge}"
                )

                sw.writerow(
                    [
                        method,
                        len(
                            rows
                        ),
                        xy_target,
                        float(
                            np.median(
                                xy_acc
                            )
                        ),
                        float(
                            np.median(
                                xy_attempted
                            )
                        ),
                        abort_count,
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
                        q_xy,
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
                        for row in rows
                    ]

                    valid = [
                        q
                        for q in hits
                        if q >= 0
                    ]

                    tw.writerow(
                        [
                            method,
                            float(
                                threshold
                            ),
                            len(
                                valid
                            ),
                            (
                                len(
                                    valid
                                )
                                / len(
                                    rows
                                )
                                if rows
                                else 0.0
                            ),
                            (
                                float(
                                    np.median(
                                        valid
                                    )
                                )
                                if valid
                                else -1.0
                            ),
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

        if torch.cuda.is_available(
        ):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
