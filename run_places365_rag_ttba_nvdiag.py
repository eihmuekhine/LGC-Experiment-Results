
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

from attacks.rag_ttba_nvdiag import (
    RAGTtBANVNormalDiagnosticAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def write_normal_csvs(
    out_dir,
    normal_diagnostics,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    replicate_path = os.path.join(
        out_dir,
        "normal_replicates.csv",
    )

    summary_path = os.path.join(
        out_dir,
        "normal_snapshot_summary.csv",
    )

    cross_path = os.path.join(
        out_dir,
        "cross_size_cosines.csv",
    )

    with open(
        replicate_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "snapshot_iteration",
                "attack_query_at_snapshot",
                "N",
                "repeat",
                "seed",
                "q_requested",
                "q_used",
                "valid",
                "alpha_v_dot_n",
                "phi_deg",
                "n_perp_norm",
            ]
        )

        for snap in normal_diagnostics:
            for level in snap.get(
                "levels",
                [],
            ):
                for row in level.get(
                    "repeats",
                    [],
                ):
                    writer.writerow(
                        [
                            snap[
                                "iteration"
                            ],
                            snap[
                                "query_at_snapshot"
                            ],
                            level[
                                "N"
                            ],
                            row[
                                "repeat"
                            ],
                            row[
                                "seed"
                            ],
                            row[
                                "q_requested"
                            ],
                            row[
                                "q_used"
                            ],
                            int(
                                row[
                                    "valid"
                                ]
                            ),
                            row[
                                "alpha_v_dot_n"
                            ],
                            row[
                                "phi_deg"
                            ],
                            row[
                                "n_perp_norm"
                            ],
                        ]
                    )

    with open(
        summary_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "snapshot_iteration",
                "attack_query_at_snapshot",
                "latent_radius",
                "N",
                "valid_repeat_count",
                "same_N_cos_mean",
                "same_N_cos_min",
                "phi_mean_deg",
                "phi_std_deg",
                "alpha_mean",
                "mean_normal_alpha",
                "mean_normal_phi_deg",
                "mean_normal_perp_norm",
                "offline_diag_queries_snapshot",
            ]
        )

        for snap in normal_diagnostics:
            for level in snap.get(
                "levels",
                [],
            ):
                writer.writerow(
                    [
                        snap[
                            "iteration"
                        ],
                        snap[
                            "query_at_snapshot"
                        ],
                        snap.get(
                            "radius_latent"
                        ),
                        level[
                            "N"
                        ],
                        level[
                            "valid_repeat_count"
                        ],
                        level[
                            "same_N_cos_mean"
                        ],
                        level[
                            "same_N_cos_min"
                        ],
                        level[
                            "phi_mean_deg"
                        ],
                        level[
                            "phi_std_deg"
                        ],
                        level[
                            "alpha_mean"
                        ],
                        level[
                            "mean_normal_alpha"
                        ],
                        level[
                            "mean_normal_phi_deg"
                        ],
                        level[
                            "mean_normal_perp_norm"
                        ],
                        snap[
                            "diagnostic_queries"
                        ],
                    ]
                )

    with open(
        cross_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "snapshot_iteration",
                "attack_query_at_snapshot",
                "N_a",
                "N_b",
                "cos_signed",
                "cos_abs",
                "angle_deg",
            ]
        )

        for snap in normal_diagnostics:
            for row in snap.get(
                "cross_size",
                [],
            ):
                writer.writerow(
                    [
                        snap[
                            "iteration"
                        ],
                        snap[
                            "query_at_snapshot"
                        ],
                        row[
                            "N_a"
                        ],
                        row[
                            "N_b"
                        ],
                        row[
                            "cos_signed"
                        ],
                        row[
                            "cos_abs"
                        ],
                        row[
                            "angle_deg"
                        ],
                    ]
                )

    return (
        replicate_path,
        summary_path,
        cross_path,
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
        "--normal_sizes",
        nargs="+",
        type=int,
        default=[
            30,
            60,
            120,
        ],
    )

    parser.add_argument(
        "--normal_repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--snapshot_iterations",
        nargs="+",
        type=int,
        default=[
            0,
            3,
            6,
            12,
        ],
    )

    parser.add_argument(
        "--diagnostic_seed",
        type=int,
        default=1992,
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

    if args.num_samples <= 0:
        raise ValueError(
            "--num_samples must be > 0."
        )

    if args.max_queries <= 0:
        raise ValueError(
            "--max_queries must be > 0."
        )

    if len(
        args.normal_sizes
    ) < 2:
        raise ValueError(
            "Use at least two --normal_sizes values."
        )

    if args.normal_repeats < 2:
        raise ValueError(
            "--normal_repeats must be >= 2."
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
        print(
            "\n"
            + "=" * 96
        )

        print(
            "NV NORMAL-QUALITY DIAGNOSTIC — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )

        print(
            "=" * 96
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

        all_final_rows = []
        all_diag_records = []

        for sample_index, pair in enumerate(
            pairs
        ):
            (
                source_name,
                source_label,
                target_name,
                target_label,
            ) = pair

            print(
                f"\n[{sample_index + 1}/"
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
                        f"[!] Skip {source_name}: "
                        "hard-label target seed failed."
                    )
                    continue

            # Restore the exact per-sample BASE seed before attack creation.
            set_seed(
                args.seed
                + sample_index
            )

            attack = RAGTtBANVNormalDiagnosticAttack(
                model=wrapper,
                z_source=z_source,
                x_orig=x_source,
                source_label=source_label,
                loss_fn_vgg=loss_fn_vgg,
                ssim_fn=ssim_fn,
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
                normal_sizes=args.normal_sizes,
                normal_repeats=args.normal_repeats,
                snapshot_iterations=args.snapshot_iterations,
                diagnostic_seed=(
                    args.diagnostic_seed
                    + 1000000
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
                    "   [FAIL] attack initialization failed."
                )
                continue

            final = result[
                "final"
            ]

            qb = result[
                "query_breakdown"
            ]

            print(
                "\n"
                "   BASE trajectory final:"
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
                "   Attack Q only: "
                f"init={qb.get('init', 0)} "
                f"normal={qb.get('normal', 0)} "
                f"bridge={qb.get('bridge', 0)} "
                f"boundary={qb.get('boundary', 0)}"
            )

            print(
                "   OFFLINE diagnostic Q "
                "(not part of attack budget): "
                f"{result['diagnostic_query_total']}"
            )

            all_final_rows.append(
                {
                    "sample_index": int(
                        sample_index
                    ),
                    "source_name": source_name,
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
                    "l2_lat": float(
                        final[
                            "l2_lat"
                        ]
                    ),
                    "l2_pix": float(
                        final[
                            "l2_pix"
                        ]
                    ),
                    "ssim": float(
                        final[
                            "ssim"
                        ]
                    ),
                    "lpips": float(
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
                    "offline_diagnostic_queries": int(
                        result[
                            "diagnostic_query_total"
                        ]
                    ),
                }
            )

            all_diag_records.append(
                {
                    "sample_index": int(
                        sample_index
                    ),
                    "source_name": source_name,
                    "normal_diagnostics": result[
                        "normal_diagnostics"
                    ],
                }
            )

        root = os.path.join(
            "Results",
            "NV_NORMAL_DIAGNOSTIC",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        final_csv = os.path.join(
            root,
            "base_final_summary.csv",
        )

        with open(
            final_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_index",
                    "source_name",
                    "best_query",
                    "budget_consumed",
                    "l2_lat",
                    "l2_pix",
                    "ssim",
                    "lpips",
                    "q_init",
                    "q_normal",
                    "q_bridge",
                    "q_boundary",
                    "offline_diagnostic_queries",
                ],
            )

            writer.writeheader()

            for row in all_final_rows:
                writer.writerow(
                    row
                )

        # This first experiment is intended for one sample, but the writer
        # supports multiple samples by creating one subdirectory per sample.
        for record in all_diag_records:
            sample_dir = os.path.join(
                root,
                f"sample_{record['sample_index']:03d}",
            )

            os.makedirs(
                sample_dir,
                exist_ok=True,
            )

            with open(
                os.path.join(
                    sample_dir,
                    "normal_diagnostics.json",
                ),
                "w",
            ) as f:
                json.dump(
                    record[
                        "normal_diagnostics"
                    ],
                    f,
                    indent=2,
                )

            write_normal_csvs(
                sample_dir,
                record[
                    "normal_diagnostics"
                ],
            )

        print(
            "\n"
            + "=" * 96
        )

        print(
            "NV NORMAL-QUALITY SUMMARY"
        )

        print(
            "=" * 96
        )

        for record in all_diag_records:
            print(
                f"\nSample {record['sample_index']}: "
                f"{record['source_name']}"
            )

            for snap in record[
                "normal_diagnostics"
            ]:
                print(
                    f"  pre-Iter{snap['iteration']} "
                    f"(attackQ={snap['query_at_snapshot']}):"
                )

                for level in snap.get(
                    "levels",
                    [],
                ):
                    same_cos = level[
                        "same_N_cos_mean"
                    ]

                    print(
                        "    "
                        f"N={level['N']:<4} "
                        f"sameNcos="
                        f"{'NA' if same_cos is None else f'{same_cos:.4f}'} "
                        f"phi="
                        f"{'NA' if level['phi_mean_deg'] is None else f'{level['phi_mean_deg']:.2f}'}"
                        " deg "
                        f"alpha="
                        f"{'NA' if level['alpha_mean'] is None else f'{level['alpha_mean']:.4f}'} "
                        f"perp="
                        f"{'NA' if level['mean_normal_perp_norm'] is None else f'{level['mean_normal_perp_norm']:.4f}'}"
                    )

                for row in snap.get(
                    "cross_size",
                    [],
                ):
                    print(
                        "    "
                        f"cross {row['N_a']}->{row['N_b']}: "
                        f"cos={row['cos_signed']:.4f}, "
                        f"angle={row['angle_deg']:.2f} deg"
                    )

        print(
            f"\n[*] Final BASE summary: "
            f"{final_csv}"
        )

        print(
            f"[*] Diagnostic root: "
            f"{root}"
        )

        del wrapper
        del classifier

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
