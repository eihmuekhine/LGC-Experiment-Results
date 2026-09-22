
import argparse
import csv
import json
import math
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

from attacks.rag_ttba_nvdiag_v2 import (
    RAGTtBANVNormalDiagnosticV2Attack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


def _write_rows_csv(
    path,
    fieldnames,
    rows,
):
    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_diagnostic_tables(
    out_dir,
    normal_diagnostics,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    estimate_rows = []
    repeat_rows = []
    cross_budget_rows = []
    cross_sigma_rows = []
    paired_vs_one_rows = []

    for snap in normal_diagnostics:
        snap_iter = int(
            snap["iteration"]
        )
        snap_q = int(
            snap["query_at_snapshot"]
        )
        radius = snap.get(
            "radius_latent"
        )

        for row in snap.get(
            "rows",
            [],
        ):
            estimate_rows.append(
                {
                    "snapshot_iteration":
                        snap_iter,
                    "attack_query_at_snapshot":
                        snap_q,
                    "latent_radius":
                        radius,
                    **row,
                }
            )

        for row in snap.get(
            "repeat_consistency",
            [],
        ):
            repeat_rows.append(
                {
                    "snapshot_iteration":
                        snap_iter,
                    "attack_query_at_snapshot":
                        snap_q,
                    "latent_radius":
                        radius,
                    **row,
                }
            )

        for row in snap.get(
            "cross_budget",
            [],
        ):
            cross_budget_rows.append(
                {
                    "snapshot_iteration":
                        snap_iter,
                    "attack_query_at_snapshot":
                        snap_q,
                    "latent_radius":
                        radius,
                    **row,
                }
            )

        for row in snap.get(
            "cross_sigma",
            [],
        ):
            cross_sigma_rows.append(
                {
                    "snapshot_iteration":
                        snap_iter,
                    "attack_query_at_snapshot":
                        snap_q,
                    "latent_radius":
                        radius,
                    **row,
                }
            )

        for row in snap.get(
            "paired_vs_one_sided",
            [],
        ):
            paired_vs_one_rows.append(
                {
                    "snapshot_iteration":
                        snap_iter,
                    "attack_query_at_snapshot":
                        snap_q,
                    "latent_radius":
                        radius,
                    **row,
                }
            )

    if estimate_rows:
        _write_rows_csv(
            os.path.join(
                out_dir,
                "normal_estimates.csv",
            ),
            list(
                estimate_rows[0].keys()
            ),
            estimate_rows,
        )

    if repeat_rows:
        _write_rows_csv(
            os.path.join(
                out_dir,
                "repeat_consistency.csv",
            ),
            list(
                repeat_rows[0].keys()
            ),
            repeat_rows,
        )

    if cross_budget_rows:
        _write_rows_csv(
            os.path.join(
                out_dir,
                "cross_budget_consistency.csv",
            ),
            list(
                cross_budget_rows[0].keys()
            ),
            cross_budget_rows,
        )

    if cross_sigma_rows:
        _write_rows_csv(
            os.path.join(
                out_dir,
                "cross_sigma_consistency.csv",
            ),
            list(
                cross_sigma_rows[0].keys()
            ),
            cross_sigma_rows,
        )

    if paired_vs_one_rows:
        _write_rows_csv(
            os.path.join(
                out_dir,
                "paired_vs_one_sided.csv",
            ),
            list(
                paired_vs_one_rows[0].keys()
            ),
            paired_vs_one_rows,
        )


def build_compact_summary(
    diagnostics,
):
    """
    Produce one compact row per:
        snapshot x estimator x sigma x budget

    including repeat consistency plus mean phi/alpha and paired informative rate.
    """
    rows = []

    for snap in diagnostics:
        for rc in snap.get(
            "repeat_consistency",
            [],
        ):
            matched = [
                row
                for row in snap.get(
                    "rows",
                    [],
                )
                if (
                    row[
                        "estimator"
                    ]
                    == rc[
                        "estimator"
                    ]
                    and abs(
                        float(
                            row[
                                "sigma"
                            ]
                        )
                        - float(
                            rc[
                                "sigma"
                            ]
                        )
                    )
                    < 1e-15
                    and int(
                        row[
                            "query_budget"
                        ]
                    )
                    == int(
                        rc[
                            "query_budget"
                        ]
                    )
                    and bool(
                        row[
                            "valid"
                        ]
                    )
                )
            ]

            def _mean_or_none(
                key,
            ):
                values = [
                    row[
                        key
                    ]
                    for row in matched
                    if row[
                        key
                    ] is not None
                    and (
                        not isinstance(
                            row[
                                key
                            ],
                            float,
                        )
                        or math.isfinite(
                            row[
                                key
                            ]
                        )
                    )
                ]
                if not values:
                    return None
                return float(
                    np.mean(
                        values
                    )
                )

            rows.append(
                {
                    "snapshot_iteration":
                        int(
                            snap[
                                "iteration"
                            ]
                        ),
                    "attack_query_at_snapshot":
                        int(
                            snap[
                                "query_at_snapshot"
                            ]
                        ),
                    "estimator":
                        rc[
                            "estimator"
                        ],
                    "sigma":
                        float(
                            rc[
                                "sigma"
                            ]
                        ),
                    "query_budget":
                        int(
                            rc[
                                "query_budget"
                            ]
                        ),
                    "valid_repeat_count":
                        int(
                            rc[
                                "valid_repeat_count"
                            ]
                        ),
                    "same_cos_signed_mean":
                        rc[
                            "cos_signed_mean"
                        ],
                    "same_cos_signed_min":
                        rc[
                            "cos_signed_min"
                        ],
                    "same_cos_abs_mean":
                        rc[
                            "cos_abs_mean"
                        ],
                    "phi_mean_deg":
                        _mean_or_none(
                            "phi_deg"
                        ),
                    "alpha_mean":
                        _mean_or_none(
                            "alpha_v_dot_n"
                        ),
                    "n_perp_norm_mean":
                        _mean_or_none(
                            "n_perp_norm"
                        ),
                    "eta_norm_mean":
                        _mean_or_none(
                            "eta_norm"
                        ),
                    "informative_pair_rate_mean":
                        _mean_or_none(
                            "informative_pair_rate"
                        ),
                }
            )

    return rows


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
        help=(
            "BASE attack sigma. "
            "Do not change for the reproduction trajectory."
        ),
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
        "--diagnostic_sigmas",
        nargs="+",
        type=float,
        default=[
            0.0002,
            0.0005,
            0.001,
            0.002,
        ],
    )

    parser.add_argument(
        "--diagnostic_query_budgets",
        nargs="+",
        type=int,
        default=[
            60,
            120,
            240,
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
        ],
        help=(
            "Start with pre-Iter0 only. If a stable paired estimator emerges, "
            "later rerun with --snapshot_iterations 0 6 12."
        ),
    )

    parser.add_argument(
        "--diagnostic_seed",
        type=int,
        default=2992,
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
        args.diagnostic_sigmas
    ) < 2:
        raise ValueError(
            "Use at least two --diagnostic_sigmas values."
        )

    if len(
        args.diagnostic_query_budgets
    ) < 2:
        raise ValueError(
            "Use at least two --diagnostic_query_budgets values."
        )

    if any(
        b < 2
        for b in args.diagnostic_query_budgets
    ):
        raise ValueError(
            "All diagnostic query budgets must be >= 2."
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
            "NV NORMAL DIAGNOSTIC V2 — "
            "ONE-SIDED vs PAIRED ANTITHETIC — "
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

        all_final_rows = []

        root = os.path.join(
            "Results",
            "NV_NORMAL_DIAGNOSTIC_V2",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

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
                        "skipping this sample."
                    )
                    continue

            # Restore validated per-sample BASE seed.
            set_seed(
                args.seed
                + sample_index
            )

            attack = RAGTtBANVNormalDiagnosticV2Attack(
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
                diagnostic_sigmas=args.diagnostic_sigmas,
                diagnostic_query_budgets=
                    args.diagnostic_query_budgets,
                normal_repeats=args.normal_repeats,
                snapshot_iterations=
                    args.snapshot_iterations,
                diagnostic_seed=(
                    args.diagnostic_seed
                    + 1_000_000
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
            diagnostics = result[
                "normal_diagnostics_v2"
            ]

            print()
            print(
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
                "(excluded from attack budget): "
                f"{result['diagnostic_query_total']}"
            )

            sample_dir = os.path.join(
                root,
                f"sample_{sample_index:03d}",
            )
            os.makedirs(
                sample_dir,
                exist_ok=True,
            )

            with open(
                os.path.join(
                    sample_dir,
                    "normal_diagnostics_v2.json",
                ),
                "w",
            ) as f:
                json.dump(
                    diagnostics,
                    f,
                    indent=2,
                )

            write_diagnostic_tables(
                sample_dir,
                diagnostics,
            )

            compact_rows = build_compact_summary(
                diagnostics
            )

            if compact_rows:
                _write_rows_csv(
                    os.path.join(
                        sample_dir,
                        "compact_stability_summary.csv",
                    ),
                    list(
                        compact_rows[0].keys()
                    ),
                    compact_rows,
                )

            all_final_rows.append(
                {
                    "sample_index":
                        int(sample_index),
                    "source_name":
                        source_name,
                    "best_query":
                        int(
                            final[
                                "query"
                            ]
                        ),
                    "budget_consumed":
                        int(
                            final[
                                "budget_consumed"
                            ]
                        ),
                    "l2_lat":
                        float(
                            final[
                                "l2_lat"
                            ]
                        ),
                    "l2_pix":
                        float(
                            final[
                                "l2_pix"
                            ]
                        ),
                    "ssim":
                        float(
                            final[
                                "ssim"
                            ]
                        ),
                    "lpips":
                        float(
                            final[
                                "lpips"
                            ]
                        ),
                    "q_init":
                        int(
                            qb.get(
                                "init",
                                0,
                            )
                        ),
                    "q_normal":
                        int(
                            qb.get(
                                "normal",
                                0,
                            )
                        ),
                    "q_bridge":
                        int(
                            qb.get(
                                "bridge",
                                0,
                            )
                        ),
                    "q_boundary":
                        int(
                            qb.get(
                                "boundary",
                                0,
                            )
                        ),
                    "offline_diagnostic_queries":
                        int(
                            result[
                                "diagnostic_query_total"
                            ]
                        ),
                }
            )

            # ----------------------------------------------------------
            # Human-readable compact diagnosis table.
            # ----------------------------------------------------------
            print()
            print(
                "=" * 100
            )
            print(
                "NV NORMAL DIAGNOSTIC V2 SUMMARY"
            )
            print(
                "=" * 100
            )

            for snap in diagnostics:
                print()
                print(
                    f"pre-Iter{snap['iteration']} "
                    f"(attackQ={snap['query_at_snapshot']}, "
                    f"radius={snap['radius_latent']:.6f})"
                )

                for sigma in args.diagnostic_sigmas:
                    print(
                        f"  sigma={sigma:.1e}"
                    )

                    for budget in (
                        args.diagnostic_query_budgets
                    ):
                        for estimator in (
                            RAGTtBANVNormalDiagnosticV2Attack.ONE_SIDED,
                            RAGTtBANVNormalDiagnosticV2Attack.PAIRED,
                        ):
                            rc = next(
                                (
                                    row
                                    for row in snap[
                                        "repeat_consistency"
                                    ]
                                    if (
                                        row[
                                            "estimator"
                                        ]
                                        == estimator
                                        and abs(
                                            float(
                                                row[
                                                    "sigma"
                                                ]
                                            )
                                            - float(
                                                sigma
                                            )
                                        )
                                        < 1e-15
                                        and int(
                                            row[
                                                "query_budget"
                                            ]
                                        )
                                        == int(
                                            budget
                                        )
                                    )
                                ),
                                None,
                            )

                            matched = [
                                row
                                for row in snap[
                                    "rows"
                                ]
                                if (
                                    row[
                                        "estimator"
                                    ]
                                    == estimator
                                    and abs(
                                        float(
                                            row[
                                                "sigma"
                                            ]
                                        )
                                        - float(
                                            sigma
                                        )
                                    )
                                    < 1e-15
                                    and int(
                                        row[
                                            "query_budget"
                                        ]
                                    )
                                    == int(
                                        budget
                                    )
                                    and bool(
                                        row[
                                            "valid"
                                        ]
                                    )
                                )
                            ]

                            same_cos = None
                            if rc is not None:
                                same_cos = rc[
                                    "cos_signed_mean"
                                ]

                            phi_mean = None
                            if matched:
                                phi_mean = float(
                                    np.mean(
                                        [
                                            row[
                                                "phi_deg"
                                            ]
                                            for row in matched
                                        ]
                                    )
                                )

                            info_mean = None
                            if (
                                estimator
                                == RAGTtBANVNormalDiagnosticV2Attack.PAIRED
                                and matched
                            ):
                                values = [
                                    row[
                                        "informative_pair_rate"
                                    ]
                                    for row in matched
                                    if row[
                                        "informative_pair_rate"
                                    ]
                                    is not None
                                ]
                                if values:
                                    info_mean = float(
                                        np.mean(values)
                                    )

                            cos_text = (
                                "NA"
                                if same_cos is None
                                else f"{same_cos:.4f}"
                            )
                            phi_text = (
                                "NA"
                                if phi_mean is None
                                else f"{phi_mean:.2f}"
                            )

                            if (
                                estimator
                                == RAGTtBANVNormalDiagnosticV2Attack.PAIRED
                            ):
                                info_text = (
                                    "NA"
                                    if info_mean is None
                                    else f"{info_mean:.3f}"
                                )
                                print(
                                    "    "
                                    f"B={budget:<4} "
                                    f"{estimator:<20} "
                                    f"sameCos={cos_text:<8} "
                                    f"phi={phi_text:<7} "
                                    f"infoRate={info_text}"
                                )
                            else:
                                print(
                                    "    "
                                    f"B={budget:<4} "
                                    f"{estimator:<20} "
                                    f"sameCos={cos_text:<8} "
                                    f"phi={phi_text:<7}"
                                )

        final_path = os.path.join(
            root,
            "base_final_summary.csv",
        )

        if all_final_rows:
            _write_rows_csv(
                final_path,
                list(
                    all_final_rows[0].keys()
                ),
                all_final_rows,
            )

        print()
        print(
            f"[*] BASE summary: "
            f"{final_path}"
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
