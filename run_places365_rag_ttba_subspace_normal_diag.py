
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

from attacks.rag_ttba_subspace_normal_diag import (
    RAGTtBASubspaceNormalDiagnosticAttack,
)

from run_places365_rag_ttba_pgr import (
    device,
    set_seed,
    load_places365_model,
    LatentModelWrapper,
    build_target_seed_hardlabel,
    prepare_strict_splits,
)


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


def flatten_diagnostic_tables(
    diagnostics,
):
    estimate_rows = []
    repeat_rows = []
    cross_dim_rows = []
    full_vs_rows = []

    for snap in diagnostics:
        common = {
            "snapshot_iteration": int(
                snap[
                    "iteration"
                ]
            ),
            "attack_query_at_snapshot": int(
                snap[
                    "query_at_snapshot"
                ]
            ),
            "latent_dim": snap.get(
                "latent_dim"
            ),
            "radius_latent": snap.get(
                "radius_latent"
            ),
            "basis_seed": snap.get(
                "basis_seed"
            ),
            "basis_orthogonality_max_abs_error":
                snap.get(
                    "basis_orthogonality_max_abs_error"
                ),
            "basis_tangent_v_max_abs_dot":
                snap.get(
                    "basis_tangent_v_max_abs_dot"
                ),
        }

        for row in snap.get(
            "rows",
            [],
        ):
            estimate_rows.append(
                {
                    **common,
                    **row,
                }
            )

        for row in snap.get(
            "repeat_consistency",
            [],
        ):
            repeat_rows.append(
                {
                    **common,
                    **row,
                }
            )

        for row in snap.get(
            "cross_dim",
            [],
        ):
            cross_dim_rows.append(
                {
                    **common,
                    **row,
                }
            )

        for row in snap.get(
            "full_vs_subspace",
            [],
        ):
            full_vs_rows.append(
                {
                    **common,
                    **row,
                }
            )

    return (
        estimate_rows,
        repeat_rows,
        cross_dim_rows,
        full_vs_rows,
    )


def build_compact_rows(
    diagnostics,
):
    """
    One compact row per exact estimator/r/budget config,
    including repeat consistency and mean geometry/informative rate.
    """
    compact = []

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
                    and row.get(
                        "subspace_dim"
                    )
                    == rc.get(
                        "subspace_dim"
                    )
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

            def mean_or_none(
                field,
            ):
                values = [
                    row[
                        field
                    ]
                    for row in matched
                    if (
                        row.get(
                            field
                        )
                        is not None
                    )
                    and (
                        not isinstance(
                            row.get(
                                field
                            ),
                            float,
                        )
                        or math.isfinite(
                            row[
                                field
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

            compact.append(
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
                    "latent_dim":
                        snap.get(
                            "latent_dim"
                        ),
                    "radius_latent":
                        snap.get(
                            "radius_latent"
                        ),
                    "estimator":
                        rc[
                            "estimator"
                        ],
                    "subspace_dim":
                        rc.get(
                            "subspace_dim"
                        ),
                    "query_budget":
                        int(
                            rc[
                                "query_budget"
                            ]
                        ),
                    "normal_cos_signed_mean":
                        rc[
                            "normal_cos_signed_mean"
                        ],
                    "normal_cos_signed_min":
                        rc[
                            "normal_cos_signed_min"
                        ],
                    "normal_cos_abs_mean":
                        rc[
                            "normal_cos_abs_mean"
                        ],
                    "y_cos_signed_mean":
                        rc[
                            "y_cos_signed_mean"
                        ],
                    "y_cos_signed_min":
                        rc[
                            "y_cos_signed_min"
                        ],
                    "y_cos_abs_mean":
                        rc[
                            "y_cos_abs_mean"
                        ],
                    "phi_mean_deg":
                        mean_or_none(
                            "phi_deg"
                        ),
                    "alpha_mean":
                        mean_or_none(
                            "alpha_v_dot_n"
                        ),
                    "n_perp_norm_mean":
                        mean_or_none(
                            "n_perp_norm"
                        ),
                    "v_dot_ey_mean":
                        mean_or_none(
                            "v_dot_ey"
                        ),
                    "eta_norm_mean":
                        mean_or_none(
                            "eta_norm"
                        ),
                    "energy_scale_mean":
                        mean_or_none(
                            "energy_scale"
                        ),
                    "informative_pair_rate_mean":
                        mean_or_none(
                            "informative_pair_rate"
                        ),
                }
            )

    return compact


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
            "BASE attack sigma. Keep 0.0002 for reference reproduction."
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
        "--subspace_dims",
        nargs="+",
        type=int,
        default=[
            16,
            32,
            64,
            128,
        ],
    )

    parser.add_argument(
        "--diagnostic_query_budgets",
        nargs="+",
        type=int,
        default=[
            60,
            120,
        ],
    )

    parser.add_argument(
        "--diagnostic_sigma",
        type=float,
        default=0.0002,
        help=(
            "Use one fixed sigma first to isolate the dimensionality effect."
        ),
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
            "First run pre-Iter0 only. "
            "Later extend to 0 6 12 only if a stable subspace emerges."
        ),
    )

    parser.add_argument(
        "--diagnostic_seed",
        type=int,
        default=3992,
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

    if any(
        r < 2
        for r in args.subspace_dims
    ):
        raise ValueError(
            "All --subspace_dims must be >= 2."
        )

    if any(
        b < 2
        for b in args.diagnostic_query_budgets
    ):
        raise ValueError(
            "All --diagnostic_query_budgets must be >= 2."
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
            "=" * 104
        )

        print(
            "SUBSPACE NORMAL DIAGNOSTIC — "
            f"{arch.upper()} — "
            f"{args.mode.upper()}"
        )

        print(
            "=" * 104
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

        root = os.path.join(
            "Results",
            "SUBSPACE_NORMAL_DIAGNOSTIC",
            args.mode,
            arch,
        )

        os.makedirs(
            root,
            exist_ok=True,
        )

        final_rows = []

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
                        "[!] Target seed failed; skipping sample."
                    )
                    continue

            # Re-establish exact BASE RNG before attack creation.
            set_seed(
                args.seed
                + sample_index
            )

            attack = RAGTtBASubspaceNormalDiagnosticAttack(
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
                subspace_dims=args.subspace_dims,
                diagnostic_query_budgets=
                    args.diagnostic_query_budgets,
                diagnostic_sigma=
                    args.diagnostic_sigma,
                normal_repeats=
                    args.normal_repeats,
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
                    "   [FAIL] attack initialization failed."
                )
                continue

            final = result[
                "final"
            ]

            qb = result[
                "query_breakdown"
            ]

            diagnostics = result[
                "subspace_diagnostics"
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
                    "subspace_normal_diagnostics.json",
                ),
                "w",
            ) as f:
                json.dump(
                    diagnostics,
                    f,
                    indent=2,
                )

            (
                estimate_rows,
                repeat_rows,
                cross_dim_rows,
                full_vs_rows,
            ) = flatten_diagnostic_tables(
                diagnostics
            )

            write_csv(
                os.path.join(
                    sample_dir,
                    "normal_estimates.csv",
                ),
                estimate_rows,
            )

            write_csv(
                os.path.join(
                    sample_dir,
                    "repeat_consistency.csv",
                ),
                repeat_rows,
            )

            write_csv(
                os.path.join(
                    sample_dir,
                    "cross_dimension_consistency.csv",
                ),
                cross_dim_rows,
            )

            write_csv(
                os.path.join(
                    sample_dir,
                    "full_vs_subspace.csv",
                ),
                full_vs_rows,
            )

            compact_rows = build_compact_rows(
                diagnostics
            )

            write_csv(
                os.path.join(
                    sample_dir,
                    "compact_stability_summary.csv",
                ),
                compact_rows,
            )

            final_rows.append(
                {
                    "sample_index":
                        int(
                            sample_index
                        ),
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
            # Concise human-readable summary.
            # ----------------------------------------------------------
            print()
            print(
                "=" * 104
            )

            print(
                "SUBSPACE NORMAL DIAGNOSTIC SUMMARY"
            )

            print(
                "=" * 104
            )

            for snap in diagnostics:
                print()
                print(
                    f"pre-Iter{snap['iteration']} "
                    f"(attackQ={snap['query_at_snapshot']}, "
                    f"D={snap['latent_dim']}, "
                    f"radius={snap['radius_latent']:.6f})"
                )

                print(
                    "  basis: "
                    f"orthErr="
                    f"{snap['basis_orthogonality_max_abs_error']:.3e}, "
                    f"max|v^T b_tan|="
                    f"{snap['basis_tangent_v_max_abs_dot']:.3e}"
                )

                for budget in (
                    args.diagnostic_query_budgets
                ):
                    full_rc = next(
                        row
                        for row in snap[
                            "repeat_consistency"
                        ]
                        if (
                            row[
                                "estimator"
                            ]
                            == RAGTtBASubspaceNormalDiagnosticAttack.FULL_PARENT
                            and int(
                                row[
                                    "query_budget"
                                ]
                            )
                            == int(
                                budget
                            )
                        )
                    )

                    n_full = (
                        "NA"
                        if full_rc[
                            "normal_cos_signed_mean"
                        ] is None
                        else f"{full_rc['normal_cos_signed_mean']:.4f}"
                    )

                    y_full = (
                        "NA"
                        if full_rc[
                            "y_cos_signed_mean"
                        ] is None
                        else f"{full_rc['y_cos_signed_mean']:.4f}"
                    )

                    print(
                        f"  B={budget:<4} "
                        f"FULL_PARENT              "
                        f"nCos={n_full:<8} "
                        f"yCos={y_full:<8}"
                    )

                    for r in args.subspace_dims:
                        for estimator in (
                            RAGTtBASubspaceNormalDiagnosticAttack.SUBSPACE_ONE,
                            RAGTtBASubspaceNormalDiagnosticAttack.SUBSPACE_PAIRED,
                        ):
                            rc = next(
                                row
                                for row in snap[
                                    "repeat_consistency"
                                ]
                                if (
                                    row[
                                        "estimator"
                                    ]
                                    == estimator
                                    and int(
                                        row[
                                            "subspace_dim"
                                        ]
                                    )
                                    == int(
                                        r
                                    )
                                    and int(
                                        row[
                                            "query_budget"
                                        ]
                                    )
                                    == int(
                                        budget
                                    )
                                )
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
                                    and int(
                                        row[
                                            "subspace_dim"
                                        ]
                                    )
                                    == int(
                                        r
                                    )
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

                            info_rate = None

                            if (
                                estimator
                                == RAGTtBASubspaceNormalDiagnosticAttack.SUBSPACE_PAIRED
                            ):
                                info_values = [
                                    row[
                                        "informative_pair_rate"
                                    ]
                                    for row in matched
                                    if row[
                                        "informative_pair_rate"
                                    ] is not None
                                ]

                                if info_values:
                                    info_rate = float(
                                        np.mean(
                                            info_values
                                        )
                                    )

                            n_text = (
                                "NA"
                                if rc[
                                    "normal_cos_signed_mean"
                                ] is None
                                else f"{rc['normal_cos_signed_mean']:.4f}"
                            )

                            y_text = (
                                "NA"
                                if rc[
                                    "y_cos_signed_mean"
                                ] is None
                                else f"{rc['y_cos_signed_mean']:.4f}"
                            )

                            if (
                                estimator
                                == RAGTtBASubspaceNormalDiagnosticAttack.SUBSPACE_PAIRED
                            ):
                                info_text = (
                                    "NA"
                                    if info_rate is None
                                    else f"{info_rate:.3f}"
                                )

                                print(
                                    f"  B={budget:<4} "
                                    f"r={r:<4} "
                                    f"{estimator:<21} "
                                    f"nCos={n_text:<8} "
                                    f"yCos={y_text:<8} "
                                    f"info={info_text}"
                                )

                            else:
                                print(
                                    f"  B={budget:<4} "
                                    f"r={r:<4} "
                                    f"{estimator:<21} "
                                    f"nCos={n_text:<8} "
                                    f"yCos={y_text:<8}"
                                )

        final_path = os.path.join(
            root,
            "base_final_summary.csv",
        )

        write_csv(
            final_path,
            final_rows,
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
