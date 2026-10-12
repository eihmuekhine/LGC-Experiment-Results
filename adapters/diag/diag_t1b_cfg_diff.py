#!/usr/bin/env python3
"""T1(b): diff the EngineConfig of the generic runner vs run_benchmark_33b.py built from run_frame_b1.sh's argv.

Prints every dataclasses.asdict(EngineConfig) field that differs.  Allowed differences: log_geometry, save_trace,
verbose.  Any other -> exit 1 (STOP).

    python adapters/diag/diag_t1b_cfg_diff.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

import yaml  # noqa: E402
from lgc_bench.engine import EngineConfig  # noqa: E402

ALLOWED = {"log_geometry", "save_trace", "verbose"}


def contract_checkpoints() -> list:
    with open(os.path.join(REPO, "contract", "paper_contract.yaml"), encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    return [int(x) for x in c["constants"]["checkpoints"]]


def generic_cfg() -> EngineConfig:
    cps = contract_checkpoints()
    return EngineConfig(max_queries=10000, checkpoints=tuple(cps), query_batch=64,
                        log_geometry=True, save_trace=False, verbose=False)


def prereg_cfg() -> EngineConfig:
    # Exact argv run_frame_b1.sh builds for resnet50 / non_targeted (run_frame_b1.sh:50-55).
    argv = [
        "run_benchmark_33b.py",
        "--dataset", "imagenet", "--mode", "non_targeted", "--archs", "resnet50",
        "--methods", "LGC_RC_EGC",
        "--data_dir", "/root/autodl-tmp/ttba_mini_flat", "--labels_file", "ttba_mini_val_labels.txt",
        "--split_csv", "splits/imagenet/ttba_github/{arch}_{mode}.csv",
        "--image_loader", "ttba_github",
        "--victim_ckpt_dir", "/root/autodl-tmp/TtBA/model",
        "--num_samples", "200", "--max_queries", "10000",
        "--checkpoints", *[str(c) for c in contract_checkpoints()],
        "--seed", "42", "--exclude_seeds", "992", "2027", "3141", "2718",
        "--out_dir", "Results/b1_resnet50", "--num_shards", "1", "--shard_index", "0", "--save_adv",
    ]
    import run_benchmark_33b as rb

    old = sys.argv
    sys.argv = argv
    try:
        args = rb.parse_args()
    finally:
        sys.argv = old

    # Replicate run_benchmark_33b.main()'s EngineConfig construction (lines 1173-1257) exactly.
    return EngineConfig(
        max_queries=args.max_queries,
        checkpoints=tuple(args.checkpoints),
        query_batch=args.query_batch,
        jac_lambda=args.jac_lambda,
        jac_mix=args.jac_mix,
        ra_mid_pvr_high=args.ra_mid_pvr_high,
        ra_lambda_wide=args.ra_lambda_wide,
        ra_theta_eps_deg=args.ra_theta_eps_deg,
        ra_h_exploit=args.ra_h_exploit,
        ra_h_escape=args.ra_h_escape,
        ra_hist_beta=args.ra_hist_beta,
        ra_dedup_deg=args.ra_dedup_deg,
        ra_race_max_queries=args.ra_race_max_queries,
        ra_race_max_rounds=args.ra_race_max_rounds,
        ra_refine_max_queries=args.ra_refine_max_queries,
        pvr_ua_reanchor_every=args.pvr_ua_reanchor_every,
        pvr_ua_uncert_q=args.pvr_ua_uncert_q,
        pvr_ua_uncert_mult=args.pvr_ua_uncert_mult,
        pvr_ua_uncert_floor=args.pvr_ua_uncert_floor,
        pvr_ua_min_errs=args.pvr_ua_min_errs,
        pvr_gc_reanchor_every=args.pvr_gc_reanchor_every,
        pvr_gc_min_model_errs=args.pvr_gc_min_model_errs,
        pvr_gc_mad_mult=args.pvr_gc_mad_mult,
        pvr_gc_mad_floor=args.pvr_gc_mad_floor,
        pvr_gc_anchor_deadband=args.pvr_gc_anchor_deadband,
        pvr_gc_stall_ratio=args.pvr_gc_stall_ratio,
        pvr_gc_stall_patience=args.pvr_gc_stall_patience,
        pvr_gc_flat_tol_deg=args.pvr_gc_flat_tol_deg,
        pvr_gc_flat_window=args.pvr_gc_flat_window,
        pvr_gc_thales_steps=args.pvr_gc_thales_steps,
        pvr_aor_reanchor_every=args.pvr_aor_reanchor_every,
        pvr_aor_min_model_errs=args.pvr_aor_min_model_errs,
        pvr_aor_mad_mult=args.pvr_aor_mad_mult,
        pvr_aor_mad_floor=args.pvr_aor_mad_floor,
        pvr_aor_anchor_deadband=args.pvr_aor_anchor_deadband,
        pvr_aor_resolve_enter=args.pvr_aor_resolve_enter,
        pvr_aor_resolve_window=args.pvr_aor_resolve_window,
        pvr_aor_planar_steps=args.pvr_aor_planar_steps,
        pvr_aor_bridge_radius_tol=args.pvr_aor_bridge_radius_tol,
        pvr_aou_recheck_every=args.pvr_aou_recheck_every,
        pvr_aou_utility_beta=args.pvr_aou_utility_beta,
        pvr_aou_switch_margin=args.pvr_aou_switch_margin,
        pvr_aou_trial_interval=args.pvr_aou_trial_interval,
        pvr_aou_min_dwell=args.pvr_aou_min_dwell,
        pvr_aou_geom_w_split=args.pvr_aou_geom_w_split,
        pvr_aou_min_model_errs=args.pvr_aou_min_model_errs,
        pvr_aou_mad_floor=args.pvr_aou_mad_floor,
        pvr_aou_bridge_radius_tol=args.pvr_aou_bridge_radius_tol,
        pvr_aou_narrow_probe_after=args.pvr_aou_narrow_probe_after,
        pvr_aou_narrow_trial_interval=args.pvr_aou_narrow_trial_interval,
        pvr_aou_narrow_switch_margin=args.pvr_aou_narrow_switch_margin,
        pvr_gte_eps_tol=args.pvr_gte_eps_tol,
        pvr_gte_window=args.pvr_gte_window,
        pvr_gte_thales_steps=args.pvr_gte_thales_steps,
        pvr_gte_narrow_tol=args.pvr_gte_narrow_tol,
        pvr_gte_model_cutoff_q=args.pvr_gte_model_cutoff_q,
        pvr_gte_eps_bracket=args.pvr_gte_eps_bracket,
        pvr_gte_debug_escape_parity=args.pvr_gte_debug_escape_parity,
        tan_rank=args.tan_rank,
        tan_frac=args.tan_frac,
        tan_energy_ratio=args.tan_energy_ratio,
        pvr_caou_recheck_every=args.pvr_caou_recheck_every,
        pvr_caou_min_model_errs=args.pvr_caou_min_model_errs,
        pvr_caou_mad_floor=args.pvr_caou_mad_floor,
        pvr_caou_bridge_radius_tol=args.pvr_caou_bridge_radius_tol,
        pvr_caou_hist_window=args.pvr_caou_hist_window,
        pvr_caou_time_decay=args.pvr_caou_time_decay,
        pvr_caou_bw_w=args.pvr_caou_bw_w, pvr_caou_bw_q=args.pvr_caou_bw_q,
        pvr_caou_bw_r=args.pvr_caou_bw_r, pvr_caou_bw_k=args.pvr_caou_bw_k,
        pvr_caou_conf_mult=args.pvr_caou_conf_mult, pvr_caou_stderr_floor=args.pvr_caou_stderr_floor,
        pvr_caou_utility_floor=args.pvr_caou_utility_floor, pvr_caou_switch_cost_frac=args.pvr_caou_switch_cost_frac,
        pvr_caou_geom_prior_frac=args.pvr_caou_geom_prior_frac, pvr_caou_trial_interval=args.pvr_caou_trial_interval,
        pvr_caou_min_dwell=args.pvr_caou_min_dwell, pvr_caou_trial_block=args.pvr_caou_trial_block,
        pvr_caou_stale_after=args.pvr_caou_stale_after, pvr_caou_min_neff=args.pvr_caou_min_neff,
        pvr_caou_sparse_strong_margin=args.pvr_caou_sparse_strong_margin,
        pvr_caou_narrow_probe_after=args.pvr_caou_narrow_probe_after,
        pvr_caou_narrow_trial_interval=args.pvr_caou_narrow_trial_interval,
        pvr_caou_narrow_trial_block=args.pvr_caou_narrow_trial_block,
        pvr_caou_narrow_switch_cost_frac=args.pvr_caou_narrow_switch_cost_frac,
        pvr_caou_narrow_sparse_strong_margin=args.pvr_caou_narrow_sparse_strong_margin,
        save_trace=args.save_traces,
        log_geometry=not args.no_geom_log,
        verbose=args.verbose,
    )


def main() -> int:
    g = asdict(generic_cfg())
    p = asdict(prereg_cfg())
    keys = sorted(set(g) | set(p))
    diffs = {k: (g.get(k), p.get(k)) for k in keys if g.get(k) != p.get(k)}
    if not diffs:
        print("IDENTICAL: no EngineConfig field differs.")
        return 0
    print(f"{'field':32s}  {'generic':>12s}  {'prereg':>12s}")
    print("-" * 64)
    bad = []
    for k in sorted(diffs):
        gv, pv = diffs[k]
        print(f"{k:32s}  {str(gv):>12s}  {str(pv):>12s}")
        if k not in ALLOWED:
            bad.append(k)
    if bad:
        print(f"\nSTOP: differing fields outside the allowed set: {sorted(bad)}")
        return 1
    print("\nOK: all differences are within {log_geometry, save_trace, verbose}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
