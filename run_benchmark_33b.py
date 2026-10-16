#!/usr/bin/env python3
"""Paired hard-label benchmark: LGC family, proposed variants and pixel baselines.

Runs on NVIDIA CUDA (AutoDL), Apple Silicon (MPS) or CPU. Results are appended
per run, so an interrupted job resumes where it stopped (same --out_dir / --mode).
Several processes may work on disjoint shards of the image list at the same time
(see run_autodl.sh); each shard writes its own CSV files.

Examples:
    python run_benchmark.py --mode non_targeted --preset core --num_samples 50      # one process
    bash run_autodl.sh non_targeted core 100 3                                       # 3 shards per GPU
    bash run_autodl.sh non_targeted geom 100 3                                       # proposed bridge geometry
    bash run_autodl.sh non_targeted pvr 50 3                                         # PVR + Thales one-search
    bash run_autodl.sh non_targeted lowbudget 50 3                                   # pixel low-budget levers

Bridge-based methods (TtBA, *_MB, *_HYB, *_GARC, *_PVR) also write geometry*.csv: one row per
iteration with k_bridge, the angle gamma_obs at the new boundary point (90 deg on a
hyperplane), the radius prediction error, the queries spent per stage and the share of +1
probe answers. runs*.csv records mean_sign_share / one_sided_pct for every probe-based method.
A results directory from an older release is continued in *_v2.csv files (new columns).

Build 32a additions: preset ``pvre_32a`` (TtBA_PVRE32 core, the A2 query-budget cutoff
ablation, Tier B LGC_T tangent-chord rows), CLI flags --pvr_gte_model_cutoff_q /
--pvr_gte_eps_bracket / --pvr_gte_debug_escape_parity / --tan_rank / --tan_frac, and the
geometry*.csv columns dbg_gate_q / dbg_escape_src (A3 diagnostic, opt-in).

Build 32b additions: presets ``pvre_32b`` (frozen Tier-A ladder: TtBA_PVRE32F, the A3 narrow-tolerance
row TtBA_PVRE32F_N2 and the A4 eps-bracket row TtBA_PVRE32F_E, with TtBA, CGBA and TtBA_PVRE_GE)
and ``tierb_32b`` (corrected tangent-chord rows LGC_TB / LGC_TB_U / LGC_TBF, run ONLY after the
diagnose_tangent.py gate passes), and CLI flag --tan_energy_ratio (frozen rho = E_J / E_S_perp).
The tangent basis is built with the thin-QR tangent_chord_basis (the 32a full SVD needed 181 GB).

Build 33a additions: presets ``lgc_33a`` (targeted: late latent->pixel switch ladder LGC_RC_H2500G /
H5000G / H7500G with pixel stage TtBA_PVRE_GNE, plus the pixel candidate TtBA_PVRE32F_NE) and ``pvre_33a``
(non-targeted: TtBA_PVRE32F_NE and the probe-policy control TtBA_PVRE_GE_CP).  Use --out_dir Results/32b.

Build 33b additions (LGC-PVRE, the single proposed method): presets ``lgc_33b`` (the single-variable ladder
LGC_RC -> LGC_RC_GE -> LGC_RC_GTE -> LGC_RC_PVRE0 -> LGC_RC_PVRE) and ``lgc_33b_main`` (LGC_RC vs LGC_RC_PVRE),
runs.csv columns pb_* (pull-back chart diagnostics, 0 queries).  Use a fresh --out_dir (e.g. Results/33b).
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
faulthandler.enable()
try:  # line-buffered output so `nohup ... > log` shows progress immediately
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass
if hasattr(signal, "SIGUSR1"):  # `kill -USR1 <pid>` prints where a long run currently is
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
print("[*] importing torch / torchvision (first run after install can take 1-2 min) ...", flush=True)

import argparse  # noqa: E402
import csv  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

from lgc_bench.analysis import print_quick_table, write_report  # noqa: E402
from lgc_bench.data import apply_dataset_defaults, load_image, prepare_split, read_labels  # noqa: E402
try:  # build 34 protocol module: needed only with --split_csv / --image_loader baseline
    from lgc_bench.baseline_protocol import baseline_tensor, duplicate_sources, read_split, split_meta  # noqa: E402
    _PROTOCOL_ERR = None
except ImportError as _err:  # keep every other mode runnable on a repo without the build-34 files
    baseline_tensor = duplicate_sources = read_split = split_meta = None
    _PROTOCOL_ERR = _err
try:  # build 38: official TtBA repository protocol (--image_loader ttba_github / --victim_ckpt_dir)
    from lgc_bench.baseline_protocol import TTBA_GH_WEIGHTS, build_ttba_github_victim, ttba_github_tensor  # noqa: E402
except ImportError:
    TTBA_GH_WEIGHTS, build_ttba_github_victim, ttba_github_tensor = {}, None, None


def _need_protocol() -> None:
    if _PROTOCOL_ERR is not None:
        raise SystemExit(f"[!] --split_csv / --image_loader baseline need lgc_bench/baseline_protocol.py (build 34): "
                         f"{_PROTOCOL_ERR}")
from lgc_bench.engine import PRESETS, REGISTRY, EngineConfig, GeometricAttack  # noqa: E402
from lgc_bench.ambient import AmbientResidualSpace  # noqa: E402  (build 36 MA-PVRE)
from lgc_bench.hybrid import run_hybrid  # noqa: E402
from lgc_bench.env import (  # noqa: E402
    configure_backend,
    derive_seed,
    describe_device,
    empty_cache,
    pick_device,
    seed_everything,
)
from lgc_bench.jacobian import cached_diag_gain, cached_subspace  # noqa: E402
from lgc_bench.metrics import build_lpips, image_metrics, uint8_repair  # noqa: E402
from lgc_bench.models import load_autoencoder, load_classifier  # noqa: E402
from lgc_bench.samplers import tangent_chord_basis  # noqa: E402  (build 32a Tier B)
from lgc_bench.spaces import (  # noqa: E402
    HardLabelOracle,
    LatentRAGSpace,
    LatentResidualSpace,
    PixelSpace,
    WhitenedLatentSpace,
)

CKPT_FIELDS = [
    "dataset", "arch", "mode", "image", "source_label", "target_label", "method", "space", "budget",
    "found", "state_query", "pix_l2", "pix_linf", "rmse", "lat_l2", "ssim", "lpips",
    "adv_float", "adv_uint8", "pix_l2_uint8", "adv_uint8_repaired", "pix_l2_uint8_repaired",
    "repair_scale", "repair_queries", "seed",
]
RUN_FIELDS = [
    "dataset", "arch", "mode", "image", "method", "space", "status", "total_queries",
    "q_init", "q_boundary", "q_normal", "q_update", "iterations", "wall_time_s",
    "final_state_query", "mean_cos_v_n", "mean_k_bridge", "final_ta_alpha", "error",
    "mean_sign_share", "one_sided_pct", "mean_n_probes", "switch_q", "escape_pct",
    "track_rho", "track_gain", "track_used",
    "qs_fallback_pct", "qs_hit_pct", "qs_K_mean", "qs_alpha_mean", "qs_extrap_pct", "qsb_span_deg_median",
    "qs_stage2_pct", "qs_hidden_pct",
    "ra_H_pct", "ra_M_pct", "ra_L_pct", "ra_lambda_mean", "ra_candidates_mean",
    "ra_race_q_mean", "ra_refine_q_mean", "ra_update_q_mean", "ra_pruned_mean", "ra_unresolved_pct", "ra_infeasible_mean",
    "ra_ttba_win_pct", "ra_bisector_win_pct", "ra_history_win_pct", "ra_escape_win_pct", "ra_pvre_win_pct",
    "ra_cgba_win_pct", "ra_mid_pvr_pct", "ra_gate_q_mean", "ra_eta_mean", "ra_hist_c_final",
    "ra_cgba_normal_q_mean", "ra_cgba_reprobe_pct", "ra_cgba_sign_share_mean",
    "ra_cgba_normal_fallback_pct",
    "ra_route_dominant", "ra_route_summary", "ra_route_sequence",
    "pvr_ua_model_pct", "pvr_ua_ttba_pct", "pvr_ua_bridge_pct", "pvr_ua_escape_pct",
    "pvr_ua_reanchor_pct", "pvr_ua_uncert_median", "pvr_ua_anchor_age_mean",
    "pvr_ua_anchor_kb_mean", "pvr_ua_choice_sequence",
    "pvr_gc_model_pct", "pvr_gc_anchor_b_pct", "pvr_gc_anchor_t_pct",
    "pvr_gc_escape_pct", "pvr_gc_rescue_pct", "pvr_gc_thales_pct",
    "pvr_gc_reanchor_pct", "pvr_gc_stall_pct", "pvr_gc_bias_median",
    "pvr_gc_scale_median", "pvr_gc_gamma_dev_median", "pvr_gc_anchor_age_mean",
    "pvr_gc_anchor_kb_mean", "pvr_gc_choice_sequence",
    "pvr_aor_model_pct", "pvr_aor_anchor_b_pct", "pvr_aor_anchor_t_pct",
    "pvr_aor_escape_pct", "pvr_aor_planar_native_pct", "pvr_aor_planar_shared_pct",
    "pvr_aor_reanchor_pct", "pvr_aor_bias_median", "pvr_aor_scale_median",
    "pvr_aor_D_median", "pvr_aor_S_median", "pvr_aor_weight_median",
    "pvr_aor_anchor_age_mean", "pvr_aor_anchor_kb_mean",
    "pvr_aor_bridge_refine_q_mean", "pvr_aor_bridge_rad_uncert_median",
    "pvr_aor_native_normal_q_mean", "pvr_aor_native_sign_share_mean",
    "pvr_aor_normal_split_deg_median", "pvr_aor_choice_sequence",
    "pvr_aou_model_pct", "pvr_aou_ttba_pct", "pvr_aou_cgba_pct",
    "pvr_aou_escape_pct", "pvr_aou_rescue_pct", "pvr_aou_trial_pct", "pvr_aou_switch_pct",
    "pvr_aou_recheck_pct", "pvr_aou_eta_median", "pvr_aou_q_action_mean",
    "pvr_aou_mu_M_final", "pvr_aou_mu_T_final", "pvr_aou_mu_C_final",
    "pvr_aou_mu_E_final", "pvr_aou_mu_R_final",
    "pvr_aou_S_median", "pvr_aou_weight_median", "pvr_aou_scale_median",
    "pvr_aou_anchor_kb_mean", "pvr_aou_anchor_age_mean",
    "pvr_aou_bridge_refine_q_mean", "pvr_aou_bridge_rad_uncert_median",
    "pvr_aou_choice_sequence",
    "pvr_caou_model_pct", "pvr_caou_ttba_pct", "pvr_caou_cgba_pct",
    "pvr_caou_escape_pct", "pvr_caou_rescue_pct",
    "pvr_caou_trial_pct", "pvr_caou_trial_start_pct", "pvr_caou_switch_pct",
    "pvr_caou_block_end_pct", "pvr_caou_recheck_pct",
    "pvr_caou_eta_median", "pvr_caou_block_eta_median", "pvr_caou_q_action_mean",
    "pvr_caou_S_median", "pvr_caou_weight_median", "pvr_caou_scale_median",
    "pvr_caou_lcb_inc_median", "pvr_caou_ucb_inc_median",
    "pvr_caou_lcb_ch_median", "pvr_caou_ucb_ch_median",
    "pvr_caou_neff_inc_median", "pvr_caou_neff_ch_median",
    "pvr_caou_choice_sequence",
    # build 31a: PVR-E v7 (Gate - Thales - Escape) and the optional mean-sign baseline estimator
    "gte_N_pct", "gte_B_pct", "gte_T_pct", "gte_F_pct", "gte_eps_median", "gte_q_update_mean",
    "gte_thales_first_q", "gte_choice_sequence", "sign_baseline_gain_mean",
    # build 32a: wide-flat model step (A2) and the A3 escape-parity diagnostic columns
    "gte_M_pct",
    # build 33b: pull-back chart diagnostics of LGC_RC_PVRE (0-query; '' for every other method)
    "pb_chart_pct", "pb_cond_median", "pb_anchor_deg_median", "pb_anchor_ratio_median",
    "pb_floor_share_median", "pb_lin_err_median", "pb_verify_err_median",
    # build 35: KKT plane / Krylov-SQP / curvature-calibrated probe scale ('' for every other method)
    "kkt_used_pct", "kkt_floor_share_median", "kkt_angle_median", "kkt_cg_res_median",
    "ksqp_S_pct", "ksqp_B_pct", "ksqp_alpha_median", "ksqp_rho_median", "ksqp_beta_median",
    "ksqp_delta_rel_median", "ksqp_q_update_mean",
    "cps_sigma_mult_median", "cps_sigma_mult_final", "cps_beta_median", "cps_gamma_median",
    "cps_agree_median", "cps_gain_median", "cps_gain_star_median", "cps_Bpart_median",
    # build 35b: why the raised / KKT chart was rejected, and its geometry
    "kkt_rej_plane_pct", "kkt_rej_frame_pct", "kkt_rej_normal_pct", "kkt_rej_verify_pct", "kkt_rej_cos_pct",
    "kkt_rej_floor_pct", "kkt_gain_ratio_median", "kkt_cond_median", "kkt_cos_median", "kkt_lam_rel_median",
    "kkt_tan2_median", "kkt_mbar_rel_median",
    # build 36: MA-PVRE block statistics
    "ma_amb_pct", "ma_first_amb_q", "ma_pshare_final", "ma_rate_L_median", "ma_rate_A_median", "ma_amb_gain_pct",
    # build 37: ambient gate and exploration
    "ma_gate_q", "ma_amb_pct_open", "ma_k_A_final", "ma_k_L_final", "ma_gate_ssim_last", "ma_gate_eps_last",
]
GEOM_FIELDS = [
    "dataset", "arch", "mode", "image", "method", "space", "it", "q", "step", "r", "g_act", "g_pred",
    "pred_err", "lin_err", "psi_deg", "psi_bridge_deg", "k_bridge", "gamma_obs_deg", "gamma_pred_deg",
    "sag_slope", "chord", "ratio", "cos_theta", "q_normal", "q_bridge", "q_radial", "sign_share",
    "c_eff", "trk_y",
    # build 32a A3 diagnostic (populated only when --pvr_gte_debug_escape_parity)
    "dbg_gate_q", "dbg_escape_src",
]

# build 32a preset (Part D): the new core, the A2 cutoff ablation, and the Tier B tangent rows.
# NOTE (pending gate): LGC_T / LGC_T_R128 / LGC_T_R512 are PENDING the B4 diagnostic
# (diagnose_tangent.py); drop them from the preset if the B4 table fails the gate.
PRESETS["pvre_32a"] = ["TtBA", "CGBA", "TtBA_PVRE32", "TtBA_PVRE32_A2",
                       "LGC_T", "LGC_T_R128", "LGC_T_R512"]
# build 32b.  LGC_T* (32a) now raise: probe-scale bug.  Pair Tier B with TtBA_PVRE_GE (same core).
PRESETS["pvre_32b"] = ["TtBA", "CGBA", "TtBA_PVRE_GE", "TtBA_PVRE32F", "TtBA_PVRE32F_N2", "TtBA_PVRE32F_E",
                      "TtBA_PVRE_GE_CP"]
# official baselines never run on real models (TA = the normal-free reference, P5)
PRESETS["baselines_32b"] = ["TA", "HSJA", "CGBA_H"]
PRESETS["tierb_32b"] = ["TtBA_PVRE_GE", "LGC_TB", "LGC_TB_U", "LGC_TBF"]
# build 33a.  Run into the SAME --out_dir as pvre_32b (Results/32b): finished (image, method) pairs are skipped,
# so TtBA / CGBA / GE / 32F_E are reused and every new row is paired on the same images.
PRESETS["lgc_33a"] = ["TtBA", "CGBA", "TtBA_PVRE32F_E", "TtBA_PVRE32F_NE", "LGC_RC",
                      "LGC_RC_H2500G", "LGC_RC_H5000G", "LGC_RC_H7500G"]          # targeted
PRESETS["pvre_33a"] = ["TtBA", "CGBA", "TtBA_PVRE_GE", "TtBA_PVRE32F_E", "TtBA_PVRE32F_NE",
                       "TtBA_PVRE_GE_CP"]                                         # non-targeted
# build 33b: LGC-PVRE (RAG + residual coordinate + pull-back PVR-E update) and its single-variable ladder
PRESETS["lgc_33b"] = ["LGC_RC", "LGC_RC_GE", "LGC_RC_GTE", "LGC_RC_PVRE0", "LGC_RC_PVRE"]
PRESETS["lgc_33b_main"] = ["LGC_RC", "LGC_RC_PVRE"]
# build 33c: run into the SAME --out_dir as lgc_33b (finished image/method pairs are skipped -> only FF runs)
PRESETS["lgc_33c"] = ["LGC_RC", "LGC_RC_GE", "LGC_RC_GTE", "LGC_RC_PVRE0", "LGC_RC_PVRE", "LGC_RC_PVRE_FF"]
# build 34b: back to the CGBA / TtBA frame; {CGBA, TtBA} x {latent chart, pull-back chart} + the frozen 33b PVRE.
# Selection split: --seed 3141 --exclude_seeds 992 2027 --num_samples 50 (both modes); analyze_34b.py decides.
PRESETS["sel_34b"] = ["LGC_RC", "LGC_RC_PB", "LGC_RC_TT", "LGC_RC_PBT", "LGC_RC_PVRE"]
PRESETS["sel_34b_h"] = ["LGC_H_RC", "LGC_RC_PBH"]  # optional CGBA-H pair (the paper's LGC-H)
# build 35: latent-space optimisation beyond the radial (CGBA/TtBA) plane.  References: LGC_RC (CGBA update in the
# latent residual coordinate), LGC_RC_PB (34b: CGBA semicircle in the pull-back chart), LGC_RC_PVRE (33b).
# Each new row changes ONE ingredient:
#   RN   : LGC_RC_PVRE in the contravariant plane span(xi - xi_s, G^-1 n)   (headline, Theorem 8)
#   RNC  : LGC_RC_PB   in the contravariant plane                          (same plane, CGBA semicircle in it)
#   KKT  : LGC_RC_PVRE in span(J^T e, n);  KKTN: in span(G^-1 J^T e, G^-1 n)
#   KSQP : faithful external Krylov-SQP proposal (closed-form tangent step + metric-normal corrector + trust region)
#   *_CPS: curvature-calibrated probe radius (normal estimator only)
# Selection split: --seed 3141 --exclude_seeds 992 2027 --num_samples 30..50, both modes (README_35.md).
PRESETS["innov_35"] = ["LGC_RC", "LGC_RC_PVRE", "LGC_RC_PB", "LGC_RC_RN", "LGC_RC_RNC"]
PRESETS["innov_35_ext"] = ["LGC_RC_KKT", "LGC_RC_KKTN", "LGC_RC_KSQP", "LGC_RC_PVRE_CPS", "LGC_RC_RN_CPS",
                           "LGC_RC_CPS"]
PRESETS["innov_35_all"] = PRESETS["innov_35"] + PRESETS["innov_35_ext"]
# build 35b: damping scan (run into the SAME --out_dir as innov_35: LGC_RC / PVRE / PB are reused, paired)
PRESETS["rn_scan_35b"] = ["LGC_RC", "LGC_RC_PVRE", "LGC_RC_PB", "LGC_RC_RN_L01", "LGC_RC_RN_L1", "LGC_RC_RN_W",
                          "LGC_RC_RNC_L1"]
# build 36: corrected versions of the 2026-10-06 idea list (README_36.md).  Smoke first (MA_X must equal PVRE).
PRESETS["smoke_36"] = ["LGC_RC_PVRE", "LGC_RC_MA_X", "LGC_RC_MA", "LGC_RC_RN_L1"]
PRESETS["ideas_36"] = ["LGC_RC", "LGC_RC_PVRE", "LGC_RC_PB", "LGC_RC_MA", "LGC_RC_MAC", "LGC_RC_PB_BL",
                       "LGC_RC_PB_CPS"]
PRESETS["ideas_36_all"] = PRESETS["ideas_36"] + ["LGC_RC_MA_S2500", "LGC_RC_RN_L01", "LGC_RC_RN_L1", "LGC_RC_RN_W",
                                                 "LGC_RC_RNC_L1", "LGC_RC_PVRE_CPS"]
# build 36 r6: references for MA-PVRE on the SAME images (pixel SOTA; latent->pixel hybrids = the "ambient only after
# the switch" ablation; ambient start 2500).  CGBA_H is the targeted CGBA variant of the CGBA paper.
PRESETS["ref_36_nt"] = ["CGBA", "TtBA", "TtBA_PVRE32F_E", "LGC_RC_H1000", "LGC_RC_HA", "LGC_RC_MA_S2500"]
PRESETS["ref_36_t"] = ["CGBA_H", "TtBA", "TtBA_PVRE32F_E", "LGC_RC_H1000", "LGC_RC_HA", "LGC_RC_MA_S2500"]
# build 37: ambient admissibility gate + doubling exploration (README_37.md)
PRESETS["smoke_37"] = ["LGC_RC", "LGC_RC_MAL_X", "LGC_RC_MALP", "LGC_RC_MACP"]
PRESETS["gate_37"] = ["LGC_RC_MAC2", "LGC_RC_MACP", "LGC_RC_MACE", "LGC_RC_MAL2", "LGC_RC_MALP"]
PRESETS["egc_37"] = ["LGC_RC_EGC", "LGC_RC_EGCI"]  # r2/r4: single-operator framework, elliptic + isotropic instance
# build 37 r5: pre-registered confirmation of the frozen LGC_RC_EGC (confirm_37.py; seed 2718, n = 100, one environment)
PRESETS["confirm_37_nt"] = ["LGC_RC_EGC", "LGC_RC_EGCI", "LGC_RC", "LGC_RC_HT1000", "CGBA", "TtBA"]
PRESETS["confirm_37_t"] = ["LGC_RC_EGC", "LGC_RC_EGCI", "LGC_RC", "LGC_RC_HT1000", "CGBA_H", "TtBA"]
PRESETS["smoke_confirm_37"] = ["LGC_RC_HT1000", "LGC_RC_EGC"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["non_targeted", "targeted"], required=True)
    p.add_argument("--dataset", choices=["places365", "imagenet"], default="places365")
    p.add_argument("--data_dir", default=None, help="default ./data/Places365 or ./data/ImageNet (by --dataset)")
    p.add_argument("--labels_file", default=None, help="default adv_test_labels.txt / val_labels.txt (by --dataset)")
    p.add_argument("--preprocess", choices=["resize", "center_crop"], default=None,
                   help="default: resize (Places365), center_crop = Resize(256)+CenterCrop(224) (ImageNet)")
    p.add_argument("--imagenet_weights", default="IMAGENET1K_V1",
                   help="torchvision weights for --dataset imagenet: IMAGENET1K_V1 (legacy pretrained=True) or DEFAULT")
    p.add_argument("--archs", nargs="+", default=["resnet50", "densenet161"])
    p.add_argument("--methods", nargs="+", default=None, help=f"any of {sorted(REGISTRY)}")
    p.add_argument("--preset", choices=sorted(PRESETS), default="core")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--max_queries", type=int, default=10000)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2500, 5000, 10000])
    p.add_argument("--seed", type=int, default=992)
    p.add_argument("--exclude_seeds", type=int, nargs="*", default=[],
                   help="held-out split: exclude every image used by the splits of these seeds (e.g. the dev seed 992)")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--query_batch", type=int, default=64)
    # RA-MB-PVRE prototype (all explicit empirical knobs for sensitivity studies)
    p.add_argument("--ra_mid_pvr_high", type=float, default=0.80,
                   help="30c upper routing gate: TtBA_PVRE for 0.2 < k_b < this value; CGBA above it")
    p.add_argument("--ra_lambda_wide", type=float, default=0.70,
                   help="wide-regime threshold on lambda_b=psi_b/theta; heuristic, tune on dev then freeze")
    p.add_argument("--ra_theta_eps_deg", type=float, default=1.0,
                   help="fall back to TtBA when theta is smaller than this (lambda_b unstable)")
    p.add_argument("--ra_h_exploit", type=float, default=0.60, help="H-regime exploit ratio times k_b")
    p.add_argument("--ra_h_escape", type=float, default=0.90, help="H-regime escape ratio times k_b")
    p.add_argument("--ra_hist_beta", type=float, default=0.90, help="consecutive-M per-instance history EMA")
    p.add_argument("--ra_dedup_deg", type=float, default=1.5, help="merge candidate directions closer than this")
    p.add_argument("--ra_race_max_queries", type=int, default=8,
                   help="hard cap for candidate screening + equal-depth race, excluding winner refinement")
    p.add_argument("--ra_race_max_rounds", type=int, default=4)
    p.add_argument("--ra_refine_max_queries", type=int, default=14,
                   help="hard cap for winner-only certified radial refinement")
    # PVR-E v2 / uncertainty-aware continuation
    p.add_argument("--pvr_ua_reanchor_every", type=int, default=4,
                   help="PVR-E v2: maximum selected/model steps between measured bridge anchors")
    p.add_argument("--pvr_ua_uncert_q", type=float, default=0.90,
                   help="PVR-E v2: quantile of recent relative radius-prediction errors")
    p.add_argument("--pvr_ua_uncert_mult", type=float, default=1.0,
                   help="PVR-E v2: model direction must beat anchored TtBA by this uncertainty margin")
    p.add_argument("--pvr_ua_uncert_floor", type=float, default=0.005,
                   help="PVR-E v2: minimum relative-radius uncertainty (0.005 = 0.5%% of r)")
    p.add_argument("--pvr_ua_min_errs", type=int, default=2,
                   help="PVR-E v2: calibrated residuals required before the model direction can win")
    # PVR-E v3 / geometry-calibrated confidence controller
    p.add_argument("--pvr_gc_reanchor_every", type=int, default=6,
                   help="PVR-E v3: hard cap on consecutive model/Thales steps without a bridge")
    p.add_argument("--pvr_gc_min_model_errs", type=int, default=2,
                   help="PVR-E v3: model-ray residuals before robust confidence gating")
    p.add_argument("--pvr_gc_mad_mult", type=float, default=1.0,
                   help="PVR-E v3: robust MAD multiplier in the conservative model-gain test")
    p.add_argument("--pvr_gc_mad_floor", type=float, default=0.0025,
                   help="PVR-E v3: minimum relative-radius robust residual scale")
    p.add_argument("--pvr_gc_anchor_deadband", type=float, default=0.001,
                   help="PVR-E v3: relative-radius edge required to select TtBA over bisector at a fresh anchor")
    p.add_argument("--pvr_gc_stall_ratio", type=float, default=0.25,
                   help="PVR-E v3: R1 stall if gain is below this fraction of recent median gain")
    p.add_argument("--pvr_gc_stall_patience", type=int, default=2,
                   help="PVR-E v3: consecutive R1 stalls before one exact-TtBA rescue")
    p.add_argument("--pvr_gc_flat_tol_deg", type=float, default=3.0,
                   help="PVR-E v3: median |gamma-90| threshold for flatness-gated Thales geometry")
    p.add_argument("--pvr_gc_flat_window", type=int, default=3)
    p.add_argument("--pvr_gc_thales_steps", type=int, default=2,
                   help="PVR-E v3: maximum consecutive Thales/CGBA-geometry steps before re-anchor")
    # PVR-E v4 / adaptive-order regularization
    p.add_argument("--pvr_aor_reanchor_every", type=int, default=6,
                   help="PVR-E v4: maximum curved/model steps between bridge anchors")
    p.add_argument("--pvr_aor_min_model_errs", type=int, default=2,
                   help="PVR-E v4: curved-radius residuals before resolvability decisions")
    p.add_argument("--pvr_aor_mad_mult", type=float, default=1.0,
                   help="PVR-E v4: robust MAD multiplier in conservative model-gain test")
    p.add_argument("--pvr_aor_mad_floor", type=float, default=0.0025,
                   help="PVR-E v4: floor on robust relative-radius uncertainty")
    p.add_argument("--pvr_aor_anchor_deadband", type=float, default=0.001,
                   help="PVR-E v4: query-free B-vs-T anchor scoring deadband")
    p.add_argument("--pvr_aor_resolve_enter", type=float, default=1.0,
                   help="PVR-E v4: curvature unresolved when median D/U is <= this")
    p.add_argument("--pvr_aor_resolve_window", type=int, default=3,
                   help="PVR-E v4: consecutive evidence window for the D/U model-order test")
    p.add_argument("--pvr_aor_planar_steps", type=int, default=4,
                   help="PVR-E v4: CGBA planar iterations before a fresh model-order check")
    p.add_argument("--pvr_aor_bridge_radius_tol", type=float, default=0.0010,
                   help="PVR-E v4: refine narrow bridge while its induced relative-radius ambiguity exceeds this")
    # PVR-E v5 / adaptive-order utility controller
    p.add_argument("--pvr_aou_recheck_every", type=int, default=6,
                   help="PVR-E v5: maximum persistent wide-action steps without a fresh bridge check")
    p.add_argument("--pvr_aou_utility_beta", type=float, default=0.80,
                   help="PVR-E v5: EMA memory for observed log-progress/query utility")
    p.add_argument("--pvr_aou_switch_margin", type=float, default=0.10,
                   help="PVR-E v5: challenger must exceed incumbent utility by this fraction")
    p.add_argument("--pvr_aou_trial_interval", type=int, default=5,
                   help="PVR-E v5: wide-regime controlled challenger-trial interval")
    p.add_argument("--pvr_aou_min_dwell", type=int, default=2,
                   help="PVR-E v5: incumbent wide-action steps before a challenger trial")
    p.add_argument("--pvr_aou_geom_w_split", type=float, default=0.50,
                   help="PVR-E v5: AOR weight only orders which challenger is tried first")
    p.add_argument("--pvr_aou_min_model_errs", type=int, default=2)
    p.add_argument("--pvr_aou_mad_floor", type=float, default=0.0025)
    p.add_argument("--pvr_aou_bridge_radius_tol", type=float, default=0.0010)
    p.add_argument("--pvr_aou_narrow_probe_after", type=int, default=4,
                   help="PVR-E v5: protect this many E steps before the first exact-TtBA counter-trial")
    p.add_argument("--pvr_aou_narrow_trial_interval", type=int, default=6)
    p.add_argument("--pvr_aou_narrow_switch_margin", type=float, default=0.10)
    # PVR-E v6 / contextual confidence-aware utility controller
    # PVR-E v7 (build 31a): a-priori values; do not tune them on the evaluation images
    p.add_argument("--pvr_gte_eps_tol", type=float, default=1.0)
    p.add_argument("--pvr_gte_window", type=int, default=3)
    p.add_argument("--pvr_gte_thales_steps", type=int, default=4)
    p.add_argument("--pvr_gte_narrow_tol", type=float, default=1e-3)
    # build 32a (A2 / A3 / A4 / Tier B)
    p.add_argument("--pvr_gte_model_cutoff_q", type=int, default=None,
                   help="A2: disable the PVR model step in the pvr_gte wide-flat branch past this "
                        "many queries (None = unlimited). Part C sweeps {500,1000,1500,2500,5000}.")
    p.add_argument("--pvr_gte_eps_bracket", action="store_true",
                   help="A4: seed the radial-search bracket at the eps-predicted centre instead of "
                        "the fixed bracket_delta margin")
    p.add_argument("--pvr_gte_debug_escape_parity", action="store_true",
                   help="A3 diagnostic: add dbg_gate_q / dbg_escape_src columns to geometry*.csv")
    p.add_argument("--tan_rank", type=int, default=256, help="Tier B: tangent-basis rank r (LGC_T)")
    p.add_argument("--tan_frac", type=float, default=0.5,
                   help="Tier B: share of probes sampled in span(Q_tan) (LGC_T)")
    p.add_argument("--tan_energy_ratio", type=float, default=None,
                   help="Tier B (32b): frozen rho = E_J / E_S_perp from diagnose_tangent.py; required by "
                        "the weighted rows LGC_TB / LGC_TBF")
    p.add_argument("--pvr_caou_recheck_every", type=int, default=6)
    p.add_argument("--pvr_caou_min_model_errs", type=int, default=2)
    p.add_argument("--pvr_caou_mad_floor", type=float, default=0.0025)
    p.add_argument("--pvr_caou_bridge_radius_tol", type=float, default=0.0010)
    p.add_argument("--pvr_caou_hist_window", type=int, default=64)
    p.add_argument("--pvr_caou_time_decay", type=float, default=0.97)
    p.add_argument("--pvr_caou_bw_w", type=float, default=0.30)
    p.add_argument("--pvr_caou_bw_q", type=float, default=0.25)
    p.add_argument("--pvr_caou_bw_r", type=float, default=0.30)
    p.add_argument("--pvr_caou_bw_k", type=float, default=0.35)
    p.add_argument("--pvr_caou_conf_mult", type=float, default=1.0)
    p.add_argument("--pvr_caou_stderr_floor", type=float, default=2.5e-5)
    p.add_argument("--pvr_caou_utility_floor", type=float, default=1.0e-6)
    p.add_argument("--pvr_caou_switch_cost_frac", type=float, default=0.10)
    p.add_argument("--pvr_caou_geom_prior_frac", type=float, default=0.05)
    p.add_argument("--pvr_caou_trial_interval", type=int, default=8)
    p.add_argument("--pvr_caou_min_dwell", type=int, default=3)
    p.add_argument("--pvr_caou_trial_block", type=int, default=2)
    p.add_argument("--pvr_caou_stale_after", type=int, default=10)
    p.add_argument("--pvr_caou_min_neff", type=float, default=1.5)
    p.add_argument("--pvr_caou_sparse_strong_margin", type=float, default=0.25)
    p.add_argument("--pvr_caou_narrow_probe_after", type=int, default=5)
    p.add_argument("--pvr_caou_narrow_trial_interval", type=int, default=8)
    p.add_argument("--pvr_caou_narrow_trial_block", type=int, default=2)
    p.add_argument("--pvr_caou_narrow_switch_cost_frac", type=float, default=0.08)
    p.add_argument("--pvr_caou_narrow_sparse_strong_margin", type=float, default=0.18)
    # autoencoder
    p.add_argument("--ae_repo", default=".", help="repo root containing models/vgg.py")
    p.add_argument("--ae_checkpoint", default="checkpoints/vgg16_imagenet.pth")
    p.add_argument("--ae_arch", default="vgg16")
    # proposed Jacobian subspace
    p.add_argument("--jac_rank", type=int, default=256)
    p.add_argument("--jac_oversample", type=int, default=16)
    p.add_argument("--jac_power_iters", type=int, default=1)
    p.add_argument("--jac_batch", type=int, default=8)
    p.add_argument("--jac_fd_eps", type=float, default=1e-2)
    p.add_argument("--jac_lambda", type=float, default=1e-2)
    p.add_argument("--jac_mix", type=float, default=1.0, help="subspace share of probe energy for LGC_JS/JSM/JSMP")
    p.add_argument("--gain_probes", type=int, default=64, help="VJP probes for diag(J^T J) (LGC_W*)")
    # bookkeeping
    p.add_argument("--out_dir", default="Results/benchmark")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--regenerate_splits", action="store_true")
    # build 34: attack the (source, target) list chosen exactly like the CGBA / TtBA scripts
    p.add_argument("--split_csv", default=None,
                   help="protocol split from make_baseline_splits.py; '{arch}' and '{mode}' are substituted, e.g. "
                        "splits/imagenet/protocol_ttba/{arch}_{mode}.csv (replaces the seed split)")
    p.add_argument("--image_loader", choices=["auto", "data", "baseline", "ttba_github"], default="auto",
                   help="baseline = Resize((224,224)) + CenterCrop(224) + ToTensor as in the CGBA/TtBA scripts; "
                        "ttba_github = Resize(256) + CenterCrop(224) + ToTensor as the official TtBA repository "
                        "(build 38); auto = baseline with --split_csv, else the data module's loader")
    p.add_argument("--victim_ckpt_dir", default=None,
                   help="build 38: load the victim exactly like the official TtBA repo (torchvision constructor + "
                        "<dir>/<README weight file>, e.g. resnet50-11ad3fa6.pth = V2) instead of load_classifier")
    p.add_argument("--allow_weight_mismatch", action="store_true",
                   help="run although --imagenet_weights differs from the victim weights used to select the split")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--save_traces", action="store_true")
    p.add_argument("--diag_plane", action="store_true",
                   help="build 35 decision gate: white-box search-plane diagnostic in every pvr_pb run (dg_* lists in "
                        "the traces; implies --save_traces; DIAGNOSTIC ONLY, the attacks never see the gradient)")
    p.add_argument("--no_geom_log", action="store_true", help="do not write geometry*.csv (bridge methods)")
    p.add_argument("--save_images", action="store_true", help="PNG of the state at the last checkpoint")
    p.add_argument("--save_adv", action="store_true", help="evaluation-only: store the adversarial image at every checkpoint in <run_dir>/adv_store (advstore.py)")
    p.add_argument("--prepare_only", action="store_true",
                   help="download classifiers/LPIPS and write the splits, then exit (run once before shards)")
    p.add_argument("--no_report", action="store_true", help="skip the final analysis (automatic for shards)")
    p.add_argument("--cooldown_s", type=float, default=0.0, help="pause after each image (laptop thermals)")
    p.add_argument("--allow_tf32", action="store_true", help="faster but less exact CUDA convolutions")
    p.add_argument("--nondeterministic", action="store_true", help="allow non-deterministic cuDNN kernels")
    p.add_argument("--smoke", action="store_true", help="2 images, 1000 queries: end-to-end check")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if not (0.2 < a.ra_mid_pvr_high < 1.0):
        p.error("--ra_mid_pvr_high must be in (0.2, 1)")
    if not (0.0 < a.ra_lambda_wide <= 1.0):
        p.error("--ra_lambda_wide must be in (0, 1]")
    if not (0.0 <= a.ra_h_exploit <= 1.0 and 0.0 <= a.ra_h_escape <= 1.0):
        p.error("--ra_h_exploit/--ra_h_escape must be in [0, 1]")
    if a.ra_h_exploit > a.ra_h_escape:
        p.error("--ra_h_exploit should not exceed --ra_h_escape")
    if not (0.0 <= a.ra_hist_beta < 1.0):
        p.error("--ra_hist_beta must be in [0, 1)")
    if a.ra_dedup_deg < 0 or a.ra_race_max_queries < 1 or a.ra_race_max_rounds < 0 or a.ra_refine_max_queries < 1:
        p.error("invalid RA-MB-PVRE query/dedup settings")
    if a.pvr_ua_reanchor_every < 1:
        p.error("--pvr_ua_reanchor_every must be >= 1")
    if not (0.5 <= a.pvr_ua_uncert_q < 1.0):
        p.error("--pvr_ua_uncert_q must be in [0.5, 1)")
    if a.pvr_ua_uncert_mult < 0.0 or a.pvr_ua_uncert_floor < 0.0 or a.pvr_ua_min_errs < 1:
        p.error("invalid PVR-E v2 uncertainty settings")
    if a.pvr_gc_reanchor_every < 1 or a.pvr_gc_min_model_errs < 1:
        p.error("invalid PVR-E v3 re-anchor/model history settings")
    if a.pvr_gc_mad_mult < 0.0 or a.pvr_gc_mad_floor < 0.0 or a.pvr_gc_anchor_deadband < 0.0:
        p.error("invalid PVR-E v3 confidence settings")
    if not (0.0 <= a.pvr_gc_stall_ratio <= 1.0) or a.pvr_gc_stall_patience < 1:
        p.error("invalid PVR-E v3 stall settings")
    if a.pvr_gc_flat_tol_deg < 0.0 or a.pvr_gc_flat_window < 1 or a.pvr_gc_thales_steps < 0:
        p.error("invalid PVR-E v3 flatness/Thales settings")
    if a.pvr_aor_reanchor_every < 1 or a.pvr_aor_min_model_errs < 1:
        p.error("invalid PVR-E v4 re-anchor/model-history settings")
    if a.pvr_aor_mad_mult < 0.0 or a.pvr_aor_mad_floor < 0.0 or a.pvr_aor_anchor_deadband < 0.0:
        p.error("invalid PVR-E v4 confidence settings")
    if a.pvr_aor_resolve_enter < 0.0 or a.pvr_aor_resolve_window < 1 or a.pvr_aor_planar_steps < 0:
        p.error("invalid PVR-E v4 resolvability/planar settings")
    if a.pvr_aor_bridge_radius_tol < 0.0:
        p.error("--pvr_aor_bridge_radius_tol must be non-negative")
    if a.pvr_aou_recheck_every < 1 or a.pvr_aou_trial_interval < 1 or a.pvr_aou_min_dwell < 0:
        p.error("invalid PVR-E v5 wide-controller cadence")
    if not (0.0 <= a.pvr_aou_utility_beta < 1.0):
        p.error("--pvr_aou_utility_beta must be in [0,1)")
    if a.pvr_aou_switch_margin < 0.0 or not (0.0 <= a.pvr_aou_geom_w_split <= 1.0):
        p.error("invalid PVR-E v5 wide-controller thresholds")
    if a.pvr_aou_min_model_errs < 1 or a.pvr_aou_mad_floor < 0.0 or a.pvr_aou_bridge_radius_tol < 0.0:
        p.error("invalid PVR-E v5 model/bridge settings")
    if a.pvr_aou_narrow_probe_after < 0 or a.pvr_aou_narrow_trial_interval < 1 or a.pvr_aou_narrow_switch_margin < 0.0:
        p.error("invalid PVR-E v5 narrow-controller settings")
    if a.pvr_gte_eps_tol < 0 or a.pvr_gte_window < 1 or a.pvr_gte_thales_steps < 0 or not (0 < a.pvr_gte_narrow_tol < 0.2):
        p.error("invalid PVR-E v7 settings")
    if a.pvr_gte_model_cutoff_q is not None and a.pvr_gte_model_cutoff_q < 0:
        p.error("--pvr_gte_model_cutoff_q must be non-negative")
    if a.tan_energy_ratio is not None and a.tan_energy_ratio < 0:
        p.error("--tan_energy_ratio must be non-negative")
    if a.tan_rank < 1 or not (0.0 <= a.tan_frac <= 1.0):
        p.error("invalid Tier B tangent settings (--tan_rank >= 1, 0 <= --tan_frac <= 1)")
    if a.pvr_caou_recheck_every < 1 or a.pvr_caou_min_model_errs < 1 or a.pvr_caou_hist_window < 4:
        p.error("invalid PVR-E v6 history/recheck settings")
    if not (0.0 < a.pvr_caou_time_decay <= 1.0):
        p.error("--pvr_caou_time_decay must be in (0,1]")
    if min(a.pvr_caou_bw_w,a.pvr_caou_bw_q,a.pvr_caou_bw_r,a.pvr_caou_bw_k) <= 0:
        p.error("PVR-E v6 kernel bandwidths must be positive")
    if min(a.pvr_caou_conf_mult,a.pvr_caou_stderr_floor,a.pvr_caou_utility_floor,a.pvr_caou_switch_cost_frac,a.pvr_caou_geom_prior_frac) < 0:
        p.error("invalid PVR-E v6 confidence/cost settings")
    if a.pvr_caou_trial_interval < 1 or a.pvr_caou_min_dwell < 0 or a.pvr_caou_trial_block < 1 or a.pvr_caou_stale_after < 1:
        p.error("invalid PVR-E v6 wide macro-action settings")
    if a.pvr_caou_min_neff < 0 or a.pvr_caou_sparse_strong_margin < 0:
        p.error("invalid PVR-E v6 confidence-switch settings")
    if a.pvr_caou_narrow_probe_after < 0 or a.pvr_caou_narrow_trial_interval < 1 or a.pvr_caou_narrow_trial_block < 1:
        p.error("invalid PVR-E v6 narrow cadence")
    if a.pvr_caou_narrow_switch_cost_frac < 0 or a.pvr_caou_narrow_sparse_strong_margin < 0:
        p.error("invalid PVR-E v6 narrow confidence settings")
    if a.smoke:
        a.num_samples = min(a.num_samples, 2)
        a.max_queries = 1000
        a.checkpoints = [250, 500, 1000]
        a.jac_rank = min(a.jac_rank, 64)
        a.out_dir = a.out_dir.rstrip("/") + "_smoke"
    a.checkpoints = sorted(set(c for c in a.checkpoints if c <= a.max_queries))
    apply_dataset_defaults(a)  # data_dir / labels_file / preprocess by dataset; activates the pre-processing
    return a


_REDIRECTED: set = set()


def _schema_path(path: str, fields) -> str:
    """path, or path_v2.csv, path_v3.csv, ... : the first file whose header equals `fields`.

    A results directory written by an older release (fewer columns) is continued in a
    versioned sibling file; the analysis globs (per_checkpoint*.csv, runs*.csv,
    geometry*.csv) read all of them and align the columns.
    """
    stem, ext = os.path.splitext(path)
    cand, k = path, 1
    while os.path.exists(cand) and os.path.getsize(cand) > 0:
        with open(cand, newline="") as f:
            header = next(csv.reader(f), [])
        if header == list(fields):
            break
        k += 1
        cand = f"{stem}_v{k}{ext}"
    if cand != path and path not in _REDIRECTED:
        _REDIRECTED.add(path)
        print(f"[*] {os.path.basename(path)} has the columns of an older release; "
              f"new rows go to {os.path.basename(cand)}")
    return cand


def append_rows(path: str, fields, rows) -> None:
    path = _schema_path(path, fields)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        f.flush()


def check_data(args) -> None:
    """Stop early with instructions if the dataset folder / label file is missing."""
    labels = os.path.join(args.data_dir, args.labels_file)
    images = os.path.join(args.data_dir, "images")
    missing = [p for p in (labels, images) if not os.path.exists(p)]
    if not missing:
        return
    msg = [f"[!] dataset '{args.dataset}' not found: missing {', '.join(missing)}"]
    if args.dataset == "imagenet":
        msg += ["    Build it once (flat images/ + val_labels.txt with torchvision class indices):",
                "      python prepare_imagenet.py --find          # locates ImageNet and prints the exact command",
                "      python prepare_imagenet.py --layout tar --src /root/autodl-pub/ImageNet/ILSVRC2012 "
                "--out ./data/ImageNet --check_arch resnet50   # AutoDL public copy (example)",
                "    or pass --data_dir / --labels_file for an existing flat layout."]
    else:
        msg += ["    expected <data_dir>/images/ and <data_dir>/<labels_file> ('<file> <label>' per line);"
                " pass --data_dir / --labels_file."]
    raise SystemExit("\n".join(msg))


def excluded_images(splits_dir: str, dataset: str, arch: str, seeds) -> set:
    """Image names (sources and targets) of the splits of other seeds, both modes, ALL architectures.

    The dev split was tuned on one architecture (resnet50); a held-out split for another
    architecture (e.g. densenet161) must not reuse those images either, so every
    ``<any arch>_<mode>_seed<sd>.csv`` of the excluded seeds is read (``arch`` is kept for the
    call signature)."""
    names = set()
    for sd in seeds:
        for mode in ("non_targeted", "targeted"):
            for path in sorted(glob.glob(os.path.join(splits_dir, dataset, f"*_{mode}_seed{sd}.csv"))):
                with open(path) as f:
                    rows = list(csv.reader(f))[1:]
                for r in rows:
                    names.add(r[0])
                    if len(r) > 2 and r[2]:
                        names.add(r[2])
    return names


def protocol_pairs(args, arch: str) -> list:
    """build 34: rows of a CGBA/TtBA protocol split (make_baseline_splits.py), in protocol order.

    - the victim weights that selected the split must equal --imagenet_weights (else the 'correctly classified'
      filter was applied by a different model);
    - repeated sources are dropped (results are keyed on the source name), first occurrence kept;
    - the first --num_samples rows are attacked, i.e. the first n images the baseline script attacks.
    Images shared with --exclude_seeds splits are reported, not dropped (dropping would change the protocol)."""
    _need_protocol()
    path = args.split_csv.format(arch=arch, mode=args.mode)
    if not os.path.exists(path):
        raise SystemExit(f"[!] --split_csv {path} not found; run make_baseline_splits.py first")
    pairs = read_split(path)
    meta = split_meta(path)
    w_split = {"V1": "IMAGENET1K_V1"}.get(meta.get("victim_weights", ""), meta.get("victim_weights", ""))
    if getattr(args, "victim_ckpt_dir", None):  # build 38: the split records the official weight FILE
        expected = TTBA_GH_WEIGHTS.get(arch, "?")
        if w_split != expected:
            msg = f"[!] {path} was selected with victim weights {w_split!r}, but --victim_ckpt_dir loads {expected!r}"
            if not args.allow_weight_mismatch:
                raise SystemExit(msg + " (pass --allow_weight_mismatch to run anyway)")
            print(msg)
    elif args.dataset == "imagenet" and w_split and w_split != args.imagenet_weights:
        msg = (f"[!] {path} was selected with victim weights {w_split}, but --imagenet_weights is "
               f"{args.imagenet_weights}")
        if not args.allow_weight_mismatch:
            raise SystemExit(msg + " (pass the same weights, or --allow_weight_mismatch)")
        print(msg)
    has_t = [bool(t) for _, _, t, _ in pairs]
    if args.mode == "targeted" and not all(has_t):
        raise SystemExit(f"[!] {path}: targeted mode needs a target in every row")
    if args.mode == "non_targeted" and any(has_t):
        print(f"[*] {path}: targets present but ignored in non_targeted mode")
        pairs = [(s, sl, "", None) for s, sl, _, _ in pairs]
    dup = duplicate_sources(pairs)
    if dup:
        seen, kept = set(), []
        for row in pairs:
            if row[0] not in seen:
                seen.add(row[0])
                kept.append(row)
        print(f"[!] {path}: {len(dup)} repeated sources dropped ({len(pairs)} -> {len(kept)} rows; first kept)")
        pairs = kept
    if len(pairs) < args.num_samples:
        print(f"[!] {path}: only {len(pairs)} rows for --num_samples {args.num_samples}")
    pairs = pairs[: args.num_samples]
    excl = excluded_images(args.splits_dir, args.dataset, arch, args.exclude_seeds)
    hit = sorted({s for s, *_ in pairs} & excl) if excl else []
    if hit:
        print(f"[!] {len(hit)} protocol images are also in the splits of seeds {args.exclude_seeds} (kept): {hit[:5]}")
    print(f"[*] protocol split {path} ({meta.get('protocol', '?')}, weights {w_split or '?'}): {len(pairs)} rows")
    return pairs


def get_victim(args, arch: str, device, cache: str, weights: str):
    """build 38: the official TtBA repo's victim (--victim_ckpt_dir) or the usual load_classifier."""
    if getattr(args, "victim_ckpt_dir", None):
        if build_ttba_github_victim is None:
            raise SystemExit("[!] --victim_ckpt_dir needs lgc_bench/baseline_protocol.py from build 38")
        print(f"[*] victim {arch}: official TtBA weights {TTBA_GH_WEIGHTS.get(arch)} from {args.victim_ckpt_dir}")
        return build_ttba_github_victim(arch, args.victim_ckpt_dir, device)
    return load_classifier(args.dataset, arch, device, cache, weights)


_LOADER_CHECKED: set = set()


def _loader_tag(args) -> str:
    """Per-image caches (Jacobian / gain) are keyed on the image name; keep the two pre-processings apart."""
    if args.image_loader == "ttba_github":
        return "_ttbagh"
    return "_baseline" if (args.image_loader == "baseline" or (args.image_loader == "auto" and args.split_csv)) else ""


def load_x(args, path: str, device):
    """x0 / x_t: the data module's tensor, or the CGBA/TtBA pre-processing in the data module's tensor layout."""
    ref = load_image(path, device)
    gh = args.image_loader == "ttba_github"
    use_baseline = gh or args.image_loader == "baseline" or (args.image_loader == "auto" and args.split_csv)
    if not use_baseline:
        return ref
    _need_protocol()
    if gh and ttba_github_tensor is None:
        raise SystemExit("[!] --image_loader ttba_github needs lgc_bench/baseline_protocol.py from build 38")
    x = ttba_github_tensor(path) if gh else baseline_tensor(path)
    if x.numel() != ref.numel():
        raise SystemExit(f"[!] baseline loader gives {tuple(x.shape)} but the data module gives {tuple(ref.shape)}; "
                         "use --image_loader data or --preprocess resize")
    if float(ref.min()) < -0.01 or float(ref.max()) > 1.01:
        raise SystemExit("[!] the data module's images are not in [0, 1]; the baseline loader cannot stand in for it")
    x = x.to(device=ref.device, dtype=ref.dtype).reshape(ref.shape)
    if "msg" not in _LOADER_CHECKED:
        _LOADER_CHECKED.add("msg")
        print(f"[*] image loader: {'official TtBA repo' if gh else 'CGBA/TtBA script'} pre-processing "
              f"(mean |x - data loader| on the first image "
              f"{float((x - ref).abs().mean()):.4f})")
    return x


def completed_runs(run_dir: str) -> set:
    """Finished (arch, image, method) triples from ALL shards' runs*.csv files."""
    done = set()
    for path in glob.glob(os.path.join(run_dir, "runs*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") in ("ok", "init_failed"):
                    done.add((r["arch"], r["image"], r["method"]))
    return done


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _one_sided(shares, lo: float = 0.1, hi: float = 0.9):
    """% of normal estimates whose +1 answer share lies outside [lo, hi] (probe depth off)."""
    xs = [float(x) for x in shares if x is not None and math.isfinite(float(x))]
    return 100.0 * sum(x < lo or x > hi for x in xs) / len(xs) if xs else float("nan")


def _ra_diag(diag: dict) -> dict:
    """runs.csv summary of the reviewer-corrected RA-MB-PVRE prototype."""
    regimes = [str(x) for x in diag.get("ra_regime", [])]
    winners = [str(x) for x in diag.get("ra_winner", [])]

    # 30d logging-only addition: make the tri-route choice explicit per image.
    # The engine is unchanged: H -> TtBA, M -> TtBA_PVRE, L -> CGBA.
    route_names = []
    route_codes = []
    for w in winners:
        if w == "ttba" or "ttba_fallback" in w:
            route_names.append("TtBA")
            route_codes.append("T")
        elif w == "pvre":
            route_names.append("TtBA_PVRE")
            route_codes.append("P")
        elif w == "cgba":
            route_names.append("CGBA")
            route_codes.append("C")
        else:
            route_names.append(w)
            route_codes.append("?")

    route_counts = {name: route_names.count(name) for name in ("TtBA", "TtBA_PVRE", "CGBA")}
    if route_names:
        best_n = max(route_counts.values())
        best = [k for k, v in route_counts.items() if v == best_n and v > 0]
        route_dominant = best[0] if len(best) == 1 else "mixed(" + "/".join(best) + ")"
        route_summary = ";".join(f"{k}={route_counts[k]}/{len(route_names)}"
                                 for k in ("TtBA", "TtBA_PVRE", "CGBA"))
        route_sequence = "".join(route_codes)
    else:
        route_dominant = ""
        route_summary = ""
        route_sequence = ""

    def pct_value(xs, pred):
        return 100.0 * sum(bool(pred(x)) for x in xs) / len(xs) if xs else ""

    def mean_key(key):
        xs = diag.get(key, [])
        return _mean(xs) if xs else ""

    return {
        "ra_H_pct": pct_value(regimes, lambda x: x == "H"),
        "ra_M_pct": pct_value(regimes, lambda x: x == "M"),
        "ra_L_pct": pct_value(regimes, lambda x: x == "L"),
        "ra_lambda_mean": mean_key("ra_lambda"),
        "ra_candidates_mean": mean_key("ra_candidates"),
        "ra_race_q_mean": mean_key("ra_race_q"),
        "ra_refine_q_mean": mean_key("ra_refine_q"),
        "ra_update_q_mean": mean_key("ra_update_q"),
        "ra_pruned_mean": mean_key("ra_pruned"),
        "ra_unresolved_pct": 100.0 * mean_key("ra_unresolved") if diag.get("ra_unresolved") else "",
        "ra_infeasible_mean": mean_key("ra_infeasible"),
        "ra_ttba_win_pct": pct_value(winners, lambda x: x == "ttba" or "ttba_fallback" in x),
        "ra_bisector_win_pct": pct_value(winners, lambda x: x == "bisector"),
        "ra_history_win_pct": pct_value(winners, lambda x: x == "history"),
        "ra_escape_win_pct": pct_value(winners, lambda x: x == "escape"),
        "ra_pvre_win_pct": pct_value(winners, lambda x: x == "pvre"),
        "ra_cgba_win_pct": pct_value(winners, lambda x: x == "cgba"),
        "ra_mid_pvr_pct": 100.0 * mean_key("ra_mid_pvr") if diag.get("ra_mid_pvr") else "",
        "ra_gate_q_mean": mean_key("ra_gate_q"),
        "ra_eta_mean": mean_key("ra_eta"),
        "ra_hist_c_final": (diag.get("ra_hist_c") or [""])[-1],
        "ra_cgba_normal_q_mean": mean_key("ra_cgba_normal_q"),
        "ra_cgba_reprobe_pct": pct_value(diag.get("ra_cgba_normal_q", []), lambda x: float(x) > 0),
        "ra_cgba_sign_share_mean": mean_key("ra_cgba_sign_share"),
        "ra_cgba_normal_fallback_pct": 100.0 * mean_key("ra_cgba_normal_fallback")
            if diag.get("ra_cgba_normal_fallback") else "",
        "ra_route_dominant": route_dominant,
        "ra_route_summary": route_summary,
        "ra_route_sequence": route_sequence,
    }



def _pvr_ua_diag(diag: dict) -> dict:
    """Per-image summary of PVR-E v2 decisions; empty strings for all other methods."""
    choices = [str(x) for x in diag.get("pvr_ua_choice", [])]
    finite_u = sorted(float(x) for x in diag.get("pvr_ua_uncert", [])
                      if x is not None and math.isfinite(float(x)))

    def pct(pred):
        return 100.0 * sum(bool(pred(x)) for x in choices) / len(choices) if choices else ""

    def mean_key(key):
        xs = diag.get(key, [])
        return _mean(xs) if xs else ""

    codes = []
    for x in choices:
        if x == "model":
            codes.append("M")
        elif x == "ttba":
            codes.append("T")
        elif x == "bridge":
            codes.append("B")
        elif "escape" in x:
            codes.append("E")
        elif "fallback" in x:
            codes.append("F")
        else:
            codes.append("?")
    return {
        "pvr_ua_model_pct": pct(lambda x: x == "model"),
        "pvr_ua_ttba_pct": pct(lambda x: x == "ttba"),
        "pvr_ua_bridge_pct": pct(lambda x: x == "bridge"),
        "pvr_ua_escape_pct": pct(lambda x: "escape" in x),
        "pvr_ua_reanchor_pct": (100.0 * _mean(diag.get("pvr_ua_reanchor", []))
                                 if diag.get("pvr_ua_reanchor") else ""),
        "pvr_ua_uncert_median": finite_u[len(finite_u)//2] if finite_u else "",
        "pvr_ua_anchor_age_mean": mean_key("pvr_ua_anchor_age"),
        "pvr_ua_anchor_kb_mean": mean_key("pvr_ua_anchor_kb"),
        "pvr_ua_choice_sequence": "".join(codes),
    }


def _pvr_gc_diag(diag: dict) -> dict:
    """Per-image summary of PVR-E v3 controller decisions."""
    choices = [str(x) for x in diag.get("pvr_gc_choice", [])]
    def pct(pred):
        return 100.0 * sum(bool(pred(x)) for x in choices) / len(choices) if choices else ""
    def finite_median(key):
        xs = sorted(float(x) for x in diag.get(key, [])
                    if x is not None and math.isfinite(float(x)))
        return xs[len(xs)//2] if xs else ""
    def mean_key(key):
        xs = [float(x) for x in diag.get(key, []) if x is not None and math.isfinite(float(x))]
        return _mean(xs) if xs else ""
    code = {
        "model":"M", "anchor_b":"B", "anchor_t":"T", "escape":"E", "rescue":"R",
        "thales":"C", "fallback":"F", "model_fail":"X", "anchor_fail":"X", "bridge_fail":"X"
    }
    seq = "".join(code.get(x, "?") for x in choices)
    return {
        "pvr_gc_model_pct": pct(lambda x: x == "model"),
        "pvr_gc_anchor_b_pct": pct(lambda x: x == "anchor_b"),
        "pvr_gc_anchor_t_pct": pct(lambda x: x == "anchor_t"),
        "pvr_gc_escape_pct": pct(lambda x: x == "escape"),
        "pvr_gc_rescue_pct": pct(lambda x: x == "rescue"),
        "pvr_gc_thales_pct": pct(lambda x: x == "thales"),
        "pvr_gc_reanchor_pct": (100.0 * _mean(diag.get("pvr_gc_reanchor", []))
                                  if diag.get("pvr_gc_reanchor") else ""),
        "pvr_gc_stall_pct": (100.0 * _mean(diag.get("pvr_gc_stall", []))
                               if diag.get("pvr_gc_stall") else ""),
        "pvr_gc_bias_median": finite_median("pvr_gc_bias"),
        "pvr_gc_scale_median": finite_median("pvr_gc_scale"),
        "pvr_gc_gamma_dev_median": finite_median("pvr_gc_gamma_dev"),
        "pvr_gc_anchor_age_mean": mean_key("pvr_gc_anchor_age"),
        "pvr_gc_anchor_kb_mean": mean_key("pvr_gc_anchor_kb"),
        "pvr_gc_choice_sequence": seq,
    }


def _pvr_aor_diag(diag: dict) -> dict:
    """Per-image summary of PVR-E v4 adaptive-order decisions."""
    choices = [str(x) for x in diag.get("pvr_aor_choice", [])]
    def pct(pred):
        return 100.0 * sum(bool(pred(x)) for x in choices) / len(choices) if choices else ""
    def finite_median(key):
        xs = sorted(float(x) for x in diag.get(key, [])
                    if x is not None and math.isfinite(float(x)))
        return xs[len(xs)//2] if xs else ""
    def mean_key(key):
        xs = [float(x) for x in diag.get(key, []) if x is not None and math.isfinite(float(x))]
        return _mean(xs) if xs else ""
    code = {
        "model":"M", "anchor_b":"B", "anchor_t":"T", "escape":"E",
        "planar_native":"C", "planar_shared":"c", "fallback":"F",
        "model_fail":"X", "anchor_fail":"X", "bridge_fail":"X"
    }
    return {
        "pvr_aor_model_pct": pct(lambda x: x == "model"),
        "pvr_aor_anchor_b_pct": pct(lambda x: x == "anchor_b"),
        "pvr_aor_anchor_t_pct": pct(lambda x: x == "anchor_t"),
        "pvr_aor_escape_pct": pct(lambda x: x == "escape"),
        "pvr_aor_planar_native_pct": pct(lambda x: x == "planar_native"),
        "pvr_aor_planar_shared_pct": pct(lambda x: x == "planar_shared"),
        "pvr_aor_reanchor_pct": (100.0 * _mean(diag.get("pvr_aor_reanchor", []))
                                   if diag.get("pvr_aor_reanchor") else ""),
        "pvr_aor_bias_median": finite_median("pvr_aor_bias"),
        "pvr_aor_scale_median": finite_median("pvr_aor_scale"),
        "pvr_aor_D_median": finite_median("pvr_aor_D"),
        "pvr_aor_S_median": finite_median("pvr_aor_S"),
        "pvr_aor_weight_median": finite_median("pvr_aor_weight"),
        "pvr_aor_anchor_age_mean": mean_key("pvr_aor_anchor_age"),
        "pvr_aor_anchor_kb_mean": mean_key("pvr_aor_anchor_kb"),
        "pvr_aor_bridge_refine_q_mean": mean_key("pvr_aor_bridge_refine_q"),
        "pvr_aor_bridge_rad_uncert_median": finite_median("pvr_aor_bridge_rad_uncert"),
        "pvr_aor_native_normal_q_mean": mean_key("pvr_aor_native_normal_q"),
        "pvr_aor_native_sign_share_mean": mean_key("pvr_aor_native_sign_share"),
        "pvr_aor_normal_split_deg_median": finite_median("pvr_aor_normal_split_deg"),
        "pvr_aor_choice_sequence": "".join(code.get(x,"?") for x in choices),
    }


def _pvr_aou_diag(diag: dict) -> dict:
    """Per-image summary of PVR-E v5 utility-controlled decisions."""
    choices=[str(x) for x in diag.get("pvr_aou_choice",[])]
    def pct(pred): return 100.0*sum(bool(pred(x)) for x in choices)/len(choices) if choices else ""
    def median(key):
        xs=sorted(float(x) for x in diag.get(key,[]) if x is not None and math.isfinite(float(x)))
        return xs[len(xs)//2] if xs else ""
    def mean(key):
        xs=[float(x) for x in diag.get(key,[]) if x is not None and math.isfinite(float(x))]
        return _mean(xs) if xs else ""
    def last(key):
        xs=[float(x) for x in diag.get(key,[]) if x is not None and math.isfinite(float(x))]
        return xs[-1] if xs else ""
    code={"model":"M","ttba":"T","cgba":"C","escape":"E","rescue":"R",
          "fallback":"F","model_fail":"X","bridge_fail":"X"}
    return {
      "pvr_aou_model_pct":pct(lambda x:x in ("model","model_fail")),
      "pvr_aou_ttba_pct":pct(lambda x:x=="ttba"),
      "pvr_aou_cgba_pct":pct(lambda x:x=="cgba"),
      "pvr_aou_escape_pct":pct(lambda x:x=="escape"),
      "pvr_aou_rescue_pct":pct(lambda x:x=="rescue"),
      "pvr_aou_trial_pct":100.0*_mean(diag.get("pvr_aou_trial",[])) if diag.get("pvr_aou_trial") else "",
      "pvr_aou_switch_pct":100.0*_mean(diag.get("pvr_aou_switch",[])) if diag.get("pvr_aou_switch") else "",
      "pvr_aou_recheck_pct":100.0*_mean(diag.get("pvr_aou_recheck",[])) if diag.get("pvr_aou_recheck") else "",
      "pvr_aou_eta_median":median("pvr_aou_eta"), "pvr_aou_q_action_mean":mean("pvr_aou_q_action"),
      "pvr_aou_mu_M_final":last("pvr_aou_mu_M"), "pvr_aou_mu_T_final":last("pvr_aou_mu_T"),
      "pvr_aou_mu_C_final":last("pvr_aou_mu_C"), "pvr_aou_mu_E_final":last("pvr_aou_mu_E"),
      "pvr_aou_mu_R_final":last("pvr_aou_mu_R"), "pvr_aou_S_median":median("pvr_aou_S"),
      "pvr_aou_weight_median":median("pvr_aou_weight"), "pvr_aou_scale_median":median("pvr_aou_scale"),
      "pvr_aou_anchor_kb_mean":mean("pvr_aou_anchor_kb"), "pvr_aou_anchor_age_mean":mean("pvr_aou_anchor_age"),
      "pvr_aou_bridge_refine_q_mean":mean("pvr_aou_bridge_refine_q"),
      "pvr_aou_bridge_rad_uncert_median":median("pvr_aou_bridge_rad_uncert"),
      "pvr_aou_choice_sequence":"".join(code.get(x,"?") for x in choices),
    }

def _pvr_caou_diag(diag: dict) -> dict:
    choices=[str(x) for x in diag.get("pvr_caou_choice",[])]
    def finite(key): return [float(x) for x in diag.get(key,[]) if isinstance(x,(int,float)) and math.isfinite(float(x))]
    def mean(key):
        xs=finite(key); return sum(xs)/len(xs) if xs else ""
    def median(key):
        xs=sorted(finite(key)); return xs[len(xs)//2] if xs else ""
    def pct(pred): return 100.0*sum(1 for x in choices if pred(x))/len(choices) if choices else ""
    code={"model":"M","model_fail":"X","ttba":"T","cgba":"C","escape":"E","rescue":"R","fallback":"F","bridge_fail":"!"}
    return {
      "pvr_caou_model_pct":pct(lambda x:x in ("model","model_fail")),
      "pvr_caou_ttba_pct":pct(lambda x:x=="ttba"), "pvr_caou_cgba_pct":pct(lambda x:x=="cgba"),
      "pvr_caou_escape_pct":pct(lambda x:x=="escape"), "pvr_caou_rescue_pct":pct(lambda x:x=="rescue"),
      "pvr_caou_trial_pct":100.0*mean("pvr_caou_trial") if mean("pvr_caou_trial")!="" else "",
      "pvr_caou_trial_start_pct":100.0*mean("pvr_caou_trial_start") if mean("pvr_caou_trial_start")!="" else "",
      "pvr_caou_switch_pct":100.0*mean("pvr_caou_switch") if mean("pvr_caou_switch")!="" else "",
      "pvr_caou_block_end_pct":100.0*mean("pvr_caou_block_end") if mean("pvr_caou_block_end")!="" else "",
      "pvr_caou_recheck_pct":100.0*mean("pvr_caou_recheck") if mean("pvr_caou_recheck")!="" else "",
      "pvr_caou_eta_median":median("pvr_caou_eta"), "pvr_caou_block_eta_median":median("pvr_caou_block_eta"),
      "pvr_caou_q_action_mean":mean("pvr_caou_q_action"), "pvr_caou_S_median":median("pvr_caou_S"),
      "pvr_caou_weight_median":median("pvr_caou_weight"), "pvr_caou_scale_median":median("pvr_caou_scale"),
      "pvr_caou_lcb_inc_median":median("pvr_caou_lcb_inc"), "pvr_caou_ucb_inc_median":median("pvr_caou_ucb_inc"),
      "pvr_caou_lcb_ch_median":median("pvr_caou_lcb_ch"), "pvr_caou_ucb_ch_median":median("pvr_caou_ucb_ch"),
      "pvr_caou_neff_inc_median":median("pvr_caou_neff_inc"), "pvr_caou_neff_ch_median":median("pvr_caou_neff_ch"),
      "pvr_caou_choice_sequence":"".join(code.get(x,"?") for x in choices),
    }

def _pb_diag(diag: dict) -> dict:
    """runs.csv pb_* columns (build 33b pull-back chart; '' when the method has no chart)."""
    fb = [float(x) for x in diag.get("pb_fallback", [])]
    keys = ("pb_chart_pct", "pb_cond_median", "pb_anchor_deg_median", "pb_anchor_ratio_median",
            "pb_floor_share_median", "pb_lin_err_median", "pb_verify_err_median")
    if not fb:
        return {k: "" for k in keys}

    def med(key):
        xs = sorted(float(x) for x in diag.get(key, []) if x is not None and math.isfinite(float(x)))
        return xs[len(xs) // 2] if xs else ""
    return {"pb_chart_pct": 100.0 * (1.0 - sum(fb) / len(fb)), "pb_cond_median": med("pb_cond"),
            "pb_anchor_deg_median": med("pb_anchor_deg"), "pb_anchor_ratio_median": med("pb_anchor_ratio"),
            "pb_floor_share_median": med("pb_floor_share"), "pb_lin_err_median": med("pb_lin_err"),
            "pb_verify_err_median": med("pb_verify_err")}


def _pvr_gte_diag(diag: dict) -> dict:
    """runs.csv columns of PVR-E v7/v8 ('' for every other method); build 33b adds the pb_* columns,
    build 35 the kkt_* / ksqp_* / cps_* columns."""
    return {**_pvr_gte_diag_core(diag), **_pb_diag(diag), **_innov_diag(diag)}


def _whitebox_normal_fn(clf, space, y0: int, yt):
    """DIAGNOSTIC ONLY (--diag_plane): z -> grad_z margin(clf(space.to_image(z))), margin > 0 on the adversarial
    side (NT: max_{c != y0} logit_c - logit_y0; targeted: logit_yt - max_{c != yt} logit_c)."""
    def fn(z: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            zz = z.detach().clone().requires_grad_(True)
            x = space.to_image(zz)
            if not x.requires_grad:
                raw = getattr(getattr(type(space), "to_image", None), "__wrapped__", None)
                if raw is not None:
                    x = raw(space, zz)
            logits = clf(x).float()
            ref = int(y0) if yt is None else int(yt)
            other = logits.clone()
            other[:, ref] = -float("inf")
            m = other.max(1).values - logits[:, ref] if yt is None else logits[:, ref] - other.max(1).values
            (g,) = torch.autograd.grad(m.sum(), zz)
        return g.detach()
    return fn


def _innov_diag(diag: dict) -> dict:
    """runs.csv columns of the build-35 rows ('' when the method does not use the ingredient)."""
    def fin(key):
        out = []
        for x in diag.get(key, []):
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fx):
                out.append(fx)
        return out

    def med(key):
        xs = sorted(fin(key))
        return xs[len(xs) // 2] if xs else ""

    out = {}
    used = fin("kkt_used")
    out["kkt_used_pct"] = 100.0 * sum(used) / len(used) if used else ""
    out["kkt_floor_share_median"] = med("kkt_floor")
    out["kkt_angle_median"] = med("kkt_angle")
    out["kkt_cg_res_median"] = med("kkt_cg_res")
    rej = [str(x) for x in diag.get("kkt_reject", [])]
    for code in ("plane", "frame", "normal", "verify", "cos", "floor"):
        out[f"kkt_rej_{code}_pct"] = 100.0 * rej.count(code) / len(rej) if rej else ""
    for k in ("gain_ratio", "cond", "cos", "lam_rel", "tan2", "mbar_rel"):
        out[f"kkt_{k}_median"] = med(f"kkt_{k}")
    ch = [str(x) for x in diag.get("ksqp_choice", [])]
    out["ksqp_S_pct"] = 100.0 * ch.count("S") / len(ch) if ch else ""
    out["ksqp_B_pct"] = 100.0 * ch.count("B") / len(ch) if ch else ""
    for k in ("alpha", "rho", "beta", "delta_rel"):
        out[f"ksqp_{k}_median"] = med(f"ksqp_{k}")
    qu = fin("ksqp_q_update")
    out["ksqp_q_update_mean"] = sum(qu) / len(qu) if qu else ""
    cm = fin("cps_mult")
    out["cps_sigma_mult_median"] = med("cps_mult")
    out["cps_sigma_mult_final"] = cm[-1] if cm else ""
    for k in ("beta", "gamma", "agree", "gain", "gain_star"):
        out[f"cps_{k}_median"] = med(f"cps_{k}")
    out["cps_Bpart_median"] = med("cps_B")  # mean-curvature share of beta at the probe scale actually used
    blk = [str(x) for x in diag.get("ma_block", [])]
    qs = fin("ma_q")
    gains = fin("ma_gain")
    out["ma_amb_pct"] = 100.0 * blk.count("A") / len(blk) if blk else ""
    first = next((qs[i] for i, b in enumerate(blk) if b == "A" and i < len(qs)), None)
    out["ma_first_amb_q"] = first if first is not None else ""
    ps = fin("ma_pshare")
    out["ma_pshare_final"] = ps[-1] if ps else ""
    out["ma_rate_L_median"] = med("ma_rate_L")
    out["ma_rate_A_median"] = med("ma_rate_A")
    tot = sum(g for g in gains)
    amb = sum(g for b, g in zip(blk, gains) if b == "A")
    out["ma_amb_gain_pct"] = 100.0 * amb / tot if (blk and tot > 0) else ""
    # build 37: gate opening query, ambient share among the iterations with the ambient block admissible
    gq = fin("ma_gate_q")
    out["ma_gate_q"] = gq[0] if gq else ""
    op = [int(x) for x in diag.get("ma_open", [])]
    blk_open = [b for b, o in zip(blk, op) if o]
    out["ma_amb_pct_open"] = 100.0 * blk_open.count("A") / len(blk_open) if blk_open else ""
    for b in ("A", "L"):
        kk = fin(f"ma_k_{b}")
        out[f"ma_k_{b}_final"] = kk[-1] if kk else ""
    gs, ge = fin("ma_gate_ssim"), fin("ma_gate_eps")
    out["ma_gate_ssim_last"] = gs[-1] if gs else ""
    out["ma_gate_eps_last"] = ge[-1] if ge else ""
    return out


def _pvr_gte_diag_core(diag: dict) -> dict:
    ch = [str(x) for x in diag.get("gte_choice", [])]
    bl = [float(x) for x in diag.get("sign_baseline_gain", []) if math.isfinite(float(x))]
    out = {"sign_baseline_gain_mean": (sum(bl) / len(bl)) if bl else ""}
    if not ch:
        return {**out, "gte_N_pct": "", "gte_B_pct": "", "gte_T_pct": "", "gte_F_pct": "", "gte_M_pct": "",
                "gte_eps_median": "", "gte_q_update_mean": "", "gte_thales_first_q": "", "gte_choice_sequence": ""}
    eps = sorted(float(x) for x in diag.get("gte_eps", []) if math.isfinite(float(x)))
    qu = [int(x) for x in diag.get("gte_q_update", [])]
    pct = lambda c: 100.0 * sum(1 for x in ch if x == c) / len(ch)
    first_t = next((i for i, x in enumerate(ch) if x == "T"), None)
    return {**out, "gte_N_pct": pct("N"), "gte_B_pct": pct("B"), "gte_T_pct": pct("T"), "gte_F_pct": pct("F"),
            "gte_M_pct": pct("M"),  # build 32a: wide-flat model steps (0 for the 31a rows)
            "gte_eps_median": eps[len(eps) // 2] if eps else "", "gte_q_update_mean": sum(qu) / max(1, len(qu)),
            "gte_thales_first_q": first_t if first_t is not None else "", "gte_choice_sequence": "".join(ch)}


def _qs_diag(diag: dict, total_queries: int = 0) -> dict:
    """runs.csv columns of the QS family ('' for every other method)."""
    def pct(key):
        xs = diag.get(key, [])
        return 100.0 * _mean(xs) if xs else ""

    def mean(key):
        xs = diag.get(key, [])
        return _mean(xs) if xs else ""

    span = sorted(float(x) for x in diag.get("qsb_span_deg", []))
    return {"qs_fallback_pct": pct("qs_fallback"), "qs_hit_pct": pct("qs_hit"), "qs_K_mean": mean("qs_K"),
            "qs_alpha_mean": mean("qs_alpha"), "qs_extrap_pct": pct("qs_extrap"),
            "qsb_span_deg_median": span[len(span) // 2] if span else "", "qs_stage2_pct": pct("qs_stage2"),
            # % of all queries that the original lgc_attack.py used but did not count (QS / QS_Bridge2 returned 0)
            "qs_hidden_pct": (100.0 * sum(diag["qs_hidden"]) / total_queries
                              if diag.get("qs_hidden") and total_queries > 0 else "")}


def main() -> None:
    args = parse_args()
    if args.diag_plane:
        args.save_traces = True  # the dg_* per-iteration lists live in traces/<arch>/<image>/<method>.json
    check_data(args)  # fail fast (before loading models) with a helpful message
    device = pick_device(args.device)
    configure_backend(device, deterministic=not args.nondeterministic, allow_tf32=args.allow_tf32)
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    print(f"[*] {describe_device(device)} | shard {args.shard_index + 1}/{args.num_shards}")

    methods = args.methods or PRESETS[args.preset]
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown methods {unknown}. Available: {sorted(REGISTRY)}")
    specs = [REGISTRY[m] for m in methods]
    need_latent = any(s.space in ("latent", "latent_w", "latent_r", "latent_ra") for s in specs)
    need_jac = any(s.needs_jacobian for s in specs)
    need_gain = any(s.needs_gain for s in specs)
    need_tan = any(s.probe == "tangent_chord" for s in specs)  # build 32a Tier B
    print(f"[*] methods: {methods}")

    cfg = EngineConfig(
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

    run_dir = os.path.join(args.out_dir, args.mode)
    os.makedirs(run_dir, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index:02d}of{args.num_shards:02d}"
    ckpt_csv = os.path.join(run_dir, f"per_checkpoint{suffix}.csv")
    runs_csv = os.path.join(run_dir, f"runs{suffix}.csv")
    geom_csv = os.path.join(run_dir, f"geometry{suffix}.csv")
    with open(os.path.join(run_dir, f"config{suffix or '_single'}.json"), "w") as f:
        json.dump({**vars(args), "methods": methods, "torch": torch.__version__,
                   "device": describe_device(device)}, f, indent=2)
    done = completed_runs(run_dir)

    encoder = decoder = None
    if need_latent or need_tan:  # Tier B needs the AE for tangent_chord_basis (0 victim queries)
        encoder, decoder = load_autoencoder(args.ae_checkpoint, args.ae_repo, device, args.ae_arch)
        print(f"[*] autoencoder {args.ae_arch} loaded from {args.ae_checkpoint}")
    lpips_net, lpips_dev = (None, None) if args.no_lpips else build_lpips(device)

    image_dir = os.path.join(args.data_dir, "images")
    img2label = read_labels(os.path.join(args.data_dir, args.labels_file))
    if not img2label:
        raise SystemExit(f"[!] {os.path.join(args.data_dir, args.labels_file)} is empty")
    jac_mem: dict = {}
    tang_mem: dict = {}  # build 32a Tier B: tangent basis per image (like jac_mem)

    if args.prepare_only:
        for arch in args.archs:
            clf = get_victim(args, arch, device, os.path.join(args.cache_dir, "classifiers"),
                                  args.imagenet_weights)
            split_file = os.path.join(args.splits_dir, args.dataset, f"{arch}_{args.mode}_seed{args.seed}.csv")
            if args.split_csv:
                pairs = protocol_pairs(args, arch)
                del clf
                continue
            excl = excluded_images(args.splits_dir, args.dataset, arch, args.exclude_seeds)
            pairs = prepare_split(split_file, image_dir, img2label, clf, device, args.mode,
                                  args.num_samples, args.seed, args.regenerate_splits, exclude=excl)
            print(f"[*] {arch}: {len(pairs)} {args.mode} samples ready in {split_file}"
                  + (f" (disjoint from {len(excl)} images of seeds {args.exclude_seeds})" if excl else ""))
            del clf
        print("[*] prepare_only: classifiers, LPIPS and splits are ready")
        return

    for arch in args.archs:
        print(f"\n{'=' * 90}\n {args.dataset.upper()} | {arch.upper()} | {args.mode.upper()}\n{'=' * 90}")
        clf = get_victim(args, arch, device, os.path.join(args.cache_dir, "classifiers"),
                              args.imagenet_weights)
        split_file = os.path.join(args.splits_dir, args.dataset, f"{arch}_{args.mode}_seed{args.seed}.csv")
        if args.split_csv:
            pairs = protocol_pairs(args, arch)
        else:
            pairs = prepare_split(split_file, image_dir, img2label, clf, device, args.mode,
                                  args.num_samples, args.seed, args.regenerate_splits,
                                  exclude=excluded_images(args.splits_dir, args.dataset, arch, args.exclude_seeds))
        indexed = list(enumerate(pairs))[args.shard_index :: args.num_shards]
        t_arch = time.time()

        for n_done, (img_idx, (s_name, s_lbl, t_name, t_lbl)) in enumerate(indexed):
            # re-read per image: a concurrent run in the same --out_dir/--mode (e.g. the curvature
            # preset next to geom) may have finished (arch, image, method) since this process started
            done |= completed_runs(run_dir)
            todo = [s for s in specs if (arch, s_name, s.name) not in done]
            if not todo:
                continue
            print(f"\n[{img_idx + 1}/{len(pairs)}] {s_name} -> {t_lbl if t_lbl is not None else 'ANY WRONG CLASS'}")
            x0 = load_x(args, os.path.join(image_dir, s_name), device)
            x_t = load_x(args, os.path.join(image_dir, t_name), device) if t_name else None
            seed_img = derive_seed(args.seed, img_idx)
            stem = os.path.splitext(s_name)[0]

            latent_space, z_s, target_seed = None, None, None
            if need_latent:
                z_s = encoder(x0)
                latent_space = LatentRAGSpace(z_s, x0, decoder)
                if x_t is not None:  # hard-label-only compensated target seed (no victim gradients)
                    pseudo = torch.clamp(x_t - (x0 - latent_space.x_rec0), 0.0, 1.0)
                    target_seed = encoder(pseudo)
            z_b = z_s
            if need_tan and z_b is None:
                # Tier B: tangent basis anchored at the decoder latent of the clean image
                z_b = encoder(x0)
            pixel_space = PixelSpace(x0)
            spaces = {"latent": latent_space, "pixel": pixel_space}
            targets = {"latent": target_seed, "pixel": None}
            if any(s.space in ("latent_r", "latent_ra") for s in todo):
                # final LGC methods: RAG latent + residual coordinate; targeted start is exactly x_t
                rspace = LatentResidualSpace(z_s, x0, decoder, encoder, x_t)
                spaces["latent_r"], targets["latent_r"] = rspace, rspace.target_point
                if any(s.space == "latent_ra" for s in todo):
                    # build 36 MA-PVRE: state [xi, p], x = clip(Phi(xi) + p); p = 0 reproduces latent_r exactly
                    aspace = AmbientResidualSpace(rspace)
                    spaces["latent_ra"], targets["latent_ra"] = aspace, aspace.target_point
            if need_gain and any(s.needs_gain for s in todo):
                gain = cached_diag_gain(os.path.join(args.cache_dir, "gain", args.ae_arch + _loader_tag(args)), stem, decoder, z_s,
                                        args.gain_probes, args.jac_batch, seed=derive_seed(args.seed, img_idx, 29))
                wspace = WhitenedLatentSpace(z_s, x0, decoder, gain)
                spaces["latent_w"] = wspace
                targets["latent_w"] = None if target_seed is None else (target_seed - z_s) / wspace.scale

            jac = None
            if need_jac and any(s.needs_jacobian and s.probe != "tangent_chord" for s in todo):
                if stem not in jac_mem:
                    jac_mem.clear()  # keep only the current image in memory
                    jac_mem[stem] = cached_subspace(
                        os.path.join(args.cache_dir, "jacobian", args.ae_arch + _loader_tag(args)), stem, decoder, z_s,
                        args.jac_rank, args.jac_oversample, args.jac_power_iters, args.jac_batch,
                        args.jac_fd_eps, seed=derive_seed(args.seed, img_idx, 17))
                jac = jac_mem[stem]

            for spec in todo:
                space = spaces[spec.space]
                oracle = HardLabelOracle(clf, space, s_lbl, t_lbl, args.max_queries, args.query_batch)
                cfg_run = cfg
                if args.diag_plane and spec.update == "pvr_pb" and space is not None:
                    cfg_run = dataclasses.replace(cfg, diag_normal_fn=_whitebox_normal_fn(
                        clf, space, s_lbl, t_lbl if args.mode == "targeted" else None))
                ckpt_rows, err, hyb_info = [], "", {}
                try:
                    jacobian = None
                    if spec.probe == "tangent_chord":  # build 32a Tier B: decoder-tangent basis
                        rank = spec.tan_rank if spec.tan_rank is not None else args.tan_rank
                        key = (stem, rank)
                        if key not in tang_mem:
                            tang_mem.clear()  # keep only the current image in memory
                            torch.manual_seed(derive_seed(args.seed, img_idx, 47))  # reproducible basis
                            tang_mem[key] = tangent_chord_basis(
                                decoder, z_b, rank=rank, oversample=args.jac_oversample,
                                eps=args.jac_fd_eps, batch=args.jac_batch)
                        jacobian = (tang_mem[key], None)
                    elif spec.needs_jacobian:
                        jacobian = jac
                    if spec.switch:  # latent -> pixel hybrid with a shared budget
                        res, hyb_info = run_hybrid(spec, cfg, space, pixel_space, clf, s_lbl, t_lbl, seed_img,
                                                   args.query_batch, target_seed=targets[spec.space],
                                                   target_image=x_t)
                    else:
                        attack = GeometricAttack(
                            spec, cfg_run, space, oracle, seed_img,
                            jacobian=jacobian if spec.needs_jacobian else None,
                            target_seed=targets[spec.space],
                            target_image=x_t if spec.space == "pixel" else None,
                        )
                        res = attack.run()
                except Exception as e:  # keep long runs alive; the run is retried on resume
                    traceback.print_exc()
                    res, err = None, f"{type(e).__name__}: {e}"

                if res is not None:
                    for cp in args.checkpoints:
                        row = {
                            "dataset": args.dataset, "arch": arch, "mode": args.mode, "image": s_name,
                            "source_label": s_lbl, "target_label": t_lbl, "method": spec.name,
                            "space": spec.space, "budget": cp, "seed": seed_img,
                        }
                        if cp in res.checkpoint_states:
                            z, sq = res.checkpoint_states[cp]
                            sp = res.checkpoint_spaces.get(cp, space)  # hybrids: latent or pixel stage
                            row["space"] = sp.name
                            img = sp.to_image(z)
                            m = image_metrics(img, x0, oracle.label_uncounted, oracle.is_adv_label,
                                              lpips_net, lpips_dev)
                            row.update(m)
                            if m["adv_uint8"] == 1:
                                row.update({"adv_uint8_repaired": 1, "pix_l2_uint8_repaired": m["pix_l2_uint8"],
                                            "repair_scale": 1.0, "repair_queries": 0})
                            else:
                                row.update(uint8_repair(z, sp.origin, sp.to_image, sp.project, x0,
                                                        oracle.label_uncounted, oracle.is_adv_label))
                            if sp.name == "latent":
                                lat = float((z - z_s).flatten().norm())
                            elif sp.name == "latent_w":
                                lat = float((sp.to_latent(z) - z_s).flatten().norm())
                            elif sp.name == "latent_r":
                                lat = float((z - sp.origin).flatten().norm())
                            elif sp.name == "latent_ra":
                                lat = sp.latent_distance(z)
                            else:
                                lat = float("nan")
                            row.update({"found": 1, "state_query": sq, "lat_l2": lat})
                            if getattr(args, "save_adv", False):  # evaluation-only (advstore): no query, no RNG
                                from advstore import store_adv
                                store_adv(os.path.join(run_dir, "adv_store"), arch, stem, spec.name, cp, img, x0,
                                          state_query=sq, adv_flag=m.get("adv_float", -1), mode=args.mode)
                            if args.save_images and cp == args.checkpoints[-1]:
                                from torchvision.utils import save_image

                                img_dir = os.path.join(run_dir, "images", arch, stem)
                                os.makedirs(img_dir, exist_ok=True)
                                save_image(img, os.path.join(img_dir, f"{spec.name}_Q{cp}.png"))
                        else:
                            row.update({"found": 0, "state_query": -1, "adv_float": 0, "adv_uint8": 0,
                                        "adv_uint8_repaired": 0})
                        ckpt_rows.append(row)
                    append_rows(ckpt_csv, CKPT_FIELDS, ckpt_rows)
                    if res.geom and not args.no_geom_log:
                        base = {"dataset": args.dataset, "arch": arch, "mode": args.mode, "image": s_name,
                                "method": spec.name, "space": spec.space}
                        append_rows(geom_csv, GEOM_FIELDS, [
                            {**base, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in g.items()}}
                            for g in res.geom])
                    if args.save_traces and res.trace:
                        tdir = os.path.join(run_dir, "traces", arch, stem)
                        os.makedirs(tdir, exist_ok=True)
                        with open(os.path.join(tdir, f"{spec.name}.json"), "w") as f:
                            json.dump({"trace": res.trace, "diag": res.diag, "by_tag": res.by_tag}, f)

                bt = res.by_tag if res is not None else {}
                append_rows(runs_csv, RUN_FIELDS, [{
                    "dataset": args.dataset, "arch": arch, "mode": args.mode, "image": s_name,
                    "method": spec.name, "space": spec.space,
                    "status": res.status if res is not None else "error",
                    "total_queries": res.queries if res is not None else oracle.queries,
                    "q_init": bt.get("init", 0), "q_boundary": bt.get("boundary", 0),
                    "q_normal": bt.get("normal", 0), "q_update": bt.get("update", 0),
                    "iterations": res.iterations if res is not None else 0,
                    "wall_time_s": round(res.wall_time, 2) if res is not None else 0,
                    "final_state_query": res.final_state_query if res is not None else -1,
                    "mean_cos_v_n": _mean(res.diag.get("cos_v_n", [])) if res is not None else "",
                    "mean_k_bridge": _mean(res.diag.get("k_bridge", [])) if res is not None else "",
                    "final_ta_alpha": (res.diag.get("ta_alpha") or [""])[-1] if res is not None else "",
                    "error": err,
                    "mean_sign_share": _mean(res.diag.get("sign_share", [])) if res is not None else "",
                    "one_sided_pct": _one_sided(res.diag.get("sign_share", [])) if res is not None else "",
                    "mean_n_probes": _mean(res.diag.get("n_probes", [])) if res is not None else "",
                    "switch_q": hyb_info.get("switch_q", ""),
                    "escape_pct": (100.0 * sum(str(g["step"]).endswith("escape") for g in res.geom)
                                   / max(1, len(res.geom))) if (res is not None and res.geom) else "",
                    # normal tracking (last stage for hybrids): rho_hat, median c^2 gain, % prior used
                    "track_rho": (res.diag.get("track_rho") or [""])[-1] if res is not None else "",
                    "track_gain": (res.diag.get("track_gain") or [""])[-1] if res is not None else "",
                    "track_used": (res.diag.get("track_used") or [""])[-1] if res is not None else "",
                    # QS family (lgc_attack.py): % iterations that fell back to LGC_H, % rays hitting within R,
                    # mean K / alpha, % accepted extrapolations, median bridge span, % iterations in LGC_H stage
                    **(_qs_diag(res.diag, res.queries) if res is not None else {}),
                    **(_ra_diag(res.diag) if res is not None else {}),
                    **(_pvr_ua_diag(res.diag) if res is not None else {}),
                    **(_pvr_gc_diag(res.diag) if res is not None else {}),
                    **(_pvr_aor_diag(res.diag) if res is not None else {}),
                    **(_pvr_aou_diag(res.diag) if res is not None else {}),
                    **(_pvr_caou_diag(res.diag) if res is not None else {}),
                    **(_pvr_gte_diag(res.diag) if res is not None else {}),
                }])
                last = ckpt_rows[-1] if ckpt_rows else {}
                if res is not None and res.status == "ok" and last.get("found") == 1:
                    geo = ""
                    if res.geom:
                        dev = sorted(abs(g["gamma_obs_deg"] - 90.0) for g in res.geom
                                     if g["step"] in ("midpoint", "garc", "hyb_bridge", "pvr_bridge", "pvr_ua_bridge",
                                                      "pvr_gc_anchor_b", "pvr_gc_anchor_t", "pvr_gc_model", "pvr_aou_model", "pvr_caou_model"))
                        n_model = sum(str(g["step"]).endswith("_model") for g in res.geom)
                        if dev:
                            geo += f" | med|gamma-90|={dev[len(dev) // 2]:.1f}deg"
                        if n_model:
                            geo += f" model_steps={n_model}/{len(res.geom)}"
                    ss = res.diag.get("sign_share", [])
                    if ss:
                        geo += f" | +1share={_mean(ss):.2f}"
                    if hyb_info:
                        geo += f" | switch@{hyb_info.get('switch_q')}"
                    n_esc = sum(str(g["step"]).endswith("escape") for g in res.geom) if res.geom else 0
                    if n_esc:
                        geo += f" escape_steps={n_esc}"
                    if res.diag.get("qs_fallback"):
                        qd = _qs_diag(res.diag, res.queries)
                        geo += f" | QS fallback={qd['qs_fallback_pct']:.0f}%"
                        if qd["qs_hidden_pct"] != "":
                            geo += f" uncounted_by_original={qd['qs_hidden_pct']:.0f}%"
                        if qd["qs_hit_pct"] != "":
                            geo += f" rays_hit={qd['qs_hit_pct']:.0f}%"
                        if qd["qsb_span_deg_median"] != "":
                            geo += f" span_med={qd['qsb_span_deg_median']:.2f}deg"
                    # 30d: show, for this image, exactly which tri-route methods were used.
                    # One image can use different routes across iterations, so report counts,
                    # percentages, dominant route, and a compact per-iteration sequence.
                    if res.diag.get("ra_winner"):
                        rd = _ra_diag(res.diag)
                        winners = [str(x) for x in res.diag.get("ra_winner", [])]
                        nroute = len(winners)
                        if nroute:
                            nt = sum((w == "ttba" or "ttba_fallback" in w) for w in winners)
                            npv = sum(w == "pvre" for w in winners)
                            nc = sum(w == "cgba" for w in winners)
                            geo += (f" | routes TtBA={nt}/{nroute}({100.0*nt/nroute:.0f}%)"
                                    f" PVRE={npv}/{nroute}({100.0*npv/nroute:.0f}%)"
                                    f" CGBA={nc}/{nroute}({100.0*nc/nroute:.0f}%)"
                                    f" dominant={rd['ra_route_dominant']}"
                                    f" seq={rd['ra_route_sequence']}")
                            if rd.get("ra_cgba_normal_q_mean", "") != "":
                                geo += (f" CGBA-nativeN={rd['ra_cgba_normal_q_mean']:.1f}q/iter"
                                        f" reprobe={rd['ra_cgba_reprobe_pct']:.0f}%")
                    if res.diag.get("pvr_ua_choice"):
                        ud = _pvr_ua_diag(res.diag)
                        geo += (f" | UA model={ud['pvr_ua_model_pct']:.0f}%"
                                f" ttba={ud['pvr_ua_ttba_pct']:.0f}%"
                                f" bridge={ud['pvr_ua_bridge_pct']:.0f}%"
                                f" escape={ud['pvr_ua_escape_pct']:.0f}%")
                        if ud["pvr_ua_uncert_median"] != "":
                            geo += f" u50={ud['pvr_ua_uncert_median']:.4f}"
                        geo += f" seq={ud['pvr_ua_choice_sequence']}"
                    if res.diag.get("pvr_gc_choice"):
                        gd = _pvr_gc_diag(res.diag)
                        geo += (f" | GC M={gd['pvr_gc_model_pct']:.0f}%"
                                f" B={gd['pvr_gc_anchor_b_pct']:.0f}%"
                                f" T={gd['pvr_gc_anchor_t_pct']:.0f}%"
                                f" E={gd['pvr_gc_escape_pct']:.0f}%"
                                f" R={gd['pvr_gc_rescue_pct']:.0f}%"
                                f" C={gd['pvr_gc_thales_pct']:.0f}%")
                        if gd["pvr_gc_scale_median"] != "":
                            geo += f" s50={gd['pvr_gc_scale_median']:.4f}"
                        if gd["pvr_gc_gamma_dev_median"] != "":
                            geo += f" gdev50={gd['pvr_gc_gamma_dev_median']:.1f}deg"
                        geo += f" seq={gd['pvr_gc_choice_sequence']}"
                    if res.diag.get("pvr_aor_choice"):
                        ad = _pvr_aor_diag(res.diag)
                        geo += (f" | AOR M={ad['pvr_aor_model_pct']:.0f}%"
                                f" B={ad['pvr_aor_anchor_b_pct']:.0f}%"
                                f" T={ad['pvr_aor_anchor_t_pct']:.0f}%"
                                f" E={ad['pvr_aor_escape_pct']:.0f}%"
                                f" C={ad['pvr_aor_planar_native_pct']:.0f}%"
                                f" c={ad['pvr_aor_planar_shared_pct']:.0f}%")
                        if ad["pvr_aor_S_median"] != "":
                            geo += f" S50={ad['pvr_aor_S_median']:.2f}"
                        if ad["pvr_aor_weight_median"] != "":
                            geo += f" w50={ad['pvr_aor_weight_median']:.2f}"
                        if ad["pvr_aor_bridge_rad_uncert_median"] != "":
                            geo += f" kU50={ad['pvr_aor_bridge_rad_uncert_median']:.4f}"
                        if ad["pvr_aor_normal_split_deg_median"] != "":
                            geo += f" nsplit50={ad['pvr_aor_normal_split_deg_median']:.1f}deg"
                        geo += f" seq={ad['pvr_aor_choice_sequence']}"
                    if res.diag.get("pvr_aou_choice"):
                        ud = _pvr_aou_diag(res.diag)
                        geo += (f" | AOU M={ud['pvr_aou_model_pct']:.0f}%"
                                f" T={ud['pvr_aou_ttba_pct']:.0f}%"
                                f" C={ud['pvr_aou_cgba_pct']:.0f}%"
                                f" E={ud['pvr_aou_escape_pct']:.0f}%"
                                f" R={ud['pvr_aou_rescue_pct']:.0f}%"
                                f" trial={ud['pvr_aou_trial_pct']:.0f}%"
                                f" sw={ud['pvr_aou_switch_pct']:.0f}%")
                        if ud["pvr_aou_eta_median"] != "":
                            geo += f" eta50={ud['pvr_aou_eta_median']:.2e}"
                        if ud["pvr_aou_S_median"] != "":
                            geo += f" S50={ud['pvr_aou_S_median']:.2f}"
                        if ud["pvr_aou_weight_median"] != "":
                            geo += f" w50={ud['pvr_aou_weight_median']:.2f}"
                        geo += f" seq={ud['pvr_aou_choice_sequence']}"
                    if res.diag.get("pvr_caou_choice"):
                        cd = _pvr_caou_diag(res.diag)
                        geo += (f" | CAOU M={cd['pvr_caou_model_pct']:.0f}%"
                                f" T={cd['pvr_caou_ttba_pct']:.0f}%"
                                f" C={cd['pvr_caou_cgba_pct']:.0f}%"
                                f" E={cd['pvr_caou_escape_pct']:.0f}%"
                                f" R={cd['pvr_caou_rescue_pct']:.0f}%"
                                f" trial={cd['pvr_caou_trial_pct']:.0f}%"
                                f" start={cd['pvr_caou_trial_start_pct']:.0f}%"
                                f" sw={cd['pvr_caou_switch_pct']:.0f}%")
                        if cd["pvr_caou_block_eta_median"] != "": geo += f" beta50={cd['pvr_caou_block_eta_median']:.2e}"
                        if cd["pvr_caou_S_median"] != "": geo += f" S50={cd['pvr_caou_S_median']:.2f}"
                        if cd["pvr_caou_weight_median"] != "": geo += f" w50={cd['pvr_caou_weight_median']:.2f}"
                        geo += f" seq={cd['pvr_caou_choice_sequence']}"
                    if res.diag.get("gte_choice"):
                        gd = _pvr_gte_diag(res.diag)
                        geo += (f" | GTE N={gd['gte_N_pct']:.0f}% B={gd['gte_B_pct']:.0f}%"
                                f" T={gd['gte_T_pct']:.0f}% F={gd['gte_F_pct']:.0f}%"
                                f" upd/it={gd['gte_q_update_mean']:.1f}")
                        if gd["gte_eps_median"] != "":
                            geo += f" eps50={gd['gte_eps_median']:.2f}"
                        geo += f" seq={gd['gte_choice_sequence']}"
                    idg = _innov_diag(res.diag)
                    if idg["kkt_used_pct"] != "":
                        geo += f" | KKT used={idg['kkt_used_pct']:.0f}%"
                        rj = {c: idg[f"kkt_rej_{c}_pct"] for c in ("frame", "verify", "cos", "floor")}
                        geo += " rej " + " ".join(f"{c}={v:.0f}%" for c, v in rj.items() if v != "" and v > 0)
                        if idg["kkt_gain_ratio_median"] != "":
                            geo += f" |Jw|/|Jv|={idg['kkt_gain_ratio_median']:.2f}"
                        if idg["kkt_angle_median"] != "":
                            geo += f" ang50={idg['kkt_angle_median']:.1f}deg"
                    if idg["ksqp_S_pct"] != "":
                        geo += (f" | KSQP S={idg['ksqp_S_pct']:.0f}% B={idg['ksqp_B_pct']:.0f}%"
                                f" upd/it={idg['ksqp_q_update_mean']:.1f}")
                        if idg["ksqp_rho_median"] != "":
                            geo += f" rho50={idg['ksqp_rho_median']:.2f}"
                    if idg["ma_amb_pct"] != "":
                        geo += f" | MA amb={idg['ma_amb_pct']:.0f}%"
                        if idg["ma_pshare_final"] != "":
                            geo += f" p-share={idg['ma_pshare_final']:.2f}"
                        if idg["ma_amb_gain_pct"] != "":
                            geo += f" amb-gain={idg['ma_amb_gain_pct']:.0f}%"
                        if idg.get("ma_gate_q", "") != "":
                            geo += f" gate@{idg['ma_gate_q']:.0f}"
                        elif idg.get("ma_gate_ssim_last", "") != "":
                            geo += f" gate closed (SSIM {idg['ma_gate_ssim_last']:.4f})"
                        elif idg.get("ma_gate_eps_last", "") != "":
                            geo += f" gate closed (eps {idg['ma_gate_eps_last']:.2f})"
                        if idg.get("ma_k_A_final", "") != "" or idg.get("ma_k_L_final", "") != "":
                            geo += f" kA={idg.get('ma_k_A_final', '') or '-'} kL={idg.get('ma_k_L_final', '') or '-'}"
                    if idg["cps_sigma_mult_final"] != "":
                        geo += (f" | CPS s_end={idg['cps_sigma_mult_final']:.2f}"
                                f" gamma50={idg['cps_gamma_median']:.2f} f50={idg['cps_gain_median']:.2f}")
                    if res.diag.get("track_rho"):
                        geo += (f" | track rho={res.diag['track_rho'][-1]:.2f}"
                                f" c2x{res.diag['track_gain'][-1]:.2f} used={res.diag['track_used'][-1]:.0f}%")
                    print(f"   {spec.name:<9} Q={res.queries:5d} | pixL2@{args.checkpoints[-1]}="
                          f"{last['pix_l2']:.4f} SSIM={last['ssim']:.4f} LPIPS={last['lpips']:.4f} "
                          f"adv float/uint8/repaired={last['adv_float']}/{last['adv_uint8']}/"
                          f"{last['adv_uint8_repaired']} | {res.wall_time:.1f}s{geo}")
                else:
                    status = res.status if res is not None else "error"
                    print(f"   {spec.name:<9} {status} {err}")
                del oracle
                empty_cache(device)

            elapsed = time.time() - t_arch
            left = len(indexed) - (n_done + 1)
            print(f"   [time] {elapsed / 60:.1f} min for {n_done + 1} images on {arch}; "
                  f"~{elapsed / (n_done + 1) * left / 60:.1f} min left on this arch (this shard)", flush=True)
            if args.cooldown_s > 0:
                time.sleep(args.cooldown_s)

        del clf
        empty_cache(device)

    if args.num_shards > 1 or args.no_report:
        print(f"\n[*] shard {args.shard_index + 1}/{args.num_shards} finished. When all shards are done run:\n"
              f"    python analyze_benchmark.py --run_dir {run_dir}")
        return
    if not glob.glob(os.path.join(run_dir, "per_checkpoint*.csv")):
        print("[!] no results were written")
        return
    print_quick_table(run_dir)
    files = write_report(run_dir)
    print("\n[*] analysis written:")
    for k, v in files.items():
        print(f"    {k:<24} {v}")


if __name__ == "__main__":
    main()
