"""Unified hard-label geometric attack engine (latent-RAG and pixel spaces).

Every method runs through the same oracle, the same initialisation protocol,
the same query accounting and the same checkpoint rules, so the comparison is
paired and fair. Update rules are ported from the official code bases:

    cgba    : CGBA semicircular sweep + BSSP          (Reza et al., ICCV 2023)
    cgba_h  : CGBA-H angular halving + radial search  (Reza et al., ICCV 2023)
    ttba    : Two-third Bridge Attack                 (Wang et al., ICML 2025)
    ta      : Triangle Attack, official get_x_hat_in_2d logic (Wang et al., ECCV 2022)
    hsja    : HopSkipJumpAttack, l2, geometric step   (Chen et al., IEEE S&P 2020)
    ngta    : normal-guided TA (TA step in span(v, n_hat)) + radial projection

Proposed bridge-geometry updates (lgc_bench/geometry.py has the formulas):
    ttba_mid: bridge bisection, then the chord midpoint (x_b + x_bridge)/2, which is
              the exact closest point for a linear boundary; short bracketed radial search
    hybrid  : right-triangle model direction (no bridge) from the pooled perpendicular
              distance d*, trust-ratio check, and bridge re-calibration when the model fails
    garc    : bridge-bisector direction + radius predicted on the inscribed-angle arc
              (gamma-arc) whose angle comes from an online hyperbolic-boundary fit
    pvr     : predict-verify-refine = hybrid + (i) gamma-arc radius prediction on model
              steps, (ii) adaptive first bracket from recent radius errors, (iii) a
              curvature-corrected secant-foot calibration from every model step (replaces
              the weak single-shot sample), (iv) bridge tolerance 1e-2 (stationarity:
              an angular error eps costs only O(eps^2) in distance)
    pvr_ua  : PVR-E v2 / uncertainty-aware PVR-E.  Keeps the published TtBA_PVRE
    pvr_gc  : PVR-E v3 / geometry-calibrated confidence controller (30g).
    pvr_aor : PVR-E v4 / adaptive-order regularized controller: curvature-resolvability
              shrinkage, effect-controlled narrow-bridge refinement, and native-CGBA
              planar bursts without double normal estimation.
              row unchanged and adds three safeguards motivated by the ImageNet diagnostics:
              (i) k_b<=0.2 uses an EXACT TtBA escape (the coarse 1e-2 bridge is refined to
              TtBA's 1e-3 tolerance and the official full radial search is used), (ii) model
              mode is periodically re-anchored by a bridge measurement, and (iii) between
              anchors a single direction is selected without a multi-branch race: the PVR
              model direction is used only when its predicted radius beats an anchored TtBA
              direction by more than the recent prediction-error uncertainty; otherwise the
              anchored TtBA direction is used.  Only the selected direction is queried.
    ramb_pvre: 30c three-regime router requested by the user.  At the current radius it first
              queries the k=0.2 gate and, only if needed, k=0.8.  Under the same local
              single-crossing assumption used by TtBA: k_b<=0.2 routes to the published TtBA
              update; 0.2<k_b<0.8 routes to the existing TtBA_PVRE update; k_b>=0.8 routes to
              the published CGBA update.  The low/mid bridge, when needed, is refined only
              inside the already certified gate bracket, avoiding a redundant [0,1] bridge.
              Published standalone TtBA/CGBA/TtBA_PVRE rows remain unchanged.
    pvr_gte : PVR-E v7 (build 31a, TtBA_PVRE_GTE) = Gate - Thales - Escape.  No model step and no
              utility controller.  Every iteration: (1) ONE gate query at k=0.2 on the current radius
              (TtBA's own threshold) decides narrow / wide under TtBA's single-crossing assumption;
              (2) narrow -> bridge inside the certified [0, 0.2] bracket to TtBA's 1e-3 tolerance, TtBA k
              rule (0.9 k_b -> 2/3 k_b), radial search bracketed at the flat prediction (PVR-E escape);
              (3) wide -> CGBA's Thales one-search with the same TtBA normal while the measured flatness
              residual eps = (g_lin - g_act) / (r - g_lin) has median |eps| <= 1 over the last 3 measured
              steps (at most 4 Thales steps in a row, then one bridge step re-measures eps); otherwise a
              bridge inside [0.2, 1] (1e-2), TtBA 2/3 rule, bracketed radial search, which measures eps.
              eps = 0 on a hyperplane and is independent of the step direction to second order; for a
              symmetric local boundary the Thales / bisector gain ratio is 4(1+eps)/(2+eps)^2.
              Ablations: gte_thales="off" (TtBA_PVRE_GE, no Thales), "always" (TtBA_PVRE_GTE_K, Thales
              whenever the gate says wide); sign_baseline (TtBA_PVRE_GTEB / TtBA_BL): HSJA's mean-sign
              baseline in the normal estimate, n ~ sum (s_i - mean s) u_i (zero extra queries).
              Build 32a (TtBA_PVRE32 / TtBA_PVRE32_A2 / LGC_T*): the wide-flat branch may choose the
              PVR model step (ported from the pvr update) before the frozen query cutoff
              pvr_gte_model_cutoff_q (A2; the 31a rows keep gte_model=False and never take it).
              Optional eps-predicted radial bracket (A4, pvr_gte_eps_bracket) and the A3
              escape-parity diagnostic columns (dbg_gate_q / dbg_escape_src).
    escape  : (spec flag, final methods *_PVRE / *_MBE) when the bridge weight k_b <= 0.2 the
              step uses TtBA's official k rule (0.9 k_b -> 2/3 k_b), i.e. TtBA's narrow-region
              escape, with a radial search bracketed at the linear-boundary prediction; above
              0.2 the bisector / model steps are kept. Targeted pixel runs spend ~80-100% of
              their steps at k_b <= 0.2, non-targeted runs 1-3%.
    track   : (spec flag, *_NT ablations) normal tracking = empirical-Bayes reuse of the previous
              normal estimate, projected orthogonal to the current direction (lgc_bench/tracking.py);
              no extra queries. Simulation: 0-2 % worse; the naive version is harmful (+7..23 %) because the
              iterate already stores the previous estimate (see tracking.py).
    thales  : (spec flag, *_PVRET) gamma-gated regime: after a bisector step with k_b > escape_k and a
              median |gamma_obs - 90| <= thales_gamma_tol (deg) over the last thales_window bisector steps,
              the next thales_recheck iterations use CGBA's Thales one-search (sweep + BSSP on the
              semicircle; exact on a flat boundary, no bridge); then a bridge step re-measures. Convex
              (gamma >> 90, targeted pixel) or narrow (k_b <= 0.2) regions keep bisector / model / escape.
    dct_factor: (spec override, *_F2 / *_F8) probe subspace = low-frequency DCT block H/f x W/f;
              the normal quality per probe is ~ E_f / d_f (energy share of the useful normal in the
              block over its dimension), the reachable floor ~ 1 / sqrt(E_f) (diagnose_frequency.py).
QS family (the user's attacks/lgc_attack.py, build 27j), latent_r, exact oracle accounting:
    qs      : fan of K rays at phi = linspace(0, pi/2, K) from x_s in span(eta, v) (+ alpha-mixed tangent),
              radial bisection 0..R per ray, duplicate bin_search, best ray, tangent extrapolation;
              K from the attacker-side SSIM gap; no hit -> LGC_H (cgba_h). FAITHFUL port.
    qs_hyb  : QS until SSIM >= 0.995, then LGC_H. FAITHFUL port of 'Hybrid'.
    qsb2    : QS_Bridge2 at radius R. FAITHFUL port, including its two geometric bugs: the angular
              bisection runs AWAY from eta (the adversarial arc lies between v and the normal), and
              phi* = phi_lo + ratio (phi_flip - phi_lo) uses the moved phi_lo, so bridge_ratio is inert.
    qsb2_o  : qsb2 with only the geometry fixed (bisection toward eta, phi* = anchor + ratio (flip - anchor));
              alpha mixing, 0..R radial bisection, duplicate bin_search and extrapolation kept (ablation).
              Still falls back: a plane tilted by alpha no longer contains x_b and misses the small adversarial
              cap on the radius-R sphere (angular radius = angle(v, in-plane normal), ~1-2 deg late).
    qsb2_oa : qsb2_o with alpha = 0 (the plane contains x_b); only the query waste of the original remains.
    qsb2_c  : corrected QS_Bridge2 = bridge on the radius-R sphere (halving + bisection to a RELATIVE
              tolerance), bisector psi_b / 2, radial search bracketed at the linear prediction
              R cos(psi_b / 2); no tangent mixing (provably harmful on a hyperplane), no extrapolation.
    qs_c    : corrected QS = the same bridge, then K rays inside the measured arc (all hit),
              bracketed radial searches, best ray.
Final LGC methods run in "latent_r" (RAG latent + one residual coordinate, exact hard-label
targeted start at x_t) and optionally switch to the pixel space (hybrid, lgc_bench/hybrid.py).
Per-method overrides: n0 (probe schedule n0 sqrt(t+1)), bridge_tol, optimal_n (probe count
N* maximising progress per query; hybrid / pvr only). "TtBA_T*" = CGBA's Thales sweep +
BSSP driven by TtBA probes (the exact one-search for a hyperplane).
Every bridge-based update logs per-iteration geometry (k_bridge, gamma_obs, ...) and every
normal estimate records the share of +1 probe answers (0.5 = balanced probe depth).

Deliberate, documented deviations (applied identically to every method):
  * candidate states are accepted only if verified adversarial AND they lower
    the objective (monotone), as in the user's comparison script;
  * CGBA sweep: if all 40 sweep points are adversarial, the last (smallest
    radius) verified point is returned instead of discarding it;
  * BSSP is initialised with the found non-adversarial point (the official code
    starts from x_0, which only affects its stopping test);
  * TtBA / radial searches verify the end point before accepting it;
  * TA: the alpha upper bound is enforced on every success path.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from .env import CPURNG
from .geometry import (
    HyperbolaFit,
    c_hat_theory,
    cos_alpha_from_bridge,
    inscribed_radius,
    model_psi,
    optimal_probe_count,
    sag_slope,
    secant_foot,
)
from .tracking import NormalTracker
from .pullback import (PBConfig, PBState, PullbackPVRE2D, chart_test_points, chart_verify_error,
                       frame_from_gram, normal_in_chart)
from . import innov35  # build 35: torch-free KSQP / CPS mathematics
from .ramb import (
    RACandidate,
    deduplicate_candidates,
    flat_radius as ramb_flat_radius,
    k_from_psi as ramb_k_from_psi,
    normalised_bridge_span,
    psi_from_k as ramb_psi_from_k,
    safe_prunable,
)
from .samplers import (
    DCTLowFreqSampler,
    GaussianSampler,
    JacobianSubspaceSampler,
    ProbeSampler,
    TAMaskAxisSampler,
    TangentChordSampler,
)
from .spaces import AttackSpace, HardLabelOracle, QueryBudgetExhausted

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    name: str
    space: str  # "latent" | "pixel" | "latent_w" | "latent_r"
    update: str  # cgba | cgba_h | ttba | ta | ngta | hsja | ttba_mid | hybrid | garc | pvr | pvr_ua | pvr_gc | pvr_aor | pvr_aou | pvr_caou | qs*
    probe: str = "gaussian"  # gaussian | dct | jacobian | none
    ta_axis: str = "gaussian"  # gaussian | jacobian | dct_mask
    metric: bool = False  # pull-back-metric direction (proposed)
    objective: str = "space"  # space | pixel (acceptance objective)
    sigma: Optional[float] = None  # probe scale override
    probe_normalize: bool = False  # TtBA official probe normalisation
    jac_mix: Optional[float] = None  # subspace share of probe energy (None -> EngineConfig.jac_mix)
    tan_rank: Optional[int] = None  # tangent_chord probe: tangent-basis rank r (None -> EngineConfig.tan_rank)
    tan_frac: Optional[float] = None  # tangent_chord probe: share of probes in span(Q_tan) (None -> EngineConfig.tan_frac)
    init: str = "noise"  # noise (CGBA/TtBA official) | blend (HSJA/TA official, pixel only)
    n0: Optional[int] = None  # probe schedule q_t = n0 sqrt(t+1) (None -> EngineConfig.n0)
    bridge_tol: Optional[float] = None  # bridge bisection tolerance in k (None -> per-update default)
    optimal_n: bool = False  # progress-optimal probe count N* (hybrid / pvr only)
    escape: bool = False  # TtBA k rule when k_b <= EngineConfig.escape_k (ttba_mid / pvr only)
    gte_thales: str = "flat"  # pvr_gte only: "flat" (eps-gated Thales) | "off" | "always"
    gte_gate_order: str = "gate_first"  # pvr_pb only (build 33c): "gate_first" (v7) | "flat_first"
    pb_rule: str = "pvre"  # pvr_pb only (build 34b): update run in the chart: pvre | cgba | cgba_h | ttba
    pb_chart: str = "pullback"  # pvr_pb only (build 34b): pullback | latent (force the latent-Euclidean chart)
    # build 35 (claude/PVRE_information_geometry_2026-10-05.md, design doc sections 16-18)
    pb_plane: str = "radial"  # pvr_pb only: search plane  radial = span(xi - xi_s, n)  (all earlier rows)
    #                           kkt_e = span(J^T e, n)  (contains the exact pixel-L2 gradient: KKT-correct fixed points)
    #                           kkt_n = span(G^-1 J^T e, G^-1 n), G = J^T J + lam I  (the KSQP plane, CG solves)
    #                           raised = span(xi - xi_s, G^-1 n)  (radial plane with the CONTRAVARIANT normal:
    #                                    the sign-probe estimate n is a covector, J^T a; the pixel-steepest
    #                                    boundary direction is the index-raised G^-1 n.  Theorem 8, design doc s19)
    cps: bool = False  # curvature-calibrated probe scale (antithetic-pair agreement + sign share -> sigma*)
    # build 35b: damping of G = J^T J + lam I for the raised / KKT planes (None -> EngineConfig.kkt_lambda_rel)
    pb_lambda_rel: Optional[float] = None  # lam = pb_lambda_rel * ||J v||^2   (mode "rel")
    pb_lambda_mode: str = "rel"  # "rel" | "wiener": lam = mean(eig J^T J) * tan^2(phi), tan^2 = pi d_eff / (2 N f)
    # build 36: MA-PVRE (update "mapvre", space "latent_ra" = lgc_bench/ambient.py): latent block with the ambient
    # block frozen, or ambient block (exact pixel plane span(e, n_pix) through x0) with the latent block frozen
    ma_start_q: int = 1000  # ambient block allowed from this query count on (latent only before)
    ma_explore_every: int = 6  # force the block not chosen for this many iterations (keeps both rates fresh)
    ma_rule_amb: str = "pvre"  # 2-D rule of the ambient block (pullback.PullbackPVRE2D rule, identity chart)
    # build 37: admissibility gate of the ambient block, exploration schedule, latent-block rule (README_37.md)
    ma_gate: str = "none"  # none | ssim (SSIM(x, x0) >= ma_gate_ssim) | elastic (eps_hat < ma_gate_eps)
    ma_gate_ssim: float = 0.99  # perceptual admissibility: ambient refinement only once the attack is imperceptible
    ma_gate_eps: float = 0.45  # floor gate: local elasticity eps = -dln d / dln q < alpha_0 / 2 (floor share > 1/2)
    ma_gate_q_elastic: int = 2500  # earliest query count of the elasticity gate (window [q/2, q])
    ma_explore: str = "fixed"  # fixed (build 36: every ma_explore_every) | doubling (losing forced trial doubles k)
    ma_explore_max: int = 48  # cap of the doubled exploration interval
    ma_latent: str = "chart"  # chart (pvr_pb step, pixel objective) | lgc (LGC_RC's own CGBA step, latent objective
    #                           while the gate is closed -> the run IS LGC_RC until the gate opens)
    gte_model: bool = False  # pvr_gte only (build 32a): allow the wide-flat branch to choose the PVR model step
    pvr_gte_model_cutoff_q: Optional[int] = None  # pvr_gte model-step cutoff override (None -> EngineConfig.pvr_gte_model_cutoff_q)
    gte_narrow_tol: Optional[float] = None  # build 32b A3: narrow-bridge tolerance override (None -> EngineConfig.pvr_gte_narrow_tol)
    gte_eps_bracket: Optional[bool] = None  # build 32b A4: eps-predicted radial bracket override (None -> EngineConfig.pvr_gte_eps_bracket)
    tan_weighted: bool = False  # build 32b Tier B: weighted two-family sign estimator (needs EngineConfig.tan_energy_ratio)
    sign_baseline: bool = False  # HSJA mean-sign baseline in the normal estimate (no extra queries)
    switch: Optional[str] = None  # hybrid latent -> pixel: "q1000", "q2500", ... or "auto"
    pixel_stage: str = "TtBA_PVRE"  # method run in pixel space after the switch
    track: bool = False  # normal tracking (empirical-Bayes rank-1 prior from the previous estimate)
    dct_factor: Optional[int] = None  # DCT probe block H/f x W/f (None -> EngineConfig.dct_factor = 4)
    thales: bool = False  # pvr only: Thales one-search (CGBA update) while the measured boundary is flat and wide
    ra_history: bool = True  # legacy RA-MB race: include per-instance history candidate in consecutive M regimes
    ra_mid_pvr: bool = False  # 30e main router: TtBA / TtBA_PVRE / CGBA using 0.2 and upper gate
    aor_native_planar: bool = False  # 30h: native CGBA probes when curvature is unresolved
    description: str = ""

    @property
    def needs_jacobian(self) -> bool:
        # tangent_chord reuses the Jacobian slot to carry its decoder-tangent basis (B2/B3)
        return self.probe in ("jacobian", "tangent_chord") or self.ta_axis == "jacobian"

    @property
    def needs_gain(self) -> bool:
        return self.space == "latent_w"


REGISTRY: Dict[str, MethodSpec] = {
    # ---- latent (RAG) family ------------------------------------------------
    "LGC": MethodSpec("LGC", "latent", "cgba", description="LGC paper: CGBA sweep+BSSP in latent"),
    "LGC_H": MethodSpec("LGC_H", "latent", "cgba_h", description="LGC-H: CGBA-H in latent"),
    "LGC_TTBA": MethodSpec("LGC_TTBA", "latent", "ttba", description="TtBA update in latent"),
    "LGC_TA": MethodSpec("LGC_TA", "latent", "ta", probe="none", ta_axis="gaussian",
                         description="Triangle Attack in latent, isotropic axis (no prior)"),
    "LGC_TA_J": MethodSpec("LGC_TA_J", "latent", "ta", probe="none", ta_axis="jacobian",
                           description="Triangle Attack in latent, Jacobian-subspace axis (fair prior)"),
    "LGC_NGTA": MethodSpec("LGC_NGTA", "latent", "ngta", description="normal-guided TA + radial projection"),
    "LGC_HSJA": MethodSpec("LGC_HSJA", "latent", "hsja", probe="none",
                           description="HSJA in the RAG latent space (Latent-HSJA-style)"),
    # ---- proposed: gain-whitened latent coordinates (diagonal pull-back metric)
    "LGC_W": MethodSpec("LGC_W", "latent_w", "cgba",
                        description="LGC (CGBA) in decoder-gain-whitened latent coordinates"),
    "LGC_W_TTBA": MethodSpec("LGC_W_TTBA", "latent_w", "ttba",
                             description="TtBA in decoder-gain-whitened latent coordinates"),
    "LGC_W_HSJA": MethodSpec("LGC_W_HSJA", "latent_w", "hsja", probe="none",
                             description="HSJA in decoder-gain-whitened latent coordinates"),
    # ---- Jacobian top-subspace ablation (refuted in the smoke test: high-gain drift)
    "LGC_JS": MethodSpec("LGC_JS", "latent", "cgba", probe="jacobian",
                         description="V1: LGC + decoder-Jacobian subspace normal estimation"),
    "LGC_JSX": MethodSpec("LGC_JSX", "latent", "cgba", probe="jacobian", jac_mix=0.5,
                          description="V1b: half subspace / half isotropic probes (safeguard)"),
    "LGC_JSM": MethodSpec("LGC_JSM", "latent", "cgba", probe="jacobian", metric=True,
                          description="V2: V1 + pull-back-metric direction (J^T J + lam I)^-1 n"),
    "LGC_JSMP": MethodSpec("LGC_JSMP", "latent", "cgba", probe="jacobian", metric=True, objective="pixel",
                           description="V3: V2 + pixel-L2 acceptance"),
    # ---- build 32a Tier B: floor-free latent tangent-chord probing (LGC_T family).
    # TODO(claude-review): B1-B3 are PENDING the B4 diagnostic gate (diagnose_tangent.py): the
    # mixture's c^2 ratio vs. DCT-f4 must be >= 1.5 at >= 3 of 5 checkpoints AND >= 1.0 at all 5
    # in both modes before these rows are benchmarked.  gte_model=True (not listed in the B3 rows)
    # is required because the wide-flat model-step choice is NEW in this engine (see A2); drop the
    # flag if the AutoDL base already carries the choice.
    "LGC_T": MethodSpec("LGC_T", "pixel", "pvr_gte", probe="tangent_chord", escape=True,
                        gte_thales="off", gte_model=True, tan_rank=256, tan_frac=0.5,
                        description="Tier B: TtBA_PVRE32 core + floor-free tangent-chord probe mix "
                                    "(pixel iterate, latent-tangent + DCT-orthogonal normal estimate)"),
    "LGC_T_R128": MethodSpec("LGC_T_R128", "pixel", "pvr_gte", probe="tangent_chord", escape=True,
                             gte_thales="off", gte_model=True, tan_rank=128, tan_frac=0.5,
                             description="Tier B, r=128"),
    "LGC_T_R512": MethodSpec("LGC_T_R512", "pixel", "pvr_gte", probe="tangent_chord", escape=True,
                             gte_thales="off", gte_model=True, tan_rank=512, tan_frac=0.5,
                             description="Tier B, r=512"),
    # ---- pixel-space baselines ----------------------------------------------
    "HSJA": MethodSpec("HSJA", "pixel", "hsja", probe="none", init="blend",
                       description="HopSkipJumpAttack (l2), official blended-uniform-noise init"),
    "CGBA": MethodSpec("CGBA", "pixel", "cgba", probe="dct", description="CGBA, DCT f=4"),
    "CGBA_H": MethodSpec("CGBA_H", "pixel", "cgba_h", probe="dct", description="CGBA-H, DCT f=4"),
    "TtBA": MethodSpec("TtBA", "pixel", "ttba", probe="dct", sigma=3e-4, probe_normalize=True,
                       description="TtBA, DCT f=4, official sigma/normalisation"),
    "TA": MethodSpec("TA", "pixel", "ta", probe="none", ta_axis="dct_mask", init="blend",
                     description="Triangle Attack, official low-frequency DCT axis and blended init"),
    # ---- proposed bridge geometry (latent) ----------------------------------------
    "LGC_MB": MethodSpec("LGC_MB", "latent", "ttba_mid",
                         description="bridge midpoint (exact closest point if the boundary is linear)"),
    "LGC_HYB": MethodSpec("LGC_HYB", "latent", "hybrid",
                          description="right-triangle model direction + bridge calibration"),
    "LGC_GARC": MethodSpec("LGC_GARC", "latent", "garc",
                           description="bridge bisector + gamma-arc radius from a hyperbolic boundary fit"),
    # ---- proposed bridge geometry (pixel, TtBA probes) ----------------------------
    "TtBA_MB": MethodSpec("TtBA_MB", "pixel", "ttba_mid", probe="dct", sigma=3e-4, probe_normalize=True,
                          description="TtBA probes + bridge midpoint"),
    "TtBA_HYB": MethodSpec("TtBA_HYB", "pixel", "hybrid", probe="dct", sigma=3e-4, probe_normalize=True,
                           description="TtBA probes + model direction + bridge calibration"),
    "TtBA_GARC": MethodSpec("TtBA_GARC", "pixel", "garc", probe="dct", sigma=3e-4, probe_normalize=True,
                            description="TtBA probes + bridge bisector + gamma-arc radius"),
    # ---- proposed predict-verify-refine (PVR) and exact one-search (Thales) variants
    "LGC_PVR": MethodSpec("LGC_PVR", "latent", "pvr",
                          description="predict-verify-refine: model direction, gamma-arc radius, "
                                      "secant-foot calibration, bridge tol 1e-2"),
    "LGC_N60": MethodSpec("LGC_N60", "latent", "cgba", n0=60,
                          description="LGC (CGBA update) with probe schedule n0 = 60"),
    "TtBA_PVR": MethodSpec("TtBA_PVR", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                           description="TtBA probes + predict-verify-refine"),
    "TtBA_PVR60": MethodSpec("TtBA_PVR60", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                             n0=60, description="TtBA probes + PVR, probe schedule n0 = 60"),
    "TtBA_PVRN": MethodSpec("TtBA_PVRN", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                            optimal_n=True, description="TtBA probes + PVR + progress-optimal probe count N*"),
    "TtBA_MB60": MethodSpec("TtBA_MB60", "pixel", "ttba_mid", probe="dct", sigma=3e-4, probe_normalize=True,
                            n0=60, bridge_tol=1e-2,
                            description="TtBA probes + bridge midpoint, n0 = 60, bridge tol 1e-2"),
    "TtBA_T": MethodSpec("TtBA_T", "pixel", "cgba", probe="dct", sigma=3e-4, probe_normalize=True,
                         description="TtBA probes + Thales sweep/BSSP (exact one-search on a hyperplane)"),
    "TtBA_T60": MethodSpec("TtBA_T60", "pixel", "cgba", probe="dct", sigma=3e-4, probe_normalize=True,
                           n0=60, description="TtBA probes + Thales sweep/BSSP, probe schedule n0 = 60"),
    # ---- FINAL methods: geometry-adaptive update (PVR-E) -------------------------------
    # regimes: k_b <= 0.2 -> TtBA k rule (narrow-region escape) + bracketed radial search;
    #          otherwise bisector (bridge step) or closed-form model step + secant calibration
    "TtBA_MBE": MethodSpec("TtBA_MBE", "pixel", "ttba_mid", probe="dct", sigma=3e-4, probe_normalize=True,
                           escape=True, description="ablation: MB + TtBA escape regime (k_b <= 0.2)"),
    "TtBA_PVRE": MethodSpec("TtBA_PVRE", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                            escape=True, description="FINAL (no LGC): PVR-E in pixel space with native TtBA normal probes"),
    "TtBA_PVRE_UA": MethodSpec(
        "TtBA_PVRE_UA", "pixel", "pvr_ua", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True,
        description="PVR-E v2: exact TtBA narrow-region escape + periodic bridge re-anchor + "
                    "uncertainty-aware single-direction selection (PVR model vs anchored TtBA)"),
    "TtBA_PVRE_GC": MethodSpec(
        "TtBA_PVRE_GC", "pixel", "pvr_gc", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True,
        description="PVR-E v3: robust signed-residual confidence + adaptive bridge re-anchor + "
                    "stall-triggered exact-TtBA rescue + flatness-gated Thales/CGBA geometry"),
    "TtBA_PVRE_AOR_SN": MethodSpec(
        "TtBA_PVRE_AOR_SN", "pixel", "pvr_aor", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, aor_native_planar=False,
        description="PVR-E v4 ablation: adaptive-order regularization with shared TtBA normal in planar mode"),
    "TtBA_PVRE_AOR": MethodSpec(
        "TtBA_PVRE_AOR", "pixel", "pvr_aor", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, aor_native_planar=True,
        description="PVR-E v4: curvature-resolvability shrinkage + effect-controlled narrow bridge + "
                    "native-CGBA planar mode without double normal estimation"),
    "TtBA_PVRE_AOU": MethodSpec(
        "TtBA_PVRE_AOU", "pixel", "pvr_aou", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True,
        description="PVR-E v5: adaptive-order geometry as a prior + persistent utility controller "
                    "over PVR/TtBA/CGBA updates + narrow E-vs-T utility trials"),
    "TtBA_PVRE_CAOU": MethodSpec(
        "TtBA_PVRE_CAOU", "pixel", "pvr_caou", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True,
        description="PVR-E v6: context-conditioned confidence-aware utility controller with "
                    "macro-action trials, switching cost and persistent M/T/C or E/R trajectories"),
    # ---- build 31a: PVR-E v7 = Gate - Thales - Escape (no model step, no utility controller)
    "TtBA_PVRE_GTE": MethodSpec(
        "TtBA_PVRE_GTE", "pixel", "pvr_gte", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, gte_thales="flat",
        description="PVR-E v7 (MAIN): k=0.2 gate every iteration; narrow -> TtBA escape; wide -> "
                    "Thales one-search while median|eps|<=1, else 2/3-rule bridge step that measures eps"),
    "TtBA_PVRE_GE": MethodSpec(
        "TtBA_PVRE_GE", "pixel", "pvr_gte", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, gte_thales="off",
        description="PVR-E v7 ablation: gate + escape + bridge steps only (no model step, no Thales)"),
    "TtBA_PVRE_GTE_K": MethodSpec(
        "TtBA_PVRE_GTE_K", "pixel", "pvr_gte", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, gte_thales="always",
        description="PVR-E v7 ablation: Thales whenever the k=0.2 gate says wide (no flatness gate)"),
    "TtBA_PVRE_GTEB": MethodSpec(
        "TtBA_PVRE_GTEB", "pixel", "pvr_gte", probe="dct", sigma=3e-4, probe_normalize=True,
        escape=True, gte_thales="flat", sign_baseline=True,
        description="PVR-E v7 + HSJA mean-sign baseline in the TtBA normal estimate (0 extra queries)"),
    "TtBA_BL": MethodSpec(
        "TtBA_BL", "pixel", "ttba", probe="dct", sigma=3e-4, probe_normalize=True, sign_baseline=True,
        description="control: official TtBA update + HSJA mean-sign baseline in the normal estimate"),
    # ---- build 32a: TtBA_PVRE32 core (A1) and the A2 query-budget cutoff ablation.
    # gte_model=True is NOT listed in the spec rows: the wide-flat model-step choice is new in
    # this engine (the 31a base had no model step), so the flag is what activates it.  The
    # TtBA_PVRE32 core is otherwise a pure alias of TtBA_PVRE_GE's config (gte_thales="off").
    "TtBA_PVRE32": MethodSpec("TtBA_PVRE32", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                              probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                              description="Build 32a core = TtBA_PVRE_GE (Thales dropped, see Q2 "
                                          "null result and the Gamma(eps)<=1 bound)"),
    "TtBA_PVRE32_A2": MethodSpec("TtBA_PVRE32_A2", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                                 probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                                 pvr_gte_model_cutoff_q=1500,
                                 description="A2: PVR model step disabled after 1500 queries (dev "
                                             "value; freeze after the Q_m sweep, see Part C)"),
    # ---- build 32b: frozen Tier-A core.  Q_m = 500 is the a-priori Part C choice (minimum NT 10k
    # geo-ratio vs CGBA over the 32a sweep {500,1000,1500,2500,5000}: 0.988 / median 1.021).
    "TtBA_PVRE32F": MethodSpec("TtBA_PVRE32F", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                               probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                               pvr_gte_model_cutoff_q=500,
                               description="32b core: GE + PVR model step before 500 queries (A1+A2 frozen)"),
    # A3 (mechanism confirmed by the 32a escape-parity log): the narrow branch spends 1 gate query plus
    # an 8-query bridge on [0, 0.2] at tol 1e-3 (9 vs 7 for the legacy escape) with no per-iteration
    # geometric gain.  tol 1e-2 on [0, 0.2] costs ceil(log2 20) = 5 bridge queries (6 with the gate).
    "TtBA_PVRE32F_N2": MethodSpec("TtBA_PVRE32F_N2", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                                  probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                                  pvr_gte_model_cutoff_q=500, gte_narrow_tol=1e-2,
                                  description="32b A3: 32F + narrow bridge tolerance 1e-2 (-3 queries/narrow iteration)"),
    "TtBA_PVRE32F_E": MethodSpec("TtBA_PVRE32F_E", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                                 probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                                 pvr_gte_model_cutoff_q=500, gte_eps_bracket=True,
                                 description="32b A4: 32F + eps-predicted radial bracket"),
    # ---- build 33a (32b results: A3 narrow tol 1e-2 = early targeted -3.4%* vs 32F; A4 eps bracket = late
    # targeted -5.9% (p=.049), NT -0.9%; the two act in different phases, so they are combined unchanged).
    "TtBA_PVRE32F_NE": MethodSpec("TtBA_PVRE32F_NE", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                                  probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                                  pvr_gte_model_cutoff_q=500, gte_narrow_tol=1e-2, gte_eps_bracket=True,
                                  description="33a pixel candidate: 32F + A3 (narrow tol 1e-2) + A4 (eps bracket)"),
    # pixel stage for latent->pixel hybrids: no model step (the stage starts at q >= 2500, where model steps
    # lose; a stage-local query counter could otherwise re-enable them), otherwise identical to 32F_NE
    "TtBA_PVRE_GNE": MethodSpec("TtBA_PVRE_GNE", "pixel", "pvr_gte", probe="dct", sigma=3e-4,
                                probe_normalize=True, escape=True, gte_thales="off", gte_narrow_tol=1e-2,
                                gte_eps_bracket=True,
                                description="33a hybrid pixel stage: GE + A3 + A4 (no model step)"),
    # build 32b single-variable control (open question in the external handoffs: does CGBA's native
    # probe policy, sigma 2e-4 un-normalised DCT, explain part of its late-NT edge?).  GE update unchanged.
    "TtBA_PVRE_GE_CP": MethodSpec("TtBA_PVRE_GE_CP", "pixel", "pvr_gte", probe="dct", escape=True,
                                  gte_thales="off",
                                  description="control: TtBA_PVRE_GE with CGBA's probes (sigma 2e-4, not normalised)"),
    # ---- build 32b Tier B (corrected; run ONLY after the diagnose_tangent.py gate passes).
    # The 32a LGC_T rows are kept for provenance but now raise (probe-scale guard).
    "LGC_TB": MethodSpec("LGC_TB", "pixel", "pvr_gte", probe="tangent_chord", sigma=3e-4,
                         probe_normalize=True, escape=True, gte_thales="off", tan_weighted=True,
                         description="Tier B probe ablation: TtBA_PVRE_GE core + tangent-chord probes, "
                                     "weighted estimator (pair with TtBA_PVRE_GE)"),
    "LGC_TB_U": MethodSpec("LGC_TB_U", "pixel", "pvr_gte", probe="tangent_chord", sigma=3e-4,
                           probe_normalize=True, escape=True, gte_thales="off", tan_weighted=False,
                           description="Tier B control: as LGC_TB with the 32a equal-per-probe sum"),
    "LGC_TBF": MethodSpec("LGC_TBF", "pixel", "pvr_gte", probe="tangent_chord", sigma=3e-4,
                          probe_normalize=True, escape=True, gte_thales="off", gte_model=True,
                          pvr_gte_model_cutoff_q=500, tan_weighted=True,
                          description="Tier B candidate final: TtBA_PVRE32F core + weighted tangent-chord probes"),
    "LGC_PVRE": MethodSpec("LGC_PVRE", "latent_r", "pvr", escape=True,
                           description="FINAL (LGC, perceptual): PVR-E in the RAG latent space with the "
                                       "residual coordinate (exact hard-label targeted start)"),
    "LGC_PVRE_H1000": MethodSpec("LGC_PVRE_H1000", "latent_r", "pvr", escape=True, switch="q1000",
                                 description="LGC_PVRE for 1000 queries, then TtBA_PVRE in pixel space"),
    "LGC_PVRE_H2500": MethodSpec("LGC_PVRE_H2500", "latent_r", "pvr", escape=True, switch="q2500",
                                 description="LGC_PVRE for 2500 queries, then TtBA_PVRE in pixel space"),
    "LGC_PVRE_HA": MethodSpec("LGC_PVRE_HA", "latent_r", "pvr", escape=True, switch="auto",
                              description="FINAL (LGC, L2): LGC_PVRE, switch to pixel TtBA_PVRE when the "
                                          "local decay exponent of the pixel L2 falls below 0.45"),
    # ---- ablation: normal tracking (*_NT), projected empirical-Bayes prior (simulation: 0-2 % worse)
    "TtBA_NT": MethodSpec("TtBA_NT", "pixel", "ttba", probe="dct", sigma=3e-4, probe_normalize=True,
                          track=True, description="ablation: official TtBA + normal tracking"),
    "CGBA_NT": MethodSpec("CGBA_NT", "pixel", "cgba", probe="dct", track=True,
                          description="ablation: official CGBA + normal tracking"),
    "TtBA_PVRE_NT": MethodSpec("TtBA_PVRE_NT", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                               escape=True, track=True, description="ablation: TtBA_PVRE + normal tracking"),
    # ---- ImageNet follow-up (build 27i): gamma-gated Thales regime, CGBA probes, LGC + residual start
    "TtBA_PVRET": MethodSpec("TtBA_PVRET", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                             escape=True, thales=True,
                             description="PVR-E + gamma-gated Thales one-search (CGBA update where the boundary "
                                         "is flat and wide; escape / bisector / model elsewhere)"),
    "TtBA_PVRE_CP": MethodSpec("TtBA_PVRE_CP", "pixel", "pvr", probe="dct", escape=True,
                               description="diagnostic: TtBA_PVRE with CGBA's probes (sigma 2e-4, not normalised)"),
    # ---- 30e three-regime router with native TtBA/CGBA normal estimation -----------------------------
    # Keeps the published standalone TtBA / CGBA / TtBA_PVRE rows unchanged.
    # Same-radius gates at k=0.2 and k=0.8 route each iteration; low/mid bridge
    # refinement is restricted to the gate-certified interval to avoid redundant queries.
    "TtBA_RAMB_PVRE": MethodSpec(
        "TtBA_RAMB_PVRE", "pixel", "ramb_pvre", probe="dct", sigma=3e-4, probe_normalize=True,
        ra_history=False, ra_mid_pvr=True,
        description="Tri-route 30e: TtBA for k_b<=0.2; TtBA_PVRE for 0.2<k_b<0.8; "
                    "CGBA for k_b>=0.8. Gates use the native TtBA normal; the CGBA route "
                    "re-estimates a native CGBA normal before the CGBA update."),
    "TtBA_RAMB_NOHIST": MethodSpec(
        "TtBA_RAMB_NOHIST", "pixel", "ramb_pvre", probe="dct", sigma=3e-4, probe_normalize=True,
        ra_history=False, ra_mid_pvr=False,
        description="RA-MB 30a ablation retained for comparison: angular M/L race, no history candidate"),
    "LGC_RC": MethodSpec("LGC_RC", "latent_r", "cgba",
                         description="LGC (CGBA update in the RAG latent) + residual coordinate (hard-label "
                                     "targeted start at x_t); non-targeted = LGC"),
    "LGC_RC_HA": MethodSpec("LGC_RC_HA", "latent_r", "cgba", switch="auto", pixel_stage="TtBA_PVRET",
                            description="LGC_RC, auto switch to pixel TtBA_PVRET"),
    "LGC_RC_H1000": MethodSpec("LGC_RC_H1000", "latent_r", "cgba", switch="q1000", pixel_stage="TtBA_PVRET",
                               description="LGC_RC for 1000 queries, then pixel TtBA_PVRET (ImageNet: the auto "
                                           "rule switched at ~3000, after the latent curve had fallen behind)"),
    # build 37 r5: the naive combination of two PUBLISHED attacks for the paper ablation -- LGC (latent, residual
    # start) for 1000 queries, then pixel TtBA (ICML 2025) on the full perturbation.  Same switch mechanism as H1000.
    "LGC_RC_HT1000": MethodSpec("LGC_RC_HT1000", "latent_r", "cgba", switch="q1000", pixel_stage="TtBA",
                                description="37r5 ablation: LGC_RC for 1000 queries, then pixel TtBA (naive switch)"),
    # ---- build 33a: late latent->pixel switch.  32b targeted evidence (geometry, matched radius): after a
    # switch at q~1000 the pixel stage is 16-34 % less efficient per query than pixel-only for r >= 20 (smaller
    # k_b: the latent point sits in a narrower region) and equal for r <= 12, while LGC_RC's own efficiency
    # falls to 0.15 log-L2/1k at 5k-10k (pixel: 0.18-0.21).  Pre-registered primary: H5000 (= Q/2); H2500 and
    # H7500 are the sensitivity ladder.  Pixel stage = TtBA_PVRE_GNE.
    "LGC_RC_H2500G": MethodSpec("LGC_RC_H2500G", "latent_r", "cgba", switch="q2500", pixel_stage="TtBA_PVRE_GNE",
                                description="LGC_RC for 2500 queries, then pixel TtBA_PVRE_GNE"),
    "LGC_RC_H5000G": MethodSpec("LGC_RC_H5000G", "latent_r", "cgba", switch="q5000", pixel_stage="TtBA_PVRE_GNE",
                                description="33a primary: LGC_RC for Q/2 = 5000 queries, then pixel TtBA_PVRE_GNE"),
    "LGC_RC_H7500G": MethodSpec("LGC_RC_H7500G", "latent_r", "cgba", switch="q7500", pixel_stage="TtBA_PVRE_GNE",
                                description="LGC_RC for 7500 queries, then pixel TtBA_PVRE_GNE"),
    # ---- build 33b: LGC-PVRE = RAG chord + residual coordinate + latent-adapted PVR-E update
    # (claude/LGC_RC_PVRE_design_2026-10-03.md).  Single-variable ladder from LGC_RC (CGBA update in latent_r),
    # all with LGC_RC's probe policy (Gaussian latent probes, EngineConfig.sigma, n0 sqrt(t+1) schedule):
    #   GE    : update -> PVR-E v7 gate + escape + bridge (A3 narrow tol, A4 eps bracket), no Thales, no model
    #   GTE   : + eps-gated Thales (= the CGBA step where the latent boundary is flat)
    #   PVRE0 : + pixel-L2 acceptance (objective)
    #   PVRE  : + pull-back chart (pixel-isometric plane geometry; the proposed method)
    "LGC_RC_GE": MethodSpec("LGC_RC_GE", "latent_r", "pvr_gte", escape=True, gte_thales="off",
                            gte_narrow_tol=1e-2, gte_eps_bracket=True,
                            description="33b ladder: LGC_RC with the PVR-E v7 update (gate + escape + bridge)"),
    "LGC_RC_GTE": MethodSpec("LGC_RC_GTE", "latent_r", "pvr_gte", escape=True, gte_thales="flat",
                             gte_narrow_tol=1e-2, gte_eps_bracket=True,
                             description="33b ladder: + eps-gated Thales (CGBA step where flat)"),
    "LGC_RC_PVRE0": MethodSpec("LGC_RC_PVRE0", "latent_r", "pvr_gte", escape=True, gte_thales="flat",
                               gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel",
                               description="33b ladder: + pixel-L2 acceptance (latent-Euclidean geometry)"),
    "LGC_RC_PVRE": MethodSpec("LGC_RC_PVRE", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                              gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel",
                              description="33b PROPOSED (LGC-PVRE): PVR-E v7 in the pull-back (pixel-isometric) "
                                          "chart of the latent plane, pixel-L2 acceptance"),
    # ---- build 33c: flat-first regime order.  33b pilot (n = 20): in NT the chart raised the narrow share
    # N 1% -> 30% and cut Thales 74% -> 53%, and PVRE / PVRE0 = 1.10 at 10k, while Thales is what wins in NT
    # (GTE / GE = 0.877).  The k = 0.2 gate measures the adversarial ARC size (curvature AND convergence AND
    # normal noise; k_b is not a curvature label), whereas eps measures departure from the hyperplane model
    # directly.  Flat-first lets measured flatness take precedence over the gate.
    "LGC_RC_PVRE_FF": MethodSpec("LGC_RC_PVRE_FF", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                                 gte_gate_order="flat_first", gte_narrow_tol=1e-2, gte_eps_bracket=True,
                                 objective="pixel",
                                 description="33c: LGC-PVRE with flat-first regime order (Thales when eps says "
                                             "flat, gate otherwise)"),
    # ---- build 34b: back to the CGBA / TtBA frame.  The 33b/34 evidence (claude/LGC_RC_PVRE_design §13):
    # every significant cost of LGC-PVRE came from the PVR-E controller and every significant gain from the
    # pull-back chart (targeted chart effect 0.786 at 10k, 94/6, n = 100 held out).  So the official CGBA /
    # CGBA-H / TtBA updates are run in the chart.  2 x 2 design {CGBA, TtBA} x {latent chart, pull-back chart}:
    #   LGC_RC     CGBA,   latent  (original LGC + RC)          LGC_RC_PB   CGBA,   pull-back chart
    #   LGC_RC_TT  TtBA,   latent                               LGC_RC_PBT  TtBA,   pull-back chart
    #   LGC_H_RC   CGBA-H, latent  (original LGC-H + RC)        LGC_RC_PBH  CGBA-H, pull-back chart
    # Chart rows accept by pixel L2 (the chart measures it; in the latent chart the objective changes no
    # decision, PVRE0 = GTE on every image of the 33c dev split).  Same probes / schedule as LGC_RC.
    "LGC_RC_TT": MethodSpec("LGC_RC_TT", "latent_r", "ttba",
                            description="34b: LGC with the official TtBA update in the RAG + RC latent"),
    "LGC_RC_PB": MethodSpec("LGC_RC_PB", "latent_r", "pvr_pb", pb_rule="cgba", objective="pixel",
                            description="34b PROPOSED candidate: LGC, CGBA update in the pull-back chart"),
    "LGC_RC_PBH": MethodSpec("LGC_RC_PBH", "latent_r", "pvr_pb", pb_rule="cgba_h", objective="pixel",
                             description="34b: LGC-H, CGBA-H update in the pull-back chart"),
    "LGC_RC_PBT": MethodSpec("LGC_RC_PBT", "latent_r", "pvr_pb", pb_rule="ttba", objective="pixel",
                             description="34b PROPOSED candidate: LGC, TtBA update in the pull-back chart"),
    # selftest controls (not in the presets): the same 2-D core with the chart switched off must reproduce
    # the engine's latent rows (LGC_RC, LGC_RC_TT), and in pixel space (chart = identity) the pixel baselines
    "LGC_RC_PBL": MethodSpec("LGC_RC_PBL", "latent_r", "pvr_pb", pb_rule="cgba", pb_chart="latent",
                             description="34b selftest: 2-D core, CGBA, latent chart, latent objective (= LGC_RC)"),
    "LGC_RC_PBTL": MethodSpec("LGC_RC_PBTL", "latent_r", "pvr_pb", pb_rule="ttba", pb_chart="latent",
                              description="34b selftest: 2-D core, TtBA, latent chart, latent objective (= LGC_RC_TT)"),
    "CGBA_PBX": MethodSpec("CGBA_PBX", "pixel", "pvr_pb", probe="dct", pb_rule="cgba", objective="pixel",
                           description="34b selftest: CGBA through the chart in pixel space (must match CGBA)"),
    "CGBA_H_PBX": MethodSpec("CGBA_H_PBX", "pixel", "pvr_pb", probe="dct", pb_rule="cgba_h", objective="pixel",
                             description="34b selftest: CGBA-H through the chart in pixel space (must match CGBA_H)"),
    "TtBA_PBX": MethodSpec("TtBA_PBX", "pixel", "pvr_pb", probe="dct", sigma=3e-4, probe_normalize=True,
                           pb_rule="ttba", objective="pixel",
                           description="34b selftest: TtBA through the chart in pixel space (must match TtBA)"),
    # selftest-only special case: in the pixel space the chart is the identity (R = I, y_b = (r, 0)), so
    # pvr_pb must reproduce pvr_gte (TtBA_PVRE_GTE) up to finite-difference round-off
    "TtBA_PVRE_PBX": MethodSpec("TtBA_PVRE_PBX", "pixel", "pvr_pb", probe="dct", sigma=3e-4, probe_normalize=True,
                                escape=True, gte_thales="flat",
                                description="33b selftest: pvr_pb in pixel space (must match TtBA_PVRE_GTE)"),
    # ---- build 35: latent-space optimisation innovations beyond the CGBA frame (design doc sections 16-18).
    # Radial-plane methods (CGBA, TtBA, PVRE) stall where the normal is parallel to the LATENT radial; the
    # pixel-L2 KKT point needs n || J^T e.  The KKT-plane rows keep PVR-E's oracle plane search (noise-robust)
    # but search a plane that contains the exact pixel gradient J^T e (one decoder VJP, 0 oracle queries).
    # KSQP is the external proposal (SQP tangent step + metric-normal corrector + trust region), kept faithful.
    # CPS calibrates the probe radius from the antithetic agreement rate and the sign share.
    "LGC_RC_KKT": MethodSpec("LGC_RC_KKT", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                             gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="kkt_e",
                             description="35: LGC-PVRE in the KKT plane span(J^T e, n) (pull-back chart)"),
    "LGC_RC_KKTN": MethodSpec("LGC_RC_KKTN", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                              gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="kkt_n",
                              description="35: LGC-PVRE in the Gauss-Newton KKT plane span(G^-1 J^T e, G^-1 n)"),
    "LGC_RC_KSQP": MethodSpec("LGC_RC_KSQP", "latent_r", "ksqp", objective="pixel",
                              description="35: Krylov-SQP tangent step + metric-normal corrector + trust region"),
    "LGC_RC_PVRE_CPS": MethodSpec("LGC_RC_PVRE_CPS", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                                  gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", cps=True,
                                  description="35: LGC-PVRE + curvature-calibrated probe scale"),
    "LGC_RC_KKT_CPS": MethodSpec("LGC_RC_KKT_CPS", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                                 gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="kkt_e",
                                 cps=True, description="35: KKT plane + curvature-calibrated probe scale"),
    "LGC_RC_CPS": MethodSpec("LGC_RC_CPS", "latent_r", "cgba", cps=True,
                             description="35 ablation: LGC (CGBA update) + curvature-calibrated probe scale"),
    # ---- build 35 headline: contravariant-normal plane (Theorem 8).  With an exact normal every radial-plane
    # method (CGBA, TtBA, PVR-E; latent or pull-back chart) is confined to span(xi_0 - xi_s, n) after its first
    # step; the pixel optimum xi* ~ G^-1 n lies in span(xi - xi_s, G^-1 n), the plane used here.
    "LGC_RC_RN": MethodSpec("LGC_RC_RN", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                            gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="raised",
                            description="35 PROPOSED candidate: LGC-PVRE in the contravariant plane span(v, G^-1 n)"),
    "LGC_RC_RNC": MethodSpec("LGC_RC_RNC", "latent_r", "pvr_pb", pb_rule="cgba", objective="pixel",
                             pb_plane="raised",
                             description="35: CGBA semicircle in the contravariant plane (= LGC_RC_PB + raised normal)"),
    "LGC_RC_RN_CPS": MethodSpec("LGC_RC_RN_CPS", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                                gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="raised",
                                cps=True, description="35: contravariant plane + curvature-calibrated probe scale"),
    # ---- build 35b: damping scan of the contravariant plane.  The 35 selection (seed 3141, NT, n = 30) used
    # lam = 1e-2 ||J v||^2 and the raised chart was rejected in 100% of iterations (RN == PVRE bit for bit).
    "LGC_RC_RN_L01": MethodSpec("LGC_RC_RN_L01", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                                gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="raised",
                                pb_lambda_rel=0.1, description="35b: LGC_RC_RN with lam = 0.1 ||J v||^2"),
    "LGC_RC_RN_L1": MethodSpec("LGC_RC_RN_L1", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                               gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="raised",
                               pb_lambda_rel=1.0, description="35b: LGC_RC_RN with lam = ||J v||^2"),
    "LGC_RC_RN_W": MethodSpec("LGC_RC_RN_W", "latent_r", "pvr_pb", escape=True, gte_thales="flat",
                              gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", pb_plane="raised",
                              pb_lambda_mode="wiener",
                              description="35b: LGC_RC_RN with the Wiener damping lam = mean eig(J^T J) tan^2 phi"),
    "LGC_RC_RNC_L1": MethodSpec("LGC_RC_RNC_L1", "latent_r", "pvr_pb", pb_rule="cgba", objective="pixel",
                                pb_plane="raised", pb_lambda_rel=1.0,
                                description="35b: CGBA semicircle in the raised plane, lam = ||J v||^2"),
    # ---- build 36: corrected versions of the 2026-10-06 idea list (README_36.md)
    # MA-PVRE: radially coupled ambient block (Theorem 3), rate-greedy block choice; NOT a latent->pixel phase switch
    "LGC_RC_MA": MethodSpec("LGC_RC_MA", "latent_ra", "mapvre", escape=True, gte_thales="flat", gte_narrow_tol=1e-2,
                            gte_eps_bracket=True, objective="pixel", ma_start_q=1000,
                            description="36: MA-PVRE, latent block = LGC-PVRE chart step, ambient block = pixel "
                                        "PVR-E in span(e, n_pix); blocks from 1000 queries"),
    "LGC_RC_MAC": MethodSpec("LGC_RC_MAC", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel", ma_start_q=1000,
                             description="36: MA-PVRE with the CGBA semicircle in the latent chart (= LGC_RC_PB block)"),
    "LGC_RC_MA_S2500": MethodSpec("LGC_RC_MA_S2500", "latent_ra", "mapvre", escape=True, gte_thales="flat",
                                  gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", ma_start_q=2500,
                                  description="36: LGC_RC_MA, ambient block from 2500 queries"),
    "LGC_RC_MA_X": MethodSpec("LGC_RC_MA_X", "latent_ra", "mapvre", escape=True, gte_thales="flat",
                              gte_narrow_tol=1e-2, gte_eps_bracket=True, objective="pixel", ma_start_q=10 ** 9,
                              description="36 sanity: ambient block never used -> must equal LGC_RC_PVRE exactly"),
    # ---- build 37 (README_37.md): ambient admissibility gate + doubling exploration; MAL = LGC_RC's own latent step
    # (latent objective until the gate opens, so a closed gate reproduces LGC_RC).  Ambient block as in LGC_RC_MAC.
    "LGC_RC_MAC2": MethodSpec("LGC_RC_MAC2", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel", ma_start_q=1000,
                              ma_explore="doubling",
                              description="37: LGC_RC_MAC with doubling exploration (no gate)"),
    "LGC_RC_MACP": MethodSpec("LGC_RC_MACP", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel", ma_start_q=1000,
                              ma_explore="doubling", ma_gate="ssim", ma_gate_ssim=0.99,
                              description="37: MAC2 + perceptual gate (ambient only once SSIM(x, x0) >= 0.99)"),
    "LGC_RC_MACE": MethodSpec("LGC_RC_MACE", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel", ma_start_q=1000,
                              ma_explore="doubling", ma_gate="elastic", ma_gate_eps=0.45, ma_gate_q_elastic=2500,
                              description="37: MAC2 + floor gate (eps_hat < 0.45 from 2500 queries)"),
    "LGC_RC_MAL2": MethodSpec("LGC_RC_MAL2", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel",
                              ma_start_q=1000, ma_explore="doubling", ma_latent="lgc",
                              description="37: LGC_RC latent step + coupled ambient block from 1000 (vs H1000)"),
    "LGC_RC_MALP": MethodSpec("LGC_RC_MALP", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel",
                              ma_start_q=1000, ma_explore="doubling", ma_latent="lgc", ma_gate="ssim",
                              ma_gate_ssim=0.99,
                              description="37: LGC_RC until SSIM(x, x0) >= 0.99, then coupled ambient block"),
    # LGC-EGC (proposed framework, README_37 r2): ONE 2-D operator, the elliptic geometric chord (Thales locus under the
    # block metric), in both blocks: pull-back metric in the latent block (ellipse; CGBA's circle when R ~ I), identity
    # metric in the ambient block (circle about x0, coupled plane); SSIM admissibility gate; doubling exploration.
    "LGC_RC_EGC": MethodSpec("LGC_RC_EGC", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel", ma_start_q=1000,
                             ma_explore="doubling", ma_gate="ssim", ma_gate_ssim=0.99, ma_rule_amb="cgba",
                             description="37r2: LGC-EGC, elliptic chord in both blocks (no TtBA bridge / escape)"),
    # r4: isotropic instance of the same framework (R ~ I in the latent block = LGC_RC's CGBA circle with the latent
    # objective until the gate opens; identity-metric circle in the ambient block).  The single-operator fallback for
    # LGC_RC_MALP, which keeps the TtBA-family PVR-E ambient step.  Pairs: EGC vs EGCI = latent ellipse | gate;
    # EGCI vs MALP = ambient rule (circle vs PVR-E) | LGC latent block.
    "LGC_RC_EGCI": MethodSpec("LGC_RC_EGCI", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel",
                              ma_start_q=1000, ma_explore="doubling", ma_latent="lgc", ma_gate="ssim",
                              ma_gate_ssim=0.99, ma_rule_amb="cgba",
                              description="37r4: LGC-EGC with R ~ I (circle in both blocks), SSIM gate"),
    "LGC_RC_MAL_X": MethodSpec("LGC_RC_MAL_X", "latent_ra", "mapvre", pb_rule="cgba", objective="pixel",
                               ma_start_q=10 ** 9, ma_explore="doubling", ma_latent="lgc",
                               description="37 sanity: gate never opens -> must equal LGC_RC"),
    # weighted estimator, corrected: for ONE isotropic probe family the sign average is the matched filter; the only
    # zero-query re-weighting left is HSJA's mean-sign baseline (gain 1 / (1 - s_bar^2), ~3% at share 0.417)
    "LGC_RC_PB_BL": MethodSpec("LGC_RC_PB_BL", "latent_r", "pvr_pb", pb_rule="cgba", objective="pixel",
                               sign_baseline=True, description="36: LGC_RC_PB + HSJA mean-sign baseline"),
    # CPS v2 on the CGBA-in-chart row (LGC_RC_PVRE_CPS exists since 35)
    "LGC_RC_PB_CPS": MethodSpec("LGC_RC_PB_CPS", "latent_r", "pvr_pb", pb_rule="cgba", objective="pixel", cps=True,
                                description="36: LGC_RC_PB + curvature-calibrated probe scale (CPS v2)"),
    # selftest-only pixel special cases: in pixel space J = I, so J^T e = e and the KKT plane IS the radial plane
    "TtBA_PVRE_KKTX": MethodSpec("TtBA_PVRE_KKTX", "pixel", "pvr_pb", probe="dct", sigma=3e-4, probe_normalize=True,
                                 escape=True, gte_thales="flat", pb_plane="kkt_e",
                                 description="35 selftest: KKT plane in pixel space (must match TtBA_PVRE_PBX)"),
    "TtBA_PVRE_RNX": MethodSpec("TtBA_PVRE_RNX", "pixel", "pvr_pb", probe="dct", sigma=3e-4, probe_normalize=True,
                                escape=True, gte_thales="flat", pb_plane="raised",
                                description="35 selftest: contravariant plane in pixel space (G = (1+lam) I: "
                                            "must match TtBA_PVRE_PBX)"),
    "KSQP_PX": MethodSpec("KSQP_PX", "pixel", "ksqp", probe="dct", sigma=3e-4, probe_normalize=True,
                          objective="pixel", description="35 selftest: KSQP in pixel space (G = (1 + lam) I)"),
    "TtBA_PVRE_CPSX": MethodSpec("TtBA_PVRE_CPSX", "pixel", "pvr_pb", probe="dct", sigma=3e-4, probe_normalize=True,
                                 escape=True, gte_thales="flat", cps=True,
                                 description="35 selftest: CPS probe scale in pixel space"),
    # ---- probe-subspace dimension (the information lever): DCT block H/f x W/f, f = 8 or 2 instead of 4
    "TtBA_F8": MethodSpec("TtBA_F8", "pixel", "ttba", probe="dct", sigma=3e-4, probe_normalize=True,
                          dct_factor=8, description="official TtBA, DCT f=8 probes (d = 3 x 28 x 28)"),
    "CGBA_F8": MethodSpec("CGBA_F8", "pixel", "cgba", probe="dct", dct_factor=8,
                          description="official CGBA, DCT f=8 probes"),
    "TtBA_PVRE_F8": MethodSpec("TtBA_PVRE_F8", "pixel", "pvr", probe="dct", sigma=3e-4, probe_normalize=True,
                               escape=True, dct_factor=8, description="TtBA_PVRE, DCT f=8 probes"),
    "TtBA_F2": MethodSpec("TtBA_F2", "pixel", "ttba", probe="dct", sigma=3e-4, probe_normalize=True,
                          dct_factor=2, description="official TtBA, DCT f=2 probes (d = 3 x 112 x 112)"),
    "CGBA_F2": MethodSpec("CGBA_F2", "pixel", "cgba", probe="dct", dct_factor=2,
                          description="official CGBA, DCT f=2 probes"),
    # ---- QS family of the user's attacks/lgc_attack.py (build 27j): latent_r = RAG latent + residual
    #      coordinate, i.e. the hard-label targeted start replaces optimize_target_latent (Adam on the
    #      victim's cross-entropy); every query goes through the counted oracle (the original returned
    #      qs = 0 for QS / QS_Bridge2, so their radial / bridge / extrapolation queries were not counted)
    "LGC_H_RC": MethodSpec("LGC_H_RC", "latent_r", "cgba_h",
                           description="LGC_H (CGBA-H in the RAG latent) + residual coordinate; parent of QS "
                                       "(its fallback) and of the Hybrid's second stage"),
    "LGC_QS": MethodSpec("LGC_QS", "latent_r", "qs",
                         description="QS (lgc_attack.py), faithful: K-ray fan, SSIM-gap K, alpha tangent mixing, "
                                     "duplicate bin_search, tangent extrapolation; exact query count"),
    "LGC_QS_HYB": MethodSpec("LGC_QS_HYB", "latent_r", "qs_hyb",
                             description="Hybrid (lgc_attack.py), faithful: QS until SSIM >= 0.995, then LGC_H"),
    "LGC_QSB2": MethodSpec("LGC_QSB2", "latent_r", "qsb2",
                           description="QS_Bridge2 (lgc_attack.py), faithful incl. the reversed bisection and the "
                                       "inert bridge_ratio; exact query count"),
    "LGC_QSB2_O": MethodSpec("LGC_QSB2_O", "latent_r", "qsb2_o",
                             description="ablation: QS_Bridge2 with the geometry fixed (bisection toward eta, "
                                         "phi* from the anchor); everything else as in the original"),
    "LGC_QSB2_OA": MethodSpec("LGC_QSB2_OA", "latent_r", "qsb2_oa",
                              description="ablation: LGC_QSB2_O with alpha = 0 (search plane contains x_b); "
                                          "only the original's query waste remains"),
    "LGC_QSB2_C": MethodSpec("LGC_QSB2_C", "latent_r", "qsb2_c",
                             description="corrected QS_Bridge2: relative-tolerance bridge on the radius-R sphere, "
                                         "bisector, bracketed radial search at R cos(psi_b/2)"),
    "LGC_QS_C": MethodSpec("LGC_QS_C", "latent_r", "qs_c",
                           description="corrected QS: bridge first, K rays inside the measured arc, bracketed "
                                       "radial searches, best ray"),
}

PRESETS: Dict[str, List[str]] = {
    "latent": ["LGC", "LGC_H", "LGC_TTBA", "LGC_TA", "LGC_TA_J", "LGC_NGTA", "LGC_HSJA"],
    "whitened": ["LGC", "LGC_W", "LGC_W_TTBA", "LGC_W_HSJA", "LGC_HSJA"],
    "jacobian": ["LGC", "LGC_JS", "LGC_JSX", "LGC_JSM", "LGC_JSMP"],
    "pixel": ["LGC", "HSJA", "CGBA", "CGBA_H", "TtBA", "TA"],
    # main paper table: latent family + strongest latent baseline + pixel baselines
    "core": ["LGC", "LGC_H", "LGC_TTBA", "LGC_HSJA", "LGC_W", "HSJA", "CGBA", "CGBA_H", "TtBA", "TA"],
    # proposed bridge geometry vs. its parents (LGC = CGBA update, LGC_TTBA / TtBA = 2/3 rule)
    "geom": ["LGC", "LGC_TTBA", "LGC_MB", "LGC_HYB", "LGC_GARC", "TtBA", "TtBA_MB", "TtBA_HYB", "TtBA_GARC"],
    # cheapest run that measures gamma_obs (boundary non-linearity) in latent and pixel space
    "curvature": ["LGC_MB", "TtBA_MB"],
    # predict-verify-refine vs. its parents and the exact one-search (Thales) in both spaces
    "pvr": ["LGC", "LGC_MB", "LGC_HYB", "LGC_PVR", "CGBA", "TtBA", "TtBA_MB", "TtBA_HYB", "TtBA_PVR",
            "TtBA_T", "TtBA_T60"],
    # pixel-only, cheapest: low-budget levers (probe schedule, bridge tolerance, N*)
    "lowbudget": ["CGBA", "TtBA", "TtBA_MB", "TtBA_MB60", "TtBA_PVR", "TtBA_PVR60", "TtBA_PVRN",
                  "TtBA_T", "TtBA_T60"],
    # dev split: freeze the escape regime and the switch rule (paired with lowbudget / pvr results)
    "final_dev": ["TtBA", "TtBA_MB", "TtBA_MBE", "TtBA_PVR", "TtBA_PVRE",
                  "LGC", "LGC_PVR", "LGC_PVRE", "LGC_PVRE_H1000", "LGC_PVRE_H2500", "LGC_PVRE_HA"],
    # paper table: existing baselines (official settings) vs the final methods
    "final": ["HSJA", "TA", "CGBA", "CGBA_H", "TtBA", "LGC", "TtBA_PVRE", "LGC_PVRE", "LGC_PVRE_HA"],
    # probe-subspace dimension on the dev split (run AFTER diagnose_frequency.py says which f is promising)
    "freq_dev": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_F8", "CGBA_F8", "TtBA_PVRE_F8", "TtBA_F2", "CGBA_F2"],
    # ImageNet comparison (run with --dataset imagenet --out_dir Results/imagenet): published baselines,
    # the bridge-geometry ladder and the final methods (QS / QS_bridges of lgc_attack are added once ported)
    "imagenet_cmp": ["TtBA", "TtBA_MB", "TtBA_PVR", "TtBA_PVRE", "TtBA_T60", "CGBA",
                     "LGC", "LGC_PVR", "LGC_PVRE", "LGC_PVRE_HA"],
    # ImageNet follow-up: why CGBA / LGC win late in non-targeted mode, and the fixes
    "imagenet_fix": ["TtBA_T", "TtBA_PVRE_CP", "TtBA_PVRET", "LGC_RC", "LGC_RC_HA", "LGC_RC_H1000"],
    # Small, query-accounted prototype requested after the reviewer/meta-review pass.
    "ramb": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_RAMB_NOHIST", "TtBA_RAMB_PVRE"],
    # PVR-E continuation requested after the 30a--30e diagnostics.  Old TtBA_PVRE is kept as baseline.
    "pvre_ua": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_UA"],
    "pvre_gc": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_GC"],
    "pvre_aor": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_GC", "TtBA_PVRE_AOR_SN", "TtBA_PVRE_AOR"],
    "pvre_aou": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_AOR", "TtBA_PVRE_AOU"],
    "pvre_caou": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_AOR", "TtBA_PVRE_AOU", "TtBA_PVRE_CAOU"],
    # build 31a: PVR-E v7 ladder.  TtBA_T (CGBA update + TtBA probes, n0=30) separates update from probes.
    "pvre_gte": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_T", "TtBA_PVRE_GE", "TtBA_PVRE_GTE", "TtBA_PVRE_GTE_K"],
    # optional second run (same --out_dir): estimator control, 30j controller, PVR-E with CGBA probes
    "pvre_gte_extra": ["TtBA_BL", "TtBA_PVRE_GTEB", "TtBA_PVRE_CAOU", "TtBA_PVRE_CP"],
    # optional real-data check of the (simulation: 0-2 % worse) normal tracking ablation, ~12 images
    "track_pilot": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_NT", "CGBA_NT", "TtBA_PVRE_NT"],
    # ablation ladder (each row adds one component to the previous)
    "ablation": ["TtBA", "TtBA_MB", "TtBA_GARC", "TtBA_HYB", "TtBA_PVR", "TtBA_PVRE", "TtBA_MBE"],
    # QS family of lgc_attack.py (faithful + corrected) with their parents; run into the SAME --out_dir as
    # imagenet_cmp / imagenet_fix so that TtBA, CGBA, LGC, TtBA_PVRE, LGC_PVRE(_HA) are paired on the same images
    "imagenet_qs": ["LGC_RC", "LGC_H_RC", "LGC_QS", "LGC_QS_HYB", "LGC_QSB2", "LGC_QSB2_O", "LGC_QSB2_OA",
                    "LGC_QSB2_C", "LGC_QS_C"],
}
PRESETS["proposed"] = PRESETS["whitened"]
PRESETS["all"] = list(dict.fromkeys(PRESETS["latent"] + PRESETS["whitened"] + PRESETS["jacobian"]
                                    + PRESETS["pixel"] + PRESETS["geom"] + PRESETS["pvr"]
                                    + PRESETS["lowbudget"] + ["LGC_N60"] + PRESETS["final_dev"]
                                    + PRESETS["final"] + PRESETS["ablation"] + PRESETS["freq_dev"]
                                    + PRESETS["track_pilot"] + PRESETS["imagenet_cmp"]
                                    + PRESETS["imagenet_fix"] + PRESETS["imagenet_qs"] + PRESETS["ramb"]
                                    + PRESETS["pvre_gte"] + PRESETS["pvre_gte_extra"]))
BRIDGE_UPDATES = ("ttba", "ttba_mid", "hybrid", "garc", "pvr", "pvr_ua", "pvr_gc", "pvr_aor", "pvr_aou", "pvr_caou", "pvr_gte", "ramb_pvre", "pvr_pb", "mapvre")
QS_UPDATES = ("qs", "qs_hyb", "qsb2", "qsb2_o", "qsb2_oa", "qsb2_c", "qs_c")
PROBE_UPDATES = ("cgba", "cgba_h", "ttba", "ngta", "ttba_mid", "hybrid", "garc", "pvr", "pvr_ua", "pvr_gc", "pvr_aor", "pvr_aou", "pvr_caou", "pvr_gte", "ramb_pvre", "pvr_pb", "ksqp", "mapvre") + QS_UPDATES


@dataclass
class EngineConfig:
    max_queries: int = 10000
    checkpoints: Tuple[int, ...] = (500, 1000, 2500, 5000, 10000)
    max_iterations: int = 100000
    tol: float = 1e-4
    sigma: float = 2e-4
    n0: int = 30
    update_reserve: int = 40
    dct_factor: int = 4
    query_batch: int = 64
    init_step: float = 0.02
    init_attempts: int = 500
    # TtBA (official)
    ttba_bridge_tol: float = 1e-3
    ttba_low: float = 0.10
    ttba_high: float = 0.20
    ttba_jump: float = 0.9
    ttba_dive: float = 2.0 / 3.0
    # Triangle Attack (official argparse defaults)
    ta_plus: float = 0.1
    ta_minus: float = 0.005
    ta_half_range: float = 0.1
    ta_max_iter: int = 2
    ta_ratio_mask: float = 0.1
    ta_dim_num: int = 1
    ta_init_alpha: float = math.pi / 2
    # HSJA (official defaults)
    hsja_gamma: float = 1.0
    hsja_init_evals: int = 100
    hsja_max_evals: int = 10000
    hsja_max_halvings: int = 30
    # proposed: pull-back metric regulariser (relative to sigma_1^2) and probe mix
    jac_lambda: float = 1e-2
    jac_mix: float = 1.0
    # proposed bridge geometry
    bracket_delta: float = 0.02  # first relative bracket around a predicted boundary radius
    hyb_lambda: float = 0.9  # forgetting factor of the pooled perpendicular-distance estimate
    hyb_bridge_weight: float = 25.0  # weight of a bridge measurement vs. one single-shot estimate
    hyb_agree: float = 0.25  # model accepted if |psi_model / psi_bridge - 1| < hyb_agree
    hyb_ratio_lo: float = 0.75  # trust ratio (actual / predicted reduction) acceptance band
    hyb_ratio_hi: float = 1.33
    garc_window: int = 12  # observations kept by the hyperbolic-boundary fit
    garc_lambda: float = 0.9
    # proposed predict-verify-refine (PVR)
    pvr_bridge_tol: float = 1e-2  # bridge tolerance (distance loss is O(eps^2) at the stationary point)
    pvr_calib_weight: float = 10.0  # weight of a model-step secant-foot calibration (x one single-shot sample)
    pvr_err_window: int = 10  # recent relative radius-prediction errors kept for the adaptive bracket
    pvr_bracket_mult: float = 3.0  # first bracket = mult * q90(recent errors), clipped to [2 tol/g, bracket_delta]
    # PVR-E v2 (TtBA_PVRE_UA).  These are experimental knobs: tune on a dev split, then freeze.
    pvr_ua_reanchor_every: int = 4  # max model/selected steps between measured bridge anchors
    pvr_ua_uncert_q: float = 0.90  # quantile of recent |g_pred-g_act|/r used as model uncertainty
    pvr_ua_uncert_mult: float = 1.0  # model must beat TtBA by this many uncertainty radii
    pvr_ua_uncert_floor: float = 0.005  # conservative floor on relative-radius uncertainty
    pvr_ua_min_errs: int = 2  # require this many calibrated residuals before model can win
    # PVR-E v3 (TtBA_PVRE_GC): evidence-driven geometry controller.
    # Defaults are development values, not theoretical constants; freeze after a dev split.
    pvr_gc_reanchor_every: int = 6  # hard cap on consecutive model/thales steps without a fresh bridge
    pvr_gc_min_model_errs: int = 2  # robust model-residual history before normal confidence gating
    pvr_gc_mad_mult: float = 1.0  # lower-confidence gain = predicted gain - mult * robust scale
    pvr_gc_mad_floor: float = 0.0025  # floor on relative-radius robust residual scale
    pvr_gc_anchor_deadband: float = 0.001  # relative-radius edge needed to choose TtBA over bisector at an anchor
    pvr_gc_stall_ratio: float = 0.25  # R1 is a stall if gain < ratio * median recent accepted gain
    pvr_gc_stall_patience: int = 2  # consecutive R1 stalls before one exact-TtBA rescue
    pvr_gc_flat_tol_deg: float = 3.0  # robust local-flatness gate, median |gamma_obs-90|
    pvr_gc_flat_window: int = 3
    pvr_gc_thales_steps: int = 2  # max consecutive CGBA/Thales geometry steps before a bridge re-anchor
    # PVR-E v4 (TtBA_PVRE_AOR): adaptive model order / resolvability controller.
    # These are dev defaults.  Freeze after a dedicated development split.
    pvr_aor_reanchor_every: int = 6
    pvr_aor_min_model_errs: int = 2
    pvr_aor_mad_mult: float = 1.0
    pvr_aor_mad_floor: float = 0.0025
    pvr_aor_anchor_deadband: float = 0.001
    pvr_aor_resolve_enter: float = 1.0  # D/U <= this means curvature is unresolved
    pvr_aor_resolve_window: int = 3
    pvr_aor_planar_steps: int = 4  # native-CGBA burst before a fresh bridge/model-order check
    pvr_aor_bridge_radius_tol: float = 0.0010  # refine k only while its induced radial ambiguity exceeds this
    # PVR-E v5 (TtBA_PVRE_AOU): adaptive-order UTILITY controller.
    # Geometry (AOR) supplies a prior; actual log-progress per total iteration query decides persistence.
    pvr_aou_recheck_every: int = 6  # max wide-mode steps without a fresh bridge/regime check
    pvr_aou_utility_beta: float = 0.80  # EMA memory of observed action utility
    pvr_aou_switch_margin: float = 0.10  # challenger utility must exceed incumbent by this fraction
    pvr_aou_trial_interval: int = 5  # controlled one-action challenger trial cadence in wide regime
    pvr_aou_min_dwell: int = 2  # minimum incumbent steps before a challenger trial
    pvr_aou_geom_w_split: float = 0.50  # AOR weight only orders WHICH challenger is tried first
    pvr_aou_min_model_errs: int = 2
    pvr_aou_mad_floor: float = 0.0025
    pvr_aou_bridge_radius_tol: float = 0.0010
    pvr_aou_narrow_probe_after: int = 4  # productive PVR-E escape gets a short protected warm-up
    pvr_aou_narrow_trial_interval: int = 6  # E<->exact-TtBA counter-action trial cadence
    pvr_aou_narrow_switch_margin: float = 0.10
    # PVR-E v6 (TtBA_PVRE_CAOU): context-conditioned confidence-aware utility control.
    # Geometry remains a prior.  Utilities are local in (model-order weight, budget stage,
    # log-radius progress), and a challenger is queried only when its optimistic bound can
    # plausibly beat the incumbent after switching cost.  Trials are short macro-actions.
    pvr_caou_recheck_every: int = 6
    pvr_caou_min_model_errs: int = 2
    pvr_caou_mad_floor: float = 0.0025
    pvr_caou_bridge_radius_tol: float = 0.0010
    pvr_caou_hist_window: int = 64
    pvr_caou_time_decay: float = 0.97
    pvr_caou_bw_w: float = 0.30
    pvr_caou_bw_q: float = 0.25
    pvr_caou_bw_r: float = 0.30
    pvr_caou_bw_k: float = 0.35
    pvr_caou_conf_mult: float = 1.0
    pvr_caou_stderr_floor: float = 2.5e-5
    pvr_caou_utility_floor: float = 1.0e-6
    pvr_caou_switch_cost_frac: float = 0.10
    pvr_caou_geom_prior_frac: float = 0.05
    pvr_caou_trial_interval: int = 8
    pvr_caou_min_dwell: int = 3
    pvr_caou_trial_block: int = 2
    pvr_caou_stale_after: int = 10
    pvr_caou_min_neff: float = 1.5
    pvr_caou_sparse_strong_margin: float = 0.25
    pvr_caou_narrow_probe_after: int = 5
    pvr_caou_narrow_trial_interval: int = 8
    pvr_caou_narrow_trial_block: int = 2
    pvr_caou_narrow_switch_cost_frac: float = 0.08
    pvr_caou_narrow_sparse_strong_margin: float = 0.18
    # PVR-E v7 (TtBA_PVRE_GTE, build 31a).  A-priori values, NOT tuned on the evaluation images:
    # eps_tol = 1 means "the flat model explains the measured step to within its own predicted gain"
    # (symmetric-model Thales loss <= 1 - 4*2/9 = 11 %); window / Thales run length are the 27i values.
    pvr_gte_eps_tol: float = 1.0
    pvr_gte_window: int = 3
    pvr_gte_thales_steps: int = 4
    pvr_gte_narrow_tol: float = 1e-3  # = TtBA's bridge tolerance inside the certified [0, 0.2] bracket
    # build 32a (TtBA_PVRE32) A2: queries after which the model step is disabled inside the
    # wide-flat branch (None = unlimited, i.e. the model step may run for the whole budget).
    # Dev-frozen value: see the Part C sweep {500, 1000, 1500, 2500, 5000} on the dev split.
    pvr_gte_model_cutoff_q: Optional[int] = None
    # build 32a A3 (diagnostic only): if True, log every field that differs between the
    # pvr_gte escape branch and the legacy pvr escape branch (gate query count charged this
    # iteration, q_bridge, q_radial, bracket source) to geometry*.csv as new columns
    # `dbg_gate_q`, `dbg_escape_src`.  Do not patch the escape step until the mechanism is
    # confirmed from these columns (see BUILD_32A_SPEC.md Part A3).
    pvr_gte_debug_escape_parity: bool = False
    # build 32a A4: if True, seed the radial-search bracket at the eps-predicted centre
    # g_hat = g_lin - eps_hat (r - g_lin) with half-width max(2 tol/g_lin, 0.3 |eps_hat| (r-g_lin)/g_lin)
    # using the running median gte_eps (already tracked in self.gte_eps), instead of the fixed
    # bracket_delta margin.  Falls back to the current bracket when len(gte_eps) < pvr_gte_window.
    # TODO(claude-review): the config comment in the spec says seed at g_lin (1 - eps/(1+eps)); the
    # Part A4 derivation says g_lin - eps (r - g_lin).  They agree only when r = g_lin (2+eps)/(1+eps).
    # This build implements the derivation.
    pvr_gte_eps_bracket: bool = False
    # build 33b: pull-back PVR-E (update "pvr_pb"): pixel-isometric chart of the latent search plane
    pb_fd_rel: float = 0.02  # central finite-difference step for J v, J w, relative to ||xi_b - xi_s||
    pb_cond_max: float = 1e3  # chart rejected above this cond(R) -> latent-Euclidean pvr_gte fallback
    pb_verify_tol: float = 0.10  # chart rejected if |D_lin - D| / D > tol at the 3 test points (0 queries)
    # build 35: KKT plane / KSQP / CPS (0 oracle queries except the KSQP corrector and the CPS pairs)
    kkt_cg_iters: int = 8  # CG iterations per solve with G = J^T J + lam I (kkt_n, ksqp)
    kkt_lambda_rel: float = 1e-2  # lam = kkt_lambda_rel * ||J v||^2 (v = latent radial unit): Tikhonov damping
    kkt_floor_max: float = 0.6  # KKT-plane chart rejected (radial plane used) if ||e_perp|| / ||e|| exceeds this
    kkt_cos_min: float = 0.02  # non-radial chart rejected if cos(n_y, y_b) < this (chart origin on the wrong side)
    kkt_cg_tol: float = 1e-3  # CG stops at relative residual <= tol
    kkt_tan2_min: float = 0.1  # wiener mode: floor of tan^2(phi)
    kkt_mbar_ema: float = 0.8  # wiener mode: EMA weight of the Hutchinson estimate of mean(eig J^T J)
    ma_sigma_amb: float = 3e-4  # MA-PVRE ambient probes: TtBA scale (normalised DCT-low probes, L2 = 3e-4 sqrt(d))
    ma_dct_factor: int = 4  # MA-PVRE ambient probe subspace: DCT block H/f x W/f (TtBA s = 4)
    ma_rate_ema: float = 0.5  # EMA weight of the previous log-progress-per-query of a block
    # build 35 decision gate (diagnose_kkt.py): white-box TRUE latent normal z -> unit(grad_z margin(Phi(z))), set
    # per image by run_benchmark_33b.py --diag_plane.  DIAGNOSTIC ONLY: never used by the attack's decisions.
    diag_normal_fn: Optional[Callable] = None
    ksqp_trust0: float = 0.3  # initial trust radius in pixel L2, relative to D
    ksqp_trust_min: float = 1e-3  # trust radius floor, relative to D
    ksqp_corr_scale: float = 0.05  # first corrector bracket, pixel L2 relative to D
    ksqp_bracket_max: int = 12  # corrector bracket doublings
    cps_pair_frac: float = 0.2  # share of the probe budget spent as antithetic pairs (+u, -u)
    cps_ema: float = 0.7  # EMA weight of the previous scale-invariant curvature estimate (B, Gam)
    cps_offset_frac: float = 0.5  # center offset delta = cps_offset_frac * tol (bisection leaves 0 <= delta <= tol)
    cps_s_min: float = 0.5  # per-iteration scale search range for sigma
    cps_s_max: float = 2.0
    cps_mult_min: float = 0.125  # sigma / sigma_0 clamp
    cps_mult_max: float = 8.0
    # build 32a Tier B (LGC_T): tangent-chord probe mix defaults (MethodSpec.tan_* overrides)
    tan_rank: int = 256  # tangent-basis rank r built by tangent_chord_basis
    tan_frac: float = 0.5  # share of the probes sampled isotropically in span(Q_tan)
    # build 32b Tier B: energy ratio rho = E_J / E_S_perp (useful-normal energy in span(Q_tan) over the
    # energy in the DCT complement family), FROZEN from diagnose_tangent.py (surrogate-calibrated for
    # paper runs).  Optimal combination of the two partial estimates (Cauchy-Schwarz):
    #     n_hat ∝ sqrt(rho) c(N_J, r) unit(sum_J s u) + c(N_S, d_S) unit(sum_S s u),
    #     c(N, d) = (1 + pi d / (2N))^(-1/2)
    # None -> rows with tan_weighted=True raise (no silent fallback to the unweighted 32a sum).
    tan_energy_ratio: Optional[float] = None
    # 30c tri-route.  k_b <= 0.2 is inherited from TtBA.  The upper 0.8 gate is an
    # empirical routing threshold: TtBA (<=0.2), TtBA_PVRE (0.2..0.8), CGBA (>=0.8).
    # The older NOHIST ablation still uses lambda_wide for its angular M/L split.
    ra_mid_pvr_high: float = 0.80  # upper routing gate; >= this uses CGBA
    ra_lambda_wide: float = 0.70  # legacy NOHIST wide threshold on lambda_b=psi_b/theta
    ra_theta_eps_deg: float = 1.0  # lambda_b is not used when theta is smaller than this
    ra_h_exploit: float = 0.60  # H-regime exploitation candidate = ratio * k_b
    ra_h_escape: float = 0.90  # H-regime escape candidate = ratio * k_b
    ra_hist_beta: float = 0.90  # gated per-attack-instance EMA, only across consecutive M regimes
    ra_hist_min: float = 0.45
    ra_hist_max: float = 0.90
    ra_dedup_deg: float = 1.5  # merge near-identical candidate directions before querying
    ra_race_max_queries: int = 8  # includes prediction/same-radius screening + equal-depth race rounds
    ra_race_max_rounds: int = 4
    ra_refine_max_queries: int = 14  # winner-only certified radial refinement
    nstar_min: int = 20  # search range of the progress-optimal probe count N*
    nstar_max: int = 1000
    nstar_window: int = 5  # iterations averaged for the per-iteration overhead O (bridge + radial queries)
    # final methods: narrow-region escape and latent -> pixel switch
    escape_k: float = 0.2  # k_b at or below which the TtBA k rule is used (TtBA's own 'k_hat' = 0.2)
    switch_min_q: int = 1000  # auto switch: earliest query
    switch_slope: float = 0.45  # auto switch: local decay exponent of the pixel L2 below which we switch
    switch_window: float = 1.5  # auto switch: exponent measured between q / window and q
    switch_max_frac: float = 0.6  # auto switch: latest switch as a fraction of the budget
    # normal tracking (*_NT methods, lgc_bench/tracking.py)
    track_beta: float = 0.2  # EMA rate of the prior-alignment mean / variance
    track_gate: float = 2.0  # prior used only if |mean alignment| >= gate * its standard error
    # gamma-gated Thales regime (*_PVRET): CGBA's one-search is exact and needs no bridge when gamma ~ 90 deg
    thales_gamma_tol: float = 3.0  # deg: median |gamma_obs - 90| of recent bisector steps must be <= tol
    thales_window: int = 3  # bisector steps in the median
    thales_recheck: int = 4  # Thales iterations before the next bridge re-measures k_b and gamma
    # QS family (lgc_attack.py / run_fairbench.py defaults)
    qs_K_schedule: Tuple[Tuple[float, int], ...] = ((0.05, 4), (0.02, 4), (0.008, 6), (0.0, 8))  # SSIM gap -> K
    qs_K_min: int = 4
    qs_K_max: int = 16
    qs_switch_ssim: float = 0.995  # reference of the SSIM gap and the Hybrid switch
    qs_warmup: int = 20  # iterations with alpha = 0
    qs_alpha_max: float = 0.3  # tangent mixing strength
    qs_radial_steps: int = 30  # radial bisection steps per ray (0..R)
    qs_extrap_iters: int = 200  # run_fairbench 'iteration': extrapolation step (0.02 + 0.06 i / (I - 1)) R
    qsb_ratio: float = 0.5  # bridge ratio (0.5 = bisector)
    qsb_steps: int = 14  # original angular bisection steps
    qsb_tol: float = 1e-2  # original angular tolerance (x pi / 2)
    qsb_alpha_mode: str = "curv"  # original alpha from the bridge span: curv | inv | sched
    qsb_span_ref: float = math.pi / 4
    qsb_rel_tol: float = 0.05  # corrected bridge: bisection to 5 % of the bridge angle
    log_geometry: bool = True
    save_trace: bool = False
    verbose: bool = False


@dataclass
class AttackResult:
    method: str
    status: str
    checkpoint_states: Dict[int, Tuple[torch.Tensor, int]]
    final_state: Optional[torch.Tensor]
    final_state_query: int
    queries: int
    by_tag: Dict[str, int]
    iterations: int
    wall_time: float
    trace: List[Tuple[int, float]] = field(default_factory=list)
    diag: Dict[str, list] = field(default_factory=dict)
    geom: List[dict] = field(default_factory=list)
    # hybrid runs: the space each checkpoint state lives in (missing -> the method's own space)
    checkpoint_spaces: Dict[int, object] = field(default_factory=dict)


class InitFailed(RuntimeError):
    pass


_STOP = object()


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.reshape(-1) * b.reshape(-1)).sum().item())


def _norm(a: torch.Tensor) -> float:
    return float(a.reshape(-1).norm().item())


def _unit(a: torch.Tensor, eps: float = 1e-12) -> Optional[torch.Tensor]:
    n = _norm(a)
    if not math.isfinite(n) or n <= eps:
        return None
    return a / n


_SSIM_FN = None
_SSIM_TRIED = False


def _ssim_value(img: torch.Tensor, ref: torch.Tensor) -> float:
    """SSIM(img, ref) with torchmetrics (same function as the evaluation); without torchmetrics (unit tests)
    a single-window SSIM from global statistics. Used by the QS family only (attacker side, no query)."""
    global _SSIM_FN, _SSIM_TRIED
    if not _SSIM_TRIED:
        _SSIM_TRIED = True
        try:
            from .metrics import load_ssim

            _SSIM_FN = load_ssim()
        except Exception:  # pragma: no cover - torchmetrics missing
            _SSIM_FN = None
    if _SSIM_FN is not None:
        return float(_SSIM_FN(img, ref, data_range=1.0).item())
    a, b = img.reshape(-1), ref.reshape(-1)
    ma, mb = float(a.mean()), float(b.mean())
    da, db = a - ma, b - mb
    va, vb, cov = float((da * da).mean()), float((db * db).mean()), float((da * db).mean())
    c1, c2 = 1e-4, 9e-4
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))


class GeometricAttack:
    def __init__(
        self,
        spec: MethodSpec,
        cfg: EngineConfig,
        space: AttackSpace,
        oracle: HardLabelOracle,
        seed: int,
        jacobian: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_seed: Optional[torch.Tensor] = None,
        target_image: Optional[torch.Tensor] = None,
        init_point: Optional[torch.Tensor] = None,
        it_offset: int = 0,
        stop_rule: Optional[Callable[[int, torch.Tensor], bool]] = None,
    ):
        """init_point: warm start (verified with one query); it_offset: continue the probe schedule
        n0 sqrt(t + it_offset + 1) of a previous stage; stop_rule(q, state) -> True ends the run."""
        if spec.space != space.name:
            raise ValueError(f"{spec.name} expects a {spec.space} space, got {space.name}")
        if spec.needs_jacobian and jacobian is None:
            raise ValueError(f"{spec.name} needs the decoder-Jacobian subspace")
        self.spec = spec
        self.cfg = cfg
        self.space = space
        self.oracle = oracle
        self.origin = space.origin
        self.device = space.origin.device
        self.target_seed = target_seed
        self.target_image = target_image
        self.init_point = init_point
        self.it_offset = int(it_offset)
        self.stop_rule = stop_rule
        self.stopped_by_rule = False
        self.jac_V, self.jac_S = (jacobian if jacobian is not None else (None, None))

        self.rng_init = CPURNG(seed, self.device)
        self.rng_normal = CPURNG(seed + 1_000_003, self.device)
        self.rng_axis = CPURNG(seed + 2_000_003, self.device)
        self.rng_hsja = CPURNG(seed + 3_000_003, self.device)

        self.sigma = spec.sigma if spec.sigma is not None else cfg.sigma
        self.dct_f = int(spec.dct_factor) if spec.dct_factor is not None else int(cfg.dct_factor)
        self.sampler = self._make_probe_sampler()
        self.axis_sampler = self._make_axis_sampler()

        self.ta_alpha = cfg.ta_init_alpha
        self.hsja_dist_post = None
        self.diag: Dict[str, list] = {"cos_v_n": [], "k_bridge": [], "ta_alpha": [], "accepted": [],
                                      "sign_share": [], "n_probes": [],
                                      # QS family: K, alpha, share of rays that hit within R, LGC_H fallback,
                                      # accepted extrapolation, bridge span (deg), Hybrid in its LGC_H stage
                                      "qs_K": [], "qs_alpha": [], "qs_hit": [], "qs_fallback": [],
                                      "qs_extrap": [], "qsb_span_deg": [], "qs_stage2": [],
                                      # queries the ORIGINAL code did not add to q_num (it returned qs = 0)
                                      "qs_hidden": [], "qs_hidden_at": [],
                                      # RA-MB-PVRE diagnostics (per iteration)
                                      "ra_regime": [], "ra_lambda": [], "ra_candidates": [],
                                      "ra_race_q": [], "ra_pruned": [], "ra_unresolved": [],
                                      "ra_winner": [], "ra_eta": [], "ra_hist_c": [],
                                      "ra_infeasible": [], "ra_refine_q": [], "ra_update_q": [],
                                      "ra_mid_pvr": [], "ra_gate_q": [],
                                      # 30e native-normal routing diagnostics.  H/M use the first
                                      # TtBA-style normal; L re-estimates CGBA's own normal.
                                      "ra_cgba_normal_q": [], "ra_cgba_sign_share": [],
                                      "ra_cgba_normal_fallback": [],
                                      # PVR-E v2 diagnostics (one entry per pvr_ua iteration where relevant)
                                      "pvr_ua_choice": [], "pvr_ua_uncert": [],
                                      "pvr_ua_reanchor": [], "pvr_ua_anchor_age": [],
                                      "pvr_ua_exact_escape": [], "pvr_ua_anchor_kb": [],
                                      "pvr_ua_pred_model": [], "pvr_ua_pred_ttba": [],
                                      # PVR-E v3 / geometry-calibrated controller diagnostics
                                      "pvr_gc_choice": [], "pvr_gc_reanchor": [],
                                      "pvr_gc_bias": [], "pvr_gc_scale": [],
                                      "pvr_gc_anchor_kb": [], "pvr_gc_anchor_age": [],
                                      "pvr_gc_stall": [], "pvr_gc_gamma_dev": [],
                                      "pvr_gc_pred_model": [], "pvr_gc_pred_anchor": [],
                                      # PVR-E v4 / adaptive-order regularization diagnostics
                                      "pvr_aor_choice": [], "pvr_aor_reanchor": [],
                                      "pvr_aor_bias": [], "pvr_aor_scale": [],
                                      "pvr_aor_D": [], "pvr_aor_S": [], "pvr_aor_weight": [],
                                      "pvr_aor_anchor_kb": [], "pvr_aor_anchor_age": [],
                                      "pvr_aor_bridge_refine_q": [], "pvr_aor_bridge_rad_uncert": [],
                                      "pvr_aor_native_normal_q": [], "pvr_aor_native_sign_share": [],
                                      "pvr_aor_normal_split_deg": [],
                                      # PVR-E v5 / adaptive-order utility controller diagnostics
                                      "pvr_aou_choice": [], "pvr_aou_mode": [],
                                      "pvr_aou_trial": [], "pvr_aou_switch": [], "pvr_aou_recheck": [],
                                      "pvr_aou_eta": [], "pvr_aou_q_action": [],
                                      "pvr_aou_mu_M": [], "pvr_aou_mu_T": [], "pvr_aou_mu_C": [],
                                      "pvr_aou_mu_E": [], "pvr_aou_mu_R": [],
                                      "pvr_aou_bias": [], "pvr_aou_scale": [],
                                      "pvr_aou_D": [], "pvr_aou_S": [], "pvr_aou_weight": [],
                                      "pvr_aou_anchor_kb": [], "pvr_aou_anchor_age": [],
                                      "pvr_aou_bridge_refine_q": [], "pvr_aou_bridge_rad_uncert": [],
                                      "pvr_caou_choice": [], "pvr_caou_mode": [],
                                      "pvr_caou_trial": [], "pvr_caou_trial_start": [],
                                      "pvr_caou_switch": [], "pvr_caou_recheck": [],
                                      "pvr_caou_block_end": [], "pvr_caou_eta": [],
                                      "pvr_caou_block_eta": [], "pvr_caou_q_action": [],
                                      "pvr_caou_ctx_w": [], "pvr_caou_ctx_q": [], "pvr_caou_ctx_r": [],
                                      "pvr_caou_lcb_inc": [], "pvr_caou_ucb_inc": [],
                                      "pvr_caou_lcb_ch": [], "pvr_caou_ucb_ch": [],
                                      "pvr_caou_neff_inc": [], "pvr_caou_neff_ch": [],
                                      "pvr_caou_bias": [], "pvr_caou_scale": [],
                                      "pvr_caou_S": [], "pvr_caou_weight": [],
                                      "pvr_caou_anchor_kb": [], "pvr_caou_anchor_age": [],
                                      "pvr_caou_bridge_refine_q": [], "pvr_caou_bridge_rad_uncert": [],
                                      # PVR-E v7 (one entry per pvr_gte iteration)
                                      "gte_choice": [], "gte_eps": [], "gte_eps_med": [],
                                      "gte_q_update": [], "gte_kb": [], "gte_gain": [],
                                      # build 33b pull-back chart (pvr_pb; 0-query diagnostics)
                                      "pb_fallback": [], "pb_cond": [], "pb_anchor_ratio": [],
                                      "pb_anchor_deg": [], "pb_floor_share": [], "pb_lin_err": [],
                                      "pb_verify_err": [],
                                      "sign_baseline_gain": []}
        self._qsb_span: Optional[float] = None  # QS_Bridge2: span of the previous bridge (drives alpha)
        self._qs_prev_psib: Optional[float] = None  # corrected bridge: previous bridge angle (warm start)
        self.trace: List[Tuple[int, float]] = []
        self.checkpoint_states: Dict[int, Tuple[torch.Tensor, int]] = {}

        # proposed bridge-geometry state
        self.geom: List[dict] = []
        self.d_eff = self._probe_dim()
        self.hyb_pool = [0.0, 0.0]  # [sum w * d*_sample, sum w]
        self.hyb_mode = "bridge"
        self.hfit = HyperbolaFit(window=cfg.garc_window, lam=cfg.garc_lambda)
        self._it = 0
        self._q_normal = 0
        # per-method overrides: probe schedule and bridge tolerance
        self.n0 = spec.n0 if spec.n0 is not None else cfg.n0
        if spec.bridge_tol is not None:
            self.bridge_tol = spec.bridge_tol
        elif spec.update in ("pvr", "pvr_ua", "pvr_gc", "pvr_aor", "pvr_aou", "pvr_caou", "pvr_gte", "ramb_pvre",
                             "pvr_pb", "mapvre"):  # build 36: MA-PVRE's latent block is a pvr_pb step
            self.bridge_tol = cfg.pvr_bridge_tol
        else:
            self.bridge_tol = cfg.ttba_bridge_tol
        self.pvr_errs: Deque[float] = deque(maxlen=cfg.pvr_err_window)
        # PVR-E v2 keeps a separate error history normalised by the CURRENT radius r,
        # plus the most recent measured bridge anchor.  It never changes the old PVR-E state.
        self.pvr_ua_errs: Deque[float] = deque(maxlen=cfg.pvr_err_window)
        self.pvr_ua_anchor_kb: Optional[float] = None
        self.pvr_ua_anchor_age: int = 0
        # PVR-E v3 state is separate so published/previous baselines are bit-for-bit unchanged.
        self.pvr_gc_model_resid: Deque[float] = deque(maxlen=cfg.pvr_err_window)  # signed (g_act-g_pred)/r
        self.pvr_gc_gain_hist: Deque[float] = deque(maxlen=max(5, cfg.pvr_err_window))
        self.pvr_gc_escape_stalls: int = 0
        self.pvr_gc_anchor_age: int = 0
        self.pvr_gc_force_model: bool = True  # one verified warm model step after a valid wide anchor
        self.pvr_gc_flat_dev: Deque[float] = deque(maxlen=max(1, cfg.pvr_gc_flat_window))
        self.pvr_gc_thales_left: int = 0
        # PVR-E v4 state.  Separate from every previous baseline.
        self.pvr_aor_radius_resid: Deque[float] = deque(maxlen=cfg.pvr_err_window)
        self.pvr_aor_resolve_hist: Deque[float] = deque(maxlen=max(1, cfg.pvr_aor_resolve_window))
        self.pvr_aor_anchor_age: int = 0
        self.pvr_aor_force_model: bool = True
        self.pvr_aor_planar_left: int = 0
        self._last_aor_normal_split_deg: float = float("nan")
        # PVR-E v5 state: all old baselines remain bit-for-bit separate.
        self.pvr_aou_radius_resid: Deque[float] = deque(maxlen=cfg.pvr_err_window)
        self.pvr_aou_anchor_age: int = 0
        self.pvr_aou_mode: str = "M"  # persistent wide action: M=model, T=TtBA, C=CGBA
        self.pvr_aou_mode_age: int = 0
        self.pvr_aou_last_trial_it: int = -10**9
        self.pvr_aou_last_used = {a: -10**9 for a in ("M", "T", "C", "E", "R")}
        self.pvr_aou_mu = {a: 0.0 for a in ("M", "T", "C", "E", "R")}
        self.pvr_aou_n = {a: 0 for a in ("M", "T", "C", "E", "R")}
        self.pvr_aou_last_S: float = 0.0
        self.pvr_aou_last_weight: float = 0.0
        self.pvr_aou_in_narrow: bool = False
        self.pvr_aou_narrow_action: str = "E"
        self.pvr_aou_narrow_steps: int = 0
        self.pvr_aou_last_narrow_trial_it: int = -10**9
        # PVR-E v6 state: contextual confidence-aware utility, separate from v5.
        self.pvr_caou_radius_resid: Deque[float] = deque(maxlen=cfg.pvr_err_window)
        self.pvr_caou_anchor_age: int = 0
        self.pvr_caou_mode: str = "M"
        self.pvr_caou_mode_age: int = 0
        self.pvr_caou_last_trial_it: int = -10**9
        self.pvr_caou_last_used = {a: -10**9 for a in ("M", "T", "C", "E", "R")}
        self.pvr_caou_hist = {a: deque(maxlen=max(4, cfg.pvr_caou_hist_window)) for a in ("M", "T", "C", "E", "R")}
        self.pvr_caou_last_S: float = 0.0
        self.pvr_caou_last_weight: float = 0.0
        self.pvr_caou_r0: Optional[float] = None
        self.pvr_caou_trial_state = None
        self.pvr_caou_in_narrow: bool = False
        self.pvr_caou_narrow_action: str = "E"
        self.pvr_caou_narrow_steps: int = 0
        self.pvr_caou_last_narrow_trial_it: int = -10**9
        self.pvr_caou_narrow_trial_state = None
        # PVR-E v7 state (build 31a): flatness residuals of measured steps, Thales run length.
        self.gte_eps: Deque[float] = deque(maxlen=max(1, cfg.pvr_gte_window))
        self.gte_thales_run: int = 0
        self.pb_conf: Optional[PBConfig] = None  # build 33b (pvr_pb only), created on first use
        self.pb_state: Optional[PBState] = None
        # build 35 state: VJP availability (None = untested), KSQP trust radius, CPS probe-scale multiplier
        self._vjp_ok: Optional[bool] = None
        self.ksqp_delta: Optional[float] = None
        self.cps_mult: float = 1.0
        self.cps_B: Optional[float] = None  # EMA of the mean-curvature coefficient B = tr K / 2
        self.kkt_mbar: Optional[float] = None  # build 35b: EMA of tr(J^T J) / k (Hutchinson, unit probes)
        self._kkt_gen = torch.Generator().manual_seed(int(seed) + 7_000_003)  # Hutchinson probes (own stream)
        self._pb_reject: Optional[str] = None  # build 35b: why the last _pb_frame returned None
        self.ma_rate = {"L": None, "A": None}  # build 36: EMA log-progress per query of each MA-PVRE block
        self.ma_since = {"L": 0, "A": 0}  # iterations since the block last ran
        # build 37: gate state, per-block exploration interval, pixel-distance history, objective switch (MAL rows)
        self._ma_open = False
        self._ma_forced: Optional[str] = None
        _k0 = max(2, int(spec.ma_explore_every))
        self.ma_k = {"L": _k0, "A": _k0}
        self._ma_hist: list = []  # (queries, best pixel distance) at the start of every mapvre iteration
        self._obj_latent = bool(spec.update == "mapvre" and str(spec.ma_latent) == "lgc")
        self._obj_dirty = False
        self.pb_conf_amb: Optional[PBConfig] = None
        self.pb_state_amb: Optional[PBState] = None
        self._amb_sampler = None
        self._pb_last: dict = {}
        self.cps_G: Optional[float] = None  # EMA of the curvature-fluctuation coefficient Gam
        # RA-MB-PVRE history is per attack instance.  It is reset when entering M from H/L,
        # so targeted/non-targeted runs are automatically separated and stale regimes do not leak.
        self.ra_prev_regime: Optional[str] = None
        self.ra_c_ema: float = cfg.ttba_dive
        self.gam_dev: Deque[float] = deque(maxlen=max(1, cfg.thales_window))  # |gamma_obs - 90| of bisector steps
        self.thales_since = -1
        self.overheads: Deque[int] = deque(maxlen=cfg.nstar_window)
        self._sign_share = float("nan")
        self._dbg_gate_q = 0  # A3 diagnostic: gate queries charged this iteration (pvr_gte sets 1, pvr 0)
        # normal tracking (no queries): None for every published / earlier method
        self.tracker: Optional[NormalTracker] = (
            NormalTracker(self.d_eff, cfg.track_beta, cfg.track_gate)
            if (spec.track and spec.update in PROBE_UPDATES and not spec.metric) else None)

    # ------------------------------------------------------------------ setup
    def _make_probe_sampler(self) -> Optional[ProbeSampler]:
        shape = self.space.shape
        if self.spec.probe == "gaussian":
            return GaussianSampler(shape)
        if self.spec.probe == "dct":
            c, h, w = shape
            norm_to = math.sqrt(c * int(h / self.dct_f) * int(w / self.dct_f))
            return DCTLowFreqSampler(shape, self.dct_f, self.device,
                                     normalize_to=norm_to if self.spec.probe_normalize else None)
        if self.spec.probe == "jacobian":
            mix = self.spec.jac_mix if self.spec.jac_mix is not None else self.cfg.jac_mix
            return JacobianSubspaceSampler(self.jac_V, shape, mix=mix)
        if self.spec.probe == "tangent_chord":
            # build 32a B3: the Jacobian slot carries the decoder-tangent basis (tangent_chord_basis,
            # 0 victim queries).  Mix isotropic probes in span(Q_tan) with DCT probes projected onto
            # the orthogonal complement (floor-free; see claude/next_design_32a_2026-10-02.md Tier B).
            c, h, w = shape
            norm_to = math.sqrt(c * int(h / self.dct_f) * int(w / self.dct_f))
            dct_sampler = DCTLowFreqSampler(shape, self.dct_f, self.device,
                                            normalize_to=norm_to if self.spec.probe_normalize else None)
            frac = self.spec.tan_frac if self.spec.tan_frac is not None else self.cfg.tan_frac
            if not self.spec.probe_normalize:
                # build 32b guard: the 32a LGC_T rows had no TtBA normalisation and the default sigma,
                # i.e. probe L2 = 2e-4 instead of 0.029 -> one-sided signs.  Fail loudly instead.
                raise ValueError(f"{self.spec.name}: tangent_chord probes require probe_normalize=True "
                                 "and sigma=3e-4 (TtBA semantics); use the LGC_TB* rows (build 32b)")
            if self.spec.tan_weighted and self.cfg.tan_energy_ratio is None:
                raise ValueError(f"{self.spec.name}: tan_weighted=True needs EngineConfig.tan_energy_ratio "
                                 "(frozen from diagnose_tangent.py; CLI --tan_energy_ratio)")
            return TangentChordSampler(self.jac_V, dct_sampler, frac, self.device, scale=norm_to)
        return None

    def _probe_dim(self) -> int:
        """Dimension of the probe distribution (enters the theoretical normal quality)."""
        if self.spec.probe == "dct":
            c, h, w = self.space.shape
            return c * int(h / self.dct_f) * int(w / self.dct_f)
        if self.spec.probe == "tangent_chord":
            # TODO(claude-review): the B1/B4 c-hat formula mixes g(N_J, r) and g(N_S, d_S); until the
            # design doc's d_eff convention is confirmed, report the DCT-complement dimension (conservative).
            c, h, w = self.space.shape
            return c * int(h / self.dct_f) * int(w / self.dct_f)
        if self.spec.probe == "jacobian" and self.jac_V is not None:
            return int(self.jac_V.shape[1])
        return self.space.dim

    def _make_axis_sampler(self) -> ProbeSampler:
        shape = self.space.shape
        if self.spec.ta_axis == "dct_mask":
            return TAMaskAxisSampler(shape, self.cfg.ta_ratio_mask, self.cfg.ta_dim_num, self.device)
        if self.spec.ta_axis == "jacobian":
            return JacobianSubspaceSampler(self.jac_V, shape)
        return GaussianSampler(shape)

    # --------------------------------------------------------------- helpers
    @property
    def remaining(self) -> int:
        return self.oracle.remaining

    def _adv(self, z: torch.Tensor, tag: str) -> bool:
        return self.oracle.is_adv(z, tag)

    def _proj(self, z: torch.Tensor) -> torch.Tensor:
        return self.space.project(z)

    def objective(self, z: torch.Tensor) -> float:
        # build 37 (MAL rows): latent-space objective until the ambient gate opens, so the run IS LGC_RC until then
        if self.spec.objective == "pixel" and not self._obj_latent:
            return _norm(self.space.to_image(z) - self.space.x0)
        return _norm(z - self.origin)

    def _radial(self, end: torch.Tensor, verified: bool, max_steps: int = 100) -> Optional[torch.Tensor]:
        """Bisection on [origin, end]; returns a VERIFIED adversarial point or None."""
        lo, hi, hi_ok = self.origin, end, verified
        for _ in range(max_steps):
            if self.remaining <= 0 or _norm(hi - lo) <= self.cfg.tol:
                break
            mid = self._proj(0.5 * (lo + hi))
            if self._adv(mid, "boundary"):
                hi, hi_ok = mid, True
            else:
                lo = mid
        if not hi_ok:
            if self.remaining <= 0 or not self._adv(hi, "boundary"):
                return None
        return hi

    # ---------------------------------------------------------- initialise
    def _init_noise(self, zs: torch.Tensor) -> torch.Tensor:
        """Random init direction; on the latent_ra space (build 36) latent-only, drawn with the BASE shape so the
        RNG stream and the start point equal those of the base latent_r rows."""
        if hasattr(self.space, "lift") and hasattr(self.space, "base_shape"):
            return self.space.lift(self.rng_init.randn(zs.shape[0], *self.space.base_shape))
        return self.rng_init.randn(*zs.shape)

    def _initial_adversarial(self) -> torch.Tensor:
        zs = self.origin
        if self.init_point is not None:  # warm start from a previous stage (e.g. latent -> pixel)
            z = self._proj(self.init_point)
            if self._adv(z, "init"):
                return z
        if self.target_seed is not None:  # latent targeted: hard-label-only seed
            direction = self.target_seed - zs
            if _norm(direction) <= 1e-12:
                raise InitFailed("target seed equals source latent")
            scales, s = [1.0], 1.15
            while s <= 5.0 + 1e-12:
                scales.append(s)
                s *= 1.15
            for sc in scales:
                z = zs + sc * direction
                if self._adv(z, "init"):
                    return z
            center = zs + scales[-1] * direction
            base = max(_norm(direction), 1.0)
            for a in range(1, 101):
                noise = _unit(self._init_noise(zs))
                z = center + (0.02 * a * base) * noise
                if self._adv(z, "init"):
                    return z
            raise InitFailed("no targeted latent initialisation")
        if self.target_image is not None:  # pixel targeted: the target image
            if self._adv(self.target_image, "init"):
                return self.target_image
            raise InitFailed("target image is not classified as the target")
        if self.spec.init == "blend" and self.space.name == "pixel":
            # official HSJA / TA initialisation: misclassified uniform noise, then
            # bisection on the blending factor (tolerance 1e-3, as in hsja.py)
            noise = None
            for _ in range(1000):
                r = self.rng_init.rand(*zs.shape)
                if self._adv(r, "init"):
                    noise = r
                    break
            if noise is None:
                raise InitFailed("no misclassified uniform noise")
            lo, hi = 0.0, 1.0
            while hi - lo > 1e-3:
                mid = 0.5 * (lo + hi)
                if self._adv((1 - mid) * zs + mid * noise, "init"):
                    hi = mid
                else:
                    lo = mid
            return (1 - hi) * zs + hi * noise
        for a in range(1, self.cfg.init_attempts + 1):
            z = self._proj(zs + (a * self.cfg.init_step) * self._init_noise(zs))
            if self._adv(z, "init"):
                return z
        raise InitFailed("no non-targeted initialisation")

    # ------------------------------------------------------ normal estimate
    def _estimate_normal_tangent(self, center: torch.Tensor, q: int) -> torch.Tensor:
        """Build 32b Tier B: weighted two-family sign estimator (TangentChordSampler only).

        The 32a path summed s_i u_i over both families with equal per-probe weight.  Because the
        noise norm (~sqrt(N)) dominates both partial sums, that gives the two unit directions almost
        equal weight, whereas the optimum (maximise cos(n_hat, n) over a, b >= 0 for
        a unit(e_J) + b unit(e_S); Cauchy-Schwarz) is (a, b) ∝ (sqrt(E_J) c_J, sqrt(E_S) c_S),
        giving c^2 = E_J c_J^2 + E_S c_S^2.  Simulation (sim_tierb.py): the equal-weight sum loses up
        to 48 % of the optimal c^2.  E_J / E_S is not identifiable from balanced hard-label signs, so
        rho = E_J / E_S is a frozen constant (EngineConfig.tan_energy_ratio).
        Same query count, tags and diagnostics as _estimate_normal.
        """
        smp = self.sampler
        acc_j = torch.zeros_like(center)
        acc_s = torch.zeros_like(center)
        done, n_pos, n_j = 0, 0, 0
        while done < q:
            b = min(self.cfg.query_batch, q - done)
            u = smp.sample(b, self.rng_normal)
            k = int(smp.last_n_tan)
            signs = self.oracle.signs(self._proj(center + self.sigma * u), "normal")
            su = signs.view(b, *([1] * (u.dim() - 1))) * u
            if k > 0:
                acc_j += su[:k].sum(0, keepdim=True)
            if k < b:
                acc_s += su[k:].sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            n_j += k
            done += b
        self._sign_share = n_pos / max(done, 1)
        self.diag["sign_share"].append(self._sign_share)
        self.diag["n_probes"].append(done)
        n_s = done - n_j
        uj, us = _unit(acc_j), _unit(acc_s)
        if self.spec.tan_weighted:
            rho = float(self.cfg.tan_energy_ratio)
            c_j = (1.0 + math.pi * smp.r / (2.0 * max(n_j, 1))) ** -0.5 if n_j > 0 else 0.0
            c_s = (1.0 + math.pi * smp.d_s / (2.0 * max(n_s, 1))) ** -0.5 if n_s > 0 else 0.0
            w_j, w_s = math.sqrt(max(rho, 0.0)) * c_j, c_s
        else:  # 32a-equivalent equal-per-probe sum (control row LGC_TB_U)
            w_j = float(acc_j.flatten().norm()) if uj is not None else 0.0
            w_s = float(acc_s.flatten().norm()) if us is not None else 0.0
        n = torch.zeros_like(center)
        if uj is not None:
            n = n + w_j * uj
        if us is not None:
            n = n + w_s * us
        n = _unit(n)
        if n is None:
            n = _unit(center - self.origin)
        return n

    def _estimate_normal(self, center: torch.Tensor, q: int) -> torch.Tensor:
        if isinstance(self.sampler, TangentChordSampler):
            return self._estimate_normal_tangent(center, q)
        if self.spec.cps:
            return self._estimate_normal_cps(center, q)
        acc = torch.zeros_like(center)
        baseline = bool(self.spec.sign_baseline)
        usum = torch.zeros_like(center) if baseline else None
        done, n_pos, u2 = 0, 0, 0.0
        while done < q:
            b = min(self.cfg.query_batch, q - done)
            u = self.sampler.sample(b, self.rng_normal)
            signs = self.oracle.signs(self._proj(center + self.sigma * u), "normal")
            acc += (signs.view(b, *([1] * (u.dim() - 1))) * u).sum(0, keepdim=True)
            if baseline:
                usum += u.sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            u2 += float((u * u).sum().item())
            done += b
        if baseline and done > 0:
            # HSJA's baseline: sum_i (s_i - s_bar) u_i.  The removed term s_bar * sum_i u_i carries no
            # information about the normal (zero-mean probes), so the estimator noise falls by the factor
            # (1 - s_bar^2) while the signal is unchanged.  No extra victim queries.
            s_bar = (2.0 * n_pos - done) / done
            acc = acc - s_bar * usum
            self.diag["sign_baseline_gain"].append(1.0 / max(1e-6, 1.0 - s_bar * s_bar))
        # probe-depth diagnostic: share of adversarial (+1) answers. The sign estimator is
        # informative only if both answers occur; ~0.5 = balanced, <0.1 or >0.9 = one-sided
        # (probe radius sigma * |u| too small / too large relative to the boundary offset).
        self._sign_share = n_pos / max(done, 1)
        self.diag["sign_share"].append(self._sign_share)
        self.diag["n_probes"].append(done)
        if self.tracker is not None and done > 0:
            # per-direction probe variance s^2 = E|u|^2 / d_eff (1 for normalised DCT / Gaussian probes);
            # the prior is projected orthogonal to the current direction (inside the probe subspace):
            # the part of the previous estimate along v has already been used by the last step
            v_dir = _unit(self.sampler.project(center - self.origin))
            n = self.tracker.update(acc, done, self._sign_share, u2 / (done * max(self.d_eff, 1)), v_dir=v_dir)
            if n is None:
                n = _unit(center - self.origin)
            return n
        n = _unit(acc)
        if n is None:  # all-same decisions and exactly cancelling probes: no query cost
            n = _unit(center - self.origin)
        if self.spec.metric:
            n = self._metric_direction(n)
        return n

    def _estimate_cgba_normal(self, center: torch.Tensor, q: int) -> Tuple[torch.Tensor, float, int]:
        """Native CGBA normal estimate used by the high-k tri-route branch.

        Standalone CGBA uses the engine default probe scale (normally 2e-4) and
        *unnormalised* low-frequency DCT Gaussian probes.  TtBA/PVR-E instead use
        sigma=3e-4 and fixed-length DCT probes.  The tri-route must first use the
        TtBA-style normal to define its k=0.2/0.8 routing gates; when the CGBA
        route is selected, this function spends a fresh CGBA probe budget so that
        the subsequent CGBA update receives the same kind of normal as standalone
        CGBA.

        Returns (unit normal, +1 sign share, queries actually used).  These victim
        queries are tagged as ``normal`` so the ordinary query breakdown remains
        exact.
        """
        q = int(max(0, q))
        if q <= 0:
            n = _unit(center - self.origin)
            return (center - self.origin if n is None else n), float("nan"), 0

        c, h, w = self.space.shape
        sampler = DCTLowFreqSampler(self.space.shape, self.dct_f, self.device, normalize_to=None)
        acc = torch.zeros_like(center)
        done, n_pos = 0, 0
        sigma = float(self.cfg.sigma)  # native CGBA: MethodSpec.sigma is None -> EngineConfig.sigma
        while done < q:
            b = min(self.cfg.query_batch, q - done)
            u = sampler.sample(b, self.rng_normal)
            signs = self.oracle.signs(self._proj(center + sigma * u), "normal")
            acc += (signs.view(b, *([1] * (u.dim() - 1))) * u).sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            done += b
        n = _unit(acc)
        if n is None:
            n = _unit(center - self.origin)
        if n is None:
            n = torch.zeros_like(center)
        share = n_pos / max(done, 1)
        return n, share, done

    def _estimate_normal_aor(self, center: torch.Tensor, q: int) -> torch.Tensor:
        """Native TtBA normal estimate plus a no-extra-query split-half stability diagnostic.

        The final estimator is identical to ``_estimate_normal`` for this non-tracked,
        non-metric method.  We additionally accumulate the first and second halves of
        the *same* queried probes and report their angle.  This is a computational
        identifiability diagnostic only; it spends no additional victim queries.
        """
        q = int(max(0, q))
        acc = torch.zeros_like(center)
        acc1 = torch.zeros_like(center)
        acc2 = torch.zeros_like(center)
        done, n_pos = 0, 0
        cut = q // 2
        while done < q:
            b = min(self.cfg.query_batch, q - done)
            u = self.sampler.sample(b, self.rng_normal)
            signs = self.oracle.signs(self._proj(center + self.sigma * u), "normal")
            contrib = signs.view(b, *([1] * (u.dim() - 1))) * u
            acc += contrib.sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            a = max(0, min(b, cut - done))
            if a > 0:
                acc1 += contrib[:a].sum(0, keepdim=True)
            if a < b:
                acc2 += contrib[a:].sum(0, keepdim=True)
            done += b
        self._sign_share = n_pos / max(done, 1)
        self.diag["sign_share"].append(self._sign_share)
        self.diag["n_probes"].append(done)
        n = _unit(acc)
        if n is None:
            n = _unit(center - self.origin)
        if n is None:
            n = torch.zeros_like(center)
        n1, n2 = _unit(acc1), _unit(acc2)
        split_deg = float("nan")
        if n1 is not None and n2 is not None:
            cc = max(-1.0, min(1.0, _dot(n1, n2)))
            split_deg = math.degrees(math.acos(cc))
        self._last_aor_normal_split_deg = split_deg
        self.diag["pvr_aor_normal_split_deg"].append(split_deg)
        return n

    def _c_hat(self) -> float:
        """Expected cos(normal estimate, true normal) of the current estimate: the tracker's c_eff
        with normal tracking, otherwise the fresh-estimate theory 1 / sqrt(1 + pi d / 2N)."""
        if self.tracker is not None and self.tracker.c_eff is not None:
            return float(self.tracker.c_eff)
        return c_hat_theory(self._q_normal, self.d_eff)

    def _metric_direction(self, n: torch.Tensor) -> torch.Tensor:
        """(V S^2 V^T + lam s1^2 I)^-1 n restricted to span(V)  (pull-back metric)."""
        V, S = self.jac_V, self.jac_S
        c = V.T @ n.reshape(-1)
        lam = self.cfg.jac_lambda * float(S[0].item()) ** 2
        w = c / (S * S + lam)
        m = _unit((V @ w).view_as(n))
        return n if m is None else m

    # --------------------------------------------------------- CGBA family
    def _plane(self, normal: torch.Tensor, xb: torch.Tensor):
        r_vec = xb - self.origin
        r = _norm(r_vec)
        v = _unit(r_vec)
        eta = _unit(normal)
        if v is None or eta is None:
            return None
        cos_t = max(-1 + 1e-7, min(1 - 1e-7, _dot(eta, v)))
        w = _unit(eta - cos_t * v)
        if w is None:
            return None
        return r, v, w, math.acos(cos_t)

    def _update_cgba(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        plane = self._plane(normal, xb)
        if plane is None:
            return xb
        r, v, w, _ = plane
        xs = self.origin
        clean, last_adv = None, None
        for i in range(1, 41):  # psi_i = pi/2 (1 - 2^-i)
            if self.remaining <= 0:
                break
            psi = 0.5 * math.pi * (1.0 - 2.0 ** (-i))
            zeta = math.cos(psi) * v + math.sin(psi) * w
            cand = self._proj(xs + (r * math.cos(psi)) * zeta)
            if not self._adv(cand, "update"):
                clean = cand
                break
            last_adv = cand
        if clean is None:
            return last_adv if last_adv is not None else xb
        clean_dir = _unit(clean - xs)
        if clean_dir is None:
            return xb
        adv_dir, adv = v, xb
        for _ in range(100):  # BSSP on the same semicircle
            if self.remaining <= 0 or _norm(adv - clean) <= self.cfg.tol:
                break
            mid_dir = _unit(adv_dir + clean_dir)
            if mid_dir is None:
                break
            mid = self._proj(xs + (r * _dot(mid_dir, v)) * mid_dir)
            if self._adv(mid, "update"):
                adv_dir, adv = mid_dir, mid
            else:
                clean_dir, clean = mid_dir, mid
        return adv

    def _update_cgba_h(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        plane = self._plane(normal, xb)
        if plane is None:
            return xb
        r, v, w, theta = plane
        xs = self.origin
        cand = None
        for i in range(1, 41):  # psi_i = theta / 2^i
            if self.remaining <= 0:
                break
            psi = theta / (2.0**i)
            zeta = math.cos(psi) * v + math.sin(psi) * w
            q = self._proj(xs + (r * math.cos(psi)) * zeta)
            if self._adv(q, "update"):
                cand = q
                break
        if cand is None:
            return xb
        res = self._radial(cand, verified=True)
        return cand if res is None else res

    # ------------------------------------------------------------------ TtBA
    def _ttba_k(self, kb: float) -> float:
        c = self.cfg
        if kb <= c.ttba_low:
            return c.ttba_jump * kb
        if kb >= c.ttba_high:
            return c.ttba_dive * kb
        slope = (c.ttba_dive * c.ttba_high - c.ttba_jump * c.ttba_low) / (c.ttba_high - c.ttba_low)
        return slope * (kb - c.ttba_low) + c.ttba_jump * c.ttba_low

    def _bridge(self, normal: torch.Tensor, xb: torch.Tensor) -> Optional[dict]:
        """TtBA bridge search: largest k with x0 + r * unit(k n + (1-k) v) adversarial (1-bit bisection)."""
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return None

        def direction(k: float) -> torch.Tensor:
            d = _unit(k * n + (1.0 - k) * v)
            return v if d is None else d

        q0 = self.oracle.queries
        lo, hi = 0.0, 1.0
        while hi - lo > self.bridge_tol and self.remaining > 0:
            mid = 0.5 * (lo + hi)
            if self._adv(self._proj(xs + r * direction(mid)), "update"):
                lo = mid
            else:
                hi = mid
        self.diag["k_bridge"].append(lo)
        cos_t = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        return {"r": r, "v": v, "n": n, "kb": lo, "lo": lo, "hi": hi,
                "direction": direction, "cos_theta": cos_t,
                "q_bridge": self.oracle.queries - q0}

    def _no_bridge_inside(self, kb: float) -> bool:
        return kb >= 1.0 - 2.0 * self.bridge_tol

    def _ttba_radial_step(self, br: dict, xb: torch.Tensor, step: str) -> torch.Tensor:
        """Official TtBA move: k from the 2/3 (0.9) rule, then full radial search."""
        k = min(1.0, max(0.0, self._ttba_k(br["kb"])))
        d = br["direction"](k)
        high = self._proj(self.origin + br["r"] * d)
        q0 = self.oracle.queries
        res = self._radial(high, verified=(k == 0.0))
        if res is None:
            return xb
        psi = math.acos(max(-1.0, min(1.0, _dot(d, br["v"]))))
        self._log_geom(step, br["r"], br["v"], xb, res, g_pred=br["r"] * math.cos(psi), br=br,
                       q_radial=self.oracle.queries - q0)
        return res

    def _update_ttba(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        br = self._bridge(normal, xb)
        if br is None:
            return xb
        return self._ttba_radial_step(br, xb, "ttba")

    # ------------------------------------------------ proposed bridge geometry
    def _bracket_radial(self, d: torch.Tensor, g_pred: float, g_cap: float,
                        d0: Optional[float] = None) -> Optional[Tuple[torch.Tensor, float]]:
        """Boundary along the unit direction d, starting from a predicted radius g_pred.

        One query at g_pred decides the side; the bracket then grows geometrically
        (d0, 2x, 4x, ...; d0 = bracket_delta unless given) and is bisected to cfg.tol.
        Returns a VERIFIED adversarial point and its radius, or None if the boundary
        lies beyond g_cap (no improvement over the current radius).
        """
        xs, c = self.origin, self.cfg

        def point(t: float) -> torch.Tensor:
            return self._proj(xs + t * d)

        g_pred = min(max(g_pred, 1e-12), g_cap)
        step = c.bracket_delta if d0 is None else max(float(d0), 1e-9)
        if self._adv(point(g_pred), "boundary"):
            hi = g_pred
            lo = hi * (1.0 - step)
            while lo > 0.0 and self._adv(point(lo), "boundary"):
                hi, step = lo, 2.0 * step
                lo = hi * (1.0 - step) if step < 1.0 else 0.0
            lo = max(lo, 0.0)
        else:
            lo = g_pred
            if lo >= g_cap:
                return None
            hi = min(g_pred * (1.0 + step), g_cap)
            while not self._adv(point(hi), "boundary"):
                if hi >= g_cap:
                    return None
                lo, step = hi, 2.0 * step
                hi = min(hi * (1.0 + step), g_cap)
        while hi - lo > c.tol and self.remaining > 0:
            m = 0.5 * (lo + hi)
            if self._adv(point(m), "boundary"):
                hi = m
            else:
                lo = m
        return point(hi), hi

    def _log_geom(self, step: str, r: float, v: torch.Tensor, xb: torch.Tensor, cand: torch.Tensor,
                  g_pred: float = float("nan"), br: Optional[dict] = None, gamma_pred: float = float("nan"),
                  ratio: float = float("nan"), cos_theta: float = float("nan"), q_radial: int = 0) -> None:
        """One row of per-iteration boundary geometry (gamma_obs = angle at the new boundary point)."""
        if not self.cfg.log_geometry:
            return
        dvec = cand - self.origin
        g = _norm(dvec)
        if g <= 1e-12 or r <= 1e-12:
            return
        psi = math.acos(max(-1.0, min(1.0, _dot(dvec, v) / g)))
        t = sag_slope(r, psi, g)
        psi_b, lin_err = float("nan"), float("nan")
        if br is not None and not self._no_bridge_inside(br["kb"]) and br["kb"] > 0.0:
            psi_b_rad = math.acos(max(-1.0, min(1.0, _dot(br["direction"](br["kb"]), br["v"]))))
            psi_b = math.degrees(psi_b_rad)
            # linear-boundary prediction for ANY rotation psi, given the measured bridge angle:
            # g_lin = r cos(a) / cos(psi - a), a = psi_bridge / 2  -> lin_err = 0 on a hyperplane
            a = 0.5 * psi_b_rad
            den = math.cos(psi - a)
            if den > 1e-9:
                lin_err = (g - r * math.cos(a) / den) / r
        self.geom.append({
            "it": self._it, "q": self.oracle.queries, "step": step, "r": r, "g_act": g,
            "g_pred": g_pred, "pred_err": (g_pred - g) / r if math.isfinite(g_pred) else float("nan"),
            "psi_deg": math.degrees(psi), "psi_bridge_deg": psi_b,
            "k_bridge": br["kb"] if br is not None else float("nan"),
            "gamma_obs_deg": math.degrees(0.5 * math.pi + math.atan(t)), "sag_slope": t,
            "gamma_pred_deg": math.degrees(gamma_pred) if math.isfinite(gamma_pred) else float("nan"),
            "chord": _norm(cand - xb), "ratio": ratio, "lin_err": lin_err,
            "cos_theta": br["cos_theta"] if br is not None else cos_theta,
            "q_normal": self._q_normal, "q_bridge": br["q_bridge"] if br is not None else 0,
            "q_radial": q_radial, "sign_share": self._sign_share,
            "c_eff": self._c_hat() if self._q_normal > 0 else float("nan"),
            "trk_y": (self.tracker.hist["y"][-1] if (self.tracker is not None and self.tracker.hist["y"])
                      else float("nan")),
            # build 32a A3 diagnostic (only when --pvr_gte_debug_escape_parity): the fields that
            # differ between the pvr_gte escape branch and the legacy pvr escape branch.
            **({"dbg_gate_q": int(self._dbg_gate_q), "dbg_escape_src": step}
               if self.cfg.pvr_gte_debug_escape_parity else {}),
        })

    def _midpoint_step(self, br: dict, xb: torch.Tensor, step: str, gamma_arc: bool,
                       fit_update: Optional[bool] = None) -> torch.Tensor:
        """Move to the chord midpoint direction; radius from Thales (linear) or the gamma-arc.

        The bisector step is where gamma_obs measures boundary curvature, so its sag slope
        feeds the hyperbolic fit (fit_update; default: only for the gamma-arc update).
        """
        if fit_update is None:
            fit_update = gamma_arc
        xs, r, v, kb = self.origin, br["r"], br["v"], br["kb"]
        if kb <= 0.0:
            return xb  # no adversarial direction in this plane at radius r
        x_bridge = self._proj(xs + r * br["direction"](kb))
        mid = 0.5 * (xb + x_bridge)
        d = _unit(mid - xs)
        if d is None:
            return xb
        psi = math.acos(max(-1.0, min(1.0, _dot(d, v))))
        gamma_pred = float("nan")
        if gamma_arc:
            gamma_pred = self.hfit.predict_gamma(r * math.sin(psi))
            g_pred = inscribed_radius(r, psi, gamma_pred)
            g_pred = min(max(g_pred, 0.05 * r), r * (1.0 - 1e-6))
        else:
            g_pred = _norm(mid - xs)  # = r cos(psi): exact on a linear boundary
        q0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r)
        if out is None:
            return xb
        cand, g_act = out
        self._log_geom(step, r, v, xb, cand, g_pred=g_pred, br=br, gamma_pred=gamma_pred,
                       q_radial=self.oracle.queries - q0)
        if fit_update and psi > 1e-9:
            # abscissa = lateral offset r sin(psi), the same quantity used by predict_gamma
            self.hfit.update(r * math.sin(psi), sag_slope(r, psi, g_act))
        return cand

    def _escape_step(self, br: dict, xb: torch.Tensor, step: str,
                     eps_bracket: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """Narrow-region regime (k_b <= escape_k): TtBA's official weight k(k_b) (0.9 k_b below 0.1,
        interpolation to 2/3 k_b at 0.2), but the radial search is bracketed at the linear-boundary
        prediction g_lin = r cos(psi_b/2) / cos(psi_k - psi_b/2) instead of bisected from the origin.

        eps_bracket (build 32a A4, pvr_gte only): (g_pred, d0) of the eps-predicted bracket,
        replacing the linear prediction and the pvr_errs/bracket_delta margin; None keeps the
        legacy bracket (every earlier method passes None)."""
        kb = br["kb"]
        if kb <= 0.0:
            return xb
        r, v = br["r"], br["v"]
        k = min(1.0, max(0.0, self._ttba_k(kb)))
        d = br["direction"](k)
        psi_k = math.acos(max(-1.0, min(1.0, _dot(d, v))))
        half_b = 0.5 * math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
        den = math.cos(psi_k - half_b)
        g_pred = r * math.cos(half_b) / den if den > 1e-9 else r
        g_pred = min(max(g_pred, 0.05 * r), r * (1.0 - 1e-6))
        d0 = None
        if eps_bracket is not None:
            g_pred, d0 = eps_bracket
        elif len(self.pvr_errs) >= 3:
            q90 = float(np.quantile(np.asarray(self.pvr_errs, dtype=float), 0.9))
            d0 = min(max(self.cfg.pvr_bracket_mult * q90, 2.0 * self.cfg.tol / max(g_pred, 1e-12)),
                     self.cfg.bracket_delta)
        q0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        if out is None:
            return xb
        cand, g_act = out
        self.pvr_errs.append(abs(g_pred - g_act) / max(g_act, 1e-12))
        self._log_geom(step, r, v, xb, cand, g_pred=g_pred, br=br, q_radial=self.oracle.queries - q0)
        return cand

    def _update_mid(self, normal: torch.Tensor, xb: torch.Tensor, gamma_arc: bool) -> torch.Tensor:
        br = self._bridge(normal, xb)
        if br is None:
            return xb
        if self._no_bridge_inside(br["kb"]):  # early phase (2 alpha_p >= theta): official TtBA move
            return self._ttba_radial_step(br, xb, "ttba_fallback")
        if self.spec.escape and br["kb"] <= self.cfg.escape_k:  # narrow region: TtBA k rule
            return self._escape_step(br, xb, "escape")
        return self._midpoint_step(br, xb, "garc" if gamma_arc else "midpoint", gamma_arc)

    def _thales_vote(self, gamma_dev_deg: float) -> None:
        """Record |gamma_obs - 90| of a bisector / model step (both in wide regions, k_b > escape_k); if the
        median of the last thales_window is <= thales_gamma_tol the boundary is flat -> Thales regime."""
        self.gam_dev.append(gamma_dev_deg)
        if (len(self.gam_dev) >= self.gam_dev.maxlen
                and float(np.median(np.asarray(self.gam_dev))) <= self.cfg.thales_gamma_tol):
            self.hyb_mode = "thales"
            self.thales_since = self._it + 1

    def _update_hybrid(self, normal: torch.Tensor, xb: torch.Tensor, pvr: bool = False,
                       bridge_override: Optional[dict] = None) -> torch.Tensor:
        """HYB (pvr=False) and predict-verify-refine (pvr=True).

        Both keep a pooled estimate of d* = r cos(alpha) (alpha = angle(v, n*)) and alternate
        between a BRIDGE step (bridge bisection -> bisector -> measured alpha) and a MODEL
        step (closed-form in-plane optimum psi*(c_hat, theta, alpha), no bridge queries),
        verified by a bracketed radial search and a trust ratio.
        PVR changes the model step only:
          predict : radius on the gamma-arc of the online hyperbolic fit (Thales if not ready);
          verify  : first bracket = 3 x q90 of recent relative radius errors;
          refine  : the two boundary points (x_b, new point) give the secant foot psi_s
                    (curvature-corrected), hence cos(alpha); it is pooled with weight
                    pvr_calib_weight instead of HYB's weak single-shot sample r cos(theta)/c_hat;
        and its bridge steps also feed the hyperbolic fit (bisector = curvature probe).
        """
        c = self.cfg
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb
        self._dbg_gate_q = 0  # A3 diagnostic: the legacy pvr path charges no k=0.2 gate query
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        w = _unit(n - ct * v)
        if w is None:
            return xb
        theta = math.acos(ct)
        tag = "pvr" if pvr else "hyb"
        c_hat = min(0.999, self._c_hat())
        pool = self.hyb_pool
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2  # inverse variance of one d* sample
        ca_model = (pool[0] / pool[1]) / r if pool[1] > 0 else None

        def pool_add(cos_alpha: float, weight: float) -> None:
            pool[0] = c.hyb_lambda * pool[0] + weight * r * cos_alpha
            pool[1] = c.hyb_lambda * pool[1] + weight

        if pvr and self.spec.thales and self.hyb_mode == "thales":
            if self._it - self.thales_since < c.thales_recheck:
                cand = self._update_cgba(n, xb)  # Thales one-search: flat boundary -> exact in-plane foot
                if cand is not xb:
                    self._log_geom(f"{tag}_thales", r, v, xb, cand, cos_theta=ct)
                return cand
            self.hyb_mode = "bridge"  # re-measure k_b and gamma with a bisector step
        if self.hyb_mode == "bridge" or ca_model is None:
            # The 30c router may already have refined k_b inside the gate-certified
            # 0.2<k_b<0.8 interval. Reuse that bridge instead of paying for a second
            # search. Published standalone TtBA_PVRE calls leave bridge_override=None.
            br = bridge_override if bridge_override is not None else self._bridge(n, xb)
            if br is None:
                return xb
            if self._no_bridge_inside(br["kb"]):
                return self._ttba_radial_step(br, xb, f"{tag}_fallback")
            if self.spec.escape and br["kb"] <= c.escape_k:  # narrow region: TtBA k rule, stay in bridge mode
                self.hyb_mode = "bridge"
                return self._escape_step(br, xb, f"{tag}_escape")
            psi_b = math.acos(max(-1.0, min(1.0, _dot(br["direction"](br["kb"]), v))))
            ap = 0.5 * psi_b
            if ca_model is not None and ap > 1e-9:
                psi_m = model_psi(c_hat, ct, min(max(ca_model, 1e-3), 0.999))
                self.hyb_mode = "model" if abs(psi_m / ap - 1.0) < c.hyb_agree else "bridge"
            ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)
            cand = self._midpoint_step(br, xb, f"{tag}_bridge", gamma_arc=False, fit_update=pvr)
            if pvr and self.spec.thales and cand is not xb:
                self._thales_vote(abs(math.degrees(math.atan(sag_slope(r, ap, _norm(cand - xs))))))
            return cand

        # ---- model step: no bridge queries
        if not pvr and ct > 0.0:  # HYB: single-shot sample r cos(theta) / c_hat (weak, biased)
            pool_add(ct / c_hat, unit_w)
        ca = min(max((pool[0] / pool[1]) / r, 1e-3), 0.999)
        psi = model_psi(c_hat, ct, ca)
        psi = min(max(psi, 0.0), 0.95 * theta)
        d = _unit(math.cos(psi) * v + math.sin(psi) * w)
        if d is None:
            return xb
        gamma_pred = float("nan")
        if pvr and self.hfit.ready():  # predict: gamma-arc radius
            gamma_pred = self.hfit.predict_gamma(r * math.sin(psi))
            g_pred = min(max(inscribed_radius(r, psi, gamma_pred), 0.05 * r), r * (1.0 - 1e-6))
        else:  # Thales radius (exact on a hyperplane)
            g_pred = r * math.cos(psi)
        d0 = None
        if pvr and len(self.pvr_errs) >= 3:  # verify: bracket matched to the recent prediction error
            q90 = float(np.quantile(np.asarray(self.pvr_errs, dtype=float), 0.9))
            d0 = min(max(c.pvr_bracket_mult * q90, 2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        q0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        if out is None:  # overshoot beyond the bridge angle: re-measure next time
            self.hyb_mode = "bridge"
            return xb
        cand, g_act = out
        self.pvr_errs.append(abs(g_pred - g_act) / max(g_act, 1e-12))
        ratio = (r - g_act) / max(r - g_pred, 1e-12)
        if not (c.hyb_ratio_lo < ratio < c.hyb_ratio_hi):
            self.hyb_mode = "bridge"
        if pvr:  # refine: secant foot from (x_b, new point), curvature from the fit
            kappa = 0.0
            if self.hfit.ready():
                s_lat = r * math.sin(psi)
                kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
            psi_s = secant_foot(r, g_act, psi, kappa, theta)
            if 1e-6 < psi_s < theta:
                ca_cal = cos_alpha_from_bridge(c_hat, ct, psi_s)
                if 0.0 < ca_cal < 1.0:
                    pool_add(ca_cal, c.pvr_calib_weight * unit_w)
        self._log_geom(f"{tag}_model", r, v, xb, cand, g_pred=g_pred, gamma_pred=gamma_pred, ratio=ratio,
                       cos_theta=ct, q_radial=self.oracle.queries - q0)
        if pvr and self.spec.thales:  # a model step also measures gamma (sag of the chord x_b -> new point)
            self._thales_vote(abs(math.degrees(math.atan(sag_slope(r, psi, g_act)))))
        return cand


    # -------------------------------- PVR-E v2: exact escape + uncertainty-aware single selection
    def _refine_bridge(self, br: dict, target_tol: float) -> dict:
        """Refine an ALREADY measured bridge bracket without repeating [0,1].

        ``_bridge`` stores its certified adversarial/non-adversarial bracket ``[lo, hi]``.
        PVR-E normally stops at 1e-2.  In the narrow escape regime we refine only that
        small interval to TtBA's 1e-3 tolerance, so the fallback is numerically the TtBA
        bridge rather than merely the same k-rule applied to a coarse bridge.
        """
        lo = float(br.get("lo", br["kb"]))
        hi = float(br.get("hi", min(1.0, lo + self.bridge_tol)))
        target_tol = max(float(target_tol), 1e-12)
        q0 = self.oracle.queries
        while hi - lo > target_tol and self.remaining > 0:
            mid = 0.5 * (lo + hi)
            x = self._proj(self.origin + br["r"] * br["direction"](mid))
            if self._adv(x, "update"):
                lo = mid
            else:
                hi = mid
        out = dict(br)
        out.update({"kb": lo, "lo": lo, "hi": hi,
                    "q_bridge": int(br.get("q_bridge", 0)) + (self.oracle.queries - q0)})
        # _bridge already wrote one diagnostic entry for this iteration; replace the
        # coarse value rather than adding a second pseudo-iteration.
        if self.diag.get("k_bridge"):
            self.diag["k_bridge"][-1] = lo
        return out

    def _pvr_ua_pred_radius(self, r: float, psi: float) -> Tuple[float, float]:
        """Prediction shared by both candidate directions in PVR-E v2.

        Returning the same boundary model for PVR and anchored-TtBA directions is
        important: selection compares DIRECTIONS rather than confounding the comparison
        with different radius estimators.  The second return is gamma (NaN if Thales).
        """
        gamma = float("nan")
        if self.hfit.ready():
            gamma = self.hfit.predict_gamma(r * math.sin(psi))
            g = inscribed_radius(r, psi, gamma)
        else:
            g = r * math.cos(psi)
        g = min(max(g, 0.05 * r), r * (1.0 - 1e-6))
        return g, gamma

    def _pvr_ua_uncertainty(self) -> float:
        """Recent calibrated relative-radius error used as a conservative model margin."""
        c = self.cfg
        if len(self.pvr_ua_errs) < max(1, c.pvr_ua_min_errs):
            return float("inf")
        q = float(np.quantile(np.asarray(self.pvr_ua_errs, dtype=float), c.pvr_ua_uncert_q))
        return max(c.pvr_ua_uncert_floor, q)

    def _pvr_ua_add_error(self, g_pred: float, g_act: float, r: float) -> None:
        if r > 1e-12 and math.isfinite(g_pred) and math.isfinite(g_act):
            self.pvr_ua_errs.append(abs(g_pred - g_act) / r)

    def _update_pvre_ua(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v2: exact escape, periodic re-anchor, uncertainty-aware single direction.

        This deliberately does *not* perform a multi-candidate victim-query race.  On a
        model step it predicts the radius of two candidate directions for free using the
        same fitted boundary model:

          P : the existing PVR closed-form model direction;
          T : the TtBA k-rule applied to the most recent measured bridge k_b.

        The P direction is selected only if

            g_P + m * u * r < g_T,

        where u is a recent q-quantile of |g_pred-g_act|/r.  Otherwise T is selected.
        Only that selected ray is queried/refined.  Every K such steps a bridge is
        re-measured.  If a measured k_b <= 0.2, the bridge is refined to 1e-3 and the
        official TtBA full radial search is used (exact escape safeguard).
        """
        c = self.cfg
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        w = _unit(n - ct * v)
        if w is None:
            return xb
        theta = math.acos(ct)
        c_hat = min(0.999, self._c_hat())
        pool = self.hyb_pool
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2
        ca_model = (pool[0] / pool[1]) / r if pool[1] > 0 else None

        def pool_add(cos_alpha: float, weight: float) -> None:
            pool[0] = c.hyb_lambda * pool[0] + weight * r * cos_alpha
            pool[1] = c.hyb_lambda * pool[1] + weight

        need_anchor = (
            self.hyb_mode == "bridge"
            or ca_model is None
            or self.pvr_ua_anchor_kb is None
            or self.pvr_ua_anchor_age >= max(1, c.pvr_ua_reanchor_every)
        )

        if need_anchor:
            periodic = int(self.pvr_ua_anchor_kb is not None and
                           self.pvr_ua_anchor_age >= max(1, c.pvr_ua_reanchor_every))
            self.diag["pvr_ua_reanchor"].append(periodic)
            self.diag["pvr_ua_anchor_age"].append(self.pvr_ua_anchor_age)
            br = self._bridge(n, xb)
            if br is None:
                self.hyb_mode = "bridge"
                self.diag["pvr_ua_choice"].append("bridge_fail")
                self.diag["pvr_ua_exact_escape"].append(0)
                return xb

            # Early no-crossing-inside case: preserve TtBA's robust full radial fallback.
            if self._no_bridge_inside(br["kb"]):
                self.pvr_ua_anchor_kb = float(br["kb"])
                self.pvr_ua_anchor_age = 0
                self.hyb_mode = "bridge"
                self.diag["pvr_ua_anchor_kb"].append(float(br["kb"]))
                self.diag["pvr_ua_choice"].append("ttba_fallback")
                self.diag["pvr_ua_exact_escape"].append(0)
                return self._ttba_radial_step(br, xb, "pvr_ua_fallback")

            # Exact narrow-region safeguard: coarse PVR bridge -> local refinement to
            # official TtBA tolerance -> official TtBA radial search.
            if br["kb"] <= c.escape_k:
                br = self._refine_bridge(br, c.ttba_bridge_tol)
                self.pvr_ua_anchor_kb = float(br["kb"])
                self.pvr_ua_anchor_age = 0
                self.hyb_mode = "bridge"
                self.diag["pvr_ua_anchor_kb"].append(float(br["kb"]))
                self.diag["pvr_ua_choice"].append("ttba_escape_exact")
                self.diag["pvr_ua_exact_escape"].append(1)
                return self._ttba_radial_step(br, xb, "pvr_ua_escape")

            # Wide bridge anchor: same bisector calibration as PVR-E, but also store k_b
            # for the conservative anchored-TtBA candidate used between anchors.
            self.pvr_ua_anchor_kb = float(br["kb"])
            self.pvr_ua_anchor_age = 0
            self.diag["pvr_ua_anchor_kb"].append(float(br["kb"]))
            self.diag["pvr_ua_choice"].append("bridge")
            self.diag["pvr_ua_exact_escape"].append(0)

            psi_b = math.acos(max(-1.0, min(1.0, _dot(br["direction"](br["kb"]), v))))
            ap = 0.5 * psi_b
            if ca_model is not None and ap > 1e-9:
                psi_m = model_psi(c_hat, ct, min(max(ca_model, 1e-3), 0.999))
                self.hyb_mode = "model" if abs(psi_m / ap - 1.0) < c.hyb_agree else "bridge"
            ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)
            cand = self._midpoint_step(br, xb, "pvr_ua_bridge", gamma_arc=False, fit_update=True)
            if cand is not xb:
                g_pred = r * math.cos(ap)
                g_act = _norm(cand - xs)
                self._pvr_ua_add_error(g_pred, g_act, r)
            return cand

        # ---- between anchors: compare two PREDICTED directions, query only the winner
        ca = min(max((pool[0] / pool[1]) / r, 1e-3), 0.999)
        psi_p = min(max(model_psi(c_hat, ct, ca), 0.0), 0.95 * theta)
        d_p = _unit(math.cos(psi_p) * v + math.sin(psi_p) * w)
        if d_p is None:
            self.hyb_mode = "bridge"
            return xb
        g_p, gamma_p = self._pvr_ua_pred_radius(r, psi_p)

        # Conservative candidate: TtBA k-rule based on the latest measured bridge anchor.
        kb_ref = min(1.0, max(0.0, float(self.pvr_ua_anchor_kb)))
        k_t = min(1.0, max(0.0, self._ttba_k(kb_ref)))
        d_t = _unit(k_t * n + (1.0 - k_t) * v)
        if d_t is None:
            d_t = v
        psi_t = math.acos(max(-1.0, min(1.0, _dot(d_t, v))))
        g_t, gamma_t = self._pvr_ua_pred_radius(r, psi_t)

        u = self._pvr_ua_uncertainty()
        margin = c.pvr_ua_uncert_mult * u * r if math.isfinite(u) else float("inf")
        choose_model = math.isfinite(margin) and (g_p + margin < g_t)
        choice = "model" if choose_model else "ttba"
        d, psi, g_pred, gamma_pred = ((d_p, psi_p, g_p, gamma_p)
                                      if choose_model else (d_t, psi_t, g_t, gamma_t))
        self.diag["pvr_ua_choice"].append(choice)
        self.diag["pvr_ua_uncert"].append(u)
        self.diag["pvr_ua_anchor_age"].append(self.pvr_ua_anchor_age)
        self.diag["pvr_ua_reanchor"].append(0)
        self.diag["pvr_ua_exact_escape"].append(0)
        self.diag["pvr_ua_anchor_kb"].append(kb_ref)
        self.diag["pvr_ua_pred_model"].append(g_p / r)
        self.diag["pvr_ua_pred_ttba"].append(g_t / r)

        d0 = None
        if len(self.pvr_ua_errs) >= max(1, c.pvr_ua_min_errs):
            qerr = float(np.quantile(np.asarray(self.pvr_ua_errs, dtype=float), c.pvr_ua_uncert_q))
            d0 = min(max(c.pvr_bracket_mult * max(qerr, c.pvr_ua_uncert_floor),
                         2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        q0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        self.pvr_ua_anchor_age += 1
        if out is None:
            self.hyb_mode = "bridge"
            return xb
        cand, g_act = out
        self._pvr_ua_add_error(g_pred, g_act, r)
        ratio = (r - g_act) / max(r - g_pred, 1e-12)
        if not (c.hyb_ratio_lo < ratio < c.hyb_ratio_hi):
            self.hyb_mode = "bridge"

        # A verified boundary point on either selected ray contains useful secant geometry.
        kappa = 0.0
        if self.hfit.ready():
            s_lat = r * math.sin(psi)
            kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
        psi_s = secant_foot(r, g_act, psi, kappa, theta)
        if 1e-6 < psi_s < theta:
            ca_cal = cos_alpha_from_bridge(c_hat, ct, psi_s)
            if 0.0 < ca_cal < 1.0:
                pool_add(ca_cal, c.pvr_calib_weight * unit_w)

        self._log_geom(f"pvr_ua_{choice}", r, v, xb, cand, g_pred=g_pred,
                       gamma_pred=gamma_pred, ratio=ratio, cos_theta=ct,
                       q_radial=self.oracle.queries - q0)
        return cand


    # -------------------------------- PVR-E v3: geometry-calibrated confidence controller
    def _pvr_gc_robust_stats(self) -> Tuple[float, float]:
        """Robust signed model-residual calibration.

        Residuals are e=(g_act-g_pred)/r.  The median estimates persistent radius bias;
        1.4826*MAD estimates a robust relative-radius scale.  Unlike 30f's q90(|e|),
        common signed bias is calibrated instead of being counted twice as uncertainty.
        """
        c = self.cfg
        if not self.pvr_gc_model_resid:
            return 0.0, float("inf")
        a = np.asarray(self.pvr_gc_model_resid, dtype=float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        scale = max(c.pvr_gc_mad_floor, 1.4826 * mad)
        return med, scale

    def _pvr_gc_record_gain(self, r: float, cand: torch.Tensor) -> float:
        if r <= 1e-12 or cand is None:
            return 0.0
        g = _norm(cand - self.origin)
        gain = max(0.0, (r - g) / r)
        if math.isfinite(gain):
            self.pvr_gc_gain_hist.append(gain)
        return gain

    def _pvr_gc_flat_vote(self, r: float, psi: float, g_act: float) -> float:
        dev = abs(math.degrees(math.atan(sag_slope(r, psi, g_act))))
        self.pvr_gc_flat_dev.append(dev)
        self.diag["pvr_gc_gamma_dev"].append(dev)
        c = self.cfg
        if (len(self.pvr_gc_flat_dev) >= max(1, c.pvr_gc_flat_window)
                and float(np.median(np.asarray(self.pvr_gc_flat_dev, dtype=float))) <= c.pvr_gc_flat_tol_deg):
            self.pvr_gc_thales_left = max(self.pvr_gc_thales_left, c.pvr_gc_thales_steps)
        return dev

    def _update_pvre_gc(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v3 / geometry-calibrated confidence controller.

        Evidence-driven corrections to 30f:
          * keep ORIGINAL PVR-E R1 as the normal narrow-region escape;
          * use exact TtBA only as a one-shot rescue after repeated R1 stalls;
          * calibrate MODEL prediction with signed median + MAD using model-ray residuals only;
          * force one model warm-up after a valid wide anchor so confidence cannot self-starve;
          * re-anchor only on model failure / age, never because a conservative step disagrees;
          * if recent verified geometry is locally flat, temporarily use CGBA's Thales one-search
            with the SAME native TtBA normal (no second normal estimate), then re-anchor.

        The flatness gate targets the empirical NT regime where CGBA is strong, while curved
        targeted regions remain in PVR-E.  No multi-branch victim-query race is used.
        """
        c = self.cfg
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        w = _unit(n - ct * v)
        if w is None:
            return xb
        theta = math.acos(ct)
        c_hat = min(0.999, self._c_hat())
        pool = self.hyb_pool
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2

        def pool_add(cos_alpha: float, weight: float) -> None:
            pool[0] = c.hyb_lambda * pool[0] + weight * r * cos_alpha
            pool[1] = c.hyb_lambda * pool[1] + weight

        def log_choice(choice: str, reanchor: int = 0, kb: float = float("nan"), stall: int = 0,
                       gp: float = float("nan"), ga: float = float("nan")) -> None:
            bias, scale = self._pvr_gc_robust_stats()
            self.diag["pvr_gc_choice"].append(choice)
            self.diag["pvr_gc_reanchor"].append(int(reanchor))
            self.diag["pvr_gc_bias"].append(bias if math.isfinite(bias) else float("nan"))
            self.diag["pvr_gc_scale"].append(scale if math.isfinite(scale) else float("nan"))
            self.diag["pvr_gc_anchor_kb"].append(kb)
            self.diag["pvr_gc_anchor_age"].append(self.pvr_gc_anchor_age)
            self.diag["pvr_gc_stall"].append(int(stall))
            self.diag["pvr_gc_pred_model"].append(gp / r if math.isfinite(gp) else float("nan"))
            self.diag["pvr_gc_pred_anchor"].append(ga / r if math.isfinite(ga) else float("nan"))

        # Flat, wide local geometry: borrow CGBA's exact-on-a-hyperplane Thales/BSSP update,
        # but DO NOT re-estimate a CGBA normal (that would roughly double the dominant normal cost).
        if self.pvr_gc_thales_left > 0 and self.hyb_mode != "bridge":
            q0 = self.oracle.queries
            cand = self._update_cgba(n, xb)
            self.pvr_gc_thales_left -= 1
            log_choice("thales")
            if cand is not xb:
                self._pvr_gc_record_gain(r, cand)
                self._log_geom("pvr_gc_thales", r, v, xb, cand, cos_theta=ct,
                               q_radial=self.oracle.queries - q0)
            self.pvr_gc_anchor_age += 1
            if self.pvr_gc_thales_left <= 0:
                self.hyb_mode = "bridge"
            return cand

        def anchor_step(reason: str) -> torch.Tensor:
            br = self._bridge(n, xb)
            if br is None:
                self.hyb_mode = "bridge"
                log_choice("bridge_fail", 1)
                return xb
            kb = float(br["kb"])
            self.pvr_gc_anchor_age = 0

            if self._no_bridge_inside(kb):
                log_choice("fallback", 1, kb)
                cand = self._ttba_radial_step(br, xb, "pvr_gc_fallback")
                self._pvr_gc_record_gain(r, cand)
                self.hyb_mode = "bridge"
                return cand

            # Narrow region: keep ORIGINAL PVR-E R1.  Exact TtBA is only a rescue after
            # repeated poor R1 progress, because 30f showed always-exact escape hurts targets.
            if kb <= c.escape_k:
                use_rescue = self.pvr_gc_escape_stalls >= max(1, c.pvr_gc_stall_patience)
                if use_rescue:
                    br2 = self._refine_bridge(br, c.ttba_bridge_tol)
                    log_choice("rescue", 1, float(br2["kb"]), stall=1)
                    cand = self._ttba_radial_step(br2, xb, "pvr_gc_rescue")
                    self.pvr_gc_escape_stalls = 0
                else:
                    cand = self._escape_step(br, xb, "pvr_gc_escape")
                    gain = max(0.0, (r - _norm(cand - xs)) / r) if cand is not xb else 0.0
                    ref = float(np.median(np.asarray(self.pvr_gc_gain_hist, dtype=float))) \
                          if len(self.pvr_gc_gain_hist) >= 3 else 0.0
                    stalled = int(ref > 0.0 and gain < c.pvr_gc_stall_ratio * ref)
                    self.pvr_gc_escape_stalls = self.pvr_gc_escape_stalls + 1 if stalled else 0
                    log_choice("escape", 1, kb, stall=stalled)
                    self._pvr_gc_record_gain(r, cand)
                self.hyb_mode = "bridge"
                return cand

            # Wide anchor: bridge gives a current-plane, query-certified angular span.
            psi_b = math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
            ap = 0.5 * psi_b
            ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)

            # Query-free current-anchor choice: bisector B vs TtBA T.  Both are scored by
            # THE SAME fitted boundary model, so common radius bias cancels in the difference.
            x_bridge = self._proj(xs + r * br["direction"](kb))
            d_b = _unit(0.5 * (xb + x_bridge) - xs)
            if d_b is None:
                d_b = v
            psi_bis = math.acos(max(-1.0, min(1.0, _dot(d_b, v))))
            g_b, gamma_b = self._pvr_ua_pred_radius(r, psi_bis)

            kt = min(1.0, max(0.0, self._ttba_k(kb)))
            d_t = br["direction"](kt)
            psi_t = math.acos(max(-1.0, min(1.0, _dot(d_t, v))))
            g_t, gamma_t = self._pvr_ua_pred_radius(r, psi_t)

            choose_t = self.hfit.ready() and (g_t + c.pvr_gc_anchor_deadband * r < g_b)
            d0 = None
            q0 = self.oracle.queries
            if choose_t:
                out = self._bracket_radial(d_t, g_t, r, d0=d0)
                choice, d_sel, psi_sel, g_pred, gamma_pred = "anchor_t", d_t, psi_t, g_t, gamma_t
            else:
                out = self._bracket_radial(d_b, g_b, r, d0=d0)
                choice, d_sel, psi_sel, g_pred, gamma_pred = "anchor_b", d_b, psi_bis, g_b, gamma_b
            if out is None:
                self.hyb_mode = "bridge"
                log_choice("anchor_fail", 1, kb, gp=float("nan"), ga=g_pred)
                return xb
            cand, g_act = out
            self._log_geom(f"pvr_gc_{choice}", r, v, xb, cand, g_pred=g_pred, br=br,
                           gamma_pred=gamma_pred, cos_theta=ct, q_radial=self.oracle.queries - q0)
            if psi_sel > 1e-9:
                self.hfit.update(r * math.sin(psi_sel), sag_slope(r, psi_sel, g_act))
                self._pvr_gc_flat_vote(r, psi_sel, g_act)
            self._pvr_gc_record_gain(r, cand)
            log_choice(choice, 1, kb, ga=g_pred)
            self.pvr_gc_force_model = True
            self.hyb_mode = "model"
            return cand

        ca_model = (pool[0] / pool[1]) / r if pool[1] > 0 else None
        if (self.hyb_mode == "bridge" or ca_model is None
                or self.pvr_gc_anchor_age >= max(1, c.pvr_gc_reanchor_every)):
            return anchor_step("periodic")

        # Between anchors: model-only continuation.  If confidence is insufficient we
        # re-anchor NOW; we do not use a stale k_b in a new (v,n) plane.
        ca = min(max(ca_model, 1e-3), 0.999)
        psi = min(max(model_psi(c_hat, ct, ca), 0.0), 0.95 * theta)
        d = _unit(math.cos(psi) * v + math.sin(psi) * w)
        if d is None:
            self.hyb_mode = "bridge"
            return anchor_step("degenerate")
        g_pred, gamma_pred = self._pvr_ua_pred_radius(r, psi)
        bias, scale = self._pvr_gc_robust_stats()
        warm = self.pvr_gc_force_model or len(self.pvr_gc_model_resid) < max(1, c.pvr_gc_min_model_errs)
        if not warm:
            g_cal = g_pred + bias * r
            conservative_gain = r - g_cal - c.pvr_gc_mad_mult * scale * r
            if conservative_gain <= 0.0:
                self.hyb_mode = "bridge"
                return anchor_step("uncertain")

        # Verify one model ray only.  Robust calibration history contains ONLY model-ray residuals.
        q0 = self.oracle.queries
        d0 = None
        if math.isfinite(scale):
            d0 = min(max(c.pvr_bracket_mult * max(scale, c.pvr_gc_mad_floor),
                         2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        self.pvr_gc_force_model = False
        if out is None:
            self.hyb_mode = "bridge"
            log_choice("model_fail", 0, gp=g_pred)
            return xb
        cand, g_act = out
        resid = (g_act - g_pred) / r
        if math.isfinite(resid):
            self.pvr_gc_model_resid.append(resid)
        bias2, scale2 = self._pvr_gc_robust_stats()
        ratio = (r - g_act) / max(r - g_pred, 1e-12)
        if not (c.hyb_ratio_lo < ratio < c.hyb_ratio_hi):
            self.hyb_mode = "bridge"
        else:
            self.hyb_mode = "model"

        # secant calibration = original PVR-E refinement
        kappa = 0.0
        if self.hfit.ready():
            s_lat = r * math.sin(psi)
            kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
        psi_s = secant_foot(r, g_act, psi, kappa, theta)
        if 1e-6 < psi_s < theta:
            ca_cal = cos_alpha_from_bridge(c_hat, ct, psi_s)
            if 0.0 < ca_cal < 1.0:
                pool_add(ca_cal, c.pvr_calib_weight * unit_w)

        self._log_geom("pvr_gc_model", r, v, xb, cand, g_pred=g_pred, gamma_pred=gamma_pred,
                       ratio=ratio, cos_theta=ct, q_radial=self.oracle.queries - q0)
        self._pvr_gc_flat_vote(r, psi, g_act)
        self._pvr_gc_record_gain(r, cand)
        self.pvr_gc_anchor_age += 1
        log_choice("model", 0, gp=g_pred)
        return cand


    # ---------------------------- PVR-E v4: adaptive-order regularized controller
    def _pvr_aor_robust_stats(self) -> Tuple[float, float]:
        """Robust error model for the curved radius predictor.

        Residuals are signed e=(g_act-g_curv)/r.  The median removes persistent
        common bias; 1.4826*MAD is a robust relative-radius uncertainty scale.
        """
        c = self.cfg
        if not self.pvr_aor_radius_resid:
            return 0.0, float("inf")
        a = np.asarray(self.pvr_aor_radius_resid, dtype=float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return med, max(c.pvr_aor_mad_floor, 1.4826 * mad)

    def _pvr_aor_pred_radius(self, r: float, psi: float) -> Tuple[float, float, float, float, float, float, float]:
        """Adaptive model-order prediction on one ray.

        g0 is the first-order (hyperplane/Thales) prediction.  g2 is the fitted
        second-order PVR-E prediction.  Their calibrated separation

            D = |g2_cal - g0| / r

        is compared with the robust prediction-noise scale U.  The shrinkage

            w = S^2/(1+S^2),   S=D/U

        continuously suppresses an unresolvable curvature correction instead of
        thresholding |gamma-90|.  Returns
        (g_reg, gamma, g0, g2_raw, D, S, w).
        """
        g0 = r * math.cos(psi)
        gamma = float("nan")
        g2 = g0
        if self.hfit.ready():
            gamma = self.hfit.predict_gamma(r * math.sin(psi))
            g2 = inscribed_radius(r, psi, gamma)
        bias, scale = self._pvr_aor_robust_stats()
        # Resolvability is about the *geometric model-order correction* itself,
        # not a persistent calibration bias.  Therefore D compares raw second-
        # versus first-order predictions; the signed median is used only to
        # calibrate the radius that will actually be queried.
        D = abs(g2 - g0) / max(r, 1e-12)
        g2_cal = g2 + (bias * r if math.isfinite(bias) else 0.0)
        if math.isfinite(scale):
            S = D / max(scale, self.cfg.pvr_aor_mad_floor, 1e-12)
        else:
            S = 0.0
        w = (S * S) / (1.0 + S * S)
        g = g0 + w * (g2_cal - g0)
        g = min(max(g, 0.05 * r), r * (1.0 - 1e-6))
        return g, gamma, g0, g2, D, S, w

    def _pvr_aor_record_resolution(self, S: float) -> None:
        """Enter the planar model only after repeated *unresolved* curvature evidence."""
        c = self.cfg
        if not math.isfinite(S) or not self.hfit.ready():
            return
        if len(self.pvr_aor_radius_resid) < max(1, c.pvr_aor_min_model_errs):
            return
        self.pvr_aor_resolve_hist.append(float(S))
        if len(self.pvr_aor_resolve_hist) < max(1, c.pvr_aor_resolve_window):
            return
        med = float(np.median(np.asarray(self.pvr_aor_resolve_hist, dtype=float)))
        if med <= c.pvr_aor_resolve_enter:
            self.pvr_aor_planar_left = max(self.pvr_aor_planar_left, c.pvr_aor_planar_steps)
            # A fresh evidence window is required after the planar burst.  This is
            # hysteresis by evidence accumulation, not a gamma-degree hard gate.
            self.pvr_aor_resolve_hist.clear()

    def _pvr_aor_bridge_rad_uncert(self, br: dict) -> float:
        """First-order radial ambiguity induced by the certified bridge k interval.

        This is dimensionless: |cos psi(k_T(lo))-cos psi(k_T(hi))|.  Refining the
        bridge only when this *effect* is material avoids both the coarse-k failure
        in very narrow regimes and the query cost of always forcing a 1e-3 bridge.
        """
        lo = float(br.get("lo", br["kb"]))
        hi = float(br.get("hi", br["kb"]))
        k0 = min(1.0, max(0.0, self._ttba_k(lo)))
        k1 = min(1.0, max(0.0, self._ttba_k(hi)))
        d0 = br["direction"](k0)
        d1 = br["direction"](k1)
        p0 = math.acos(max(-1.0, min(1.0, _dot(d0, br["v"]))))
        p1 = math.acos(max(-1.0, min(1.0, _dot(d1, br["v"]))))
        return abs(math.cos(p0) - math.cos(p1))

    def _pvr_aor_refine_narrow_bridge(self, br: dict) -> Tuple[dict, int, float]:
        """Refine a coarse PVR bridge only while its induced radial ambiguity matters."""
        c = self.cfg
        lo = float(br.get("lo", br["kb"]))
        hi = float(br.get("hi", min(1.0, lo + self.bridge_tol)))
        q0 = self.oracle.queries
        out = dict(br)
        out.update({"lo": lo, "hi": hi, "kb": lo})
        unc = self._pvr_aor_bridge_rad_uncert(out)
        while (unc > c.pvr_aor_bridge_radius_tol
               and hi - lo > c.ttba_bridge_tol
               and self.remaining > 0):
            mid = 0.5 * (lo + hi)
            x = self._proj(self.origin + br["r"] * br["direction"](mid))
            if self._adv(x, "update"):
                lo = mid
            else:
                hi = mid
            out.update({"lo": lo, "hi": hi, "kb": lo})
            unc = self._pvr_aor_bridge_rad_uncert(out)
        q = self.oracle.queries - q0
        out["q_bridge"] = int(br.get("q_bridge", 0)) + q
        if self.diag.get("k_bridge"):
            self.diag["k_bridge"][-1] = lo
        return out, q, unc

    def _pvr_aor_record_gain(self, r: float, cand: torch.Tensor) -> float:
        if cand is None or r <= 1e-12:
            return 0.0
        return max(0.0, (r - _norm(cand - self.origin)) / r)

    def _update_pvre_aor(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v4: adaptive-order regularized PVR-E.

        Three corrections are driven directly by the 30g results:
          1) low-curvature is detected by *resolvability* D/U, not |gamma-90|;
          2) the second-order radius is smoothly shrunk toward the hyperplane model;
          3) when curvature stays unresolved, a short CGBA planar burst is used.  The
             main method estimates CGBA's native normal at the START of those iterations,
             so it never pays for both TtBA and CGBA normals in one iteration.  The SN
             ablation keeps the shared TtBA normal to isolate this effect.

        In the narrow R1 regime the original PVR-E escape is preserved, but the bridge
        is refined only when the certified k interval causes a material radial ambiguity.
        This replaces 30g's ineffective stall/exact-rescue heuristic.
        """
        c = self.cfg
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        wdir = _unit(n - ct * v)
        if wdir is None:
            return xb
        theta = math.acos(ct)
        c_hat = min(0.999, self._c_hat())
        pool = self.hyb_pool
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2

        def pool_add(cos_alpha: float, weight: float) -> None:
            pool[0] = c.hyb_lambda * pool[0] + weight * r * cos_alpha
            pool[1] = c.hyb_lambda * pool[1] + weight

        def log_choice(choice: str, reanchor: int = 0, kb: float = float("nan"),
                       D: float = float("nan"), S: float = float("nan"),
                       wt: float = float("nan"), qref: int = 0,
                       kunc: float = float("nan")) -> None:
            bias, scale = self._pvr_aor_robust_stats()
            self.diag["pvr_aor_choice"].append(choice)
            self.diag["pvr_aor_reanchor"].append(int(reanchor))
            self.diag["pvr_aor_bias"].append(bias if math.isfinite(bias) else float("nan"))
            self.diag["pvr_aor_scale"].append(scale if math.isfinite(scale) else float("nan"))
            self.diag["pvr_aor_D"].append(D)
            self.diag["pvr_aor_S"].append(S)
            self.diag["pvr_aor_weight"].append(wt)
            self.diag["pvr_aor_anchor_kb"].append(kb)
            self.diag["pvr_aor_anchor_age"].append(self.pvr_aor_anchor_age)
            self.diag["pvr_aor_bridge_refine_q"].append(int(qref))
            self.diag["pvr_aor_bridge_rad_uncert"].append(kunc)

        # A planar burst is a first-order model-order decision.  In the main AOR method
        # ``_step`` already supplied a native CGBA normal; the SN ablation supplied the
        # ordinary TtBA normal.  Both use exactly the same update here.
        if self.pvr_aor_planar_left > 0:
            q0 = self.oracle.queries
            cand = self._update_cgba(n, xb)
            self.pvr_aor_planar_left -= 1
            choice = "planar_native" if self.spec.aor_native_planar else "planar_shared"
            log_choice(choice)
            gain = self._pvr_aor_record_gain(r, cand)
            if cand is not xb:
                self._log_geom(f"pvr_aor_{choice}", r, v, xb, cand, cos_theta=ct,
                               q_radial=self.oracle.queries - q0)
            self.pvr_aor_anchor_age += 1
            if gain <= 1e-10 or self.pvr_aor_planar_left <= 0:
                self.pvr_aor_planar_left = 0
                self.hyb_mode = "bridge"
                self.pvr_aor_force_model = True
            return cand

        def anchor_step(reason: str) -> torch.Tensor:
            br = self._bridge(n, xb)
            if br is None:
                self.hyb_mode = "bridge"
                log_choice("bridge_fail", 1)
                return xb
            kb = float(br["kb"])
            self.pvr_aor_anchor_age = 0

            if self._no_bridge_inside(kb):
                cand = self._ttba_radial_step(br, xb, "pvr_aor_fallback")
                log_choice("fallback", 1, kb)
                self.hyb_mode = "bridge"
                return cand

            # Narrow R1: preserve PVR-E's successful bracketed escape, but condition the
            # *bridge precision* by its induced L2-radius effect rather than k itself.
            if kb <= c.escape_k:
                br2, qref, kunc = self._pvr_aor_refine_narrow_bridge(br)
                cand = self._escape_step(br2, xb, "pvr_aor_escape")
                log_choice("escape", 1, float(br2["kb"]), qref=qref, kunc=kunc)
                self.hyb_mode = "bridge"
                return cand

            psi_bridge = math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
            ap = 0.5 * psi_bridge
            ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)

            x_bridge = self._proj(xs + r * br["direction"](kb))
            d_b = _unit(0.5 * (xb + x_bridge) - xs)
            if d_b is None:
                d_b = v
            psi_b = math.acos(max(-1.0, min(1.0, _dot(d_b, v))))
            pb = self._pvr_aor_pred_radius(r, psi_b)
            g_b, gamma_b, _, g2_b, D_b, S_b, w_b = pb

            kt = min(1.0, max(0.0, self._ttba_k(kb)))
            d_t = br["direction"](kt)
            psi_t = math.acos(max(-1.0, min(1.0, _dot(d_t, v))))
            pt = self._pvr_aor_pred_radius(r, psi_t)
            g_t, gamma_t, _, g2_t, D_t, S_t, w_t = pt

            choose_t = self.hfit.ready() and (g_t + c.pvr_aor_anchor_deadband * r < g_b)
            if choose_t:
                d_sel, psi_sel, g_pred, gamma_pred, g2_raw, D, S, wt = d_t, psi_t, g_t, gamma_t, g2_t, D_t, S_t, w_t
                choice = "anchor_t"
            else:
                d_sel, psi_sel, g_pred, gamma_pred, g2_raw, D, S, wt = d_b, psi_b, g_b, gamma_b, g2_b, D_b, S_b, w_b
                choice = "anchor_b"
            q0 = self.oracle.queries
            out = self._bracket_radial(d_sel, g_pred, r, d0=None)
            if out is None:
                self.hyb_mode = "bridge"
                log_choice("anchor_fail", 1, kb, D=D, S=S, wt=wt)
                return xb
            cand, g_act = out
            was_fit = math.isfinite(gamma_pred)
            if was_fit and math.isfinite(g2_raw):
                self.pvr_aor_radius_resid.append((g_act - g2_raw) / r)
                self._pvr_aor_record_resolution(S)
            if psi_sel > 1e-9:
                self.hfit.update(r * math.sin(psi_sel), sag_slope(r, psi_sel, g_act))
            self._log_geom(f"pvr_aor_{choice}", r, v, xb, cand, g_pred=g_pred, br=br,
                           gamma_pred=gamma_pred, cos_theta=ct, q_radial=self.oracle.queries - q0)
            log_choice(choice, 1, kb, D=D, S=S, wt=wt)
            self.pvr_aor_force_model = True
            self.hyb_mode = "model"
            return cand

        ca_model = (pool[0] / pool[1]) / r if pool[1] > 0 else None
        if (self.hyb_mode == "bridge" or ca_model is None
                or self.pvr_aor_anchor_age >= max(1, c.pvr_aor_reanchor_every)):
            return anchor_step("periodic")

        ca = min(max(ca_model, 1e-3), 0.999)
        psi = min(max(model_psi(c_hat, ct, ca), 0.0), 0.95 * theta)
        d = _unit(math.cos(psi) * v + math.sin(psi) * wdir)
        if d is None:
            self.hyb_mode = "bridge"
            return anchor_step("degenerate")
        g_pred, gamma_pred, g0, g2_raw, D, S, wt = self._pvr_aor_pred_radius(r, psi)
        bias, scale = self._pvr_aor_robust_stats()
        warm = self.pvr_aor_force_model or len(self.pvr_aor_radius_resid) < max(1, c.pvr_aor_min_model_errs)
        if not warm and math.isfinite(scale):
            conservative_gain = r - g_pred - c.pvr_aor_mad_mult * scale * r
            if conservative_gain <= 0.0:
                self.hyb_mode = "bridge"
                return anchor_step("uncertain")

        q0 = self.oracle.queries
        d0 = None
        if math.isfinite(scale):
            d0 = min(max(c.pvr_bracket_mult * max(scale, c.pvr_aor_mad_floor),
                         2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        self.pvr_aor_force_model = False
        if out is None:
            self.hyb_mode = "bridge"
            log_choice("model_fail", 0, D=D, S=S, wt=wt)
            return xb
        cand, g_act = out
        if math.isfinite(gamma_pred) and math.isfinite(g2_raw):
            self.pvr_aor_radius_resid.append((g_act - g2_raw) / r)
            self._pvr_aor_record_resolution(S)
        ratio = (r - g_act) / max(r - g_pred, 1e-12)
        self.hyb_mode = "model" if (c.hyb_ratio_lo < ratio < c.hyb_ratio_hi) else "bridge"

        kappa = 0.0
        if self.hfit.ready():
            s_lat = r * math.sin(psi)
            kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
        psi_s = secant_foot(r, g_act, psi, kappa, theta)
        if 1e-6 < psi_s < theta:
            ca_cal = cos_alpha_from_bridge(c_hat, ct, psi_s)
            if 0.0 < ca_cal < 1.0:
                pool_add(ca_cal, c.pvr_calib_weight * unit_w)
        if psi > 1e-9:
            self.hfit.update(r * math.sin(psi), sag_slope(r, psi, g_act))

        self._log_geom("pvr_aor_model", r, v, xb, cand, g_pred=g_pred, gamma_pred=gamma_pred,
                       ratio=ratio, cos_theta=ct, q_radial=self.oracle.queries - q0)
        self.pvr_aor_anchor_age += 1
        log_choice("model", 0, D=D, S=S, wt=wt)
        return cand


    # ---------------------------- PVR-E v5: adaptive-order utility controller
    def _pvr_aou_robust_stats(self) -> Tuple[float, float]:
        """Signed median + MAD of MODEL-ray second-order radius residuals."""
        c = self.cfg
        if not self.pvr_aou_radius_resid:
            return 0.0, float("inf")
        a = np.asarray(self.pvr_aou_radius_resid, dtype=float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        return med, max(c.pvr_aou_mad_floor, 1.4826 * mad)

    def _pvr_aou_pred_radius(self, r: float, psi: float) -> Tuple[float, float, float, float, float, float, float]:
        """AOR geometry used ONLY as a calibrated radius predictor / context prior.

        Returns (g_reg, gamma, g0, g2_raw, D, S, w), with
          D=|g2-g0|/r, S=D/U and w=S^2/(1+S^2).
        Unlike build 30h, w does not directly decide the optimizer; observed
        query-normalized utility controls persistent M/T/C action selection.
        """
        g0 = r * math.cos(psi)
        gamma = float("nan")
        g2 = g0
        if self.hfit.ready():
            gamma = self.hfit.predict_gamma(r * math.sin(psi))
            g2 = inscribed_radius(r, psi, gamma)
        bias, scale = self._pvr_aou_robust_stats()
        D = abs(g2 - g0) / max(r, 1e-12)
        g2_cal = g2 + (bias * r if math.isfinite(bias) else 0.0)
        S = D / max(scale, self.cfg.pvr_aou_mad_floor, 1e-12) if math.isfinite(scale) else 0.0
        w = (S * S) / (1.0 + S * S)
        g = g0 + w * (g2_cal - g0)
        g = min(max(g, 0.05 * r), r * (1.0 - 1e-6))
        return g, gamma, g0, g2, D, S, w

    def _pvr_aou_bridge_rad_uncert(self, br: dict) -> float:
        """Dimensionless radial ambiguity induced by the certified bridge-k interval."""
        lo = float(br.get("lo", br["kb"]))
        hi = float(br.get("hi", br["kb"]))
        k0 = min(1.0, max(0.0, self._ttba_k(lo)))
        k1 = min(1.0, max(0.0, self._ttba_k(hi)))
        d0 = br["direction"](k0); d1 = br["direction"](k1)
        p0 = math.acos(max(-1.0, min(1.0, _dot(d0, br["v"]))))
        p1 = math.acos(max(-1.0, min(1.0, _dot(d1, br["v"]))))
        return abs(math.cos(p0) - math.cos(p1))

    def _pvr_aou_refine_narrow_bridge(self, br: dict) -> Tuple[dict, int, float]:
        """Effect-controlled R1 bridge refinement, never finer than native TtBA tol."""
        c = self.cfg
        lo = float(br.get("lo", br["kb"]))
        hi = float(br.get("hi", min(1.0, lo + self.bridge_tol)))
        q0 = self.oracle.queries
        out = dict(br); out.update({"lo": lo, "hi": hi, "kb": lo})
        unc = self._pvr_aou_bridge_rad_uncert(out)
        while (unc > c.pvr_aou_bridge_radius_tol and hi - lo > c.ttba_bridge_tol and self.remaining > 0):
            mid = 0.5 * (lo + hi)
            x = self._proj(self.origin + br["r"] * br["direction"](mid))
            if self._adv(x, "update"):
                lo = mid
            else:
                hi = mid
            out.update({"lo": lo, "hi": hi, "kb": lo})
            unc = self._pvr_aou_bridge_rad_uncert(out)
        q = self.oracle.queries - q0
        out["q_bridge"] = int(br.get("q_bridge", 0)) + q
        if self.diag.get("k_bridge"):
            self.diag["k_bridge"][-1] = lo
        return out, q, unc

    def _pvr_aou_eta(self, r: float, cand: torch.Tensor, q_action: int) -> float:
        """Scale-free observed utility: log-radius progress per TOTAL iteration query.

        Normal probes are included because the research objective is total victim-query
        efficiency.  They are common across AOU actions, so this also preserves a fair
        comparison of the action-specific bridge/radial overhead.
        """
        if cand is None or r <= 1e-12:
            return 0.0
        rn = _norm(cand - self.origin)
        if not math.isfinite(rn) or rn >= r * (1.0 - 1e-12):
            return 0.0
        qtot = max(1, int(self._q_normal) + int(max(0, q_action)))
        return max(0.0, math.log(max(r, 1e-12) / max(rn, 1e-12))) / qtot

    def _pvr_aou_update_utility(self, action: str, eta: float) -> None:
        if action not in self.pvr_aou_mu:
            return
        b = min(max(float(self.cfg.pvr_aou_utility_beta), 0.0), 0.999999)
        n = self.pvr_aou_n[action]
        self.pvr_aou_mu[action] = float(eta) if n == 0 else b * self.pvr_aou_mu[action] + (1.0 - b) * float(eta)
        self.pvr_aou_n[action] = n + 1
        self.pvr_aou_last_used[action] = int(self._it)

    def _pvr_aou_better(self, challenger: str, incumbent: str, margin: float) -> bool:
        if self.pvr_aou_n.get(challenger, 0) <= 0:
            return False
        if self.pvr_aou_n.get(incumbent, 0) <= 0:
            return True
        return self.pvr_aou_mu[challenger] > (1.0 + max(0.0, float(margin))) * self.pvr_aou_mu[incumbent] + 1e-15

    def _pvr_aou_choose_wide_trial(self) -> Optional[str]:
        """Geometry orders exploration; it never directly chooses the final optimizer."""
        c = self.cfg
        if self.pvr_aou_mode_age < max(0, c.pvr_aou_min_dwell):
            return None
        if self._it - self.pvr_aou_last_trial_it < max(1, c.pvr_aou_trial_interval):
            return None
        cur = self.pvr_aou_mode
        if self.pvr_aou_last_weight <= c.pvr_aou_geom_w_split:
            pref = ["C", "T", "M"]
        else:
            pref = ["M", "T", "C"]
        cand = [a for a in pref if a != cur]
        # Prefer the least recently observed challenger; geometry breaks ties.
        cand.sort(key=lambda a: (self.pvr_aou_last_used.get(a, -10**9), pref.index(a)))
        return cand[0] if cand else None

    def _pvr_aou_finish_wide(self, action: str, cand: torch.Tensor, r: float, q0: int,
                             trial_from: Optional[str], choice_name: str,
                             S: float = float("nan"), wt: float = float("nan"),
                             kb: float = float("nan"), recheck: int = 0,
                             qref: int = 0, kunc: float = float("nan")) -> torch.Tensor:
        q_action = self.oracle.queries - q0
        eta = self._pvr_aou_eta(r, cand, q_action)
        self._pvr_aou_update_utility(action, eta)
        switched = 0
        if trial_from is not None:
            self.pvr_aou_last_trial_it = int(self._it)
            if self._pvr_aou_better(action, trial_from, self.cfg.pvr_aou_switch_margin):
                self.pvr_aou_mode = action; self.pvr_aou_mode_age = 0; switched = 1
            else:
                self.pvr_aou_mode = trial_from; self.pvr_aou_mode_age = max(1, self.pvr_aou_mode_age)
        else:
            self.pvr_aou_mode = action
            self.pvr_aou_mode_age += 1
        self._pvr_aou_log(choice_name, action, int(trial_from is not None), switched, recheck,
                          eta, q_action, S=S, wt=wt, kb=kb, qref=qref, kunc=kunc)
        return cand

    def _pvr_aou_log(self, choice: str, mode: str, trial: int, switched: int, recheck: int,
                     eta: float, q_action: int, S: float = float("nan"),
                     wt: float = float("nan"), kb: float = float("nan"),
                     qref: int = 0, kunc: float = float("nan")) -> None:
        bias, scale = self._pvr_aou_robust_stats()
        d = self.diag
        d["pvr_aou_choice"].append(choice); d["pvr_aou_mode"].append(mode)
        d["pvr_aou_trial"].append(int(trial)); d["pvr_aou_switch"].append(int(switched))
        d["pvr_aou_recheck"].append(int(recheck)); d["pvr_aou_eta"].append(float(eta))
        d["pvr_aou_q_action"].append(int(q_action))
        for a in ("M", "T", "C", "E", "R"):
            d[f"pvr_aou_mu_{a}"].append(self.pvr_aou_mu[a] if self.pvr_aou_n[a] else float("nan"))
        d["pvr_aou_bias"].append(bias if math.isfinite(bias) else float("nan"))
        d["pvr_aou_scale"].append(scale if math.isfinite(scale) else float("nan"))
        d["pvr_aou_D"].append(float("nan"))
        d["pvr_aou_S"].append(S); d["pvr_aou_weight"].append(wt)
        d["pvr_aou_anchor_kb"].append(kb); d["pvr_aou_anchor_age"].append(self.pvr_aou_anchor_age)
        d["pvr_aou_bridge_refine_q"].append(int(qref)); d["pvr_aou_bridge_rad_uncert"].append(kunc)

    def _pvr_aou_fit_from_boundary(self, r: float, v: torch.Tensor, cand: torch.Tensor) -> None:
        """Use any verified improving boundary point to keep the 1-D curvature fit current."""
        if cand is None or cand is self.origin:
            return
        g = _norm(cand - self.origin)
        if not math.isfinite(g) or g <= 0.0 or g >= r:
            return
        d = _unit(cand - self.origin)
        if d is None:
            return
        psi = math.acos(max(-1.0, min(1.0, _dot(d, v))))
        if psi > 1e-9:
            self.hfit.update(r * math.sin(psi), sag_slope(r, psi, g))

    def _pvr_aou_model_step(self, n: torch.Tensor, xb: torch.Tensor, r: float, v: torch.Tensor,
                             wdir: torch.Tensor, theta: float, c_hat: float,
                             unit_w: float) -> Tuple[torch.Tensor, float, float]:
        """One regularized PVR model step; returns (candidate,S,w)."""
        c = self.cfg; pool = self.hyb_pool
        if pool[1] <= 0:
            return xb, float("nan"), float("nan")
        ca = min(max((pool[0] / pool[1]) / r, 1e-3), 0.999)
        psi = min(max(model_psi(c_hat, max(-1+1e-7,min(1-1e-7,_dot(n,v))), ca), 0.0), 0.95 * theta)
        d = _unit(math.cos(psi) * v + math.sin(psi) * wdir)
        if d is None:
            return xb, float("nan"), float("nan")
        g_pred, gamma_pred, g0, g2_raw, D, S, wt = self._pvr_aou_pred_radius(r, psi)
        bias, scale = self._pvr_aou_robust_stats()
        d0 = None
        if math.isfinite(scale):
            d0 = min(max(c.pvr_bracket_mult * max(scale, c.pvr_aou_mad_floor),
                         2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        qrad0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        if out is None:
            return xb, S, wt
        cand, g_act = out
        if math.isfinite(gamma_pred) and math.isfinite(g2_raw):
            self.pvr_aou_radius_resid.append((g_act - g2_raw) / r)
        # secant calibration of the pooled normal-foot distance
        kappa = 0.0
        if self.hfit.ready():
            s_lat = r * math.sin(psi)
            kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
        psi_s = secant_foot(r, g_act, psi, kappa, theta)
        if 1e-6 < psi_s < theta:
            ca_cal = cos_alpha_from_bridge(c_hat, max(-1+1e-7,min(1-1e-7,_dot(n,v))), psi_s)
            if 0.0 < ca_cal < 1.0:
                pool[0] = c.hyb_lambda * pool[0] + c.pvr_calib_weight * unit_w * r * ca_cal
                pool[1] = c.hyb_lambda * pool[1] + c.pvr_calib_weight * unit_w
        if psi > 1e-9:
            self.hfit.update(r * math.sin(psi), sag_slope(r, psi, g_act))
        self._log_geom("pvr_aou_model", r, v, xb, cand, g_pred=g_pred, gamma_pred=gamma_pred,
                       cos_theta=max(-1+1e-7,min(1-1e-7,_dot(n,v))),
                       q_radial=self.oracle.queries - qrad0)
        self.pvr_aou_last_S = S; self.pvr_aou_last_weight = wt
        return cand, S, wt

    def _pvr_aou_choose_narrow(self) -> Tuple[str, Optional[str]]:
        """Return (action, trial_from) for the local E-vs-exact-TtBA controller."""
        c = self.cfg; cur = self.pvr_aou_narrow_action
        if self.pvr_aou_narrow_steps < max(0, c.pvr_aou_narrow_probe_after):
            return cur, None
        if self._it - self.pvr_aou_last_narrow_trial_it < max(1, c.pvr_aou_narrow_trial_interval):
            return cur, None
        challenger = "R" if cur == "E" else "E"
        return challenger, cur

    def _update_pvre_aou(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v5: adaptive-order, utility-controlled, path-persistent attack.

        Key design: AOR's curvature resolvability is CONTEXT, not an optimizer selector.
        The persistent wide action M/T/C is changed only by a controlled one-action trial
        whose observed reward is log(r_old/r_new) per total iteration victim query.  In
        the narrow k_b<=0.2 regime, the same principle compares original PVR-E escape E
        against an exact-TtBA counter-action R.  No multi-branch victim-query race occurs.
        """
        c = self.cfg; xs = self.origin
        r_vec = xb - xs; r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        wdir = _unit(n - ct * v)
        if wdir is None:
            return xb
        theta = math.acos(ct); c_hat = min(0.999, self._c_hat())
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2
        pool = self.hyb_pool

        def pool_add(cos_alpha: float, weight: float) -> None:
            pool[0] = c.hyb_lambda * pool[0] + weight * r * cos_alpha
            pool[1] = c.hyb_lambda * pool[1] + weight

        # Geometry is intentionally only an exploration prior.  Do not trial before
        # at least a small amount of model-residual evidence exists.
        trial = None
        if len(self.pvr_aou_radius_resid) >= max(1, c.pvr_aou_min_model_errs):
            trial = self._pvr_aou_choose_wide_trial()
        action = trial if trial is not None else self.pvr_aou_mode
        trial_from = self.pvr_aou_mode if trial is not None else None

        # A fresh bridge is needed initially, in a known narrow regime, for every T
        # action, or periodically to prevent a persistent C/M path from becoming stale.
        need_bridge = (pool[1] <= 0 or self.pvr_aou_in_narrow or action == "T" or
                       self.pvr_aou_anchor_age >= max(1, c.pvr_aou_recheck_every))
        br = None; qref = 0; kunc = float("nan"); kb = float("nan"); recheck = int(need_bridge)
        q0 = self.oracle.queries
        if need_bridge:
            br = self._bridge(n, xb)
            self.pvr_aou_anchor_age = 0
            if br is None:
                self._pvr_aou_log("bridge_fail", self.pvr_aou_mode, int(trial is not None), 0, 1,
                                  0.0, self.oracle.queries-q0)
                return xb
            kb = float(br["kb"])
            if self._no_bridge_inside(kb):
                cand = self._ttba_radial_step(br, xb, "pvr_aou_fallback")
                self.pvr_aou_in_narrow = False
                eta = self._pvr_aou_eta(r, cand, self.oracle.queries-q0)
                self._pvr_aou_update_utility("T", eta)
                self._pvr_aou_log("fallback", "T", 0, 0, 1, eta, self.oracle.queries-q0, kb=kb)
                return cand

            # Narrow local region: learn whether original PVR-E escape E or exact TtBA R
            # is productive for THIS trajectory.  Small k_b alone never forces R.
            if kb <= c.escape_k:
                self.pvr_aou_in_narrow = True; self.pvr_aou_narrow_steps += 1
                br2, qref, kunc = self._pvr_aou_refine_narrow_bridge(br)
                na, ntrial_from = self._pvr_aou_choose_narrow()
                if na == "R":
                    # A counter-action advertised as exact TtBA uses the actual TtBA bridge tolerance.
                    brx = self._refine_bridge(br2, c.ttba_bridge_tol)
                    cand = self._ttba_radial_step(brx, xb, "pvr_aou_rescue")
                    choice = "rescue"
                else:
                    cand = self._escape_step(br2, xb, "pvr_aou_escape")
                    choice = "escape"
                eta = self._pvr_aou_eta(r, cand, self.oracle.queries-q0)
                self._pvr_aou_update_utility(na, eta)
                switched = 0
                if ntrial_from is not None:
                    self.pvr_aou_last_narrow_trial_it = int(self._it)
                    if self._pvr_aou_better(na, ntrial_from, c.pvr_aou_narrow_switch_margin):
                        self.pvr_aou_narrow_action = na; switched = 1
                    else:
                        self.pvr_aou_narrow_action = ntrial_from
                self._pvr_aou_fit_from_boundary(r, v, cand)
                self._pvr_aou_log(choice, na, int(ntrial_from is not None), switched, 1, eta,
                                  self.oracle.queries-q0, kb=kb, qref=qref, kunc=kunc)
                return cand

            # Exiting a narrow patch: old E/R evidence is local and should not dominate
            # a later unrelated narrow patch.
            if self.pvr_aou_in_narrow:
                self.pvr_aou_in_narrow = False; self.pvr_aou_narrow_steps = 0
                self.pvr_aou_narrow_action = "E"; self.pvr_aou_last_narrow_trial_it = -10**9
                for aa in ("E", "R"):
                    self.pvr_aou_mu[aa] = 0.0; self.pvr_aou_n[aa] = 0; self.pvr_aou_last_used[aa] = -10**9

            # Fresh bridge calibrates the model-foot pool, but does NOT force a bisector
            # update.  This separates measurement from optimization action.
            psi_bridge = math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
            ap = 0.5 * psi_bridge
            ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)
            if pool[1] > 0:
                ca_tmp = min(max((pool[0] / pool[1]) / r, 1e-3), 0.999)
                psi_tmp = min(max(model_psi(c_hat, ct, ca_tmp), 0.0), 0.95 * theta)
                _, _, _, _, _, St, wt = self._pvr_aou_pred_radius(r, psi_tmp)
                self.pvr_aou_last_S = St; self.pvr_aou_last_weight = wt

        # No useful trial is allowed during cold start: first acquire M residual evidence.
        if len(self.pvr_aou_radius_resid) < max(1, c.pvr_aou_min_model_errs):
            action = "M"; trial_from = None

        # If M lacks an anchor/pool, get one now (can happen after a failed bridge state).
        if action == "M" and pool[1] <= 0:
            if br is None:
                br = self._bridge(n, xb); recheck = 1; self.pvr_aou_anchor_age = 0
                if br is None:
                    self._pvr_aou_log("bridge_fail", "M", 0, 0, 1, 0.0, self.oracle.queries-q0)
                    return xb
                kb = float(br["kb"])
            psi_bridge = math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
            ca_meas = cos_alpha_from_bridge(c_hat, ct, 0.5 * psi_bridge)
            if 0.0 < ca_meas < 1.0:
                pool_add(ca_meas, c.hyb_bridge_weight * unit_w)

        if action == "C":
            cand = self._update_cgba(n, xb)
            self._pvr_aou_fit_from_boundary(r, v, cand)
            self.pvr_aou_anchor_age += 1
            return self._pvr_aou_finish_wide("C", cand, r, q0, trial_from, "cgba",
                                              S=self.pvr_aou_last_S, wt=self.pvr_aou_last_weight,
                                              kb=kb, recheck=recheck)

        if action == "T":
            if br is None:
                br = self._bridge(n, xb); recheck = 1; self.pvr_aou_anchor_age = 0
                if br is None:
                    self._pvr_aou_log("bridge_fail", "T", int(trial_from is not None), 0, 1,
                                      0.0, self.oracle.queries-q0)
                    return xb
                kb = float(br["kb"])
            cand = self._ttba_radial_step(br, xb, "pvr_aou_ttba")
            self._pvr_aou_fit_from_boundary(r, v, cand)
            self.pvr_aou_anchor_age += 1
            return self._pvr_aou_finish_wide("T", cand, r, q0, trial_from, "ttba",
                                              S=self.pvr_aou_last_S, wt=self.pvr_aou_last_weight,
                                              kb=kb, recheck=recheck)

        # M: regularized PVR direction/radius.  A failed M step receives eta=0 and the
        # controller can replace it only through a later controlled challenger trial.
        cand, S, wt = self._pvr_aou_model_step(n, xb, r, v, wdir, theta, c_hat, unit_w)
        self.pvr_aou_anchor_age += 1
        choice = "model" if cand is not xb else "model_fail"
        return self._pvr_aou_finish_wide("M", cand, r, q0, trial_from, choice,
                                          S=S, wt=wt, kb=kb, recheck=recheck)


    # ---------------- PVR-E v6: context-conditioned confidence-aware utility controller
    def _pvr_caou_robust_stats(self) -> Tuple[float, float]:
        c = self.cfg
        if not self.pvr_caou_radius_resid:
            return 0.0, float("inf")
        a = np.asarray(self.pvr_caou_radius_resid, dtype=float)
        med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
        return med, max(c.pvr_caou_mad_floor, 1.4826 * mad)

    def _pvr_caou_pred_radius(self, r: float, psi: float) -> Tuple[float, float, float, float, float, float, float]:
        g0 = r * math.cos(psi); gamma = float("nan"); g2 = g0
        if self.hfit.ready():
            gamma = self.hfit.predict_gamma(r * math.sin(psi)); g2 = inscribed_radius(r, psi, gamma)
        bias, scale = self._pvr_caou_robust_stats()
        D = abs(g2 - g0) / max(r, 1e-12)
        g2_cal = g2 + (bias * r if math.isfinite(bias) else 0.0)
        S = D / max(scale, self.cfg.pvr_caou_mad_floor, 1e-12) if math.isfinite(scale) else 0.0
        w = (S * S) / (1.0 + S * S)
        g = g0 + w * (g2_cal - g0)
        g = min(max(g, 0.05 * r), r * (1.0 - 1e-6))
        return g, gamma, g0, g2, D, S, w

    def _pvr_caou_bridge_rad_uncert(self, br: dict) -> float:
        lo = float(br.get("lo", br["kb"])); hi = float(br.get("hi", br["kb"]))
        k0 = min(1.0, max(0.0, self._ttba_k(lo))); k1 = min(1.0, max(0.0, self._ttba_k(hi)))
        d0 = br["direction"](k0); d1 = br["direction"](k1)
        p0 = math.acos(max(-1.0, min(1.0, _dot(d0, br["v"])))); p1 = math.acos(max(-1.0, min(1.0, _dot(d1, br["v"]))))
        return abs(math.cos(p0) - math.cos(p1))

    def _pvr_caou_refine_narrow_bridge(self, br: dict) -> Tuple[dict, int, float]:
        c = self.cfg; lo = float(br.get("lo", br["kb"])); hi = float(br.get("hi", min(1.0, lo + self.bridge_tol)))
        q0 = self.oracle.queries; out = dict(br); out.update({"lo": lo, "hi": hi, "kb": lo})
        unc = self._pvr_caou_bridge_rad_uncert(out)
        while unc > c.pvr_caou_bridge_radius_tol and hi - lo > c.ttba_bridge_tol and self.remaining > 0:
            mid = 0.5 * (lo + hi); x = self._proj(self.origin + br["r"] * br["direction"](mid))
            if self._adv(x, "update"): lo = mid
            else: hi = mid
            out.update({"lo": lo, "hi": hi, "kb": lo}); unc = self._pvr_caou_bridge_rad_uncert(out)
        q = self.oracle.queries - q0; out["q_bridge"] = int(br.get("q_bridge", 0)) + q
        if self.diag.get("k_bridge"): self.diag["k_bridge"][-1] = lo
        return out, q, unc

    def _pvr_caou_eta(self, r: float, cand: torch.Tensor, q_action: int) -> float:
        if cand is None or r <= 1e-12: return 0.0
        rn = _norm(cand - self.origin)
        if not math.isfinite(rn) or rn >= r * (1.0 - 1e-12): return 0.0
        qtot = max(1, int(self._q_normal) + int(max(0, q_action)))
        return max(0.0, math.log(max(r,1e-12)/max(rn,1e-12))) / qtot

    def _pvr_caou_progress(self, r: float) -> float:
        if self.pvr_caou_r0 is None or not math.isfinite(self.pvr_caou_r0) or self.pvr_caou_r0 <= 0:
            self.pvr_caou_r0 = max(float(r), 1e-12)
        return min(1.0, max(0.0, math.log(max(self.pvr_caou_r0,1e-12)/max(r,1e-12)) / 4.0))

    def _pvr_caou_wide_ctx(self, r: float) -> Tuple[float, float, float]:
        w = self.pvr_caou_last_weight if math.isfinite(self.pvr_caou_last_weight) else 0.5
        qf = min(1.0, max(0.0, self.oracle.queries / max(1.0, float(self.cfg.max_queries))))
        return (min(1.0,max(0.0,w)), qf, self._pvr_caou_progress(r))

    def _pvr_caou_narrow_ctx(self, r: float, kb: float) -> Tuple[float, float, float]:
        kr = min(1.0, max(0.0, float(kb) / max(self.cfg.escape_k, 1e-12)))
        qf = min(1.0, max(0.0, self.oracle.queries / max(1.0, float(self.cfg.max_queries))))
        return (kr, qf, self._pvr_caou_progress(r))

    def _pvr_caou_add_obs(self, action: str, ctx: Tuple[float, ...], eta: float) -> None:
        if action not in self.pvr_caou_hist or not math.isfinite(eta): return
        self.pvr_caou_hist[action].append((tuple(float(x) for x in ctx), float(max(0.0,eta)), int(self._it)))
        self.pvr_caou_last_used[action] = int(self._it)

    def _pvr_caou_stats(self, action: str, ctx: Tuple[float, ...], narrow: bool=False) -> Tuple[float,float,float,float,float]:
        """Kernel/local, time-decayed mean, standard error, n_eff, LCB, UCB."""
        c = self.cfg; hist = list(self.pvr_caou_hist.get(action, ()))
        if not hist: return (float("nan"), float("inf"), 0.0, -float("inf"), float("inf"))
        bws = (c.pvr_caou_bw_k, c.pvr_caou_bw_q, c.pvr_caou_bw_r) if narrow else (c.pvr_caou_bw_w, c.pvr_caou_bw_q, c.pvr_caou_bw_r)
        ws=[]; ys=[]
        for z,y,it0 in hist:
            if len(z)!=len(ctx): continue
            d2=0.0
            for j,(x0,x1) in enumerate(zip(ctx,z)):
                bw=max(1e-6,float(bws[min(j,len(bws)-1)])); d2 += ((float(x0)-float(x1))/bw)**2
            age=max(0,int(self._it)-int(it0)); wt=math.exp(-0.5*d2)*(max(1e-6,min(1.0,c.pvr_caou_time_decay))**age)
            if wt>1e-10: ws.append(wt); ys.append(float(y))
        if not ws: return (float("nan"), float("inf"), 0.0, -float("inf"), float("inf"))
        sw=sum(ws); mu=sum(w*y for w,y in zip(ws,ys))/sw
        var=sum(w*(y-mu)**2 for w,y in zip(ws,ys))/sw
        sw2=sum(w*w for w in ws); neff=(sw*sw)/max(sw2,1e-12)
        se=max(float(c.pvr_caou_stderr_floor), math.sqrt(max(0.0,var)/max(neff,1.0)))
        cc=max(0.0,float(c.pvr_caou_conf_mult)); return mu,se,neff,mu-cc*se,mu+cc*se

    def _pvr_caou_geom_bonus(self, action: str, w: float, ref: float) -> float:
        if action=="M": p=w
        elif action=="C": p=1.0-w
        else: p=0.5
        return max(0.0,float(self.cfg.pvr_caou_geom_prior_frac))*max(ref,self.cfg.pvr_caou_utility_floor)*p

    def _pvr_caou_trial_candidate(self, r: float) -> Optional[str]:
        c=self.cfg; cur=self.pvr_caou_mode
        if self.pvr_caou_trial_state is not None: return self.pvr_caou_trial_state["action"]
        if self.pvr_caou_mode_age < max(0,c.pvr_caou_min_dwell): return None
        if self._it-self.pvr_caou_last_trial_it < max(1,c.pvr_caou_trial_interval): return None
        ctx=self._pvr_caou_wide_ctx(r); inc=self._pvr_caou_stats(cur,ctx,False)
        mu_i,_,_,lcb_i,ucb_i=inc
        ref=max(c.pvr_caou_utility_floor, mu_i if math.isfinite(mu_i) else c.pvr_caou_utility_floor)
        cost=max(0.0,c.pvr_caou_switch_cost_frac)*ref
        pref=["C","T","M"] if self.pvr_caou_last_weight<=0.5 else ["M","T","C"]
        best=None; bestscore=-float("inf")
        for a in pref:
            if a==cur: continue
            st=self._pvr_caou_stats(a,ctx,False); mu,se,neff,lcb,ucb=st
            stale=(self._it-self.pvr_caou_last_used.get(a,-10**9)>=max(1,c.pvr_caou_stale_after))
            bonus=self._pvr_caou_geom_bonus(a,self.pvr_caou_last_weight,ref)
            if neff<=0 or stale:
                score=ref+bonus+c.pvr_caou_conf_mult*c.pvr_caou_stderr_floor
                eligible=True
            else:
                score=ucb+bonus; eligible=(not math.isfinite(lcb_i)) or score>lcb_i+cost
            if eligible and score>bestscore: best,bestscore=a,score
        return best

    def _pvr_caou_should_switch(self, challenger: str, incumbent: str, ctx: Tuple[float,...], block_eta: float, narrow: bool=False) -> Tuple[bool,Tuple[float,float,float,float]]:
        c=self.cfg; ch=self._pvr_caou_stats(challenger,ctx,narrow); inc=self._pvr_caou_stats(incumbent,ctx,narrow)
        mu_c,se_c,ne_c,lcb_c,ucb_c=ch; mu_i,se_i,ne_i,lcb_i,ucb_i=inc
        ref=max(c.pvr_caou_utility_floor, mu_i if math.isfinite(mu_i) else c.pvr_caou_utility_floor)
        frac=c.pvr_caou_narrow_switch_cost_frac if narrow else c.pvr_caou_switch_cost_frac
        cost=max(0.0,frac)*ref
        confident=(ne_c>=c.pvr_caou_min_neff and ne_i>=c.pvr_caou_min_neff and lcb_c>ucb_i+cost)
        strong_margin=c.pvr_caou_narrow_sparse_strong_margin if narrow else c.pvr_caou_sparse_strong_margin
        strong=(math.isfinite(block_eta) and block_eta>(1.0+max(0.0,strong_margin))*max(ref, ucb_i if math.isfinite(ucb_i) else ref))
        return bool(confident or strong),(lcb_i,ucb_i,lcb_c,ucb_c)

    def _pvr_caou_start_wide_trial(self, action: str, r: float) -> None:
        self.pvr_caou_trial_state={"action":action,"from":self.pvr_caou_mode,"left":max(1,self.cfg.pvr_caou_trial_block),
                                   "r0":float(r),"q":0,"ctx":self._pvr_caou_wide_ctx(r),"start_it":int(self._it)}

    def _pvr_caou_start_narrow_trial(self, action: str, r: float, kb: float) -> None:
        self.pvr_caou_narrow_trial_state={"action":action,"from":self.pvr_caou_narrow_action,
                                          "left":max(1,self.cfg.pvr_caou_narrow_trial_block),"r0":float(r),"q":0,
                                          "ctx":self._pvr_caou_narrow_ctx(r,kb),"start_it":int(self._it)}

    def _pvr_caou_log(self, choice: str, mode: str, trial: int, trial_start: int, switched: int, recheck: int,
                      block_end: int, eta: float, block_eta: float, q_action: int, ctx: Tuple[float,...],
                      bounds=(float("nan"),)*4, neffs=(float("nan"),float("nan")),
                      S=float("nan"), wt=float("nan"), kb=float("nan"), qref=0, kunc=float("nan")) -> None:
        bias,scale=self._pvr_caou_robust_stats(); d=self.diag
        d["pvr_caou_choice"].append(choice); d["pvr_caou_mode"].append(mode); d["pvr_caou_trial"].append(int(trial))
        d["pvr_caou_trial_start"].append(int(trial_start)); d["pvr_caou_switch"].append(int(switched)); d["pvr_caou_recheck"].append(int(recheck))
        d["pvr_caou_block_end"].append(int(block_end)); d["pvr_caou_eta"].append(float(eta)); d["pvr_caou_block_eta"].append(float(block_eta))
        d["pvr_caou_q_action"].append(int(q_action)); d["pvr_caou_ctx_w"].append(float(ctx[0]) if ctx else float("nan"))
        d["pvr_caou_ctx_q"].append(float(ctx[1]) if len(ctx)>1 else float("nan")); d["pvr_caou_ctx_r"].append(float(ctx[2]) if len(ctx)>2 else float("nan"))
        li,ui,lc,uc=bounds; ni,nc=neffs
        d["pvr_caou_lcb_inc"].append(li); d["pvr_caou_ucb_inc"].append(ui); d["pvr_caou_lcb_ch"].append(lc); d["pvr_caou_ucb_ch"].append(uc)
        d["pvr_caou_neff_inc"].append(ni); d["pvr_caou_neff_ch"].append(nc)
        d["pvr_caou_bias"].append(bias if math.isfinite(bias) else float("nan")); d["pvr_caou_scale"].append(scale if math.isfinite(scale) else float("nan"))
        d["pvr_caou_S"].append(S); d["pvr_caou_weight"].append(wt); d["pvr_caou_anchor_kb"].append(kb); d["pvr_caou_anchor_age"].append(self.pvr_caou_anchor_age)
        d["pvr_caou_bridge_refine_q"].append(int(qref)); d["pvr_caou_bridge_rad_uncert"].append(kunc)

    def _pvr_caou_finish(self, action: str, cand: torch.Tensor, r: float, q0: int, choice: str,
                         ctx: Tuple[float,...], narrow: bool=False, recheck: int=0,
                         S=float("nan"), wt=float("nan"), kb=float("nan"), qref=0, kunc=float("nan")) -> torch.Tensor:
        q_action=self.oracle.queries-q0; eta=self._pvr_caou_eta(r,cand,q_action); qtot=max(1,int(self._q_normal)+int(max(0,q_action)))
        trial_state=self.pvr_caou_narrow_trial_state if narrow else self.pvr_caou_trial_state
        trial=int(trial_state is not None and trial_state.get("action")==action); trial_start=int(trial and trial_state.get("start_it")==int(self._it))
        switched=0; block_end=0; block_eta=float("nan"); bounds=(float("nan"),)*4; neffs=(float("nan"),float("nan"))
        if trial:
            trial_state["q"] += qtot; trial_state["left"] -= 1
            if trial_state["left"]<=0 or cand is None:
                block_end=1; rn=_norm(cand-self.origin) if cand is not None else trial_state["r0"]
                block_eta=max(0.0,math.log(max(trial_state["r0"],1e-12)/max(rn,1e-12)))/max(1,int(trial_state["q"])) if rn<trial_state["r0"] else 0.0
                self._pvr_caou_add_obs(action,trial_state["ctx"],block_eta)
                inc=trial_state["from"]; do,bounds=self._pvr_caou_should_switch(action,inc,trial_state["ctx"],block_eta,narrow)
                si=self._pvr_caou_stats(inc,trial_state["ctx"],narrow); sc=self._pvr_caou_stats(action,trial_state["ctx"],narrow); neffs=(si[2],sc[2])
                if narrow:
                    self.pvr_caou_last_narrow_trial_it=int(self._it); self.pvr_caou_narrow_action=action if do else inc; self.pvr_caou_narrow_trial_state=None
                else:
                    self.pvr_caou_last_trial_it=int(self._it); self.pvr_caou_mode=action if do else inc; self.pvr_caou_mode_age=0 if do else max(1,self.pvr_caou_mode_age); self.pvr_caou_trial_state=None
                switched=int(do)
        else:
            self._pvr_caou_add_obs(action,ctx,eta)
            if not narrow:
                self.pvr_caou_mode=action; self.pvr_caou_mode_age+=1
        self._pvr_caou_log(choice,action,trial,trial_start,switched,recheck,block_end,eta,block_eta,q_action,ctx,bounds,neffs,S,wt,kb,qref,kunc)
        return cand

    def _pvr_caou_fit_from_boundary(self, r: float, v: torch.Tensor, cand: torch.Tensor) -> None:
        if cand is None or cand is self.origin: return
        g=_norm(cand-self.origin)
        if not math.isfinite(g) or g<=0.0 or g>=r: return
        d=_unit(cand-self.origin)
        if d is None: return
        psi=math.acos(max(-1.0,min(1.0,_dot(d,v))))
        if psi>1e-9: self.hfit.update(r*math.sin(psi),sag_slope(r,psi,g))

    def _pvr_caou_model_step(self, n: torch.Tensor, xb: torch.Tensor, r: float, v: torch.Tensor,
                              wdir: torch.Tensor, theta: float, c_hat: float, unit_w: float) -> Tuple[torch.Tensor,float,float]:
        c=self.cfg; pool=self.hyb_pool
        if pool[1]<=0: return xb,float("nan"),float("nan")
        ct=max(-1+1e-7,min(1-1e-7,_dot(n,v))); ca=min(max((pool[0]/pool[1])/r,1e-3),0.999)
        psi=min(max(model_psi(c_hat,ct,ca),0.0),0.95*theta); d=_unit(math.cos(psi)*v+math.sin(psi)*wdir)
        if d is None: return xb,float("nan"),float("nan")
        g_pred,gamma_pred,g0,g2_raw,D,S,wt=self._pvr_caou_pred_radius(r,psi); bias,scale=self._pvr_caou_robust_stats(); d0=None
        if math.isfinite(scale): d0=min(max(c.pvr_bracket_mult*max(scale,c.pvr_caou_mad_floor),2.0*c.tol/max(g_pred,1e-12)),c.bracket_delta)
        qrad0=self.oracle.queries; out=self._bracket_radial(d,g_pred,r,d0=d0)
        if out is None: return xb,S,wt
        cand,g_act=out
        if math.isfinite(gamma_pred) and math.isfinite(g2_raw): self.pvr_caou_radius_resid.append((g_act-g2_raw)/r)
        kappa=0.0
        if self.hfit.ready():
            s_lat=r*math.sin(psi); kappa=2.0*self.hfit.predict_t(s_lat)/max(s_lat,1e-12)
        psi_s=secant_foot(r,g_act,psi,kappa,theta)
        if 1e-6<psi_s<theta:
            ca_cal=cos_alpha_from_bridge(c_hat,ct,psi_s)
            if 0.0<ca_cal<1.0:
                pool[0]=c.hyb_lambda*pool[0]+c.pvr_calib_weight*unit_w*r*ca_cal; pool[1]=c.hyb_lambda*pool[1]+c.pvr_calib_weight*unit_w
        if psi>1e-9: self.hfit.update(r*math.sin(psi),sag_slope(r,psi,g_act))
        self._log_geom("pvr_caou_model",r,v,xb,cand,g_pred=g_pred,gamma_pred=gamma_pred,cos_theta=ct,q_radial=self.oracle.queries-qrad0)
        self.pvr_caou_last_S=S; self.pvr_caou_last_weight=wt; return cand,S,wt

    def _pvr_caou_choose_narrow_action(self, r: float, kb: float) -> str:
        c=self.cfg; cur=self.pvr_caou_narrow_action
        if self.pvr_caou_narrow_trial_state is not None: return self.pvr_caou_narrow_trial_state["action"]
        if self.pvr_caou_narrow_steps < max(0,c.pvr_caou_narrow_probe_after): return cur
        if self._it-self.pvr_caou_last_narrow_trial_it < max(1,c.pvr_caou_narrow_trial_interval): return cur
        ch="R" if cur=="E" else "E"; ctx=self._pvr_caou_narrow_ctx(r,kb); inc=self._pvr_caou_stats(cur,ctx,True); st=self._pvr_caou_stats(ch,ctx,True)
        mu_i,_,_,lcb_i,_=inc; mu_c,_,ne_c,_,ucb_c=st; ref=max(c.pvr_caou_utility_floor,mu_i if math.isfinite(mu_i) else c.pvr_caou_utility_floor)
        cost=c.pvr_caou_narrow_switch_cost_frac*ref; stale=(self._it-self.pvr_caou_last_used.get(ch,-10**9)>=max(1,c.pvr_caou_stale_after))
        if ne_c<=0 or stale or (not math.isfinite(lcb_i)) or ucb_c>lcb_i+cost:
            self._pvr_caou_start_narrow_trial(ch,r,kb); return ch
        return cur

    def _update_pvre_caou(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v6: contextual confidence-aware adaptive-order utility controller."""
        c=self.cfg; xs=self.origin; r_vec=xb-xs; r=_norm(r_vec); v,n=_unit(r_vec),_unit(normal)
        if v is None or n is None or r<=1e-12: return xb
        if self.pvr_caou_r0 is None: self.pvr_caou_r0=r
        ct=max(-1+1e-7,min(1-1e-7,_dot(n,v))); wdir=_unit(n-ct*v)
        if wdir is None: return xb
        theta=math.acos(ct); c_hat=min(0.999,self._c_hat()); unit_w=(c_hat*math.sqrt(self.d_eff)/r)**2; pool=self.hyb_pool
        def pool_add(ca,weight):
            pool[0]=c.hyb_lambda*pool[0]+weight*r*ca; pool[1]=c.hyb_lambda*pool[1]+weight

        # Continue an active macro trial; otherwise start one only when confidence says it is plausible.
        trial_start=False
        if self.pvr_caou_trial_state is not None:
            action=self.pvr_caou_trial_state["action"]
        else:
            action=self.pvr_caou_mode
            if len(self.pvr_caou_radius_resid)>=max(1,c.pvr_caou_min_model_errs):
                ch=self._pvr_caou_trial_candidate(r)
                if ch is not None:
                    self._pvr_caou_start_wide_trial(ch,r); action=ch; trial_start=True

        need_bridge=(pool[1]<=0 or self.pvr_caou_in_narrow or action=="T" or self.pvr_caou_anchor_age>=max(1,c.pvr_caou_recheck_every))
        br=None; qref=0; kunc=float("nan"); kb=float("nan"); recheck=int(need_bridge); q0=self.oracle.queries
        if need_bridge:
            br=self._bridge(n,xb); self.pvr_caou_anchor_age=0
            if br is None:
                ctx=self._pvr_caou_wide_ctx(r); self._pvr_caou_log("bridge_fail",self.pvr_caou_mode,0,0,0,1,0,0.0,float("nan"),self.oracle.queries-q0,ctx)
                self.pvr_caou_trial_state=None; return xb
            kb=float(br["kb"])
            if self._no_bridge_inside(kb):
                # The certified bridge interval has no interior crossing for the planned
                # macro-action.  Cancel any pending wide trial rather than carrying stale
                # trial state across a forced TtBA fallback.
                self.pvr_caou_trial_state=None
                cand=self._ttba_radial_step(br,xb,"pvr_caou_fallback"); ctx=self._pvr_caou_wide_ctx(r); self.pvr_caou_in_narrow=False
                return self._pvr_caou_finish("T",cand,r,q0,"fallback",ctx,recheck=1,kb=kb)

            if kb<=c.escape_k:
                # A wide trial cannot be interpreted across a regime transition; cancel it.
                self.pvr_caou_trial_state=None; self.pvr_caou_in_narrow=True; self.pvr_caou_narrow_steps+=1
                br2,qref,kunc=self._pvr_caou_refine_narrow_bridge(br); na=self._pvr_caou_choose_narrow_action(r,kb); nctx=self._pvr_caou_narrow_ctx(r,kb)
                if na=="R":
                    brx=self._refine_bridge(br2,c.ttba_bridge_tol); cand=self._ttba_radial_step(brx,xb,"pvr_caou_rescue"); choice="rescue"
                else:
                    cand=self._escape_step(br2,xb,"pvr_caou_escape"); choice="escape"
                self._pvr_caou_fit_from_boundary(r,v,cand)
                return self._pvr_caou_finish(na,cand,r,q0,choice,nctx,narrow=True,recheck=1,kb=kb,qref=qref,kunc=kunc)

            if self.pvr_caou_in_narrow:
                self.pvr_caou_in_narrow=False; self.pvr_caou_narrow_steps=0; self.pvr_caou_narrow_action="E"; self.pvr_caou_narrow_trial_state=None
            psi_bridge=math.acos(max(-1.0,min(1.0,_dot(br["direction"](kb),v)))); ca_meas=cos_alpha_from_bridge(c_hat,ct,0.5*psi_bridge)
            if 0.0<ca_meas<1.0: pool_add(ca_meas,c.hyb_bridge_weight*unit_w)
            if pool[1]>0:
                ca_tmp=min(max((pool[0]/pool[1])/r,1e-3),0.999); psi_tmp=min(max(model_psi(c_hat,ct,ca_tmp),0.0),0.95*theta)
                _,_,_,_,_,St,wt=self._pvr_caou_pred_radius(r,psi_tmp); self.pvr_caou_last_S=St; self.pvr_caou_last_weight=wt

        # cold start always obtains model residual evidence and never spends trial budget.
        if len(self.pvr_caou_radius_resid)<max(1,c.pvr_caou_min_model_errs):
            self.pvr_caou_trial_state=None; action="M"
        ctx=self._pvr_caou_wide_ctx(r)
        if action=="M" and pool[1]<=0:
            if br is None:
                br=self._bridge(n,xb); recheck=1; self.pvr_caou_anchor_age=0
                if br is None:
                    self._pvr_caou_log("bridge_fail","M",0,0,0,1,0,0.0,float("nan"),self.oracle.queries-q0,ctx); return xb
                kb=float(br["kb"])
            psi_bridge=math.acos(max(-1.0,min(1.0,_dot(br["direction"](kb),v)))); ca_meas=cos_alpha_from_bridge(c_hat,ct,0.5*psi_bridge)
            if 0.0<ca_meas<1.0: pool_add(ca_meas,c.hyb_bridge_weight*unit_w)

        if action=="C":
            cand=self._update_cgba(n,xb); self._pvr_caou_fit_from_boundary(r,v,cand); self.pvr_caou_anchor_age+=1
            return self._pvr_caou_finish("C",cand,r,q0,"cgba",ctx,recheck=recheck,S=self.pvr_caou_last_S,wt=self.pvr_caou_last_weight,kb=kb)
        if action=="T":
            if br is None:
                br=self._bridge(n,xb); recheck=1; self.pvr_caou_anchor_age=0
                if br is None:
                    self._pvr_caou_log("bridge_fail","T",0,0,0,1,0,0.0,float("nan"),self.oracle.queries-q0,ctx); self.pvr_caou_trial_state=None; return xb
                kb=float(br["kb"])
            cand=self._ttba_radial_step(br,xb,"pvr_caou_ttba"); self._pvr_caou_fit_from_boundary(r,v,cand); self.pvr_caou_anchor_age+=1
            return self._pvr_caou_finish("T",cand,r,q0,"ttba",ctx,recheck=recheck,S=self.pvr_caou_last_S,wt=self.pvr_caou_last_weight,kb=kb)
        cand,S,wt=self._pvr_caou_model_step(n,xb,r,v,wdir,theta,c_hat,unit_w); self.pvr_caou_anchor_age+=1
        return self._pvr_caou_finish("M",cand,r,q0,"model" if cand is not xb else "model_fail",ctx,recheck=recheck,S=S,wt=wt,kb=kb)

    # -------------------------------- PVR-E v7 (build 31a): Gate - Thales - Escape
    def _pvr_gte_eps(self, br: dict, cand: torch.Tensor, xb: torch.Tensor, r: float,
                     v: torch.Tensor) -> float:
        """Flatness residual of a measured step, normalised by the flat model's own predicted gain.

        With the two current-radius boundary points (x_b at psi=0, bridge point at psi_b) the
        hyperplane through them predicts, for the step direction psi,
            g_lin(psi) = r cos(psi_b/2) / cos(psi - psi_b/2),
        and eps = (g_lin - g_act) / (r - g_lin).  eps = 0 on a hyperplane, eps > 0 when the boundary
        is closer than the hyperplane (obtuse, gamma_obs > 90 deg).  For a locally symmetric
        boundary g(psi) = h + a (psi - psi_b/2)^2 / 2 one has eps = a / r - 1 for every direction
        near the bisector (second order), and the Thales point gains 4 (1 + eps) / (2 + eps)^2 of
        the bisector gain.  Returns nan when the step did not move or the geometry is degenerate.
        """
        if cand is None or cand is xb or br is None or br.get("kb", 0.0) <= 0.0:
            return float("nan")
        dvec = cand - self.origin
        g = _norm(dvec)
        if not (0.0 < g < r):
            return float("nan")
        psi = math.acos(max(-1.0, min(1.0, _dot(dvec, v) / g)))
        psi_b = math.acos(max(-1.0, min(1.0, _dot(br["direction"](br["kb"]), v))))
        a = 0.5 * psi_b
        den = math.cos(psi - a)
        if den <= 1e-9:
            return float("nan")
        g_lin = r * math.cos(a) / den
        if r - g_lin <= 1e-9 * r:
            return float("nan")
        return (g_lin - g) / (r - g_lin)

    def _pvr_gte_flat(self) -> Tuple[bool, float]:
        """Thales gate: median eps of the last `pvr_gte_window` measured steps inside [-tol, tol]."""
        if len(self.gte_eps) < max(1, self.cfg.pvr_gte_window):
            return False, float("nan")
        med = float(np.median(np.asarray(self.gte_eps, dtype=float)))
        return (abs(med) <= self.cfg.pvr_gte_eps_tol), med

    def _pvr_gte_finish(self, choice: str, cand: torch.Tensor, r: float, q0: int,
                        eps: float = float("nan"), kb: float = float("nan")) -> torch.Tensor:
        if math.isfinite(eps):
            self.gte_eps.append(eps)
        _, med = self._pvr_gte_flat()
        g = _norm(cand - self.origin) if cand is not None else r
        self.diag["gte_choice"].append(choice)
        self.diag["gte_eps"].append(eps)
        self.diag["gte_eps_med"].append(med)
        self.diag["gte_q_update"].append(self.oracle.queries - q0)
        self.diag["gte_kb"].append(kb)
        self.diag["gte_gain"].append(math.log(r / g) if 0.0 < g < r else 0.0)
        return cand

    def _update_pvre_gte(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """PVR-E v7: one k=0.2 gate query, then exactly one of
             N  narrow (gate non-adversarial): certified [0, 0.2] bridge (1e-3) + TtBA k rule +
                bracketed radial search (the PVR-E escape) -> measures eps;
             T  wide and flat (median|eps| <= 1, < 4 Thales steps in a row): CGBA's Thales
                one-search (sweep + BSSP) in span(v, n_hat) with the SAME TtBA normal;
             M  wide, not Thales, build 32a (spec.gte_model): the PVR model step (no bridge
                queries), only before the frozen query cutoff pvr_gte_model_cutoff_q (A2);
             B  wide otherwise: certified [0.2, 1] bridge (1e-2) + TtBA 2/3 rule + bracketed radial
                search -> measures eps;
             F  wide with no bridge inside (k_b >= 1 - 2 tol): official TtBA radial step.
        The 31a rows (GTE / GE / GTE_K / GTEB) keep gte_model=False and therefore never take M:
        their choice sequences stay in NBTFX.  Acceptance stays monotone in run().
        """
        c = self.cfg
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12 or self.remaining <= 0:
            return xb
        q0 = self.oracle.queries

        def direction(k: float) -> torch.Tensor:
            d = _unit(k * n + (1.0 - k) * v)
            return v if d is None else d

        kg = float(c.escape_k)
        gate_adv = self._adv(self._proj(xs + r * direction(kg)), "update")
        self._dbg_gate_q = 1  # A3 diagnostic: the gate query is charged on every pvr_gte iteration
        if not gate_adv:  # ---- N: k_b < 0.2 under TtBA's local single-crossing assumption
            self.gte_thales_run = 0
            narrow_tol = self.spec.gte_narrow_tol if self.spec.gte_narrow_tol is not None else c.pvr_gte_narrow_tol
            br = self._route_bridge_from_bracket(normal, xb, 0.0, kg, narrow_tol, q0)
            if br is None:
                return self._pvr_gte_finish("X", xb, r, q0)
            cand = self._escape_step(br, xb, "pvr_gte_escape",
                                     eps_bracket=self._pvr_gte_eps_seed(r, br))
            return self._pvr_gte_finish("N", cand, r, q0, self._pvr_gte_eps(br, cand, xb, r, v), br["kb"])

        mode = self.spec.gte_thales
        flat, _ = self._pvr_gte_flat()
        use_thales = (mode == "always") or (
            mode == "flat" and flat and self.gte_thales_run < max(0, c.pvr_gte_thales_steps))
        if use_thales:  # ---- T: exact one-search on a hyperplane, no bridge, no radial search
            self.gte_thales_run += 1
            cand = self._update_cgba(n, xb)
            if cand is not xb:
                self._log_geom("pvr_gte_thales", r, v, xb, cand, cos_theta=_dot(n, v))
            return self._pvr_gte_finish("T", cand, r, q0)

        self.gte_thales_run = 0  # ---- B / F / M: wide, bridge or model in the certified [0.2, 1] bracket
        # A2 gate: the model step may be chosen only before the frozen query cutoff.  Once past the
        # cutoff the bridge step is forced.  (self.oracle.queries is the engine's single query counter.)
        q_cut = self.spec.pvr_gte_model_cutoff_q
        if q_cut is None:
            q_cut = c.pvr_gte_model_cutoff_q
        choice = "pvr_gte_bridge"
        if self.spec.gte_model and self.hyb_pool[1] > 0 and self.hyb_mode == "model":
            choice = "pvr_gte_model"
        if q_cut is not None and self.oracle.queries >= q_cut:
            choice = "pvr_gte_bridge"  # force the bridge step once past cutoff (A2)
        if choice == "pvr_gte_model":
            cand = self._pvr_gte_model_step(n, xb, r, v)
            if cand is not xb:
                return self._pvr_gte_finish("M", cand, r, q0)
            # failed model step (no verified improvement): mirrors PVR-E, which forces a bridge
            # re-measurement next iteration instead of racing a stale pool estimate
            self.hyb_mode = "bridge"
            return self._pvr_gte_finish("M", xb, r, q0)
        br = self._route_bridge_from_bracket(normal, xb, kg, 1.0, self.bridge_tol, q0)
        if br is None:
            return self._pvr_gte_finish("X", xb, r, q0)
        if self._no_bridge_inside(br["kb"]):
            cand = self._ttba_radial_step(br, xb, "pvr_gte_fallback")
            return self._pvr_gte_finish("F", cand, r, q0, kb=br["kb"])
        if self.spec.gte_model:
            self._gte_pool_bridge(br)  # warm the perpendicular-distance pool / hyb_mode (A2)
        cand = self._escape_step(br, xb, "pvr_gte_bridge",  # TtBA k(k_b) = 2/3 k_b for k_b >= 0.2
                                 eps_bracket=self._pvr_gte_eps_seed(r, br))
        return self._pvr_gte_finish("B", cand, r, q0, self._pvr_gte_eps(br, cand, xb, r, v), br["kb"])

    def _pvr_gte_model_step(self, n: torch.Tensor, xb: torch.Tensor, r: float,
                            v: torch.Tensor) -> torch.Tensor:
        """One PVR model step in the pvr_gte wide-flat branch (build 32a A2).

        Ported from the PVR-E model branch of _update_hybrid: predict the radius on the gamma-arc
        of the online hyperbolic fit (Thales if not ready), verify with the adaptive bracket, refine
        the pooled perpendicular distance by the secant foot.  No bridge queries.  Only reachable
        when spec.gte_model (never for the 31a GTE/GE/GTE_K rows).  Returns xb on failure.
        """
        c = self.cfg
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        w = _unit(n - ct * v)
        if w is None:
            return xb
        theta = math.acos(ct)
        c_hat = min(0.999, self._c_hat())
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2
        ca = min(max((self.hyb_pool[0] / self.hyb_pool[1]) / r, 1e-3), 0.999)
        psi = model_psi(c_hat, ct, ca)
        psi = min(max(psi, 0.0), 0.95 * theta)
        d = _unit(math.cos(psi) * v + math.sin(psi) * w)
        if d is None:
            return xb
        gamma_pred = float("nan")
        if self.hfit.ready():  # predict: gamma-arc radius
            gamma_pred = self.hfit.predict_gamma(r * math.sin(psi))
            g_pred = min(max(inscribed_radius(r, psi, gamma_pred), 0.05 * r), r * (1.0 - 1e-6))
        else:  # Thales radius (exact on a hyperplane)
            g_pred = r * math.cos(psi)
        d0 = None
        if len(self.pvr_errs) >= 3:  # verify: bracket matched to the recent prediction error
            q90 = float(np.quantile(np.asarray(self.pvr_errs, dtype=float), 0.9))
            d0 = min(max(c.pvr_bracket_mult * q90, 2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        q0 = self.oracle.queries
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        if out is None:  # overshoot beyond the bridge angle: re-measure next time
            self.hyb_mode = "bridge"
            return xb
        cand, g_act = out
        self.pvr_errs.append(abs(g_pred - g_act) / max(g_act, 1e-12))
        ratio = (r - g_act) / max(r - g_pred, 1e-12)
        if not (c.hyb_ratio_lo < ratio < c.hyb_ratio_hi):
            self.hyb_mode = "bridge"
        # refine: secant foot from (x_b, new point), curvature from the fit
        kappa = 0.0
        if self.hfit.ready():
            s_lat = r * math.sin(psi)
            kappa = 2.0 * self.hfit.predict_t(s_lat) / max(s_lat, 1e-12)
        psi_s = secant_foot(r, g_act, psi, kappa, theta)
        if 1e-6 < psi_s < theta:
            ca_cal = cos_alpha_from_bridge(c_hat, ct, psi_s)
            if 0.0 < ca_cal < 1.0:
                self.hyb_pool[0] = c.hyb_lambda * self.hyb_pool[0] + c.pvr_calib_weight * unit_w * r * ca_cal
                self.hyb_pool[1] = c.hyb_lambda * self.hyb_pool[1] + c.pvr_calib_weight * unit_w
        if psi > 1e-9:
            self.hfit.update(r * math.sin(psi), sag_slope(r, psi, g_act))
        self._log_geom("pvr_gte_model", r, v, xb, cand, g_pred=g_pred, gamma_pred=gamma_pred,
                       ratio=ratio, cos_theta=ct, q_radial=self.oracle.queries - q0)
        return cand

    def _gte_pool_bridge(self, br: dict) -> None:
        """Warm the PVR-E perpendicular-distance pool from a wide bridge measurement and set
        hyb_mode (build 32a A2).  Ported from _update_hybrid's bridge branch; only runs for rows
        with spec.gte_model, so the 31a GTE/GE/GTE_K rows are bit-for-bit unchanged.
        """
        c = self.cfg
        r, v, ct, kb = br["r"], br["v"], br["cos_theta"], br["kb"]
        psi_b = math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), v))))
        ap = 0.5 * psi_b
        c_hat = min(0.999, self._c_hat())
        unit_w = (c_hat * math.sqrt(self.d_eff) / r) ** 2
        ca_meas = cos_alpha_from_bridge(c_hat, ct, ap)
        if 0.0 < ca_meas < 1.0:
            self.hyb_pool[0] = c.hyb_lambda * self.hyb_pool[0] + c.hyb_bridge_weight * unit_w * r * ca_meas
            self.hyb_pool[1] = c.hyb_lambda * self.hyb_pool[1] + c.hyb_bridge_weight * unit_w
        ca = (self.hyb_pool[0] / self.hyb_pool[1]) / r if self.hyb_pool[1] > 0 else None
        if ca is not None and ap > 1e-9:
            psi_m = model_psi(c_hat, ct, min(max(ca, 1e-3), 0.999))
            self.hyb_mode = "model" if abs(psi_m / ap - 1.0) < c.hyb_agree else "bridge"

    def _pvr_gte_eps_seed(self, r: float, br: dict) -> Optional[Tuple[float, float]]:
        """A4: eps-predicted radial bracket (g_pred, d0) for the escape step, or None to fall back
        to the current bracket.  Falls back whenever fewer than pvr_gte_window eps are measured
        (and whenever the flag is off), so earlier methods are exactly unchanged.
        """
        c = self.cfg
        use = self.spec.gte_eps_bracket if self.spec.gte_eps_bracket is not None else c.pvr_gte_eps_bracket
        if not use:
            return None
        if len(self.gte_eps) < max(1, c.pvr_gte_window):
            return None
        eps_hat = float(np.median(np.asarray(self.gte_eps, dtype=float)))
        if not math.isfinite(eps_hat):
            return None
        kb = br["kb"]
        if kb <= 0.0:
            return None
        # g_lin for the escape direction (same formula as _escape_step)
        k = min(1.0, max(0.0, self._ttba_k(kb)))
        d = br["direction"](k)
        psi_k = math.acos(max(-1.0, min(1.0, _dot(d, br["v"]))))
        half_b = 0.5 * math.acos(max(-1.0, min(1.0, _dot(br["direction"](kb), br["v"]))))
        den = math.cos(psi_k - half_b)
        g_lin = r * math.cos(half_b) / den if den > 1e-9 else r
        g_lin = min(max(g_lin, 0.05 * r), r * (1.0 - 1e-6))
        # Derivation (implemented exactly, not re-derived): on a locally circular boundary
        #     g_act = g_lin - eps (r - g_lin),
        # so a bracket centred at g_hat = g_lin - eps_hat (r - g_lin) with half-width
        #     hw = max(2 tol / g_lin, 0.3 |eps_hat| (r - g_lin) / g_lin)
        # is a tighter a-priori bracket than the fixed bracket_delta margin.
        g_hat = g_lin - eps_hat * (r - g_lin)
        g_hat = min(max(g_hat, 0.05 * r), r * (1.0 - 1e-6))
        hw = max(2.0 * c.tol / g_lin, 0.3 * abs(eps_hat) * (r - g_lin) / g_lin)
        return g_hat, hw

    # -------------------------------- build 33b: pull-back PVR-E (LGC-PVRE)
    def _pb_config(self) -> PBConfig:
        """PBConfig resolved exactly like pvr_gte resolves its knobs (spec override, else EngineConfig)."""
        if self.pb_conf is None:
            c, s = self.cfg, self.spec
            self.pb_conf = PBConfig(
                tol=float(c.tol), bracket_delta=float(c.bracket_delta), escape_k=float(c.escape_k),
                narrow_tol=float(s.gte_narrow_tol if s.gte_narrow_tol is not None else c.pvr_gte_narrow_tol),
                bridge_tol=float(self.bridge_tol), eps_tol=float(c.pvr_gte_eps_tol),
                eps_window=int(c.pvr_gte_window), thales_steps=int(c.pvr_gte_thales_steps),
                thales_mode=str(s.gte_thales), gate_order=str(s.gte_gate_order),
                eps_bracket=bool(s.gte_eps_bracket if s.gte_eps_bracket is not None else c.pvr_gte_eps_bracket),
                bracket_mult=float(c.pvr_bracket_mult), err_window=int(c.pvr_err_window),
                ttba_low=float(c.ttba_low), ttba_high=float(c.ttba_high),
                ttba_jump=float(c.ttba_jump), ttba_dive=float(c.ttba_dive),
                rule=str(s.pb_rule), ttba_bridge_tol=float(c.ttba_bridge_tol))
            if self.pb_conf.rule not in ("pvre", "cgba", "cgba_h", "ttba") or s.pb_chart not in ("pullback", "latent"):
                raise ValueError(f"{s.name}: pb_rule={s.pb_rule!r} / pb_chart={s.pb_chart!r} not supported")
            self.pb_state = PBState.from_cfg(self.pb_conf)
        return self.pb_conf

    def _pb_frame(self, xb: torch.Tensor, r_xi: float, v: torch.Tensor, w: torch.Tensor, ct: float,
                  nu: Optional[np.ndarray] = None):
        """Pixel-isometric chart of span(v, w) at xi_b (0 oracle queries, 8 decoder evaluations).

        J v, J w by central differences of Phi = space.to_image o project with step
        h = pb_fd_rel * ||xi_b - xi_s||; Gram scalars accumulated in float64.  Returns (Frame2D, n_y), or
        None if the chart is degenerate, ill-conditioned (pb_cond_max) or fails the 0-query Verify step
        (|D_lin - D| / D > pb_verify_tol at the 3 chart_test_points, D from the decoder).
        """
        h = max(float(self.cfg.pb_fd_rel) * r_xi, 1e-8)
        x0f = self.space.x0.reshape(-1).double()

        def img(z: torch.Tensor) -> torch.Tensor:
            return self.space.to_image(self._proj(z)).reshape(-1).double()

        e = img(xb) - x0f
        jv = (img(xb + h * v) - img(xb - h * v)) / (2.0 * h)
        jw = (img(xb + h * w) - img(xb - h * w)) / (2.0 * h)
        g11, g12, g22 = float(torch.dot(jv, jv)), float(torch.dot(jv, jw)), float(torch.dot(jw, jw))
        self._pb_reject, self._pb_last = None, {"gain_ratio": math.sqrt(max(g22, 0.0) / max(g11, 1e-300))}
        try:  # cond(R) = sqrt(cond(Gram)), logged even when frame_from_gram rejects the chart
            ev = np.linalg.eigvalsh(np.array([[g11, g12], [g12, g22]], dtype=float))
            self._pb_last["cond"] = math.sqrt(ev[1] / ev[0]) if ev[0] > 0 else float("inf")
        except np.linalg.LinAlgError:
            self._pb_last["cond"] = float("nan")
        fr = frame_from_gram(g11, g12, g22, float(torch.dot(jv, e)), float(torch.dot(jw, e)), float(torch.dot(e, e)),
                             r_xi, float(self.cfg.pb_cond_max))
        if fr is None:
            self._pb_reject = "frame"
            return None
        if nu is None:
            nu = np.array([ct, math.sqrt(max(0.0, 1.0 - ct * ct))])
        n_y = normal_in_chart(fr, np.asarray(nu, dtype=float))
        if n_y is None:
            self._pb_reject = "normal"
            return None
        pts = chart_test_points(fr, n_y)  # Verify: first-order chart vs the true decoder distance
        d_true = []
        for y in pts:
            a, b = fr.ab(y)
            d_true.append(float(torch.linalg.vector_norm(img(xb + a * v + b * w) - x0f)))
        verr = chart_verify_error(fr, pts, d_true)
        self.diag["pb_verify_err"].append(verr)
        self._pb_last["verify"] = verr
        if not (verr <= float(self.cfg.pb_verify_tol)):
            self._pb_reject = "verify"
            return None
        return fr, n_y

    def _update_pvre_pb(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """LGC-PVRE update: PVR-E v7 (gate - Thales - escape) executed in a 2-D chart of span(v, w).

        Pull-back chart (default): every oracle query is issued at xi(y) = xi_b + a v + b w with
        (a, b) = R^{-1} (y - y_b), so gate, bridge, Thales and radial searches see pixel-isometric geometry
        (lgc_bench/pullback.py).  If the chart is rejected (degenerate, cond > pb_cond_max, or the 0-query
        Verify step fails) the SAME solver runs in the latent-Euclidean chart xi(y) = xi_s + y_1 v + y_2 w
        (= pvr_gte geometry).  Acceptance is the engine's monotone rule with the spec's objective.
        """
        q0 = self.oracle.queries
        pbc = self._pb_config()
        xs = self.origin
        r_vec = xb - xs
        r_xi = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        w = None
        ct = 0.0
        if v is not None and n is not None and r_xi > 1e-12:
            ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
            w = _unit(n - ct * v)
        if w is None or self.remaining <= 0:
            for k in ("gte_choice", "gte_eps", "gte_eps_med", "gte_q_update", "gte_kb", "gte_gain"):
                self.diag[k].append("X" if k == "gte_choice" else (0 if k == "gte_q_update" else float("nan")))
            self.diag["pb_fallback"].append(1)
            self.diag["pb_lin_err"].append(float("nan"))
            return xb
        if self.cfg.diag_normal_fn is not None:
            self._plane_diag(xb, n, v, r_xi)
        out = None
        if self.spec.pb_plane != "radial" and self.spec.pb_chart == "pullback":
            # build 35: contravariant / KKT / Gauss-Newton plane; the radial plane is the fallback
            alt = self._kkt_plane(xb, n, v, r_xi)
            used, reason = 0, "plane"
            if alt is not None:
                v2, w2, nu2 = alt
                out2 = self._pb_frame(xb, r_xi, v2, w2, 0.0, nu=nu2)
                self.diag.setdefault("kkt_gain_ratio", []).append(self._pb_last.get("gain_ratio", float("nan")))
                self.diag.setdefault("kkt_cond", []).append(self._pb_last.get("cond", float("nan")))
                reason = self._pb_reject or "ok"
                if out2 is not None:
                    yb2 = out2[0].y_b
                    cos2 = float(np.dot(out2[1], yb2)) / max(float(np.linalg.norm(yb2)), 1e-300)
                    self.diag.setdefault("kkt_cos", []).append(cos2)
                    if cos2 < float(self.cfg.kkt_cos_min):
                        out2, reason = None, "cos"
                if out2 is not None and out2[0].floor_share > float(self.cfg.kkt_floor_max):
                    out2, reason = None, "floor"
                if out2 is not None:
                    out, v, w, used, reason = out2, v2, w2, 1, "ok"
                    self.diag.setdefault("kkt_floor", []).append(out2[0].floor_share)
            self.diag.setdefault("kkt_used", []).append(used)
            self.diag.setdefault("kkt_reject", []).append(reason)
        if out is None:
            out = self._pb_frame(xb, r_xi, v, w, ct) if self.spec.pb_chart == "pullback" else None
        if out is not None:
            fr, n_y = out
            y_b = fr.y_b

            def to_xi(y) -> torch.Tensor:
                a, b = fr.ab(y)
                return self._proj(xb + a * v + b * w)
        else:  # latent-Euclidean chart (pvr_gte geometry), same solver and state
            fr = None
            n_y = np.array([ct, math.sqrt(max(0.0, 1.0 - ct * ct))])
            y_b = np.array([r_xi, 0.0])

            def to_xi(y) -> torch.Tensor:
                return self._proj(xs + float(y[0]) * v + float(y[1]) * w)

        solver = PullbackPVRE2D(pbc, self.pb_state, lambda y, tag: self._adv(to_xi(y), tag),
                                lambda: self.remaining)
        res = solver.step(y_b, n_y)
        cand, lin_err, gain = xb, float("nan"), 0.0
        if res.y is not None:
            cand = to_xi(res.y)
            d_new = _norm(self.space.to_image(cand) - self.space.x0)
            d_old = _norm(self.space.to_image(xb) - self.space.x0)
            if fr is not None:
                lin_err = abs(fr.d_lin(res.y) - d_new) / max(d_new, 1e-12)
            if 0.0 < d_new < d_old:
                gain = math.log(d_old / d_new)
        # per-iteration diagnostics are appended only after the step completed (a QueryBudgetExhausted
        # raised inside the solver leaves every per-iteration list aligned, as in pvr_gte)
        self.diag["pb_fallback"].append(0 if fr is not None else 1)
        if fr is not None:
            self.diag["pb_cond"].append(fr.cond)
            self.diag["pb_anchor_ratio"].append(fr.anchor_ratio)
            self.diag["pb_anchor_deg"].append(fr.anchor_deg)
            self.diag["pb_floor_share"].append(fr.floor_share)
        self.diag["pb_lin_err"].append(lin_err)
        eps_hist = list(self.pb_state.eps)
        med = (float(np.median(np.asarray(eps_hist, dtype=float)))
               if len(eps_hist) >= max(1, pbc.eps_window) else float("nan"))
        # pvr_gte-compatible bookkeeping (runs.csv gte_* columns; aligned per-iteration lists)
        self.diag["gte_choice"].append(res.choice)
        self.diag["gte_eps"].append(res.eps)
        self.diag["gte_eps_med"].append(med)
        self.diag["gte_q_update"].append(self.oracle.queries - q0)
        self.diag["gte_kb"].append(res.kb)
        self.diag["gte_gain"].append(gain)
        return cand

    # -------------------------------- build 35: decoder calculus (0 oracle queries)
    def _img_flat(self, z: torch.Tensor) -> torch.Tensor:
        return self.space.to_image(self._proj(z)).reshape(-1).double()

    def _jvp_fd(self, xb: torch.Tensor, d: torch.Tensor, h: float) -> torch.Tensor:
        """J d by central differences (2 decoder evaluations); d need not be a unit vector."""
        nd = _norm(d)
        if nd <= 1e-300:
            return torch.zeros_like(self._img_flat(xb))
        u = d / nd
        try:  # one batched decoder call (the oracle already evaluates batches through to_image)
            x2 = self.space.to_image(self._proj(torch.cat([xb + h * u, xb - h * u], 0)))
            x2 = x2.reshape(2, -1).double()
            return nd * (x2[0] - x2[1]) / (2.0 * h)
        except Exception:  # noqa: BLE001 - spaces without batch support
            return nd * (self._img_flat(xb + h * u) - self._img_flat(xb - h * u)) / (2.0 * h)

    def _vjp(self, xb: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        """J^T y by reverse-mode autodiff through space.to_image (1 decoder forward + backward).

        Returns None (and disables the KKT/KSQP paths for this run) if the space is not differentiable, e.g.
        when to_image is wrapped in torch.no_grad.  selftest_35.py Part C checks this on the real autoencoder."""
        if self._vjp_ok is False:
            return None
        try:
            with torch.enable_grad():
                z = xb.detach().clone().requires_grad_(True)
                x = self.space.to_image(z)
                if not x.requires_grad:
                    # a method decorated with @torch.no_grad() keeps the undecorated function in __wrapped__
                    raw = getattr(getattr(type(self.space), "to_image", None), "__wrapped__", None)
                    if raw is not None:
                        x = raw(self.space, z)
                if not x.requires_grad:
                    raise RuntimeError("space.to_image returned a tensor without grad (torch.no_grad inside?)")
                yy = y.detach().to(dtype=x.dtype, device=x.device).reshape(x.shape)
                (g,) = torch.autograd.grad((x * yy).sum(), z)
            if not torch.isfinite(g).all():
                raise RuntimeError("non-finite VJP")
            self._vjp_ok = True
            return g.detach().to(dtype=xb.dtype)
        except Exception as err:  # noqa: BLE001 - any failure disables the build-35 decoder paths
            if self._vjp_ok is None:
                print(f"[!] {self.spec.name}: VJP through the decoder failed ({err}); KKT/KSQP fall back to "
                      "the radial plane / no move")
            self._vjp_ok = False
            return None

    def _gn_operator(self, xb: torch.Tensor, r_xi: float, noisy: bool = True):
        """(matvec, lam, h): G p = J^T (J p) + lam p.

        mode "rel"   : lam = lam_rel ||J v||^2 (v = latent radial unit; lam_rel = spec.pb_lambda_rel or config).
        mode "wiener": lam = mbar tan^2(phi): the posterior mean of the contravariant normal M^-1 c given the
                       estimate n_hat = cos(phi) c_hat + noise (isotropic noise, prior Cov(c) ~ M) is
                       (M + lam I)^-1 n_hat (design doc s20); mbar = tr(M)/k by Hutchinson (one JVP per call, EMA),
                       tan^2(phi) = pi d_eff / (2 N f) with N the probe count and f = exp(-Phi^-1(share)^2).
        noisy=False (exact white-box normal, diagnostics): lam = config kkt_lambda_rel ||J v||^2."""
        h = max(float(self.cfg.pb_fd_rel) * r_xi, 1e-8)
        v = _unit(xb - self.origin)
        jv = self._jvp_fd(xb, v, h) if v is not None else None
        gain2 = max(float(torch.dot(jv, jv)) if jv is not None else 1.0, 1e-12)
        mode = str(self.spec.pb_lambda_mode) if noisy else "rel"
        if mode == "wiener":
            u = torch.randn(tuple(xb.shape), generator=self._kkt_gen).to(device=xb.device, dtype=xb.dtype)
            u = u / max(_norm(u), 1e-300)
            ju = self._jvp_fd(xb, u, h)
            m_t = float(torch.dot(ju, ju))
            a = float(self.cfg.kkt_mbar_ema)
            self.kkt_mbar = m_t if self.kkt_mbar is None else a * self.kkt_mbar + (1.0 - a) * m_t
            share = float(getattr(self, "_sign_share", 0.5))
            share = min(max(share, 0.01), 0.99) if math.isfinite(share) else 0.5
            f = min(1.0, max(0.05, math.exp(-innov35.Phi_inv(share) ** 2)))
            nq = max(1, int(getattr(self, "_q_normal", 0) or 1))
            tan2 = max(math.pi * float(self.d_eff) / (2.0 * nq * f), float(self.cfg.kkt_tan2_min))
            lam = max(self.kkt_mbar, 1e-12) * tan2
            self.diag.setdefault("kkt_tan2", []).append(tan2)
            self.diag.setdefault("kkt_mbar_rel", []).append(self.kkt_mbar / gain2)
        elif mode == "rel":
            rel = self.spec.pb_lambda_rel if (noisy and self.spec.pb_lambda_rel is not None) else self.cfg.kkt_lambda_rel
            lam = float(rel) * gain2
        else:
            raise ValueError(f"{self.spec.name}: pb_lambda_mode={mode!r} not supported")
        if noisy:
            self.diag.setdefault("kkt_lam_rel", []).append(lam / gain2)

        def matvec(p: torch.Tensor) -> Optional[torch.Tensor]:
            jp = self._jvp_fd(xb, p, h)
            jtjp = self._vjp(xb, jp)
            if jtjp is None:
                return None
            return jtjp + lam * p
        return matvec, lam, h

    def _cg(self, matvec, b: torch.Tensor, iters: int) -> Tuple[Optional[torch.Tensor], float]:
        """Conjugate gradients for G x = b (SPD), dot products in float64; (x, relative residual)."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rr = _dot(r, r)
        b2 = rr
        if not math.isfinite(b2) or b2 <= 0:
            return None, float("nan")
        for _ in range(int(iters)):
            Ap = matvec(p)
            if Ap is None:
                return None, float("nan")
            pAp = _dot(p, Ap)
            if not math.isfinite(pAp) or pAp <= 0:
                break
            a = rr / pAp
            x = x + a * p
            r = r - a * Ap
            rn = _dot(r, r)
            if rn <= (float(self.cfg.kkt_cg_tol) ** 2) * b2:
                rr = rn
                break
            p = r + (rn / rr) * p
            rr = rn
        return x, math.sqrt(max(rr, 0.0) / b2)

    def _kkt_plane(self, xb: torch.Tensor, n: torch.Tensor, v_rad: torch.Tensor, r_xi: float):
        """Basis (v, w) and in-plane normal coordinates nu of the build-35 search plane, or None.

        kkt_e: span(g, n) with g = J^T e, the exact latent gradient of 0.5 ||Phi - x0||^2 (one VJP).
        kkt_n: span(G^-1 g, G^-1 n) (the KSQP plane; 2 CG solves with kkt_cg_iters iterations).
        raised: span(v, G^-1 n) (1 CG solve); diag kkt_angle = angle(n, G^-1 n), the covariant/contravariant split."""
        if self.spec.pb_plane == "raised":
            matvec, _, _ = self._gn_operator(xb, r_xi)
            zn, res_n = self._cg(matvec, n, self.cfg.kkt_cg_iters)
            if zn is None:
                return None
            zu = _unit(zn)
            if zu is None:
                return None
            self.diag.setdefault("kkt_cg_res", []).append(res_n)
            self.diag.setdefault("kkt_angle", []).append(
                math.degrees(math.acos(max(-1.0, min(1.0, _dot(zu, n))))))
            w2 = _unit(zn - _dot(zn, v_rad) * v_rad)
            if w2 is None:
                return None
            return v_rad, w2, np.array([_dot(n, v_rad), _dot(n, w2)])
        e = self._img_flat(xb) - self.space.x0.reshape(-1).double()
        g = self._vjp(xb, e)
        if g is None:
            return None
        gu = _unit(g)
        if gu is None:
            return None
        self.diag.setdefault("kkt_angle", []).append(
            math.degrees(math.acos(max(-1.0, min(1.0, _dot(gu, n))))))
        if self.spec.pb_plane == "kkt_e":
            ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, gu)))
            w = _unit(n - ct * gu)
            if w is None:
                return None
            return gu, w, np.array([ct, math.sqrt(max(0.0, 1.0 - ct * ct))])
        if self.spec.pb_plane == "kkt_n":
            matvec, _, _ = self._gn_operator(xb, r_xi)
            zg, res_g = self._cg(matvec, g, self.cfg.kkt_cg_iters)
            zn, res_n = self._cg(matvec, n, self.cfg.kkt_cg_iters)
            if zg is None or zn is None:
                return None
            self.diag.setdefault("kkt_cg_res", []).append(max(res_g, res_n))
            v2 = _unit(zg)
            if v2 is None:
                return None
            w2 = _unit(zn - _dot(zn, v2) * v2)
            if w2 is None:
                return None
            return v2, w2, np.array([_dot(n, v2), _dot(n, w2)])
        raise ValueError(f"{self.spec.name}: pb_plane={self.spec.pb_plane!r} not supported")

    def _frame_only(self, xb: torch.Tensor, r_xi: float, v: torch.Tensor, w: torch.Tensor):
        """frame_from_gram of span(v, w) at xi_b without the Verify step and without touching self.diag."""
        h = max(float(self.cfg.pb_fd_rel) * r_xi, 1e-8)
        x0f = self.space.x0.reshape(-1).double()
        e = self._img_flat(xb) - x0f
        jv = (self._img_flat(xb + h * v) - self._img_flat(xb - h * v)) / (2.0 * h)
        jw = (self._img_flat(xb + h * w) - self._img_flat(xb - h * w)) / (2.0 * h)
        return frame_from_gram(float(torch.dot(jv, jv)), float(torch.dot(jv, jw)), float(torch.dot(jw, jw)),
                               float(torch.dot(jv, e)), float(torch.dot(jw, e)), float(torch.dot(e, e)),
                               r_xi, float(self.cfg.pb_cond_max))

    def _plane_diag(self, xb: torch.Tensor, n_hat: torch.Tensor, v: torch.Tensor, r_xi: float) -> None:
        """0-query white-box diagnostic of the search plane (design doc s19 decision gate).

        For P in {radial, raised} built from the ESTIMATED normal (what the attack does) and from the TRUE normal
        (the plane's potential), the first-order distance after one exact in-plane step is
            D+ = sqrt(||e_perp||^2 + (n_y . y_b)^2),   n_y = unit(R^-T (n.v, n.w)),  n = TRUE normal,
        recorded as D+ / D in dg_{est,true}_{radial,raised}; dg_cos = cos(n_hat, n), dg_raise_deg = angle(n, G^-1 n),
        dg_q = oracle queries so far.  Never changes the attack."""
        keys = ("dg_est_radial", "dg_est_raised", "dg_true_radial", "dg_true_raised", "dg_cos", "dg_raise_deg", "dg_q")
        vals = {k: float("nan") for k in keys}
        vals["dg_q"] = float(self.oracle.queries)
        try:
            nt = _unit(self.cfg.diag_normal_fn(xb).to(dtype=xb.dtype).reshape(xb.shape))
            if nt is None:
                raise RuntimeError("zero white-box gradient")
            vals["dg_cos"] = _dot(nt, n_hat)
            ops = {"est": self._gn_operator(xb, r_xi, noisy=True)[0], "true": self._gn_operator(xb, r_xi, noisy=False)[0]}
            for src, nn in (("est", n_hat), ("true", nt)):
                for plane in ("radial", "raised"):
                    d = nn
                    if plane == "raised":
                        d, _ = self._cg(ops[src], nn, self.cfg.kkt_cg_iters)
                        if d is None:
                            continue
                        if src == "true":
                            du = _unit(d)
                            if du is not None:
                                vals["dg_raise_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, _dot(du, nt)))))
                    w = _unit(d - _dot(d, v) * v)
                    if w is None:
                        continue
                    fr = self._frame_only(xb, r_xi, v, w)
                    if fr is None:
                        continue
                    n_y = normal_in_chart(fr, np.array([_dot(nt, v), _dot(nt, w)]))
                    if n_y is None:
                        continue
                    d_plus = math.sqrt(fr.floor2 + float(np.dot(n_y, fr.y_b)) ** 2)
                    vals[f"dg_{src}_{plane}"] = d_plus / max(fr.e_norm, 1e-300)
        except Exception as err:  # noqa: BLE001 - diagnostics must never stop a run
            if not getattr(self, "_dg_warned", False):
                print(f"[!] {self.spec.name}: plane diagnostic failed ({err})")
                self._dg_warned = True
        for k in keys:
            self.diag.setdefault(k, []).append(vals[k])

    # -------------------------------- build 36: MA-PVRE (manifold-ambient PVR-E)
    def _ma_pix_dist(self, z: torch.Tensor) -> float:
        return _norm(self.space.to_image(z) - self.space.x0)

    def _ma_elasticity(self) -> float:
        """eps_hat(q) = ln(d(q_j) / d(q)) / ln(q / q_j), q_j = last recorded query count <= q / 2 (build 37).
        Under d(q) = d_V + C q^-alpha0 (latent floor d_V): eps = alpha0 (1 - d_V / d), i.e. the floor share is
        1 - eps / alpha0."""
        return innov35.elasticity(self._ma_hist)

    def _ma_gate(self, current: torch.Tensor) -> bool:
        """Is the ambient block admissible?  Never before ma_start_q; then per ma_gate (build 37):
        none (build 36), ssim (SSIM(x, x0) >= ma_gate_ssim, the evaluation's SSIM, 0 queries) or
        elastic (q >= ma_gate_q_elastic and eps_hat < ma_gate_eps).  Once open it stays open."""
        if self._ma_open:
            return True
        s = self.spec
        q = self.oracle.queries
        if q < int(s.ma_start_q):
            return False
        g = str(s.ma_gate)
        if g == "none":
            ok = True
        elif g == "ssim":
            val = _ssim_value(self.space.to_image(current), self.space.x0)
            self.diag.setdefault("ma_gate_ssim", []).append(val)
            ok = math.isfinite(val) and val >= float(s.ma_gate_ssim)
        elif g == "elastic":
            eps = self._ma_elasticity()
            self.diag.setdefault("ma_gate_eps", []).append(eps)
            ok = q >= int(s.ma_gate_q_elastic) and math.isfinite(eps) and eps < float(s.ma_gate_eps)
        else:
            raise ValueError(f"{s.name}: ma_gate={g!r} (none | ssim | elastic)")
        if ok:
            self._ma_open = True
            if g != "none":
                self.diag["ma_gate_q"] = [q]
            if self._obj_latent:  # MAL: from now on pixel objective; latent-objective rates are not comparable
                self._obj_latent = False
                self._obj_dirty = True
                self.ma_rate = {"L": None, "A": None}
        return ok

    def _ma_choose(self, current: torch.Tensor) -> str:
        """Block for this iteration: latent only while the ambient block is not admissible (ma_start_q, gate);
        then the block with the larger EMA log-progress per query, each block forced at least every k_B
        iterations (k_B = ma_explore_every; with ma_explore = doubling, k_B doubles after a forced trial of B that
        stays slower than the other block and resets after one that is faster)."""
        self._ma_forced = None
        if not self._ma_gate(current):
            return "L"
        if self.ma_rate["A"] is None:
            return "A"
        if self.ma_since["A"] + 1 >= self.ma_k["A"]:
            self._ma_forced = "A"
            return "A"
        if self.ma_since["L"] + 1 >= self.ma_k["L"]:
            self._ma_forced = "L"
            return "L"
        r_l = self.ma_rate["L"] if self.ma_rate["L"] is not None else 0.0
        return "A" if self.ma_rate["A"] > r_l else "L"

    def _ma_normal(self, center: torch.Tensor, q: int, block: str) -> torch.Tensor:
        """Sign-sum normal of ONE block (the other block's probe coordinates are 0).
        L: N(0, I) probes in the latent block, sigma = spec/config sigma, drawn with the base shape (same stream as
        the base latent_r rows).  A: normalised DCT-low probes in the pixel block (TtBA scale ma_sigma_amb)."""
        sp, c = self.space, self.cfg
        if block == "L":
            sig = float(self.sigma)
        else:
            sig = float(c.ma_sigma_amb)
            if self._amb_sampler is None:
                ch, hh, ww = sp.pix_shape
                f = int(c.ma_dct_factor)
                self._amb_sampler = DCTLowFreqSampler(sp.pix_shape, f, self.device,
                                                      normalize_to=math.sqrt(ch * int(hh / f) * int(ww / f)))
        # accumulated exactly as GeometricAttack._estimate_normal (zeros_like(center), in-place add), so that the
        # latent block with p = 0 reproduces the latent_r rows up to the reduction order of the longer vector
        acc = torch.zeros_like(center)
        done, n_pos = 0, 0
        while done < q:
            b = min(c.query_batch, q - done)
            if block == "L":
                u = sp.lift(self.rng_normal.randn(b, *sp.base_shape))
            else:
                u = sp.lift_pix(self._amb_sampler.sample(b, self.rng_normal))
            signs = self.oracle.signs(self._proj(center + sig * u), "normal")
            acc += (signs.view(b, *([1] * (u.dim() - 1))) * u).sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            done += b
        self._sign_share = n_pos / max(done, 1)
        self.diag["sign_share"].append(self._sign_share)
        self.diag["n_probes"].append(done)
        n = _unit(acc)
        if n is None:
            n = _unit(center - self.origin)
        return n

    def _ma_amb_config(self) -> PBConfig:
        if self.pb_conf_amb is None:
            self.pb_conf_amb = _dc_replace(self._pb_config(), rule=str(self.spec.ma_rule_amb))
            self.pb_state_amb = PBState.from_cfg(self.pb_conf_amb)
        return self.pb_conf_amb

    def _ma_ambient(self, n_full: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Ambient block: PVR-E (2-D solver, identity chart) in the pixel plane x0 + span(e, t), e = x(xb) - x0,
        t = unit(n_pix - (n_pix.e^) e^).  The plane contains the whole perturbation (semantic + ambient): every
        step may also shrink the latent-induced part (Theorem 3, no floor).  xi is frozen; only p moves."""
        sp = self.space
        _, n_pix = sp.split(n_full)
        npx = _unit(n_pix.reshape(-1).double())
        x = self._img_flat(xb)
        x0f = sp.x0.reshape(-1).double()
        e = x - x0f
        dist = float(torch.linalg.vector_norm(e))
        if npx is None or dist <= 1e-12:
            return xb
        eh = e / dist
        ct = float(torch.dot(npx, eh))
        if ct <= 1e-6:
            return xb
        t = npx - ct * eh
        tn = float(torch.linalg.vector_norm(t))
        if tn <= 1e-12:
            return xb
        th = t / tn
        xi, p = sp.split(xb)

        def to_z(y) -> torch.Tensor:
            target = x0f + float(y[0]) * eh + float(y[1]) * th
            dp = (target - x).to(dtype=p.dtype).reshape(p.shape)
            return sp.join(xi, p + dp)

        pbc = self._ma_amb_config()
        solver = PullbackPVRE2D(pbc, self.pb_state_amb, lambda y, tag: self._adv(to_z(y), tag), lambda: self.remaining)
        res = solver.step(np.array([dist, 0.0]), np.array([ct, math.sqrt(max(0.0, 1.0 - ct * ct))]))
        self.diag.setdefault("ma_amb_choice", []).append(res.choice)
        return xb if res.y is None else to_z(res.y)

    def _step_mapvre(self, it: int, current: torch.Tensor):
        sp = self.space
        if not (hasattr(sp, "split") and hasattr(sp, "base")):
            raise ValueError(f"{self.spec.name}: update 'mapvre' needs the latent_ra space (lgc_bench/ambient.py)")
        if self.spec.objective != "pixel":
            raise ValueError(f"{self.spec.name}: MA-PVRE needs objective='pixel'")
        q = min(self._n_probes(it, current), self.remaining - self.cfg.update_reserve)
        if q <= 0:
            return _STOP
        self._q_normal = q
        if str(self.spec.ma_gate) == "elastic" and not self._ma_open:
            dp = self._ma_pix_dist(current)
            if self._ma_hist:
                dp = min(dp, self._ma_hist[-1][1])
            self._ma_hist.append((self.oracle.queries, dp))
        block = self._ma_choose(current)
        q0 = self.oracle.queries
        d0 = self.objective(current)
        if block == "L":
            n = self._ma_normal(current, q, "L")
            _, p = sp.split(current)
            saved = self.origin
            self.origin = sp.join(sp.base.origin, p)  # latent block: radial plane through [xi_s, p]
            try:
                if str(self.spec.ma_latent) == "lgc":
                    out = self._update_cgba(n, current)  # LGC_RC's own step (build 37, MAL rows)
                else:
                    out = self._update_pvre_pb(n, current)
            finally:
                self.origin = saved
        else:
            n = self._ma_normal(current, q, "A")
            out = self._ma_ambient(n, current)
        dq = max(1, self.oracle.queries - q0)
        d1 = self.objective(out) if out is not current else d0
        gain = math.log(d0 / d1) if (0.0 < d1 < d0) else 0.0
        a = float(self.cfg.ma_rate_ema)
        r = gain / dq
        self.ma_rate[block] = r if self.ma_rate[block] is None else a * self.ma_rate[block] + (1.0 - a) * r
        for k in ("L", "A"):
            self.ma_since[k] = 0 if k == block else self.ma_since[k] + 1
        if str(self.spec.ma_explore) == "doubling" and self._ma_forced == block:
            other = "A" if block == "L" else "L"
            r_o = self.ma_rate[other]
            lost = r_o is not None and self.ma_rate[block] < r_o  # the forced trial lost: explore B half as often
            self.ma_k[block] = innov35.explore_interval(self.ma_k[block], int(self.spec.ma_explore_every),
                                                        int(self.spec.ma_explore_max), lost)
            self.diag.setdefault("ma_k_" + block, []).append(self.ma_k[block])
        self.diag.setdefault("ma_open", []).append(int(self._ma_open))
        self.diag.setdefault("ma_block", []).append(block)
        self.diag.setdefault("ma_q", []).append(q0)
        self.diag.setdefault("ma_gain", []).append(gain)
        self.diag.setdefault("ma_rate_L", []).append(self.ma_rate["L"] if self.ma_rate["L"] is not None else float("nan"))
        self.diag.setdefault("ma_rate_A", []).append(self.ma_rate["A"] if self.ma_rate["A"] is not None else float("nan"))
        self.diag.setdefault("ma_pshare", []).append(sp.ambient_share(out if gain > 0 else current))
        return out

    # -------------------------------- build 35: Krylov-SQP (external proposal, faithful)
    def _update_ksqp(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """min 0.5 ||Phi(xi) - x0||^2 on the boundary by one SQP step:
            p = -z_g + z_n (n.z_g)/(n.z_n),  z = G^-1 (.) by CG,  G = J^T J + lam I,  g = J^T e
        predictor xi + alpha p with the pixel trust radius alpha = min(1, Delta / sqrt(p^T G p)); metric-normal
        corrector xi(beta) = pred + beta q_n, q_n = z_n / (n.z_n), beta bracketed and bisected to the adversarial
        side of the boundary (queries tagged 'update'); trust ratio rho = dF / dm adapts Delta."""
        q0 = self.oracle.queries
        c = self.cfg
        n = _unit(normal)
        r_vec = xb - self.origin
        r_xi = _norm(r_vec)
        x0f = self.space.x0.reshape(-1).double()
        e = self._img_flat(xb) - x0f
        D = float(torch.linalg.vector_norm(e))
        out, tag = xb, "X"
        try:
            if n is None or r_xi <= 1e-12 or D <= 0:
                return xb
            if self.ksqp_delta is None:
                self.ksqp_delta = float(c.ksqp_trust0) * D
            g = self._vjp(xb, e)
            if g is None:
                return xb
            matvec, lam, h = self._gn_operator(xb, r_xi)
            zg, _ = self._cg(matvec, g, c.kkt_cg_iters)
            zn, _ = self._cg(matvec, n, c.kkt_cg_iters)
            if zg is None or zn is None:
                return xb
            den = _dot(n, zn)
            if not math.isfinite(den) or den <= 0:
                return xb
            p = -zg + zn * (_dot(n, zg) / den)
            jp = self._jvp_fd(xb, p, h)
            pGp = float(torch.dot(jp, jp)) + lam * _dot(p, p)
            if not math.isfinite(pGp) or pGp <= 0:
                return xb
            alpha = min(1.0, self.ksqp_delta / math.sqrt(pGp))
            pred = self._proj(xb + alpha * p)
            dm = -(alpha * _dot(g, p) + 0.5 * alpha * alpha * pGp)
            qn = zn / den
            jq = float(torch.linalg.vector_norm(self._jvp_fd(xb, qn, h)))  # pixel length per unit beta
            if not math.isfinite(jq) or jq <= 0:
                return xb
            s0 = float(c.ksqp_corr_scale) * D / jq
            tol_b = float(c.tol) / jq
            if self._adv(pred, "update"):
                hi, lo, ok = 0.0, -s0, False
                for _ in range(int(c.ksqp_bracket_max)):
                    if not self._adv(self._proj(pred + lo * qn), "update"):
                        ok = True
                        break
                    hi, lo = lo, 2.0 * lo
            else:
                lo, hi, ok = 0.0, s0, False
                for _ in range(int(c.ksqp_bracket_max)):
                    if self._adv(self._proj(pred + hi * qn), "update"):
                        ok = True
                        break
                    lo, hi = hi, 2.0 * hi
            if not ok:
                self.ksqp_delta = max(0.5 * self.ksqp_delta, float(c.ksqp_trust_min) * D)
                tag = "B"
                return xb
            for _ in range(60):
                if abs(hi - lo) <= tol_b or self.remaining <= 0:
                    break
                mid = 0.5 * (lo + hi)
                if self._adv(self._proj(pred + mid * qn), "update"):
                    hi = mid
                else:
                    lo = mid
            cand = self._proj(pred + hi * qn)
            Dc = float(torch.linalg.vector_norm(self._img_flat(cand) - x0f))
            rho = (0.5 * (D * D - Dc * Dc) / dm) if dm > 0 else -1.0
            if rho < 0.25:
                self.ksqp_delta = max(0.5 * self.ksqp_delta, float(c.ksqp_trust_min) * D)
            elif rho > 0.75:
                self.ksqp_delta = min(1.5 * self.ksqp_delta, D)
            self.diag.setdefault("ksqp_alpha", []).append(alpha)
            self.diag.setdefault("ksqp_rho", []).append(rho)
            self.diag.setdefault("ksqp_beta", []).append(hi * jq / D)
            out, tag = cand, "S"
            return cand
        finally:
            self.diag.setdefault("ksqp_choice", []).append(tag)
            self.diag.setdefault("ksqp_q_update", []).append(self.oracle.queries - q0)
            self.diag.setdefault("ksqp_delta_rel", []).append(
                (self.ksqp_delta / D) if (self.ksqp_delta is not None and D > 0) else float("nan"))

    # -------------------------------- build 35: curvature-calibrated probe scale (CPS)
    def _estimate_normal_cps(self, center: torch.Tensor, q: int) -> torch.Tensor:
        """Sign-sum normal estimate with an adaptive probe radius sigma_t = sigma * cps_mult (innov35 docstring).

        A share cps_pair_frac of the probes are antithetic pairs (+u, -u): both signs enter the estimator as
        (s+ - s-) u (= the two single-probe terms), and their agreement rate measures the even-order part of the
        boundary at the probe scale.  (share, agreement) -> (beta, gamma) -> scale-invariant (B, Gam) with the
        bisection offset delta = cps_offset_frac * tol; the next radius maximises the c^2 multiplier
        f(sigma_n) = exp(-(delta/sigma_n + B sigma_n)^2 / (1 + Gam^2 sigma_n^2)) / (1 + Gam^2 sigma_n^2)
        inside [s_min, s_max] x the current radius (geometric damping sqrt, clamp [mult_min, mult_max])."""
        c = self.cfg
        sig = float(self.sigma) * float(self.cps_mult)
        n_pairs = max(2, int(round(q * float(c.cps_pair_frac) / 2.0)))
        n_pairs = min(n_pairs, q // 2)
        n_single = q - 2 * n_pairs
        acc = torch.zeros_like(center)
        done, n_pos, agree, u2 = 0, 0, 0, 0.0
        while done < n_single:
            b = min(c.query_batch, n_single - done)
            u = self.sampler.sample(b, self.rng_normal)
            signs = self.oracle.signs(self._proj(center + sig * u), "normal")
            acc += (signs.view(b, *([1] * (u.dim() - 1))) * u).sum(0, keepdim=True)
            n_pos += int((signs > 0).sum().item())
            u2 += float((u * u).sum().item())
            done += b
        pairs_done = 0
        while pairs_done < n_pairs:
            b = min(max(1, c.query_batch // 2), n_pairs - pairs_done)
            u = self.sampler.sample(b, self.rng_normal)
            s1 = self.oracle.signs(self._proj(center + sig * u), "normal")
            s2 = self.oracle.signs(self._proj(center - sig * u), "normal")
            acc += ((s1 - s2).view(b, *([1] * (u.dim() - 1))) * u).sum(0, keepdim=True)
            n_pos += int((s1 > 0).sum().item()) + int((s2 > 0).sum().item())
            agree += int((s1 == s2).sum().item())
            u2 += 2.0 * float((u * u).sum().item())
            pairs_done += b
            done += 2 * b
        share = n_pos / max(done, 1)
        agree_rate = agree / max(pairs_done, 1)
        # normal-component std of one probe: sigma * sqrt(E|u|^2 / d_eff) (1 x sigma for N(0, I) probes)
        sig_n = sig * math.sqrt(max(u2, 1e-300) / (max(done, 1) * max(self.d_eff, 1)))
        delta = float(c.cps_offset_frac) * float(c.tol)
        beta, gamma = innov35.cps_solve(share, agree_rate, n_pairs=pairs_done)
        B_t, G_t = innov35.cps_decompose(beta, gamma, sig_n, delta)
        a = float(c.cps_ema)
        self.cps_B = B_t if self.cps_B is None else a * self.cps_B + (1 - a) * B_t
        self.cps_G = G_t if self.cps_G is None else a * self.cps_G + (1 - a) * G_t
        sig_star = innov35.cps_best_sigma(delta, self.cps_B, self.cps_G,
                                          sig_n * float(c.cps_s_min), sig_n * float(c.cps_s_max))
        f_now = innov35.cps_signal(sig_n, delta, self.cps_B, self.cps_G)
        f_star = innov35.cps_signal(sig_star, delta, self.cps_B, self.cps_G)
        self.cps_mult = float(min(max(self.cps_mult * math.sqrt(sig_star / sig_n), float(c.cps_mult_min)),
                                  float(c.cps_mult_max)))
        self.diag.setdefault("cps_mult", []).append(self.cps_mult)
        self.diag.setdefault("cps_beta", []).append(beta)
        self.diag.setdefault("cps_gamma", []).append(gamma)
        self.diag.setdefault("cps_B", []).append(self.cps_B * sig_n)  # mean-curvature part of beta (current)
        self.diag.setdefault("cps_agree", []).append(agree_rate)
        self.diag.setdefault("cps_gain", []).append(f_now)
        self.diag.setdefault("cps_gain_star", []).append(f_star)
        self._sign_share = share
        self.diag["sign_share"].append(share)
        self.diag["n_probes"].append(done)
        nrm = _unit(acc)
        if nrm is None:
            nrm = _unit(center - self.origin)
        return nrm

    # -------------------------------- reviewer-corrected RA-MB-PVRE prototype
    def _ra_candidate(self, name: str, k: float, br: dict, psi_b: float) -> RACandidate:
        """Construct one RA-MB candidate without spending a victim-model query."""
        k = min(1.0, max(0.0, float(k)))
        d = br["direction"](k)
        psi = math.acos(max(-1.0, min(1.0, _dot(d, br["v"]))))
        g_pred = ramb_flat_radius(br["r"], psi, psi_b)
        return RACandidate(name=name, k=k, psi=psi, direction=d, g_pred=g_pred)

    def _ra_init_bracket(self, cand: RACandidate, br: dict, race_start_q: int) -> bool:
        """Verify the flat-radius prediction and create a certified radial interval.

        The race budget includes this screening.  If the prediction is adversarial,
        [0, g_pred] is a certified interval because the source is non-adversarial.  If
        it is non-adversarial we spend one same-radius query; only an adversarial point
        at r creates [g_pred, r].  A non-adversarial r point is a direct diagnostic that
        the candidate/bridge single-crossing picture is not reliable for this ray.
        """
        c = self.cfg
        if self.remaining <= 0 or self.oracle.queries - race_start_q >= c.ra_race_max_queries:
            return False
        xs = self.origin

        def point(t: float) -> torch.Tensor:
            return self._proj(xs + t * cand.direction)

        q0 = self.oracle.queries
        pp = point(cand.g_pred)
        if self._adv(pp, "boundary"):
            cand.lo, cand.hi = 0.0, cand.g_pred
            cand.lo_obj, cand.hi_obj = 0.0, self.objective(pp)
            cand.high_point, cand.valid = pp, True
        else:
            cand.lo, cand.lo_obj = cand.g_pred, self.objective(pp)
            if self.remaining <= 0 or self.oracle.queries - race_start_q >= c.ra_race_max_queries:
                cand.queries += self.oracle.queries - q0
                return False
            pr = point(br["r"])
            if not self._adv(pr, "boundary"):
                cand.valid = False
                cand.queries += self.oracle.queries - q0
                return False
            cand.hi, cand.hi_obj = br["r"], self.objective(pr)
            cand.high_point, cand.valid = pr, True
        cand.queries += self.oracle.queries - q0
        return cand.valid

    def _ra_bisect_once(self, cand: RACandidate) -> None:
        """One deterministic interval-refinement query for an active candidate."""
        if not cand.valid or self.remaining <= 0:
            return
        xs = self.origin
        m = 0.5 * (cand.lo + cand.hi)
        pm = self._proj(xs + m * cand.direction)
        if self._adv(pm, "boundary"):
            cand.hi, cand.hi_obj, cand.high_point = m, self.objective(pm), pm
        else:
            cand.lo, cand.lo_obj = m, self.objective(pm)
        cand.queries += 1

    def _ra_refine(self, cand: RACandidate) -> Tuple[Optional[torch.Tensor], float, int]:
        """Winner-only full verification/refinement inside its certified bracket.

        Unlike selecting a noisy midpoint estimate, this always returns the verified
        adversarial upper endpoint.  The maximum number of refinement queries is capped
        so the method remains comparable at fixed total query budgets.
        """
        if not cand.valid or cand.high_point is None:
            return None, float("inf"), 0
        q0 = self.oracle.queries
        steps = 0
        while (cand.hi - cand.lo > self.cfg.tol and self.remaining > 0
               and steps < self.cfg.ra_refine_max_queries):
            self._ra_bisect_once(cand)
            steps += 1
        return cand.high_point, cand.hi_obj, self.oracle.queries - q0

    def _ra_prune(self, active: List[RACandidate]) -> Tuple[List[RACandidate], int]:
        """Prune only branches that are certified unable to beat the best upper bound."""
        if len(active) <= 1:
            return active, 0
        incumbent = min(active, key=lambda x: x.hi_obj)
        umin = incumbent.hi_obj
        kept, pruned = [], 0
        for cand in active:
            if cand is not incumbent and safe_prunable(cand, umin):
                pruned += 1
            else:
                kept.append(cand)
        return kept, pruned

    def _route_bridge_from_bracket(self, normal: torch.Tensor, xb: torch.Tensor,
                                   lo: float, hi: float, tol: float,
                                   q0: int) -> Optional[dict]:
        """Refine k_bridge only inside a gate-certified interval [lo, hi].

        Preconditions (under TtBA's local single-crossing model): the current-radius
        point at ``lo`` is adversarial and the point at ``hi`` is non-adversarial.
        Gate queries have already been counted; q_bridge in the returned record includes
        those gate queries by measuring from ``q0``.
        """
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return None

        def direction(k: float) -> torch.Tensor:
            d = _unit(k * n + (1.0 - k) * v)
            return v if d is None else d

        lo = float(max(0.0, min(1.0, lo)))
        hi = float(max(lo, min(1.0, hi)))
        while hi - lo > tol and self.remaining > 0:
            mid = 0.5 * (lo + hi)
            if self._adv(self._proj(xs + r * direction(mid)), "update"):
                lo = mid
            else:
                hi = mid
        self.diag["k_bridge"].append(lo)
        cos_t = max(-1 + 1e-7, min(1 - 1e-7, _dot(n, v)))
        return {"r": r, "v": v, "n": n, "kb": lo, "direction": direction,
                "cos_theta": cos_t, "q_bridge": self.oracle.queries - q0}

    def _update_ramb_pvre(self, normal: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """30e three-regime router with *native normal estimation* per routed update.

        The first normal passed into this method is the official TtBA-style estimate
        (DCT f=4 by default, sigma=3e-4, fixed probe norm).  It is used for the two
        same-radius routing gates because k_b and the 0.2 TtBA threshold are defined
        in the TtBA direction family.

          H: B(0.2) = non-adversarial -> k_b <= 0.2
             -> TtBA update using that same native TtBA normal.
          M: B(0.2) = adversarial and B(0.8) = non-adversarial
             -> TtBA_PVRE using that same native TtBA normal.
          L: B(0.8) = adversarial -> k_b >= 0.8
             -> re-estimate the normal with native CGBA probes (EngineConfig.sigma,
                normally 2e-4, unnormalised low-frequency DCT Gaussian probes), then
                run the CGBA update with the CGBA normal.

        Therefore the routed TtBA and PVR-E calculations match TtBA's normal
        estimator, while the routed CGBA calculation matches CGBA's normal estimator.
        The CGBA re-estimation costs additional victim queries and is included in the
        global query budget and in the ``normal`` query tag.
        """
        c = self.cfg
        q_update0 = self.oracle.queries
        xs = self.origin
        r_vec = xb - xs
        r = _norm(r_vec)
        v, n = _unit(r_vec), _unit(normal)
        if v is None or n is None or r <= 1e-12:
            return xb

        def direction(k: float) -> torch.Tensor:
            d = _unit(k * n + (1.0 - k) * v)
            return v if d is None else d

        def log_route(regime: str, winner: str, out: torch.Tensor, gate_q: int,
                      bridge_q: int = 0, mid_pvr: int = 0,
                      cgba_normal_q: int = 0, cgba_sign_share: float = float("nan"),
                      cgba_normal_fallback: int = 0) -> torch.Tensor:
            q_update = max(1, self.oracle.queries - q_update0)
            g_act = _norm(out - xs)
            self.diag["ra_regime"].append(regime)
            self.diag["ra_lambda"].append(float("nan"))
            self.diag["ra_candidates"].append(0)
            self.diag["ra_race_q"].append(0)
            self.diag["ra_pruned"].append(0)
            self.diag["ra_unresolved"].append(0)
            self.diag["ra_winner"].append(winner)
            self.diag["ra_eta"].append(max(0.0, r - g_act) / q_update)
            self.diag["ra_hist_c"].append(self.ra_c_ema)
            self.diag["ra_infeasible"].append(0)
            # refinement/update accounting excludes the route gates, bridge and any
            # second normal estimate; global by_tag still counts every victim query.
            self.diag["ra_refine_q"].append(max(0, q_update - gate_q - bridge_q - cgba_normal_q))
            self.diag["ra_update_q"].append(q_update)
            self.diag["ra_mid_pvr"].append(mid_pvr)
            self.diag["ra_gate_q"].append(gate_q)
            self.diag["ra_cgba_normal_q"].append(int(cgba_normal_q))
            self.diag["ra_cgba_sign_share"].append(float(cgba_sign_share))
            self.diag["ra_cgba_normal_fallback"].append(int(cgba_normal_fallback))
            self.ra_prev_regime = regime
            return out

        # Gate 1: k=0.2, using the native TtBA normal.
        q_gate0 = self.oracle.queries
        p02 = self._proj(xs + r * direction(c.escape_k))
        adv02 = self._adv(p02, "update")
        if not adv02:
            self.hyb_mode = "bridge"
            br = self._route_bridge_from_bracket(normal, xb, 0.0, c.escape_k,
                                                 c.ttba_bridge_tol, q_gate0)
            if br is None:
                return xb
            out = self._ttba_radial_step(br, xb, "ramb_route_ttba")
            gate_q = 1
            bridge_extra = max(0, br.get("q_bridge", 0) - gate_q)
            return log_route("H", "ttba", out, gate_q, bridge_extra, 0)

        # Gate 2: k=0.8 (or configured upper gate), still using the TtBA normal.
        high = float(c.ra_mid_pvr_high)
        p08 = self._proj(xs + r * direction(high))
        adv08 = self._adv(p08, "update")
        gate_q = self.oracle.queries - q_gate0

        if not adv08:
            # Native TtBA_PVRE normal: use the same TtBA estimate already computed.
            if self.ra_prev_regime != "M":
                self.hyb_mode = "bridge"
            needs_bridge = (self.hyb_mode == "bridge" or self.hyb_pool[1] <= 0.0)
            br = None
            bridge_extra = 0
            if needs_bridge:
                br = self._route_bridge_from_bracket(normal, xb, c.escape_k, high,
                                                     c.pvr_bridge_tol, q_gate0)
                if br is None:
                    return xb
                bridge_extra = max(0, br.get("q_bridge", 0) - gate_q)
            out = self._update_hybrid(normal, xb, pvr=True, bridge_override=br)
            return log_route("M", "pvre", out, gate_q, bridge_extra, 1)

        # L: re-estimate the normal exactly in CGBA probe style before the CGBA update.
        # The scheduled number of probes is the same q_t used by standalone CGBA at this
        # iteration.  Near the budget limit, use as many as can be afforded while preserving
        # the standard update reserve; if none can be afforded, fall back to the gate normal.
        self.hyb_mode = "bridge"
        q_cgba = min(int(self._q_normal), max(0, self.remaining - c.update_reserve))
        cgba_fallback = 0
        cgba_share = float("nan")
        if q_cgba > 0:
            cgba_n, cgba_share, q_cgba_done = self._estimate_cgba_normal(xb, q_cgba)
        else:
            cgba_n, q_cgba_done, cgba_fallback = normal, 0, 1
        out = self._update_cgba(cgba_n, xb)
        return log_route("L", "cgba", out, gate_q, 0, 0,
                         q_cgba_done, cgba_share, cgba_fallback)

    # ------------------------------------------ QS family (user's lgc_attack.py)
    def _attacker_ssim(self, z: torch.Tensor) -> float:
        """SSIM(x(z), x0). The attacker owns x0 and the generator, so this costs no query."""
        return _ssim_value(self.space.to_image(z), self.space.x0)

    def _qs_K(self, ssim: float) -> int:
        """lgc_attack._compute_QS_K ('gap_based'): K from the gap to the switch SSIM."""
        c = self.cfg
        gap = max(0.0, c.qs_switch_ssim - ssim)
        sched = sorted(c.qs_K_schedule, reverse=True)
        k = sched[-1][1]
        for thr, kk in sched:
            if gap >= thr:
                k = kk
                break
        return int(max(c.qs_K_min, min(c.qs_K_max, k)))

    def _qs_alpha_sched(self) -> float:
        """QS: alpha = 0 during the warm-up, then linear to qs_alpha_max."""
        c, i = self.cfg, self._it
        if c.qs_warmup <= 0 or i < c.qs_warmup:
            return 0.0
        return max(0.0, min(c.qs_alpha_max, c.qs_alpha_max * (i - c.qs_warmup + 1) / max(1, c.qs_warmup)))

    def _qsb_alpha(self) -> float:
        """QS_Bridge2 (_bridge2_alpha): alpha from the previous bridge span."""
        c, i, span = self.cfg, self._it, self._qsb_span
        if c.qsb_alpha_mode in ("curv", "inv"):
            if span is None:
                return c.qs_alpha_max * min(1.0, (i + 1) / max(1, c.qs_warmup))
            x = float(span) / float(c.qsb_span_ref)
            x = 1.0 - x if c.qsb_alpha_mode == "curv" else x
            return c.qs_alpha_max * max(0.0, min(1.0, x))
        if i < c.qs_warmup:
            return 0.0
        return c.qs_alpha_max * min(1.0, (i - c.qs_warmup + 1) / max(1, c.qs_warmup))

    def _qs_tangent(self, e1: torch.Tensor) -> Optional[torch.Tensor]:
        """The original scores probe i by |label_i - 0.5| / |P_t u_i| (a CLASS INDEX, not a decision);
        |P_t u_i| ~ sqrt(d) for every probe, so the pick is an arbitrary probe and its tangential part is
        an isotropic random tangent direction (its alignment with the normal is O(1 / sqrt(d))).
        It is reproduced as such (no query)."""
        g = self.axis_sampler.sample(1, self.rng_axis)
        return _unit(g - _dot(g, e1) * e1)

    @staticmethod
    def _qs_plane(e1: torch.Tensor, v: torch.Tensor, t: Optional[torch.Tensor], alpha: float):
        """lgc_attack._plane_basis: e1 = eta, e2 = unit((1 - alpha) e2_v + alpha t), orthogonal to e1."""
        e2v = _unit(v - _dot(v, e1) * e1)
        if e2v is None:
            return None
        if t is None or alpha <= 0.0:
            return e1, e2v
        t = _unit(t - _dot(t, e1) * e1)
        if t is None:
            return e1, e2v
        e2 = (1.0 - alpha) * e2v + alpha * t
        e2 = _unit(e2 - _dot(e2, e1) * e1)
        return (e1, e2v) if e2 is None else (e1, e2)

    def _qs_ray(self, d: torch.Tensor, R: float, tol: float) -> Optional[torch.Tensor]:
        """Original radial bisection on [0, R] along d (<= qs_radial_steps queries); None if no hit."""
        xs, lo, hi, cand = self.origin, 0.0, R, None
        for _ in range(self.cfg.qs_radial_steps):
            mid = 0.5 * (lo + hi)
            pt = self._proj(xs + mid * d)
            if self._adv(pt, "boundary"):
                cand, hi = pt, mid
            else:
                lo = mid
            if hi - lo < tol:
                break
        return cand

    def _qs_bin_search(self, adv: torch.Tensor) -> torch.Tensor:
        """Original bin_search(x_0, adv): >= 1 query, stops at |adv - cln| < tol or 100 queries. After
        _qs_ray it repeats the same ray (the candidate is already within tol of the boundary)."""
        cln = self.origin
        for _ in range(100):
            mid = self._proj(0.5 * (cln + adv))
            if self._adv(mid, "boundary"):
                adv = mid
            else:
                cln = mid
            if _norm(adv - cln) < self.cfg.tol:
                break
        return adv

    def _qs_extrapolate(self, best: torch.Tensor, v: torch.Tensor, eta: torch.Tensor, R: float) -> torch.Tensor:
        """Original tangent extrapolation: best + (0.02 + 0.06 i / (I - 1)) R e2_v, 1 query + bin_search."""
        tang = _unit(v - _dot(v, eta) * eta)
        if tang is None:
            self.diag["qs_extrap"].append(0)
            return best
        T = max(self.cfg.qs_extrap_iters - 1, 1)
        step = (0.02 + 0.06 * min(self._it, T) / T) * R
        ext = self._proj(best + step * tang)
        if self._adv(ext, "update"):
            ext = self._qs_bin_search(ext)
            if _norm(ext - self.origin) < _norm(best - self.origin):
                self.diag["qs_extrap"].append(1)
                return ext
        self.diag["qs_extrap"].append(0)
        return best

    def _qs_hide(self, q0: int) -> None:
        """Queries since q0 that the original code did not add to q_num, and when (for a reported-query axis)."""
        self.diag["qs_hidden"].append(self.oracle.queries - q0)
        self.diag["qs_hidden_at"].append(self.oracle.queries)

    def _qs_fallback(self, n: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        self.diag["qs_fallback"].append(1)
        return self._update_cgba_h(n, xb)

    def _update_qs(self, n: torch.Tensor, xb: torch.Tensor, ssim: Optional[float] = None) -> torch.Tensor:
        """FAITHFUL go_to_boundary_QS (every query counted)."""
        q0 = self.oracle.queries
        xs = self.origin
        R = _norm(xb - xs)
        v, eta = _unit(xb - xs), _unit(n)
        if v is None or eta is None or R <= 1e-12:
            return xb
        K = self._qs_K(self._attacker_ssim(xb) if ssim is None else ssim)
        alpha = self._qs_alpha_sched()
        t = self._qs_tangent(eta) if alpha > 0.0 else None
        pl = self._qs_plane(eta, v, t, alpha)
        if pl is None:
            return xb
        e1, e2 = pl
        best, best_r, hits = None, float("inf"), 0
        for j in range(K):
            phi = 0.5 * math.pi * j / (K - 1) if K > 1 else 0.0  # torch.linspace(0, pi/2, K)
            cand = self._qs_ray(math.cos(phi) * e1 + math.sin(phi) * e2, R, self.cfg.tol)
            if cand is None:
                continue
            hits += 1
            cand = self._qs_bin_search(cand)
            rr = _norm(cand - xs)
            if rr < best_r:
                best, best_r = cand, rr
        self.diag["qs_K"].append(K)
        self.diag["qs_alpha"].append(alpha)
        self.diag["qs_hit"].append(hits / K)
        if best is None:
            self._qs_hide(q0)  # the fan; LGC_H's own count was returned
            return self._qs_fallback(n, xb)
        self.diag["qs_fallback"].append(0)
        try:
            best = self._qs_extrapolate(best, v, eta, R)
        finally:
            self._qs_hide(q0)  # returned (best_adv, 0)
        self._log_geom("qs", R, v, xb, best, cos_theta=_dot(eta, v))
        return best

    def _update_qs_hyb(self, n: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """FAITHFUL go_to_boundary_Hybrid: QS while SSIM < qs_switch_ssim, LGC_H afterwards."""
        ssim = self._attacker_ssim(xb)
        if ssim >= self.cfg.qs_switch_ssim:
            self.diag["qs_stage2"].append(1)
            return self._update_cgba_h(n, xb)
        self.diag["qs_stage2"].append(0)
        return self._update_qs(n, xb, ssim=ssim)

    def _update_qsb2(self, n: torch.Tensor, xb: torch.Tensor, fixed: bool = False,
                     no_alpha: bool = False) -> torch.Tensor:
        """go_to_boundary_QS_Bridge2. fixed=False: FAITHFUL; fixed=True (qsb2_o): the bisection runs from the
        anchor TOWARD eta and phi* = anchor + ratio (flip - anchor); no_alpha=True (qsb2_oa): alpha = 0."""
        c, xs = self.cfg, self.origin
        q0 = self.oracle.queries
        R = _norm(xb - xs)
        v, eta = _unit(xb - xs), _unit(n)
        if v is None or eta is None or R <= 1e-12:
            return xb
        alpha = 0.0 if no_alpha else self._qsb_alpha()
        t = self._qs_tangent(eta) if alpha > 0.0 else None
        pl = self._qs_plane(eta, v, t, alpha)
        if pl is None:
            return xb
        e1, e2 = pl
        a1, a2 = _dot(v, e1), _dot(v, e2)
        anchor = 0.0 if math.hypot(a1, a2) < 1e-12 else max(-0.5 * math.pi, min(0.5 * math.pi, math.atan2(a2, a1)))

        def point(phi: float) -> torch.Tensor:
            return self._proj(xs + R * (math.cos(phi) * e1 + math.sin(phi) * e2))

        sgn = -1.0 if fixed else 1.0  # phi grows AWAY from e1 = eta; the adversarial arc lies toward eta
        lo, hi = anchor, anchor + sgn * 0.5 * math.pi
        self.diag["qs_alpha"].append(alpha)
        if not self._adv(point(lo), "update"):
            lo = anchor - sgn * 0.5 * math.pi
            if not self._adv(point(lo), "update"):
                self._qs_hide(q0)
                return self._qs_fallback(n, xb)
        tol = max(1e-4, c.qsb_tol * 0.5 * math.pi)
        for _ in range(c.qsb_steps):
            mid = 0.5 * (lo + hi)
            if self._adv(point(mid), "update"):
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                break
        flip = hi
        hit_end = sgn * (flip - anchor) >= 0.5 * math.pi - 0.5 * tol
        if fixed:
            phi_star = anchor + c.qsb_ratio * (flip - anchor)
            span = abs(flip - anchor)
        elif hit_end:
            phi_star, span = 0.5 * (lo + hi), abs(hi - lo)
        else:
            phi_star = lo + c.qsb_ratio * (flip - lo)  # lo has moved to the flip: ratio has no effect
            span = flip - anchor
        self._qsb_span = span
        self.diag["qsb_span_deg"].append(math.degrees(span))
        cand = self._qs_ray(math.cos(phi_star) * e1 + math.sin(phi_star) * e2, R, max(1e-5, c.tol))
        if cand is None:
            self._qs_hide(q0)
            return self._qs_fallback(n, xb)
        self.diag["qs_fallback"].append(0)
        try:
            best = self._qs_bin_search(cand)
            best = self._qs_extrapolate(best, v, eta, R)
        finally:
            self._qs_hide(q0)
        self._log_geom(("qsb2_oa" if no_alpha else "qsb2_o") if fixed else "qsb2", R, v, xb, best,
                       cos_theta=_dot(eta, v))
        return best

    def _qs_bridge(self, n: torch.Tensor, xb: torch.Tensor) -> Optional[dict]:
        """Corrected bridge: largest rotation psi from v toward eta with x_s + R d(psi) adversarial, on the
        radius-R sphere. Halving from pi/2 finds the scale (x_b itself, psi = 0, is adversarial), bisection
        refines psi_b to a RELATIVE tolerance (a fixed angular tolerance is ~50 % of a 2 deg late bridge)."""
        xs = self.origin
        R = _norm(xb - xs)
        v, eta = _unit(xb - xs), _unit(n)
        if v is None or eta is None or R <= 1e-12:
            return None
        ct = max(-1 + 1e-7, min(1 - 1e-7, _dot(eta, v)))
        w = _unit(eta - ct * v)
        if w is None:
            return None

        def direction(psi: float) -> torch.Tensor:
            return math.cos(psi) * v + math.sin(psi) * w

        def adv(psi: float) -> bool:
            return self._adv(self._proj(xs + R * direction(psi)), "update")

        q0 = self.oracle.queries
        # first trial: 1.5 x the previous bridge angle (the bridge shrinks slowly), else pi/2
        psi0 = 0.5 * math.pi if self._qs_prev_psib is None else min(0.5 * math.pi, 1.5 * self._qs_prev_psib)
        if adv(psi0):  # the arc reaches psi0: double until a non-adversarial angle (pi = -v is assumed clean)
            lo, hi = psi0, min(math.pi, 2.0 * psi0)
            while hi < math.pi and adv(hi):
                lo, hi = hi, min(math.pi, 2.0 * hi)
        else:  # halve until adversarial
            lo, hi, psi = 0.0, psi0, 0.5 * psi0
            while psi > 1e-6:
                if adv(psi):
                    lo = psi
                    break
                hi, psi = psi, 0.5 * psi
        if lo <= 0.0:
            self._qs_prev_psib = None
            return None
        while hi - lo > self.cfg.qsb_rel_tol * lo and self.remaining > 0:
            mid = 0.5 * (lo + hi)
            if adv(mid):
                lo = mid
            else:
                hi = mid
        self._qs_prev_psib = lo
        self.diag["qsb_span_deg"].append(math.degrees(lo))
        return {"R": R, "v": v, "w": w, "psi_b": lo, "direction": direction, "cos_theta": ct,
                "q_bridge": self.oracle.queries - q0}

    def _qs_radial_lin(self, br: dict, psi: float) -> Optional[Tuple[torch.Tensor, float, float]]:
        """Radial search along d(psi), bracketed at the linear-boundary prediction
        g_lin = R cos(psi_b / 2) / cos(psi - psi_b / 2) (= R cos(psi_b / 2) at the bisector)."""
        R, half = br["R"], 0.5 * br["psi_b"]
        den = math.cos(psi - half)
        g_pred = R * math.cos(half) / den if den > 1e-9 else R
        g_pred = min(max(g_pred, 0.05 * R), R * (1.0 - 1e-6))
        d0 = None
        if len(self.pvr_errs) >= 3:
            q90 = float(np.quantile(np.asarray(self.pvr_errs, dtype=float), 0.9))
            d0 = min(max(self.cfg.pvr_bracket_mult * q90, 2.0 * self.cfg.tol / max(g_pred, 1e-12)),
                     self.cfg.bracket_delta)
        out = self._bracket_radial(br["direction"](psi), g_pred, R, d0=d0)
        if out is None:
            return None
        cand, g_act = out
        self.pvr_errs.append(abs(g_pred - g_act) / max(g_act, 1e-12))
        return cand, g_act, g_pred

    def _update_qsb2_c(self, n: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Corrected QS_Bridge2: relative-tolerance bridge, psi* = ratio psi_b (0.5 = bisector, exact on a
        hyperplane), one bracketed radial search. No tangent mixing, no extrapolation, no duplicate search."""
        br = self._qs_bridge(n, xb)
        if br is None:
            return self._qs_fallback(n, xb)
        self.diag["qs_fallback"].append(0)
        psi = self.cfg.qsb_ratio * br["psi_b"]
        q0 = self.oracle.queries
        out = self._qs_radial_lin(br, psi)
        if out is None:
            return xb
        cand, _, g_pred = out
        self._log_geom("qsb2_c", br["R"], br["v"], xb, cand, g_pred=g_pred, cos_theta=br["cos_theta"],
                       q_radial=self.oracle.queries - q0)
        return cand

    def _update_qs_c(self, n: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        """Corrected QS: the same bridge, then K rays at psi_j = psi_b j / (K + 1) inside the measured arc
        (every ray hits within R on a flat or convex boundary), bracketed radial searches, best ray."""
        br = self._qs_bridge(n, xb)
        if br is None:
            return self._qs_fallback(n, xb)
        self.diag["qs_fallback"].append(0)
        K = self._qs_K(self._attacker_ssim(xb))
        self.diag["qs_K"].append(K)
        best, best_g, g_best_pred, hits = None, float("inf"), float("nan"), 0
        q0 = self.oracle.queries
        for j in range(1, K + 1):
            out = self._qs_radial_lin(br, br["psi_b"] * j / (K + 1))
            if out is None:
                continue
            hits += 1
            cand, g_act, g_pred = out
            if g_act < best_g:
                best, best_g, g_best_pred = cand, g_act, g_pred
        self.diag["qs_hit"].append(hits / K)
        if best is None:
            return xb
        self._log_geom("qs_c", br["R"], br["v"], xb, best, g_pred=g_best_pred, cos_theta=br["cos_theta"],
                       q_radial=self.oracle.queries - q0)
        return best

    # -------------------------------------------------------- Triangle Attack
    def _ta_axis(self, u1: torch.Tensor, prefer: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if prefer is not None:
            u2 = _unit(prefer - _dot(prefer, u1) * u1)
            if u2 is not None:
                return u2
        for _ in range(8):
            g = self.axis_sampler.sample(1, self.rng_axis)
            u2 = _unit(g - _dot(g, u1) * u1)
            if u2 is not None:
                return u2
        return None

    def _ta_step(self, x_adv: torch.Tensor, u2: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Port of official get_x_hat_in_2d (TA repo, attack_mask.py)."""
        c = self.cfg
        xo = self.origin
        diff = x_adv - xo
        d = _norm(diff)
        if d <= 1e-12:
            return x_adv, False
        u1 = diff / d
        upper, lower = math.pi / 2 + c.ta_half_range, math.pi / 2 - c.ta_half_range

        def cand(theta: float, flag: float) -> torch.Tensor:
            a = self.ta_alpha
            scale = d * math.sin(a + theta) / math.sin(a)
            return self._proj(xo + scale * (math.cos(theta) * u1 + flag * math.sin(theta) * u2))

        def dec() -> None:
            self.ta_alpha = max(lower, self.ta_alpha - c.ta_minus)

        def inc() -> None:
            self.ta_alpha = min(upper, self.ta_alpha + c.ta_plus)

        x_hat = x_adv
        theta = max(math.pi - 2 * self.ta_alpha, 0.0) + min(math.pi / 16, self.ta_alpha / 2)
        right = math.pi - self.ta_alpha
        x = cand(theta, 1.0)
        if self._adv(x, "update"):
            x_hat, left, flag = x, theta, 1.0
        else:
            dec()
            theta = max(theta, math.pi - 2 * self.ta_alpha + math.pi / 64)
            x = cand(theta, -1.0)
            if self._adv(x, "update"):
                x_hat, left, flag = x, theta, -1.0
            else:
                dec()
                return x_adv, False

        theta = 0.5 * (left + right)
        for _ in range(c.ta_max_iter):
            x = cand(theta, flag)
            if self._adv(x, "update"):
                inc()
                return x, True
            dec()
            theta = max(theta, math.pi - 2 * self.ta_alpha + math.pi / 64)
            flag = -flag
            x = cand(theta, flag)
            if self._adv(x, "update"):
                inc()
                return x, True
            dec()
            left = max(math.pi - 2 * self.ta_alpha, 0.0) + min(math.pi / 16, self.ta_alpha / 2)
            right = theta
            theta = 0.5 * (left + right)
        inc()
        return x_hat, True

    # ------------------------------------------------------------------ HSJA
    def _hsja_step(self, t: int, x: torch.Tensor):
        c = self.cfg
        xs = self.origin
        d_dim = self.space.dim
        theta_bs = c.hsja_gamma / (d_dim**1.5)
        dist = _norm(x - xs)
        if t == 1 or self.hsja_dist_post is None:
            delta = 0.1 * (1.0 if self.space.name == "pixel" else dist)
        else:
            delta = math.sqrt(d_dim) * theta_bs * self.hsja_dist_post
        num = min(int(c.hsja_init_evals * math.sqrt(t)), c.hsja_max_evals)
        num = min(num, self.remaining - c.update_reserve)
        if num <= 0:
            return _STOP
        shape = self.space.shape
        s0 = torch.zeros_like(x)
        s1 = torch.zeros_like(x)
        n_adv, done = 0, 0
        while done < num:
            b = min(c.query_batch, num - done)
            rv = self.rng_hsja.randn(b, *shape)
            rv = rv / rv.flatten(1).norm(dim=1).clamp_min(1e-12).view(b, *([1] * len(shape)))
            pert = self._proj(x + delta * rv)
            rv = (pert - x) / delta
            s = self.oracle.signs(pert, "normal")
            s0 += rv.sum(0, keepdim=True)
            s1 += (s.view(b, *([1] * len(shape))) * rv).sum(0, keepdim=True)
            n_adv += int((s > 0).sum().item())
            done += b
        if n_adv == num:
            g = s0 / num
        elif n_adv == 0:
            g = -s0 / num
        else:
            mean_f = (2.0 * n_adv - num) / num
            g = s1 / num - mean_f * s0 / num
        g = _unit(g)
        if g is None:
            return x
        eps = dist / math.sqrt(t)
        cand = None
        for _ in range(c.hsja_max_halvings):
            if self.remaining <= 0:
                break
            y = self._proj(x + eps * g)
            if self._adv(y, "update"):
                cand = y
                break
            eps /= 2.0
        if cand is None:
            return x
        self.hsja_dist_post = _norm(cand - xs)
        lo, hi = 0.0, 1.0
        while (hi - lo) / theta_bs > 1 and self.remaining > 0:
            mid = 0.5 * (lo + hi)
            if self._adv(self._proj((1 - mid) * xs + mid * cand), "boundary"):
                hi = mid
            else:
                lo = mid
        return cand if hi >= 1.0 else self._proj((1 - hi) * xs + hi * cand)

    # ---------------------------------------------------------------- step
    def _n_probes(self, it: int, current: torch.Tensor) -> int:
        """Probe count of iteration it: n0 sqrt(t+1), or N* (optimal_n) once d* is pooled."""
        q = int(self.n0 * math.sqrt(it + self.it_offset + 1))
        if self.spec.optimal_n and self.hyb_pool[1] > 0:
            r = _norm(current - self.origin)
            if r > 1e-12:
                ca = (self.hyb_pool[0] / self.hyb_pool[1]) / r
                overhead = float(np.mean(self.overheads)) if self.overheads else 26.0
                q = optimal_probe_count(ca, self.d_eff, overhead, self.cfg.nstar_min, self.cfg.nstar_max)
        return q

    def _dispatch(self, u: str, n: torch.Tensor, current: torch.Tensor):
        if u == "cgba":
            return self._update_cgba(n, current)
        if u == "cgba_h":
            return self._update_cgba_h(n, current)
        if u == "ttba":
            return self._update_ttba(n, current)
        if u == "ttba_mid":
            return self._update_mid(n, current, gamma_arc=False)
        if u == "garc":
            return self._update_mid(n, current, gamma_arc=True)
        if u == "hybrid":
            return self._update_hybrid(n, current, pvr=False)
        if u == "pvr":
            return self._update_hybrid(n, current, pvr=True)
        if u == "pvr_ua":
            return self._update_pvre_ua(n, current)
        if u == "pvr_gc":
            return self._update_pvre_gc(n, current)
        if u == "pvr_aor":
            return self._update_pvre_aor(n, current)
        if u == "pvr_aou":
            return self._update_pvre_aou(n, current)
        if u == "pvr_caou":
            return self._update_pvre_caou(n, current)
        if u == "pvr_gte":
            return self._update_pvre_gte(n, current)
        if u == "pvr_pb":
            return self._update_pvre_pb(n, current)
        if u == "ksqp":
            return self._update_ksqp(n, current)
        if u == "ramb_pvre":
            return self._update_ramb_pvre(n, current)
        if u == "qs":
            return self._update_qs(n, current)
        if u == "qs_hyb":
            return self._update_qs_hyb(n, current)
        if u == "qsb2":
            return self._update_qsb2(n, current, fixed=False)
        if u == "qsb2_o":
            return self._update_qsb2(n, current, fixed=True)
        if u == "qsb2_oa":
            return self._update_qsb2(n, current, fixed=True, no_alpha=True)
        if u == "qsb2_c":
            return self._update_qsb2_c(n, current)
        if u == "qs_c":
            return self._update_qs_c(n, current)
        u1 = _unit(current - self.origin)  # ngta
        if u1 is None:
            return current
        u2 = self._ta_axis(u1, prefer=n)
        if u2 is None:
            return current
        x_hat, changed = self._ta_step(current, u2)
        self.diag["ta_alpha"].append(self.ta_alpha)
        if not changed or x_hat is current:
            return current
        res = self._radial(x_hat, verified=True)
        return x_hat if res is None else res

    def _step(self, it: int, current: torch.Tensor):
        u = self.spec.update
        self._it = it
        if u == "mapvre":
            return self._step_mapvre(it, current)
        if u in PROBE_UPDATES:
            q = min(self._n_probes(it, current), self.remaining - self.cfg.update_reserve)
            if q <= 0:
                return _STOP
            self._q_normal = q
            if u == "pvr_aor" and self.pvr_aor_planar_left > 0 and self.spec.aor_native_planar:
                # Model order was selected on PREVIOUS verified evidence.  Estimate the
                # native CGBA normal directly at the next iteration; do not first spend
                # a TtBA normal and then re-probe.
                n, share, q_done = self._estimate_cgba_normal(current, q)
                self._q_normal = q_done
                self._sign_share = share
                self.diag["sign_share"].append(share)
                self.diag["n_probes"].append(q_done)
                self.diag["pvr_aor_native_normal_q"].append(q_done)
                self.diag["pvr_aor_native_sign_share"].append(share)
                self._last_aor_normal_split_deg = float("nan")
                self.diag["pvr_aor_normal_split_deg"].append(float("nan"))
            elif u == "pvr_aor":
                n = self._estimate_normal_aor(current, q)
                self.diag["pvr_aor_native_normal_q"].append(0)
                self.diag["pvr_aor_native_sign_share"].append(float("nan"))
            else:
                n = self._estimate_normal(current, q)
            v = _unit(current - self.origin)
            if v is not None:
                self.diag["cos_v_n"].append(_dot(n, v))
            q_after_normal = self.oracle.queries
            out = self._dispatch(u, n, current)
            self.overheads.append(self.oracle.queries - q_after_normal)  # bridge + radial queries
            return out
        if u == "ta":
            if self.remaining < 2:
                return _STOP
            u1 = _unit(current - self.origin)
            if u1 is None:
                return current
            u2 = self._ta_axis(u1)
            if u2 is None:
                return current
            x_hat, _ = self._ta_step(current, u2)
            self.diag["ta_alpha"].append(self.ta_alpha)
            return x_hat
        if u == "hsja":
            if self.remaining <= self.cfg.update_reserve:
                return _STOP
            return self._hsja_step(it + 1, current)
        raise ValueError(u)

    # ----------------------------------------------------------- checkpoints
    def _fill_crossed(self, q_before: int, q_after: int, old: torch.Tensor, old_q: int) -> None:
        """An update that straddles a budget leaves the PRE-update state valid there."""
        for cp in self.cfg.checkpoints:
            if cp not in self.checkpoint_states and q_before <= cp < q_after:
                self.checkpoint_states[cp] = (old.detach().clone(), old_q)

    # ------------------------------------------------------------------- run
    def run(self) -> AttackResult:
        t0 = time.time()
        try:
            init = self._initial_adversarial()
            current = self._radial(init, verified=True)
            if current is None:
                current = init
        except (InitFailed, QueryBudgetExhausted) as err:
            if self.cfg.verbose:
                print(f"      [{self.spec.name}] init failed: {err}")
            return AttackResult(self.spec.name, "init_failed", {}, None, self.oracle.queries,
                                self.oracle.queries, dict(self.oracle.by_tag), 0, time.time() - t0)
        self.hsja_dist_post = _norm(init - self.origin)
        state_q = self.oracle.queries
        cur_obj = self.objective(current)
        if self.cfg.save_trace:
            self.trace.append((state_q, cur_obj))

        it = 0
        while it < self.cfg.max_iterations and self.remaining > 0:
            q_before = self.oracle.queries
            old, old_q = current, state_q
            stop = False
            try:
                cand = self._step(it, current)
                if cand is _STOP:
                    stop, cand = True, current
            except QueryBudgetExhausted:
                cand = current
            self._fill_crossed(q_before, self.oracle.queries, old, old_q)
            if self._obj_dirty:  # build 37: the acceptance objective switched (MAL gate opened) -> re-measure
                cur_obj = self.objective(current)
                self._obj_dirty = False
            accepted = False
            if cand is not current:
                c_obj = self.objective(cand)
                if c_obj < cur_obj - 1e-12:
                    current, cur_obj, state_q, accepted = cand.detach(), c_obj, self.oracle.queries, True
            self.diag["accepted"].append(int(accepted))
            for cp in self.cfg.checkpoints:
                if cp not in self.checkpoint_states and cp == self.oracle.queries:
                    self.checkpoint_states[cp] = (current.detach().clone(), state_q)
            if accepted and self.cfg.save_trace:
                self.trace.append((state_q, cur_obj))
            if self.stop_rule is not None and not stop and self.stop_rule(self.oracle.queries, current):
                self.stopped_by_rule = True
                stop = True
            if self.cfg.verbose and (it < 3 or (it + 1) % 25 == 0):
                print(f"      [{self.spec.name} it {it:04d}] Q={self.oracle.queries:5d} obj={cur_obj:.5f}")
            it += 1
            if stop or self.oracle.queries == q_before:
                break

        if self.stopped_by_rule:
            self.diag["stopped_at"] = [self.oracle.queries]
        if self.tracker is not None:
            self.diag["track_rho"] = [self.tracker.rho_hat()]
            self.diag["track_gain"] = [self.tracker.gain_c2()]
            self.diag["track_used"] = [self.tracker.used_pct()]
        # A state reached with <= Q queries is valid for every LATER checkpoint;
        # never back-fill a checkpoint that was passed before a state existed.
        for cp in self.cfg.checkpoints:
            if cp not in self.checkpoint_states and self.oracle.queries < cp <= self.cfg.max_queries:
                self.checkpoint_states[cp] = (current.detach().clone(), state_q)

        return AttackResult(
            method=self.spec.name,
            status="ok",
            checkpoint_states=self.checkpoint_states,
            final_state=current.detach(),
            final_state_query=state_q,
            queries=self.oracle.queries,
            by_tag=dict(self.oracle.by_tag),
            iterations=it,
            wall_time=time.time() - t0,
            trace=self.trace,
            diag=self.diag,
            geom=self.geom,
        )
