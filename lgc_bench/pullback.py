"""Pull-back PVR-E (build 33b): the PVR-E v7 update (gate - Thales - escape) carried out in the
pixel-isometric coordinates of the latent search plane of LGC.

Geometry (claude/LGC_RC_PVRE_design_2026-10-03.md, section 2)
-----------------------------------------------------------
Phi(xi) is the RAG map with the residual coordinate, xi = (z, u):
    Phi(z, u) = clip(x0 + G(z) - G(z_s) + (u / lambda) * Delta_r),   Phi(xi_s) = x0.
At the current boundary point xi_b, with the hard-label latent normal estimate n_hat, the search
plane is span(v, w):
    v = unit(xi_b - xi_s),   w = unit(n_hat - (n_hat . v) v).
Let J = dPhi/dxi at xi_b (central finite differences, 0 oracle queries), A = [J v, J w],
e = Phi(xi_b) - x0 and the Gram matrix A^T A = R^T R (Cholesky, R upper triangular). To first
order in the decoder,
    D^2(xi_b + a v + b w) = ||Phi(xi_b + a v + b w) - x0||^2 ~= ||e_perp||^2 + ||y||^2,
    y = y_b + R [a, b]^T,   y_b = R^{-T} A^T e,   ||e_perp||^2 = ||e||^2 - ||y_b||^2.
So in y the in-plane problem is the Euclidean pixel problem, with origin y = 0 and current point
y_b. Every PVR-E operation is therefore the 2-D pixel-space operation, mapped back by
    xi(y) = xi_b + [v w] R^{-1} (y - y_b).
The boundary normal transforms covariantly: n_y = unit(R^{-T} (n_hat . v, n_hat . w)).

Special cases:
- Phi = identity (pixel space): R = I and y_b = (r, 0), so the update is exactly PVR-E v7 (pvr_gte).
- R proportional to I with the pivot at xi_s, and the gate always flat: the update is the CGBA
  (Thales) step, i.e. the original LGC update.

Build 34b: the official CGBA / CGBA-H / TtBA updates in the chart (PBConfig.rule)
- rule "cgba":   CGBA (Reza et al., ICCV 2023): sweep psi_i = pi/2 (1 - 2^-i) on the Thales circle of
                 [0, y_b], then the semicircular boundary search (BSSP).  = GeometricAttack._update_cgba.
- rule "cgba_h": CGBA-H: sweep psi_i = theta / 2^i (theta = angle(n_y, y_b)) until adversarial, then radial
                 bisection.  = _update_cgba_h.
- rule "ttba":   TtBA (Wang et al., ICML 2025): bridge bisection k in [0, 1] to tolerance 1e-3, the 2/3 (0.9)
                 rule k(k_b), radial bisection along unit(k n + (1-k) v).  = _update_ttba.
- rule "pvre":   PVR-E v7 (build 33b, default).
Why the chart matters for these rules: every one of them is a construction in the EUCLIDEAN geometry of the
search plane (the Thales circle over [x0, x_b], the angle bisection, the k-mixture of unit vectors and its
tuned constants 2/3, 0.9, 0.1, 0.2).  Executed in latent coordinates they use the latent metric, which differs
from the pixel metric by the plane's pull-back Gram matrix A^T A = R^T R.  In y = y_b + R [a; b] the plane is
pixel-isometric to first order, so the constructions (and TtBA's pixel-tuned constants) apply as designed, at
zero oracle cost.  selftest_34b.py checks each rule against a line-by-line transcription of the official code.

Verify step (0 queries):
- Before use, the chart is checked against the true decoder distance at three in-plane test points
  (chart_test_points).
- If the first-order model is off by more than pb_verify_tol, the iteration uses the
  latent-Euclidean chart instead. This happens early in a run, when the latent point sits in a
  saturated decoder region and the in-plane floor dominates.

Diagnostics (0 queries):
- cond(R): metric anisotropy.
- The anchor mismatch. The latent pivot xi_s maps to y_s = y_b - r_xi R e1 and is generally
  != 0. We report ||y_s|| / ||y_b|| and the angle between R e1 and y_b.
- The floor share ||e_perp|| / ||e||: the in-plane manifold floor.

This module is torch-free so the 2-D core can be unit-tested anywhere (selftest_pvre_33b.py,
sim/sim_pullback.py). The engine (lgc_bench/engine.py, update "pvr_pb") supplies the Gram
scalars and an oracle callback.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple

import numpy as np

Vec = np.ndarray  # shape (2,), float64


def _unit2(x: Vec) -> Optional[Vec]:
    n = float(math.hypot(float(x[0]), float(x[1])))
    return None if n <= 1e-300 else np.asarray(x, dtype=float) / n


def _angle(a: Vec, b: Vec) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-300 or nb <= 1e-300:
        return float("nan")
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb))))


# --------------------------------------------------------------------------- frame
@dataclass
class Frame2D:
    """Pixel-isometric chart of the latent search plane (all quantities 0-query)."""

    R: np.ndarray  # (2, 2) upper triangular, A^T A = R^T R
    Rinv: np.ndarray  # (2, 2)
    y_b: Vec  # current point in y (pixel units)
    floor2: float  # ||e_perp||^2
    e_norm: float  # ||e|| = D(xi_b)
    cond: float  # cond(R)
    anchor_ratio: float  # ||y_s|| / ||y_b||, y_s = image of the latent pivot xi_s
    anchor_deg: float  # angle(R e1, y_b) in degrees
    floor_share: float  # ||e_perp|| / ||e||

    def ab(self, y: Vec) -> Tuple[float, float]:
        """Plane coordinates (a, b): xi(y) = xi_b + a v + b w."""
        d = self.Rinv @ (np.asarray(y, dtype=float) - self.y_b)
        return float(d[0]), float(d[1])

    def d_lin(self, y: Vec) -> float:
        """First-order pixel distance D(xi(y)) ~= sqrt(||e_perp||^2 + ||y||^2)."""
        return math.sqrt(max(0.0, self.floor2) + float(np.dot(y, y)))


def frame_from_gram(g11: float, g12: float, g22: float, ae1: float, ae2: float, e2: float,
                    r_xi: float, cond_max: float = 1e3) -> Optional[Frame2D]:
    """Build the chart from the 6 scalars ||Jv||^2, Jv.Jw, ||Jw||^2, Jv.e, Jw.e, ||e||^2.

    Returns None when the chart is unusable:
    - the Gram matrix is not positive definite;
    - cond(R) > cond_max;
    - ||y_b|| ~ 0.
    The caller then falls back to the latent-Euclidean update."""
    G = np.array([[g11, g12], [g12, g22]], dtype=float)
    if not np.all(np.isfinite(G)) or g11 <= 0.0 or g22 <= 0.0:
        return None
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        return None
    R = L.T  # upper triangular, positive diagonal
    sv = np.linalg.svd(R, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))
    if not math.isfinite(cond) or cond > cond_max:
        return None
    Rinv = np.linalg.inv(R)
    y_b = np.linalg.solve(R.T, np.array([ae1, ae2], dtype=float))  # R^{-T} A^T e
    ry = float(np.linalg.norm(y_b))
    if not math.isfinite(ry) or ry <= 1e-12:
        return None
    floor2 = max(0.0, float(e2) - ry * ry)
    e_norm = math.sqrt(max(float(e2), 0.0))
    y_s = y_b - float(r_xi) * R[:, 0]
    return Frame2D(R=R, Rinv=Rinv, y_b=y_b, floor2=floor2, e_norm=e_norm, cond=cond,
                   anchor_ratio=float(np.linalg.norm(y_s)) / ry,
                   anchor_deg=math.degrees(_angle(R[:, 0], y_b)),
                   floor_share=math.sqrt(floor2) / max(e_norm, 1e-300))


def normal_in_chart(frame: Frame2D, nu: Vec) -> Optional[Vec]:
    """Covariant transform of the in-plane latent normal nu = (n.v, n.w): n_y = unit(R^{-T} nu)."""
    return _unit2(frame.Rinv.T @ np.asarray(nu, dtype=float))


def chart_test_points(frame: Frame2D, n_y: Vec) -> list:
    """Points where the chart is verified before use (decoder-side, 0 oracle queries).

    They cover the region the PVR-E operations visit: the radial midpoint 0.5 y_b, and the Thales
    points at psi = pi/4 and pi/3 towards the normal (the gate, bridge and Thales queries live on or
    inside the circle |y| = |y_b|)."""
    r = float(np.linalg.norm(frame.y_b))
    v = frame.y_b / r
    ct = max(-1 + 1e-7, min(1 - 1e-7, float(np.dot(n_y, v))))
    w = _unit2(np.asarray(n_y, dtype=float) - ct * v)
    pts = [0.5 * frame.y_b]
    if w is not None:
        for psi in (0.25 * math.pi, math.pi / 3.0):
            pts.append(r * math.cos(psi) * (math.cos(psi) * v + math.sin(psi) * w))
    return pts


def chart_verify_error(frame: Frame2D, pts: list, d_true: list) -> float:
    """max_i |D_lin(y_i) - D(xi(y_i))| / D(xi(y_i)); D(xi(y)) is computed by the attacker (decoder)."""
    err = 0.0
    for y, d in zip(pts, d_true):
        err = max(err, abs(frame.d_lin(y) - float(d)) / max(float(d), 1e-12))
    return err


# ---------------------------------------------------------------------- 2-D PVR-E
@dataclass
class PBConfig:
    """Hyper-parameters; the engine fills them from EngineConfig / MethodSpec exactly as pvr_gte."""

    tol: float = 1e-4  # radial / BSSP tolerance in y units (= pixel L2)
    bracket_delta: float = 0.02
    escape_k: float = 0.2  # gate k (TtBA k_hat)
    narrow_tol: float = 1e-3  # bridge tolerance inside the certified [0, 0.2]
    bridge_tol: float = 1e-2  # bridge tolerance inside the certified [0.2, 1]
    eps_tol: float = 1.0
    eps_window: int = 3
    thales_steps: int = 4
    thales_mode: str = "flat"  # flat | off | always
    # build 33c: order of the two regime tests.  "gate_first" (v7): the k = escape_k gate query decides
    # narrow (N) before flatness is consulted.  "flat_first": if the eps history says the boundary is
    # locally flat (median |eps| <= eps_tol, < thales_steps Thales steps in a row) take the Thales (CGBA)
    # step directly, without the gate query; otherwise run the v7 logic unchanged.
    gate_order: str = "gate_first"
    eps_bracket: bool = False  # A4
    bracket_mult: float = 3.0
    err_window: int = 10
    ttba_low: float = 0.10
    ttba_high: float = 0.20
    ttba_jump: float = 0.9
    ttba_dive: float = 2.0 / 3.0
    # build 34b: which update the chart runs ("pvre" | "cgba" | "cgba_h" | "ttba"), and TtBA's own bridge
    # tolerance (official delta = 1e-3; PVR-E's B regime keeps bridge_tol)
    rule: str = "pvre"
    ttba_bridge_tol: float = 1e-3
    # CGBA / CGBA-H sweep length: 39 queries as in the official code (num_calls = 1 .. 39, then "return x_b");
    # GeometricAttack._update_cgba / _update_cgba_h use 40 (one extra query only when the sweep fails)
    cgba_sweep: int = 39


@dataclass
class PBState:
    """Cross-iteration state (eps history, Thales run length, radius-prediction errors)."""

    eps: Deque[float] = field(default_factory=lambda: deque(maxlen=3))
    errs: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    thales_run: int = 0

    @classmethod
    def from_cfg(cls, cfg: PBConfig) -> "PBState":
        return cls(eps=deque(maxlen=max(1, cfg.eps_window)), errs=deque(maxlen=max(1, cfg.err_window)))


@dataclass
class PBResult:
    y: Optional[Vec]  # verified adversarial new point in y (None: no move)
    choice: str  # N (narrow escape) | T (Thales) | B (wide bridge) | F (TtBA fallback) | X (degenerate)
    #              build 34b rules: C (CGBA) | H (CGBA-H) | D (TtBA two-third bridge, "dive" rule)
    eps: float = float("nan")
    kb: float = float("nan")


class PullbackPVRE2D:
    """PVR-E v7 (or, build 34b, the official CGBA / CGBA-H / TtBA update; PBConfig.rule) in a 2-D chart. adv(y, tag) issues ONE oracle query at xi(y) and returns True if
    adversarial. tag is "update" for gate, bridge and Thales points, and "boundary" for radial
    points, matching the engine's accounting. remaining() returns the oracle budget left.

    Mirrors GeometricAttack._update_pvre_gte with gte_model=False, operation for operation:
    - gate at k = escape_k;
    - N: bridge in [0, k_g] + TtBA k rule + bracketed radial (A4 eps seed);
    - T: CGBA sweep + BSSP on the Thales circle, when the median |eps| <= eps_tol;
    - B: bridge in [k_g, 1] + 2/3 rule + bracketed radial;
    - F: official TtBA radial step when no bridge is inside.
    """

    def __init__(self, cfg: PBConfig, state: PBState, adv: Callable[[Vec, str], bool],
                 remaining: Callable[[], int]):
        self.c, self.st, self.adv, self.remaining = cfg, state, adv, remaining

    # ------------------------------------------------------------ helpers
    def _ttba_k(self, kb: float) -> float:
        c = self.c
        if kb <= c.ttba_low:
            return c.ttba_jump * kb
        if kb >= c.ttba_high:
            return c.ttba_dive * kb
        slope = (c.ttba_dive * c.ttba_high - c.ttba_jump * c.ttba_low) / (c.ttba_high - c.ttba_low)
        return slope * (kb - c.ttba_low) + c.ttba_jump * c.ttba_low

    @staticmethod
    def _dir_fn(n: Vec, v: Vec) -> Callable[[float], Vec]:
        def direction(k: float) -> Vec:
            d = _unit2(k * n + (1.0 - k) * v)
            return v if d is None else d
        return direction

    def _bridge(self, r: float, direction: Callable[[float], Vec], lo: float, hi: float,
                tol: float) -> float:
        """Largest adversarial k inside the gate-certified [lo, hi] at radius r (bisection)."""
        lo, hi = float(max(0.0, min(1.0, lo))), float(max(lo, min(1.0, hi)))
        while hi - lo > tol and self.remaining() > 0:
            mid = 0.5 * (lo + hi)
            if self.adv(r * direction(mid), "update"):
                lo = mid
            else:
                hi = mid
        return lo

    def _bracket_radial(self, d: Vec, g_pred: float, g_cap: float,
                        d0: Optional[float] = None) -> Optional[Tuple[Vec, float]]:
        """= GeometricAttack._bracket_radial along the y-ray t d from the chart origin."""
        c = self.c
        g_pred = min(max(g_pred, 1e-12), g_cap)
        step = c.bracket_delta if d0 is None else max(float(d0), 1e-9)
        if self.adv(g_pred * d, "boundary"):
            hi = g_pred
            lo = hi * (1.0 - step)
            while lo > 0.0 and self.adv(lo * d, "boundary"):
                hi, step = lo, 2.0 * step
                lo = hi * (1.0 - step) if step < 1.0 else 0.0
            lo = max(lo, 0.0)
        else:
            lo = g_pred
            if lo >= g_cap:
                return None
            hi = min(g_pred * (1.0 + step), g_cap)
            while not self.adv(hi * d, "boundary"):
                if hi >= g_cap:
                    return None
                lo, step = hi, 2.0 * step
                hi = min(hi * (1.0 + step), g_cap)
        while hi - lo > c.tol and self.remaining() > 0:
            m = 0.5 * (lo + hi)
            if self.adv(m * d, "boundary"):
                hi = m
            else:
                lo = m
        return hi * d, hi

    def _radial_full(self, end: Vec, verified: bool) -> Optional[Vec]:
        """= GeometricAttack._radial: bisection on [0, end]; returns a verified adversarial point."""
        lo, hi, hi_ok = np.zeros(2), np.asarray(end, dtype=float), verified
        for _ in range(100):
            if self.remaining() <= 0 or float(np.linalg.norm(hi - lo)) <= self.c.tol:
                break
            mid = 0.5 * (lo + hi)
            if self.adv(mid, "boundary"):
                hi, hi_ok = mid, True
            else:
                lo = mid
        if not hi_ok:
            if self.remaining() <= 0 or not self.adv(hi, "boundary"):
                return None
        return hi

    def _g_lin(self, r: float, v: Vec, direction: Callable[[float], Vec], kb: float,
               k: float) -> float:
        d = direction(k)
        psi_k = _angle(d, v)
        half_b = 0.5 * _angle(direction(kb), v)
        den = math.cos(psi_k - half_b)
        g = r * math.cos(half_b) / den if den > 1e-9 else r
        return min(max(g, 0.05 * r), r * (1.0 - 1e-6))

    def _eps_seed(self, r: float, v: Vec, direction: Callable[[float], Vec],
                  kb: float) -> Optional[Tuple[float, float]]:
        """A4: eps-predicted bracket (g_hat, half-width), = _pvr_gte_eps_seed."""
        c = self.c
        if not c.eps_bracket or len(self.st.eps) < max(1, c.eps_window) or kb <= 0.0:
            return None
        eps_hat = float(np.median(np.asarray(self.st.eps, dtype=float)))
        if not math.isfinite(eps_hat):
            return None
        k = min(1.0, max(0.0, self._ttba_k(kb)))
        g_lin = self._g_lin(r, v, direction, kb, k)
        g_hat = min(max(g_lin - eps_hat * (r - g_lin), 0.05 * r), r * (1.0 - 1e-6))
        hw = max(2.0 * c.tol / g_lin, 0.3 * abs(eps_hat) * (r - g_lin) / g_lin)
        return g_hat, hw

    def _escape(self, r: float, v: Vec, direction: Callable[[float], Vec], kb: float,
                seed: Optional[Tuple[float, float]]) -> Optional[Vec]:
        """= _escape_step: TtBA k(k_b), radial search bracketed at g_lin (or at the eps seed)."""
        if kb <= 0.0:
            return None
        c = self.c
        k = min(1.0, max(0.0, self._ttba_k(kb)))
        d = direction(k)
        g_pred = self._g_lin(r, v, direction, kb, k)
        d0 = None
        if seed is not None:
            g_pred, d0 = seed
        elif len(self.st.errs) >= 3:
            q90 = float(np.quantile(np.asarray(self.st.errs, dtype=float), 0.9))
            d0 = min(max(c.bracket_mult * q90, 2.0 * c.tol / max(g_pred, 1e-12)), c.bracket_delta)
        out = self._bracket_radial(d, g_pred, r, d0=d0)
        if out is None:
            return None
        y, g_act = out
        self.st.errs.append(abs(g_pred - g_act) / max(g_act, 1e-12))
        return y

    def _eps(self, r: float, v: Vec, direction: Callable[[float], Vec], kb: float,
             y: Optional[Vec]) -> float:
        """= _pvr_gte_eps: flatness residual (g_lin - g) / (r - g_lin) of a measured step."""
        if y is None or kb <= 0.0:
            return float("nan")
        g = float(np.linalg.norm(y))
        if not (0.0 < g < r):
            return float("nan")
        psi = _angle(y, v)
        a = 0.5 * _angle(direction(kb), v)
        den = math.cos(psi - a)
        if den <= 1e-9:
            return float("nan")
        g_lin = r * math.cos(a) / den
        if r - g_lin <= 1e-9 * r:
            return float("nan")
        return (g_lin - g) / (r - g_lin)

    def _flat(self) -> bool:
        if len(self.st.eps) < max(1, self.c.eps_window):
            return False
        return abs(float(np.median(np.asarray(self.st.eps, dtype=float)))) <= self.c.eps_tol

    def thales(self, y_b: Vec, n: Vec) -> Optional[Vec]:
        """= _update_cgba in the chart: sweep psi_i = pi/2 (1 - 2^-i), then BSSP on the Thales circle.

        If every sweep point is adversarial the official code keeps x_b; this returns the last (verified
        adversarial, closer) sweep point instead, as GeometricAttack._update_cgba always has."""
        r = float(np.linalg.norm(y_b))
        v = y_b / r
        ct = max(-1 + 1e-7, min(1 - 1e-7, float(np.dot(n, v))))
        w = _unit2(n - ct * v)
        if w is None:
            return None
        clean, last_adv = None, None
        n_sweep = 40 if self.c.rule == "pvre" else max(1, int(self.c.cgba_sweep))  # pvre: frozen 33b behaviour
        for i in range(1, n_sweep + 1):
            if self.remaining() <= 0:
                break
            psi = 0.5 * math.pi * (1.0 - 2.0 ** (-i))
            zeta = math.cos(psi) * v + math.sin(psi) * w
            cand = (r * math.cos(psi)) * zeta
            if not self.adv(cand, "update"):
                clean = cand
                break
            last_adv = cand
        if clean is None:
            return last_adv
        clean_dir = _unit2(clean)
        if clean_dir is None:
            return None
        adv_dir, advp, moved = v, np.asarray(y_b, dtype=float), False
        for _ in range(100):
            if self.remaining() <= 0 or float(np.linalg.norm(advp - clean)) <= self.c.tol:
                break
            mid_dir = _unit2(adv_dir + clean_dir)
            if mid_dir is None:
                break
            mid = (r * float(np.dot(mid_dir, v))) * mid_dir
            if self.adv(mid, "update"):
                adv_dir, advp, moved = mid_dir, mid, True
            else:
                clean_dir, clean = mid_dir, mid
        return advp if moved else None

    def cgba_h(self, y_b: Vec, n: Vec) -> Optional[Vec]:
        """= _update_cgba_h in the chart: sweep psi_i = theta / 2^i towards v until adversarial (Thales points
        r cos(psi) unit(cos psi v + sin psi w)), then radial bisection from the origin (official bin_search)."""
        r = float(np.linalg.norm(y_b))
        v = y_b / r
        ct = max(-1 + 1e-7, min(1 - 1e-7, float(np.dot(n, v))))
        w = _unit2(n - ct * v)
        if w is None:
            return None
        theta = math.acos(ct)
        cand = None
        for i in range(1, max(1, int(self.c.cgba_sweep)) + 1):
            if self.remaining() <= 0:
                break
            psi = theta / (2.0 ** i)
            q = (r * math.cos(psi)) * (math.cos(psi) * v + math.sin(psi) * w)
            if self.adv(q, "update"):
                cand = q
                break
        if cand is None:
            return None
        res = self._radial_full(cand, verified=True)
        return cand if res is None else res

    def ttba(self, y_b: Vec, n: Vec) -> Tuple[Optional[Vec], float]:
        """= _update_ttba in the chart (official TtBA.TtBA): k_b by bisection on [0, 1] to ttba_bridge_tol at
        radius r, k = rule(k_b) (0.9 k_b below 0.1, 2/3 k_b above 0.2, linear in between), radial bisection
        on [0, r d(k)].  Returns (point or None, k_b)."""
        r = float(np.linalg.norm(y_b))
        v = y_b / r
        direction = self._dir_fn(n, v)
        kb = self._bridge(r, direction, 0.0, 1.0, self.c.ttba_bridge_tol)
        k = min(1.0, max(0.0, self._ttba_k(kb)))
        return self._radial_full(r * direction(k), verified=(k == 0.0)), kb

    # ---------------------------------------------------------------- step
    def step(self, y_b: Vec, n_y: Vec) -> PBResult:
        c, st = self.c, self.st
        y_b = np.asarray(y_b, dtype=float)
        r = float(np.linalg.norm(y_b))
        n = _unit2(n_y) if n_y is not None else None
        if r <= 1e-12 or n is None or self.remaining() <= 0:
            return PBResult(None, "X")
        if c.rule == "cgba":  # build 34b: official updates in the chart (no gate, no eps bookkeeping)
            return PBResult(self.thales(y_b, n), "C")
        if c.rule == "cgba_h":
            return PBResult(self.cgba_h(y_b, n), "H")
        if c.rule == "ttba":
            y, kb = self.ttba(y_b, n)
            return PBResult(y, "D", kb=kb)
        if c.rule != "pvre":
            raise ValueError(f"unknown PBConfig.rule {c.rule!r}")
        v = y_b / r
        direction = self._dir_fn(n, v)
        kg = float(c.escape_k)
        if (c.gate_order == "flat_first" and c.thales_mode == "flat" and self._flat()
                and st.thales_run < max(0, c.thales_steps)):  # ---- T (flat-first, no gate query)
            st.thales_run += 1
            return self._finish(PBResult(self.thales(y_b, n), "T"))
        if not self.adv(r * direction(kg), "update"):  # ---- N
            st.thales_run = 0
            kb = self._bridge(r, direction, 0.0, kg, c.narrow_tol)
            y = self._escape(r, v, direction, kb, self._eps_seed(r, v, direction, kb))
            return self._finish(PBResult(y, "N", self._eps(r, v, direction, kb, y), kb))
        use_thales = c.thales_mode == "always" or (
            c.thales_mode == "flat" and self._flat() and st.thales_run < max(0, c.thales_steps))
        if use_thales:  # ---- T
            st.thales_run += 1
            return self._finish(PBResult(self.thales(y_b, n), "T"))
        st.thales_run = 0  # ---- B / F
        kb = self._bridge(r, direction, kg, 1.0, c.bridge_tol)
        if kb >= 1.0 - 2.0 * c.bridge_tol:
            k = min(1.0, max(0.0, self._ttba_k(kb)))
            y = self._radial_full(r * direction(k), verified=(k == 0.0))
            return self._finish(PBResult(y, "F", kb=kb))
        y = self._escape(r, v, direction, kb, self._eps_seed(r, v, direction, kb))
        return self._finish(PBResult(y, "B", self._eps(r, v, direction, kb, y), kb))

    def _finish(self, res: PBResult) -> PBResult:
        if math.isfinite(res.eps):
            self.st.eps.append(res.eps)
        return res
