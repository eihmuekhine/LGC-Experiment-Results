"""Planar boundary geometry used by the bridge / model / gamma-arc updates.

All quantities live in the 2-D search plane spanned by the current direction v
(from the source x0 to the current boundary point x_b, |x_b - x0| = r) and the
estimated normal. Angles are measured at x0 from v.

Linear boundary (tangent-hyperplane model, used by CGBA, SurFree, TA with
alpha = pi/2):
  * the closest boundary point x* sees the chord x_b-x* under a right angle
    (Thales), so a point in direction psi is predicted at radius r cos(psi);
  * the bridge point (second boundary point at radius r) lies at 2*alpha and
    x* = (x_b + x_bridge) / 2.

Curved boundary: the angle at the new vertex becomes gamma != pi/2 and, by the
law of sines, the new boundary point in direction psi lies at radius
    g = r sin(gamma + psi) / sin(gamma).
For a hyperbolic boundary h(s) = d* + a (sqrt(1 + s^2/b^2) - 1),
    gamma = pi/2 + atan(t),  t = (h(s) - d*) / s  ("sag slope" of the chord),
which is linear in a for fixed b. The fit below is therefore a 1-D grid over
b with a closed-form weighted least-squares a. The abscissa is the lateral offset
s = r sin(psi) of x_b from the new ray (the same quantity at fit and at prediction).

Predict-verify-refine (PVR) additionally uses
  * secant_foot: the foot direction from the TWO boundary points of a model step
    (x_b and the new point), curvature-corrected with the fitted kappa;
  * optimal_probe_count: the probe count N* that maximises progress per query.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

TA_HALF_RANGE = 0.1  # official Triangle Attack bound |alpha - pi/2| <= 0.1 rad


def c_hat_theory(n_probes: int, d_eff: int) -> float:
    """Expected cos(normal estimate, true normal) for n sign probes (Gaussian, locally linear)."""
    return 1.0 / math.sqrt(1.0 + math.pi * max(d_eff, 1) / (2.0 * max(n_probes, 1)))


def model_psi(c_hat: float, cos_theta: float, cos_alpha: float) -> float:
    """Exact in-plane optimal rotation for a linear boundary.

    tan psi* = (cos beta - cos theta cos alpha) / (sin theta cos alpha), with
    cos beta = c_hat (normal quality), theta = angle(v, n_hat), alpha = angle(v, n*).
    """
    st = math.sqrt(max(1.0 - cos_theta * cos_theta, 1e-12))
    return math.atan2(c_hat - cos_theta * cos_alpha, st * cos_alpha)


def cos_alpha_from_bridge(c_hat: float, cos_theta: float, alpha_p: float) -> float:
    """Invert model_psi: the bridge measures alpha_p = psi_bridge / 2 -> cos(alpha)."""
    st = math.sqrt(max(1.0 - cos_theta * cos_theta, 1e-12))
    return c_hat / max(st * math.tan(alpha_p) + cos_theta, 1e-12)


def sag_slope(r: float, psi: float, g: float) -> float:
    """t = tan(gamma - pi/2) of the triangle (x0, x_b, x_new); t = 0 for a linear boundary."""
    s = math.sin(psi)
    if s <= 1e-12:
        return 0.0
    return (math.cos(psi) - g / r) / s


def gamma_from_triangle(r: float, psi: float, g: float) -> float:
    """Angle at the new boundary point, from |x_b - x0| = r, the rotation psi and the new radius g."""
    return 0.5 * math.pi + math.atan(sag_slope(r, psi, g))


def inscribed_radius(r: float, psi: float, gamma: float) -> float:
    """Radius of the point in direction psi on the arc that sees [x0, x_b] under angle gamma."""
    s = math.sin(gamma)
    if s <= 1e-12:
        return r * math.cos(psi)
    return r * math.sin(gamma + psi) / s


def secant_foot(r: float, g: float, psi: float, kappa: float = 0.0, psi_max: float = 0.5 * math.pi) -> float:
    """Direction (angle from v at x0) of the foot of the perpendicular, from TWO boundary points.

    The points are x_b = x0 + r v and p = x0 + g (cos psi v + sin psi w). Model the in-plane
    boundary slice as a parabola with vertex (foot) in direction p* at distance d:
        x.u = d + (kappa / 2) (x.e)^2,   u = foot direction, e perpendicular to u.
    Both points give the same d, so p* solves
        F(p) = [r cos p - kappa/2 r^2 sin^2 p] - [g cos(psi - p) - kappa/2 g^2 sin^2(psi - p)] = 0.
    For kappa = 0 (hyperplane) the root is explicit (secant through the two points):
        tan p* = (r - g cos psi) / (g sin psi).
    A curved root is searched on (0, psi_max) by bisection; the linear root is the fallback.
    """
    lin = math.atan2(r - g * math.cos(psi), max(g * math.sin(psi), 1e-12))
    if abs(kappa) < 1e-12 or not math.isfinite(kappa):
        return lin

    def f(p: float) -> float:
        return ((r * math.cos(p) - 0.5 * kappa * r * r * math.sin(p) ** 2)
                - (g * math.cos(psi - p) - 0.5 * kappa * g * g * math.sin(psi - p) ** 2))

    a, b = 1e-6, min(psi_max, 0.5 * math.pi - 1e-6)
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)) or fa * fb > 0.0:
        return lin
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if b - a < 1e-12:
            break
    return 0.5 * (a + b)


def exact_foot_radius(r: float, cos_alpha: float) -> float:
    """Closest boundary point radius on a hyperplane: d* = r cos(alpha) (x* = x0 + r (n.v) n)."""
    return r * cos_alpha


def optimal_probe_count(cos_alpha: float, d_eff: int, overhead: float, n_min: int = 20,
                        n_max: int = 1000, step: int = 5) -> int:
    """Probe count N* that maximises log-distance progress per query on a hyperplane.

    With c(N)^2 = 2N / (2N + pi d) (sign-probe normal quality) the exact in-plane step
    reduces the distance by   -log(d'/d) = 1/2 log(1 + c(N)^2 tan^2(alpha) sin^2(alpha)),
    where alpha = angle(v, n*). The per-iteration cost is N + overhead (bridge / radial
    queries). N* maximises the ratio; it grows as the attack converges (alpha -> 0).
    """
    ca = min(max(cos_alpha, 1e-3), 0.999)
    beta = (1.0 / (ca * ca) - 1.0) * (1.0 - ca * ca)  # tan^2(alpha) sin^2(alpha)
    best_n, best_rate = n_min, -1.0
    for n in range(n_min, n_max + 1, step):
        c2 = 2.0 * n / (2.0 * n + math.pi * max(d_eff, 1))
        rate = 0.5 * math.log1p(c2 * beta) / (n + max(overhead, 0.0))
        if rate > best_rate:
            best_n, best_rate = n, rate
    return best_n


class HyperbolaFit:
    """Online fit of the chord sag slope t(c) = a * (sqrt(1 + c^2/b^2) - 1) / c.

    b -> inf with a/b^2 fixed gives a parabola (t ~ kappa c / 2); b -> 0 gives a
    corner between two hyperplanes (t ~ a/b). Observations are (lateral offset
    c = r sin(psi) of x_b from the new ray, sag slope t); older ones are
    down-weighted by `lam` per update.
    """

    def __init__(self, window: int = 12, lam: float = 0.9, min_obs: int = 3, t_max: float = 1.7):
        self.obs: Deque[Tuple[float, float]] = deque(maxlen=window)
        self.lam = lam
        self.min_obs = min_obs
        self.t_max = t_max  # |gamma - pi/2| <= atan(1.7) ~ 60 deg
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    @staticmethod
    def basis(c: np.ndarray, b: float) -> np.ndarray:
        c = np.maximum(np.asarray(c, dtype=float), 1e-12)
        return (np.sqrt(1.0 + (c / b) ** 2) - 1.0) / c

    def update(self, c: float, t: float) -> None:
        if not (math.isfinite(c) and math.isfinite(t)) or c <= 0:
            return
        self.obs.append((float(c), float(np.clip(t, -self.t_max, self.t_max))))
        if len(self.obs) < self.min_obs:
            return
        cs = np.array([o[0] for o in self.obs])
        ts = np.array([o[1] for o in self.obs])
        w = self.lam ** np.arange(len(cs))[::-1]
        med = float(np.median(cs))
        best = None
        for b in med * 2.0 ** np.arange(-4.0, 7.0, 0.5):
            phi = self.basis(cs, b)
            den = float(np.sum(w * phi * phi))
            if den <= 1e-30:
                continue
            a = float(np.sum(w * phi * ts) / den)
            res = float(np.sum(w * (ts - a * phi) ** 2))
            if best is None or res < best[0]:
                best = (res, a, b)
        if best is not None:
            _, self.a, self.b = best

    def ready(self) -> bool:
        return self.a is not None

    def predict_t(self, c: float) -> float:
        if not self.ready():
            if not self.obs:
                return 0.0
            ts = np.array([o[1] for o in self.obs])
            w = self.lam ** np.arange(len(ts))[::-1]
            return float(np.sum(w * ts) / np.sum(w))
        return float(np.clip(self.a * self.basis(np.array([c]), self.b)[0], -self.t_max, self.t_max))

    def predict_gamma(self, c: float) -> float:
        return 0.5 * math.pi + math.atan(self.predict_t(c))

    @property
    def kappa0(self) -> float:
        """Curvature at the vertex a / b^2 (nan before the first fit)."""
        return float("nan") if not self.ready() else self.a / (self.b * self.b)
