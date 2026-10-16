"""Build 35: torch-free mathematics of the latent-space innovations (claude/PVRE_information_geometry_2026-10-05.md,
design doc sections 16-18).  Used by lgc_bench/engine.py and unit-tested by selftest_35.py (Part A, NumPy only).

1. KSQP closed form (equality-constrained Gauss-Newton step on the estimated boundary tangent)
       min_p  g^T p + 1/2 p^T G p   s.t.  n^T p = 0
       p = -z_g + z_n (n^T z_g) / (n^T z_n),   z_g = G^-1 g,  z_n = G^-1 n
   and the metric-normal corrector q_n = z_n / (n^T z_n)  (minimum-G-norm displacement with n^T q = 1).
2. CPS: curvature-calibrated probe scale.  At a boundary point x_b (adversarial side, offset delta >= 0 from the
   boundary after bisection) a probe x_b + sigma u has the signed boundary value
       phi = delta + sigma_n t + 1/2 sigma^2 u_T^T K u_T + O(sigma^3),   t ~ N(0, 1)
   (sigma_n = sigma * per-direction probe std, K = second fundamental form on the tangent block).  In units of
   sigma_n the sign is s = sign(t + beta + gamma xi) with xi ~ N(0, 1) (CLT for the even part) and
       beta(sigma_n) = delta / sigma_n + B sigma_n,  B = tr K / 2         (offset ~ 1/sigma, mean curvature ~ sigma)
       gamma(sigma_n) = Gam sigma_n,                 Gam = sqrt(sum kappa_i^2 / 2)   (curvature fluctuation)
   Observables at one scale:
       sign share    p = Phi(beta / sqrt(1 + gamma^2))
       pair agreement (antithetic +-u)  A = E_xi[ 2 Phi(|beta + gamma xi|) - 1 ]
       signal factor E[t s] / E[|t|] = (1 + gamma^2)^-1/2 exp(-beta^2 / (2 (1 + gamma^2)))
   so the c^2 multiplier of the sign estimator is  f = exp(-beta^2 / (1 + gamma^2)) / (1 + gamma^2).
   (p, A) -> (beta, gamma) by bisection; with delta known (cps_offset_frac * tol) -> (B, Gam), which are
   scale-invariant; sigma_n* = argmax_sigma f(sigma) on a bounded log grid.  Closed forms (Gam = 0):
       B < 0:  sigma* = sqrt(delta / -B), f* = 1  (offset and mean curvature cancel)
       B > 0:  sigma* = sqrt(delta /  B), f* = exp(-4 delta B)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np



def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def Phi_inv(p: float) -> float:
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------- KSQP
def ksqp_direction(G: np.ndarray, g: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dense reference implementation (selftest): returns (p, q_n)."""
    zg, zn = np.linalg.solve(G, g), np.linalg.solve(G, n)
    den = float(n @ zn)
    return -zg + zn * float(n @ zg) / den, zn / den


def cg_solve(matvec, b: np.ndarray, iters: int) -> Tuple[np.ndarray, float]:
    """Conjugate gradients for an SPD operator (reference for the torch version in the engine)."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = float(r @ r)
    b2 = rr
    if b2 <= 0:
        return x, 0.0
    for _ in range(int(iters)):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if not math.isfinite(pAp) or pAp <= 0:
            break
        a = rr / pAp
        x = x + a * p
        r = r - a * Ap
        rn = float(r @ r)
        if rn <= 1e-24 * b2:
            rr = rn
            break
        p = r + (rn / rr) * p
        rr = rn
    return x, math.sqrt(max(rr, 0.0) / b2)


# -------------------------------------------------------------------------------------------------- CPS
def cps_share(beta: float, gamma: float) -> float:
    return Phi(beta / math.sqrt(1.0 + gamma * gamma))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def owens_t(h: float, a: float) -> float:
    """Owen's T(h, a) = (1/2pi) int_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx  (a >= 0; T is even in h).

    a <= 1: 64-point Gauss-Legendre on [0, a] (smooth integrand).  a > 1: the reflection
    T(h, a) = Phi(h)/2 + Phi(a h)/2 - Phi(h) Phi(a h) - T(a h, 1/a)   (h >= 0)."""
    h, a = abs(float(h)), float(a)
    if a <= 0.0:
        return 0.0
    if math.isinf(a):
        return 0.5 * (1.0 - Phi(h))
    if a > 1.0:
        ah = a * h
        return 0.5 * Phi(h) + 0.5 * Phi(ah) - Phi(h) * Phi(ah) - owens_t(ah, 1.0 / a)
    x = 0.5 * a * (_GL_X + 1.0)
    f = np.exp(-0.5 * h * h * (1.0 + x * x)) / (1.0 + x * x)
    return float(0.5 * a * np.sum(_GL_W * f) / (2.0 * math.pi))


def cps_agree(beta: float, gamma: float) -> float:
    """A = E_xi[2 Phi(|beta + gamma xi|) - 1] = P(|t| < |c|), c ~ N(beta, gamma^2), in closed form:
    with U = c - t, V = c + t (corr (gamma^2 - 1)/(gamma^2 + 1)),  A = P(U V > 0) = 1 - 4 T(beta / sqrt(1 + gamma^2), 1/gamma)."""
    gamma = abs(float(gamma))
    if gamma < 1e-12:
        return 2.0 * Phi(abs(float(beta))) - 1.0
    return 1.0 - 4.0 * owens_t(float(beta) / math.sqrt(1.0 + gamma * gamma), 1.0 / gamma)


def cps_solve(share: float, agree: float, gamma_max: float = 20.0,
              n_pairs: int = 0) -> Tuple[float, float]:
    """(beta, gamma) from the sign share and the antithetic agreement rate (both in [0, 1]).

    At a fixed share the agreement is flat in gamma near 0 (second order, and numerically 0 up to gamma ~ 0.3 when
    |beta| > 1), so gamma is identifiable only through a significant excess A - A(gamma = 0).  With n_pairs > 0 the
    excess must exceed 2 binomial standard errors, otherwise gamma = 0 (sampling noise would bias gamma upwards).
    A one-sided share (outside (0.01, 0.99)) returns the conservative |beta| bound and gamma = 0."""
    raw = float(share)
    share = min(max(raw, 0.005), 0.995)
    q = Phi_inv(share)
    if not (0.01 < raw < 0.99):
        return q, 0.0

    def a_of(g: float) -> float:
        return cps_agree(q * math.sqrt(1.0 + g * g), g)
    a0 = a_of(0.0)
    thr = 2.0 * math.sqrt(max(a0 * (1.0 - a0), 1e-4) / n_pairs) if n_pairs > 0 else 0.0
    if not math.isfinite(agree) or agree <= a0 + thr + 1e-9:  # 1e-9: numerical flatness of A(gamma) near 0
        return q, 0.0
    if agree >= a_of(gamma_max):
        g = gamma_max
    else:
        lo, hi = 0.0, gamma_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if a_of(mid) < agree:
                lo = mid
            else:
                hi = mid
        g = 0.5 * (lo + hi)
    return q * math.sqrt(1.0 + g * g), g


def cps_signal(sig: float, delta: float, B: float, Gam: float) -> float:
    """c^2 multiplier of the sign estimator at normal-probe std sig (beta = delta/sig + B sig, gamma = Gam sig)."""
    if not (sig > 0):
        return 0.0
    b = delta / sig + B * sig
    c2 = (Gam * sig) ** 2
    return math.exp(-b * b / (1.0 + c2)) / (1.0 + c2)


def cps_decompose(beta: float, gamma: float, sig: float, delta: float) -> Tuple[float, float]:
    """Scale-invariant (B, Gam) from (beta, gamma) measured at normal-probe std sig, center offset delta."""
    return (beta - delta / sig) / sig, gamma / sig


def cps_best_sigma(delta: float, B: float, Gam: float, lo: float, hi: float, n: int = 121) -> float:
    """argmax of cps_signal on a log grid of [lo, hi] (refined by one golden-section pass around the grid max)."""
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), n))
    vals = np.array([cps_signal(float(x), delta, B, Gam) for x in grid])
    i = int(np.argmax(vals))
    a, b = math.log(grid[max(0, i - 1)]), math.log(grid[min(n - 1, i + 1)])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    f = lambda t: cps_signal(math.exp(t), delta, B, Gam)
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(40):
        if f(c) >= f(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    t = 0.5 * (a + b)
    return math.exp(t) if f(t) >= vals[i] else float(grid[i])


# --------------------------------------------------------------------------------------------- build 37
def elasticity(hist) -> float:
    """Local elasticity eps_hat(q) = ln(d(q_j) / d(q)) / ln(q / q_j) of a best-distance history
    hist = [(q_1, d_1), ..., (q, d)] (queries increasing), q_j = last recorded query count <= q / 2.

    Floor model d(q) = d_V + C q^-alpha0 (latent floor d_V = d* / ||P_V n||):
        eps(q) = -d ln d / d ln q = alpha0 (1 - d_V / d(q)),  i.e. the floor share d_V / d = 1 - eps / alpha0.
    NaN when no recorded point lies at or below q / 2."""
    if len(hist) < 2:
        return float("nan")
    q, d = hist[-1]
    for qj, dj in reversed(hist[:-1]):
        if qj <= 0.5 * q:
            if qj > 0 and dj > 0 and d > 0 and q > qj:
                return math.log(dj / d) / math.log(q / qj)
            return float("nan")
    return float("nan")


def explore_interval(k: int, k0: int, k_max: int, lost: bool) -> int:
    """Doubling exploration (build 37): a forced trial of a block that stays slower than the other block doubles
    its exploration interval (capped at k_max); a trial that is faster resets it to k0."""
    k0 = max(2, int(k0))
    return min(2 * int(k), max(k0, int(k_max))) if lost else k0
