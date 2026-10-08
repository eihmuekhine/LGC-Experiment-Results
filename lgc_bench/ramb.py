"""Utilities for RA-MB-PVRE (regime-adaptive multi-branch PVR-E).

The module intentionally contains only deterministic geometry and candidate
book-keeping. Victim-model queries remain in :mod:`lgc_bench.engine`, so all
methods continue to use the same oracle and query accounting.

The prototype follows the reviewer-corrected design:
  * k_b <= 0.2 is the inherited TtBA narrow/high-curvature gate;
  * for k_b > 0.2, a *normalised angular bridge span*
        lambda_b = psi_b / theta
    is used instead of fixed k=0.8/0.9 gates;
  * the wide threshold is a tunable heuristic, not a theorem;
  * candidate directions are de-duplicated in angle, because k itself is not
    an angularly invariant coordinate;
  * branch selection is based on certified radial intervals, not midpoint
    estimates or stochastic-bandit assumptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class RACandidate:
    """One candidate direction in the RA-MB race.

    ``lo`` is a certified non-adversarial scalar radius and ``hi`` a certified
    adversarial scalar radius along this direction. ``lo_obj``/``hi_obj`` are
    the corresponding attack objectives and are used for cross-direction safe
    pruning. ``high_point`` is the verified adversarial state at ``hi``.
    """

    name: str
    k: float
    psi: float
    direction: object
    g_pred: float
    lo: float = 0.0
    hi: float = float("inf")
    lo_obj: float = 0.0
    hi_obj: float = float("inf")
    high_point: Optional[object] = None
    valid: bool = False
    queries: int = 0
    merged_from: str = ""


def psi_from_k(k: float, cos_theta: float) -> float:
    """Angle psi between v and unit(k*n + (1-k)*v).

    This is numerically more stable than explicitly forming the direction when
    only the angle is needed. ``cos_theta = <v,n>`` and theta in [0, pi].
    """
    k = min(1.0, max(0.0, float(k)))
    ct = min(1.0, max(-1.0, float(cos_theta)))
    st = math.sqrt(max(0.0, 1.0 - ct * ct))
    x = 1.0 - k + k * ct
    y = k * st
    return math.atan2(y, x)


def k_from_psi(psi: float, theta: float) -> float:
    """Inverse of :func:`psi_from_k` for 0 <= psi <= theta < pi.

    tan(psi) = k sin(theta) / [1-k+k cos(theta)].
    """
    theta = min(math.pi - 1e-8, max(0.0, float(theta)))
    psi = min(theta, max(0.0, float(psi)))
    if psi <= 1e-12:
        return 0.0
    st, ct = math.sin(theta), math.cos(theta)
    t = math.tan(psi)
    den = st + t * (1.0 - ct)
    if abs(den) <= 1e-12:
        return 0.0
    return min(1.0, max(0.0, t / den))


def normalised_bridge_span(psi_b: float, theta: float, eps: float = 1e-8) -> Optional[float]:
    """Return lambda_b = psi_b/theta, or None when theta is too small.

    ``lambda_b`` is a parameterisation-normalised angular span; it is *not*
    claimed to remove normal-estimation error or to be geometry invariant.
    """
    if not (math.isfinite(psi_b) and math.isfinite(theta)) or theta <= eps:
        return None
    return min(1.0, max(0.0, psi_b / theta))


def flat_radius(r: float, psi: float, psi_b: float, min_frac: float = 0.05) -> float:
    """Locally planar bridge prediction for the radial boundary.

        g_flat(psi) = r cos(psi_b/2) / cos(psi - psi_b/2).

    This is only a *predictor*. The engine always verifies it with hard-label
    queries and constructs a certified bracket before accepting a point.
    """
    a = 0.5 * psi_b
    den = math.cos(psi - a)
    if den <= 1e-10:
        g = r
    else:
        g = r * math.cos(a) / den
    return min(max(float(g), min_frac * r), r * (1.0 - 1e-6))


def deduplicate_candidates(cands: Iterable[RACandidate], eps_angle: float) -> List[RACandidate]:
    """Merge candidates whose angular separation is below ``eps_angle``.

    Priority is the input order, except that a candidate named ``escape`` is
    retained over a near-duplicate in the H regime so that exploration is not
    silently removed. The merged names are retained for diagnostics.
    """
    out: List[RACandidate] = []
    for c in cands:
        hit = None
        for i, old in enumerate(out):
            if abs(c.psi - old.psi) < eps_angle:
                hit = i
                break
        if hit is None:
            out.append(c)
            continue
        old = out[hit]
        names = [x for x in (old.merged_from or old.name, c.merged_from or c.name) if x]
        merged = "+".join(dict.fromkeys("+".join(names).split("+")))
        if c.name == "escape" and old.name != "escape":
            c.merged_from = merged
            out[hit] = c
        else:
            old.merged_from = merged
    return out


def safe_prunable(candidate: RACandidate, incumbent_u: float, tol: float = 1e-12) -> bool:
    """Certified branch-and-bound pruning rule.

    If the candidate's lower objective bound is no smaller than the best
    certified adversarial upper bound, the candidate cannot beat the incumbent
    under the maintained interval assumptions.
    """
    return candidate.valid and candidate.lo_obj >= incumbent_u - tol
