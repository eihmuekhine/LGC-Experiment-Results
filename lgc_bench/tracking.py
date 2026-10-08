"""Normal tracking: empirical-Bayes reuse of the previous normal estimate (rank-1 prior).

Motivation (ceiling analysis, final_dev n = 50): 84-92 % of every probe-based attack's queries
estimate the boundary normal, whose quality with N sign probes in a d-dimensional probe
subspace is only
    c_f^2 = cos^2(n_hat, n*) = 1 / (1 + d tau^2),     tau^2 = pi / (2 N)       (balanced probes)
(c_f ~ 0.10-0.12 for DCT f=4, d = 9408, N = 150-200). Per-step progress is first order in c^2,
the in-plane update geometry only second order. TtBA, CGBA and HSJA discard every previous
estimate; consecutive boundary points are close, so their normals may be correlated.

Model (units of the per-direction probe standard deviation s, s^2 = E|u|^2 / d):
    acc = sum_i phi_i u_i,  phi_i = sign(<u_i, n*> sigma + b)                (engine estimator)
    E[phi u] = a n*,        a = sqrt(2/pi) s exp(-z^2/2),  z = Phi^-1(share of +1 answers)
    g = acc / (N a) = n* + e,   e isotropic, per-direction variance tau^2 = (pi / 2N) exp(z^2)
The previous tracked estimate p (unit) is independent of the fresh noise e, so
    y = <g, p> ~ N(c_p, tau^2),    c_p = <n*_t, p>   (alignment of the prior with the NEW normal;
                                                     it already contains the normal drift).
Estimator (empirical Bayes, two scalars per iteration, no extra queries):
    prior on c_p       : N(mu, v),  mu = EMA_beta(y),  v = max(EMA_beta((y - mu)^2) - tau^2, 0)
    gate               : if |mu| < gate * se(mu) the prior is not significant -> fresh estimate
    posterior          : c_p^ = (mu tau^2 + y v) / (v + tau^2),  pv = v tau^2 / (v + tau^2)
    Wiener gain (perp) : eta = (1 - m2) / ((1 - m2) + (d - 1) tau^2),  m2 = c_p^2 + pv
    estimate           : n_hat = unit(c_p^ p + eta (g - y p))
    expected quality   : c_eff^2 = num^2 / den,  num = c_p^2 + eta (1 - m2),
                         den = c_p^2 + eta^2 ((1 - m2) + (d - 1) tau^2)
Simulation (sim/sim_track_estimator.py, d = 9408, n0 = 30, per-step normal correlation rho):
    c^2 gain x16.9 (rho = 1), x4.2 (0.9), x2.1 (0.8), x1.14 (0.5), x0.99 (0.3), x0.97 (0).
So the worst case (normals decorrelate completely every step) costs ~3 % in c^2.
Diagnostic: rho_hat = sum_t y_t / sum_t c_eff_{t-1} (alignment the prior keeps after one step).

FULL-LOOP RESULT (sim/sim_track.py, sim/sim_track_useful.py) - the estimator-only gain does NOT carry
over to the attack. After each step the iterate x_b has already moved along the previous estimate,
so the information in p is stored in the direction v = unit(x_b - x0). Only the part of the normal
ORTHOGONAL to v drives the next step, and there the naive tracker is worse than the fresh estimate
(hyperplane, TtBA rule: cos(n_hat, n*) 0.26 vs 0.10, but useful cos 0.074 vs 0.093; distance +8 %
to +24 % at 1k-10k queries, 0/8 seeds better). With ``v_dir`` the prior is first projected
orthogonal to v (the unexploited part only); its alignment with the next normal is then NEGATIVE
(rho_hat ~ -0.1 to -0.5: the leftover of p is the noise the last step moved along), the prior passes
the gate in only 2-14 % of the iterations and the attack is 0-2 % WORSE (sim_track.py, 6 seeds, 10k). Conclusion: the iterate is (to second order) a sufficient statistic of the
past normal estimates; normal memory is not a first-order lever. Kept as an ablation (*_NT).

Vectors may be torch tensors or numpy arrays (only elementwise *, +, sum are used).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

_ND = NormalDist()


def _dot(a: Any, b: Any) -> float:
    return float((a * b).sum())


class NormalTracker:
    """Empirical-Bayes rank-1 normal tracker (see module docstring).

    Args:
        d_eff: dimension of the probe subspace (DCT block size, latent dimension, ...).
        beta: EMA rate of the prior mean / variance of the alignment c_p (per iteration).
        gate: the prior is used only if |mu| >= gate * se(mu) (se from the EMA weights).
        share_clip: the +1 share is clipped to [share_clip, 1 - share_clip] before Phi^-1.
    """

    def __init__(self, d_eff: int, beta: float = 0.2, gate: float = 2.0, share_clip: float = 0.02):
        self.d = max(int(d_eff), 2)
        self.beta = float(beta)
        self.gate = float(gate)
        self.clip = float(share_clip)
        self.reset()
        self.hist: Dict[str, List[float]] = {"y": [], "tau": [], "mu": [], "c_eff": [], "c_fresh": [],
                                              "c_prev": [], "used": []}

    def reset(self) -> None:
        self.p: Optional[Any] = None  # previous tracked unit estimate
        self.mu: Optional[float] = None  # EMA of y
        self.s2: Optional[float] = None  # EMA of (y - mu)^2
        self.w2: float = 0.0  # sum of squared EMA weights (variance factor of mu)
        self.c_eff: Optional[float] = None

    def update(self, acc: Any, n_probes: int, share: float, probe_var: float,
               v_dir: Optional[Any] = None) -> Optional[Any]:
        """acc = sum_i phi_i u_i over n_probes probes; returns the tracked UNIT normal (or None).

        v_dir: unit direction x_b - x0 projected onto the probe subspace (and renormalised). If given,
        the prior is projected orthogonal to it first (only the part of the previous estimate that the
        last step has not exploited is reused); None = naive tracking (estimator-only setting)."""
        n = max(int(n_probes), 1)
        if v_dir is not None and self.p is not None:
            q = self.p - _dot(self.p, v_dir) * v_dir
            nq = math.sqrt(max(_dot(q, q), 0.0))
            self.p = q / nq if (math.isfinite(nq) and nq > 1e-12) else None
        sh = min(max(float(share), self.clip), 1.0 - self.clip)
        z = _ND.inv_cdf(sh)
        s = math.sqrt(max(float(probe_var), 1e-30))
        a = math.sqrt(2.0 / math.pi) * s * math.exp(-0.5 * z * z)
        tau2 = (math.pi / (2.0 * n)) * math.exp(z * z)
        g = acc / (n * a)
        c_fresh2 = 1.0 / (1.0 + self.d * tau2)
        y, used, c_prev = float("nan"), 0.0, float("nan")
        vec, c_eff2 = g, c_fresh2
        if self.p is not None:
            c_prev = float(self.c_eff) if self.c_eff is not None else float("nan")
            y = _dot(g, self.p)
            b = self.beta
            if self.mu is None:
                self.mu, self.s2, self.w2 = y, 0.0, 1.0
            else:
                dev = (y - self.mu) ** 2
                self.mu = (1.0 - b) * self.mu + b * y
                self.s2 = (1.0 - b) * self.s2 + b * dev
                self.w2 = (1.0 - b) ** 2 * self.w2 + b * b
            se2 = tau2 * self.w2
            if abs(self.mu) >= self.gate * math.sqrt(se2):
                v = max(self.s2 - tau2, 0.0)
                cp = (self.mu * tau2 + y * v) / (v + tau2)
                pv = v * tau2 / (v + tau2)
                m2 = min(cp * cp + pv, 1.0 - 1e-6)
                eta = (1.0 - m2) / ((1.0 - m2) + (self.d - 1) * tau2)
                vec = cp * self.p + eta * (g - y * self.p)
                num = cp * cp + eta * (1.0 - m2)
                den = cp * cp + eta * eta * ((1.0 - m2) + (self.d - 1) * tau2)
                c_eff2 = num * num / den if den > 0 else c_fresh2
                used = 1.0
        nv = math.sqrt(max(_dot(vec, vec), 0.0))
        if not math.isfinite(nv) or nv <= 1e-30:
            ng = math.sqrt(max(_dot(g, g), 0.0))
            if not math.isfinite(ng) or ng <= 1e-30:
                return None
            vec, nv, c_eff2, used = g, ng, c_fresh2, 0.0
        out = vec / nv
        self.p = out
        self.c_eff = math.sqrt(min(max(c_eff2, 0.0), 1.0))
        h = self.hist
        h["y"].append(y)
        h["tau"].append(math.sqrt(tau2))
        h["mu"].append(float("nan") if self.mu is None else self.mu)
        h["c_eff"].append(self.c_eff)
        h["c_fresh"].append(math.sqrt(c_fresh2))
        h["c_prev"].append(c_prev)
        h["used"].append(used)
        return out

    # ------------------------------------------------------------ run summaries
    def rho_hat(self, skip: int = 3) -> float:
        """sum y_t / sum c_eff_{t-1} over t > skip: the alignment the prior keeps after one step."""
        pairs = [(y, c) for y, c in zip(self.hist["y"][skip:], self.hist["c_prev"][skip:])
                 if math.isfinite(y) and math.isfinite(c)]
        den = sum(c for _, c in pairs)
        return sum(y for y, _ in pairs) / den if den > 0 else float("nan")

    def gain_c2(self, last_frac: float = 0.5) -> float:
        """Median over the last part of the run of c_eff^2 / c_fresh^2 (expected c^2 gain)."""
        ce, cf = self.hist["c_eff"], self.hist["c_fresh"]
        k = int(len(ce) * (1.0 - last_frac))
        r = sorted((a * a) / (b * b) for a, b in zip(ce[k:], cf[k:]) if b > 0)
        return r[len(r) // 2] if r else float("nan")

    def used_pct(self) -> float:
        u = self.hist["used"]
        return 100.0 * sum(u) / len(u) if u else float("nan")
