#!/usr/bin/env python3
"""T3(b): CGBA resnet50 NT row 1 — verify the normalised<->pixel L2 mapping from the returned iterate.

Captures the FINAL returned iterate x_n (normalised space) from Proposed_attack.Attack(), then:
    d_n = ||x_n - normalize(x0)||_2
    d_p = ||to_pixel(x_n) - x0||_2
Require d_n/d_p in [4.37, 4.46] AND tracker pix_l2@10000 <= d_p.  Otherwise the L2 mapping is wrong -> STOP.

    python adapters/diag/diag_t3b_cgba_l2map.py
"""
from __future__ import annotations

import os
import sys

REPO = "/root/autodl-tmp/LGC_frozen"
sys.path.insert(0, os.path.join(REPO, "adapters"))
sys.path.insert(0, os.path.join(REPO, "tools", "eval"))
sys.path.insert(0, REPO)
sys.path.insert(0, "/root/autodl-tmp/baselines/CGBA")

import _common as C  # noqa: E402
import torch  # noqa: E402

from adapters.run_cgba import _faithful_model, _patched_attack_class, PARAMS  # noqa: E402

ARCH = "resnet50"
SPLIT = os.path.join(REPO, "splits", "imagenet", "ttba_github", "resnet50_non_targeted.csv")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    victim = C.load_victim_any(ARCH, device)
    checkpoints = C.contract_checkpoints()
    budget = C.contract_budget()
    r = C.load_split_rows(SPLIT)[0]
    name, y_src = r["source"], int(r["source_label"])
    x0 = C.load_x0_any(ARCH, name, device, data_dir="/root/autodl-tmp/ttba_mini_flat")

    tracker = C.HardLabelTracker(_faithful_model(victim), x0, y_src, None, checkpoints, budget,
                                 to_pixel=C.to_pixel_unnormalize, init_adv_pixel=None)

    src = C.normalize(x0)
    lb = C.normalize(torch.zeros_like(x0))
    ub = C.normalize(torch.ones_like(x0))
    Patched = _patched_attack_class()

    class Recording(Patched):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._last_iterate = None

        def _rec(self, out):
            if torch.isfinite(out).all():
                self._last_iterate = out.detach().clone()
            return out

        def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
            out, n = super().go_to_boundary_CGBA(x_s, eta_o, x_b)
            return self._rec(out), n

        def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):
            out, n = super().go_to_boundary_CGBA_H(x_s, eta_o, x_b)
            return self._rec(out), n

    C.seed_all(0)  # same per-image seed as the adapter (C.run: seed_base 0 + idx 0)
    attack = Recording(tracker, src, list(C.MEAN), list(C.STD), lb, ub,
                       dim_reduc_factor=PARAMS["dim_reduc_factor"], attack_method="CGBA",
                       iteration=PARAMS["iteration"], initial_query=PARAMS["initial_query"],
                       tol=PARAMS["tol"], sigma=PARAMS["sigma"], verbose_control="No")
    x_n = None
    try:
        x_n, n_query, norms = attack.Attack()
    except C.BudgetExhausted:
        x_n = attack._last_iterate

    if x_n is None:
        print("STOP: no final iterate captured")
        return 1

    if not torch.isfinite(x_n).all():
        print("STOP: final iterate non-finite")
        return 1

    # Attack() returns ``x_adv = clip_image_values(x_adv, self.lb, self.ub)`` (proposed_attack.py:264):
    # clamp the captured boundary point to the normalised box exactly as the returned iterate is clamped.
    x_n = torch.min(torch.max(x_n, lb), ub)

    x0_n = src
    d_n = float(torch.norm(x_n - x0_n).item())
    to_pixel = C.to_pixel_unnormalize(x_n)
    d_p = float(torch.norm(to_pixel - x0).item())
    ratio = d_n / d_p

    pix_l2_10000 = float(tracker.core.recorded.get(10000, (float("inf"), -1))[0])

    print(f"image={name} queries={tracker.queries}")
    print(f"d_n (normalised) = {d_n:.6f}")
    print(f"d_p (pixel)      = {d_p:.6f}")
    print(f"d_n/d_p          = {ratio:.6f}   (required in [4.37, 4.46])")
    print(f"tracker pix_l2@10000 = {pix_l2_10000:.6f}  (required <= d_p={d_p:.6f})")

    ok = 4.37 <= ratio <= 4.46 and pix_l2_10000 <= d_p + 1e-6
    print("VERDICT:", "OK" if ok else "STOP (L2 mapping wrong)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
