#!/usr/bin/env python3
"""B4e T1(d) — run the FIXED CGBA variant on the stalled images and report the post-fix behaviour.

For every row (default rows 1-3) of a CGBA non_targeted cell:
  * pix_l2 at 1000 / 2000 / 5000 / 10000 (found = 1 rows only);
  * nonfinite_queries (from the NonfiniteGuard in the guarded HardLabelTracker);
  * the number of SemiCircular_boundary_search walks that hit the 100-call cap (``num_calls > 100``);
  * stalled = pix_l2@2000 == pix_l2@10000 (must be False after the D11 acos clip).

    python adapters/diag/diag_b4e_t1_stalled.py --arch resnet50
    python adapters/diag/diag_b4e_t1_stalled.py --arch Salman2020Do_R50
    python adapters/diag/diag_b4e_t1_stalled.py --arch resnet50_places365
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))
sys.path.insert(0, "/root/autodl-tmp/baselines/CGBA")

import _common as C  # noqa: E402
import run_cgba as A  # noqa: E402
from hardlabel_tracker import BudgetExhausted  # noqa: E402
import torch  # noqa: E402

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
BUDGET = 10000
BUDGETS = [1000, 2000, 5000, 10000]

SPLITS = {
    "resnet50": "splits/imagenet/ttba_github/resnet50_non_targeted.csv",
    "Salman2020Do_R50": "splits/robust/Salman2020Do_R50_non_targeted.csv",
    "resnet50_places365": "splits/places365/resnet50_places365_non_targeted.csv",
}


def run_image(arch, r, idx):
    name = r["source"]
    y_src = int(r["source_label"])
    x0 = C.load_x0_any(arch, name, "cuda", DATA_DIR)
    victim = C.load_victim_any(arch, "cuda", CKPT_DIR)
    raw = A._faithful_model(victim)
    params = dict(A.PARAMS, attack_method="CGBA")
    ckpts = C.contract_checkpoints(C.load_contract())

    tracker = C.HardLabelTracker(raw, x0, y_src, None, ckpts, BUDGET,
                                 to_pixel=C.to_pixel_unnormalize, init_adv_pixel=None)

    FixedPatched = A._patched_attack_class("fixed")

    class CountingAttack(FixedPatched):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bssp_cap_hits = 0

        def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
            out, n_calls = super().SemiCircular_boundary_search(x_0, x_b, p_near_boundary)
            if n_calls > 100:
                self.bssp_cap_hits += 1
            return out, n_calls

    C.seed_all(idx)
    src = C.normalize(x0)
    lb = C.normalize(torch.zeros_like(x0))
    ub = C.normalize(torch.ones_like(x0))
    atk = CountingAttack(tracker, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=None,
                         dim_reduc_factor=params["dim_reduc_factor"], attack_method="CGBA",
                         iteration=params["iteration"], initial_query=params["initial_query"],
                         tol=params["tol"], sigma=params["sigma"], verbose_control="No")
    try:
        atk.Attack()
    except BudgetExhausted:
        pass

    l2 = {b: (float(tracker.core.recorded[b][0]) if tracker.core.recorded.get(b, (float("inf"), -1))[0] != float("inf")
               else float("nan")) for b in BUDGETS}
    stalled = l2[2000] == l2[10000]
    print(f"RESULT {C.stem_of(name)}: pix_l2 1k={l2[1000]:.4f} 2k={l2[2000]:.4f} 5k={l2[5000]:.4f} "
          f"10k={l2[10000]:.4f} nonfinite_queries={tracker.nonfinite_queries} "
          f"bssp_100call_cap_hits={atk.bssp_cap_hits} stalled={stalled}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=sorted(SPLITS))
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args()
    rows = C.load_split_rows(SPLITS[a.arch])[: a.rows]
    for i, r in enumerate(rows):
        run_image(a.arch, r, i)
    print(f"DONE diag_b4e_t1_stalled {a.arch}")


if __name__ == "__main__":
    main()
