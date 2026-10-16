#!/usr/bin/env python3
"""B4d T4(c) — unpatched official Proposed_attack (faithful model, no PatchedProposedAttack).

Run the UNPATCHED official CGBA (non_targeted) on resnet50 rows 1-3 and report:
  * whether proposed_attack.py:252 (``assert self.all_queries == q_num``) raises,
  * at which query (self.all_queries at the crash) and iteration,
  * the tracker best pixel L2 at that moment.

This is the D8 off-by-one: the 40-iteration cap branch returns num_calls==40 after only 39 is_adversarial
calls, so q_num overshoots all_queries by one and the assert trips on the first capped walk.

    python adapters/diag/diag_b4d_t4c_unpatched.py
"""
from __future__ import annotations

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
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
import torch  # noqa: E402
from proposed_attack import Proposed_attack  # noqa: E402

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
BUDGET = 10000
ARCH = "resnet50"


def run_image(r, idx):
    name = r["source"]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, name, "cuda")
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = A._faithful_model(victim)
    params = dict(A.PARAMS, attack_method="CGBA")
    ckpts = C.contract_checkpoints(C.load_contract())
    tracker = HardLabelTracker(raw, x0, y_src, None, ckpts, BUDGET,
                               to_pixel=C.to_pixel_unnormalize, init_adv_pixel=None)
    C.seed_all(idx)
    src = C.normalize(x0)
    lb = C.normalize(torch.zeros_like(x0))
    ub = C.normalize(torch.ones_like(x0))
    atk = Proposed_attack(tracker, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=None,
                          dim_reduc_factor=params["dim_reduc_factor"], attack_method="CGBA",
                          iteration=params["iteration"], initial_query=params["initial_query"],
                          tol=params["tol"], sigma=params["sigma"], verbose_control="No")
    crash = None
    try:
        atk.Attack()
    except BudgetExhausted:
        crash = None
    except AssertionError as e:
        tb = e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "Attack" and tb.tb_lineno == 252:
                loc = tb.tb_frame.f_locals
                crash = {"query": int(loc.get("q_num", -1)), "all_queries": int(atk.all_queries),
                         "iter": int(loc.get("i", -1)), "best": float(tracker.core.best_l2)}
                break
            tb = tb.tb_next
    if crash is None:
        print(f"RESULT T4c {C.stem_of(name)}: NO crash at :252 (completed, queries={tracker.queries})")
    else:
        print(f"RESULT T4c {C.stem_of(name)}: crash at proposed_attack.py:252  iter={crash['iter']} "
              f"q_num={crash['query']} all_queries={crash['all_queries']} "
              f"tracker_best_pixel_l2={crash['best']:.4f}")


def main():
    rows = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_non_targeted.csv")[:3]
    for i, r in enumerate(rows):
        run_image(r, i)
    print("DONE T4c")


if __name__ == "__main__":
    main()
