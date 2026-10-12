#!/usr/bin/env python3
"""B3b T3 — CGBA boundary-walk count off-by-one: patched vs unpatched trajectory.

The official ``Attack()`` (proposed_attack.py) maintains ``q_num`` and ``n_query``; the only uses that affect
control flow are ``n_query.append`` and ``assert self.all_queries == q_num`` (proposed_attack.py:252).  The boundary
walks return the wrong count at the 40-iteration cap, so the assert crashes on non_targeted images whose initial
boundary is large.  run_cgba.py now subclasses Proposed_attack and overrides go_to_boundary_CGBA /
go_to_boundary_CGBA_H to return ``all_queries``-diff (the true count), leaving the returned boundary point
unchanged, so the trajectory must be identical.

Checks (faithful path, ResNet-50 V2, budget 10000):
  (a) cells whose UNPATCHED run completes: per-checkpoint pix_l2 and state_query identical patched vs unpatched;
  (b) cells whose UNPATCHED run crashes: the patched run completes and its (n_query, norms) trajectory equals the
      unpatched run up to the crash iteration.
"""
from __future__ import annotations

import math
import os
import sys
import time

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


def tof(x):
    if isinstance(x, (int, float)):
        return float(x)
    return float(torch.as_tensor(x).reshape(-1)[0].item())


def list_eq(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


def run_cell(mode, r, idx):
    name = r["source"]
    y_src = int(r["source_label"])
    y_tgt = int(r["target_label"]) if (mode == "targeted" and r.get("target_label")) else None
    x0 = C.load_image(DATA_DIR, name, "cuda")
    x_tgt = C.load_image(DATA_DIR, r["target"], "cuda") if y_tgt is not None else None
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = A._faithful_model(victim)
    params = dict(A.PARAMS, attack_method=("CGBA_H" if y_tgt is not None else "CGBA"))
    ckpts = C.contract_checkpoints(C.load_contract())

    def build_tracker():
        return HardLabelTracker(raw, x0, y_src, y_tgt, ckpts, BUDGET,
                                to_pixel=C.to_pixel_unnormalize,
                                init_adv_pixel=x_tgt if y_tgt is not None else None)

    def make_attack(cls, model, iteration=None):
        src = C.normalize(x0)
        tar = C.normalize(x_tgt) if x_tgt is not None else None
        lb = C.normalize(torch.zeros_like(x0))
        ub = C.normalize(torch.ones_like(x0))
        return cls(model, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=tar,
                   dim_reduc_factor=params["dim_reduc_factor"], attack_method=params["attack_method"],
                   iteration=(params["iteration"] if iteration is None else iteration),
                   initial_query=params["initial_query"], tol=params["tol"],
                   sigma=params["sigma"], verbose_control=params["verbose_control"])

    # ---- unpatched ----
    unpatched_tr = build_tracker()
    C.seed_all(idx)
    unpatched = make_attack(Proposed_attack, unpatched_tr)
    unpatched_status = "complete"
    unpatched_trace = None
    try:
        unpatched.Attack()
    except BudgetExhausted:
        unpatched_status = "budget"
    except AssertionError as e:
        unpatched_status = "crash"
        tb = e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "Attack":
                loc = tb.tb_frame.f_locals
                unpatched_trace = ([tof(x) for x in loc.get("n_query", [])],
                                   [tof(x) for x in loc.get("norms", [])],
                                   int(loc.get("i", -1)))
                break
            tb = tb.tb_next
    unpatched_rec = {b: unpatched_tr.core.recorded.get(b) for b in ckpts}

    if unpatched_status == "crash":
        uq, un, ui = unpatched_trace
        # patched run capped just past the crash so it completes normally and returns its (n_query, norms)
        patched_tr = build_tracker()
        C.seed_all(idx)
        Patched = A._patched_attack_class()
        patched = make_attack(Patched, patched_tr, iteration=ui + 1)
        try:
            out = patched.Attack()
        except BudgetExhausted:
            print(f"RESULT T3 {mode} {C.stem_of(name)} unpatched=crash FAIL (patched capped run hit budget)", flush=True)
            return "FAIL"
        pq = [tof(x) for x in out[1]]
        pn = [tof(x) for x in out[2]]
        nq_ok = list_eq(pq[: len(uq)], uq)
        nm_ok = list_eq(pn[: len(un)], un)
        verdict = "PASS" if (nq_ok and nm_ok) else "FAIL"
        print(f"RESULT T3 {mode} {C.stem_of(name)} unpatched=crash {verdict} "
              f"crash_iter={ui} prefix_len={len(uq)} nq_match={nq_ok} norms_match={nm_ok} "
              f"cap_fix_log={len(patched._cap_fix_log)}", flush=True)
        return verdict

    # ---- patched (full budget) ----
    patched_tr = build_tracker()
    C.seed_all(idx)
    Patched = A._patched_attack_class()
    patched = make_attack(Patched, patched_tr)
    try:
        patched.Attack()
    except BudgetExhausted:
        pass
    patched_rec = {b: patched_tr.core.recorded.get(b) for b in ckpts}
    diffs = []
    for b in ckpts:
        u, p = unpatched_rec[b], patched_rec[b]
        if u is None or p is None:
            diffs.append(f"{b}:missing")
        elif u != p:
            diffs.append(f"{b}:unp{u}!=pat{p}")
    verdict = "PASS" if not diffs else "FAIL"
    print(f"RESULT T3 {mode} {C.stem_of(name)} unpatched={unpatched_status} {verdict} "
          f"patched_q={patched_tr.queries} unpatched_q={unpatched_tr.queries} diffs={diffs or 'none'}", flush=True)
    return verdict


def main():
    rows_nt = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_non_targeted.csv")[:3]
    rows_t = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_targeted.csv")[:3]
    t0 = time.time()
    for i, r in enumerate(rows_nt):
        run_cell("non_targeted", r, i)
    for i, r in enumerate(rows_t):
        run_cell("targeted", r, i)
    print(f"DONE T3 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
