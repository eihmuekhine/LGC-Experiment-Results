#!/usr/bin/env python3
"""B3b T5 — bypass audit: raw-victim forward image counter vs tracker query count (row 1 non_targeted).

For TtBA, CGBA and ARS-OPT the raw victim forward is wrapped with an image counter (counts batch elements); after the
attack the counter must equal tracker.queries -- i.e. every tracker query issues exactly one victim classification and
nothing classifies an image outside the tracker.  The tracker.finalize() eval_model calls (uint8 re-check) are then
reported separately (they are uncounted by design).

Also reports each method's own query counter for T4: CGBA all_queries, TtBA Attacker.queries (attributes);
ARS-OPT has no attribute counter (returned as out[1]).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import _common as C  # noqa: E402
import run_cgba as A  # noqa: E402
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
import torch  # noqa: E402

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
BUDGET = 10000
ARCH = "resnet50"
name = "ILSVRC2012_val_00007450.JPEG"


class Counter:
    def __init__(self):
        self.images = 0
        self.calls = 0

    def wrap(self, mod):
        orig = mod.forward

        def fwd(x, *a, **k):
            self.calls += 1
            self.images += int(x.shape[0])
            return orig(x, *a, **k)

        mod.forward = fwd
        return mod


def cgba():
    sys.path.insert(0, "/root/autodl-tmp/baselines/CGBA")
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = A._faithful_model(victim)
    rows = C.load_split_rows(os.path.join(REPO_ROOT, "splits", "imagenet", "ttba_github", f"{ARCH}_non_targeted.csv"))[:1]
    r = rows[0]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, r["source"], "cuda")
    ckpts = C.contract_checkpoints(C.load_contract())
    params = dict(A.PARAMS, attack_method="CGBA")

    ctr = Counter()
    ctr.wrap(raw)
    tracker = A.make_tracker(victim, x0, y_src, None, None, ckpts, BUDGET)
    C.seed_all(0)
    attack = None
    try:
        Patched = A._patched_attack_class()
        src = C.normalize(x0)
        lb = C.normalize(torch.zeros_like(x0))
        ub = C.normalize(torch.ones_like(x0))
        attack = Patched(tracker, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=None,
                         dim_reduc_factor=params["dim_reduc_factor"], attack_method="CGBA",
                         iteration=params["iteration"], initial_query=params["initial_query"],
                         tol=params["tol"], sigma=params["sigma"], verbose_control=params["verbose_control"])
        attack.Attack()
    except BudgetExhausted:
        pass
    atk_time = ctr.images
    ok = (atk_time == tracker.queries)
    official_q = attack.all_queries if attack is not None else None
    before = ctr.images
    tracker.finalize(None, ARCH, C.stem_of(name), "CGBA", "non_targeted", eval_model=victim)
    eval_calls = ctr.images - before
    print(f"RESULT T5 CGBA {'PASS' if ok else 'FAIL'} atk_images={atk_time} tracker_q={tracker.queries} "
          f"official_all_queries={official_q} finalize_eval_calls={eval_calls}", flush=True)


def arsopt():
    sys.path.insert(0, "/root/autodl-tmp/baselines/ARS-OPT")
    import run_arsopt as RA

    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = victim.m
    rows = C.load_split_rows(os.path.join(REPO_ROOT, "splits", "imagenet", "ttba_github", f"{ARCH}_non_targeted.csv"))[:1]
    r = rows[0]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, r["source"], "cuda")
    ckpts = C.contract_checkpoints(C.load_contract())
    RA._ARCH = ARCH
    RA._TARGETED = False
    params = dict(RA.BASE_PARAMS, lr=0.4)

    ctr = Counter()
    ctr.wrap(raw)
    tracker = RA.make_tracker(victim, x0, y_src, None, None, ckpts, BUDGET)
    C.seed_all(0)
    out = None
    try:
        RA._install_shims()
        from PARS_OPT.pars_opt_l2_norm_attack import ParsOptL2Norm

        attacker = ParsOptL2Norm(tracker, [], "ImageNet", params["epsilon"], False, batch_size=1,
                                 k=params["k"], alpha=params["lr"], beta=params["beta"],
                                 iterations=params["iterations"], maximum_queries=params["maximum_queries"],
                                 method="ARS", clip_grad_max_norm=params["clip_grad_max_norm"], tol=params["tol"],
                                 grad_binary_search_tol=None, best_initial_target_sample=False, resized=False,
                                 dim_reduce_factor=params["dim_reduce_factor"], line_search=False)
        true_labels = torch.tensor([int(y_src)], device=x0.device)
        out = attacker.untargeted_attack(0, x0, true_labels)
    except BudgetExhausted:
        pass
    atk_time = ctr.images
    ok = (atk_time == tracker.queries)
    official_q = int(out[1][0].item()) if out is not None else None
    before = ctr.images
    tracker.finalize(None, ARCH, C.stem_of(name), "ARS_OPT", "non_targeted", eval_model=victim)
    eval_calls = ctr.images - before
    print(f"RESULT T5 ARS-OPT {'PASS' if ok else 'FAIL'} atk_images={atk_time} tracker_q={tracker.queries} "
          f"official_q(return out[1])={official_q} finalize_eval_calls={eval_calls}", flush=True)


def ttba():
    sys.path.insert(0, "/root/autodl-tmp/TtBA")
    os.chdir("/root/autodl-tmp/TtBA")  # dataset.py reads model/<file> by relative path
    import dataset as ttds
    import TtBA as ttba

    test_loader, torch_model = ttds.load_dataset_model("imagenet-resnet50")
    ds = test_loader.dataset
    by_name = {os.path.basename(p): i for i, (p, _) in enumerate(ds.samples)}
    img, lab = ds[by_name[name]]
    x0 = img.unsqueeze(0).cuda()
    y_src = int(lab)
    ckpts = C.contract_checkpoints(C.load_contract())

    ctr = Counter()
    ctr.wrap(torch_model.model)  # DataParallel.forward: the actual victim classification
    tracker = HardLabelTracker(C.ClampedVictim(torch_model), x0, y_src, None, ckpts, BUDGET,
                               to_pixel=C.to_pixel_clamp)
    args = SimpleNamespace(budget=BUDGET, epsilon=2.5, early=0, remember=0, targeted=0, norm="TtBA")
    C.seed_all(0)
    atk = ttba.Attacker(args, tracker, x0, 0, "TtBA", None, None)
    try:
        atk.attack()
    except BudgetExhausted:
        pass
    atk_time = ctr.images
    ok = (atk_time == tracker.queries)
    before = ctr.images
    tracker.finalize(None, ARCH, C.stem_of(name), "TtBA", "non_targeted", eval_model=torch_model)
    eval_calls = ctr.images - before
    print(f"RESULT T5 TtBA {'PASS' if ok else 'FAIL'} atk_images={atk_time} tracker_q={tracker.queries} "
          f"official_Attacker_queries={atk.queries} finalize_eval_calls={eval_calls}", flush=True)


METHODS = {"ttba": ttba, "cgba": cgba, "arsopt": arsopt}


def main():
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(METHODS))
    args = ap.parse_args()
    if args.only:
        METHODS[args.only]()
        return
    t0 = time.time()
    for m in METHODS:
        subprocess.run([sys.executable, os.path.abspath(__file__), "--only", m], check=True)
    print(f"DONE T5 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
