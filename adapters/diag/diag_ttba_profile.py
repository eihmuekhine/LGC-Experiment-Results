#!/usr/bin/env python3
"""B4d T2 — TtBA DataParallel diagnosis (OBSERVATION ONLY, changes nothing).

Run the OFFICIAL TtBA Attacker (exact main.py path, via ttba_github/ttba_run_official.py ==
adapters/run_ttba_v2.py) on resnet50 NT row 1 and, at the point the HardLabelTracker calls the victim,
print:
  * the model object chain (tracker.model -> ... -> raw net) and its types,
  * whether an ``nn.DataParallel`` forward is in the call chain,
  * the per-query victim-forward wall time (median over 200 real attack queries),
  * a control: median forward time of the SAME resnet50 WITHOUT DataParallel (``.module``) over 200 calls,
    to separate DataParallel dispatch overhead from precision/backend effects.

Does not change the attack, seeds, hyper-parameters or any precision setting.

    python adapters/diag/diag_ttba_profile.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))
sys.path.insert(0, os.path.join(REPO_ROOT, "ttba_github"))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))

TTBA_REPO = "/root/autodl-tmp/TtBA"
sys.path.insert(0, TTBA_REPO)

import _common as C  # noqa: E402
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
from ttba_common import CheckpointRecorder, HookList, read_split_rows  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ARCH = "resnet50"
SPLIT = os.path.join(REPO_ROOT, "splits", "imagenet", "ttba_github", f"{ARCH}_non_targeted.csv")
DATASET = "imagenet-resnet50"
BUDGET = 10000


class TimingVictim(nn.Module):
    """Observation-only wrapper that records per-forward wall time around the real victim."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.times = []

    def forward(self, x):
        t0 = time.perf_counter()
        out = self.model(x)
        self.times.append(time.perf_counter() - t0)
        return out


def describe_chain(tracker):
    """Print the model object chain from tracker.model down to the raw net."""
    print("=== model object chain at predict_label ===")
    node = tracker.model
    depth = 0
    while node is not None and depth < 6:
        print(f"  {'  ' * depth}{type(node).__module__}.{type(node).__name__}")
        if isinstance(node, nn.DataParallel):
            node = node.module
        elif isinstance(node, (C.ClampedVictim,)):
            node = node.victim
        elif isinstance(node, TimingVictim):
            node = node.model
        elif isinstance(node, torch.nn.Sequential):
            node = None
        else:
            node = getattr(node, "model", None)
            if node is None:
                node = getattr(node, "net", None)
        depth += 1
    return tracker.model


def data_parallel_in_chain(tracker):
    node = tracker.model
    while node is not None:
        if isinstance(node, nn.DataParallel):
            return True
        if isinstance(node, C.ClampedVictim):
            node = node.victim
        elif isinstance(node, TimingVictim):
            node = node.model
        else:
            node = getattr(node, "model", None)
    return False


def main() -> int:
    import dataset as ttds  # official  (sets cudnn.deterministic=True at dataset.py:220)
    import TtBA as ttba  # official

    os.chdir(TTBA_REPO)
    test_loader, torch_model = ttds.load_dataset_model(DATASET)  # == main.py
    ds = test_loader.dataset
    it = iter(test_loader)
    next(it)
    by_name = {os.path.basename(p): i for i, (p, _) in enumerate(ds.samples)}

    print(f"torch_model.type = {type(torch_model).__module__}.{type(torch_model).__name__}")
    print(f"torch_model.model.type = {type(torch_model.model).__module__}.{type(torch_model.model).__name__}")
    print(f"torch_model.model.module.type = {type(torch_model.model.module).__module__}."
          f"{type(torch_model.model.module).__name__}")
    is_dp = isinstance(torch_model.model, nn.DataParallel)
    print(f"DataParallel in chain (nn.DataParallel): {is_dp}  (created at TtBA/dataset.py:293)")

    rows = read_split_rows(SPLIT)
    r = rows[0]
    name = r["source"]
    img, lab = ds[by_name[name]]
    if int(lab) != int(r["source_label"]):
        raise SystemExit(f"label mismatch: {lab} vs {r['source_label']}")
    x0 = img.unsqueeze(0).cuda()

    contract = C.load_contract()
    checkpoints = C.contract_checkpoints(contract)

    timing = TimingVictim(torch_model)
    tracker = HardLabelTracker(C.ClampedVictim(timing), x0, int(r["source_label"]), None,
                               checkpoints, BUDGET, to_pixel=C.to_pixel_clamp)

    print()
    print("=== DataParallel in the tracker call chain ===")
    describe_chain(tracker)
    print(f"DataParallel in tracker call chain: {data_parallel_in_chain(tracker)}")
    print()

    class SnapAttacker(ttba.Attacker):
        def __init__(self, *args, checkpoints=(), **kw):
            super().__init__(*args, **kw)
            self._last_bx = None
            self.recorder = CheckpointRecorder(checkpoints, self.L2_line[0][1])
            self.L2_line = HookList(self.L2_line, self)

        def bin_search_fast(self, adv_x, tol):
            out, l2, n = super().bin_search_fast(adv_x, tol)
            self._last_bx = out[1].detach().clone()
            return out, l2, n

    import argparse
    args = argparse.Namespace(budget=BUDGET, epsilon=2.5, early=0, remember=0, targeted=0, norm="TtBA")
    atk = SnapAttacker(args, tracker, x0, int(r.get("loader_pos", 0) or 0), "TtBA", None, None,
                       checkpoints=checkpoints)
    t0 = time.time()
    status = "ok"
    try:
        atk.attack()
    except BudgetExhausted:
        status = "budget"
    wall = time.time() - t0

    times = timing.times
    first200 = times[:200]
    med_all = statistics.median(times) if times else float("nan")
    med200 = statistics.median(first200) if first200 else float("nan")
    print(f"attack status={status} wall={wall:.2f}s queries={tracker.queries} "
          f"n_forward_calls_recorded={len(times)}")
    print(f"per-query victim-forward wall time: median(200)={med200*1e3:.3f} ms  "
          f"median(all)={med_all*1e3:.3f} ms")

    # --- control: SAME resnet50 WITHOUT DataParallel ---
    torch.cuda.synchronize()
    xnorm = torch_model.preprocess(x0.clamp(0.0, 1.0))
    raw_net = torch_model.model.module  # the resnet50 inside DataParallel
    t_dp, t_raw = [], []
    for _ in range(200):
        t0 = time.perf_counter(); torch_model.model(xnorm); torch.cuda.synchronize()
        t_dp.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); raw_net(xnorm); torch.cuda.synchronize()
        t_raw.append(time.perf_counter() - t0)
    med_dp = statistics.median(t_dp)
    med_raw = statistics.median(t_raw)
    print(f"control (200 calls, sync'd): DataParallel.forward median={med_dp*1e3:.3f} ms  "
          f"raw resnet50.forward median={med_raw*1e3:.3f} ms  "
          f"DataParallel overhead={(med_dp-med_raw)*1e3:.3f} ms/call ({100*(med_dp-med_raw)/med_dp:.1f}%)")

    print()
    print("CONCLUSION (T2): see report. DataParallel dispatch overhead is measured directly above; "
          "precision settings are the TtBA adapter defaults (cudnn.allow_tf32=True, cudnn.deterministic=True "
          "from dataset.py:220). NOTHING changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
