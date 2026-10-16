"""TEMPLATE - run ONE official baseline implementation on the shared split through the uniform hard-label tracker.

Copy to adapters/run_<method>.py and fill the three FILL functions after READING the official repository.
Rules (from the attack-eval-protocol skill):
  * import the official attack code; do not edit its algorithm (only the model object and the data it receives);
  * default hyper-parameters of the paper / repository, written to params.json and the env file;
  * same split CSV, pre-processing and victim weights as LGC-EGC (TtBA protocol: Resize 256 -> CenterCrop 224);
  * budget 10000, checkpoints 500 1000 2000 2500 5000 10000, every victim evaluation counted (tracker);
  * the attack sees ONLY top-1 decisions (tracker returns one-hot scores / labels);
  * fidelity check before the full run: 3-5 images with the repo's own script vs. this adapter (same numbers
    up to RNG) and, for TtBA, against the published Table 1 magnitude.

    python adapters/run_<method>.py --method SignOPT --arch resnet50 --mode targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_targeted.csv --data_dir /root/autodl-tmp/ttba_mini_flat \
        --repo /root/autodl-tmp/baselines/SignOPT --out_dir Results/b1_resnet50_official --limit 200
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402

BUDGETS = (500, 1000, 2000, 2500, 5000, 10000)
CKPT_FIELDS = ["arch", "mode", "image", "method", "budget", "found", "state_query", "pix_l2", "pix_linf",
               "adv_float", "adv_uint8"]
RUN_FIELDS = ["arch", "mode", "image", "method", "queries", "status", "wall_time_s", "error"]


# ----------------------------------------------------------------------------------------------- FILL 1
def load_victim(arch: str, ckpt_dir: str, device: str) -> torch.nn.Module:
    """Victim taking PIXEL input in [0, 1] (normalisation inside), eval mode, identical weights to our runner.
    Reuse the loader of run_benchmark_33b.py (--victim_ckpt_dir) or TtBA's GeneralTorchModel."""
    raise NotImplementedError


# ----------------------------------------------------------------------------------------------- FILL 2
def load_images(split_csv: str, data_dir: str, limit: int, device: str) -> Iterator[Tuple[str, Any, int, Any, Any]]:
    """Yield (image_name, x0 (1,3,224,224) in [0,1], y_src, x_tgt or None, y_tgt or None) in split order.
    Use EXACTLY the pre-processing of the TtBA protocol loader (the runner prints mean |x - loader| = 0)."""
    raise NotImplementedError


# ----------------------------------------------------------------------------------------------- FILL 3
def run_official(tracker: HardLabelTracker, x0: Any, y_src: int, x_tgt: Any, y_tgt: Any, params: dict) -> None:
    """Call the official attack with the tracker in place of the model.  Examples (adapt after reading the code):
        torch-module API : attack(model=tracker, ...)            -> tracker(x) returns one-hot scores
        predict_label API: model.predict_label = tracker.predict_label
        numpy API        : model.predict = lambda a: tracker.predict_numpy(a, channels_last=True)
    If the official code needs inputs in another space (e.g. normalised or 0-255), build the tracker with
    to_pixel=<map to [0,1]> so L2 / storage are always in pixel space.  Return when the attack ends."""
    raise NotImplementedError


def append(path: str, fields: list, rows: list) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def done_images(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {r["image"] for r in csv.DictReader(fh) if r.get("status") in ("ok", "budget")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", choices=["non_targeted", "targeted"], required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt_dir", default="/root/autodl-tmp/TtBA/model")
    ap.add_argument("--repo", required=True, help="official repository (its git commit is recorded)")
    ap.add_argument("--params", default="{}", help="JSON of hyper-parameters (defaults of the paper / repo)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_queries", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.repo))
    params = json.loads(a.params)
    run_dir = os.path.join(a.out_dir, a.mode)
    os.makedirs(run_dir, exist_ok=True)
    ck = os.path.join(run_dir, f"per_checkpoint_{a.method}.csv")
    rp = os.path.join(run_dir, f"runs_{a.method}.csv")
    with open(os.path.join(run_dir, f"env_{a.method}.json"), "w") as fh:
        commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        json.dump({"method": a.method, "repo": os.path.abspath(a.repo), "repo_commit": commit, "params": params,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "adapter_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                   "budgets": BUDGETS, "max_queries": a.max_queries, "split_csv": os.path.abspath(a.split_csv)},
                  fh, indent=1)
    victim = load_victim(a.arch, a.ckpt_dir, a.device)
    done = done_images(rp)
    for idx, (name, x0, y_src, x_tgt, y_tgt) in enumerate(load_images(a.split_csv, a.data_dir, a.limit, a.device)):
        stem = os.path.splitext(name)[0]
        if stem in done:
            continue
        s = a.seed * 1000003 + idx  # per-image seed: resume-safe and order-independent
        random.seed(s), np.random.seed(s % (2 ** 32)), torch.manual_seed(s)
        tracker = HardLabelTracker(victim, x0, y_src, y_tgt if a.mode == "targeted" else None, BUDGETS,
                                   a.max_queries, init_adv_pixel=x_tgt if a.mode == "targeted" else None)
        t0, status, err = time.time(), "ok", ""
        try:
            run_official(tracker, x0, y_src, x_tgt, y_tgt, params)
        except BudgetExhausted:
            status = "budget"
        except Exception as e:  # noqa: BLE001  (keep the run alive; the image is retried on resume)
            traceback.print_exc()
            status, err = "error", f"{type(e).__name__}: {e}"
        rows = tracker.finalize(os.path.join(run_dir, "adv_store"), a.arch, stem, a.method, a.mode, eval_model=victim)
        append(ck, CKPT_FIELDS, rows)
        tracker.save_trace(os.path.join(run_dir, "traces", a.arch, stem, f"{a.method}.npz"))
        append(rp, RUN_FIELDS, [{"arch": a.arch, "mode": a.mode, "image": stem, "method": a.method,
                                 "queries": tracker.queries, "status": status,
                                 "wall_time_s": round(time.time() - t0, 2), "error": err}])
        last = rows[-1]
        print(f"[{idx + 1}] {stem} {a.method} q={tracker.queries} L2@{last['budget']}={last['pix_l2']:.4f} "
              f"{status} {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
