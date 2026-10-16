#!/usr/bin/env python3
"""B3b T4 — TtBA completeness: adapter pix_l2_iter vs official pix_l2_reported at all 8 checkpoints, 6 cells.

Read-only comparison of the two already-written result sets:
  * adapter:  Results/b3_ttba_adapter/{mode}/per_checkpoint_TtBA.csv  (column pix_l2_iter, recorded by
              run_ttba_v2.py from the official Attacker.L2_line via SnapAttacker/CheckpointRecorder);
  * official: Results/b3_ttba_official/{mode}/ttbagh_per_checkpoint.csv (column pix_l2_reported, recorded by
              ttba_github/ttba_run_official.py from the same Attacker.L2_line).

Both record the official iterate (unclamped bisection distance) of the last completed iteration with queries <= c;
they must be bit-identical.  Also reports the queries official/tracker column (TtBA Attacker.queries attribute).
"""
from __future__ import annotations

import csv
import os

REPO = "/root/autodl-tmp/LGC_frozen"
AD = os.path.join(REPO, "Results", "b3_ttba_adapter")
OF = os.path.join(REPO, "Results", "b3_ttba_official")
CKPT = [100, 200, 500, 1000, 2000, 2500, 5000, 10000]


def load_ad(mode):
    rows = {}
    with open(os.path.join(AD, mode, "per_checkpoint_TtBA.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["image"], int(r["budget"]))] = (r["pix_l2_iter"], r["state_query"])
    return rows


def load_of(mode):
    rows = {}
    with open(os.path.join(OF, mode, "ttbagh_per_checkpoint.csv"), newline="") as f:
        for r in csv.DictReader(f):
            stem = r["image"].replace(".JPEG", "")
            rows[(stem, int(r["budget"]))] = (r["pix_l2_reported"], r["state_query"])
    return rows


def load_runs(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["image"].replace(".JPEG", "")] = (r.get("total_queries"), r.get("queries"))
    return out


def main():
    ok = True
    for mode in ("non_targeted", "targeted"):
        ad = load_ad(mode)
        of = load_of(mode)
        images = sorted({k[0] for k in ad})
        print(f"== {mode} (adapter pix_l2_iter vs official pix_l2_reported) ==")
        for img in images:
            diffs = []
            for b in CKPT:
                a = ad.get((img, b))
                o = of.get((img, b))
                if a is None or o is None:
                    diffs.append(f"{b}:missing(ad={a is not None},of={o is not None})")
                    continue
                if a[0] != o[0]:
                    diffs.append(f"{b}:ad={a[0]}!=of={o[0]}")
                    ok = False
            print(f"  {img}: {'PASS (identical at all 8)' if not diffs else 'FAIL ' + '; '.join(diffs)}")

    print()
    print("== queries official/tracker (TtBA Attacker.queries attribute) ==")
    for mode in ("non_targeted", "targeted"):
        runs = load_runs(os.path.join(OF, mode, "ttbagh_runs.csv"))
        ad_runs = load_runs(os.path.join(AD, mode, "runs_TtBA.csv"))
        for img in sorted(runs):
            oq = runs[img][0]
            aq = ad_runs.get(img)
            print(f"  {mode} {img}: official total_queries={oq} tracker(adapter runs)={aq[1] if aq else '?'}")

    print()
    print("T4 VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
