#!/usr/bin/env python3
"""B4d T4(a) — CGBA / CGBA_H stall scan (ZERO GPU, reads existing smoke results).

For every CGBA (non_targeted) / CGBA_H (targeted) smoke cell (B4, B4b, and the B4c/T1 reruns), per image:
  pix_l2 @1000/2000/5000/10000, last-improvement state_query (state_query at budget 10000),
  stalled = (pix_l2@2000 == pix_l2@10000) i.e. best-so-far flat from 2k to 10k.

    python adapters/diag/diag_b4d_t4a_stall.py
"""
from __future__ import annotations

import csv
import glob
import os

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")
BUDGETS = [1000, 2000, 5000, 10000]

# (label, root, mode, method-file)
CELLS = [
    ("B4 resnet50 NT", "b4_smoke_cgba_resnet50", "non_targeted", "CGBA"),
    ("B4 resnet50 T",  "b4_smoke_cgba_resnet50", "targeted", "CGBA_H"),
    ("B4 vgg19 NT",    "b4_smoke_cgba_vgg19",    "non_targeted", "CGBA"),
    ("B4 vgg19 T",     "b4_smoke_cgba_vgg19",    "targeted", "CGBA_H"),
    ("B4 vit_b_32 NT", "b4_smoke_cgba_vit_b_32", "non_targeted", "CGBA"),
    ("B4 vit_b_32 T",  "b4_smoke_cgba_vit_b_32", "targeted", "CGBA_H"),
    ("B4b resnet50_places365 NT", "b4b_smoke_CGBA_resnet50_places365", "non_targeted", "CGBA"),
    ("B4b resnet50_places365 T",  "b4b_smoke_CGBA_H_resnet50_places365", "targeted", "CGBA_H"),
    ("B4b densenet161_places365 NT", "b4b_smoke_CGBA_densenet161_places365", "non_targeted", "CGBA"),
    ("B4b densenet161_places365 T",  "b4b_smoke_CGBA_H_densenet161_places365", "targeted", "CGBA_H"),
    ("B4b Salman2020Do_R50 NT", "b4b_smoke_CGBA_Salman2020Do_R50", "non_targeted", "CGBA"),
    ("B4b Salman2020Do_R50 T",  "b4b_smoke_CGBA_H_Salman2020Do_R50", "targeted", "CGBA_H"),
    ("T1 rerun resnet50 NT", "b4c_regress_cgba", "non_targeted", "CGBA"),
    ("T1 rerun resnet50 T",  "b4c_regress_cgba_h", "targeted", "CGBA_H"),
]


def load(root, mode, mfile):
    p = os.path.join(R, root, mode, f"per_checkpoint_{mfile}.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            b = int(r["budget"])
            if b not in BUDGETS:
                continue
            img = r["image"]
            out.setdefault(img, {})[b] = r
    return out


def main() -> int:
    n_stall = 0
    for label, root, mode, mfile in CELLS:
        rows = load(root, mode, mfile)
        if not rows:
            print(f"{label:32s}: (no results)")
            continue
        for img in sorted(rows):
            rec = rows[img]
            l2 = {b: float(rec[b]["pix_l2"]) if int(rec[b].get("found", 0)) == 1 else float("nan")
                  for b in BUDGETS}
            sq = rec[10000].get("state_query", "-")
            stalled = (l2[2000] == l2[10000])
            if stalled:
                n_stall += 1
            print(f"{label:32s} {img:24s} 1k={l2[1000]:8.3f} 2k={l2[2000]:8.3f} "
                  f"5k={l2[5000]:8.3f} 10k={l2[10000]:8.3f} state_query@10k={sq} stalled={stalled}")
    print(f"\nstalled images (pix_l2@2000 == pix_l2@10000): {n_stall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
