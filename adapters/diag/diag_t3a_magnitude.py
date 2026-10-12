#!/usr/bin/env python3
"""T3(a): mean/median pix_l2 @2000/5000/10000 for CGBA / CGBA_H / TtBA per victim/mode from b4_smoke (n=3).

Quotes the keys of ttba_github/ttba_table1.json if present; otherwise reports MISSING/UNVERIFIED.

    python adapters/diag/diag_t3a_magnitude.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")
BUDGETS = [2000, 5000, 10000]
ARCHS = ["resnet50", "vgg19", "vit_b_32"]


def load(root: str, mode: str, method: str) -> dict:
    p = os.path.join(R, root, mode, f"per_checkpoint_{method}.csv")
    vals = {b: [] for b in BUDGETS}
    if not os.path.exists(p):
        return vals
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            b = int(r["budget"])
            if b in vals and int(r.get("found", 0)) == 1:
                vals[b].append(float(r["pix_l2"]))
    return vals


def main() -> int:
    table1_path = os.path.join(REPO, "ttba_github", "ttba_table1.json")
    table1 = None
    if os.path.exists(table1_path):
        table1 = json.load(open(table1_path))
        print("ttba_table1.json keys:", sorted(table1.keys()))
    else:
        print("ttba_table1.json: MISSING (UNVERIFIED) — no published ratios computable.")

    print(f"\n{'method':8s} {'arch':9s} {'mode':12s} " +
          " ".join(f"@{b:>5d}" for b in BUDGETS))
    print("-" * 90)
    for meth, mode, mfile in [("CGBA", "non_targeted", "CGBA"),
                              ("CGBA_H", "targeted", "CGBA_H"),
                              ("TtBA", "non_targeted", "TtBA"),
                              ("TtBA", "targeted", "TtBA")]:
        for arch in ARCHS:
            root = f"b4_smoke_cgba_{arch}" if meth in ("CGBA", "CGBA_H") else f"b4_smoke_ttba_{arch}"
            vals = load(root, mode, mfile)
            cells = []
            for b in BUDGETS:
                v = vals[b]
                cells.append(f"mean={st.mean(v):.2f} med={st.median(v):.2f}" if v else "none")
            print(f"{meth:8s} {arch:9s} {mode:12s} " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
