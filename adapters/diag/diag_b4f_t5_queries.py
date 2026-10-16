#!/usr/bin/env python3
"""B4f T5(d,e): per-cell query accounting from the B4e2 smoke runs.

(d) per cell: images with queries == budget, < budget (with status), > budget.
(e) tracker count vs official counter where one is recorded (TtBA total_queries = official atk.queries).

Read-only.  Usage: python adapters/diag/diag_b4f_t5_queries.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

BUDGET = 10000


def read_runs(md: str, prefix: str) -> list:
    out = []
    for f in sorted(glob.glob(os.path.join(md, f"{prefix}*.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_file"] = os.path.basename(f)
                out.append(r)
    return out


def qcol(r: dict) -> int:
    if "queries" in r and r["queries"] not in ("", None):
        return int(float(r["queries"]))
    if "total_queries" in r and r["total_queries"] not in ("", None):
        return int(float(r["total_queries"]))
    return None


def main() -> int:
    roots_txt = os.path.join(REPO_ROOT, "reports", "B4e2_smoke_roots.txt")
    roots = [ln.strip() for ln in open(roots_txt) if ln.strip()]
    print(f"[*] roots: {len(roots)}")
    print("\n## (d) per-cell query accounting (budget 10000)\n")
    print("| root/mode | method | n | ==10000 | <10000 (status) | >10000 |")
    print("|---|---|---|---|---|---|")
    bad = 0
    for root in roots:
        for md in sorted(glob.glob(os.path.join(REPO_ROOT, root, "*"))):
            if not os.path.isdir(md):
                continue
            runs = [r for r in read_runs(md, "runs") if "method" in r]
            if not runs:
                continue
            meths = {}
            for r in runs:
                meths.setdefault(r["method"], []).append(r)
            for meth, g in meths.items():
                qs = []
                for r in g:
                    q = qcol(r)
                    qs.append((q, r.get("status", "?")))
                eq = sum(1 for q, s in qs if q == BUDGET)
                lo = [(s) for q, s in qs if q is not None and q < BUDGET]
                hi = sum(1 for q, s in qs if q is not None and q > BUDGET)
                lo_str = ", ".join(f"{s}" for s in lo) if lo else "-"
                if hi:
                    bad += 1
                print(f"| {root}/{os.path.basename(md)} | {meth} | {len(g)} | {eq} | {lo_str} | {hi} |")

    print("\n## (e) tracker vs official counter (TtBA: queries=tracker, total_queries=official atk.queries)\n")
    print("| root/mode | image | tracker queries | official total_queries | offset (tracker-official) |")
    print("|---|---|---|---|---|")
    for root in roots:
        for md in sorted(glob.glob(os.path.join(REPO_ROOT, root, "*"))):
            for r in read_runs(md, "runs_TtBA"):
                q = int(float(r["queries"]))
                oq = int(float(r.get("total_queries", -1)))
                print(f"| {root}/{os.path.basename(md)} | {r['image']} | {q} | {oq} | {q - oq} |")
    print(f"\n[*] cells with any >budget image: {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
