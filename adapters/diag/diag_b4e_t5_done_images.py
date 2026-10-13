#!/usr/bin/env python3
"""B4e2 T5 — sandbox unit test for ttba_github.ttba_common.done_images (contract D6 resume fix).

After the fix, ``done_images`` treats an image as done if it has ANY row in runs_*.csv, whatever its status
(ok / budget / error).  An image with no row is not done (it runs).

    python adapters/diag/diag_b4e_t5_done_images.py
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "ttba_github"))

from ttba_common import done_images  # noqa: E402

FIELDS = ["image", "status", "queries"]


def write_runs(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        runs = os.path.join(d, "runs_TtBA.csv")
        write_runs(runs, [
            {"image": "img_ok", "status": "ok", "queries": "10000"},
            {"image": "img_budget", "status": "budget", "queries": "10000"},
            {"image": "img_error", "status": "error", "queries": "500"},
        ])
        got = done_images(runs)
        expect = {"img_ok", "img_budget", "img_error"}
        checks = [
            ("ok row skipped", "img_ok" in got),
            ("budget row skipped", "img_budget" in got),
            ("error row skipped", "img_error" in got),
            ("missing image not done (runs)", "img_missing" not in got),
            ("missing image not done (no file)", done_images(os.path.join(d, "nope.csv")) == set()),
        ]
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
        print("done_images returned:", sorted(got))
        if not all(ok for _, ok in checks) or got != expect:
            print("VERDICT: FAIL")
            return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
