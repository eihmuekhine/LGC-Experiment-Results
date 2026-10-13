#!/usr/bin/env python3
"""B4e2 F1 — CPU sandbox test for the resume-safe clip_events_182 merge in adapters/run_cgba.py.

Regression for the checker_review finding: ``_write_clip_events`` used to merge ``_CLIP_EVENTS`` by row index,
so on a resumed CGBA/CGBA_H run the counts landed on the wrong rows (done images are skipped without appending
to ``_CLIP_EVENTS``).  The fix snapshots the image stems already present in runs_<method>.csv BEFORE C.run and
attaches the new ``_CLIP_EVENTS`` entries to the NEW rows only, in processing order.

This test builds a runs_CGBA.csv with one pre-existing row (image A, status ok, no clip_events_182 column),
snapshots the pre-existing stems via ``_existing_stems``, appends two new rows (images B, C) exactly as C.run
would, then merges two counts via ``_write_clip_events`` and asserts:
  (a) A's clip_events_182 is untouched (empty — it had no column);
  (b) B and C receive the counts in processing order.

    python adapters/diag/diag_b4e2_t6_resume.py
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import run_cgba as A  # noqa: E402

RUN_FIELDS = ["arch", "mode", "image", "method", "queries", "nonfinite_queries", "status", "wall_time_s", "error"]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        mode = "non_targeted"
        run_path = os.path.join(d, mode, "runs_CGBA.csv")
        os.makedirs(os.path.dirname(run_path), exist_ok=True)

        # 1. pre-existing row (image A, status ok, no clip_events_182 column)
        with open(run_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerow({"arch": "resnet50", "mode": mode, "image": "A", "method": "CGBA",
                        "queries": "10000", "nonfinite_queries": "0", "status": "ok",
                        "wall_time_s": "20.0", "error": ""})

        # 2. snapshot the stems already present BEFORE C.run (what main() now does)
        pre_existing = A._existing_stems(d, mode, "CGBA")

        # 3. C.run attacks only the NEW images (B, C) and appends their rows in processing order
        with open(run_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
            for img in ("B", "C"):
                w.writerow({"arch": "resnet50", "mode": mode, "image": img, "method": "CGBA",
                            "queries": "10000", "nonfinite_queries": "0", "status": "budget",
                            "wall_time_s": "21.0", "error": ""})

        # 4. the per-image clip-event counts collected by run_official, in processing order
        A._CLIP_EVENTS[:] = [7, 13]

        # 5. merge (what main() does after C.run returns)
        A._write_clip_events("CGBA", d, mode, pre_existing)

        with open(run_path, newline="") as fh:
            rows = {r["image"]: r for r in csv.DictReader(fh)}

        checks = [
            ("pre-existing A untouched (empty)", rows["A"].get("clip_events_182", "") == ""),
            ("new B gets first count", rows["B"].get("clip_events_182") == "7"),
            ("new C gets second count", rows["C"].get("clip_events_182") == "13"),
            ("snapshot captured only A", pre_existing == {"A"}),
            ("column name preserved", "clip_events_182" in rows["A"]),
        ]
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
        if not all(ok for _, ok in checks):
            print("rows:", {k: dict(v) for k, v in rows.items()})
            print("VERDICT: FAIL")
            return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
