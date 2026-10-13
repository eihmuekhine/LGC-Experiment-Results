#!/usr/bin/env python3
"""B4e2 F1 + G1 — CPU sandbox test for the resume-safe clip_events_182 merge in adapters/run_cgba.py.

Regression for the checker_review finding (F1): ``_write_clip_events`` used to merge ``_CLIP_EVENTS`` by row
index, so on a resumed CGBA/CGBA_H run the counts landed on the wrong rows (done images are skipped without
appending to ``_CLIP_EVENTS``).  The fix snapshots the number of rows already present in runs_<method>.csv
BEFORE C.run and attaches the new ``_CLIP_EVENTS`` entries to the rows appended THIS run only (index >= snapshot
count), in processing order.

Scenario (c) — done-image resume (the original F1 scenario): one pre-existing row (image A, status ok, no
clip_events_182 column), then two new rows (B, C).  Merges counts [7, 13] and asserts A is untouched and B/C
receive the counts in order.

Scenario (b) — error-row resume (G1, checker_review_followup finding 1): one pre-existing row (image X, status
error, no clip_events_182), then the re-attacked X (new row, same stem) and a new row Y.  Merges counts [5, 9]
and asserts the OLD error row stays empty while X-new receives 5 and Y receives 9 (the reviewer's demo now passes).

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


def _row(arch="resnet50", mode="non_targeted", image="", method="CGBA", queries="10000",
         nonfinite="0", status="ok", wall="20.0", error=""):
    return {"arch": arch, "mode": mode, "image": image, "method": method, "queries": queries,
            "nonfinite_queries": nonfinite, "status": status, "wall_time_s": wall, "error": error}


def _scenario_c(tmp) -> int:
    mode = "non_targeted"
    run_path = os.path.join(tmp, mode, "runs_CGBA.csv")
    os.makedirs(os.path.dirname(run_path), exist_ok=True)

    # 1. pre-existing row (image A, status ok, no clip_events_182 column)
    with open(run_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(_row(image="A", status="ok"))

    # 2. snapshot the number of rows already present BEFORE C.run (what main() now does)
    pre_existing = A._pre_existing_count(tmp, mode, "CGBA")

    # 3. C.run attacks only the NEW images (B, C) and appends their rows in processing order
    with open(run_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        for img in ("B", "C"):
            w.writerow(_row(image=img, status="budget"))

    # 4. the per-image clip-event counts collected by run_official, in processing order
    A._CLIP_EVENTS[:] = [7, 13]

    # 5. merge (what main() does after C.run returns)
    A._write_clip_events("CGBA", tmp, mode, pre_existing)

    with open(run_path, newline="") as fh:
        rows = {r["image"]: r for r in csv.DictReader(fh)}

    checks = [
        ("(c) pre-existing A untouched (empty)", rows["A"].get("clip_events_182", "") == ""),
        ("(c) new B gets first count", rows["B"].get("clip_events_182") == "7"),
        ("(c) new C gets second count", rows["C"].get("clip_events_182") == "13"),
        ("(c) snapshot captured count 1 (A)", pre_existing == 1),
        ("(c) column name preserved", "clip_events_182" in rows["A"]),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    return 0 if all(ok for _, ok in checks) else 1


def _scenario_b(tmp) -> int:
    mode = "non_targeted"
    run_path = os.path.join(tmp, mode, "runs_CGBA.csv")
    os.makedirs(os.path.dirname(run_path), exist_ok=True)

    # 1. pre-existing row (image X, status error, no clip_events_182 column)
    with open(run_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerow(_row(image="X", status="error", queries="500"))

    # 2. snapshot the number of rows already present BEFORE C.run
    pre_existing = A._pre_existing_count(tmp, mode, "CGBA")

    # 3. C.run re-attacks the errored X (appends a SECOND row with the same stem) then attacks Y
    with open(run_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writerow(_row(image="X", status="ok"))
        w.writerow(_row(image="Y", status="budget"))

    # 4. the per-image clip-event counts collected by run_official, in processing order
    A._CLIP_EVENTS[:] = [5, 9]

    # 5. merge
    A._write_clip_events("CGBA", tmp, mode, pre_existing)

    with open(run_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    checks = [
        ("(b) snapshot captured count 1 (error X)", pre_existing == 1),
        ("(b) old error X untouched (empty)", rows[0].get("clip_events_182", "") == ""),
        ("(b) X-new gets first count", rows[1].get("clip_events_182") == "5"),
        ("(b) Y gets second count", rows[2].get("clip_events_182") == "9"),
        ("(b) column name preserved", "clip_events_182" in rows[0]),
    ]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if not all(ok for _, ok in checks):
        print("rows:", [dict(r) for r in rows])
    return 0 if all(ok for _, ok in checks) else 1


def main() -> int:
    rc = 0
    with tempfile.TemporaryDirectory() as d:
        print("--- scenario (c) done-image resume ---")
        rc |= _scenario_c(d)
    with tempfile.TemporaryDirectory() as d:
        print("--- scenario (b) error-row resume ---")
        rc |= _scenario_b(d)
    print("VERDICT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
