#!/usr/bin/env python3
"""Tests for tools/eval/g5_check.py (D13): synthetic PASS, synthetic FAIL, image-pairing with a missing
image, and n < 30 refusal.

    python tools/eval/tests_g5/test_g5.py      -> prints PASS/FAIL per test, exit 0 if all pass

Pure stdlib + numpy; no torch, no GPU.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVAL = os.path.join(REPO, "tools", "eval")
sys.path.insert(0, EVAL)

import g5_check as g5  # noqa: E402

BUDGETS = [2000, 5000, 10000]
RESULT: list = []


def check(name: str, cond: bool) -> None:
    RESULT.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


def write_ck(root, mode, stem, images, field, val_fn):
    d = os.path.join(root, mode)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"per_checkpoint_{stem}.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "mode", "image", "method", "budget", "found", "state_query", "pix_l2",
                    "pix_linf", "adv_float", "adv_uint8", "source_label", "target_label", "ssim", "lpips",
                    "pix_l2_iter"])
        for img in images:
            for b in BUDGETS:
                found, pix_l2, pix_l2_iter = val_fn(img, b)
                w.writerow(["resnet50", mode, img, stem, b, found, b, f"{pix_l2:.6f}",
                            "0.0", "1", "1", "0", "", "", "",
                            f"{pix_l2_iter:.6f}" if field == "iter" else ""])


def write_table1(root, ttba_med, cgba_med):
    t = {"table1": {"resnet50": {}}}
    for meth, meds in (("TtBA", ttba_med), ("CGBA", cgba_med)):
        d = {}
        for b in (2000, 5000, 10000):
            d[str(b)] = [meds[b], meds[b]]
        t["table1"]["resnet50"][meth] = {"non_targeted": d}
    p = os.path.join(root, "ttba_table1.json")
    json.dump(t, open(p, "w"))
    return p


def run_g5(ttba_root, cgba_root, table1):
    cmd = [sys.executable, os.path.join(EVAL, "g5_check.py"), "--arch", "resnet50", "--mode", "non_targeted",
           "--ttba_root", ttba_root, "--table1", table1, "--n_boot", "2000", "--seed", "0"]
    if cgba_root is not None:
        cmd += ["--cgba_root", cgba_root]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def make_cell(tmp, stem, images, l2):
    """Write a cell dir containing only per_checkpoint_TtBA.csv (from `l2`) — the TtBA cell."""
    root = os.path.join(tmp, f"cell_{stem}")
    write_ck(root, "non_targeted", "TtBA", images, "iter",
             lambda img, b: (1, l2[img][b], l2[img][b]))
    return root


def make_cgba_cell(tmp, images, l2_cgba):
    """Write a cell dir containing only per_checkpoint_CGBA.csv — the CGBA cell."""
    root = os.path.join(tmp, "cell_cgba")
    write_ck(root, "non_targeted", "CGBA", images, "pix_l2",
             lambda img, b: (1, l2_cgba[img][b], 0.0))
    return root


def test_synthetic_pass():
    print("[1] synthetic PASS (n=30, both ratios near 1)")
    tmp = tempfile.mkdtemp()
    n = 30
    images = [f"img{i:03d}.JPEG" for i in range(n)]
    ttba_med = {2000: 7.0, 5000: 2.8, 10000: 1.7}
    cgba_med = {2000: 12.7, 5000: 5.1, 10000: 2.5}
    l2_ttba = {img: {b: ttba_med[b] * (1 + 0.01 * (i % 5)) for b in ttba_med} for i, img in enumerate(images)}
    l2_cgba = {img: {b: cgba_med[b] * (1 + 0.01 * (i % 5)) for b in cgba_med} for i, img in enumerate(images)}
    ttba_root = make_cell(tmp, "ttba", images, l2_ttba)
    cgba_root = make_cgba_cell(tmp, images, l2_cgba)
    table1 = write_table1(tmp, ttba_med, cgba_med)
    rc, out = run_g5(ttba_root, cgba_root, table1)
    check("rc==0 (PASS)", rc == 0)
    check("verdict PASS in output", "VERDICT: PASS" in out)


def test_synthetic_fail():
    print("[2] synthetic FAIL (CGBA off by 5x -> rho CI misses [0.5,2])")
    tmp = tempfile.mkdtemp()
    n = 30
    images = [f"img{i:03d}.JPEG" for i in range(n)]
    ttba_med = {2000: 7.0, 5000: 2.8, 10000: 1.7}
    cgba_med = {2000: 12.7, 5000: 5.1, 10000: 2.5}
    # CGBA 5x too large -> rho ~5
    l2_ttba = {img: {b: ttba_med[b] * (1 + 0.01 * (i % 5)) for b in ttba_med} for i, img in enumerate(images)}
    l2_cgba = {img: {b: cgba_med[b] * 5.0 * (1 + 0.01 * (i % 5)) for b in cgba_med} for i, img in enumerate(images)}
    ttba_root = make_cell(tmp, "ttba", images, l2_ttba)
    cgba_root = make_cgba_cell(tmp, images, l2_cgba)
    table1 = write_table1(tmp, ttba_med, cgba_med)
    rc, out = run_g5(ttba_root, cgba_root, table1)
    check("rc==1 (FAIL)", rc == 1)
    check("verdict FAIL in output", "VERDICT: FAIL" in out)


def test_missing_image_pairing():
    print("[3] image-pairing with a missing image (one CGBA image absent -> excluded, no crash)")
    tmp = tempfile.mkdtemp()
    n = 31
    images = [f"img{i:03d}.JPEG" for i in range(n)]
    ttba_med = {2000: 7.0, 5000: 2.8, 10000: 1.7}
    cgba_med = {2000: 12.7, 5000: 5.1, 10000: 2.5}
    l2_ttba = {img: {b: ttba_med[b] * (1 + 0.01 * (i % 5)) for b in ttba_med} for i, img in enumerate(images)}
    l2_cgba = {img: {b: cgba_med[b] * (1 + 0.01 * (i % 5)) for b in cgba_med}
               for i, img in enumerate(images) if img != images[3]}  # drop one CGBA image
    ttba_root = make_cell(tmp, "ttba", images, l2_ttba)
    cgba_root = make_cgba_cell(tmp, [i for i in images if i != images[3]], l2_cgba)
    table1 = write_table1(tmp, ttba_med, cgba_med)
    rc, out = run_g5(ttba_root, cgba_root, table1)
    check("rc==0 (PASS, pairing handled)", rc == 0)
    check("verdict PASS in output", "VERDICT: PASS" in out)


def test_n_below_30_refusal():
    print("[4] n < 30 -> 'tool check only'")
    tmp = tempfile.mkdtemp()
    n = 3
    images = [f"img{i:03d}.JPEG" for i in range(n)]
    ttba_med = {2000: 7.0, 5000: 2.8, 10000: 1.7}
    cgba_med = {2000: 12.7, 5000: 5.1, 10000: 2.5}
    l2_ttba = {img: {b: ttba_med[b] for b in ttba_med} for img in images}
    l2_cgba = {img: {b: cgba_med[b] for b in cgba_med} for img in images}
    ttba_root = make_cell(tmp, "ttba", images, l2_ttba)
    cgba_root = make_cgba_cell(tmp, images, l2_cgba)
    table1 = write_table1(tmp, ttba_med, cgba_med)
    rc, out = run_g5(ttba_root, cgba_root, table1)
    check("rc==0 (tool check only)", rc == 0)
    check("'tool check only' in output", "tool check only" in out)


def main() -> int:
    test_synthetic_pass()
    test_synthetic_fail()
    test_missing_image_pairing()
    test_n_below_30_refusal()
    n_fail = sum(1 for _, ok in RESULT if not ok)
    print(f"\n{n_fail} failed / {len(RESULT)} checks")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
