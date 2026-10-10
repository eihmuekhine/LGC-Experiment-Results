"""Self-tests that need only numpy / pandas / matplotlib / scipy (no torch, no GPU).

    python selftest_evalkit.py          -> prints PASS/FAIL per check, exit code 0 if all pass
torch-dependent parts are tested on AutoDL by:  python eval_metrics_offline.py --selftest
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from advstore import load_adv, store_adv  # noqa: E402
from hardlabel_tracker import TrackerCore  # noqa: E402

RESULTS = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


def test_core() -> None:
    print("[A] TrackerCore bookkeeping")
    c = TrackerCore(budgets=[3, 5, 10], max_queries=10)
    n, ev = c.process([5.0, 4.0, 6.0, 3.0], [False, True, True, True])     # q1..q4; best 4.0 @q2, 3.0 @q4
    check("A1 counts all allowed queries", n == 4 and c.queries == 4)
    check("A2 best is min L2 among adversarial queries", c.best_l2 == 3.0 and c.best_query == 4)
    check("A3 checkpoint 3 records the best AT q=3 (4.0), not the later 3.0", c.recorded[3] == (4.0, 2))
    check("A4 event order: improve(1), checkpoint(3), improve(3)",
          ev == [("improve", 1), ("checkpoint", 3), ("improve", 3)])
    n, ev = c.process([1.0] * 10, [False] * 10)                            # 6 allowed, then budget is spent
    check("A5 budget enforced inside a batch", n == 6 and c.queries == 10 and c.remaining == 0)
    check("A6 checkpoints 5 and 10 recorded with best 3.0", c.recorded[5] == (3.0, 4) and c.recorded[10] == (3.0, 4))
    c2 = TrackerCore(budgets=[2, 4, 8], max_queries=8)
    c2.process([2.0, 1.0, 9.0], [True, True, False])                       # stops early at q=3
    added = c2.carry_forward()
    check("A7 early stop: later budgets carried forward, none back-filled", added == [4, 8] and c2.recorded[2] == (1.0, 2))
    c3 = TrackerCore(budgets=[1, 2], max_queries=2)
    c3.set_initial(10.0)
    c3.process([12.0, 9.0], [True, False])
    check("A8 initial known adversarial (no query) kept until beaten", c3.recorded[1] == (10.0, 0) and c3.best_l2 == 10.0)
    c4 = TrackerCore(budgets=[1], max_queries=5)
    c4.process([0.5], [False])
    check("A9 no adversarial yet -> recorded L2 is inf (found = 0)", c4.recorded[1][0] == float("inf"))


def synthetic_metrics(path: str) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for arch in ("resnet50",):
        for mode in ("targeted",):
            for i in range(40):
                img = f"img{i:03d}"
                base_l2 = rng.lognormal(2.0, 0.4)
                for m, l2f, lpf, ssf in (("OURS", 0.9, 0.4, 0.02), ("B1", 1.0, 1.0, 0.0), ("B2", 1.2, 1.1, -0.01)):
                    for b in (500, 1000, 2000, 2500, 5000, 10000):
                        decay = (10000 / b) ** 0.5
                        l2 = base_l2 * l2f * decay * rng.lognormal(0, 0.05)
                        lp = min(0.9, 0.02 * l2 * lpf * rng.lognormal(0, 0.05))
                        rows.append({"root": "r", "arch": arch, "mode": mode, "image": img, "method": m, "budget": b,
                                     "state_query": b - 3, "adv_flag": 1, "l2_meta": l2, "l2_relerr": 0.0, "l2": l2,
                                     "linf": l2 / 50, "psnr": 30.0, "ssim": min(1.0, 0.97 + ssf - 0.002 * l2),
                                     "ms_ssim": 0.98, "lpips_alex": lp * 1.2, "lpips_vgg": lp, "dists": lp / 2,
                                     "fsim": 0.99})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_tables(tmp: str) -> None:
    print("[B] tables, paired statistics and figures")
    mcsv = os.path.join(tmp, "metrics.csv")
    synthetic_metrics(mcsv)
    out = os.path.join(tmp, "assets")
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_tables_figures.py"), "--metrics", mcsv, "--ours", "OURS",
                        "--order", "OURS", "B1", "B2", "--rename", "OURS=LGC-EGC (ours)", "--boot", "2000",
                        "--outdir", out], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    check("B1 script ran", r.returncode == 0)
    need = ["tab_l2_resnet50_targeted.tex", "tab_percep_resnet50_targeted.tex", "tab_paired_resnet50_targeted.tex",
            "tab_asrperc_resnet50_targeted.tex", "fig_curves_resnet50_targeted.pdf", "fig_pareto_resnet50_targeted.pdf",
            "summary_resnet50_targeted.json"]
    check("B2 all outputs written", all(os.path.exists(os.path.join(out, f)) for f in need))
    if os.path.exists(os.path.join(out, "summary_resnet50_targeted.json")):
        import json

        s = json.load(open(os.path.join(out, "summary_resnet50_targeted.json")))
        p = s["paired"]["B1"]
        check("B3 L2 ratio ours/B1 ~ 0.9 with CI covering it", 0.85 < p["l2"]["est"] < 0.95 and p["l2"]["lo"] < 0.9 < p["l2"]["hi"])
        check("B4 LPIPS difference negative (ours better), wins dominate", p["lpips_vgg"]["est"] < 0 and p["lpips_vgg"]["wins"] > 35)
        check("B5 Holm-adjusted p present", "p_holm" in p["l2"])
        tex = open(os.path.join(out, "tab_l2_resnet50_targeted.tex")).read()
        check("B6 best L2 in bold belongs to ours", "LGC-EGC (ours) & \\textbf" in tex)


def test_store_grid_package(tmp: str) -> None:
    print("[C] adversarial store, comparison grid, reviewer package")
    rng = np.random.default_rng(1)
    res = os.path.join(tmp, "Results", "run1")
    store = os.path.join(res, "targeted", "adv_store")
    yy, xx = np.mgrid[0:64, 0:64] / 64
    for i in range(4):
        x0 = np.stack([0.5 + 0.3 * np.sin((3 + i) * xx), 0.5 + 0.3 * np.cos(4 * yy), 0.4 + 0.2 * xx]).astype(np.float32)
        for m, s in (("OURS", 0.005), ("B1", 0.02)):
            xa = np.clip(x0 + s * rng.standard_normal(x0.shape).astype(np.float32), 0, 1)
            store_adv(store, "resnet50", f"img{i:03d}", m, 10000, xa, x0, 9990, 1, "targeted")
    xa, x0, meta = load_adv(os.path.join(store, "resnet50", "img000", "OURS_Q10000.npz"))
    check("C1 store round trip keeps shape and metadata", xa.shape == (3, 64, 64) and meta["budget"] == 10000)
    stem = os.path.join(tmp, "assets", "fig_examples")
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_comparison_grid.py"), "--roots", store, "--arch", "resnet50",
                        "--budget", "10000", "--methods", "OURS", "B1", "--ours", "OURS", "--n", "2", "--out", stem],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    check("C2 comparison grid written with caption", r.returncode == 0 and os.path.exists(stem + ".pdf")
          and "one amplification per row" in open(stem + "_caption.txt").read())
    pkg = os.path.join(tmp, "reviewer_package")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_reviewer_package.py"), "--out", pkg, "--code",
                        os.path.join(HERE, "advstore.py"), "--results", res, "--assets", os.path.join(tmp, "assets"),
                        "--store_images", "2", "--tar"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
    man = os.path.join(pkg, "MANIFEST.csv")
    ok = r.returncode == 0 and os.path.exists(man) and os.path.exists(pkg + ".tar.gz")
    check("C3 package + manifest + tar written", ok)
    if ok:
        rows = list(csv.DictReader(open(man)))
        n_npz = sum(r_["path"].endswith(".npz") for r_ in rows)
        check("C4 store subset respected (2 images x 2 methods = 4 npz)", n_npz == 4)
        import hashlib

        p0 = rows[0]
        h = hashlib.sha256(open(os.path.join(pkg, p0["path"]), "rb").read()).hexdigest()
        check("C5 manifest sha256 verifies", h == p0["sha256"])


def main() -> int:
    test_core()
    with tempfile.TemporaryDirectory() as tmp:
        test_tables(tmp)
        test_store_grid_package(tmp)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
