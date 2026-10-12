#!/usr/bin/env python3
"""Acceptance tests T1-T14 for tools/eval/check_contract.py (skill lgc-paper-contract, Section 2).

Each test builds a synthetic result tree (3 images x 2 methods x 8 checkpoints, adv_store npz,
env file, split CSVs), injects one defect, runs the checker and asserts the EXACT stated verdict.

    python tools/eval/tests_contract/test_contract.py      -> prints PASS/FAIL per test, exit 0 if all pass

Pure stdlib + numpy + pandas + PyYAML; no torch, no GPU; runs in < 1 min.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVAL = os.path.join(REPO, "tools", "eval")
sys.path.insert(0, EVAL)

import yaml  # noqa: E402

import check_contract as cc  # noqa: E402

CONTRACT = cc.load_yaml(os.path.join(REPO, "contract", "paper_contract.yaml"))
COLMAP = cc.load_yaml(os.path.join(REPO, "contract", "column_map.yaml"))
CONST = CONTRACT["constants"]
BUDGETS = [int(x) for x in CONST["checkpoints"]]
ARCH = "resnet50"
MODE = "non_targeted"
IMAGES = ["img001.JPEG", "img002.JPEG", "img003.JPEG"]
METHODS = {"LGC_RC_EGC": "engine_state", "TtBA": "tracker_best"}

RESULTS: list = []


def l2(b: int) -> float:
    return round(500.0 / (int(b) ** 0.5), 6)


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


# -------------------------------------------------------------------------------------------------
def write_split(split_dir: str) -> None:
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, f"{ARCH}_{MODE}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "source_label", "target", "target_label",
                                           "order", "protocol", "victim_weights", "loader_pos",
                                           "source_relpath", "target_relpath"])
        w.writeheader()
        for i, img in enumerate(IMAGES):
            w.writerow({"source": img, "source_label": 579, "target": "", "target_label": "",
                        "order": i, "protocol": f"ttba_github:imagenet-resnet50:targeted0",
                        "victim_weights": "resnet50-11ad3fa6.pth", "loader_pos": i,
                        "source_relpath": f"val/{img}", "target_relpath": ""})


def write_x0(store: str, stem: str) -> None:
    d = os.path.join(store, ARCH, stem)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x0.npy"), (0.5 * np.ones((3, 224, 224), dtype=np.float32)))


def write_npz(store: str, stem: str, method: str, budget: int, l2v: float, sq: int) -> None:
    d = os.path.join(store, ARCH, stem)
    os.makedirs(d, exist_ok=True)
    delta = np.array([l2v], dtype=np.float64)
    np.savez_compressed(
        os.path.join(d, f"{method}_Q{budget}.npz"),
        delta=delta, budget=np.int64(budget), state_query=np.int64(sq), adv_flag=np.int64(1),
        l2=np.float64(l2v), linf=np.float64(l2v / 10), mode=np.array(MODE), method=np.array(method),
        image=np.array(stem), arch=np.array(ARCH))


def ck_row(img: str, method: str, budget: int, sq: int, l2v: float, found: int = 1) -> dict:
    return {"arch": ARCH, "mode": MODE, "image": img, "method": method, "budget": budget,
            "found": found, "state_query": sq, "pix_l2": l2v, "pix_linf": l2v / 10,
            "adv_float": found, "adv_uint8": found, "ssim": 0.9, "lpips": 0.1,
            "source_label": 579, "target_label": ""}


def base_ck_rows() -> list:
    rows = []
    for img in IMAGES:
        for method in METHODS:
            for b in BUDGETS:
                rows.append(ck_row(img, method, b, b - 5, l2(b)))
    return rows


def base_runs_rows() -> list:
    rows = []
    for img in IMAGES:
        for method in METHODS:
            rows.append({"arch": ARCH, "mode": MODE, "image": img, "method": method,
                         "status": "ok", "total_queries": 9990, "wall_time_s": 1.0})
    return rows


def write_csv(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def env_text(split_sha: str, engine_sha: str, adv_store_line: str) -> str:
    return "\n".join([
        "NVIDIA GeForce RTX 5090",
        "torch 2.7.1+cu128",
        "git rev-parse HEAD -> 9e2822e8bb5c1aee6b04d39fea7f813ed0086454",
        f"{engine_sha}  lgc_bench/engine.py",
        f"{split_sha}  splits/imagenet/ttba_github/resnet50_non_targeted.csv",
        "checkpoints 100 200 500 1000 2000 2500 5000 10000",
        adv_store_line,
    ]) + "\n"


def build_tree(tmp: str, *, ck_rows=None, runs_rows=None, npz_skip=None, npz_l2=None,
               x0_images=None, mismatch=False, env=None, extra_image=None) -> dict:
    """Build a synthetic result tree; return the paths used by the checker."""
    root = os.path.join(tmp, "results", "run1")
    split_dir = os.path.join(tmp, "split")
    mode_dir = os.path.join(root, MODE)
    store = os.path.join(mode_dir, "adv_store")

    write_split(split_dir)
    ck = ck_rows if ck_rows is not None else base_ck_rows()
    runs = runs_rows if runs_rows is not None else base_runs_rows()
    if extra_image:
        for method in METHODS:
            for b in BUDGETS:
                ck.append(ck_row(extra_image, method, b, b - 5, l2(b), found=0))
    write_csv(os.path.join(mode_dir, "per_checkpoint.csv"), ck)
    write_csv(os.path.join(mode_dir, "runs.csv"), runs)

    npz_skip = set(npz_skip or [])
    npz_l2 = dict(npz_l2 or {})
    x0_images = set(x0_images or [])
    for r in ck:
        if r["found"] != 1:
            continue
        stem = os.path.splitext(r["image"])[0]
        if r["image"] not in x0_images:
            write_x0(store, stem)
        key = (stem, r["method"], int(r["budget"]))
        if key in npz_skip:
            continue
        write_npz(store, stem, r["method"], int(r["budget"]), npz_l2.get(key, r["pix_l2"]),
                  int(r["state_query"]))
    if mismatch:
        os.makedirs(os.path.join(store), exist_ok=True)
        with open(os.path.join(store, "x0_mismatch.csv"), "w") as fh:
            fh.write("arch,image,method,max_abs_diff\nresnet50,img001,JPG,0.5\n")
    if env is not None:
        with open(os.path.join(root, "env_frame.txt"), "w") as fh:
            fh.write(env)
    return {"root": root, "split_dir": split_dir}


def run(stage: str, root: str, split_dir: str, n: int = 3, metrics_csv=None, paper_dir=None,
        generated_dir=None, cut=None) -> cc.Report:
    return cc.check(CONTRACT, COLMAP, [root], stage, n, metrics_csv, paper_dir, generated_dir,
                    split_dir, cut=cut)


def v(report: cc.Report, cid: str) -> str:
    return report.verdict_of(cid)


def offending(report: cc.Report, cid: str) -> list:
    for c in report.checks:
        if c.id == cid:
            return c.offending
    return []


def good_env() -> str:
    return env_text(CONST["split_sha256"]["resnet50_non_targeted"],
                    CONST["engine_sha256_prefix"] + "0" * 52,
                    "ADV_STORE_DTYPE float16 ADV_STORE_BUDGETS unset")


# E10/E11 are REQUIRED since contract v1.5 (D9): only E9b and E12 remain CONDITIONAL/cuttable.
COND_CUT = {"E9b_operator_substitution", "E12_low_query_breadth"}


def fresh(tmp: str) -> str:
    return tempfile.mkdtemp(dir=tmp)


# =================================================================================================
def test_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # ---- T1: complete 3-image x 2-method cell -> all PASS
        tree = build_tree(fresh(tmp), env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T1 complete cell -> no FAIL (all PASS)",
              rep.exit_code() == 0 and all(v(rep, f"I{i:02d}") == "PASS" for i in range(1, 13)))

        # ---- T2: one image lacks budget 500 -> I01 FAIL naming that image
        rows = base_ck_rows()
        rows = [r for r in rows if not (r["image"] == "img001.JPEG" and r["budget"] == 500)]
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T2 missing budget 500 -> I01 FAIL", v(rep, "I01") == "FAIL" and
              any("img001" in o and "500" in o for o in rep.checks[0].offending))

        # ---- T3: one npz missing for a found row -> I05 FAIL
        tree = build_tree(fresh(tmp), npz_skip={("img002", "TtBA", 1000)}, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T3 missing npz -> I05 FAIL", v(rep, "I05") == "FAIL")

        # ---- T4: npz l2 differs from pix_l2 by 1% -> I06 FAIL
        tree = build_tree(fresh(tmp), npz_l2={("img003", "LGC_RC_EGC", 2000): l2(2000) * 1.01}, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T4 npz l2 != pix_l2 by 1% -> I06 FAIL", v(rep, "I06") == "FAIL")

        # ---- T5: state_query 10050 at budget 10000 -> I04 FAIL
        rows = base_ck_rows()
        for r in rows:
            if r["image"] == "img001.JPEG" and r["method"] == "TtBA" and r["budget"] == 10000:
                r["state_query"] = 10050
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T5 state_query > budget -> I04 FAIL", v(rep, "I04") == "FAIL")

        # ---- T6: an image not in rows 1-3 of the split -> I02 FAIL
        tree = build_tree(fresh(tmp), extra_image="img999.JPEG", env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T6 extra image -> I02 FAIL", v(rep, "I02") == "FAIL")

        # ---- T7: tracker-best method with L2 increasing -> I08 FAIL
        rows = base_ck_rows()
        for r in rows:
            if r["image"] == "img001.JPEG" and r["method"] == "TtBA":
                r["pix_l2"] = 3.0 if r["budget"] == 100 else (3.2 if r["budget"] == 200 else l2(r["budget"]))
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T7 tracker_best L2 increase -> I08 FAIL", v(rep, "I08") == "FAIL")

        # ---- T8: engine-state method with L2 increasing -> I08 WARN only
        rows = base_ck_rows()
        for r in rows:
            if r["image"] == "img002.JPEG" and r["method"] == "LGC_RC_EGC":
                r["pix_l2"] = 3.0 if r["budget"] == 100 else (3.2 if r["budget"] == 200 else l2(r["budget"]))
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T8 engine_state L2 increase -> I08 WARN (not FAIL)", v(rep, "I08") == "WARN")

        # ---- T9: x0_mismatch.csv present -> I07 FAIL
        tree = build_tree(fresh(tmp), mismatch=True, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T9 x0_mismatch.csv -> I07 FAIL", v(rep, "I07") == "FAIL")

        # ---- T10: env split sha256 differs -> I10 FAIL
        bad_env = env_text("0" * 64, CONST["engine_sha256_prefix"] + "0" * 52,
                           "ADV_STORE_DTYPE float16 ADV_STORE_BUDGETS unset")
        tree = build_tree(fresh(tmp), env=bad_env)
        rep = run("final", tree["root"], tree["split_dir"])
        check("T10 env split sha differs -> I10 FAIL", v(rep, "I10") == "FAIL")

        # ---- T11: ADV_STORE_BUDGETS=1000 10000 recorded -> I11 FAIL
        bad_env = env_text(CONST["split_sha256"]["resnet50_non_targeted"],
                           CONST["engine_sha256_prefix"] + "0" * 52,
                           "ADV_STORE_DTYPE float16 ADV_STORE_BUDGETS=1000 10000")
        tree = build_tree(fresh(tmp), env=bad_env)
        rep = run("final", tree["root"], tree["split_dir"])
        check("T11 ADV_STORE_BUDGETS set -> I11 FAIL", v(rep, "I11") == "FAIL")

        # ---- T12: duplicate (image, method, budget) row -> I01 FAIL
        rows = base_ck_rows()
        rows.append(dict(rows[0]))  # duplicate the first row
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T12 duplicate row -> I01 FAIL", v(rep, "I01") == "FAIL")

        # ---- T15: unregistered split (arch_mode not in constants.split_sha256) -> I10 FAIL
        rows = base_ck_rows()
        for r in rows:
            r["arch"] = "unknown_arch"
        tree = build_tree(fresh(tmp), ck_rows=rows, env=good_env())
        rep = run("final", tree["root"], tree["split_dir"])
        check("T15 unregistered split -> I10 FAIL", v(rep, "I10") == "FAIL" and
              any("unregistered split" in o for o in offending(rep, "I10")))

        # ---- T16: env holds the OTHER mode's split sha -> I10 FAIL
        bad_env = env_text(CONST["split_sha256"]["resnet50_targeted"],
                           CONST["engine_sha256_prefix"] + "0" * 52,
                           "ADV_STORE_DTYPE float16 ADV_STORE_BUDGETS unset")
        tree = build_tree(fresh(tmp), env=bad_env)
        rep = run("final", tree["root"], tree["split_dir"])
        check("T16 env holds other mode's split -> I10 FAIL", v(rep, "I10") == "FAIL")

        # ---- T17: our env (env_frame.txt) without an ADV_STORE_BUDGETS token -> I11 FAIL
        no_adv_env = "\n".join([
            "NVIDIA GeForce RTX 5090",
            "torch 2.7.1+cu128",
            CONST["engine_sha256_prefix"] + "0" * 52 + "  lgc_bench/engine.py",
            CONST["split_sha256"]["resnet50_non_targeted"] + "  splits/imagenet/ttba_github/resnet50_non_targeted.csv",
            "checkpoints 100 200 500 1000 2000 2500 5000 10000",
        ]) + "\n"
        tree = build_tree(fresh(tmp), env=no_adv_env)
        rep = run("final", tree["root"], tree["split_dir"])
        check("T17 our env without ADV_STORE_BUDGETS -> I11 FAIL", v(rep, "I11") == "FAIL")

    with tempfile.TemporaryDirectory() as tmp:
        # ---- T13: numbers.tex has \NumX = '??' used in a .tex -> I15 FAIL
        tree = build_tree(fresh(tmp), env=good_env())
        paper = os.path.join(tmp, "paper")
        gen = os.path.join(tmp, "generated")
        os.makedirs(paper), os.makedirs(gen)
        with open(os.path.join(paper, "main.tex"), "w") as fh:
            fh.write(r"\documentclass{article}\begin{document}\NumMainL2 \end{document}")
        with open(os.path.join(gen, "numbers.tex"), "w") as fh:
            fh.write(r"\newcommand{\NumMainL2}{??}")
        m_csv = os.path.join(tmp, "metrics.csv")
        write_csv(m_csv, [{"root": "r", "arch": ARCH, "mode": MODE, "image": IMAGES[0],
                           "method": "LGC_RC_EGC", "budget": 10000, "state_query": 9990,
                           "adv_flag": 1, "l2_meta": 5.0, "l2_relerr": 0.0, "l2": 5.0,
                           "linf": 0.5, "psnr": 30.0, "ssim": 0.9, "ms_ssim": 0.9,
                           "lpips_alex": 0.1, "lpips_vgg": 0.1, "dists": 0.05, "fsim": 0.9}])
        rep = run("manuscript", tree["root"], tree["split_dir"], metrics_csv=m_csv,
                  paper_dir=paper, generated_dir=gen, cut=COND_CUT)
        check("T13 \\NumX = '??' -> I15 FAIL", v(rep, "I15") == "FAIL")

    with tempfile.TemporaryDirectory() as tmp:
        # ---- T14: REQUIRED E10/E11 absent -> I14 FAIL; cutting the CONDITIONAL elements does not rescue them
        paper = os.path.join(tmp, "paper")
        gen = os.path.join(tmp, "generated")
        os.makedirs(paper), os.makedirs(gen)
        with open(os.path.join(paper, "main.tex"), "w") as fh:
            fh.write(r"\documentclass{article}\begin{document}x\end{document}")
        m_csv = os.path.join(tmp, "metrics.csv")
        write_csv(m_csv, [{"arch": ARCH, "mode": MODE, "image": IMAGES[0], "method": "LGC_RC_EGC",
                           "budget": 10000}])
        tree = build_tree(fresh(tmp), env=good_env())
        rep = run("manuscript", tree["root"], tree["split_dir"], metrics_csv=m_csv,
                  paper_dir=paper, generated_dir=gen, cut=set())
        check("T14a E10/E11 REQUIRED absent -> I14 FAIL", v(rep, "I14") == "FAIL")
        rep = run("manuscript", tree["root"], tree["split_dir"], metrics_csv=m_csv,
                  paper_dir=paper, generated_dir=gen, cut=COND_CUT)
        check("T14b REQUIRED E10/E11 cannot be cut -> I14 FAIL", v(rep, "I14") == "FAIL")


def main() -> int:
    test_contract()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
