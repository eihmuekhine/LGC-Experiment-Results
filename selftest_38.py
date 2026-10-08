#!/usr/bin/env python3
"""Build 38 self-test: official TtBA protocol tools.

Part A (no torch):
  A1  CheckpointRecorder: value at budget c = last record with q <= c, on random record sequences.
  A2  The recorder agrees with the repo's own convention (OutResult.Summary: L2_LINE_sum[i, l:r] = record value).
  A3  HookList feeds the recorder with the owner's last boundary point.
  A4  Split CSV written with SPLIT_FIELDS is read by lgc_bench.baseline_protocol.read_split / split_meta.
  A5  ttba_table1.json holds the paper's ImageNet rows (spot values).
  A6  confirm_ttba.py on synthetic results: runs, finds the claims and the integrity lines.
Part B (AutoDL, torch; pass --ttba_repo):
  B1  ttba_github_tensor == the repo's ImageFolder transform on the first val images (max |diff| = 0).
  B2  build_ttba_github_victim gives the same logits as the repo's GeneralTorchModel (same file, same input).
  B3  run_benchmark_33b.py accepts --image_loader ttba_github and --victim_ckpt_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ttba_github"))
from ttba_common import CKPT_FIELDS, RUN_FIELDS, SPLIT_FIELDS, CheckpointRecorder, HookList, write_rows  # noqa: E402

FAILS = []


def check(ok: bool, msg: str) -> None:
    print(("[ok] " if ok else "[FAIL] ") + msg)
    if not ok:
        FAILS.append(msg)


def repo_value(records, q):
    """OutResult.Summary: L2_LINE_sum[l:r] = rec[pair][1] for l = rec[pair][0], r = rec[pair + 1][0]."""
    budget = max(r[0] for r in records) + 10
    line = [None] * (budget + 2)
    for k in range(len(records) - 1):
        l, r = records[k][0], min(records[k + 1][0], budget + 1)
        for i in range(l, r):
            line[i] = records[k][1]
    return line[q]


def part_a() -> None:
    rng = random.Random(0)
    ok1 = ok2 = True
    for _ in range(300):
        cps = sorted(rng.sample(range(50, 12000, 50), 5))
        q, recs = 0, []
        for _ in range(rng.randint(1, 60)):
            q += rng.randint(1, 600)
            recs.append((q, rng.random()))
        rec = CheckpointRecorder(cps, 99.0)
        for qq, v in recs:
            rec.on_record(qq, v, ("payload", qq))
        snap = rec.finish()
        allr = [(0, 99.0)] + recs
        for c in cps:
            exp = [r for r in allr if r[0] <= c][-1]
            ok1 &= snap[c][0] == exp[0] and snap[c][1] == exp[1]
            ok1 &= (snap[c][2] is None) if exp[0] == 0 else (snap[c][2] == ("payload", exp[0]))
            if c < allr[-1][0]:  # inside the recorded range the repo's step function is defined
                ok2 &= repo_value(allr + [(allr[-1][0] + 1, allr[-1][1])], c) == exp[1]
    check(ok1, "A1 recorder: state at c = last record with q <= c (300 random sequences, payload included)")
    check(ok2, "A2 recorder == the repo's L2_LINE_sum step function at every checkpoint")

    class Owner:
        pass

    o = Owner()
    o.recorder = CheckpointRecorder([1000], 5.0)
    o.L2_line = HookList([[0, 5.0]], o)
    o._last_bx = "bx1"
    o.L2_line.append([400, 3.0])
    o._last_bx = "bx2"
    o.L2_line.append([1200, 2.0])
    s = o.recorder.finish()
    check(s[1000] == (400, 3.0, "bx1") and len(o.L2_line) == 3, "A3 HookList passes (q, L2, boundary point)")

    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "resnet50_targeted.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
            w.writeheader()
            w.writerow({"source": "a.JPEG", "source_label": 3, "target": "b.JPEG", "target_label": 7, "order": 0,
                        "protocol": "ttba_github:imagenet-resnet50:targeted1",
                        "victim_weights": "resnet50-11ad3fa6.pth", "loader_pos": 4, "source_relpath": "x",
                        "target_relpath": "y"})
        try:
            from lgc_bench.baseline_protocol import read_split, split_meta
            rs, meta = read_split(p), split_meta(p)
            check(rs == [("a.JPEG", 3, "b.JPEG", 7)] and meta["victim_weights"] == "resnet50-11ad3fa6.pth",
                  "A4 split CSV is read by baseline_protocol.read_split / split_meta")
        except ImportError as err:
            print(f"[!] A4 skipped ({err})")
    finally:
        shutil.rmtree(tmp)

    t1 = json.load(open(os.path.join(HERE, "ttba_github", "ttba_table1.json")))["table1"]
    check(t1["resnet50"]["TtBA"]["non_targeted"]["10000"] == [3.448, 1.7]
          and t1["vgg19"]["CGBA-H"]["targeted"]["10000"] == [14.32, 10.59]
          and t1["vit_b_32"]["TtBA"]["targeted"]["2000"] == [22.2, 19.64]
          and t1["resnet50"]["HSJA"]["targeted"]["2000"] == [72.01, 61.64],
          "A5 Table 1 JSON spot values (ResNet50 TtBA NT 10k, VGG19 CGBA-H T 10k, ViT TtBA T 2k, ResNet50 HSJA T 2k)")

    try:
        import numpy as np
        import pandas  # noqa: F401
    except ImportError as err:
        print(f"[!] A6 skipped ({err})")
        return
    tmp = tempfile.mkdtemp()
    try:
        rng2 = np.random.default_rng(1)
        res = os.path.join(tmp, "Results", "ttbagh_resnet50")
        sp = os.path.join(tmp, "splits")
        os.makedirs(sp)
        for mode in ("non_targeted", "targeted"):
            os.makedirs(os.path.join(res, mode))
            imgs = [f"im{i}.JPEG" for i in range(60)]
            with open(os.path.join(sp, f"resnet50_{mode}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
                w.writeheader()
                for i, n in enumerate(imgs):
                    w.writerow({"source": n, "source_label": 1, "order": i})
            ours, ref = [], []
            for n in imgs:
                u = rng2.normal(0, 0.3)
                for b in (500, 1000, 2000, 2500, 5000, 10000):
                    base = 30 / (b / 500) ** 0.7 * np.exp(u)
                    f_m = 0.85 if mode == "targeted" else 1.0
                    ours.append({"image": n, "method": "LGC_RC_EGC", "budget": b, "found": 1,
                                 "pix_l2": base * f_m * np.exp(rng2.normal(0, 0.05)), "lpips": 0.02})
                    ref.append({"image": n, "method": "TtBA_GH", "budget": b, "found": 1,
                                "pix_l2": base * np.exp(rng2.normal(0, 0.05)), "lpips": 0.08, "lpips_net": "alex",
                                "pix_l2_reported": base * 1.01})
            write_rows(os.path.join(res, mode, "per_checkpoint.csv"), ["image", "method", "budget", "found", "pix_l2",
                                                                       "lpips"], ours)
            write_rows(os.path.join(res, mode, "ttbagh_per_checkpoint.csv"), CKPT_FIELDS, ref)
            write_rows(os.path.join(res, mode, "ttbagh_runs.csv"), RUN_FIELDS,
                       [{"image": n, "method": "TtBA_GH", "status": "ok", "x0_pred_ok": 1} for n in imgs])
        out = subprocess.run([sys.executable, os.path.join(HERE, "confirm_ttba.py"), "--archs", "resnet50",
                              "--res_pattern", os.path.join(tmp, "Results", "ttbagh_{arch}"),
                              "--split_pattern", os.path.join(sp, "{arch}_{mode}.csv"), "--our_lpips_net", "alex",
                              "--out", os.path.join(tmp, "out"), "--B", "500"], capture_output=True, text=True)
        txt = out.stdout + out.stderr
        check(out.returncode == 0 and "CONFIRMED     T1 resnet50" in txt and "P1 resnet50 NT" in txt
              and "paper Table 1: TtBA (cited)" in txt and "[!]" not in txt,
              "A6 confirm_ttba.py on synthetic data: T1 confirmed (true ratio 0.85), P1 tested, Table 1 cited, "
              "no integrity warning" + ("" if out.returncode == 0 else "\n" + txt[-2000:]))
    finally:
        shutil.rmtree(tmp)


def part_b(ttba_repo: str) -> None:
    try:
        import torch
    except ImportError as err:
        print(f"[!] Part B skipped ({err})")
        return
    from lgc_bench.baseline_protocol import build_ttba_github_victim, ttba_github_tensor
    repo = os.path.abspath(ttba_repo)
    cwd = os.getcwd()
    sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        import dataset as ttds
        loader, tm = ttds.load_dataset_model("imagenet-resnet50")
        ds = loader.dataset
        diffs = []
        for i in range(3):
            x_repo, _ = ds[i]
            x_ours = ttba_github_tensor(ds.samples[i][0])
            diffs.append(float((x_repo - x_ours).abs().max()))
        check(max(diffs) == 0.0, f"B1 ttba_github_tensor == repo ImageFolder transform (max |diff| {max(diffs):.2e})")
        v = build_ttba_github_victim("resnet50", os.path.join(repo, "model"), "cuda")
        with torch.no_grad():
            x = torch.stack([ds[i][0] for i in range(3)]).cuda()
            a, b = tm.predict_prob(x), v(x)
        check(float((a - b).abs().max()) < 1e-3 and bool((a.argmax(1) == b.argmax(1)).all()),
              f"B2 victim logits equal the repo's GeneralTorchModel (max |diff| {float((a - b).abs().max()):.2e})")
    finally:
        os.chdir(cwd)
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_benchmark_33b.py"), "--help"],
                         capture_output=True, text=True)
    check("ttba_github" in out.stdout and "--victim_ckpt_dir" in out.stdout,
          "B3 run_benchmark_33b.py exposes --image_loader ttba_github and --victim_ckpt_dir")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttba_repo", default="")
    a = ap.parse_args()
    part_a()
    if a.ttba_repo:
        part_b(a.ttba_repo)
    else:
        print("[!] Part B skipped (pass --ttba_repo /root/autodl-tmp/TtBA on AutoDL)")
    print("\nBuild 38 self-test: " + ("PASS" if not FAILS else f"FAIL ({len(FAILS)})"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
