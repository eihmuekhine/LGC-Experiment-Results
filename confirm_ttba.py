#!/usr/bin/env python3
"""Pre-registered comparison under the OFFICIAL TtBA protocol (build 38): LGC_RC_EGC vs the official TtBA code, run by
us on the same images in the same environment.

REGISTERED 2026-10-07 13:30 (UTC+8), BEFORE any TtBA-protocol run. The method is frozen as LGC_RC_EGC (build 37 r5).

Protocol
  github.com/BUPTAIOC/TtBA main.py selection: ImageNet-mini val, Resize(256) + CenterCrop(224), DataLoader shuffle
  with seed 42, first `imgnum` correctly classified images (1000), victim weights from the README files.
  Split dumped by ttba_github/ttba_dump_split.py.
  TtBA_GH = the official TtBA.Attacker, run by ttba_github/ttba_run_official.py.
  Ours    = run_benchmark_33b.py --split_csv ... --image_loader ttba_github --victim_ckpt_dir ...
            --methods LGC_RC_EGC.
  Victims: resnet50 (V2 file), vgg19, vit_b_32. Both modes. Budgets 500 / 1k / 2k / 2.5k / 5k / 10k.
Metrics
  pix_l2       : L2 of the clamped adversarial image (both methods). For TtBA_GH the repo's own reported value
                 (unclamped bisection distance, >= the clamped one) is also shown in the Table-1 view.
  L2 ratio     : paired geometric mean.
  AUC          : per-image mean of log L2 over the budgets 500/1k/2.5k/5k/10k (= confirm_37).
  CI           : image bootstrap, 95% percentile (B = 10000, seed 0).
  Test         : one-sided Wilcoxon signed-rank.
  LPIPS AUC    : per-image mean LPIPS over the budgets; effect = median paired difference.
Claims (method fixed; a failed claim is reported and dropped, never fixed by changing the method)
  Tested family, Holm over every test present (victims x modes):
    T1  targeted  : M / TtBA_GH L2 AUC < 1, AND upper 95% bound < 1.
    P1  both modes: LPIPS AUC (M - TtBA_GH) < 0, AND upper bound < 0. Evaluated only if --our_lpips_net equals
                    the net that ttba_run_official.py used; otherwise "not evaluable".
  CI-only (non-inferiority):
    T2  targeted     : M / TtBA_GH L2 at 10k, upper bound <= 1.03.
    N1  non-targeted : M / TtBA_GH L2 AUC and L2 at 10k, upper bound <= 1.15.
Descriptive
  - Table-1 view, mean (median) at 2k / 5k / 10k, next to the paper's Table 1. The paper's numbers come from another
    list and environment, so they are cited, not paired.
  - ASR at eps = 2.5 (paper's Table 5 threshold) per budget.
  - Paired ratios per budget.
  - Integrity: image lists equal to the split, success shares, x0 predictions.

Usage
  python confirm_ttba.py --archs resnet50 vgg19 vit_b_32 --res_pattern Results/ttbagh_{arch} \
      --split_pattern splits/imagenet/ttba_github/{arch}_{mode}.csv --our_lpips_net alex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confirm_34 import boot_ci, per_image, wilcoxon  # noqa: E402
from analyze_34b import holm, l2_summary  # noqa: E402

M, R = "LGC_RC_EGC", "TtBA_GH"
MODES = ("non_targeted", "targeted")
TABLE_B = (2000, 5000, 10000)
AUC_B = (500, 1000, 2500, 5000, 10000)  # AUC budgets = confirm_37 (pre-registered 2026-10-08, before any build-38 data)
EPS = 2.5
NI_T, NI_NT = 1.03, 1.15
HERE = os.path.dirname(os.path.abspath(__file__))


def _read(paths):
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False)
    return df.drop_duplicates(["image", "method", "budget"], keep="last")


def load_pair(res: str, mode: str):
    ours = _read(sorted(glob.glob(os.path.join(res, mode, "per_checkpoint*.csv"))))
    ref = _read(sorted(glob.glob(os.path.join(res, mode, "ttbagh_per_checkpoint*.csv"))))
    parts = []
    if ours is not None:
        parts.append(ours[ours.method == M])
    if ref is not None:
        parts.append(ref[ref.method == R])
    if not parts:
        return None, None
    rr = sorted(glob.glob(os.path.join(res, mode, "ttbagh_runs*.csv")))
    runs = pd.concat([pd.read_csv(p) for p in rr], ignore_index=True) if rr else None
    return pd.concat(parts, ignore_index=True, sort=False), runs


def table_view(pc, method, col="pix_l2"):
    cells = []
    for b in TABLE_B:
        v = pc[(pc.method == method) & (pc.budget == b)][col].astype(float)
        cells.append(f"{v.mean():.3f}({v.median():.3f})" if len(v) else "n/a")
    return " ".join(f"{c:>15s}" for c in cells)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=["resnet50", "vgg19", "vit_b_32"])
    ap.add_argument("--res_pattern", default="Results/ttbagh_{arch}")
    ap.add_argument("--split_pattern", default="splits/imagenet/ttba_github/{arch}_{mode}.csv")
    ap.add_argument("--our_lpips_net", default=None, help="net of lgc_bench/metrics.py build_lpips (alex | vgg)")
    ap.add_argument("--out", default="Results/confirm_ttba")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    t1 = json.load(open(os.path.join(HERE, "ttba_github", "ttba_table1.json")))["table1"]
    txt = [f"TtBA-protocol comparison (pre-registered 2026-10-07): {M} vs {R} (official code, run by us)"]
    rows, tests, ci_claims = [], [], []
    for arch in a.archs:
        res = a.res_pattern.format(arch=arch)
        env = os.path.join(res, "env.txt")
        if os.path.exists(env):
            txt.append(f"\n##### {arch}  ({' | '.join(l.strip() for l in open(env) if l.strip())})")
        else:
            txt.append(f"\n##### {arch}")
        for mode in MODES:
            pc, runs = load_pair(res, mode)
            if pc is None:
                txt.append(f"[{mode}] no results under {res}/{mode}")
                continue
            sp = a.split_pattern.format(arch=arch, mode=mode)
            split = pd.read_csv(sp) if os.path.exists(sp) else None
            have = {m: set(pc[pc.method == m].image) for m in (M, R)}
            info = {m: len(v) for m, v in have.items()}
            txt.append(f"[{mode}] images: {info}" + (f"; split rows {len(split)}" if split is not None else ""))
            if split is not None:
                ss = set(split.source)
                for m in (M, R):
                    extra, miss = have[m] - ss, ss - have[m]
                    if extra or miss:
                        txt.append(f"  [!] {m}: {len(miss)} split images missing, {len(extra)} not in the split")
            if runs is not None and "x0_pred_ok" in runs:
                bad = int((runs.x0_pred_ok == 0).sum())
                if bad:
                    txt.append(f"  [!] {R}: x0 misclassified on {bad} images (environment differs from the split)")
            lp_nets = set(pc[pc.method == R].get("lpips_net", pd.Series(dtype=str)).dropna().astype(str))
            # ---- Table-1 view
            txt.append(f"  Table-1 view, mean(median) L2 at {', '.join(map(str, TABLE_B))} queries")
            txt.append(f"    {'ours ' + M:34s}" + table_view(pc, M))
            txt.append(f"    {R + ' (clamped image)':34s}" + table_view(pc, R))
            if "pix_l2_reported" in pc:
                txt.append(f"    {R + ' (repo-reported L2)':34s}" + table_view(pc, R, "pix_l2_reported"))
            paper = t1.get(arch, {})
            for att in ("HSJA", "TA", "CGBA", "CGBA-H", "TtBA"):
                if att in paper:
                    cells = [f"{paper[att][mode][str(b)][0]:.3f}({paper[att][mode][str(b)][1]:.3f})" for b in TABLE_B]
                    txt.append(f"    {'paper Table 1: ' + att + ' (cited)':34s}" + " ".join(f"{c:>15s}" for c in cells))
            # ---- ASR at eps
            asr = []
            for m in (M, R):
                cells = []
                for b in sorted(pc.budget.unique()):
                    s = pc[(pc.method == m) & (pc.budget == b)]
                    ok = (s.get("found", 1).fillna(0).astype(float) > 0) & (s.pix_l2.astype(float) <= EPS)
                    cells.append(f"{int(b)}:{ok.mean():.3f}")
                asr.append(f"    {m:14s} " + " ".join(cells))
            txt.append(f"  ASR at L2 <= {EPS}:")
            txt += asr
            # ---- paired contrasts
            budgets = sorted(set(pc[pc.method == M].budget) & set(pc[pc.method == R].budget))
            pcc = pc[pc.budget.isin(budgets)]; pca = pcc[pcc.budget.isin(AUC_B)]
            cells = []
            for b in budgets + ["AUC"]:
                d = per_image(pca if b == "AUC" else pcc, M, R, "pix_l2", b)
                if len(d):
                    s = l2_summary(d, a.B)
                    rows.append({"arch": arch, "mode": mode, "contrast": f"{M}/{R}", "budget": b, "metric": "pix_l2",
                                 **s})
                    cells.append(f"{b}: {s['ratio']:.3f}{'*' if s['p'] < 0.05 else ''} "
                                 f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] ({s['wins']}/{s['losses']})")
            txt.append(f"  paired L2 {M}/{R}: " + " | ".join(cells))
            dl = per_image(pca, M, R, "lpips", "AUC")
            lp_ok = bool(a.our_lpips_net) and lp_nets == {a.our_lpips_net}
            if len(dl) and np.isfinite(dl).all():
                lo, hi = boot_ci(dl, np.median, a.B)
                txt.append(f"  LPIPS AUC delta (M - R): {np.median(dl):+.4f} [{lo:+.4f},{hi:+.4f}] "
                           f"({int((dl < 0).sum())}/{int((dl > 0).sum())}); R net {sorted(lp_nets)}, ours "
                           f"{a.our_lpips_net or 'not given'}")
            # ---- claims
            tg = "T" if mode == "targeted" else "NT"
            d_auc, d_10k = per_image(pca, M, R, "pix_l2", "AUC"), per_image(pcc, M, R, "pix_l2", 10000)
            s_auc = l2_summary(d_auc, a.B) if len(d_auc) else None
            s_10k = l2_summary(d_10k, a.B) if len(d_10k) else None
            if mode == "targeted":
                tests.append((f"T1 {arch} T: L2 AUC M/R < 1, upper < 1", s_auc["p_less"] if s_auc else np.nan,
                              bool(s_auc and s_auc["ci_hi"] < 1), s_auc))
                ci_claims.append((f"T2 {arch} T: L2@10k M/R upper <= {NI_T}", bool(s_10k and s_10k["ci_hi"] <= NI_T),
                                  s_10k))
            else:
                ci_claims.append((f"N1 {arch} NT: L2 AUC M/R upper <= {NI_NT}",
                                  bool(s_auc and s_auc["ci_hi"] <= NI_NT), s_auc))
                ci_claims.append((f"N1 {arch} NT: L2@10k M/R upper <= {NI_NT}",
                                  bool(s_10k and s_10k["ci_hi"] <= NI_NT), s_10k))
            if len(dl) and np.isfinite(dl).all() and lp_ok:
                lo, hi = boot_ci(dl, np.median, a.B)
                tests.append((f"P1 {arch} {tg}: LPIPS AUC (M - R) < 0, upper < 0", wilcoxon(dl, "less"), hi < 0,
                              {"delta": float(np.median(dl)), "ci_lo": lo, "ci_hi": hi, "n": len(dl)}))
            else:
                txt.append(f"  P1 {arch} {tg}: not evaluable (LPIPS missing, or --our_lpips_net {a.our_lpips_net} "
                           f"!= R net {sorted(lp_nets)})")
    adj = holm([t[1] for t in tests]) if tests else []
    txt.append("\nCLAIMS (Holm over the tested family; tested claims need p_holm < 0.05 AND the CI condition)")
    for (name, p, ci_ok, s), ph in zip(tests, adj):
        ok = bool(np.isfinite(ph) and ph < 0.05 and ci_ok)
        eff = (f"{s['ratio']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}]" if s and "ratio" in s else
               f"{s['delta']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]" if s else "n/a")
        txt.append(f"  {'CONFIRMED' if ok else 'not confirmed':13s} {name}: {eff} (n={s['n'] if s else 0}) "
                   f"p_holm={ph:.3g}")
        rows.append({"contrast": name, "p": p, "p_holm": ph, "confirmed": ok})
    for name, ok, s in ci_claims:
        eff = f"{s['ratio']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}] (n={s['n']})" if s else "n/a"
        txt.append(f"  {'CONFIRMED' if ok else 'not confirmed':13s} {name}: {eff}")
        rows.append({"contrast": name, "confirmed": ok})
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "confirm_ttba.csv"), index=False)
    with open(os.path.join(a.out, "confirm_ttba.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
