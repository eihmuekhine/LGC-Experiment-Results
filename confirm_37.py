#!/usr/bin/env python3
"""Pre-registered confirmation of the frozen method LGC_RC_EGC (LGC-EGC), build 37 r5.

REGISTERED 2026-10-07 10:45 (UTC+8), BEFORE any seed-2718 run.

The method was frozen by analyze_37 rule v2 + amendment r4 on the single-environment selection set Results/sel37_5090
(seed 3141, n = 20; RTX 5090, torch 2.7.1+cu128).  All four candidates (MALP, MACP, EGC, EGCI) passed rule v2, with
NT AUC 0.771-0.775.  Amendment r4 therefore selects LGC_RC_EGC.

Data
  ResNet-50, ImageNet, both modes: --seed 2718 --exclude_seeds 992 2027 3141 --num_samples 100.
  One environment, recorded in <res>/env.txt.
Rows
  LGC_RC_EGC (= M, the method), LGC_RC_EGCI, LGC_RC, LGC_RC_HT1000, TtBA, and the mode's CGBA:
  CGBA (non-targeted) / CGBA_H (targeted).
Statistics (as analyze_37)
  L2 ratio  : paired geometric-mean L2 ratio.
  L2 AUC    : per-image mean of log L2 over 500 / 1k / 2.5k / 5k / 10k.
  CI        : image bootstrap, 95% percentile CI (B = 10000, seed 0).
  Test      : one-sided Wilcoxon signed-rank.
  LPIPS AUC : per-image mean LPIPS over the same budgets; effect = median paired difference.
  A contrast uses only the images that both of its rows finished.

CONFIRMATORY CLAIMS
  The method is fixed.  A claim that fails is dropped or re-scoped in the paper; the method is NOT changed.

  Tested family: Holm over the 8 one-sided tests below, alpha 0.05.  A tested claim needs its Holm-adjusted p < 0.05
  AND the stated CI condition.
    C1   NT floor removal: M / LGC_RC L2 at 10k < 1, AND upper 95% bound <= 0.67 (claim: >= 1.5x lower L2 at 10k).
    C4a  T vs CGBA_H     : M / CGBA_H L2 AUC < 1, AND upper bound < 1.
    C4b  T vs TtBA       : M / TtBA L2 AUC < 1, AND upper bound < 1.
    C5   Perceptual      : LPIPS AUC (M - X) < 0, AND upper bound < 0.
                           X = CGBA and TtBA (NT); X = CGBA_H and TtBA (T).  This gives 4 tests.
    C6   NT LPIPS at low budget: LPIPS at 1000 queries (M - CGBA) < 0, AND upper bound < 0.

  CI-only claims (non-inferiority, no test):
    C2   T vs LGC_RC      : L2 AUC upper 95% bound <= 1.03.
    C3   NT vs CGBA and TtBA: L2 AUC and L2 at 10k, upper 95% bound <= 1.15.
         The 1.15 margin is fixed a priori.  The dev per-image log-ratio sd is about 0.34, so n = 100 gives a CI
         half-width of about 7%.

DESCRIPTIVE (no decision)
  - Per-budget medians.
  - M vs LGC_RC_EGCI: the pull-back ellipse; dev T 10k 0.862*, LPIPS +0.019.
  - LGC_RC_HT1000 vs LGC_RC and M vs LGC_RC_HT1000: the naive LGC -> TtBA switch.
  - LGC_RC vs the pixel attacks: the decoder floor.
  - SSIM >= 0.99 share, gate statistics, wall time per image.

Usage
  python confirm_37.py --res Results/confirm37_resnet50 --arch ResNet-50
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confirm_34 import boot_ci, load, per_image, wilcoxon  # noqa: E402
from analyze_34b import holm, l2_summary  # noqa: E402
from analyze_36 import contrast  # noqa: E402

M, ISO, REF, NAIVE = "LGC_RC_EGC", "LGC_RC_EGCI", "LGC_RC", "LGC_RC_HT1000"
PIX = {"non_targeted": ["CGBA", "TtBA"], "targeted": ["CGBA_H", "TtBA"]}
ROWS = {mode: [M, ISO, REF, NAIVE] + px for mode, px in PIX.items()}
N_MIN = 100
NI_T, NI_NT, FLOOR_UB = 1.03, 1.15, 0.67
DIAG = ["ma_gate_q", "ma_amb_pct_open", "ma_k_A_final", "ma_k_L_final", "wall_time_s"]
TAG = {"non_targeted": "NT", "targeted": "T"}


def l2(pc, a, b, budget, B):
    d = per_image(pc, a, b, "pix_l2", budget)
    return l2_summary(d, B) if len(d) else None


def lp(pc, a, b, budget, B):
    d = per_image(pc, a, b, "lpips", budget)
    if not len(d):
        return None
    lo, hi = boot_ci(d, np.median, B)
    return {"n": len(d), "delta": float(np.median(d)), "ci_lo": lo, "ci_hi": hi, "wins": int((d < 0).sum()),
            "losses": int((d > 0).sum()), "p_less": wilcoxon(d, "less")}


def fmt_l2(s):
    return (f"{s['ratio']:.3f} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}] ({s['wins']}/{s['losses']}; n={s['n']})"
            if s else "n/a")


def fmt_lp(s):
    return (f"{s['delta']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}] ({s['wins']}/{s['losses']}; n={s['n']})"
            if s else "n/a")


def runs_table(res, mode):
    paths = glob.glob(os.path.join(res, mode, "runs*.csv"))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True, sort=False)
    df = df.drop_duplicates(["image", "method"], keep="last")
    cols = [c for c in DIAG if c in df.columns]
    out = df.groupby("method")[cols].median(numeric_only=True)
    if "ma_gate_q" in df.columns:
        out["gate_open_pct"] = df.groupby("method")["ma_gate_q"].apply(lambda s: 100.0 * s.notna().mean())
    return out.reindex([r for r in ROWS[mode] if r in out.index])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--arch", default="?")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    txt = [f"Build 37 r5 CONFIRMATION of {M} (pre-registered 2026-10-07), {a.arch}, res = {a.res}"]
    env = os.path.join(a.res, "env.txt")
    if os.path.exists(env):
        txt.append("environment: " + " | ".join(l.strip() for l in open(env) if l.strip()))
    data = {m: load(a.res, m) for m in ("non_targeted", "targeted")}
    data = {m: v for m, v in data.items() if v is not None}
    if not data:
        raise SystemExit(f"no per_checkpoint*.csv under {a.res}/<mode>/")
    rows, tests, ci_claims = [], [], []
    for mode, pc in data.items():
        tg = TAG[mode]
        budgets = sorted(pc.budget.unique())
        present = [r for r in ROWS[mode] if r in set(pc.method)]
        missing = [r for r in ROWS[mode] if r not in present]
        counts = {r: pc[pc.method == r].image.nunique() for r in present}
        txt.append(f"\n[{mode}] images per row: {counts}" + (f"  MISSING rows: {missing}" if missing else ""))
        if min(counts.values()) < N_MIN:
            txt.append(f"  [!] fewer than {N_MIN} images for some rows -> the confirmation is INCOMPLETE")
        common = set.intersection(*[set(pc[pc.method == r].image) for r in present])
        sub = pc[pc.method.isin(present) & pc.image.isin(common)]
        txt.append(f"[{mode}] medians on the {len(common)} images all rows finished (L2, LPIPS, SSIM>=0.99 share)")
        for v, f_ in (("pix_l2", "{:.3f}"), ("lpips", "{:.4f}")):
            t = sub.pivot_table(index="method", columns="budget", values=v, aggfunc="median").reindex(present)
            txt += ["  " + s for s in t.to_string(float_format=lambda x, f=f_: f.format(x)).splitlines()]
        if "ssim" in sub.columns:
            t = sub.assign(ok=(sub.ssim >= 0.99).astype(float)).pivot_table(index="method", columns="budget",
                                                                             values="ok", aggfunc="mean")
            txt += ["  " + s for s in t.reindex(present).to_string(float_format=lambda x: f"{x:.2f}").splitlines()]
        # ---- confirmatory quantities
        last = budgets[-1]
        if mode == "non_targeted":
            s = l2(pc, M, REF, last, a.B)
            tests.append(("C1 NT floor: M/LGC_RC L2@10k < 1, upper <= 0.67", s["p_less"] if s else float("nan"),
                          bool(s and s["ci_hi"] <= FLOOR_UB), fmt_l2(s)))
            for px in PIX[mode]:
                for b in ("AUC", last):
                    s = l2(pc, M, px, b, a.B)
                    ci_claims.append((f"C3 NT non-inferiority M/{px} L2 {b}: upper <= {NI_NT}",
                                      bool(s and s["ci_hi"] <= NI_NT), fmt_l2(s)))
            s = lp(pc, M, "CGBA", 1000, a.B)
            tests.append(("C6 NT LPIPS@1000 (M - CGBA) < 0, upper < 0", s["p_less"] if s else float("nan"),
                          bool(s and s["ci_hi"] < 0), fmt_lp(s)))
        else:
            s = l2(pc, M, REF, "AUC", a.B)
            ci_claims.append((f"C2 T non-inferiority M/LGC_RC L2 AUC: upper <= {NI_T}",
                              bool(s and s["ci_hi"] <= NI_T), fmt_l2(s)))
            for px, tag in (("CGBA_H", "C4a"), ("TtBA", "C4b")):
                s = l2(pc, M, px, "AUC", a.B)
                tests.append((f"{tag} T superiority M/{px} L2 AUC < 1, upper < 1", s["p_less"] if s else float("nan"),
                              bool(s and s["ci_hi"] < 1), fmt_l2(s)))
        for px in PIX[mode]:
            s = lp(pc, M, px, "AUC", a.B)
            tests.append((f"C5 {tg} LPIPS AUC (M - {px}) < 0, upper < 0", s["p_less"] if s else float("nan"),
                          bool(s and s["ci_hi"] < 0), fmt_lp(s)))
        # ---- descriptive contrasts
        txt.append(f"[{mode}] descriptive contrasts (L2 ratio per budget / AUC; LPIPS AUC delta)")
        pairs = [(M, REF), (M, ISO), (M, NAIVE), (NAIVE, REF), (REF, PIX[mode][0]), (REF, PIX[mode][1])]
        pairs += [(M, px) for px in PIX[mode]] + [(ISO, px) for px in PIX[mode]]
        for x, y in pairs:
            if x in present and y in present:
                out, cells = contrast(pc, x, y, a.B, budgets)
                rows += [{"mode": mode, "contrast": f"{x} / {y}", **r} for r in out]
                txt.append(f"  {x} / {y}: " + " | ".join(cells))
        rt = runs_table(a.res, mode)
        if rt is not None and not rt.empty:
            txt.append(f"[{mode}] run diagnostics (medians; gate_open_pct = share of images)")
            txt += ["  " + s for s in rt.to_string(float_format=lambda x: f"{x:.3g}").splitlines()]
    # ---- verdicts
    adj = holm([t[1] for t in tests])
    txt.append("\nCONFIRMATORY VERDICTS (Holm over the tested family; a claim needs p_holm < 0.05 AND its CI condition)")
    for (name, p, ci_ok, cell), ph in zip(tests, adj):
        ok = bool(np.isfinite(ph) and ph < 0.05 and ci_ok)
        txt.append(f"  {'CONFIRMED' if ok else 'not confirmed':13s} {name}: {cell}  p_holm={ph:.3g}")
        rows.append({"mode": "claim", "contrast": name, "p": p, "p_holm": ph, "ci_ok": ci_ok, "confirmed": ok})
    for name, ok, cell in ci_claims:
        txt.append(f"  {'CONFIRMED' if ok else 'not confirmed':13s} {name}: {cell}")
        rows.append({"mode": "claim", "contrast": name, "ci_ok": ok, "confirmed": ok})
    txt.append("  (failed claims are dropped or re-scoped in the paper; the method stays LGC_RC_EGC)")
    pd.DataFrame(rows).to_csv(os.path.join(a.res, "confirm_37.csv"), index=False)
    with open(os.path.join(a.res, "confirm_37.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
