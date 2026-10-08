#!/usr/bin/env python3
"""Build 34b selection analysis: {CGBA, TtBA} x {latent chart, pull-back chart} (+ the 33b PVRE), paired.

Usage (AutoDL, after the sel_34b runs of both modes into one --out_dir):
    python analyze_34b.py --res Results/sel34b_resnet50 --arch ResNet-50

Reads <res>/<mode>/per_checkpoint*.csv; writes <res>/analyze_34b.txt and <res>/analyze_34b.csv.

Statistics (all paired over the images every compared method finished; same definitions as report_33b /
confirm_34): L2 = geometric-mean ratio exp(mean log(L2_a / L2_b)); L2 AUC = per image mean over the checkpoint
budgets of log L2, then the geo-mean ratio; LPIPS = median paired difference; W/L; Wilcoxon signed-rank;
image-level bootstrap 95% CI (B = 10000).

Pre-registered selection rule (fixed before the selection run; seed 3141, n = 50, 992 and 2027 excluded):
  candidates: LGC_RC_PB, LGC_RC_PBT, LGC_RC_TT, LGC_RC_PVRE; reference LGC_RC (the original LGC update + RC)
  primary   : targeted L2 AUC ratio vs LGC_RC <= 0.97 and one-sided Wilcoxon p < 0.05 after Holm over the
              four candidates;
  safety    : non-targeted L2 AUC ratio, upper 95% bound <= 1.03; LPIPS AUC delta (median) <= +0.005 in both
              modes;
  choice    : among the candidates that pass, the smallest targeted L2 AUC ratio; within 1% of it, the
              simpler one in the order PB < PBT < TT < PVRE.  If none passes: no change (LGC_RC).
Factorial contrasts (reported, not part of the rule): chart effect under CGBA (PB / LGC_RC) and under TtBA
(PBT / TT); update effect in each chart (TT / LGC_RC, PBT / PB); interaction (PBT/TT) / (PB/LGC_RC).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from confirm_34 import boot_ci, load, per_image, wilcoxon  # noqa: E402

REF = "LGC_RC"
CANDIDATES = ["LGC_RC_PB", "LGC_RC_PBT", "LGC_RC_TT", "LGC_RC_PVRE"]
SIMPLICITY = {m: i for i, m in enumerate(CANDIDATES)}
FACTORIAL = [("chart | CGBA", "LGC_RC_PB", "LGC_RC"), ("chart | TtBA", "LGC_RC_PBT", "LGC_RC_TT"),
             ("chart | CGBA-H", "LGC_RC_PBH", "LGC_H_RC"), ("TtBA vs CGBA | latent", "LGC_RC_TT", "LGC_RC"),
             ("TtBA vs CGBA | chart", "LGC_RC_PBT", "LGC_RC_PB"), ("PVRE vs PB", "LGC_RC_PVRE", "LGC_RC_PB")]


def holm(ps):
    """Holm-adjusted p-values (same order as ps)."""
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(np.where(np.isfinite(ps), ps, np.inf))
    adj, run = np.full(len(ps), np.nan), 0.0
    m = int(np.isfinite(ps).sum())
    for rank, i in enumerate(order):
        if not math.isfinite(ps[i]):
            continue
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def l2_summary(d: np.ndarray, B: int) -> dict:
    lo, hi = boot_ci(d, np.mean, B)
    return {"n": len(d), "ratio": math.exp(d.mean()), "ci_lo": math.exp(lo), "ci_hi": math.exp(hi),
            "wins": int((d < 0).sum()), "losses": int((d > 0).sum()), "p": wilcoxon(d), "p_less": wilcoxon(d, "less")}


def interaction(pc, a1, b1, a2, b2, budget):
    """Per-image (log a1 - log b1) - (log a2 - log b2) on the images all four finished."""
    w = pc.pivot_table(index=["image", "budget"], columns="method", values="pix_l2")
    if any(m not in w for m in (a1, b1, a2, b2)):
        return np.array([])
    w = np.log(w[[a1, b1, a2, b2]].dropna())
    x = w.groupby(level="image").mean() if budget == "AUC" else w.xs(budget, level="budget")
    return ((x[a1] - x[b1]) - (x[a2] - x[b2])).to_numpy(dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--arch", default="?")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    rows, txt = [], [f"Build 34b selection analysis, {a.arch}, res = {a.res}"]
    data = {m: load(a.res, m) for m in ("targeted", "non_targeted")}
    data = {m: v for m, v in data.items() if v is not None}
    if not data:
        raise SystemExit(f"no per_checkpoint*.csv under {a.res}/<mode>/")
    cand_stats = {c: {} for c in CANDIDATES}
    for mode, pc in data.items():
        budgets = sorted(pc.budget.unique())
        present = set(pc.method)
        txt.append(f"\n[{mode}] images per method: "
                   f"{pc.groupby('method').image.nunique().to_dict()}")
        txt.append(f"[{mode}] L2 ratio vs {REF} (geo-mean, 95% CI, W/L, two-sided p) at "
                   f"{', '.join(str(b) for b in budgets)} and AUC")
        for m in [c for c in CANDIDATES + ["LGC_H_RC", "LGC_RC_PBH"] if c in present]:
            cells = []
            for b in list(budgets) + ["AUC"]:
                d = per_image(pc, m, REF, "pix_l2", b)
                if not len(d):
                    continue
                s = l2_summary(d, a.B)
                rows.append({"mode": mode, "contrast": f"{m} / {REF}", "metric": "pix_l2", "budget": b, **s})
                star = "*" if s["p"] < 0.05 else ""
                cells.append(f"{b}: {s['ratio']:.3f}{star} [{s['ci_lo']:.3f},{s['ci_hi']:.3f}] ({s['wins']}/{s['losses']})")
                if b == "AUC" and m in cand_stats:
                    cand_stats[m][mode] = s
            dl = per_image(pc, m, REF, "lpips", "AUC")
            if len(dl):
                lo, hi = boot_ci(dl, np.median, a.B)
                lp = {"n": len(dl), "delta": float(np.median(dl)), "ci_lo": lo, "ci_hi": hi,
                      "wins": int((dl < 0).sum()), "losses": int((dl > 0).sum()), "p": wilcoxon(dl)}
                rows.append({"mode": mode, "contrast": f"{m} / {REF}", "metric": "lpips", "budget": "AUC", **lp})
                if m in cand_stats:
                    cand_stats[m][mode + "_lpips"] = lp
                cells.append(f"LPIPS AUC d {lp['delta']:+.4f} [{lp['ci_lo']:+.4f},{lp['ci_hi']:+.4f}]")
            txt.append(f"  {m:12s} " + " | ".join(cells))
        txt.append(f"[{mode}] factorial contrasts (L2, AUC and last budget {budgets[-1]})")
        for lab, x, y in FACTORIAL:
            if x in present and y in present:
                cells = []
                for b in (budgets[-1], "AUC"):
                    d = per_image(pc, x, y, "pix_l2", b)
                    if len(d):
                        s = l2_summary(d, a.B)
                        rows.append({"mode": mode, "contrast": f"{x} / {y} ({lab})", "metric": "pix_l2", "budget": b, **s})
                        cells.append(f"{b}: {s['ratio']:.3f}{'*' if s['p'] < 0.05 else ''} "
                                     f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] ({s['wins']}/{s['losses']})")
                txt.append(f"  {lab:24s} {x}/{y}: " + " | ".join(cells))
        if all(m in present for m in ("LGC_RC_PB", "LGC_RC", "LGC_RC_PBT", "LGC_RC_TT")):
            d = interaction(pc, "LGC_RC_PBT", "LGC_RC_TT", "LGC_RC_PB", "LGC_RC", "AUC")
            s = l2_summary(d, a.B)
            rows.append({"mode": mode, "contrast": "interaction (PBT/TT)/(PB/LGC_RC)", "metric": "pix_l2",
                         "budget": "AUC", **s})
            txt.append(f"  {'interaction':24s} (PBT/TT)/(PB/LGC_RC) AUC: {s['ratio']:.3f} "
                       f"[{s['ci_lo']:.3f},{s['ci_hi']:.3f}] p={s['p']:.3g}")

    # ---- pre-registered selection rule
    txt.append("\nPre-registered selection rule")
    avail = [c for c in CANDIDATES if "targeted" in cand_stats[c]]
    p_primary = [cand_stats[c]["targeted"]["p_less"] for c in avail]
    p_holm = holm(p_primary)
    passing = []
    for c, ph in zip(avail, p_holm):
        st = cand_stats[c]
        t = st["targeted"]
        prim = t["ratio"] <= 0.97 and ph < 0.05
        nt = st.get("non_targeted")
        safe_nt = nt is not None and nt["ci_hi"] <= 1.03
        lp_ok = all(st.get(f"{m}_lpips", {}).get("delta", np.inf) <= 0.005 for m in ("targeted", "non_targeted"))
        ok = prim and safe_nt and lp_ok
        if ok:
            passing.append(c)
        txt.append(f"  {c:12s} primary T AUC {t['ratio']:.3f} (Holm one-sided p={ph:.3g}) {'PASS' if prim else 'FAIL'}"
                   f" | NT AUC upper {nt['ci_hi'] if nt else float('nan'):.3f} {'PASS' if safe_nt else 'FAIL'}"
                   f" | LPIPS {'PASS' if lp_ok else 'FAIL'}  ->  {'PASS' if ok else 'fail'}")
    if passing:
        best = min(cand_stats[c]["targeted"]["ratio"] for c in passing)
        near = [c for c in passing if cand_stats[c]["targeted"]["ratio"] <= best * 1.01]
        winner = min(near, key=lambda c: SIMPLICITY[c])
        txt.append(f"  -> selected: {winner} (freeze it; confirm on a fresh split / the final protocols)")
    else:
        txt.append("  -> no candidate passes: keep LGC_RC (no change)")
    os.makedirs(a.res, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.res, "analyze_34b.csv"), index=False)
    with open(os.path.join(a.res, "analyze_34b.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
