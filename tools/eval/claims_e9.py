#!/usr/bin/env python3
"""Registered ablation claims of prereg_build38_amendment3.txt (section 6; n = 200 per cell). L, 2026-10-23.

Rows (frozen engine; same rows, victims, budget, seed and command line as the B5 main study; only --methods differs):
  LGC_RC_EGC     the method: the B5 cells, read from the B5 export (paper_exports/b5a2/cells, .../metrics), not re-run
  LGC_RC         no ambient block: isotropic latent chord (CGBA circle) + residual coordinate
  LGC_RC_EGCI    circle instead of the pull-back ellipse in the latent block (R ~ I), everything else as EGC
  LGC_RC_HT1000  naive switch: LGC_RC for 1000 queries, then pixel TtBA on the full perturbation

Registered claims (Holm within each family, alpha = 0.05; "<" claims are one-sided Wilcoxon, alternative "less"):
  A1 (m = 3, non-targeted, every victim): L2@10k ratio EGC / LGC_RC < 1 and upper 95% bound < 1.
  A2 (m = 3, targeted, every victim):     LPIPS-vgg AUC delta (EGC - HT1000) < 0 and upper 95% bound < 0.
  A3 (m = 3, targeted, every victim):     L2@10k ratio EGC / EGCI < 1 and upper 95% bound < 1.
  A4 (interval only, targeted, every victim): L2 AUC ratio EGC / LGC_RC, upper 95% bound <= 1.03.
Every other (variant, estimand, mode) pair is descriptive (decision "descriptive", no p-value): L2 AUC ratio,
L2@10k ratio, and LPIPS-vgg / LPIPS-alex / DISTS / SSIM AUC deltas of EGC against each variant, both modes.

Statistics: the frozen conventions of tools/eval/claims_registered.py, imported (not copied): per-image AUC over
contract constants.auc_budgets (mean of log L2; mean of the metric otherwise); L2 ratio = paired geometric mean with
the bootstrap 95% percentile interval of exp(mean log ratio); deltas = median paired difference with the bootstrap
interval of the median; B = contract constants.stats.bootstrap_B; bootstrap seed 0. SSIM is "higher is better": its
delta (EGC - variant) is descriptive only. A comparison uses the images for which both rows have a value;
n_excluded = n_main - n_pairs. --no-devoverlap recomputes every row without the dev-overlap rows (all descriptive).

Also writes levels_e9.csv: per (method, victim, mode, metric, budget) the n, mean and median of pix_l2 (per
checkpoint), adv_float, lpips_vgg, lpips_alex, dists and ssim (offline metrics) at budgets 1000 and 10000.

usage (repository root, after tools/eval/e9_finish.sh copied the cells and computed the metrics):
  python tools/eval/claims_e9.py [--no-devoverlap]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claims_registered as cr  # noqa: E402  (frozen: auc_series, at_budget, paired_l2, paired_delta, holm, sha256)

OURS = "LGC_RC_EGC"
VARIANTS = ["LGC_RC", "LGC_RC_EGCI", "LGC_RC_HT1000"]
VSHORT = {"LGC_RC": "RC", "LGC_RC_EGCI": "EGCI", "LGC_RC_HT1000": "HT1000"}
VICTIMS = ["vgg19", "resnet50", "vit_b_32"]
MODES = ["non_targeted", "targeted"]
SHORT = {"non_targeted": "NT", "targeted": "T"}
FINAL = 10000
LEVEL_BUDGETS = [1000, 10000]
NI_A4 = 1.03

KINDS = {  # kind -> (metric column, estimand label, short tag)
    "l2_auc": ("pix_l2", "L2 AUC ratio", "l2auc"),
    "l2_10k": ("pix_l2", "L2@10k ratio", "l2k"),
    "lpips_vgg_auc": ("lpips_vgg", "LPIPS-vgg AUC delta", "lpvgg"),
    "lpips_alex_auc": ("lpips_alex", "LPIPS-alex AUC delta", "lpalex"),
    "dists_auc": ("dists", "DISTS AUC delta", "dists"),
    "ssim_auc": ("ssim", "SSIM AUC delta", "ssim"),
}
# (family, mode, variant, kind): the registered claims of amendment 3, section 6
REGISTERED = [
    ("A1", "non_targeted", "LGC_RC", "l2_10k"),
    ("A2", "targeted", "LGC_RC_HT1000", "lpips_vgg_auc"),
    ("A3", "targeted", "LGC_RC_EGCI", "l2_10k"),
    ("A4", "targeted", "LGC_RC", "l2_auc"),
]
HOLM_FAMILIES = ("A1", "A2", "A3")


def cell_name(method: str, arch: str, mode: str) -> str:
    if method == OURS:
        return f"b5_main_{OURS}_{arch}_{SHORT[mode]}"
    return f"e9_{method}_{arch}_{SHORT[mode]}"


def read_pc(root: str, cell: str, mode: str) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(root, cell, mode, "per_checkpoint*.csv")))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["image"] = df["image"].astype(str).str.replace(".JPEG", "", regex=False)
    return df


def read_metrics(root: str, cell: str) -> Optional[pd.DataFrame]:
    p = os.path.join(root, cell, "metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["image"] = df["image"].astype(str).str.replace(".JPEG", "", regex=False)
    return df


def series(pc: pd.DataFrame, met: pd.DataFrame, method: str, budgets: List[int]) -> Dict[str, pd.Series]:
    return {
        "l2_auc": cr.auc_series(pc, method, "pix_l2", budgets, log=True),
        "l2_10k": cr.at_budget(pc, method, "pix_l2", FINAL, log=True),
        "lpips_vgg_auc": cr.auc_series(met, method, "lpips_vgg", budgets),
        "lpips_alex_auc": cr.auc_series(met, method, "lpips_alex", budgets),
        "dists_auc": cr.auc_series(met, method, "dists", budgets),
        "ssim_auc": cr.auc_series(met, method, "ssim", budgets),
    }


def levels(pc: pd.DataFrame, met: pd.DataFrame, method: str, arch: str, mode: str) -> List[dict]:
    out = []
    for src, cols in ((pc, ("pix_l2", "adv_float")), (met, ("lpips_vgg", "lpips_alex", "dists", "ssim"))):
        for col in cols:
            for b in LEVEL_BUDGETS:
                v = pd.to_numeric(src[(src.method == method) & (src.budget == b)][col], errors="coerce").dropna()
                out.append({"method": method, "victim": arch, "mode": mode, "metric": col, "budget": b,
                            "n": int(len(v)), "mean": float(v.mean()) if len(v) else float("nan"),
                            "median": float(v.median()) if len(v) else float("nan")})
    return out


def decide(r: dict) -> str:
    fam = r["family"]
    if fam in ("A1", "A3"):
        return "confirmed" if (r["p_holm"] is not None and r["p_holm"] < 0.05 and r["ci_high"] < 1) else "not_confirmed"
    if fam == "A2":
        return "confirmed" if (r["p_holm"] is not None and r["p_holm"] < 0.05 and r["ci_high"] < 0) else "not_confirmed"
    if fam == "A4":
        return "confirmed" if r["ci_high"] <= NI_A4 else "not_confirmed"
    return "descriptive"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b5-cells", default="paper_exports/b5a2/cells")
    ap.add_argument("--b5-metrics", default="paper_exports/b5a2/metrics")
    ap.add_argument("--e9-cells", default="paper_exports/e9/cells")
    ap.add_argument("--e9-metrics", default="paper_exports/e9/metrics")
    ap.add_argument("--contract", default="contract/paper_contract.yaml")
    ap.add_argument("--amendment", default="prereg_build38_amendment3.txt")
    ap.add_argument("--dev-overlap", default="paper_exports/b5a2/derived/dev_overlap_rows.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper_exports/e9/claims")
    ap.add_argument("--no-devoverlap", action="store_true")
    a = ap.parse_args(argv)

    c = cr.load_contract(a.contract)
    budgets = [int(x) for x in c["constants"]["auc_budgets"]]
    B = int(c["constants"]["stats"]["bootstrap_B"])
    n_main = int(c["constants"]["n_main"])
    dev: Dict[str, list] = {}
    if a.no_devoverlap:
        with open(a.dev_overlap) as fh:
            dev = json.load(fh)["dev_overlap_rows_1_200"]

    data: Dict[tuple, dict] = {}
    lev: List[dict] = []
    missing: List[str] = []
    for arch in VICTIMS:
        for mode in MODES:
            for method in [OURS] + VARIANTS:
                cell = cell_name(method, arch, mode)
                croot, mroot = (a.b5_cells, a.b5_metrics) if method == OURS else (a.e9_cells, a.e9_metrics)
                pc, met = read_pc(croot, cell, mode), read_metrics(mroot, cell)
                if pc is None or met is None:
                    missing.append(cell)
                    continue
                data[(method, arch, mode)] = series(pc, met, method, budgets)
                lev += levels(pc, met, method, arch, mode)
    if missing:
        print("[!] missing per_checkpoint or metrics for: " + ", ".join(missing))
        return 1

    def paired(arch: str, mode: str, base: str, kind: str) -> Optional[dict]:
        o, b = data[(OURS, arch, mode)][kind], data[(base, arch, mode)][kind]
        if a.no_devoverlap:
            drop = set(dev.get(f"{arch}_{mode}", []))
            o, b = o[~o.index.isin(drop)], b[~b.index.isin(drop)]
        j = pd.DataFrame({"o": o, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(j) < 2:
            return None
        d = j["o"].to_numpy(float) - j["b"].to_numpy(float)
        s = cr.paired_l2(d, B, a.seed) if kind in ("l2_auc", "l2_10k") else cr.paired_delta(d, B, a.seed)
        s["n_excluded"] = n_main - s["n_pairs"]
        return s

    rows: List[dict] = []
    registered = {(m, v, k): f for f, m, v, k in REGISTERED}
    for f, mode, v, kind in REGISTERED:
        for arch in VICTIMS:
            s = paired(arch, mode, v, kind)
            if s is None:
                continue
            col, lab, _ = KINDS[kind]
            rows.append(dict(claim_id=f"{f}_{arch}", family=f, victim=arch, mode=mode, baseline=v, metric=col,
                             estimand=lab, **s))
    for arch in VICTIMS:
        for mode in MODES:
            for v in VARIANTS:
                for kind, (col, lab, tag) in KINDS.items():
                    if (mode, v, kind) in registered:
                        continue
                    s = paired(arch, mode, v, kind)
                    if s is None:
                        continue
                    rows.append(dict(claim_id=f"D_{VSHORT[v]}_{tag}_{arch}_{SHORT[mode]}", family=None,
                                     victim=arch, mode=mode, baseline=v, metric=col, estimand=lab, **s))

    for f in HOLM_FAMILIES:
        idx = [i for i, r in enumerate(rows) if r["family"] == f]
        for i, ph in zip(idx, cr.holm([rows[i]["p_one_sided"] for i in idx])):
            rows[i]["p_holm"] = ph
    for r in rows:
        if r["family"] not in HOLM_FAMILIES:
            r["p_one_sided"], r["p_holm"] = None, None
        r["decision"] = "descriptive" if a.no_devoverlap else decide(r)

    here = os.path.abspath(__file__)
    meta = {
        "script": "tools/eval/claims_e9.py", "script_sha256": cr.sha256(here),
        "conventions": "tools/eval/claims_registered.py",
        "conventions_sha256": cr.sha256(os.path.join(os.path.dirname(here), "claims_registered.py")),
        "amendment": a.amendment,
        "amendment_sha256": cr.sha256(a.amendment) if os.path.exists(a.amendment) else None,
        "contract_sha256": cr.sha256(a.contract), "bootstrap_B": B, "bootstrap_seed": a.seed,
        "auc_budgets": budgets, "n_main": n_main, "alpha": 0.05, "test": "wilcoxon signed-rank, one-sided (less)",
        "families": {f: sum(r["family"] == f for r in rows) for f in ("A1", "A2", "A3", "A4")},
        "holm_families": list(HOLM_FAMILIES), "a4_margin": NI_A4, "no_devoverlap": bool(a.no_devoverlap),
        "ours_source": f"{a.b5_cells} + {a.b5_metrics} (B5, tag b5a2-done-2026-10-23)",
    }
    os.makedirs(a.out, exist_ok=True)
    name = "claims_no_devoverlap.json" if a.no_devoverlap else "claims.json"
    with open(os.path.join(a.out, name), "w") as fh:
        json.dump({"meta": meta, "rows": rows, "excluded_comparisons": []}, fh, indent=1, default=float)
    if not a.no_devoverlap:
        pd.DataFrame(lev).to_csv(os.path.join(a.out, "levels_e9.csv"), index=False)

    print("families (m): %s" % meta["families"])
    for r in rows:
        if r["family"]:
            ph = "-" if r["p_holm"] is None else "%.3g" % r["p_holm"]
            print("  %-14s %-3s %-13s est=%.4f [%.4f, %.4f] p_holm=%s n=%d excl=%d -> %s" % (
                r["claim_id"], r["family"], r["baseline"], r["estimate"], r["ci_low"], r["ci_high"], ph,
                r["n_pairs"], r["n_excluded"], r["decision"]))
    print("rows: %d (%d descriptive); wrote %s" % (
        len(rows), sum(r["family"] is None for r in rows), os.path.join(a.out, name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
