#!/usr/bin/env python3
"""Registered claims for the LGC-EGC paper (build 38 amendment 2, n = 200 per cell).

Implements ONLY the quoted claim text (prereg_build38.txt section 6 and
prereg_build38_amendment2.txt section 7; see reports/raw/B5a2/export/claims_spec.md):

  T1 (Holm family, m = 3)  targeted : L2 AUC ratio EGC / TtBA < 1 and upper 95% bound < 1.
  P1 (Holm family, m = 6)  both     : LPIPS-vgg AUC delta (EGC - TtBA) < 0 and upper 95% bound < 0.
  T2 (CI-only)             targeted : L2@10k ratio upper bound <= 1.03.
  N1 (CI-only)             NT       : L2 AUC ratio and L2@10k ratio, upper bound <= 1.15.
  P2 (Holm family)         all      : LPIPS-vgg AUC delta (EGC - baseline) < 0, upper 95% bound < 0,
                                      baseline in {TtBA, CGBA(NT), CGBA-H(T), ARS_OPT}, over baselines x victims x modes.
  Supporting (CI-only)     all      : the same as P2 for LPIPS-alex and DISTS; perceptual ASR AUC.

Statistics (frozen convention, confirm_ttba.py):
  AUC      : per-image mean of log(pix_l2) over budgets {500,1000,2500,5000,10000} (L2);
             per-image mean of the metric over the same budgets (LPIPS / DISTS); per-image mean of the
             perceptual-ASR indicator over the same budgets (perceptual ASR AUC).
  L2 ratio : paired geometric mean, exp(mean(log o - log b)); CI = exp(bootstrap 95% percentile of the mean).
  delta    : median paired difference; CI = bootstrap 95% percentile of the median (confirm_ttba.py:176,196).
  test     : one-sided Wilcoxon signed-rank, alternative "less" (d < 0 => ours better).
  Holm     : within each family (T1, P1, P2).

Reads bootstrap B from contract constants.stats.bootstrap_B (10000). The bootstrap seed is a CLI argument
(default 0); the contract has no seed constant (make_tables_figures.py also defaults to 0).

Decision rule:
  - Holm families: confirmed iff p_holm < 0.05 AND the CI condition (T1: ci_high < 1; P1/P2: ci_high < 0).
  - T2 / N1 (CI-only): confirmed iff ci_high <= 1.03 (T2) / 1.15 (N1).
  - Supporting LPIPS-alex / DISTS (CI-only): confirmed iff ci_high < 0.
  - Supporting perceptual ASR AUC: no registered direction; decision = "descriptive".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

OURS = "LGC_RC_EGC"
VICTIMS = ["vgg19", "resnet50", "vit_b_32"]
MODES = ["non_targeted", "targeted"]
FINAL = 10000
NI_T, NI_NT = 1.03, 1.15

# EXCLUDED-FROM-CLAIMS cells (main study only), from reports/B5a2_stop_decisions.md
EXCLUDED_CELLS = {
    ("CGBA_H", "vgg19", "targeted"),
    ("CGBA", "resnet50", "non_targeted"),
}


def methods_for(mode: str) -> List[str]:
    if mode == "non_targeted":
        return [OURS, "TtBA", "CGBA", "ARS_OPT"]
    return [OURS, "TtBA", "CGBA_H", "ARS_OPT"]


def cell_dir(method: str, arch: str, mode: str) -> str:
    m = "NT" if mode == "non_targeted" else "T"
    return f"b5_main_{method}_{arch}_{m}"


def load_pc(results_root: str, method: str, arch: str, mode: str) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(results_root, cell_dir(method, arch, mode), mode, "per_checkpoint*.csv")))
    if not paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["image"] = df["image"].astype(str).str.replace(".JPEG", "", regex=False)
    return df


def load_metrics(metrics_root: str, method: str, arch: str, mode: str) -> Optional[pd.DataFrame]:
    p = os.path.join(metrics_root, cell_dir(method, arch, mode), "metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["image"] = df["image"].astype(str)
    return df


def auc_series(df: pd.DataFrame, method: str, col: str, budgets: List[int], log: bool = False) -> pd.Series:
    sub = df[df.method == method]
    sub = sub[sub.budget.isin(budgets)].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    if log:
        sub[col] = np.log(sub[col])
    return sub.groupby("image")[col].mean()


def at_budget(df: pd.DataFrame, method: str, col: str, budget: int, log: bool = False) -> pd.Series:
    sub = df[(df.method == method) & (df.budget == budget)].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    if log:
        sub[col] = np.log(sub[col])
    return sub.set_index("image")[col]


def wilcoxon_less(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(d, zero_method="wilcox", alternative="less").pvalue)


def bootstrap_ci(d: np.ndarray, stat, B: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    vals = stat(d[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_l2(d: np.ndarray, B: int, seed: int) -> dict:
    lo, hi = bootstrap_ci(d, lambda x: np.exp(x.mean(axis=1)), B, seed)
    return {
        "estimate": float(np.exp(d.mean())),
        "ci_low": lo, "ci_high": hi,
        "p_one_sided": wilcoxon_less(d),
        "n_pairs": int(len(d)),
    }


def paired_delta(d: np.ndarray, B: int, seed: int) -> dict:
    lo, hi = bootstrap_ci(d, lambda x: np.median(x, axis=1), B, seed)
    return {
        "estimate": float(np.median(d)),
        "ci_low": lo, "ci_high": hi,
        "p_one_sided": wilcoxon_less(d),
        "n_pairs": int(len(d)),
    }


def holm(ps: List[float]) -> List[float]:
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(np.where(np.isfinite(ps), ps, np.inf))
    adj, run = np.full(len(ps), np.nan), 0.0
    m = int(np.isfinite(ps).sum())
    for rank, i in enumerate(order):
        if not math.isfinite(ps[i]):
            continue
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return [float(x) for x in adj]


def load_contract(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_asr_auc(met: pd.DataFrame, method: str, budgets: List[int], tau: dict, mode: str) -> pd.Series:
    sub = met[met.method == method]
    sub = sub[sub.budget.isin(budgets)].copy()
    tl = float(tau[mode]["tau_L"])
    ts = float(tau[mode]["tau_S"])
    for col in ("adv_flag", "lpips_vgg", "ssim"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub["ind"] = ((sub["adv_flag"] == 1) & (sub["lpips_vgg"] <= tl) & (sub["ssim"] >= ts)).astype(float)
    return sub.groupby("image")["ind"].mean()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="Results")
    ap.add_argument("--metrics-root", default="paper_exports/b5a2/metrics")
    ap.add_argument("--contract", default="contract/paper_contract.yaml")
    ap.add_argument("--decisions", default="reports/B5a2_stop_decisions.md")
    ap.add_argument("--dev-overlap", default="paper_exports/b5a2/derived/dev_overlap_rows.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper_exports/b5a2/claims")
    ap.add_argument("--no-devoverlap", action="store_true")
    a = ap.parse_args()

    c = load_contract(a.contract)
    AUC_BUDGETS = [int(x) for x in c["constants"]["auc_budgets"]]
    B = int(c["constants"]["stats"]["bootstrap_B"])
    N_MAIN = int(c["constants"]["n_main"])
    tau = c["constants"]["asr_perc"]

    dev_overlap = {}
    if os.path.exists(a.dev_overlap):
        dev_overlap = json.load(open(a.dev_overlap))["dev_overlap_rows_1_200"]

    def dev_set(arch: str, mode: str) -> set:
        return set(dev_overlap.get(f"{arch}_{mode}", []))

    # ---- preload per-image AUC values for every (method, arch, mode)
    data: Dict[tuple, dict] = {}
    for arch in VICTIMS:
        for mode in MODES:
            for method in methods_for(mode):
                pc = load_pc(a.results_root, method, arch, mode)
                met = load_metrics(a.metrics_root, method, arch, mode)
                d = {}
                if pc is not None:
                    d["l2_auc"] = auc_series(pc, method, "pix_l2", AUC_BUDGETS, log=True)
                    d["l2_10k"] = at_budget(pc, method, "pix_l2", FINAL, log=True)
                if met is not None:
                    d["lpips_vgg_auc"] = auc_series(met, method, "lpips_vgg", AUC_BUDGETS)
                    d["lpips_alex_auc"] = auc_series(met, method, "lpips_alex", AUC_BUDGETS)
                    d["dists_auc"] = auc_series(met, method, "dists", AUC_BUDGETS)
                    d["asr_auc"] = perceptual_asr_auc(met, method, AUC_BUDGETS, tau, mode)
                data[(method, arch, mode)] = d

    rows: List[dict] = []
    excluded: List[dict] = []

    def paired(arch: str, mode: str, base: str, kind: str) -> Optional[dict]:
        ours = data[(OURS, arch, mode)].get(kind)
        b = data[(base, arch, mode)].get(kind)
        if ours is None or b is None:
            return None
        if a.no_devoverlap:
            drop = dev_set(arch, mode)
            ours = ours[~ours.index.isin(drop)]
            b = b[~b.index.isin(drop)]
        j = pd.DataFrame({"o": ours, "b": b}).dropna()
        if len(j) < 2:
            return None
        o = j["o"].to_numpy(float)
        bb = j["b"].to_numpy(float)
        if kind in ("l2_auc", "l2_10k"):
            s = paired_l2(o - bb, B, a.seed)
        else:
            s = paired_delta(o - bb, B, a.seed)
        s["n_excluded"] = N_MAIN - s["n_pairs"]
        return s

    # ---- T1, T2, N1 (vs TtBA)
    t1 = []
    for arch in VICTIMS:
        s = paired(arch, "targeted", "TtBA", "l2_auc")
        if s:
            rows.append(dict(claim_id=f"T1_{arch}", family="T1", victim=arch, mode="targeted",
                             baseline="TtBA", metric="pix_l2", estimand="L2 AUC ratio", **s))
            t1.append(len(rows) - 1)
    for arch in VICTIMS:
        s = paired(arch, "targeted", "TtBA", "l2_10k")
        if s:
            rows.append(dict(claim_id=f"T2_{arch}", family=None, victim=arch, mode="targeted",
                             baseline="TtBA", metric="pix_l2", estimand="L2@10k ratio",
                             **s))
    for arch in VICTIMS:
        s = paired(arch, "non_targeted", "TtBA", "l2_auc")
        if s:
            rows.append(dict(claim_id=f"N1_{arch}_AUC", family=None, victim=arch, mode="non_targeted",
                             baseline="TtBA", metric="pix_l2", estimand="L2 AUC ratio",
                             **s))
        s = paired(arch, "non_targeted", "TtBA", "l2_10k")
        if s:
            rows.append(dict(claim_id=f"N1_{arch}_10k", family=None, victim=arch, mode="non_targeted",
                             baseline="TtBA", metric="pix_l2", estimand="L2@10k ratio",
                             **s))

    # ---- P1 (LPIPS-vgg AUC delta vs TtBA)
    p1 = []
    for arch in VICTIMS:
        for mode in MODES:
            s = paired(arch, mode, "TtBA", "lpips_vgg_auc")
            if s:
                rows.append(dict(claim_id=f"P1_{arch}_{mode}", family="P1", victim=arch, mode=mode,
                                 baseline="TtBA", metric="lpips_vgg", estimand="LPIPS-vgg AUC delta", **s))
                p1.append(len(rows) - 1)

    # ---- P2 + supporting (all baselines x victims x modes)
    p2 = []
    for arch in VICTIMS:
        for mode in MODES:
            for base in methods_for(mode):
                if base == OURS:
                    continue
                is_excluded = (base, arch, mode) in EXCLUDED_CELLS
                vgg = paired(arch, mode, base, "lpips_vgg_auc")
                if vgg is None:
                    continue
                if is_excluded:
                    excluded.append({"baseline": base, "victim": arch, "mode": mode,
                                     "metric": "lpips_vgg",
                                     "reason": "EXCLUDED-FROM-CLAIMS cell (domain gate failed)"})
                    continue
                rows.append(dict(claim_id=f"P2_{arch}_{mode}_{base}", family="P2", victim=arch, mode=mode,
                                 baseline=base, metric="lpips_vgg", estimand="LPIPS-vgg AUC delta", **vgg))
                p2.append(len(rows) - 1)
                for kind, metric, lab in (("lpips_alex_auc", "lpips_alex", "LPIPS-alex AUC delta"),
                                          ("dists_auc", "dists", "DISTS AUC delta")):
                    s = paired(arch, mode, base, kind)
                    if s:
                        rows.append(dict(claim_id=f"SUP_{metric}_{arch}_{mode}_{base}", family=None,
                                         victim=arch, mode=mode, baseline=base, metric=metric,
                                         estimand=lab, **s))
                s = paired(arch, mode, base, "asr_auc")
                if s:
                    rows.append(dict(claim_id=f"SUP_asr_{arch}_{mode}_{base}", family=None,
                                     victim=arch, mode=mode, baseline=base, metric="perceptual_asr",
                                     estimand="perceptual ASR AUC delta", **s))

    # ---- Holm within each family
    def apply_holm(indices: List[int]) -> None:
        ps = [rows[i]["p_one_sided"] for i in indices]
        phs = holm(ps)
        for i, ph in zip(indices, phs):
            rows[i]["p_holm"] = ph

    apply_holm(t1)
    apply_holm(p1)
    apply_holm(p2)

    # CI-only rows carry null p-values
    for r in rows:
        if r["family"] is None:
            r["p_one_sided"] = None
            r["p_holm"] = None

    # ---- decisions
    for r in rows:
        fam = r["family"]
        if fam == "T1":
            ok = (r["p_holm"] is not None and r["p_holm"] < 0.05 and r["ci_high"] < 1)
            r["decision"] = "confirmed" if ok else "not_confirmed"
        elif fam in ("P1", "P2"):
            ok = (r["p_holm"] is not None and r["p_holm"] < 0.05 and r["ci_high"] < 0)
            r["decision"] = "confirmed" if ok else "not_confirmed"
        elif r["estimand"] == "L2@10k ratio":
            thr = NI_T if r["mode"] == "targeted" else NI_NT
            r["decision"] = "confirmed" if r["ci_high"] <= thr else "not_confirmed"
        elif r["estimand"] == "L2 AUC ratio":
            r["decision"] = "confirmed" if r["ci_high"] <= NI_NT else "not_confirmed"
        elif r["estimand"] in ("LPIPS-alex AUC delta", "DISTS AUC delta"):
            r["decision"] = "confirmed" if r["ci_high"] < 0 else "not_confirmed"
        elif r["estimand"] == "perceptual ASR AUC delta":
            r["decision"] = "descriptive"
        else:
            r["decision"] = "not_confirmed"

    if a.no_devoverlap:
        for r in rows:
            r["decision"] = "descriptive"

    meta = {
        "script": "tools/eval/claims_registered.py",
        "script_sha256": sha256("tools/eval/claims_registered.py"),
        "prereg_sha256": sha256("prereg_build38.txt"),
        "amendment_sha256": sha256("prereg_build38_amendment2.txt"),
        "contract_sha256": sha256(a.contract),
        "decisions_sha256": sha256(a.decisions),
        "bootstrap_B": B,
        "bootstrap_seed": a.seed,
        "auc_budgets": AUC_BUDGETS,
        "n_main": N_MAIN,
        "families": {"T1": len(t1), "P1": len(p1), "P2": len(p2)},
        "excluded_cells": sorted([f"{b}_{arch}_{m}" for b, arch, m in EXCLUDED_CELLS]),
        "no_devoverlap": bool(a.no_devoverlap),
    }

    out = {"meta": meta, "rows": rows, "excluded_comparisons": excluded}
    os.makedirs(a.out, exist_ok=True)
    name = "claims_no_devoverlap.json" if a.no_devoverlap else "claims.json"
    with open(os.path.join(a.out, name), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"families (m): {meta['families']}")
    print(f"rows: {len(rows)}; excluded_comparisons: {len(excluded)}")
    for r in rows:
        p = r["p_one_sided"] if r["p_one_sided"] is None else round(r["p_one_sided"], 4)
        ph = r["p_holm"] if r["p_holm"] is None else round(r["p_holm"], 4)
        print(f"  {r['claim_id']:26s} {r['family'] or '-':4s} {r['baseline']:8s} {r['victim']:9s} {r['mode']:13s} "
              f"{r['estimand']:23s} est={r['estimate']:.4f} [{r['ci_low']:.4f},{r['ci_high']:.4f}] "
              f"p={p} holm={ph} n={r['n_pairs']} excl={r['n_excluded']} -> {r['decision']}")
    print(f"wrote {os.path.join(a.out, name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
