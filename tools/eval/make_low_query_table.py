"""E8 low-query table (paper element E8_low_query_table) from the per_checkpoint CSVs + offline metrics CSV.

For each (arch, mode) cell and each method, at each low-query budget {100, 200, 500, 1000} it reports:
  * found rate        : fraction of images with found = 1 at that budget (from per_checkpoint);
  * ASR at RMSE       : fraction of images with RMSE <= tau for tau in {0.1, 0.05, 0.01}, where
                         RMSE = pix_l2 / sqrt(3*224*224).  An image with found = 0 or adv_flag = 0 counts as a
                         failure (L2 = inf, contract D6);
  * median pix_l2     : over found rows only (from per_checkpoint);
  * median LPIPS-vgg  : over found rows (from the metrics CSV, one implementation for all methods).

Direction-aware bolding: the best value per column is bolded (highest found rate / ASR, lowest L2 / LPIPS).  n (the
number of images per method) is printed.  Ours is always listed first.

    python make_low_query_table.py --roots $(cat reports/B4e2_smoke_roots.txt) \
        --metrics generated/smoke_canary/metrics.csv --ours LGC_RC_EGC \
        --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --budgets 100 200 500 1000 --rmse 0.1 0.05 0.01 \
        --outdir generated/smoke_canary
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SQRT = float(np.sqrt(3 * 224 * 224))          # RMSE normaliser sqrt(3*224*224)
CKPT_COLS = ["arch", "mode", "image", "method", "budget", "found", "pix_l2"]
METRIC_COLS = ["arch", "mode", "image", "method", "budget", "adv_flag", "lpips_vgg"]

SUB = [("found_rate", True), ("asr_1", True), ("asr_2", True), ("asr_3", True), ("med_l2", False), ("med_lpips", False)]
SUB_LABEL = {"found_rate": "found", "asr_1": "ASR$_{1}$", "asr_2": "ASR$_{2}$", "asr_3": "ASR$_{3}$",
             "med_l2": "L$_2$", "med_lpips": "LPIPS"}


def stem(name) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_per_checkpoint(roots: List[str]) -> pd.DataFrame:
    """Concatenate every ``<root>/*/per_checkpoint*.csv`` into one long frame with normalised columns."""
    frames = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d):
                continue
            for f in sorted(glob.glob(os.path.join(d, "per_checkpoint*.csv"))):
                df = pd.read_csv(f)
                if not {"arch", "mode", "image", "method", "budget", "found", "pix_l2"} <= set(df.columns):
                    continue
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=CKPT_COLS)
    ck = pd.concat(frames, ignore_index=True)
    ck = ck[CKPT_COLS].copy()
    ck["image"] = ck["image"].map(stem)
    ck["budget"] = pd.to_numeric(ck["budget"], errors="coerce")
    ck["found"] = pd.to_numeric(ck["found"], errors="coerce")
    ck["pix_l2"] = pd.to_numeric(ck["pix_l2"], errors="coerce")
    ck["method"] = ck["method"].astype(str)
    ck["arch"] = ck["arch"].astype(str)
    ck["mode"] = ck["mode"].astype(str)
    return ck.drop_duplicates(subset=["arch", "mode", "image", "method", "budget"], keep="last")


def load_metrics(paths: List[str]) -> pd.DataFrame:
    """Concatenate metrics CSVs (eval_metrics_offline.py long format) and keep the needed columns."""
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if not {"arch", "mode", "image", "method", "budget"} <= set(df.columns):
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=METRIC_COLS)
    m = pd.concat(frames, ignore_index=True)
    keep = ["arch", "mode", "image", "method", "budget"]
    for c in ("adv_flag", "lpips_vgg"):
        if c in m.columns:
            keep.append(c)
        else:
            m[c] = np.nan
    m = m[keep].copy()
    m["image"] = m["image"].map(stem)
    m["budget"] = pd.to_numeric(m["budget"], errors="coerce")
    m["method"] = m["method"].astype(str)
    m["arch"] = m["arch"].astype(str)
    m["mode"] = m["mode"].astype(str)
    if "adv_flag" in m.columns:
        m["adv_flag"] = pd.to_numeric(m["adv_flag"], errors="coerce")
    if "lpips_vgg" in m.columns:
        m["lpips_vgg"] = pd.to_numeric(m["lpips_vgg"], errors="coerce")
    return m.drop_duplicates(subset=["arch", "mode", "image", "method", "budget"], keep="last")


def compute(ckpt: pd.DataFrame, metrics: pd.DataFrame, budgets: List[int],
            rmse_thresholds: List[float]) -> pd.DataFrame:
    """Per (arch, mode, method, budget) -> found_rate, asr_<i>, med_l2, med_lpips, n.

    ``ckpt`` / ``metrics`` must already have the normalised columns of ``load_per_checkpoint`` /
    ``load_metrics`` (image names stemmed)."""
    budgets = sorted(int(b) for b in budgets)
    rows = []
    for (arch, mode), c in ckpt.groupby(["arch", "mode"]):
        met = metrics[(metrics["arch"] == arch) & (metrics["mode"] == mode)]
        for method, g in c.groupby("method"):
            n = int(g["image"].nunique())
            for b in budgets:
                gb = g[g["budget"] == b]
                if gb.empty:
                    continue
                mm = met[(met.method == method) & (met.budget == b)][["image", "adv_flag", "lpips_vgg"]]
                j = gb.merge(mm, on="image", how="left")
                found = j["found"].fillna(0).astype(int) == 1
                found_n = int(found.sum())
                found_rate = 100.0 * found_n / max(1, len(j))
                adv_ok = j["adv_flag"].isna() | (j["adv_flag"] == 1)
                rmse = j["pix_l2"] / SQRT
                rec = {"arch": arch, "mode": mode, "method": method, "budget": b,
                       "n": n, "found_rate": found_rate,
                       "med_l2": float(j.loc[found, "pix_l2"].median()) if found_n else float("nan"),
                       "med_lpips": float(j.loc[found, "lpips_vgg"].median()) if found_n else float("nan")}
                for i, tau in enumerate(rmse_thresholds, 1):
                    ok = found & adv_ok & (rmse <= tau)
                    rec[f"asr_{i}"] = 100.0 * int(ok.sum()) / max(1, len(j))
                rows.append(rec)
    return pd.DataFrame(rows)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(x, dec: int, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{dec}f}" + ("\\%" if pct else "")


def render(df: pd.DataFrame, methods: List[str], names: Dict[str, str], budgets: List[int],
           rmse_thresholds: List[float], arch: str, mode: str) -> str:
    sub = [("found_rate", True)] + [(f"asr_{i}", True) for i in range(1, len(rmse_thresholds) + 1)] + \
          [("med_l2", False), ("med_lpips", False)]
    nsub = len(sub)
    n_per_method = {m: int(df[df.method == m]["n"].max()) if not df[df.method == m].empty else 0 for m in methods}
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize",
             rf"\caption{{E8 low-query regime for {tex_escape(arch)}, {tex_escape(mode)}: found rate (\%), "
             rf"ASR (\%) at RMSE $\le$ {', '.join(str(t) for t in rmse_thresholds)}, median pixel $\ell_2$ and "
             rf"median LPIPS-vgg.  n = {', '.join(f'{names.get(m, m)}:{n_per_method[m]}' for m in methods)}; "
             rf"RMSE = $\ell_2/\sqrt{{3\cdot 224 \cdot 224}}$.}}",
             r"\begin{tabular}{l" + "c" * (nsub * len(budgets)) + "}", r"\toprule"]
    header = ["Method"]
    for b in budgets:
        header += [rf"\multicolumn{{{nsub}}}{{c}}{{{b}}}"]
    lines.append(" & ".join(header) + r" \\")
    cmids = " ".join(rf"\cmidrule(lr){{{2 + k * nsub}-{1 + (k + 1) * nsub}}}" for k in range(len(budgets)))
    lines.append(cmids)
    subcells = [SUB_LABEL[c] for c, _ in sub]
    lines.append(" & " + " & ".join(subcells * len(budgets)) + r" \\")
    lines.append(r"\midrule")
    best = {}
    for c, higher in sub:
        for b in budgets:
            col = df[df.budget == b]
            vals = {m: col[col.method == m][c].iloc[0] if not col[col.method == m].empty else None for m in methods}
            vv = {m: x for m, x in vals.items() if x is not None and (not isinstance(x, float) or np.isfinite(x))}
            best[(c, b)] = (max if higher else min)(vv, key=vv.get) if vv else None
    for m in methods:
        cells = []
        for b in budgets:
            row = df[(df.method == m) & (df.budget == b)]
            if row.empty:
                cells += ["--"] * nsub
                continue
            r = row.iloc[0]
            for c, _ in sub:
                pct = c != "med_l2" and c != "med_lpips"
                dec = 1 if pct else (2 if c == "med_l2" else 4)
                s = fmt(r.get(c), dec, pct=pct)
                cells.append(rf"\textbf{{{s}}}" if best[(c, b)] == m and s != "--" else s)
        lines.append(tex_escape(names.get(m, m)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Found rate and ASR are percentages over the n images; "
              r"an image with found = 0 or adv\_flag = 0 counts as a failure ($\ell_2=\infty$, contract D6). "
              r"ASR$_{1}$/ASR$_{2}$/ASR$_{3}$ = ASR at RMSE $\le$ " + ", ".join(str(t) for t in rmse_thresholds) +
              r".  Median $\ell_2$ and LPIPS-vgg over found rows only.}",
              r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="result folders containing <mode>/per_checkpoint*.csv")
    ap.add_argument("--metrics", nargs="+", required=True, help="metrics CSV(s) from eval_metrics_offline.py")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--order", nargs="*", default=None, help="method order (ours first is recommended)")
    ap.add_argument("--rename", nargs="*", default=[], help="internal=Display pairs")
    ap.add_argument("--budgets", nargs="+", type=int, default=[100, 200, 500, 1000])
    ap.add_argument("--rmse", nargs="+", type=float, default=[0.1, 0.05, 0.01])
    ap.add_argument("--outdir", default="generated/smoke_canary")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ckpt = load_per_checkpoint(a.roots)
    metrics = load_metrics(a.metrics)
    df = compute(ckpt, metrics, a.budgets, a.rmse)
    names = dict(kv.split("=", 1) for kv in a.rename)
    for (arch, mode), d in df.groupby(["arch", "mode"]):
        methods = [m for m in (a.order or sorted(d.method.unique())) if m in set(d.method)]
        if a.ours in methods:
            methods = [a.ours] + [m for m in methods if m != a.ours]
        tag = f"{arch}_{mode}"
        nimg = {m: int(d[d.method == m]["n"].max()) for m in methods}
        print(f"[*] {tag}: images per method {nimg}")
        tex = render(d, methods, names, a.budgets, a.rmse, arch, mode)
        with open(os.path.join(a.outdir, f"tab_lowq_{tag}.tex"), "w", encoding="utf-8") as fh:
            fh.write(tex)
    print(f"[+] low-query tables in {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
