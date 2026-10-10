"""Paper tables (LaTeX, booktabs) and figures (PDF + PNG) from the offline metrics CSV.

Input: one or more CSVs written by eval_metrics_offline.py (long format: arch, mode, image, method, budget, l2,
ssim, ms_ssim, lpips_alex, lpips_vgg, dists, fsim, ...).  Every number in the outputs is computed here from that CSV,
so a reviewer can regenerate all tables and figures with one command.

    python make_tables_figures.py --metrics metrics_b1.csv --ours LGC_RC_EGC \
        --order LGC_RC_EGC TtBA CGBA CGBA_H HSJA SignOPT ARS_OPT \
        --rename LGC_RC_EGC="LGC-EGC (ours)" CGBA_H=CGBA-H SignOPT=Sign-OPT ARS_OPT=ARS-OPT \
        --outdir paper_assets

Outputs per (arch, mode): tab_l2_*, tab_percep_*, tab_paired_*, tab_asrperc_* (.tex), fig_curves_*, fig_pareto_*
(.pdf/.png) and summary_*.json.  Pairing is always on the images that BOTH methods attacked (n is printed).
Statistics: paired geometric-mean ratio (L2) or mean difference (other metrics), percentile bootstrap 95% CI
(B resamples of images), two-sided Wilcoxon signed-rank test, Holm correction across baselines, wins/losses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# metric -> (label for tables, higher_is_better, decimals)
METRICS: Dict[str, Tuple[str, bool, int]] = {
    "l2": (r"$\ell_2$", False, 2), "linf": (r"$\ell_\infty$", False, 3), "psnr": ("PSNR", True, 2),
    "ssim": ("SSIM", True, 4), "ms_ssim": ("MS-SSIM", True, 4), "fsim": ("FSIM", True, 4),
    "lpips_alex": ("LPIPS$_{\\mathrm{alex}}$", False, 4), "lpips_vgg": ("LPIPS$_{\\mathrm{vgg}}$", False, 4),
    "dists": ("DISTS", False, 4),
}
PERCEPTUAL = ["ssim", "ms_ssim", "fsim", "lpips_alex", "lpips_vgg", "dists"]
PAIRED = ["l2", "lpips_alex", "lpips_vgg", "dists", "ssim"]
# validated categorical palette (dataviz reference, light mode, fixed order; ours always slot 1)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd7"
ARROW = {True: r"$\uparrow$", False: r"$\downarrow$"}
PLAIN = {"l2": "L2", "lpips_alex": "LPIPS-alex", "lpips_vgg": "LPIPS-vgg", "dists": "DISTS", "ssim": "SSIM"}


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def holm(p: List[float]) -> List[float]:
    m, order = len(p), np.argsort(p)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    d = d[np.isfinite(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return float("nan")
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def bootstrap_ci(d: np.ndarray, stat, b: int, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(d)
    idx = rng.integers(0, n, size=(b, n))
    vals = stat(d[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired(ours: pd.DataFrame, base: pd.DataFrame, metric: str, b: int, rng) -> Optional[dict]:
    """Paired comparison on common images.  L2: ratio ours/base (geo-mean); others: mean difference ours-base."""
    j = ours[["image", metric]].merge(base[["image", metric]], on="image", suffixes=("_o", "_b")).dropna()
    if metric == "l2":
        j = j[(j[f"{metric}_o"] > 0) & (j[f"{metric}_b"] > 0)]
    if len(j) < 2:
        return None
    o, bb = j[f"{metric}_o"].to_numpy(float), j[f"{metric}_b"].to_numpy(float)
    hib = METRICS[metric][1]
    if metric == "l2":
        d = np.log(o) - np.log(bb)
        est = float(np.exp(d.mean()))
        lo, hi = bootstrap_ci(d, lambda x: np.exp(x.mean(axis=1)), b, rng)
    else:
        d = o - bb
        est = float(d.mean())
        lo, hi = bootstrap_ci(d, lambda x: x.mean(axis=1), b, rng)
    wins = int(np.sum(o > bb)) if hib else int(np.sum(o < bb))
    losses = int(np.sum(o < bb)) if hib else int(np.sum(o > bb))
    return {"n": int(len(j)), "est": est, "lo": lo, "hi": hi, "p": wilcoxon_p(d), "wins": wins, "losses": losses}


def fmt(x: float, dec: int) -> str:
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{dec}f}"


def bold_best(vals: Dict[str, float], higher: bool) -> Optional[str]:
    v = {k: x for k, x in vals.items() if x is not None and math.isfinite(x)}
    if not v:
        return None
    return (max if higher else min)(v, key=v.get)


def write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def table_l2(df: pd.DataFrame, methods: List[str], names: Dict[str, str], budgets: List[int], cap: str) -> str:
    g = df.groupby(["method", "budget"])["l2"]
    mean, med = g.mean(), g.median()
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", rf"\caption{{{cap}}}",
             r"\begin{tabular}{l" + "c" * len(budgets) + "}", r"\toprule",
             "Method & " + " & ".join(f"{b // 1000}k" if b % 1000 == 0 else str(b) for b in budgets) + r" \\",
             r"\midrule"]
    best = {b: bold_best({m: mean.get((m, b), float("nan")) for m in methods}, False) for b in budgets}
    for m in methods:
        cells = []
        for b in budgets:
            c = f"{fmt(mean.get((m, b), float('nan')), 2)} ({fmt(med.get((m, b), float('nan')), 2)})"
            cells.append(rf"\textbf{{{c}}}" if best[b] == m else c)
        lines.append(tex_escape(names.get(m, m)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_percep(df: pd.DataFrame, methods: List[str], names: Dict[str, str], budget: int, cap: str) -> str:
    cols = [c for c in PERCEPTUAL if c in df.columns and df[c].notna().any()]
    d = df[df.budget == budget].groupby("method")[cols].mean()
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", rf"\caption{{{cap}}}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Method & " + " & ".join(METRICS[c][0] + " " + ARROW[METRICS[c][1]] for c in cols) + r" \\",
             r"\midrule"]
    best = {c: bold_best({m: d[c].get(m, float("nan")) for m in methods}, METRICS[c][1]) for c in cols}
    for m in methods:
        cells = []
        for c in cols:
            s = fmt(d[c].get(m, float("nan")), METRICS[c][2])
            cells.append(rf"\textbf{{{s}}}" if best[c] == m else s)
        lines.append(tex_escape(names.get(m, m)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_paired(res: Dict[str, Dict[str, dict]], baselines: List[str], names: Dict[str, str], ours: str,
                 cap: str) -> str:
    mets = [m for m in PAIRED if any(res.get(b, {}).get(m) for b in baselines)]
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", rf"\caption{{{cap}}}",
             r"\begin{tabular}{l" + "c" * len(mets) + "}", r"\toprule",
             f"{tex_escape(names.get(ours, ours))} vs. & " + " & ".join(
                 ("$\\ell_2$ ratio" if m == "l2" else f"$\\Delta$ {METRICS[m][0]}") for m in mets) + r" \\",
             r"\midrule"]
    for b in baselines:
        cells = []
        for m in mets:
            r = res.get(b, {}).get(m)
            if not r:
                cells.append("--")
                continue
            dec = 3 if m == "l2" else METRICS[m][2]
            star = "$^{*}$" if r.get("p_holm", 1.0) < 0.05 else ""
            cells.append(f"{r['est']:.{dec}f}{star} [{r['lo']:.{dec}f}, {r['hi']:.{dec}f}] ({r['wins']}/{r['losses']})")
        lines.append(tex_escape(names.get(b, b)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\\[2pt]\parbox{\linewidth}{\scriptsize Paired over common images; 95\% bootstrap CI; "
              r"$^{*}$ Holm-corrected two-sided Wilcoxon $p<0.05$; (wins/losses) for ours.}", r"\end{table*}", ""]
    return "\n".join(lines)


def asr_perc(df: pd.DataFrame, methods: List[str], budgets: List[int], tl: float, ts: float, lp: str) -> pd.DataFrame:
    common = set.intersection(*[set(df[df.method == m].image) for m in methods]) if methods else set()
    out = {}
    for m in methods:
        dm = df[(df.method == m) & df.image.isin(common)]
        out[m] = {}
        for b in budgets:
            db = dm[dm.budget == b]
            ok = (db[lp] <= tl) & (db["ssim"] >= ts)
            out[m][b] = 100.0 * float(ok.sum()) / max(1, len(common))
    return pd.DataFrame(out).T


def _logfmt(ax, x: bool = True, y: bool = True) -> None:
    """Readable log axes: ticks at 1-2-5 x 10^k labelled as plain numbers, unlabelled minor ticks."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on:
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


def figures(df: pd.DataFrame, methods: List[str], names: Dict[str, str], ours: str, final: int,
            asr: pd.DataFrame, lp: str, stem: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    sty = {m: (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]) for i, m in enumerate(methods)}
    panels = [("l2", "median L2 (IQR band)"), (lp, f"median {PLAIN.get(lp, lp)} (IQR band)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for ax, (col, lab) in zip(axes[:2], panels):
        for m in methods:
            g = df[df.method == m].groupby("budget")[col]
            q = g.quantile([0.25, 0.5, 0.75]).unstack()
            if q.empty:
                continue
            c, mk = sty[m]
            lw = 2.0 if m == ours else 1.2
            ax.plot(q.index, q[0.5], color=c, marker=mk, ms=4, lw=lw, label=names.get(m, m))
            ax.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _logfmt(ax)
        ax.set_xlabel("queries")
        ax.set_ylabel(lab)
        ax.grid(True, color=GRID, lw=0.5)
    ax = axes[2]
    for m in methods:
        if m in asr.index:
            c, mk = sty[m]
            ax.plot(asr.columns, asr.loc[m], color=c, marker=mk, ms=4, lw=2.0 if m == ours else 1.2)
    ax.set_xscale("log")
    _logfmt(ax, y=False)
    ax.set_xlabel("queries")
    ax.set_ylabel("perceptual ASR (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, lw=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(len(methods), 4), frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem.replace('XX', 'curves')}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    base = [m for m in methods if m != ours]
    if not base:
        return
    ncol = min(3, len(base))
    nrow = int(math.ceil(len(base) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.2 * nrow), squeeze=False)
    fin = df[df.budget == final]
    o = fin[fin.method == ours][["image", "l2", lp]]
    for k, b in enumerate(base):
        ax = axes[k // ncol][k % ncol]
        j = o.merge(fin[fin.method == b][["image", "l2", lp]], on="image", suffixes=("_o", "_b")).dropna()
        for side, col, mk, lab in (("_b", PALETTE[1], MARKERS[1], names.get(b, b)),
                                   ("_o", PALETTE[0], MARKERS[0], names.get(ours, ours))):
            ax.scatter(j["l2" + side], j[lp + side], s=9, color=col, marker=mk, alpha=0.55, lw=0, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _logfmt(ax)
        ax.set_xlabel(f"L2 at {final // 1000}k queries")
        ax.set_ylabel(PLAIN.get(lp, lp))
        ax.set_title(f"vs. {names.get(b, b)} (n={len(j)})", fontsize=8, color=INK)
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    for k in range(len(base), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem.replace('XX', 'pareto')}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--order", nargs="*", default=None, help="method order (ours first is recommended)")
    ap.add_argument("--rename", nargs="*", default=[], help="internal=Display pairs")
    ap.add_argument("--budgets", nargs="+", type=int, default=[2000, 5000, 10000])
    ap.add_argument("--final", type=int, default=10000)
    ap.add_argument("--asr_lpips", default="lpips_vgg", help="LPIPS column used by the perceptual ASR")
    ap.add_argument("--thr", nargs="*", default=["non_targeted:0.05:0.99", "targeted:0.3:0.75"],
                    help="mode:tau_lpips:tau_ssim (pre-registered)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="paper_assets")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in a.metrics], ignore_index=True)
    df["budget"] = df["budget"].astype(int)
    df = df.drop_duplicates(subset=["arch", "mode", "image", "method", "budget"], keep="last")
    names = dict(kv.split("=", 1) for kv in a.rename)
    thr = {t.split(":")[0]: (float(t.split(":")[1]), float(t.split(":")[2])) for t in a.thr}
    rng = np.random.default_rng(a.seed)
    for (arch, mode), d in df.groupby(["arch", "mode"]):
        methods = [m for m in (a.order or sorted(d.method.unique())) if m in set(d.method)]
        if a.ours in methods:
            methods = [a.ours] + [m for m in methods if m != a.ours]
        tag = f"{arch}_{mode}"
        nimg = {m: int(d[d.method == m].image.nunique()) for m in methods}
        print(f"[*] {tag}: images per method {nimg}")
        tb = [b for b in a.budgets if b in set(d.budget)]
        write(os.path.join(a.outdir, f"tab_l2_{tag}.tex"), table_l2(
            d, methods, names, tb, f"Mean (median) pixel $\\ell_2$ vs. query budget, {tex_escape(arch)}, {tex_escape(mode)}."))
        write(os.path.join(a.outdir, f"tab_percep_{tag}.tex"), table_percep(
            d, methods, names, a.final, f"Perceptual metrics at {a.final} queries (mean), {tex_escape(arch)}, {tex_escape(mode)}."))
        res: Dict[str, Dict[str, dict]] = {}
        fin = d[d.budget == a.final]
        o = fin[fin.method == a.ours]
        baselines = [m for m in methods if m != a.ours]
        for m in PAIRED:
            if m not in d.columns:
                continue
            ps, keys = [], []
            for b in baselines:
                r = paired(o, fin[fin.method == b], m, a.boot, rng)
                if r:
                    res.setdefault(b, {})[m] = r
                    ps.append(r["p"] if math.isfinite(r["p"]) else 1.0)
                    keys.append(b)
            for b, ph in zip(keys, holm(ps) if ps else []):
                res[b][m]["p_holm"] = ph
        write(os.path.join(a.outdir, f"tab_paired_{tag}.tex"), table_paired(
            res, baselines, names, a.ours, f"Paired comparison at {a.final} queries, {tex_escape(arch)}, {tex_escape(mode)}."))
        tl, ts = thr.get(mode, (0.05, 0.99))
        asr = asr_perc(d, methods, sorted(set(d.budget)), tl, ts, a.asr_lpips)
        lines = [r"\begin{table}[t]", r"\centering\footnotesize",
                 rf"\caption{{Perceptual ASR (\%): adversarial with {METRICS[a.asr_lpips][0]} $\le {tl}$ and SSIM "
                 rf"$\ge {ts}$, {tex_escape(arch)}, {tex_escape(mode)}.}}",
                 r"\begin{tabular}{l" + "c" * len(asr.columns) + "}", r"\toprule",
                 "Method & " + " & ".join(str(c) for c in asr.columns) + r" \\", r"\midrule"]
        for m in asr.index:
            lines.append(tex_escape(names.get(m, m)) + " & " + " & ".join(f"{v:.1f}" for v in asr.loc[m]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        write(os.path.join(a.outdir, f"tab_asrperc_{tag}.tex"), "\n".join(lines))
        figures(d, methods, names, a.ours, a.final, asr, a.asr_lpips, os.path.join(a.outdir, f"fig_XX_{tag}"))
        summary = {"arch": arch, "mode": mode, "images_per_method": nimg, "paired": res,
                   "asr_perc": asr.to_dict(orient="index"), "thresholds": {"lpips": tl, "ssim": ts},
                   "means": d.groupby(["method", "budget"]).mean(numeric_only=True).reset_index()
                   .to_dict(orient="records")}
        write(os.path.join(a.outdir, f"summary_{tag}.json"), json.dumps(summary, indent=1, default=float))
    print(f"[+] tables and figures in {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
