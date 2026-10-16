"""Statistics and figures for benchmark outputs (paired, per-image design)."""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Fixed categorical order (validated palette, light mode). Reference method = slot 1.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

PLOT_GROUPS: Dict[str, List[str]] = {
    "Latent optimisers": ["LGC", "LGC_H", "LGC_TTBA", "LGC_TA", "LGC_NGTA", "LGC_HSJA"],
    "Whitened latent (proposed)": ["LGC", "LGC_W", "LGC_W_TTBA", "LGC_W_HSJA", "LGC_HSJA"],
    "Jacobian top-subspace (ablation)": ["LGC", "LGC_JS", "LGC_JSX", "LGC_JSM", "LGC_JSMP", "LGC_TA_J"],
    "Pixel-space baselines": ["LGC", "HSJA", "CGBA", "CGBA_H", "TtBA", "TA"],
    "Bridge geometry, latent (proposed)": ["LGC", "LGC_TTBA", "LGC_MB", "LGC_HYB", "LGC_GARC"],
    "Bridge geometry, pixel (proposed)": ["LGC", "TtBA", "TtBA_MB", "TtBA_HYB", "TtBA_GARC"],
    "Predict-verify-refine, latent (proposed)": ["LGC", "LGC_MB", "LGC_HYB", "LGC_PVR", "LGC_N60"],
    "Predict-verify-refine, pixel (proposed)": ["CGBA", "TtBA", "TtBA_MB", "TtBA_HYB", "TtBA_PVR", "TtBA_T",
                                                "TtBA_T60"],
    "Low-budget levers, pixel": ["CGBA", "TtBA", "TtBA_MB", "TtBA_MB60", "TtBA_PVR", "TtBA_PVR60", "TtBA_PVRN",
                                 "TtBA_T60"],
    "Final methods vs existing attacks": ["TtBA", "HSJA", "TA", "CGBA", "CGBA_H", "TtBA_PVRE", "LGC_PVRE",
                                          "LGC_PVRE_HA"],
    "Final methods vs current variants": ["TtBA", "TtBA_MB", "TtBA_PVR", "TtBA_T60", "TtBA_PVRE", "LGC",
                                          "LGC_PVRE", "LGC_PVRE_HA"],
    "Ablation ladder (pixel)": ["TtBA", "TtBA_MB", "TtBA_GARC", "TtBA_HYB", "TtBA_PVR", "TtBA_PVRE", "TtBA_PVRE_GC"],
    "PVRE geometry-controller": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_PVRE_UA", "TtBA_PVRE_GC",
                                 "TtBA_PVRE_AOR_SN", "TtBA_PVRE_AOR", "TtBA_PVRE_AOU", "TtBA_PVRE_CAOU"],
    "LGC switch rule": ["LGC_PVRE", "LGC_PVRE_H1000", "LGC_PVRE_H2500", "LGC_PVRE_HA", "TtBA_PVRE"],
    "RA-MB-PVRE prototype": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_RAMB_NOHIST", "TtBA_RAMB_PVRE"],
    "PVR-E v7 (Gate-Thales-Escape)": ["TtBA", "CGBA", "TtBA_PVRE", "TtBA_T", "TtBA_PVRE_GE", "TtBA_PVRE_GTE",
                                      "TtBA_PVRE_GTE_K"],
    "PVR-E v7 extras": ["TtBA", "TtBA_BL", "TtBA_PVRE_GTE", "TtBA_PVRE_GTEB", "TtBA_PVRE_CAOU", "TtBA_PVRE_CP"],
}

# steps whose new point is the bridge bisector (or the model optimum): there the angle
# gamma_obs at the new boundary point is exactly 90 deg on a hyperplane, so gamma_obs - 90
# measures boundary non-linearity. TA_HALF_RANGE_DEG is Triangle Attack's allowed |alpha - 90|.
CURVATURE_STEPS = (
    "midpoint", "garc", "hyb_bridge", "pvr_bridge", "pvr_ua_bridge",
    # 30g/30h used new step tags; omitting them made gamma appear NaN even though
    # the engine had logged verified boundary geometry.
    "pvr_gc_anchor_b", "pvr_gc_anchor_t", "pvr_gc_model",
    "pvr_aor_anchor_b", "pvr_aor_anchor_t", "pvr_aor_model",
    "pvr_aou_model", "pvr_caou_model",
)
TA_HALF_RANGE_DEG = float(np.degrees(0.1))


def _read_all(run_dir: str, pattern: str, keys: List[str]) -> pd.DataFrame:
    """Concatenate the single-process file and every shard file; keep the last copy of a row."""
    import glob

    paths = sorted(glob.glob(os.path.join(run_dir, pattern)))
    frames = [pd.read_csv(p) for p in paths if os.path.getsize(p) > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in keys if k in df.columns]
    return df.drop_duplicates(subset=keys, keep="last") if keys else df


def load_results(run_dir: str):
    ck = _read_all(run_dir, "per_checkpoint*.csv", ["arch", "image", "method", "budget"])
    if ck.empty:
        raise FileNotFoundError(f"no per_checkpoint*.csv in {run_dir}")
    runs = _read_all(run_dir, "runs*.csv", ["arch", "image", "method", "status"])
    if not runs.empty:  # an errored run that later succeeded: keep the success
        runs = runs.sort_values("status", key=lambda s: s.eq("error")).drop_duplicates(
            subset=[c for c in ("arch", "image", "method") if c in runs.columns], keep="first")
    for col, default in (("adv_uint8_repaired", np.nan), ("pix_l2_uint8_repaired", np.nan),
                         ("repair_queries", np.nan)):
        if col not in ck.columns:
            ck[col] = default
    # Primary success = verified adversarial float image (standard in HSJA/CGBA/TtBA/TA papers).
    ck["success"] = (ck["found"] == 1) & (ck["adv_float"] == 1)
    ck["success_uint8"] = (ck["found"] == 1) & (ck["adv_uint8"] == 1)
    ck["success_uint8_rep"] = (ck["found"] == 1) & (ck["adv_uint8_repaired"] == 1)
    return ck, runs


def _with_all(ck: pd.DataFrame) -> pd.DataFrame:
    if ck["arch"].nunique() <= 1:
        return ck
    both = ck.copy()
    both["image"] = both["arch"] + "/" + both["image"].astype(str)
    both["arch"] = "ALL"
    return pd.concat([ck, both], ignore_index=True)


def _boot_ci(x: np.ndarray, stat=np.median, n_boot: int = 2000, seed: int = 0):
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = stat(x[idx], axis=1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def summarize(ck: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _summarize(ck)


def _summarize(ck: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, method, budget), g in _with_all(ck).groupby(["arch", "method", "budget"]):
        s = g[g["success"]]
        l2 = s["pix_l2"].to_numpy(dtype=float)
        lo, hi = _boot_ci(l2)
        rows.append({
            "arch": arch, "method": method, "budget": int(budget), "n": len(g),
            "ASR": 100.0 * g["success"].mean(),
            "ASR_uint8_raw": 100.0 * g["success_uint8"].mean(),
            "ASR_uint8_repaired": 100.0 * g["success_uint8_rep"].mean(),
            "repair_q_mean": g["repair_queries"].mean(),
            "pixL2_uint8_rep_median": g.loc[g["success_uint8_rep"], "pix_l2_uint8_repaired"].median(),
            "pixL2_mean": np.mean(l2) if len(l2) else np.nan,
            "pixL2_median": np.median(l2) if len(l2) else np.nan,
            "pixL2_med_CI95": f"[{lo:.3f}, {hi:.3f}]",
            "latL2_median": s["lat_l2"].median(),
            "SSIM_mean": s["ssim"].mean(), "SSIM_median": s["ssim"].median(),
            "LPIPS_mean": s["lpips"].mean(), "LPIPS_median": s["lpips"].median(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["arch", "budget", "pixL2_median"])


def _holm(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def paired_vs_reference(ck: pd.DataFrame, reference: str = "LGC", metric: str = "pix_l2") -> pd.DataFrame:
    """Geo-mean ratio, wins and Wilcoxon signed-rank (Holm within arch x budget)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _paired(ck, reference, metric)


def _paired(ck: pd.DataFrame, reference: str, metric: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    data = _with_all(ck)
    out = []
    for (arch, budget), g in data.groupby(["arch", "budget"]):
        ref = g[(g["method"] == reference) & g["success"]].set_index("image")[metric]
        block = []
        for method, gm in g.groupby("method"):
            if method == reference:
                continue
            other = gm[gm["success"]].set_index("image")[metric]
            common = ref.index.intersection(other.index)
            if len(common) == 0:
                continue
            a = other.loc[common].to_numpy(float)
            b = ref.loc[common].to_numpy(float)
            ratio = float(np.exp(np.mean(np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12)))))
            try:
                p = float(wilcoxon(a, b).pvalue) if len(common) >= 6 and np.any(a != b) else np.nan
            except ValueError:
                p = np.nan
            block.append({
                "arch": arch, "budget": int(budget), "method": method, "vs": reference,
                "n_pairs": len(common), "geo_ratio": ratio, "wins": int((a < b).sum()),
                "median_rel_%": float(np.median((a - b) / b) * 100), "p_wilcoxon": p,
            })
        valid = [i for i, r in enumerate(block) if np.isfinite(r["p_wilcoxon"])]
        if valid:
            adj = _holm([block[i]["p_wilcoxon"] for i in valid])
            for i, a_p in zip(valid, adj):
                block[i]["p_holm"] = a_p
        out.extend(block)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["arch", "budget", "geo_ratio"])


def threshold_asr(ck: pd.DataFrame, rmse: Iterable[float] = (0.01, 0.005, 0.0025),
                  ssim: float = 0.99, lpips: float = 0.01) -> pd.DataFrame:
    data = _with_all(ck)
    rows = []
    for (arch, method, budget), g in data.groupby(["arch", "method", "budget"]):
        row = {"arch": arch, "method": method, "budget": int(budget), "n": len(g)}
        for t in rmse:
            row[f"ASR@RMSE<={t}"] = 100.0 * (g["success"] & (g["rmse"] <= t)).mean()
        row[f"ASR@SSIM>={ssim}"] = 100.0 * (g["success"] & (g["ssim"] >= ssim)).mean()
        row[f"ASR@LPIPS<={lpips}"] = 100.0 * (g["success"] & (g["lpips"] <= lpips)).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arch", "budget", "method"])


def query_breakdown(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    ok = runs[runs["status"] == "ok"].copy()
    cols = ["q_init", "q_boundary", "q_normal", "q_update"]
    for c in cols:
        ok[c + "_%"] = 100.0 * ok[c] / ok["total_queries"].clip(lower=1)
    extra = [c for c in ("mean_sign_share", "one_sided_pct", "mean_n_probes") if c in ok.columns]
    for c in extra:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    agg = ok.groupby("method")[[c + "_%" for c in cols] + ["iterations", "wall_time_s"] + extra].mean()
    return agg.sort_values("q_normal_%", ascending=False)


def plot_convergence(summary: pd.DataFrame, out_dir: str, reference: str = "LGC") -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def _fmt_q(v, _):
        return f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"

    paths = []
    for arch, sa in summary.groupby("arch"):
        groups = {k: [m for m in v if m in set(sa["method"])] for k, v in PLOT_GROUPS.items()}
        groups = {k: v for k, v in groups.items() if len(v) > 1}
        if not groups:
            continue
        ncol = min(2, len(groups)) if len(groups) > 3 else len(groups)
        nrow = -(-len(groups) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow), sharey=True, squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax in flat[len(groups):]:
            ax.set_visible(False)
        for ax, (title, methods) in zip(flat, groups.items()):
            ordered = [reference] + [m for m in methods if m != reference] if reference in methods else methods
            for slot, m in enumerate(ordered[: len(PALETTE)]):
                d = sa[sa["method"] == m].sort_values("budget")
                ax.plot(d["budget"], d["pixL2_median"], color=PALETTE[slot], marker=MARKERS[slot],
                        markersize=5, linewidth=1.6, label=m)
            ax.set_xscale("log")
            ax.set_yscale("log")
            budgets = sorted(sa["budget"].unique())
            ax.set_xticks(budgets)
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_q))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(axis="y", which="minor", labelsize=7)
            ax.set_title(title, fontsize=11, color="#222222")
            ax.set_xlabel("Query budget", color="#444444")
            ax.grid(True, which="both", color="#e6e6e6", linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.legend(fontsize=8, frameon=False)
        for row in axes:
            row[0].set_ylabel("Median pixel $\\ell_2$ (successful)", color="#444444")
        fig.suptitle(f"{arch}: median pixel L2 vs. queries", fontsize=12, color="#222222")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(out_dir, f"convergence_{arch}.{ext}")
            fig.savefig(p, dpi=200)
            paths.append(p)
        plt.close(fig)
    return paths


def load_geometry(run_dir: str) -> pd.DataFrame:
    return _read_all(run_dir, "geometry*.csv", ["arch", "image", "method", "it"])


def geometry_summary(geo: pd.DataFrame) -> pd.DataFrame:
    """Per arch x method: boundary non-linearity (gamma_obs), prediction errors and query split."""
    if geo.empty:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = geo.copy()
        if data["arch"].nunique() > 1:
            both = data.copy()
            both["arch"] = "ALL"
            data = pd.concat([data, both], ignore_index=True)
        rows = []
        for (arch, method), g in data.groupby(["arch", "method"]):
            cur = g[g["step"].isin(CURVATURE_STEPS)]
            dev = (cur["gamma_obs_deg"] - 90.0).to_numpy(float)
            dev = dev[np.isfinite(dev)]
            it_max = g.groupby("image")["it"].transform("max").clip(lower=1)
            phase = g["it"] / it_max
            early = cur[phase.loc[cur.index] <= 1 / 3]
            late = cur[phase.loc[cur.index] >= 2 / 3]
            rows.append({
                "arch": arch, "method": method, "images": g["image"].nunique(), "steps": len(g),
                "curv_steps": len(dev),
                "gamma_obs_median": 90.0 + float(np.median(dev)) if len(dev) else np.nan,
                "gamma_obs_IQR": float(np.subtract(*np.percentile(dev, [75, 25]))) if len(dev) else np.nan,
                "abs_dev_median": float(np.median(np.abs(dev))) if len(dev) else np.nan,
                "abs_dev_early": float(np.median(np.abs(early["gamma_obs_deg"] - 90))) if len(early) else np.nan,
                "abs_dev_late": float(np.median(np.abs(late["gamma_obs_deg"] - 90))) if len(late) else np.nan,
                "frac_obtuse_%": 100.0 * float(np.mean(dev > 0)) if len(dev) else np.nan,
                "frac_beyond_TA_%": 100.0 * float(np.mean(np.abs(dev) > TA_HALF_RANGE_DEG)) if len(dev) else np.nan,
                "abs_lin_err_median": float(np.nanmedian(np.abs(g["lin_err"]))) if g["lin_err"].notna().any() else np.nan,
                "abs_pred_err_median": float(np.nanmedian(np.abs(g["pred_err"]))) if g["pred_err"].notna().any() else np.nan,
                "k_bridge_median": float(np.nanmedian(g["k_bridge"])) if g["k_bridge"].notna().any() else np.nan,
                "model_step_%": 100.0 * float(g["step"].astype(str).str.endswith("_model").mean()),
                "fallback_%": 100.0 * float(g["step"].astype(str).str.endswith("fallback").mean()),
                "q_normal_mean": float(g["q_normal"].mean()),
                "q_bridge_mean": float(g["q_bridge"].mean()),
                "q_radial_mean": float(g["q_radial"].mean()),
                "sign_share_median": (float(np.nanmedian(g["sign_share"]))
                                      if "sign_share" in g and g["sign_share"].notna().any() else np.nan),
            })
    return pd.DataFrame(rows).sort_values(["arch", "method"])


def plot_geometry(geo: pd.DataFrame, out_dir: str) -> List[str]:
    """Histogram of gamma_obs - 90 deg on bisector steps (0 = locally linear boundary)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: List[str] = []
    cur = geo[geo["step"].isin(CURVATURE_STEPS)]
    for arch, ga in cur.groupby("arch"):
        methods = sorted(ga["method"].unique())
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        # range follows the data (targeted pixel attacks reach 80+ deg); values outside are clipped
        lim = float(np.clip(np.nanpercentile(np.abs(ga["gamma_obs_deg"] - 90.0), 99.5), 15.0, 90.0))
        bins = np.linspace(-lim, lim, 61)
        for slot, m in enumerate(methods[: len(PALETTE)]):
            dev = np.clip((ga.loc[ga["method"] == m, "gamma_obs_deg"] - 90.0).to_numpy(float), -lim, lim)
            ax.hist(dev, bins=bins, histtype="step", linewidth=1.6, color=PALETTE[slot],
                    label=f"{m} (n={len(dev)})", density=True)
        for x in (-TA_HALF_RANGE_DEG, TA_HALF_RANGE_DEG):
            ax.axvline(x, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("gamma_obs - 90 deg  (angle at the new boundary point; 0 = hyperplane)", color="#444444")
        ax.set_ylabel("density", color="#444444")
        ax.set_title(f"{arch}: boundary non-linearity (dashed = TA's +-5.7 deg)", fontsize=11, color="#222222")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(out_dir, f"geometry_gamma_{arch}.{ext}")
            fig.savefig(p, dpi=200)
            paths.append(p)
        plt.close(fig)
    return paths


def write_report(run_dir: str, reference: str = "LGC", rmse: Optional[Sequence[float]] = None) -> Dict[str, str]:
    ck, runs = load_results(run_dir)
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(ck)
    if summary.empty:
        return {}
    paired = paired_vs_reference(ck, reference=reference)
    thr = threshold_asr(ck, rmse=rmse or (0.01, 0.005, 0.0025))
    qb = query_breakdown(runs)
    files = {
        "summary": os.path.join(out_dir, "summary.csv"),
        "paired": os.path.join(out_dir, f"paired_vs_{reference}.csv"),
        "threshold_asr": os.path.join(out_dir, "threshold_asr.csv"),
        "query_breakdown": os.path.join(out_dir, "query_breakdown.csv"),
    }
    summary.to_csv(files["summary"], index=False)
    paired.to_csv(files["paired"], index=False)
    thr.to_csv(files["threshold_asr"], index=False)
    if not qb.empty:
        qb.to_csv(files["query_breakdown"])
    for p in plot_convergence(summary, out_dir, reference=reference):
        files[os.path.basename(p)] = p
    geo = load_geometry(run_dir)
    if not geo.empty:
        gs = geometry_summary(geo)
        files["geometry_summary"] = os.path.join(out_dir, "geometry_summary.csv")
        gs.to_csv(files["geometry_summary"], index=False)
        for p in plot_geometry(geo, out_dir):
            files[os.path.basename(p)] = p
    return files


def print_quick_table(run_dir: str, reference: str = "LGC") -> None:
    ck, runs = load_results(run_dir)
    summary = summarize(ck)
    if summary.empty:
        print("[analysis] no checkpoint rows yet")
        return
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    cols = ["arch", "method", "budget", "n", "ASR", "ASR_uint8_raw", "ASR_uint8_repaired", "pixL2_mean",
            "pixL2_median", "SSIM_mean", "LPIPS_mean"]
    for arch, s in summary.groupby("arch"):
        top = s[s["budget"] == s["budget"].max()][cols]
        print(f"\n=== {arch} @ budget {int(s['budget'].max())} (sorted by median pixel L2) ===")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    paired = paired_vs_reference(ck, reference=reference)
    if not paired.empty:
        last = paired[paired["budget"] == paired["budget"].max()]
        print(f"\n=== paired vs {reference} @ max budget (geo_ratio < 1 = better than {reference}) ===")
        print(last.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    qb = query_breakdown(runs)
    if not qb.empty:
        print("\n=== query breakdown (% of queries) ===")
        print(qb.to_string(float_format=lambda v: f"{v:.1f}"))
    geo = load_geometry(run_dir)
    if not geo.empty:
        gs = geometry_summary(geo)
        cols = ["arch", "method", "images", "curv_steps", "gamma_obs_median", "abs_dev_median", "abs_dev_early",
                "abs_dev_late", "frac_beyond_TA_%", "abs_lin_err_median", "model_step_%", "q_bridge_mean",
                "q_radial_mean"]
        print("\n=== boundary geometry (gamma_obs = 90 deg on a hyperplane; bisector steps only) ===")
        print(gs[[c for c in cols if c in gs.columns]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if not runs.empty and "status" in runs.columns:
        prog = runs.groupby(["arch", "method"])["status"].value_counts().unstack(fill_value=0)
        print("\n=== progress: finished runs per arch x method ===")
        print(prog.to_string())
