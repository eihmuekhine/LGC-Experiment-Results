"""Visual comparison figure from stored adversarial examples (adv_store), fair across methods.

Rows = images, columns = clean image, then for every method the adversarial image (top) and its signed
perturbation (bottom).  The perturbation of ALL methods in a row uses ONE common amplification
(0.5 + amp * delta, amp chosen so the largest 99th-percentile |delta| among the shown methods maps to 0.45),
or a fixed --amp for the whole figure.  Per-image min-max normalisation is NOT used: it makes a tiny
perturbation look as strong as a large one (the error in the legacy _save_visual_milestone).

Image selection is rule-based and written into the caption file (no cherry-picking):
    --select random --seed 0     (default)  |  --select quantile  (25/50/75th percentile of ours' LPIPS)
    --images name1 name2 ...     (explicit; then say why in the paper)

    python make_comparison_grid.py --roots Results/b1_resnet50/targeted/adv_store Results/b1_resnet50_ports/targeted/adv_store \
        --arch resnet50 --budget 10000 --methods LGC_RC_EGC TtBA CGBA_H --ours LGC_RC_EGC \
        --metrics metrics_b1.csv --n 3 --out paper_assets/fig_examples_resnet50_targeted
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advstore import load_adv  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"


def index(roots: List[str], arch: str, budget: int) -> Dict[Tuple[str, str], str]:
    """(image, method) -> npz path for the given arch and budget."""
    out = {}
    for r in roots:
        for p in glob.glob(os.path.join(r, arch, "*", f"*_Q{budget}.npz")):
            image = os.path.basename(os.path.dirname(p))
            method = os.path.basename(p)[: -len(f"_Q{budget}.npz")]
            out[(image, method)] = p
    return out


def pick(images: List[str], how: str, n: int, seed: int, metrics, ours: str, budget: int, arch: str) -> List[str]:
    images = sorted(images)
    if how == "quantile" and metrics is not None:
        d = metrics[(metrics.method == ours) & (metrics.budget == budget) & (metrics.arch == arch)]
        d = d[d.image.isin(images)].sort_values("lpips_vgg" if "lpips_vgg" in d else "l2")
        if len(d):
            qs = np.linspace(0.25, 0.75, n) if n > 1 else [0.5]
            return [d.image.iloc[int(round(q * (len(d) - 1)))] for q in qs]
    rng = np.random.default_rng(seed)
    return list(rng.choice(images, size=min(n, len(images)), replace=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="adv_store folders (one per run dir)")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--rename", nargs="*", default=[])
    ap.add_argument("--metrics", default=None, help="metrics CSV (to print SSIM / LPIPS under each image)")
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--select", choices=["random", "quantile"], default="random")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=float, default=0.0, help="fixed amplification; 0 = common per row")
    ap.add_argument("--out", required=True, help="output path stem (writes .pdf, .png, _caption.txt)")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = dict(kv.split("=", 1) for kv in a.rename)
    idx = index(a.roots, a.arch, a.budget)
    common = sorted(set.intersection(*[{i for (i, m) in idx if m == meth} for meth in a.methods]))
    if not common:
        print("[!] no image has a stored state for every method at this budget")
        return 1
    met = None
    if a.metrics:
        import pandas as pd

        met = pd.read_csv(a.metrics)
        met["image"] = met["image"].astype(str)
    imgs = a.images or pick(common, a.select, a.n, a.seed, met, a.ours or a.methods[0], a.budget, a.arch)
    cols = 1 + len(a.methods)
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6})
    fig, axes = plt.subplots(2 * len(imgs), cols, figsize=(1.25 * cols, 2.55 * len(imgs)), squeeze=False)
    cap_lines = []
    for r, im in enumerate(imgs):
        data = {m: load_adv(idx[(im, m)]) for m in a.methods}
        x0 = next(iter(data.values()))[1]
        deltas = {m: data[m][0] - data[m][1] for m in a.methods}
        amp = a.amp or 0.45 / max(1e-8, max(float(np.percentile(np.abs(d), 99)) for d in deltas.values()))
        ax0, ax1 = axes[2 * r][0], axes[2 * r + 1][0]
        ax0.imshow(np.transpose(x0, (1, 2, 0)))
        ax0.set_title("clean" if r == 0 else "", color=INK)
        ax1.text(0.5, 0.5, f"amp x{amp:.0f}", ha="center", va="center", color=INK2, transform=ax1.transAxes)
        for c, m in enumerate(a.methods, start=1):
            xa, _, meta = data[m]
            d = deltas[m]
            l2 = float(np.linalg.norm(d.ravel()))
            lab = f"L2 {l2:.2f}"
            if met is not None:
                row = met[(met.method == m) & (met.image == im) & (met.budget == a.budget) & (met.arch == a.arch)]
                if len(row):
                    s = row.iloc[-1]
                    lab += f"\nSSIM {s.get('ssim', float('nan')):.4f}\nLPIPS {s.get('lpips_vgg', float('nan')):.4f}"
            axes[2 * r][c].imshow(np.transpose(xa, (1, 2, 0)))
            axes[2 * r][c].set_title(names.get(m, m) if r == 0 else "", color=INK, fontweight="bold" if m == a.ours else "normal")
            axes[2 * r][c].set_xlabel(lab, color=INK2, fontsize=6)
            axes[2 * r + 1][c].imshow(np.clip(0.5 + amp * np.transpose(d, (1, 2, 0)), 0, 1))
        for ax in axes[2 * r].tolist() + axes[2 * r + 1].tolist():
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        cap_lines.append(f"{im}: common amplification x{amp:.1f}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    with open(f"{a.out}_caption.txt", "w") as fh:
        fh.write(f"Images selected by rule '{'explicit' if a.images else a.select}' (seed {a.seed}) among {len(common)} "
                 f"images with a stored state for every method; budget {a.budget}; perturbations shown as 0.5 + amp*delta "
                 f"with one amplification per row.\n" + "\n".join(cap_lines) + "\n")
    print(f"[+] {a.out}.pdf/.png and caption")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
