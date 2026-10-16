"""Build the reproducibility package for editors and reviewers (code + data provenance + results).

    python build_reviewer_package.py --out reviewer_package \
        --code advstore.py hardlabel_tracker.py eval_metrics_offline.py make_tables_figures.py make_comparison_grid.py \
               ../../run_benchmark_33b.py ../../lgc_bench ../../ttba_github \
        --results Results/b1_resnet50 Results/b1_vgg19 Results/b1_vit_b_32 \
        --assets paper_assets metrics_b1.csv --splits splits/imagenet/ttba_github \
        --store_images 20 --provenance provenance.csv --tar

What it does
  * copies code, split lists, per-checkpoint CSVs, run logs, env files, metric CSVs and paper assets;
  * copies the adversarial-image store for the first N images per (arch, mode, method) (--store_images; 0 = all);
  * writes MANIFEST.csv (relative path, bytes, sha256) for every file and a README_REVIEWERS.md skeleton;
  * optional --tar: reviewer_package.tar.gz + its sha256.
Large binaries (checkpoints, datasets) are NOT copied; list them with their sha256 in provenance.csv instead.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tarfile
from typing import List

SKIP_DIRS = {"__pycache__", ".git", "adv_store"}
SKIP_EXT = {".pth", ".pt", ".ckpt", ".tar", ".bin", ".safetensors"}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: str, dst: str) -> None:
    if os.path.isfile(src):
        if os.path.splitext(src)[1] not in SKIP_EXT:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
        return
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in SKIP_EXT:
                continue
            s = os.path.join(root, f)
            d = os.path.join(dst, os.path.relpath(s, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def copy_store_subset(results: List[str], out: str, n: int) -> int:
    """Copy adv_store folders, keeping at most n images per (store, arch) (all if n == 0)."""
    copied = 0
    for r in results:
        for root, dirs, _ in os.walk(r):
            if os.path.basename(root) != "adv_store":
                continue
            for arch in sorted(os.listdir(root)):
                ad = os.path.join(root, arch)
                if not os.path.isdir(ad):
                    continue
                imgs = sorted(os.listdir(ad))
                for im in imgs[: n or None]:
                    s = os.path.join(ad, im)
                    d = os.path.join(out, "results", os.path.relpath(s, os.path.dirname(os.path.abspath(r))))
                    shutil.copytree(s, d, dirs_exist_ok=True)
                    copied += 1
            mm = os.path.join(root, "x0_mismatch.csv")
            if os.path.exists(mm):
                shutil.copy2(mm, os.path.join(out, "results", "x0_mismatch_" + os.path.basename(r) + ".csv"))
            dirs[:] = []
    return copied


README = """# Reproducibility package (reviewers / editors)

## 1. What is inside
- `code/`: attack, adapters and evaluation code (exact versions; see MANIFEST.csv for sha256).
- `splits/`: image lists and target pairs (identical for every method).
- `results/`: per-checkpoint CSVs, run logs, env files, trace files, and the adversarial-image store
  (`adv_store/`, float16 perturbations + exact float32 L2) for the first {n_store} images per setting.
- `assets/`: metric CSVs, LaTeX tables and figures exactly as in the paper.
- `provenance.csv`: external artefacts NOT included (datasets, victim / autoencoder checkpoints, official baseline
  repositories) with source URL, version / commit and sha256.

## 2. Threat model and fairness (fill in)
- Hard-label (top-1 decision only); every victim evaluation is one query, including initialisation; budget [..].
- Identical images, target pairs, pre-processing and victim weights for every method; official baseline code with
  default hyper-parameters (table below); no method was tuned on the evaluation images.

| Method | Venue | Code source (commit) | Hyper-parameters |
|---|---|---|---|
| [..] | [..] | [..] | [..] |

## 3. Regenerate every table and figure (no GPU)
```bash
python code/eval_metrics_offline.py --roots results/... --out metrics.csv     # needs the full store (or use assets/)
python code/make_tables_figures.py --metrics assets/metrics_*.csv --ours [..] --outdir regenerated
```

## 4. Re-run a single attack (GPU)
[command per method, from the run scripts]

## 5. Integrity checks
- `MANIFEST.csv`: sha256 of every file.  `results/x0_mismatch_*.csv` must be absent (identical inputs).
- `l2_relerr` in the metric CSV compares recomputed and run-time L2 (< 1e-3 expected).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reviewer_package")
    ap.add_argument("--code", nargs="*", default=[])
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--assets", nargs="*", default=[])
    ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--provenance", default=None)
    ap.add_argument("--store_images", type=int, default=20)
    ap.add_argument("--tar", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for group, paths in (("code", a.code), ("splits", a.splits), ("assets", a.assets)):
        for p in paths:
            copy_tree(p, os.path.join(a.out, group, os.path.basename(os.path.normpath(p))))
    for r in a.results:  # everything except adv_store and model files
        copy_tree(r, os.path.join(a.out, "results", os.path.basename(os.path.normpath(r))))
    n_store = copy_store_subset(a.results, a.out, a.store_images)
    if a.provenance:
        shutil.copy2(a.provenance, os.path.join(a.out, "provenance.csv"))
    with open(os.path.join(a.out, "README_REVIEWERS.md"), "w") as fh:
        fh.write(README.format(n_store=a.store_images or "all"))
    rows = []
    for root, _, files in os.walk(a.out):
        for f in sorted(files):
            p = os.path.join(root, f)
            rows.append({"path": os.path.relpath(p, a.out), "bytes": os.path.getsize(p), "sha256": sha256(p)})
    with open(os.path.join(a.out, "MANIFEST.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    print(f"[+] {a.out}: {len(rows)} files, {n_store} stored image folders, MANIFEST.csv written")
    if a.tar:
        tgz = a.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tgz, "w:gz") as t:
            t.add(a.out, arcname=os.path.basename(a.out.rstrip("/")))
        with open(tgz + ".sha256", "w") as fh:
            fh.write(f"{sha256(tgz)}  {os.path.basename(tgz)}\n")
        print(f"[+] {tgz} (+ .sha256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
