#!/usr/bin/env python3
"""E9 ablation cells (prereg_build38_amendment3.txt): queue order and the per-cell completeness gate. L, 2026-10-23.

CPU only. The gate never reads a result value beyond what completeness needs (row counts, status, method,
image set, per-image seed); no L2 / LPIPS value is printed.

  python tools/eval/e9_gate.py --print_order            cell<TAB>outdir, longest expected first (LPT)
  python tools/eval/e9_gate.py --print_errors --root D  error rows of every runs*.csv under D (exit 1 if any)
  python tools/eval/e9_gate.py CELL                     gate one finished cell (exit 0 PASS, 1 FAIL)
  python tools/eval/e9_gate.py --all                    gate all 18 cells (exit 0 only if all PASS)
  python tools/eval/e9_gate.py --check_inputs           inputs before launch (exit 0 all present, 1 otherwise):
      the autoencoder (sha256 1d004c634f1b...), the three victim files (sha256 prefix = file-name suffix), the
      labels file, the six split CSVs (sha256 = contract constants.split_sha256), every source / target image of
      rows 1-200 under <data_dir>/images, the six B5 EGC cells and metrics files of paper_exports/b5a2

Gate (every item must hold):
  G1 runs*.csv: exactly n_main (200) rows, no status "error", one row per image;
  G2 per_checkpoint*.csv: method column == the cell method only; every image has every registered checkpoint;
  G3 the image set equals that of the B5 LGC_RC_EGC cell of the same victim and mode (paper_exports/b5a2/cells),
     and the per-image seed column equals the B5 seed of that image (same row -> same RNG stream);
  G4 config_single.json: methods == [METHOD], num_samples 200, max_queries 10000, seed 42, the B5 checkpoints and
     exclude_seeds, save_adv true, torch 2.7.1+cu128.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

METHODS = ["LGC_RC_EGCI", "LGC_RC", "LGC_RC_HT1000"]
ARCHS = ["vgg19", "resnet50", "vit_b_32"]
MODES = ["non_targeted", "targeted"]
SHORT = {"non_targeted": "NT", "targeted": "T"}
N_MAIN = 200
CHECKPOINTS = [100, 200, 500, 1000, 2000, 2500, 5000, 10000]
EXCLUDE = [992, 2027, 3141, 2718]
TORCH = "2.7.1+cu128"
# expected seconds per image on one RTX 5090, one process per GPU: the B5 LGC_RC_EGC medians (E13 cost table) times
# a method factor (LGC_RC and EGCI as EGC; HT1000 runs pixel TtBA after 1000 queries: 0.6). Only for the queue order.
EGC_S = {("vgg19", "targeted"): 28.5, ("vgg19", "non_targeted"): 25.2, ("resnet50", "targeted"): 24.0,
         ("vit_b_32", "targeted"): 23.7, ("resnet50", "non_targeted"): 21.3, ("vit_b_32", "non_targeted"): 20.7}
FACTOR = {"LGC_RC_EGCI": 1.05, "LGC_RC": 1.0, "LGC_RC_HT1000": 0.6}
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
LABELS = "ttba_mini_val_labels.txt"
VICTIM_DIR = "/root/autodl-tmp/TtBA/model"
VICTIM_FILES = ["vgg19-dcbb9e9d.pth", "resnet50-11ad3fa6.pth", "vit_b_32-d86f8d99.pth"]
AE_PATH = "checkpoints/vgg16_imagenet.pth"
AE_PREFIX = "1d004c634f1b"


def cells() -> List[Tuple[str, str, str, str]]:
    out = [(f"e9_{m}_{a}_{SHORT[o]}", m, a, o) for m in METHODS for a in ARCHS for o in MODES]
    return sorted(out, key=lambda c: -EGC_S[(c[2], c[3])] * FACTOR[c[1]])


CELL = {c[0]: c for c in cells()}


def outdir(name: str, results_root: str) -> str:
    return os.path.join(results_root, name)


def _rows(paths: List[str]) -> List[dict]:
    out = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            out += list(csv.DictReader(fh))
    return out


def error_rows(root: str) -> List[Tuple[str, dict]]:
    out = []
    for dp, _d, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("runs") and fn.endswith(".csv"):
                p = os.path.join(dp, fn)
                out += [(p, r) for r in _rows([p]) if r.get("status") == "error"]
    return out


def stem(s: str) -> str:
    return s.replace(".JPEG", "")


def gate(name: str, results_root: str, b5_cells: str) -> List[str]:
    """Return the list of failures (empty = PASS)."""
    _n, method, arch, mode = CELL[name]
    root = os.path.join(outdir(name, results_root), mode)
    fails: List[str] = []
    runs = _rows(sorted(glob.glob(os.path.join(root, "runs*.csv"))))
    pcs = _rows(sorted(glob.glob(os.path.join(root, "per_checkpoint*.csv"))))
    if not runs or not pcs:
        return [f"G1/G2 missing runs*.csv or per_checkpoint*.csv under {root}"]
    imgs = [stem(r["image"]) for r in runs]
    if len(runs) != N_MAIN or len(set(imgs)) != N_MAIN:
        fails.append(f"G1 runs rows {len(runs)}, distinct images {len(set(imgs))} (expected {N_MAIN})")
    errs = [r for r in runs if r.get("status") == "error"]
    if errs:
        fails.append(f"G1 {len(errs)} error rows")
    meth = {r["method"] for r in pcs}
    if meth != {method}:
        fails.append(f"G2 method column {sorted(meth)} != [{method}]")
    have: Dict[str, set] = {}
    seeds: Dict[str, set] = {}
    for r in pcs:
        have.setdefault(stem(r["image"]), set()).add(int(r["budget"]))
        seeds.setdefault(stem(r["image"]), set()).add(r.get("seed", ""))
    short = [i for i, b in have.items() if not set(CHECKPOINTS) <= b]
    if short:
        fails.append(f"G2 {len(short)} images lack a registered checkpoint (e.g. {short[0]})")
    ref_paths = sorted(glob.glob(os.path.join(b5_cells, f"b5_main_LGC_RC_EGC_{arch}_{SHORT[mode]}", mode,
                                              "per_checkpoint*.csv")))
    if not ref_paths:
        fails.append(f"G3 B5 reference cell not found under {b5_cells}")
    else:
        ref = _rows(ref_paths)
        ref_seed: Dict[str, set] = {}
        for r in ref:
            ref_seed.setdefault(stem(r["image"]), set()).add(r.get("seed", ""))
        if set(ref_seed) != set(have):
            fails.append(f"G3 image set differs from B5 EGC ({len(set(have) ^ set(ref_seed))} images differ)")
        bad = [i for i in have if i in ref_seed and (len(seeds[i]) != 1 or seeds[i] != ref_seed[i])]
        if bad:
            fails.append(f"G3 per-image seed differs from B5 EGC on {len(bad)} images (e.g. {bad[0]})")
    cfgp = os.path.join(root, "config_single.json")
    if not os.path.exists(cfgp):
        fails.append("G4 config_single.json missing")
    else:
        with open(cfgp) as fh:
            cfg = json.load(fh)
        want = {"methods": [method], "num_samples": N_MAIN, "max_queries": 10000, "seed": 42,
                "checkpoints": CHECKPOINTS, "exclude_seeds": EXCLUDE, "save_adv": True, "torch": TORCH}
        for k, v in want.items():
            if cfg.get(k) != v:
                fails.append(f"G4 config {k}={cfg.get(k)!r} (expected {v!r})")
    return fails


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inputs(repo: str = ".", data_dir: str = DATA_DIR, victim_dir: str = VICTIM_DIR,
                 victim_files: List[str] = VICTIM_FILES, ae_path: str = AE_PATH, ae_prefix: str = AE_PREFIX,
                 contract: str = "contract/paper_contract.yaml", b5_root: str = "paper_exports/b5a2") -> List[str]:
    """Return the list of missing or wrong inputs (empty = all present). Prints one line per item."""
    fails: List[str] = []

    def item(ok: bool, what: str) -> None:
        print(f"[inputs] {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    ae = os.path.join(repo, ae_path)
    item(os.path.isfile(ae) and _sha(ae).startswith(ae_prefix), f"autoencoder {ae_path} (sha256 {ae_prefix}...)")
    for f in victim_files:
        p = os.path.join(victim_dir, f)
        pre = os.path.splitext(f)[0].split("-")[-1]
        item(os.path.isfile(p) and _sha(p).startswith(pre), f"victim {p} (sha256 {pre}...)")
    lab = os.path.join(data_dir, LABELS)
    item(os.path.isfile(lab) and os.path.getsize(lab) > 0, f"labels {lab}")
    want = {}
    try:
        import yaml  # the experiment environment has PyYAML (claims_registered.py uses it)
        with open(os.path.join(repo, contract)) as fh:
            want = (yaml.safe_load(fh).get("constants") or {}).get("split_sha256") or {}
    except Exception as e:  # noqa: BLE001
        item(False, f"contract {contract} readable ({type(e).__name__})")
    missing_img = 0
    for arch in ARCHS:
        for mode in MODES:
            sp = os.path.join(repo, "splits", "imagenet", "ttba_github", f"{arch}_{mode}.csv")
            key = f"{arch}_{mode}"
            ok = os.path.isfile(sp) and want.get(key) == _sha(sp)
            item(ok, f"split {sp} (sha256 = contract {key})")
            if os.path.isfile(sp):
                for r in _rows([sp])[:N_MAIN]:
                    for col in ("source", "target"):
                        n = r.get(col) or ""
                        if n and not os.path.isfile(os.path.join(data_dir, "images", n)):
                            missing_img += 1
            cell = f"b5_main_LGC_RC_EGC_{arch}_{SHORT[mode]}"
            pc = glob.glob(os.path.join(repo, b5_root, "cells", cell, mode, "per_checkpoint*.csv"))
            met = os.path.join(repo, b5_root, "metrics", cell, "metrics.csv")
            item(bool(pc) and os.path.isfile(met), f"B5 EGC cell and metrics {cell}")
    item(missing_img == 0, f"images of rows 1-{N_MAIN} under {os.path.join(data_dir, 'images')} ({missing_img} missing)")
    print(f"[inputs] {'ALL PRESENT' if not fails else str(len(fails)) + ' missing or wrong'}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", nargs="?")
    ap.add_argument("--results_root", default="Results")
    ap.add_argument("--b5_cells", default="paper_exports/b5a2/cells")
    ap.add_argument("--root", default=None)
    ap.add_argument("--print_order", action="store_true")
    ap.add_argument("--print_errors", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check_inputs", action="store_true")
    a = ap.parse_args(argv)
    if a.check_inputs:
        return 1 if check_inputs() else 0
    if a.print_order:
        for name, m, arch, mode in cells():
            print(f"{name}\t{outdir(name, a.results_root)}")
        return 0
    if a.print_errors:
        root = a.root or (outdir(a.cell, a.results_root) if a.cell in CELL else None)
        if not root:
            ap.error("--print_errors needs --root or a known cell")
        errs = error_rows(root)
        print(f"error rows under {root}: {len(errs)}")
        for p, r in errs:
            print(f"  {p}: image={r.get('image')} error={r.get('error')}")
        return 1 if errs else 0
    names = [c[0] for c in cells()] if a.all else [a.cell]
    if not names or names == [None] or any(n not in CELL for n in names):
        ap.error(f"a known cell name or --all is required (cells: {', '.join(sorted(CELL))})")
    nfail = 0
    for n in names:
        f = gate(n, a.results_root, a.b5_cells)
        print(f"[gate] {n}: {'PASS' if not f else 'FAIL'}")
        for x in f:
            print(f"    {x}")
        nfail += bool(f)
    if a.all:
        print(f"[gate] {len(names) - nfail} of {len(names)} cells PASS")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
