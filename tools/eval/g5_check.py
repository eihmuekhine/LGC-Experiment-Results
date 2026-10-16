#!/usr/bin/env python3
"""G5 magnitude-sanity gate (D13) — run after the first 30 images of each TtBA / CGBA / CGBA_H cell.

Replaces ``adapters/diag/g5_table1.py`` as the B5 gate.  Two checks, both compared to the published
TtBA Table 1 (``ttba_github/ttba_table1.json``, format ``[mean, median]`` per budget; n = 1000, their
image list and environment — NOT paired with our runs):

  * **TtBA**: ratio of OUR median ``pix_l2_iter`` to the published TtBA median, at budgets 2000/5000/10000.
  * **CGBA / CGBA_H**: rho_B = [med_i L_m(B) / med_i L_TtBA(B)] / [pub_med_m(B) / pub_med_TtBA(B)] on images
    completed by BOTH methods (paired on image stem).  ARS_OPT: n/a.

For each check the 95 % percentile bootstrap CI (paired over images, 10000 resamples, seed 0) is computed on the
ratio; **PASS iff the CI intersects [0.5, 2] at every B**.  Ratios of *means* are printed as descriptive only.
When n < 30 the tool prints "tool check only" and never emits a verdict (smoke / dry-run).  Exit 0 on PASS, 1 on
FAIL, 2 on usage/read error.  Reads only; never writes.

    python tools/eval/g5_check.py --arch resnet50 --mode non_targeted \
        --ttba_root Results/b5_ttba_resnet50 --cgba_root Results/b5_cgba_resnet50
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUDGETS = [2000, 5000, 10000]
# published method key -> our per_checkpoint file stem / method id
PUBKEY_CGBA = {"CGBA": "CGBA", "CGBA_H": "CGBA_H"}
N_MIN = 30


def _find_table1(path):
    if path and os.path.exists(path):
        return path
    cand = os.path.join(REPO, "ttba_github", "ttba_table1.json")
    return cand if os.path.exists(cand) else None


def _load_ck(root, mode, stem):
    """Load <root>/<mode>/per_checkpoint_<stem>.csv -> {image: {budget: row}}."""
    f = os.path.join(root, mode, f"per_checkpoint_{stem}.csv")
    if not os.path.exists(f):
        return None, f
    out = {}
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["image"], {})[int(r["budget"])] = r
    return out, f


def _val(row, key):
    return float(row[key])


def _median(xs):
    return float(np.median(np.asarray(xs, dtype=float)))


def _bootstrap_ci(stat_fn, images, n_boot, seed):
    """Percentile 95 % CI of stat_fn(images[subset]) over n_boot resamples of `images` (list of image keys)."""
    rng = np.random.RandomState(seed)
    n = len(images)
    idx = np.arange(n)
    stats = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        stats[k] = stat_fn(idx[rng.choice(idx, size=n, replace=True)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _intersect_ci(lo, hi):
    return lo <= 2.0 and hi >= 0.5


def _pub(table1, arch, method, mode, budget):
    return table1["table1"][arch][method][mode][str(budget)]


def check_ttba(ttba_ck, table1, arch, mode, n_boot, seed):
    """TtBA magnitude check: our median pix_l2_iter / published median."""
    lines = []
    n = 0
    for img, rec in ttba_ck.items():
        if rec.get(BUDGETS[0]) is not None:
            n += 1
    all_pass = True
    for b in BUDGETS:
        imgs = [img for img, rec in ttba_ck.items()
                if b in rec and int(rec[b].get("found", 0)) == 1]
        if not imgs:
            lines.append((b, None, "no found rows", False))
            all_pass = False
            continue
        pub_med = _pub(table1, arch, "TtBA", mode, b)[1]
        pub_mean = _pub(table1, arch, "TtBA", mode, b)[0]
        med = _median([_val(ttba_ck[i][b], "pix_l2_iter") for i in imgs])

        def stat(sel):
            vals = [_val(ttba_ck[imgs[i]][b], "pix_l2_iter") for i in sel]
            return _median(vals) / pub_med

        lo, hi = _bootstrap_ci(stat, imgs, n_boot, seed)
        ok = _intersect_ci(lo, hi)
        all_pass &= ok
        mean_ratio = np.mean([_val(ttba_ck[i][b], "pix_l2_iter") for i in imgs]) / pub_mean
        lines.append((b, (med / pub_med, lo, hi, med, pub_med, mean_ratio), None, ok))
    return n, all_pass, lines


def check_cgba(cgba_ck, ttba_ck, table1, arch, mode, method, n_boot, seed):
    """CGBA / CGBA_H rho check on images completed by both methods."""
    lines = []
    pub_key = PUBKEY_CGBA[method]
    n = 0
    for img in set(cgba_ck) & set(ttba_ck):
        if cgba_ck[img].get(BUDGETS[0]) is not None or ttba_ck[img].get(BUDGETS[0]) is not None:
            n += 1
    all_pass = True
    for b in BUDGETS:
        paired = [img for img in set(cgba_ck) & set(ttba_ck)
                  if b in cgba_ck[img] and b in ttba_ck[img]
                  and int(cgba_ck[img][b].get("found", 0)) == 1
                  and int(ttba_ck[img][b].get("found", 0)) == 1]
        if not paired:
            lines.append((b, None, "no paired found rows", False))
            all_pass = False
            continue
        pub_m = _pub(table1, arch, pub_key, mode, b)[1]
        pub_t = _pub(table1, arch, "TtBA", mode, b)[1]
        pub_mean_m = _pub(table1, arch, pub_key, mode, b)[0]
        pub_mean_t = _pub(table1, arch, "TtBA", mode, b)[0]

        def rho(sel):
            med_m = _median([_val(cgba_ck[paired[i]][b], "pix_l2") for i in sel])
            med_t = _median([_val(ttba_ck[paired[i]][b], "pix_l2_iter") for i in sel])
            return (med_m / med_t) / (pub_m / pub_t)

        lo, hi = _bootstrap_ci(rho, paired, n_boot, seed)
        ok = _intersect_ci(lo, hi)
        all_pass &= ok
        rho_pt = rho(np.arange(len(paired)))
        rho_mean = (np.mean([_val(cgba_ck[i][b], "pix_l2") for i in paired]) /
                    np.mean([_val(ttba_ck[i][b], "pix_l2_iter") for i in paired])) / \
                   (pub_mean_m / pub_mean_t)
        lines.append((b, (rho_pt, lo, hi, rho_mean), None, ok))
    return n, all_pass, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", required=True, choices=["non_targeted", "targeted"])
    ap.add_argument("--ttba_root", required=True)
    ap.add_argument("--cgba_root", default=None, help="optional: run the CGBA/CGBA_H rho check too")
    ap.add_argument("--table1", default=None)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    table1_path = _find_table1(a.table1)
    if table1_path is None:
        print("FLAG (error): ttba_table1.json not found")
        return 2
    table1 = json.load(open(table1_path))
    if a.arch not in table1["table1"]:
        print(f"FLAG (error): arch {a.arch} not in ttba_table1.json "
              f"({sorted(table1['table1'])})")
        return 2

    ttba_ck, ttba_f = _load_ck(a.ttba_root, a.mode, "TtBA")
    if ttba_ck is None:
        print(f"FLAG (error): TtBA per_checkpoint missing: {ttba_f}")
        return 2

    cgba_method = None
    cgba_ck = None
    if a.cgba_root is not None:
        for stem, meth in (("CGBA", "CGBA"), ("CGBA_H", "CGBA_H")):
            ck, f = _load_ck(a.cgba_root, a.mode, stem)
            if ck is not None:
                cgba_ck, cgba_method = ck, meth
                break
        if cgba_ck is None:
            print(f"FLAG (error): no CGBA / CGBA_H per_checkpoint in {a.cgba_root}/{a.mode}")
            return 2

    print(f"g5_check arch={a.arch} mode={a.mode} table1={table1_path} n_boot={a.n_boot} seed={a.seed}")

    verdicts = []

    # --- TtBA magnitude check ---
    n_t, ttba_pass, ttba_lines = check_ttba(ttba_ck, table1, a.arch, a.mode, a.n_boot, a.seed)
    print(f"\nTtBA (median pix_l2_iter vs published median)  n={n_t}")
    for b, stat, note, ok in ttba_lines:
        if stat is None:
            print(f"  @{b}: {note}  -> FLAG")
            continue
        ratio, lo, hi, med, pub_med, mean_ratio = stat
        print(f"  @{b}: ratio(med)={ratio:.3f} 95%CI=[{lo:.3f},{hi:.3f}] intersect[0.5,2]={ok} "
              f"(our med={med:.3f} pub med={pub_med:.3f}; ratio(mean)={mean_ratio:.3f} descriptive)")
    verdicts.append(("TtBA", ttba_pass, n_t))

    # --- CGBA / CGBA_H rho check ---
    if cgba_ck is not None:
        n_c, cgba_pass, cgba_lines = check_cgba(cgba_ck, ttba_ck, table1, a.arch, a.mode,
                                                 cgba_method, a.n_boot, a.seed)
        print(f"\n{cgba_method} (rho_B ratio-of-ratios on paired images)  n={n_c}")
        for b, stat, note, ok in cgba_lines:
            if stat is None:
                print(f"  @{b}: {note}  -> FLAG")
                continue
            rho, lo, hi, rho_mean = stat
            print(f"  @{b}: rho={rho:.3f} 95%CI=[{lo:.3f},{hi:.3f}] intersect[0.5,2]={ok} "
                  f"(rho(mean)={rho_mean:.3f} descriptive)")
        verdicts.append((cgba_method, cgba_pass, n_c))

    # --- verdict ---
    n_all = [n for _, _, n in verdicts]
    n_min = min(n_all) if n_all else 0
    if n_min < N_MIN:
        print(f"\nVERDICT: tool check only (n={n_min} < {N_MIN}; not a gate)")
        return 0
    fail = [name for name, ok, _ in verdicts if not ok]
    if fail:
        print(f"\nVERDICT: FAIL (checks failing: {fail})")
        return 1
    print(f"\nVERDICT: PASS (all CIs intersect [0.5, 2] at every budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
