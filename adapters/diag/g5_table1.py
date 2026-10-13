#!/usr/bin/env python3
"""B4d T6 — G5 gate tool (zero GPU): compare ONE cell's mean/median pix_l2 @2000/5000/10000 with the
published TtBA Table 1 (ttba_github/ttba_table1.json) for the same victim/mode.

Usage:
    python adapters/diag/g5_table1.py --out_dir <cell out dir> --arch <resnet50|vgg19|vit_b_32> \
        --mode <non_targeted|targeted> [--table1 ttba_github/ttba_table1.json]

The cell out dir contains <mode>/per_checkpoint_<METHOD>.csv; the method is read from the file name
(CGBA / CGBA_H / TtBA).  It computes mean and median pix_l2 at budgets 2000/5000/10000 over rows with
found == 1, and prints ratio(mean) and ratio(median) vs the published table.  PASS iff EVERY ratio is in
[0.5, 2]; otherwise FLAG.  Exit code 0 on PASS, 1 on FLAG (and 2 on usage/read errors).

This is the B5 G5 gate; it never writes any file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

REPO = "/root/autodl-tmp/LGC_frozen"
BUDGETS = [2000, 5000, 10000]
# our method file stem -> published table key
PUBKEY = {"CGBA": "CGBA", "CGBA_H": "CGBA-H", "TtBA": "TtBA"}
# published method key -> (method file stem, mode)
CELL_METHODS = {
    "CGBA": "CGBA",
    "CGBA-H": "CGBA_H",
    "TtBA": "TtBA",
}


def find_table1(path):
    if path and os.path.exists(path):
        return path
    cand = os.path.join(REPO, "ttba_github", "ttba_table1.json")
    if os.path.exists(cand):
        return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="cell out dir containing <mode>/per_checkpoint_<METHOD>.csv")
    ap.add_argument("--arch", required=True, choices=["resnet50", "vgg19", "vit_b_32"])
    ap.add_argument("--mode", required=True, choices=["non_targeted", "targeted"])
    ap.add_argument("--table1", default=None, help="path to ttba_table1.json (default: worktree copy)")
    a = ap.parse_args()

    mode_dir = os.path.join(a.out_dir, a.mode)
    if not os.path.isdir(mode_dir):
        print(f"FLAG (error): no such cell dir {mode_dir}")
        return 2
    ck_files = sorted(glob.glob(os.path.join(mode_dir, "per_checkpoint_*.csv")))
    if not ck_files:
        print(f"FLAG (error): no per_checkpoint_*.csv in {mode_dir}")
        return 2
    if len(ck_files) > 1:
        print(f"WARN: multiple per_checkpoint files: {ck_files}; using {ck_files[0]}")

    fname = os.path.basename(ck_files[0])
    stem = fname[len("per_checkpoint_"):-len(".csv")]
    if stem not in PUBKEY:
        print(f"FLAG (error): unknown method file {fname} (expected CGBA / CGBA_H / TtBA)")
        return 2
    method = stem

    vals = {b: [] for b in BUDGETS}
    with open(ck_files[0], newline="") as fh:
        for r in csv.DictReader(fh):
            b = int(r["budget"])
            if b in vals and int(r.get("found", 0)) == 1:
                vals[b].append(float(r["pix_l2"]))

    table1_path = find_table1(a.table1)
    if table1_path is None:
        print("FLAG (error): ttba_table1.json not found")
        return 2
    table1 = json.load(open(table1_path))
    if a.arch not in table1["table1"]:
        print(f"FLAG (error): arch {a.arch} not in ttba_table1.json victims "
              f"{sorted(table1['table1'])}")
        return 2
    pubkey = PUBKEY[method]
    if pubkey not in table1["table1"][a.arch]:
        print(f"FLAG (error): method {pubkey} not in ttba_table1.json for {a.arch}")
        return 2

    print(f"cell: arch={a.arch} mode={a.mode} method={method} n={len(vals[BUDGETS[0]])} "
          f"table1={table1_path}")
    flags = 0
    for b in BUDGETS:
        v = vals[b]
        if not v:
            print(f"  @{b}: no found rows -> ratio nan/nan FLAG")
            flags += 1
            continue
        pub = table1["table1"][a.arch][pubkey][a.mode][str(b)]
        pm, pmed = pub[0], pub[1]
        rm = statistics.mean(v) / pm
        rmed = statistics.median(v) / pmed
        bad = not (0.5 <= rm <= 2.0 and 0.5 <= rmed <= 2.0)
        if bad:
            flags += 1
        print(f"  @{b}: our mean={statistics.mean(v):.3f} median={statistics.median(v):.3f} "
              f"pub mean={pm:.3f} median={pmed:.3f} "
              f"ratio(mean)={rm:.3f} ratio(median)={rmed:.3f} {'FLAG' if bad else 'ok'}")

    verdict = "PASS" if flags == 0 else "FLAG"
    print(f"VERDICT: {verdict} (flags={flags})")
    return 0 if flags == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
