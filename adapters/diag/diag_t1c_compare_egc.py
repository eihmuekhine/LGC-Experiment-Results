#!/usr/bin/env python3
"""T1(c): compare generic-runner resnet50 NT+T rows 1-3 vs Results/b4_smoke_egc_resnet50.

Per (mode, image, budget): found, state_query, pix_l2 exactly equal; npz `delta` bitwise equal.
Prints IDENTICAL or the first diverging row.

    python adapters/diag/diag_t1c_compare_egc.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

REPO = "/root/autodl-tmp/LGC_frozen"
A = os.path.join(REPO, "Results", "b4_smoke_egc_resnet50")
B = os.path.join(REPO, "Results", "b4c_egc_generic_resnet50")

KEYS = ["found", "state_query", "pix_l2"]


def load_ck(root: str) -> dict:
    out = {}
    for mode in ("non_targeted", "targeted"):
        p = os.path.join(root, mode, "per_checkpoint.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (mode, r["image"], int(r["budget"]))
                out[key] = r
    return out


def npz_path(root: str, mode: str, image: str, budget: int) -> str:
    stem = os.path.splitext(image)[0]
    p = os.path.join(root, mode, "adv_store", "resnet50", stem, f"LGC_RC_EGC_Q{budget}.npz")
    return p if os.path.exists(p) else None


def main() -> int:
    a = load_ck(A)
    b = load_ck(B)
    keys = sorted(set(a) | set(b))
    print(f"rows: smoke={len(a)} generic={len(b)}")

    first = None
    n_npz = 0
    for key in keys:
        if key not in a:
            print(f"MISSING in smoke: {key}"); return 1
        if key not in b:
            print(f"MISSING in generic: {key}"); return 1
        ra, rb = a[key], b[key]
        for k in KEYS:
            va, vb = ra[k], rb[k]
            # pix_l2 exact float equality
            if k == "pix_l2":
                eq = float(va) == float(vb)
            else:
                eq = str(va).strip() == str(vb).strip()
            if not eq:
                print(f"DIVERGE {key} field={k}: smoke={va!r} generic={vb!r}")
                if first is None:
                    first = (key, k)
        if int(ra["found"]) == 1:
            pa, pb = npz_path(A, key[0], key[1], key[2]), npz_path(B, key[0], key[1], key[2])
            if pa is None or pb is None:
                print(f"npz missing for {key}: smoke={pa} generic={pb}")
                if first is None:
                    first = (key, "npz")
                continue
            da = np.load(pa, allow_pickle=False)["delta"]
            db = np.load(pb, allow_pickle=False)["delta"]
            if da.shape != db.shape or not np.array_equal(da, db):
                print(f"npz delta DIVERGE {key}: shapes {da.shape} vs {db.shape}, equal={np.array_equal(da, db)}")
                if first is None:
                    first = (key, "npz-delta")
            else:
                n_npz += 1

    if first is None:
        print(f"IDENTICAL: all {len(keys)} rows match; {n_npz} npz deltas bitwise equal.")
        return 0
    print(f"STOP: first diverging row {first[0]} field {first[1]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
