#!/usr/bin/env python3
"""T2: ImageNet adapter regression — compare b4c_regress_* (row 1) vs b4_smoke_* for the same image.

Per method/mode: found, state_query, pix_l2 at all 8 checkpoints; npz `delta` bitwise.
Verdict per method.

    python adapters/diag/diag_t2_regress.py
"""
from __future__ import annotations

import csv
import os

import numpy as np

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")
IMAGE = "ILSVRC2012_val_00007450"

# method -> list of (mode, smoke_ck, regress_ck, smoke_root, regress_root, arch)
CELLS = [
    ("TtBA", "non_targeted", "b4_smoke_ttba_resnet50", "b4c_regress_ttba", "resnet50", "TtBA"),
    ("TtBA", "targeted", "b4_smoke_ttba_resnet50", "b4c_regress_ttba", "resnet50", "TtBA"),
    ("CGBA", "non_targeted", "b4_smoke_cgba_resnet50", "b4c_regress_cgba", "resnet50", "CGBA"),
    ("CGBA_H", "targeted", "b4_smoke_cgba_resnet50", "b4c_regress_cgba_h", "resnet50", "CGBA_H"),
    ("ARS_OPT", "non_targeted", "b4_smoke_arsopt_resnet50", "b4c_regress_arsopt", "resnet50", "ARS_OPT"),
    ("ARS_OPT", "targeted", "b4_smoke_arsopt_resnet50", "b4c_regress_arsopt", "resnet50", "ARS_OPT"),
]


def load_ck(root: str, mode: str, method: str) -> dict:
    p = os.path.join(R, root, mode, f"per_checkpoint_{method}.csv")
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            stem = os.path.splitext(r["image"])[0]
            if stem != IMAGE:
                continue
            out[int(r["budget"])] = (stem, r)
    return out


def npz(root: str, mode: str, arch: str, method: str, budget: int) -> str:
    stem = os.path.splitext(IMAGE)[0]
    p = os.path.join(R, root, mode, "adv_store", arch, stem, f"{method}_Q{budget}.npz")
    return p if os.path.exists(p) else None


def main() -> int:
    all_ok = True
    for method, mode, smoke_root, reg_root, arch, mfile in CELLS:
        sc = load_ck(smoke_root, mode, mfile)
        rc = load_ck(reg_root, mode, mfile)
        problems = []
        n_npz = 0
        for b in sorted(set(sc) | set(rc)):
            if b not in sc or b not in rc:
                problems.append(f"budget {b} missing in one side")
                continue
            ss, sr = sc[b][1], rc[b][1]
            for k in ("found", "state_query", "pix_l2"):
                va = float(sr[k]) if k == "pix_l2" else str(sr[k]).strip()
                vb = float(ss[k]) if k == "pix_l2" else str(ss[k]).strip()
                if va != vb:
                    problems.append(f"budget {b} {k}: smoke={vb!r} regress={va!r}")
            if int(sr["found"]) == 1:
                pa, pb = npz(smoke_root, mode, arch, mfile, b), npz(reg_root, mode, arch, mfile, b)
                if pa is None or pb is None:
                    problems.append(f"budget {b} npz missing (smoke={pa}, regress={pb})")
                    continue
                da = np.load(pa, allow_pickle=False)["delta"]
                db = np.load(pb, allow_pickle=False)["delta"]
                if da.shape != db.shape or not np.array_equal(da, db):
                    problems.append(f"budget {b} npz delta differs")
                else:
                    n_npz += 1
        verdict = "IDENTICAL" if not problems else "DIVERGED"
        if problems:
            all_ok = False
        print(f"{method:8s} {mode:12s}: {verdict}  ({n_npz} npz bitwise equal)")
        for p in problems:
            print(f"    - {p}")
    print("\nOVERALL:", "IDENTICAL" if all_ok else "STOP (divergence)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
