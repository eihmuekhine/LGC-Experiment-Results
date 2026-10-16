#!/usr/bin/env python3
"""B4d T3 — concurrency neutrality: compare solo vs 2-process concurrent results for the SAME image.

For TtBA (per_checkpoint_TtBA.csv) and LGC_RC_EGC (per_checkpoint.csv), resnet50 NT row 1:
  * found / state_query / pix_l2 must be exactly equal at every checkpoint;
  * the adv_store npz `delta` must be bitwise equal (sha256 of the delta bytes).

Verdict IDENTICAL, or the first diverging checkpoint.

    python adapters/diag/diag_b4d_t3_compare.py
"""
from __future__ import annotations

import csv
import hashlib
import os

import numpy as np

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")
IMAGE = "ILSVRC2012_val_00007450.JPEG"
STEM = os.path.splitext(IMAGE)[0]
CKPT = [100, 200, 500, 1000, 2000, 2500, 5000, 10000]

CELLS = [
    ("TtBA", "b4d_t3_solo_ttba", "b4d_t3_conc2_ttba", "TtBA", "per_checkpoint_TtBA.csv"),
    ("LGC_RC_EGC", "b4d_t3_solo_egc_resnet50", "b4d_t3_conc2_egc_resnet50", "LGC_RC_EGC", "per_checkpoint.csv"),
]


def load_ck(root, fname):
    p = os.path.join(R, root, "non_targeted", fname)
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            if os.path.splitext(r["image"])[0] == STEM:
                out[int(r["budget"])] = r
    return out


def npz_delta_sha(root, method, budget):
    p = os.path.join(R, root, "non_targeted", "adv_store", "resnet50", STEM, f"{method}_Q{budget}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)["delta"]
    return hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()


def main() -> int:
    for method, solo_root, conc_root, mfile, fname in CELLS:
        sc = load_ck(solo_root, fname)
        cc = load_ck(conc_root, fname)
        first = None
        n_npz = 0
        for b in CKPT:
            if b not in sc or b not in cc:
                print(f"{method}: checkpoint {b} missing (solo={b in sc}, conc={b in cc})")
                if first is None:
                    first = b
                continue
            sr, cr = sc[b], cc[b]
            for k in ("found", "state_query", "pix_l2"):
                vs = float(sr[k]) if k == "pix_l2" else str(sr[k]).strip()
                vc = float(cr[k]) if k == "pix_l2" else str(cr[k]).strip()
                if vs != vc:
                    print(f"{method}: checkpoint {b} {k} DIVERGES: solo={vs!r} conc={vc!r}")
                    if first is None:
                        first = b
            if int(cr["found"]) == 1:
                hs = npz_delta_sha(solo_root, mfile, b)
                hc = npz_delta_sha(conc_root, mfile, b)
                if hs is None or hc is None:
                    print(f"{method}: checkpoint {b} npz missing (solo={bool(hs)}, conc={bool(hc)})")
                    if first is None:
                        first = b
                elif hs != hc:
                    print(f"{method}: checkpoint {b} npz delta sha DIVERGES: solo={hs[:16]} conc={hc[:16]}")
                    if first is None:
                        first = b
                else:
                    n_npz += 1
        verdict = "IDENTICAL" if first is None else f"DIVERGE at checkpoint {first}"
        print(f"{method}: {verdict}  ({n_npz} npz deltas bitwise equal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
