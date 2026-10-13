#!/usr/bin/env python3
"""B4e T1(d) — numeric + adv_store `delta` comparison of a NEW run vs an EXISTING (old) smoke cell.

Compares per-image `found` / `state_query` / `pix_l2` at all 8 contract checkpoints and the sha256 of the adv_store
npz `delta` bytes.  The `method` column (and the npz file name) differs by design (e.g. ``CGBA_D8only`` vs ``CGBA``)
and is passed in separately, never compared.

Modes:
  * identical: every checkpoint must match exactly (found/state_query/pix_l2 bit-equal, npz delta bit-equal).
  * prefix: compare checkpoints in increasing budget order and stop at the FIRST divergence; report that budget and
    assert every budget before it is identical (used for the fixed-vs-stalled comparison: budgets below the
    first-NaN query index are identical, afterwards the fixed variant keeps improving).

    python adapters/diag/diag_b4e_t1_compare.py --new-root ... --new-method CGBA_D8only \
        --old-root Results/b4_smoke_cgba_resnet50 --old-method CGBA --arch resnet50 --mode non_targeted
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os

import numpy as np

REPO = "/root/autodl-tmp/LGC_frozen"
CKPT = [100, 200, 500, 1000, 2000, 2500, 5000, 10000]


def load_ck(root, mode, method):
    p = os.path.join(root, mode, f"per_checkpoint_{method}.csv")
    if not os.path.exists(p):
        return None, p
    out = {}
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["image"], {})[int(r["budget"])] = r
    return out, p


def npz_delta_sha(root, mode, arch, image, method, budget):
    p = os.path.join(root, mode, "adv_store", arch, image, f"{method}_Q{budget}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)["delta"]
    return hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()


def compare_image(img, new_rec, old_rec, args, out):
    diverged = False
    n_npz = 0
    for b in CKPT:
        if b not in new_rec or b not in old_rec:
            out.append(f"  {img} @{b}: MISSING (new={b in new_rec}, old={b in old_rec})")
            return 1
        nr, orow = new_rec[b], old_rec[b]
        keys = ("found", "state_query", "pix_l2")
        for k in keys:
            vn = str(nr[k]).strip()
            vo = str(orow[k]).strip()
            if vn != vo:
                out.append(f"  {img} @{b}: {k} DIVERGES new={vn!r} old={vo!r}")
                diverged = True
                break
        if diverged:
            break
        if int(nr["found"]) == 1:
            hn = npz_delta_sha(args.new_root, args.mode, args.arch, img, args.new_method, b)
            ho = npz_delta_sha(args.old_root, args.mode, args.arch, img, args.old_method, b)
            if hn is None or ho is None:
                out.append(f"  {img} @{b}: npz MISSING (new={bool(hn)}, old={bool(ho)})")
                diverged = True
                break
            if hn != ho:
                out.append(f"  {img} @{b}: npz delta DIVERGES new={hn[:16]} old={ho[:16]}")
                diverged = True
                break
            n_npz += 1
    if diverged:
        if args.cmp == "prefix":
            out.append(f"  {img}: prefix IDENTICAL through budget {CKPT[max(0, CKPT.index(b) - 1)]}, "
                       f"diverges at {b} (npz_deltas_equal_before={n_npz})")
            return 0
        return 1
    out.append(f"  {img}: IDENTICAL ({n_npz} npz deltas bitwise equal)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--old-root", required=True)
    ap.add_argument("--new-method", required=True)
    ap.add_argument("--old-method", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", required=True, choices=["non_targeted", "targeted"])
    ap.add_argument("--cmp", choices=["identical", "prefix"], default="identical")
    a = ap.parse_args()

    new_ck, pn = load_ck(a.new_root, a.mode, a.new_method)
    old_ck, po = load_ck(a.old_root, a.mode, a.old_method)
    if new_ck is None:
        print(f"FAIL: new per_checkpoint missing: {pn}")
        return 2
    if old_ck is None:
        print(f"FAIL: old per_checkpoint missing: {po}")
        return 2

    imgs = sorted(set(new_ck) & set(old_ck))
    print(f"compare {a.cmp}: new={a.new_root} method={a.new_method}  old={a.old_root} method={a.old_method} "
          f"images={len(imgs)}")
    n_fail = 0
    for img in imgs:
        detail = []
        n_fail += compare_image(img, new_ck[img], old_ck[img], a, detail)
        for line in detail:
            print(line)
    verdict = "PASS" if n_fail == 0 else "FAIL"
    print(f"VERDICT: {verdict} ({n_fail} diverging images / {len(imgs)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
