#!/usr/bin/env python3
"""Read-only bitwise comparison of two result roots, for a HUMAN to verify an agent's "IDENTICAL" claim.

Use it on B5-host T1 (A vs A2, A vs B), T2/T3 (new host vs smoke reference) and T5 (GPU i vs GPU 0).

Usage (from the worktree root):
    python tools/eval/compare_roots.py --a Results/b5host_A --b Results/b5host_A2
    python tools/eval/compare_roots.py --a <ref root> --b <new root> --budgets 100 200 500 1000 2000 10000

What is compared, per (mode dir, arch, method, image, budget) present in BOTH roots:
    per_checkpoint*.csv : found and state_query exactly; pix_l2 exactly (float64 equality, no tolerance)
    adv_store/**.npz    : every array except keys containing "time" (dtype, shape and bytes)
Rows present in only one root are counted and listed (first 10); they are not differences, but a verdict
IDENTICAL is printed only when the shared key set is non-empty and nothing differs.
Exit code: 0 IDENTICAL, 1 DIFFERENT, 2 nothing to compare.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

Key = Tuple[str, str, str, str, int]


def stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_checkpoints(root: str) -> Dict[Key, Tuple[int, int, float]]:
    out: Dict[Key, Tuple[int, int, float]] = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "per_checkpoint*.csv"), recursive=True)):
        try:
            df = pd.read_csv(f, float_precision="round_trip")  # exact text -> float64
        except pd.errors.EmptyDataError:
            continue
        need = {"image", "budget", "found", "pix_l2"}
        if not need <= set(df.columns):
            print(f"[!] {f}: missing columns {sorted(need - set(df.columns))}; skipped")
            continue
        mode = os.path.basename(os.path.dirname(f))
        meth_default = os.path.basename(f)[len("per_checkpoint"):].lstrip("_").rsplit(".", 1)[0] or "?"
        for r in df.itertuples(index=False):
            d = r._asdict()
            meth = str(d.get("method", meth_default))
            sq = int(d["state_query"]) if "state_query" in d and pd.notna(d["state_query"]) else -1
            pix = float(d["pix_l2"]) if pd.notna(d["pix_l2"]) else float("nan")
            arch = str(d.get("arch", "?"))
            out[(mode, arch, meth, stem(d["image"]), int(d["budget"]))] = (int(d["found"]), sq, pix)
    return out


def npz_index(root: str) -> Dict[Tuple[str, str, str, str], str]:
    """(mode dir, arch, image, file name) -> path for every stored npz (x0.npy is ignored here)."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "adv_store", "**", "*.npz"), recursive=True):
        rel = os.path.relpath(p, root).split(os.sep)
        i = rel.index("adv_store")
        mode = rel[i - 1] if i > 0 else "?"
        arch = rel[i + 1] if len(rel) > i + 3 else "?"
        out[(mode, arch, rel[-2], rel[-1])] = p
    return out


def array_digest(path: str) -> Dict[str, str]:
    with np.load(path, allow_pickle=False) as z:
        res = {}
        for k in sorted(z.files):
            if "time" in k.lower():
                continue
            a = np.ascontiguousarray(z[k])
            res[k] = f"{a.dtype}|{a.shape}|" + hashlib.sha256(a.tobytes()).hexdigest()
        return res


def same_float(x: float, y: float) -> bool:
    return (np.isnan(x) and np.isnan(y)) or x == y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference root")
    ap.add_argument("--b", required=True, help="root to compare")
    ap.add_argument("--budgets", type=int, nargs="*", default=None, help="restrict to these budgets (default: all shared)")
    ap.add_argument("--no_npz", action="store_true", help="skip the npz array comparison")
    a = ap.parse_args()

    ca, cb = load_checkpoints(a.a), load_checkpoints(a.b)
    if a.budgets:
        keep = set(a.budgets)
        ca = {k: v for k, v in ca.items() if k[4] in keep}
        cb = {k: v for k, v in cb.items() if k[4] in keep}
    shared = sorted(set(ca) & set(cb))
    only_a, only_b = sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))
    diffs: List[str] = []
    max_abs = 0.0
    for k in shared:
        fa, sa, pa = ca[k]
        fb, sb, pb = cb[k]
        if not same_float(pa, pb):
            max_abs = max(max_abs, abs(pa - pb)) if not (np.isnan(pa) or np.isnan(pb)) else float("inf")
        if fa != fb or sa != sb or not same_float(pa, pb):
            diffs.append(f"{k}: A(found={fa}, state_query={sa}, pix_l2={pa!r}) B(found={fb}, state_query={sb}, pix_l2={pb!r})")

    print(f"# compare_roots\nA = {a.a}\nB = {a.b}\n")
    print(f"checkpoint keys: shared {len(shared)}, only in A {len(only_a)}, only in B {len(only_b)}")
    for label, lst in (("only in A", only_a), ("only in B", only_b)):
        for k in lst[:10]:
            print(f"  {label}: {k}")
    print(f"checkpoint differences: {len(diffs)}; max |pix_l2 A - B| = {max_abs:.3e}")
    for d in diffs[:10]:
        print(f"  DIFF {d}")

    n_npz_shared, npz_diffs = 0, []
    if not a.no_npz:
        na, nb = npz_index(a.a), npz_index(a.b)
        if a.budgets:
            keep_q = {f"_Q{b}.npz" for b in a.budgets}
            na = {k: v for k, v in na.items() if any(k[2].endswith(q) for q in keep_q)}
            nb = {k: v for k, v in nb.items() if any(k[2].endswith(q) for q in keep_q)}
        for k in sorted(set(na) & set(nb)):
            n_npz_shared += 1
            da, db = array_digest(na[k]), array_digest(nb[k])
            if da != db:
                bad = sorted(x for x in set(da) | set(db) if da.get(x) != db.get(x))
                npz_diffs.append(f"{k}: arrays differ {bad}")
        print(f"npz files: shared {n_npz_shared}, only in A {len(set(na) - set(nb))}, only in B {len(set(nb) - set(na))}; "
              f"differences: {len(npz_diffs)}")
        for d in npz_diffs[:10]:
            print(f"  DIFF {d}")

    if not shared:
        print("VERDICT: NOTHING-TO-COMPARE")
        return 2
    first: Optional[str] = (diffs or npz_diffs or [None])[0]
    if diffs or npz_diffs:
        print(f"VERDICT: DIFFERENT (first: {first})")
        return 1
    print("VERDICT: IDENTICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
