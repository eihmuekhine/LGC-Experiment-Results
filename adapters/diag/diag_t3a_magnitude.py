#!/usr/bin/env python3
"""T3(a): mean/median pix_l2 @2000/5000/10000 for CGBA / CGBA_H / TtBA per victim/mode from b4_smoke (n=3),
next to the published TtBA Table 1 (ttba_github/ttba_table1.json), with the ours/published ratio.

Flags any cell whose ratio (mean or median) is > 2 or < 0.5, per the G5 magnitude sanity check. The
published table is n = 1000, unpaired (their image list and environment), so this is a descriptive
sanity check only, never a tuning decision.

    python adapters/diag/diag_t3a_magnitude.py

The table file is looked up in this order: <repo>/ttba_github/ttba_table1.json, then the sibling
main repo /root/autodl-tmp/Latent-Geometric-Chords/ttba_github/ttba_table1.json.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")
BUDGETS = [2000, 5000, 10000]
ARCHS = ["resnet50", "vgg19", "vit_b_32"]
TABLE1_CANDIDATES = [
    os.path.join(REPO, "ttba_github", "ttba_table1.json"),
    "/root/autodl-tmp/Latent-Geometric-Chords/ttba_github/ttba_table1.json",
]
# our cell -> published table key ("CGBA-H" uses a hyphen in the published table)
PUBKEY = {"CGBA": "CGBA", "CGBA_H": "CGBA-H", "TtBA": "TtBA"}


def load(root: str, mode: str, method: str) -> dict:
    p = os.path.join(R, root, mode, f"per_checkpoint_{method}.csv")
    vals = {b: [] for b in BUDGETS}
    if not os.path.exists(p):
        return vals
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            b = int(r["budget"])
            if b in vals and int(r.get("found", 0)) == 1:
                vals[b].append(float(r["pix_l2"]))
    return vals


def find_table1():
    for p in TABLE1_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def main() -> int:
    table1_path = find_table1()
    if table1_path is None:
        print("ttba_table1.json: MISSING (UNVERIFIED) — no published ratios computable.")
        return 1
    table1 = json.load(open(table1_path))
    print(f"ttba_table1.json: {table1_path}")
    print("top-level keys:", sorted(table1.keys()))
    print('"table1" victim keys:', sorted(table1["table1"].keys()))
    print('"format":', table1.get("format"))

    pub = table1["table1"]

    flags = []
    print(f"\n{'method':8s} {'arch':9s} {'mode':12s} " +
          " ".join(f"@{b:>5d}" for b in BUDGETS))
    print("-" * 118)
    for meth, mode, mfile in [("CGBA", "non_targeted", "CGBA"),
                              ("CGBA_H", "targeted", "CGBA_H"),
                              ("TtBA", "non_targeted", "TtBA"),
                              ("TtBA", "targeted", "TtBA")]:
        for arch in ARCHS:
            root = f"b4_smoke_cgba_{arch}" if meth in ("CGBA", "CGBA_H") else f"b4_smoke_ttba_{arch}"
            vals = load(root, mode, mfile)
            pcell = pub[arch][PUBKEY[meth]][mode]
            cells = []
            for b in BUDGETS:
                v = vals[b]
                if not v:
                    cells.append("none")
                    continue
                pm, pmed = pcell[str(b)]
                rm = st.mean(v) / pm
                rmed = st.median(v) / pmed
                cell = f"mean={st.mean(v):.2f} med={st.median(v):.2f} (r={rm:.2f}/{rmed:.2f})"
                if rm > 2 or rmed > 2 or rm < 0.5 or rmed < 0.5:
                    cell += " FLAG"
                    flags.append((meth, arch, mode, b, round(rm, 2), round(rmed, 2)))
                cells.append(cell)
            print(f"{meth:8s} {arch:9s} {mode:12s} " + "  ".join(cells))

    print("\nFLAGGED cells (ratio > 2 or < 0.5):", len(flags))
    for meth, arch, mode, b, rm, rmed in flags:
        print(f"  {meth:7s} {arch:9s} {mode:12s} @{b}: ratio mean={rm} median={rmed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
