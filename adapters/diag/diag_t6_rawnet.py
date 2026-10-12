#!/usr/bin/env python3
"""T6: CGBA raw-net equivalence on E10/E11.

For rows 1-3 of each new victim, print max|victim(x0) - raw_net((x0-mu)/sd)| and argmax equality, with mu/sd
exactly as run_cgba.py passes them (C.MEAN/C.STD, _common.py:42-43).  Also print the RobustBench normalize
constants.

    python adapters/diag/diag_t6_rawnet.py
"""
from __future__ import annotations

import os
import sys

REPO = "/root/autodl-tmp/LGC_frozen"
sys.path.insert(0, os.path.join(REPO, "adapters"))
sys.path.insert(0, REPO)

import _common as C  # noqa: E402
import torch  # noqa: E402

VICTIMS = [
    ("resnet50_places365", "splits/places365/resnet50_places365_non_targeted.csv"),
    ("densenet161_places365", "splits/places365/densenet161_places365_non_targeted.csv"),
    ("Salman2020Do_R50", "splits/robust/Salman2020Do_R50_non_targeted.csv"),
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu = C.MEAN
    sd = C.STD
    print(f"mu/sd passed by run_cgba.py (C.MEAN/C.STD, _common.py:42-43): mu={list(mu)} sd={list(sd)}")
    for arch, split in VICTIMS:
        victim = C.load_victim_any(arch, device)
        raw = C.raw_net(victim)
        print(f"\n=== {arch} ===")
        rows = C.load_split_rows(os.path.join(REPO, split))[:3]
        for i, r in enumerate(rows):
            name = r["source"]
            x0 = C.load_x0_any(arch, name, device, data_dir="/root/autodl-tmp/ttba_mini_flat")
            with torch.no_grad():
                lv = victim(x0)
                lr = raw(C.normalize(x0))
            maxdiff = float((lv - lr).abs().max())
            same = bool((lv.argmax(1) == lr.argmax(1)).all())
            print(f"  row{i + 1} {name}: max|victim - raw_net| = {maxdiff:.3e}  argmax_equal={same}")

    # RobustBench normalize constants
    rb = C.load_victim_any("Salman2020Do_R50", device)
    if isinstance(rb, torch.nn.Sequential):
        kids = list(rb._modules.keys())
        if kids and kids[0] == "normalize":
            norm = rb.normalize
            mean = norm.mean if hasattr(norm, "mean") else None
            std = norm.std if hasattr(norm, "std") else None
            if hasattr(norm, "normalize"):
                mean = norm.normalize.mean
                std = norm.normalize.std
            print("\nRobustBench normalize constants:")
            print("  mean =", [round(float(x), 6) for x in (mean.flatten() if mean is not None else [])])
            print("  std  =", [round(float(x), 6) for x in (std.flatten() if std is not None else [])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
