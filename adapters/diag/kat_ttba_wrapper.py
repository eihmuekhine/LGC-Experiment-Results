#!/usr/bin/env python3
"""B4g T1 helper: run the OFFICIAL TtBA adapter (adapters/run_ttba_v2.py) on the ``kat_linear`` victim.

``run_ttba_v2.py`` is not in the B4g allowed-edit list, so its ``--arch`` argparse choices cannot be extended to
``kat_linear``.  This wrapper imports the adapter and calls its generic (E10/E11) entry point ``_main_generic``
unchanged, with ``arch="kat_linear"`` — the exact same code path that runs the official ``TtBA.Attacker`` through
``C.load_victim_any`` / ``C.load_x0_any`` and the HardLabelTracker.  Nothing in the attack algorithm is changed.

Usage (from the worktree root):
    KAT_FORWARD_COUNT_FILE=<path> python adapters/diag/kat_ttba_wrapper.py \
        --targeted {0,1} --split_csv splits/kat/kat_linear_<mode>.csv \
        --out_dir generated/smoke_canary/kat_runs/TtBA --limit 5
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import run_ttba_v2  # noqa: E402  (imports _common -> configure_precision runs at import -> D12 flags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targeted", type=int, choices=[0, 1], required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--epsilon", type=float, default=2.5)
    a = ap.parse_args()
    ns = SimpleNamespace(arch="kat_linear", targeted=a.targeted, split_csv=a.split_csv, out_dir=a.out_dir,
                         budget=a.budget, epsilon=a.epsilon, seed_mode="per_image", limit=a.limit,
                         ttba_repo="/root/autodl-tmp/TtBA")
    return run_ttba_v2._main_generic(ns)


if __name__ == "__main__":
    raise SystemExit(main())
