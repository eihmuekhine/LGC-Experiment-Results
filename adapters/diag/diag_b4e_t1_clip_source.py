#!/usr/bin/env python3
"""B4e T1(b) — verify the D11 acos-clip override differs from the official method in EXACTLY one line.

For ``--cgba_variant fixed`` the adapter's ``PatchedProposedAttack.SemiCircular_boundary_search`` must be a
verbatim copy of the official ``Proposed_attack.SemiCircular_boundary_search`` (proposed_attack.py:171-196) whose
ONLY change wraps the acos argument at :182 in ``torch.clamp(..., -1.0, 1.0)``.  For ``--cgba_variant d8only`` it
must NOT override the method at all (the D8 count fix is inherited unchanged).

The comparison is done with ``inspect.getsource``, whitespace-normalised (each line stripped, empty lines dropped);
the diff must show exactly one changed line, and that line must be the acos line.

    python adapters/diag/diag_b4e_t1_clip_source.py
"""
from __future__ import annotations

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))
sys.path.insert(0, "/root/autodl-tmp/baselines/CGBA")

import _common as C  # noqa: E402
import run_cgba as A  # noqa: E402
from proposed_attack import Proposed_attack  # noqa: E402


def norm_lines(src: str):
    return [ln.strip() for ln in src.splitlines() if ln.strip()]


def main() -> int:
    official = inspect.getsource(Proposed_attack.SemiCircular_boundary_search)
    off_lines = norm_lines(official)

    # d8only: must inherit the official method (no override)
    D8 = A._patched_attack_class("d8only")
    ok_d8only = D8.SemiCircular_boundary_search is Proposed_attack.SemiCircular_boundary_search
    print(f"d8only: SemiCircular_boundary_search is official (not overridden) = {ok_d8only}")

    # fixed: exactly one changed line, the acos line
    Fixed = A._patched_attack_class("fixed")
    override = inspect.getsource(Fixed.SemiCircular_boundary_search)
    ovr_lines = norm_lines(override)

    if len(off_lines) != len(ovr_lines):
        print(f"FAIL: line count differs official={len(off_lines)} override={len(ovr_lines)}")
        return 1

    diff = [(i, o, v) for i, (o, v) in enumerate(zip(off_lines, ovr_lines)) if o != v]
    if len(diff) != 1:
        print(f"FAIL: expected exactly 1 changed line, got {len(diff)}")
        for i, o, v in diff:
            print(f"  line {i}:\n    official : {o}\n    override : {v}")
        return 1

    i, o_line, v_line = diff[0]
    is_acos = ("torch.acos" in o_line) and ("torch.acos" in v_line) and ("torch.clamp" in v_line)
    if not is_acos:
        print(f"FAIL: the changed line is not the acos line:\n  official: {o_line}\n  override: {v_line}")
        return 1

    print(f"PASS: exactly one changed line (index {i}), the acos line")
    print(f"  official: {o_line}")
    print(f"  override: {v_line}")

    ok = ok_d8only and len(diff) == 1 and is_acos
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
