#!/usr/bin/env python3
"""B4e T1(b)/T6 — verify the D11 acos-clip override changes VALUES in EXACTLY one way: the clamp.

For ``--cgba_variant fixed`` the adapter's ``PatchedProposedAttack.SemiCircular_boundary_search`` must be a
verbatim copy of the official ``Proposed_attack.SemiCircular_boundary_search`` (proposed_attack.py:171-196) whose
ONLY value-changing edit wraps the acos argument at :182 in ``torch.clamp(..., -1.0, 1.0)``.  The D11 clip-event
counter (``clip_events_182``, T6) is allowed as an observation-only addition: it hoists the acos argument into an
``arg`` variable (verbatim copy of the official expression), tests ``arg < -1.0 or arg > 1.0`` and increments
``self.clip_events_182``, then computes ``theta = torch.acos(torch.clamp(arg, -1.0, 1.0))``.  None of those three
added lines changes any attack value; the clamp is the only value-changing edit.

For ``--cgba_variant d8only`` it must NOT override the method at all (the D8 count fix is inherited unchanged).

The comparison is done with ``inspect.getsource``, whitespace-normalised (each line stripped, empty lines dropped).
Assertions:
  (a) d8only inherits the official method (no override);
  (b) the override assigns ``arg =`` the official acos argument VERBATIM;
  (c) the override's theta line is ``theta = torch.acos(torch.clamp(arg, -1.0, 1.0))``;
  (d) the observation-only counter lines are present and are the only lines added besides the theta edit;
  (e) after removing those lines from both sides, every remaining line is byte-identical.

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


def _single(pred, lines, what: str):
    hits = [ln for ln in lines if pred(ln)]
    if len(hits) != 1:
        raise SystemExit(f"FAIL: expected exactly one {what!r} line, got {len(hits)}")
    return hits[0]


def main() -> int:
    official = inspect.getsource(Proposed_attack.SemiCircular_boundary_search)
    off_lines = norm_lines(official)

    # d8only: must inherit the official method (no override)
    D8 = A._patched_attack_class("d8only")
    ok_d8only = D8.SemiCircular_boundary_search is Proposed_attack.SemiCircular_boundary_search
    print(f"d8only: SemiCircular_boundary_search is official (not overridden) = {ok_d8only}")

    Fixed = A._patched_attack_class("fixed")
    override = inspect.getsource(Fixed.SemiCircular_boundary_search)
    ovr_lines = norm_lines(override)

    # (b) official acos argument must be assigned verbatim to `arg`
    off_theta = _single(lambda l: l.startswith("theta = torch.acos("), off_lines, "official theta")
    if not off_theta.endswith(")"):
        raise SystemExit("FAIL: cannot parse official acos argument")
    off_arg = off_theta[len("theta = torch.acos("):-1]
    ovr_arg = _single(lambda l: l.startswith("arg = "), ovr_lines, "override arg")
    if ovr_arg != "arg = " + off_arg:
        print(f"FAIL: override arg != official acos argument\n  official arg: {off_arg}\n  override arg: {ovr_arg[6:]}")
        return 1

    # (c) theta must be acos(clamp(arg))
    ovr_theta = _single(lambda l: l.startswith("theta = torch.acos("), ovr_lines, "override theta")
    expect_theta = "theta = torch.acos(torch.clamp(arg, -1.0, 1.0))"
    if ovr_theta != expect_theta:
        print(f"FAIL: override theta line is not the clamp form\n  expected: {expect_theta}\n  actual  : {ovr_theta}")
        return 1

    # (d) the observation-only counter lines are present
    if "if arg < -1.0 or arg > 1.0:" not in ovr_lines or "self.clip_events_182 += 1" not in ovr_lines:
        print("FAIL: missing observation-only clip-event counter lines")
        return 1

    # (e) everything else is byte-identical
    drop_off = {off_theta}
    drop_ovr = {ovr_theta, ovr_arg, "if arg < -1.0 or arg > 1.0:", "self.clip_events_182 += 1"}
    off_rest = [ln for ln in off_lines if ln not in drop_off]
    ovr_rest = [ln for ln in ovr_lines if ln not in drop_ovr]
    if off_rest != ovr_rest:
        print("FAIL: non-theta lines differ between official and override")
        import difflib
        for d in difflib.unified_diff(off_rest, ovr_rest, "official", "override", lineterm=""):
            print("  " + d)
        return 1

    print("PASS: exactly one value-changing edit (the clamp); counter lines are observation-only")
    print(f"  official theta: {off_theta}")
    print(f"  override arg  : {ovr_arg}")
    print(f"  override theta: {ovr_theta}")

    ok = ok_d8only and ovr_arg == "arg = " + off_arg and ovr_theta == expect_theta and off_rest == ovr_rest
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
