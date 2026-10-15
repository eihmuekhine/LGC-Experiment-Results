"""G1 gate (D19): ClampedVictimNormalized clamps a NORMALISED-space input channel-wise to [lb, ub] before the raw
net, and agrees with C.normalize(C.to_pixel_unnormalize(x)) (the same [0,1]-clamped pixel image the tracker stores).

CPU only, no GPU.  Run:  python adapters/tests/test_g1_clamped_victim.py   (exit 0 iff all pass)
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))   # adapters/ (holds _common.py)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import _common as C  # noqa: E402  (configure_precision runs at import -> D12 flags, CPU-safe)

RESULTS = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


class Recorder(torch.nn.Module):
    """Raw net that records the (already clamped) input it receives and returns it unchanged."""

    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, x):
        self.last = x.clone()
        return x


def main() -> int:
    print("[1] ClampedVictimNormalized: channel-wise clamp to [lb, ub]; == normalize(to_pixel_unnormalize(x))")
    rng = torch.Generator().manual_seed(0)
    x0 = torch.rand(2, 3, 8, 8, generator=rng)                    # [0,1] pixel tensor (batch of 2)
    lb = C.normalize(torch.zeros_like(x0))
    ub = C.normalize(torch.ones_like(x0))
    raw = Recorder()
    cvn = C.ClampedVictimNormalized(raw, lb, ub)

    # out-of-box normalised input: values far outside [lb, ub]
    x = C.normalize(x0) + 6.0 * torch.randn(2, 3, 8, 8, generator=rng)
    check("input is out of box (some element < lb or > ub)",
          bool(((x < lb) | (x > ub)).any()))

    out = cvn(x)
    clamped = torch.clamp(x, lb, ub)
    check("raw net received torch.clamp(x, lb, ub)", bool((raw.last - clamped).abs().max() < 1e-7))
    check("lb/ub are per-channel constants (channel-wise clamp)",
          bool((lb - lb[:, :, :1, :1]).abs().max() < 1e-6) and bool((ub - ub[:, :, :1, :1]).abs().max() < 1e-6))

    # the D19 identity: normalize(to_pixel_unnormalize(x)) == clamp(x, lb, ub) (up to FP round-off)
    via_pixel = C.normalize(C.to_pixel_unnormalize(x))
    check("clamp(x, lb, ub) == normalize(to_pixel_unnormalize(x)) (allclose)",
          bool(torch.allclose(clamped, via_pixel, atol=1e-6, rtol=1e-5)))
    check("ClampedVictimNormalized(x) == raw_net(normalize(to_pixel_unnormalize(x)))",
          bool(torch.allclose(out, raw(via_pixel), atol=1e-6, rtol=1e-5)))

    # a fully in-range normalised input passes through unchanged
    x_in = C.normalize(x0)
    cvn(x_in)
    check("in-range input passes through unchanged", bool((raw.last - x_in).abs().max() < 1e-7))

    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
