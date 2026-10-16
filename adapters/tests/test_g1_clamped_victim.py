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

    # ---- B4h T2: oob_queries (observation-only, per image) ----
    print("[2] oob_queries: pixel-space (ClampedVictim) and normalised-space (ClampedVictimNormalized) counting")

    class FlatLogits(torch.nn.Module):
        def forward(self, z):
            return torch.zeros(z.shape[0], 2, device=z.device, dtype=z.dtype)

    rng2 = torch.Generator().manual_seed(1)
    px0 = torch.rand(1, 3, 8, 8, generator=rng2)                  # pixel [0,1], batch of 1

    # pixel space: guard sees xb directly (before ClampedVictim clamps it)
    trk = C.HardLabelTracker(C.ClampedVictim(FlatLogits()), px0, 0, budgets=[100], max_queries=100,
                             to_pixel=C.to_pixel_clamp)
    trk.labels(px0)
    check("pixel: in-range query -> oob 0", trk.oob_queries == 0)
    oob_hi = px0.clone(); oob_hi[0, 0, 0, 0] = 1.5
    trk.labels(oob_hi)
    check("pixel: coordinate > 1+1e-6 -> oob 1", trk.oob_queries == 1)
    oob_lo = px0.clone(); oob_lo[0, 1, 2, 3] = -1e-3
    trk.labels(oob_lo)
    check("pixel: coordinate < -1e-6 -> oob 2", trk.oob_queries == 2)

    # normalised space: guard un-normalises xb (xb*sd+mu, no clamp) before the [0,1] check
    lb = C.normalize(torch.zeros_like(px0))
    ub = C.normalize(torch.ones_like(px0))
    trk2 = C.HardLabelTracker(C.ClampedVictimNormalized(FlatLogits(), lb, ub), px0, 0,
                              budgets=[100], max_queries=100, to_pixel=C.to_pixel_unnormalize)
    trk2.labels(C.normalize(px0))
    check("norm: in-range query -> oob 0", trk2.oob_queries == 0)
    oob_hi_n = C.normalize(px0).clone(); oob_hi_n[0, 0, 0, 0] = 10.0   # pixel = 10*sd+mu >> 1
    trk2.labels(oob_hi_n)
    check("norm: un-normalised coordinate > 1+1e-6 -> oob 1", trk2.oob_queries == 1)
    oob_lo_n = C.normalize(px0).clone(); oob_lo_n[0, 2, 1, 0] = -10.0  # pixel = -10*sd+mu << -1e-6
    trk2.labels(oob_lo_n)
    check("norm: un-normalised coordinate < -1e-6 -> oob 2", trk2.oob_queries == 2)

    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
