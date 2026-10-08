"""Build 36: manifold-ambient state space for MA-PVRE (design doc s16 A, theory doc Theorems 1-3).

State z = [xi, p] stored as ONE flat tensor of shape (B, D_lat + D_pix) so that every engine operation (probes,
bisection, finite differences, uint8 repair, checkpoints) works unchanged:

    x(z) = clip( base.to_image(xi) + p, 0, 1 ),     origin = [xi_s, 0],

where `base` is any lgc_bench space (LatentResidualSpace for the LGC rows, PixelSpace in the selftest).  xi is the
latent (semantic) block, p the ambient pixel block.  The MA-PVRE update (engine `_step_mapvre`) moves ONE block
per iteration: the latent block with p frozen (pull-back chart of span(v_xi, w_xi)), or the ambient block with xi
frozen in the exact pixel plane span(e, n_pix) through x0 (radially coupled, Theorem 3: no floor).
With p = 0 the space reproduces `base` exactly (selftest_36 B3: LGC_RC_MA_X == LGC_RC_PVRE bit for bit).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class AmbientResidualSpace:
    name = "latent_ra"

    def __init__(self, base) -> None:
        self.base = base
        o = base.origin
        self.base_shape = tuple(o.shape[1:])  # per-sample latent shape of the base space
        self.d_lat = int(o[0].numel())
        x0 = base.x0
        self.x0 = x0
        self.pix_shape = tuple(x0.shape[1:])  # (C, H, W)
        self.d_pix = int(x0[0].numel())
        self.dim = self.d_lat + self.d_pix
        self.shape = (self.dim,)
        self.origin = self.join(o, torch.zeros_like(x0))
        tp = getattr(base, "target_point", None)
        self.target_point = None if tp is None else self.join(tp, torch.zeros_like(x0))

    # ---------------------------------------------------------------- packing
    def split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, D) -> xi (B, *base_shape), p (B, C, H, W)."""
        b = z.shape[0]
        zf = z.reshape(b, -1)
        return zf[:, : self.d_lat].reshape(b, *self.base_shape), zf[:, self.d_lat:].reshape(b, *self.pix_shape)

    def join(self, xi: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        b = xi.shape[0]
        if p.shape[0] != b:
            p = p.expand(b, *p.shape[1:])
        return torch.cat([xi.reshape(b, -1), p.reshape(b, -1).to(dtype=xi.dtype, device=xi.device)], dim=1)

    def lift(self, xi_noise: torch.Tensor) -> torch.Tensor:
        """Latent-only vector (B, *base_shape) -> full state direction with a zero ambient block."""
        return self.join(xi_noise, torch.zeros((xi_noise.shape[0],) + self.pix_shape, dtype=xi_noise.dtype,
                                               device=xi_noise.device))

    def lift_pix(self, p_noise: torch.Tensor) -> torch.Tensor:
        """Pixel-only vector (B, C, H, W) -> full state direction with a zero latent block."""
        b = p_noise.shape[0]
        return self.join(torch.zeros((b,) + self.base_shape, dtype=p_noise.dtype, device=p_noise.device), p_noise)

    # ---------------------------------------------------------------- space API used by the engine / runner
    def _base_image(self, xi: torch.Tensor) -> torch.Tensor:
        out = self.base.to_image(xi)
        if xi.requires_grad and not out.requires_grad:  # base.to_image decorated with @torch.no_grad()
            raw = getattr(getattr(type(self.base), "to_image", None), "__wrapped__", None)
            if raw is not None:
                out = raw(self.base, xi)
        return out

    def to_image(self, z: torch.Tensor) -> torch.Tensor:
        xi, p = self.split(z)
        return torch.clamp(self._base_image(xi) + p.to(dtype=self.x0.dtype), 0.0, 1.0)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        xi, p = self.split(z)
        return self.join(self.base.project(xi), p)

    def latent_distance(self, z: torch.Tensor) -> float:
        xi, _ = self.split(z)
        return float((xi - self.base.origin).flatten().norm())

    def ambient_share(self, z: torch.Tensor) -> float:
        """||p|| / ||x(z) - x0||: share of the perturbation carried by the ambient block."""
        _, p = self.split(z)
        e = float((self.to_image(z) - self.x0).flatten().norm())
        return float(p.flatten().norm()) / max(e, 1e-12)


def wrap_target(base_target: Optional[torch.Tensor], space: AmbientResidualSpace) -> Optional[torch.Tensor]:
    return None if base_target is None else space.join(base_target, torch.zeros_like(space.x0))
