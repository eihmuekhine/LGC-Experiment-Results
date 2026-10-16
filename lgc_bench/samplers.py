"""Probe distributions for normal-vector estimation and plane-axis sampling.

* GaussianSampler         : isotropic N(0, I) in the full space (LGC, HSJA).
* DCTLowFreqSampler       : N(0, I) on the top-left (H/f x W/f) block of the
                            orthonormal 2-D DCT, mapped back by IDCT
                            (official CGBA / TtBA, dimension-reduction factor f).
* JacobianSubspaceSampler : u = sqrt(k/r) V_r xi, xi ~ N(0, I_r), where V_r are
                            the top-r right singular vectors of the decoder
                            Jacobian at z_s (proposed; zero victim queries).
* TangentChordSampler     : build 32a Tier B: isotropic unit probes in the decoder-tangent
                            subspace (tangent_chord_basis, zero victim queries) mixed with
                            unit DCT probes projected onto its orthogonal complement.
* TAMaskAxisSampler       : official Triangle-Attack axis: a sparse random vector
                            on a few low-frequency DCT coefficients (1 per channel
                            by default) inside the top-left ratio*H block.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .env import CPURNG


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix D (n x n): X = D x, x = D^T X."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0, :] *= 1.0 / math.sqrt(n)
    d[1:, :] *= math.sqrt(2.0 / n)
    return d


class ProbeSampler:
    def sample(self, b: int, rng: CPURNG) -> torch.Tensor:  # (b, *shape)
        raise NotImplementedError

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of x (1, *shape) onto the span of the probes (identity if full-rank)."""
        return x


class GaussianSampler(ProbeSampler):
    def __init__(self, shape: tuple):
        self.shape = tuple(shape)

    def sample(self, b: int, rng: CPURNG) -> torch.Tensor:
        return rng.randn(b, *self.shape)


class DCTLowFreqSampler(ProbeSampler):
    """Low-frequency DCT noise for images of shape (C, H, W)."""

    def __init__(self, shape: tuple, factor: int, device: torch.device, normalize_to: Optional[float] = None):
        c, h, w = shape
        self.shape = (c, h, w)
        self.sh = max(1, int(h / factor))
        self.sw = max(1, int(w / factor))
        self.dh_t = torch.tensor(dct_matrix(h)[: self.sh, :].T, dtype=torch.float32, device=device)  # (H, sh)
        self.dw = torch.tensor(dct_matrix(w)[: self.sw, :], dtype=torch.float32, device=device)  # (sw, W)
        # forward transforms (for project / dct_block), stored explicitly
        self.dh = torch.tensor(dct_matrix(h)[: self.sh, :], dtype=torch.float32, device=device)  # (sh, H)
        self.dw_t = torch.tensor(dct_matrix(w)[: self.sw, :].T, dtype=torch.float32, device=device)  # (W, sw)
        self.normalize_to = normalize_to

    def idct_block(self, coeff: torch.Tensor) -> torch.Tensor:
        """coeff (b, C, sh, sw) -> image-domain (b, C, H, W)."""
        return self.dh_t @ coeff @ self.dw

    def dct_block(self, x: torch.Tensor) -> torch.Tensor:
        """image-domain (b, C, H, W) -> low-frequency coefficients (b, C, sh, sw) (orthonormal DCT)."""
        return self.dh @ x @ self.dw_t

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projection onto the low-frequency DCT block (the span of the probes)."""
        return self.idct_block(self.dct_block(x))

    def sample(self, b: int, rng: CPURNG) -> torch.Tensor:
        coeff = rng.randn(b, self.shape[0], self.sh, self.sw)
        u = self.idct_block(coeff)
        if self.normalize_to is not None:
            n = u.flatten(1).norm(dim=1).clamp_min(1e-12).view(b, 1, 1, 1)
            u = u / n * self.normalize_to
        return u


class JacobianSubspaceSampler(ProbeSampler):
    """Probes in span(V_r), optionally mixed with isotropic noise.

    u = sqrt(mix) * sqrt(k/r) V_r xi + sqrt(1 - mix) * g,   xi ~ N(0, I_r), g ~ N(0, I_k)
    so E||u||^2 = k for every mix (same sigma semantics as the Gaussian probes).
    mix = 1 -> pure subspace (fast when the normal lies in span(V_r));
    mix < 1 -> keeps an isotropic component (safe when it does not).
    """

    def __init__(self, V: torch.Tensor, shape: tuple, mix: float = 1.0):
        self.V = V  # (k, r), orthonormal columns, on device
        self.shape = tuple(shape)
        k, r = V.shape
        self.scale = math.sqrt(k / r)
        self.mix = float(min(1.0, max(0.0, mix)))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.mix < 1.0:  # isotropic component: the probes span the full space
            return x
        flat = x.reshape(-1)
        return (self.V @ (self.V.T @ flat)).view_as(x)

    def sample(self, b: int, rng: CPURNG) -> torch.Tensor:
        xi = rng.randn(b, self.V.shape[1])
        u = ((xi @ self.V.T) * (self.scale * math.sqrt(self.mix))).view(b, *self.shape)
        if self.mix < 1.0:
            u = u + math.sqrt(1.0 - self.mix) * rng.randn(b, *self.shape)
        return u


class TAMaskAxisSampler(ProbeSampler):
    """Official TA second axis (before orthogonalisation), in the image domain."""

    def __init__(self, shape: tuple, ratio_mask: float, dim_num: int, device: torch.device):
        c, h, w = shape
        self.shape = (c, h, w)
        self.size = max(1, int(h * ratio_mask))
        self.dim_num = int(dim_num)
        d = dct_matrix(h)
        self.d_t = torch.tensor(d.T, dtype=torch.float32, device=device)  # IDCT along H: D^T X
        self.dw = torch.tensor(dct_matrix(w), dtype=torch.float32, device=device)
        self.device = device

    def sample(self, b: int, rng: CPURNG) -> torch.Tensor:
        c, h, w = self.shape
        out = []
        n_pos = self.size * self.size
        for _ in range(b):
            coeff = torch.zeros(c, h, w, device=self.device)
            vals = rng.randn(c, self.dim_num)
            for ch in range(c):  # different random positions per channel (mask1/2/3)
                for j, pos in enumerate(rng.choice(n_pos, self.dim_num).tolist()):
                    ri, ci = divmod(int(pos), self.size)
                    coeff[ch, ri, ci] = vals[ch, j]
            out.append(self.d_t @ coeff @ self.dw)
        return torch.stack(out, 0)


# ---------------------------------------------------------------------------
# Build 32a Tier B: floor-free latent tangent-chord probing (LGC_T family)
# ---------------------------------------------------------------------------


@torch.no_grad()
def tangent_chord_basis(decode: nn.Module, z_b: torch.Tensor, rank: int = 256, oversample: int = 16,
                        eps: float = 1e-2, batch: int = 16) -> torch.Tensor:
    """Orthonormal pixel-space basis (shape: pixel_dim x rank) spanning
    {G(z_b + eps*xi) - G(z_b)}, via SVD of (rank+oversample) finite-difference columns.

    decode: the autoencoder's G(.).  No calls to the victim oracle.  Called once per
    `jac_reanchor_every` iterations (reuse the existing Jacobian-reanchor cadence/config field
    from the LGC_JS path, do not add a second cadence parameter).

    Cadence (32b decision): the basis is built once per image at z_s = E(x0) by the runner (the
    LGC_JS convention, cached_subspace); diagnose_tangent.py measures whether anchoring at
    E(x_b) matters.  The caller seeds torch (derive_seed in the runner) for reproducibility.

    Build 32b fix (memory): 32a called torch.linalg.svd(C) with the default full_matrices=True on the
    (pixel_dim x l) chord matrix, which allocates a pixel_dim x pixel_dim U (150528^2 float64 = 181 GB
    at 3x224x224) and cannot run on real images.  32b uses a thin QR on the device followed by an SVD
    of the small l x l factor:  C = Q0 R,  R = U_r S V^T  =>  C = (Q0 U_r) S V^T,  so the top-r left
    singular vectors of C are Q0 U_r[:, :r].  Memory O(pixel_dim * l), time O(pixel_dim * l^2).
    """
    decode.eval()
    device = z_b.device
    z_shape = tuple(z_b.shape[1:])
    z_dim = int(np.prod(z_shape))
    l = min(z_dim, int(rank) + int(oversample))
    if int(rank) > l:
        raise ValueError(f"tangent_chord_basis: rank {rank} exceeds the latent dimension {z_dim}")
    cols = torch.randn(z_dim, l, device=device)
    cols = cols / cols.norm(dim=0).clamp_min(1e-12)  # unit directions (jacobian.py convention)
    g0 = decode(z_b).flatten(1)  # G(z_b), computed once
    out = []
    i = 0
    while i < l:
        b = min(batch, l - i)
        d = cols[:, i:i + b].T.reshape(b, *z_shape)
        yp = decode(z_b + eps * d).flatten(1)
        out.append(((yp - g0) / eps).float())  # forward finite difference {G(z_b + eps xi) - G(z_b)} / eps
        i += b
    c = torch.cat(out, 0).T.contiguous()  # (pixel_dim, l)
    q0, r_fac = torch.linalg.qr(c, mode="reduced")  # (pixel_dim, l), (l, l)
    u_r, _, _ = torch.linalg.svd(r_fac.detach().cpu().double(), full_matrices=False)  # l x l, float64 on CPU
    q = q0 @ u_r[:, : int(rank)].to(dtype=q0.dtype, device=q0.device)
    return q.contiguous()


class TangentChordSampler(ProbeSampler):
    """Mixes isotropic probes in a decoder-tangent subspace (0 victim queries to build) with
    DCT probes on its orthogonal complement in pixel space. See claude/next_design_32a_2026-10-02.md
    Tier B for the derivation of e_tan / e_perp and the c-hat formula.

    sample(n) returns n probe directions in pixel space, split n_tan = round(n*n_tan_frac)
    isotropic in span(tangent_basis), n - n_tan projected DCT onto the orthogonal complement.
    The first n_tan directions lie in span(Q), the rest in its orthogonal complement;
    ``last_n_tan`` records the split of the most recent call so the engine can form the two
    partial sign sums separately (weighted estimator, build 32b).

    Build 32b fix (probe scale): 32a returned UNIT-norm directions, so with the engine's
    ``center + sigma * u`` and sigma = 2e-4 the probe L2 was 2e-4 instead of TtBA's
    3e-4 * sqrt(d_S) = 0.029; the normal displacement then falls far below the base-point depth and
    the signs become one-sided (Remark "probe depth" in proposed_methods_vs_CGBA_TA_TtBA.tex;
    simulation sim_tierb.py: sign share 1.00, c^2 ratio 0.01-0.24).  32b rescales BOTH families to
    ``scale`` (default: the DCT sampler's normalize_to, i.e. sqrt(d_S) for TtBA-normalised probes,
    else sqrt(d_S) as well, which is the expected norm of raw DCT probes), so the per-probe semantics
    of sigma are exactly those of the DCT path it replaces.
    """

    def __init__(self, tangent_basis: torch.Tensor, dct_sampler: "DCTLowFreqSampler",
                 n_tan_frac: float, device: torch.device, scale: Optional[float] = None):
        self.Q = tangent_basis  # (pixel_dim, r), orthonormal columns, on device
        self.dct = dct_sampler
        self.shape = tuple(dct_sampler.shape)
        self.frac = float(min(1.0, max(0.0, n_tan_frac)))
        self.device = device
        self.r = int(tangent_basis.shape[1])
        self.d_block = int(self.shape[0] * dct_sampler.sh * dct_sampler.sw)  # DCT-f block dimension
        self.d_s = max(1, self.d_block - self.r)  # complement-family dimension (span(Q) mostly inside the block)
        if scale is None:
            scale = dct_sampler.normalize_to if dct_sampler.normalize_to is not None else math.sqrt(self.d_block)
        self.scale = float(scale)
        self.last_n_tan = 0

    def sample(self, n: int, rng: Optional[CPURNG] = None) -> torch.Tensor:
        """(n, *shape) probes: first n_tan in span(Q), rest in Q^perp; every probe has norm ``scale``.
        rng is the engine's CPURNG (rng=None falls back to torch.randn, used by diagnose_tangent.py)."""
        n = int(n)
        n_tan = min(n, max(0, int(round(n * self.frac))))
        out = []
        if n_tan > 0:
            if rng is not None:
                xi = rng.randn(n_tan, self.Q.shape[1])
            else:
                xi = torch.randn(n_tan, self.Q.shape[1], device=self.device)
            u_tan = xi @ self.Q.T  # (n_tan, pixel_dim), isotropic in span(Q)
            u_tan = u_tan / u_tan.norm(dim=1).clamp_min(1e-12).view(-1, 1)
            out.append(u_tan * self.scale)
        n_perp = n - n_tan
        if n_perp > 0:
            if rng is not None:
                u_dct = self.dct.sample(n_perp, rng)
            else:
                coeff = torch.randn(n_perp, self.shape[0], self.dct.sh, self.dct.sw, device=self.device)
                u_dct = self.dct.idct_block(coeff)
            u_perp = u_dct.flatten(1) - (u_dct.flatten(1) @ self.Q) @ self.Q.T  # onto Q^perp
            u_perp = u_perp / u_perp.norm(dim=1).clamp_min(1e-12).view(-1, 1)
            out.append(u_perp * self.scale)
        self.last_n_tan = n_tan
        return torch.cat(out, 0).view(n, *self.shape)
