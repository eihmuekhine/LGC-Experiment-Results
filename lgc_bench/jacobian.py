r"""Top-r right-singular subspace of the decoder Jacobian J = dG/dz at z_s.

Costs ZERO victim queries: only the attacker-owned decoder is evaluated.

Randomised subspace iteration on M = J^T J:
    Q_0 = orth(Omega),  Omega ~ N(0, I_k) of size k x l  (l = r + oversample)
    repeat (power_iters + 1) times:  Q <- orth(J^T (J Q))
    Rayleigh-Ritz:  B = J Q,  B^T B = W diag(s^2) W^T,  V_r = Q W[:, :r]

* J v   : central finite differences, batched forward passes (backend-safe on MPS)
* J^T u : batched reverse-mode VJP (standard autograd)
* QR / eigh are done in float64 on the CPU (MPS has no float64).
"""

from __future__ import annotations

import os
import time
from typing import Tuple

import torch
import torch.nn as nn

from .env import CPURNG, empty_cache, is_oom


def _jvp_fd(decoder: nn.Module, z_s: torch.Tensor, cols: torch.Tensor, h: float, batch: int) -> torch.Tensor:
    """J @ cols via central differences. cols: (k, l) -> returns (l, m)."""
    k, l = cols.shape
    shape = z_s.shape[1:]
    out = []
    i = 0
    with torch.no_grad():
        while i < l:
            b = min(batch, l - i)
            d = cols[:, i : i + b].T.reshape(b, *shape)
            try:
                yp = decoder(z_s + h * d).flatten(1)
                ym = decoder(z_s - h * d).flatten(1)
            except RuntimeError as err:
                if is_oom(err) and batch > 1:
                    batch = max(1, batch // 2)
                    empty_cache(z_s.device)
                    continue
                raise
            out.append((yp - ym) / (2.0 * h))
            i += b
    return torch.cat(out, 0)


def _vjp(decoder: nn.Module, z_s: torch.Tensor, rows: torch.Tensor, batch: int) -> torch.Tensor:
    """J^T @ rows^T. rows: (l, m) -> returns (k, l). Requires decoder in eval mode."""
    l = rows.shape[0]
    shape = z_s.shape[1:]
    out = []
    i = 0
    while i < l:
        b = min(batch, l - i)
        try:
            with torch.enable_grad():
                z = z_s.detach().expand(b, *shape).clone().requires_grad_(True)
                y = decoder(z).flatten(1)
                g = torch.autograd.grad((y * rows[i : i + b]).sum(), z)[0]
        except RuntimeError as err:
            if is_oom(err) and batch > 1:
                batch = max(1, batch // 2)
                empty_cache(z_s.device)
                continue
            raise
        out.append(g.flatten(1).detach())
        i += b
    return torch.cat(out, 0).T


def _orth_cpu(a: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(a.detach().cpu().double())
    return q.float().to(a.device)


def decoder_right_subspace(
    decoder: nn.Module,
    z_s: torch.Tensor,
    rank: int = 256,
    oversample: int = 16,
    power_iters: int = 1,
    batch: int = 8,
    fd_eps: float = 1e-2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return V (k x r, orthonormal, on device) and singular values s (r,) on device."""
    decoder.eval()
    device = z_s.device
    k = z_s.numel()
    rank = min(rank, k)
    l = min(k, rank + oversample)
    rng = CPURNG(seed, device)
    q = _orth_cpu(rng.randn(k, l))
    for _ in range(power_iters + 1):
        y = _jvp_fd(decoder, z_s, q, fd_eps, batch)  # (l, m)
        z = _vjp(decoder, z_s, y, batch)  # (k, l) = J^T J Q
        q = _orth_cpu(z)
    b = _jvp_fd(decoder, z_s, q, fd_eps, batch)  # (l, m) rows = (J q_i)^T
    gram = (b @ b.T).detach().cpu().double()  # (l, l) = Q^T J^T J Q
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)[:rank]
    w = evecs[:, order].float().to(device)
    s = evals[order].clamp_min(0.0).sqrt().float().to(device)
    v = q @ w
    return v.contiguous(), s.contiguous()


def decoder_diag_gain(
    decoder: nn.Module,
    z_s: torch.Tensor,
    n_probes: int = 64,
    batch: int = 8,
    seed: int = 0,
) -> torch.Tensor:
    """Hutchinson estimate of diag(J^T J): d_j = E_u[(J^T u)_j^2], u ~ N(0, I_m).

    d_j is the pixel energy produced by a unit change of latent coordinate j
    (its 'decoder gain'). Zero victim queries; n_probes batched VJPs.
    """
    decoder.eval()
    m = int(decoder(z_s).numel())
    rng = CPURNG(seed, z_s.device)
    acc = torch.zeros(z_s.numel(), device=z_s.device)
    done = 0
    while done < n_probes:
        b = min(batch, n_probes - done)
        g = _vjp(decoder, z_s, rng.randn(b, m), batch)  # (k, b)
        acc += (g * g).sum(1)
        done += b
    return (acc / n_probes).view_as(z_s)


def cached_diag_gain(cache_dir: str, key: str, decoder: nn.Module, z_s: torch.Tensor,
                     n_probes: int, batch: int, seed: int, verbose: bool = True) -> torch.Tensor:
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{key}_diaggain_T{n_probes}.pt")
    if os.path.exists(fname):
        return torch.load(fname, map_location="cpu").to(z_s.device)
    t0 = time.time()
    d = decoder_diag_gain(decoder, z_s, n_probes, batch, seed)
    torch.save(d.cpu(), fname)
    if verbose:
        q = torch.quantile(d.flatten().cpu(), torch.tensor([0.05, 0.5, 0.95]))
        print(f"      [gain] diag(J^T J) with {n_probes} probes in {time.time() - t0:.1f}s | "
              f"p5={q[0]:.3g} median={q[1]:.3g} p95={q[2]:.3g}")
    return d


def cached_subspace(
    cache_dir: str,
    key: str,
    decoder: nn.Module,
    z_s: torch.Tensor,
    rank: int,
    oversample: int,
    power_iters: int,
    batch: int,
    fd_eps: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Disk-cached subspace. Depends only on the AE and the image, not on the victim."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{key}_r{rank}_o{oversample}_p{power_iters}_h{fd_eps:g}.pt")
    if os.path.exists(fname):
        blob = torch.load(fname, map_location="cpu")
        return blob["V"].to(z_s.device), blob["S"].to(z_s.device)
    t0 = time.time()
    v, s = decoder_right_subspace(decoder, z_s, rank, oversample, power_iters, batch, fd_eps, seed)
    torch.save({"V": v.cpu(), "S": s.cpu()}, fname)
    if verbose:
        print(
            f"      [jacobian] r={rank} computed in {time.time() - t0:.1f}s | "
            f"sigma_1={float(s[0]):.3g} sigma_r={float(s[-1]):.3g}"
        )
    return v, s
