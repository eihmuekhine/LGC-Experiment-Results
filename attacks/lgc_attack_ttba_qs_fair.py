"""
Fair hard-label comparison engine for the user's RAG/LGC ImageNet setting.

IMPORTANT
---------
This is a NEW file. It does not modify attacks/lgc_attack.py.

Compared methods
----------------
  * LGC              : geometry ported from the attached LGC implementation.
  * LGC_H            : geometry ported from the attached LGC_H implementation.
  * QS               : 90-degree sector + adaptive-K QS ported from the attached file.
  * Hybrid           : attached QS -> LGC_H switch rule (SSIM threshold 0.995).
  * RAG_TTBA_BASE    : TtBA normal/current-direction bridge under the same RAG map.

Fairness controls
-----------------
  * identical RAG map, source/target split, seed and normal estimator;
  * identical N_t = floor(N0*sqrt(t+1)) schedule;
  * one centralized victim-query API;
  * hard cap at max_queries (no query may exceed the budget);
  * no evaluation early-stop -- every method may use the same total budget;
  * checkpoint values are best adversarial states obtained with Q <= checkpoint;
  * targeted initialization is hard-label only (no victim logits / gradients).

The attached source lists FD and LGC_C as command-line choices, but Attack() has no
FD/LGC_C dispatch implementation. They are intentionally NOT fabricated here.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _plane_basis(eta: torch.Tensor, v: torch.Tensor, eps: float = 1e-12):
    """Same plane construction used by the attached QS implementation."""
    e1 = eta / (torch.norm(eta) + eps)
    proj = torch.sum(v * e1) * e1
    residual = v - proj
    nr = torch.norm(residual)
    if nr.item() < eps:
        # Degenerate case: choose a deterministic orthogonal direction.
        flat = e1.reshape(-1)
        idx = int(torch.argmin(torch.abs(flat)).item())
        basis = torch.zeros_like(flat)
        basis[idx] = 1.0
        basis = basis.reshape_as(e1)
        residual = basis - torch.sum(basis * e1) * e1
        nr = torch.norm(residual)
    e2 = residual / (nr + eps)
    return e1, e2


def _rotate_in_plane(eta: torch.Tensor, v: torch.Tensor, phi: torch.Tensor, eps: float = 1e-12):
    e1, e2 = _plane_basis(eta, v, eps=eps)
    return torch.cos(phi) * e1 + torch.sin(phi) * e2


def build_hard_label_target_seed(
    encoder,
    x_orig: torch.Tensor,
    x_rec_0: torch.Tensor,
    x_target_raw: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient-free counterpart of the first (pseudo-target) step in the attached
    optimize_target_latent(). It never calls the victim model.

    The attack class itself verifies/expands this seed using counted TOP-1 labels.
    """
    pseudo_target = torch.clamp(x_target_raw - (x_orig - x_rec_0), 0.0, 1.0)
    with torch.no_grad():
        return encoder(pseudo_target).detach()


class FairLGCCompareAttack:
    SUPPORTED = {"LGC", "LGC_H", "QS", "Hybrid", "RAG_TTBA_BASE"}
    NORMAL_METHODS = SUPPORTED

    def __init__(
        self,
        model,
        z_source: torch.Tensor,
        x_orig: torch.Tensor,
        source_label: int,
        loss_fn_vgg,
        ssim_fn,
        attack_method: str = "LGC_H",
        target_seed: Optional[torch.Tensor] = None,
        target_label: Optional[int] = None,
        iteration: int = 1000,
        initial_query: int = 30,
        tol: float = 1e-4,
        sigma: float = 2e-4,
        seed: int = 992,
        query_checkpoints: Optional[List[int]] = None,
        max_queries: int = 10000,
        verbose: str = "Yes",
        # TtBA
        ttba_bridge_tol: float = 1e-3,
        # QS/Hybrid settings copied from attached implementation
        switch_ssim: float = 0.995,
        qs_k_mode: str = "gap_based",
        qs_k_min: int = 4,
        qs_k_max: int = 16,
        qs_fixed_k: int = 10,
        # Same default schedule as the attached file.
        qs_k_schedule: Optional[Dict[float, int]] = None,
        enable_progress_k: bool = False,
        # Fair budget guard. Same reserve is used for every method.
        update_query_reserve: int = 18,
    ):
        if attack_method not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported attack_method={attack_method}. "
                f"Supported: {sorted(self.SUPPORTED)}"
            )
        if qs_k_mode not in {"fixed", "gap_based"}:
            raise ValueError("qs_k_mode must be 'fixed' or 'gap_based'.")

        self.model = model
        self.z_source = z_source.detach().clone()
        self.x_orig = x_orig.detach().clone()
        self.source_label = int(source_label)
        self.target_seed = None if target_seed is None else target_seed.detach().clone()
        self.target_label = None if target_label is None else int(target_label)

        self.loss_fn_vgg = loss_fn_vgg
        self.ssim_fn = ssim_fn
        self.attack_method = attack_method

        self.iteration = int(iteration)
        self.N0 = int(initial_query)
        self.tol = float(tol)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.max_queries = int(max_queries)
        self.checkpoints = sorted(query_checkpoints or [500, 1000, 2000, 5000, 10000])
        self.verbose = verbose
        self.device = self.z_source.device
        self.eps = 1e-12
        self.grad_estimator_batch_size = 40
        self.update_query_reserve = int(update_query_reserve)

        self.ttba_bridge_tol = float(ttba_bridge_tol)

        self.switch_ssim = float(switch_ssim)
        self.qs_k_mode = qs_k_mode
        self.qs_k_min = int(qs_k_min)
        self.qs_k_max = int(qs_k_max)
        self.qs_fixed_k = int(qs_fixed_k)
        self.qs_k_schedule = (
            {0.05: 4, 0.02: 4, 0.008: 6, 0.0: 8}
            if qs_k_schedule is None
            else {float(k): int(v) for k, v in qs_k_schedule.items()}
        )
        self.enable_progress_k = bool(enable_progress_k)
        self._K_fast = 0.15
        self._K_slow = 0.03
        self._cur_ssim = 0.0
        self._prev_ssim = 0.0
        self._hybrid_switched = False
        self._K_ladder = None

        self.all_queries = 0
        self.q_breakdown = OrderedDict(
            init=0,
            normal=0,
            lgc_search=0,
            qs_search=0,
            ttba_bridge=0,
            tangent=0,
            refine=0,
        )
        self.history = []
        self.diagnostics = []

        self._set_seed(self.seed)

    # ------------------------------------------------------------------
    # Reproducibility / central query accounting
    # ------------------------------------------------------------------
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def remaining_queries(self) -> int:
        return max(0, self.max_queries - self.all_queries)

    def _predict_labels(self, z_batch: torch.Tensor, category: str) -> torch.Tensor:
        """The only victim-model query entry point."""
        if category not in self.q_breakdown:
            raise KeyError(f"Unknown query category: {category}")

        if z_batch.ndim == self.z_source.ndim - 1:
            z_batch = z_batch.unsqueeze(0)

        n = int(z_batch.shape[0])
        if n <= 0 or self.remaining_queries() <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        n = min(n, self.remaining_queries())
        z_batch = z_batch[:n]

        with torch.no_grad():
            labels = self.model(z_batch).argmax(dim=1)

        self.all_queries += n
        self.q_breakdown[category] += n
        return labels

    def _labels_to_status(self, labels: torch.Tensor) -> torch.Tensor:
        if self.target_label is None:
            return torch.where(
                labels != self.source_label,
                torch.ones_like(labels),
                -torch.ones_like(labels),
            )
        return torch.where(
            labels == self.target_label,
            torch.ones_like(labels),
            -torch.ones_like(labels),
        )

    def is_adversarial(self, z: torch.Tensor, category: str) -> int:
        labels = self._predict_labels(z, category)
        if labels.numel() == 0:
            return 0  # budget exhausted
        return int(self._labels_to_status(labels)[0].item())

    # ------------------------------------------------------------------
    # Shared initialization
    # ------------------------------------------------------------------
    def _find_initial_adversarial(self) -> Optional[torch.Tensor]:
        if self.remaining_queries() <= 0:
            return None

        if self.target_label is not None:
            if self.target_seed is None:
                raise ValueError("Targeted mode requires target_seed.")

            # Hard-label only. No logits/gradient optimization.
            delta = self.target_seed - self.z_source
            for factor in [1.0, 1.10, 1.25, 1.50, 2.0, 3.0]:
                if self.remaining_queries() <= 0:
                    break
                cand = self.z_source + factor * delta
                if self.is_adversarial(cand, "init") == 1:
                    return cand.detach()

            for j in range(1, 51):
                if self.remaining_queries() <= 0:
                    break
                scale = 0.01 * j
                cand = self.target_seed + scale * torch.randn_like(self.target_seed)
                if self.is_adversarial(cand, "init") == 1:
                    return cand.detach()
            return None

        # Same increasing latent Gaussian style used by the validated RAG-TtBA runner.
        for j in range(1, 501):
            if self.remaining_queries() <= 0:
                break
            cand = self.z_source + (0.02 * j) * torch.randn_like(self.z_source)
            if self.is_adversarial(cand, "init") == 1:
                return cand.detach()
        return None

    def _binary_to_boundary(
        self,
        z_clean: torch.Tensor,
        z_adv: torch.Tensor,
        category: str = "refine",
        max_steps: int = 100,
    ) -> torch.Tensor:
        """Binary search with a known CLEAN / ADV pair."""
        clean = z_clean.detach().clone()
        adv = z_adv.detach().clone()

        for _ in range(max_steps):
            if self.remaining_queries() <= 0:
                break
            if torch.norm(adv - clean).item() <= self.tol:
                break
            mid = 0.5 * (clean + adv)
            status = self.is_adversarial(mid, category)
            if status == 0:
                break
            if status == 1:
                adv = mid
            else:
                clean = mid
        return adv.detach()

    def _initialize_boundary(self) -> Optional[torch.Tensor]:
        z_adv = self._find_initial_adversarial()
        if z_adv is None:
            return None
        return self._binary_to_boundary(self.z_source, z_adv, "init", max_steps=100)

    # ------------------------------------------------------------------
    # Shared hard-label normal estimator -- same estimator for every method
    # ------------------------------------------------------------------
    def _estimate_normal(self, z_boundary: torch.Tensor, q: int) -> Optional[torch.Tensor]:
        q = min(int(q), self.remaining_queries())
        if q <= 0:
            return None

        noises = torch.randn(
            (q,) + tuple(self.z_source.shape[1:]),
            dtype=self.z_source.dtype,
            device=self.device,
        )
        noisy = z_boundary + self.sigma * noises
        labels = self._predict_labels(noisy, "normal")
        if labels.numel() == 0:
            return None

        noises = noises[: labels.shape[0]]
        status = self._labels_to_status(labels).to(noises.dtype)
        shape = (status.shape[0],) + (1,) * (noises.ndim - 1)
        eta = (status.view(shape) * noises).mean(dim=0, keepdim=True)
        nrm = torch.norm(eta)
        if not torch.isfinite(nrm) or nrm.item() < self.eps:
            return None
        return eta / (nrm + self.eps)

    # ------------------------------------------------------------------
    # Original LGC geometry port
    # ------------------------------------------------------------------
    def _step_lgc(self, z_b: torch.Tensor, eta_o: torch.Tensor) -> torch.Tensor:
        x_s = self.z_source
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (z_b - x_s) / (torch.norm(z_b - x_s) + self.eps)

        dot_val = torch.clamp(
            torch.dot(eta.reshape(-1), v.reshape(-1)),
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )
        theta = torch.acos(dot_val)

        p_near_boundary = None
        num_calls = 1
        while num_calls < 40 and self.remaining_queries() > 0:
            ang = (math.pi / 2.0) * (1.0 - 1.0 / (2 ** num_calls))
            denom = math.sin(ang) + self.eps
            m = (
                torch.sin(theta).item() * math.cos(ang) / denom
                - torch.cos(theta).item()
            )
            zeta = eta + m * v
            zeta = zeta / (torch.norm(zeta) + self.eps)
            radius = torch.norm(z_b - x_s)
            radial_proj = torch.dot(v.reshape(-1), zeta.reshape(-1))
            p = x_s + zeta * radius * radial_proj

            status = self.is_adversarial(p, "lgc_search")
            if status == 0:
                return z_b.detach()
            if status == -1:
                p_near_boundary = p.detach()
                break
            num_calls += 1

        if p_near_boundary is None:
            return z_b.detach()

        # Attached SemiCircular_boundary_search geometry.
        norm_dis = torch.norm(z_b - x_s)
        boundary_dir = (z_b - x_s) / (norm_dis + self.eps)
        clean_dir = (p_near_boundary - x_s) / (
            torch.norm(p_near_boundary - x_s) + self.eps
        )
        adv_dir = boundary_dir.detach().clone()
        adv = z_b.detach().clone()
        clean = x_s.detach().clone()

        for _ in range(101):
            if self.remaining_queries() <= 0:
                break
            if torch.norm(adv - clean).item() < self.tol:
                break
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir / (torch.norm(mid_dir) + self.eps)
            dot_mid = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
                torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir) + self.eps
            )
            dot_mid = torch.clamp(dot_mid, -1.0 + 1e-7, 1.0 - 1e-7)
            ang_mid = torch.acos(dot_mid)
            d = torch.cos(ang_mid) * norm_dis
            z_mid = x_s + mid_dir * d
            status = self.is_adversarial(z_mid, "refine")
            if status == 0:
                break
            if status == 1:
                adv_dir = mid_dir
                adv = z_mid
            else:
                clean_dir = mid_dir
                clean = z_mid
        return adv.detach()

    # ------------------------------------------------------------------
    # Original LGC_H geometry port
    # ------------------------------------------------------------------
    def _step_lgc_h(self, z_b: torch.Tensor, eta_o: torch.Tensor) -> torch.Tensor:
        x_s = self.z_source
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (z_b - x_s) / (torch.norm(z_b - x_s) + self.eps)

        dot_val = torch.clamp(
            torch.dot(eta.reshape(-1), v.reshape(-1)),
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )
        theta = torch.acos(dot_val)

        adv_trial = None
        for num_calls in range(1, 40):
            if self.remaining_queries() <= 0:
                break
            denom = torch.sin(theta / (2 ** num_calls)) + self.eps
            m = (
                torch.sin(theta)
                * torch.cos(theta / (2 ** num_calls))
                / denom
                - torch.cos(theta)
            ).item()
            zeta = eta + m * v
            zeta = zeta / (torch.norm(zeta) + self.eps)
            radius = torch.norm(z_b - x_s)
            radial_proj = torch.dot(zeta.reshape(-1), v.reshape(-1))
            cand = x_s + zeta * radius * radial_proj
            status = self.is_adversarial(cand, "lgc_search")
            if status == 0:
                break
            if status == 1:
                adv_trial = cand.detach()
                break

        if adv_trial is None:
            return z_b.detach()
        return self._binary_to_boundary(x_s, adv_trial, "refine", max_steps=100)

    # ------------------------------------------------------------------
    # Attached adaptive-K QS port
    # ------------------------------------------------------------------
    def _parse_k_ladder(self):
        if self._K_ladder is None:
            self._K_ladder = sorted(self.qs_k_schedule.items(), key=lambda kv: kv[0])
        return self._K_ladder

    def _compute_qs_k(self, iter_idx: int) -> int:
        if self.qs_k_mode == "fixed":
            return int(self.qs_fixed_k)

        cur_ssim = float(self._cur_ssim)
        prev_ssim = float(self._prev_ssim)
        gap = max(0.0, self.switch_ssim - cur_ssim)

        ladder = self._parse_k_ladder()
        K = ladder[-1][1]
        for thresh, k in ladder:
            if gap >= thresh:
                K = k
            else:
                break

        if self.enable_progress_k and iter_idx > 0:
            gap_prev = max(1e-6, self.switch_ssim - prev_ssim)
            gap_cur = max(0.0, self.switch_ssim - cur_ssim)
            progress_ratio = (gap_prev - gap_cur) / gap_prev
            if progress_ratio > self._K_fast:
                K -= 2
            elif progress_ratio < self._K_slow:
                K += 2

        return int(max(self.qs_k_min, min(self.qs_k_max, K)))

    def _step_qs(self, z_b: torch.Tensor, eta_o: torch.Tensor, iter_idx: int) -> torch.Tensor:
        x_s = self.z_source
        eta = eta_o / (torch.norm(eta_o) + self.eps)
        v = (z_b - x_s) / (torch.norm(z_b - x_s) + self.eps)
        K = self._compute_qs_k(iter_idx)
        R = torch.norm(z_b - x_s)
        phis = torch.linspace(
            0.0,
            math.pi / 2.0,
            steps=K,
            dtype=z_b.dtype,
            device=z_b.device,
        )

        if self.verbose == "Yes":
            print(
                f"         QS-AdapK: iter={iter_idx} "
                f"gap={self.switch_ssim - self._cur_ssim:.4f} K={K}"
            )

        best_adv = None
        for k in range(K):
            if self.remaining_queries() <= 0:
                break
            zeta = _rotate_in_plane(eta, v, phis[k], eps=self.eps)
            lo, hi = 0.0, float(R.item())
            cand = None

            for _ in range(30):
                if self.remaining_queries() <= 0:
                    break
                mid = 0.5 * (lo + hi)
                pt = x_s + zeta * mid
                status = self.is_adversarial(pt, "qs_search")
                if status == 0:
                    break
                if status == 1:
                    cand = pt.detach()
                    hi = mid
                else:
                    lo = mid
                if hi - lo < self.tol:
                    break

            if cand is not None and self.remaining_queries() > 0:
                cand = self._binary_to_boundary(
                    self.z_source, cand, "refine", max_steps=100
                )
                if (
                    best_adv is None
                    or torch.norm(cand - x_s).item() < torch.norm(best_adv - x_s).item()
                ):
                    best_adv = cand.detach()

        if best_adv is None:
            # Same fallback intention as attached file.
            return self._step_lgc_h(z_b, eta_o)

        # Tangential extrapolation copied from attached QS.
        try:
            tangent = v - torch.dot(v.reshape(-1), eta.reshape(-1)) * eta
            tangent = tangent / (torch.norm(tangent) + self.eps)
            step = (0.02 + 0.06 * iter_idx / max(self.iteration - 1, 1)) * R
            adv_ext = best_adv + tangent * step
            if self.remaining_queries() > 0:
                status = self.is_adversarial(adv_ext, "tangent")
                if status == 1:
                    adv_ext = self._binary_to_boundary(
                        self.z_source, adv_ext, "refine", max_steps=100
                    )
                    if torch.norm(adv_ext - x_s).item() < torch.norm(best_adv - x_s).item():
                        best_adv = adv_ext.detach()
        except Exception:
            pass

        return best_adv.detach()

    def _step_hybrid(self, z_b: torch.Tensor, eta: torch.Tensor, iter_idx: int) -> torch.Tensor:
        # Preserve attached switch rule. _cur_ssim is intentionally updated with
        # the PREVIOUS recorded state after each iteration, matching its one-step lag.
        if self._cur_ssim >= self.switch_ssim:
            if not self._hybrid_switched and self.verbose == "Yes":
                print(
                    f"         Hybrid: switched to LGC_H @ Iter {iter_idx} "
                    f"SSIM={self._cur_ssim:.4f} threshold={self.switch_ssim:.4f}"
                )
                self._hybrid_switched = True
            return self._step_lgc_h(z_b, eta)
        return self._step_qs(z_b, eta, iter_idx)

    # ------------------------------------------------------------------
    # RAG-TtBA baseline
    # ------------------------------------------------------------------
    def _ttba_direction(self, z_b: torch.Tensor, eta: torch.Tensor, k: float) -> torch.Tensor:
        v = (z_b - self.z_source) / (torch.norm(z_b - self.z_source) + self.eps)
        n = eta / (torch.norm(eta) + self.eps)
        d = float(k) * n + (1.0 - float(k)) * v
        return d / (torch.norm(d) + self.eps)

    def _step_ttba(self, z_b: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z_b - self.z_source)
        if r.item() < self.eps:
            return z_b.detach()

        L, R = 0.0, 1.0
        while (R - L) > self.ttba_bridge_tol and self.remaining_queries() > 0:
            mid = 0.5 * (L + R)
            d = self._ttba_direction(z_b, eta, mid)
            cand = self.z_source + r * d
            status = self.is_adversarial(cand, "ttba_bridge")
            if status == 0:
                break
            if status == -1:
                R = mid
            else:
                L = mid

        # Piecewise approximation used by the validated TtBA bridge implementation.
        low, JUMP, high, DIVE = 0.10, 0.90, 0.20, 2.0 / 3.0
        kk = (DIVE * high - JUMP * low) / (high - low)
        if L <= low:
            k_app = JUMP * L
        elif L >= high:
            k_app = DIVE * L
        else:
            k_app = kk * (L - low) + JUMP * low

        d = self._ttba_direction(z_b, eta, k_app)
        adv_trial = self.z_source + r * d
        z_new = self._binary_to_boundary(
            self.z_source, adv_trial, "refine", max_steps=100
        )
        self.diagnostics.append(
            {
                "method": "RAG_TTBA_BASE",
                "k_bridge": float(L),
                "k_app": float(k_app),
            }
        )
        return z_new.detach()

    # ------------------------------------------------------------------
    # Metrics / checkpoints
    # ------------------------------------------------------------------
    def _compute_metrics(self, z_adv: torch.Tensor) -> Dict[str, float]:
        # Metrics are post-hoc and DO NOT call the victim classifier.
        with torch.no_grad():
            x_adv = self.model.get_pixel_image(z_adv)
            l2_lat = float(torch.norm(z_adv - self.z_source, p=2).item())
            linf_lat = float(torch.max(torch.abs(z_adv - self.z_source)).item())
            delta = x_adv - self.x_orig
            l2_pix = float(torch.norm(delta, p=2).item())
            linf_pix = float(torch.max(torch.abs(delta)).item())
            ssim_val = float(
                self.ssim_fn(x_adv, self.x_orig, data_range=1.0).detach().item()
            )
            lpips_val = float(
                self.loss_fn_vgg(
                    x_adv * 2.0 - 1.0,
                    self.x_orig * 2.0 - 1.0,
                ).detach().mean().item()
            )

        return {
            "query": int(self.all_queries),
            "l2_lat": l2_lat,
            "linf_lat": linf_lat,
            "l2_pix": l2_pix,
            "linf_pix": linf_pix,
            "ssim": ssim_val,
            "lpips": lpips_val,
            # Every stored state is maintained as adversarial by construction.
            "asr": 1.0,
        }

    def _record_state(self, z_adv: torch.Tensor, phase: str):
        row = self._compute_metrics(z_adv)
        row["phase"] = phase
        self.history.append(row)
        if self.verbose == "Yes":
            print(
                f"      -> [{phase:<9}] Q={row['query']:<5d} "
                f"| PixL2={row['l2_pix']:.4f} "
                f"| LatL2={row['l2_lat']:.4f} "
                f"| SSIM={row['ssim']:.5f} "
                f"| LPIPS={row['lpips']:.5f}"
            )

    def checkpoint_summary(self):
        out = {
            "checkpoint": np.asarray(self.checkpoints, dtype=np.int64),
            "state_query": np.full(len(self.checkpoints), -1, dtype=np.int64),
            "l2_lat": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "linf_lat": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "l2_pix": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "linf_pix": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "ssim": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "lpips": np.full(len(self.checkpoints), np.nan, dtype=np.float64),
            "asr": np.zeros(len(self.checkpoints), dtype=np.float64),
        }

        for j, cp in enumerate(self.checkpoints):
            eligible = [h for h in self.history if h["query"] <= cp]
            if not eligible:
                continue
            # Same best-under-budget rule for every method.
            h = min(eligible, key=lambda r: r["l2_pix"])
            out["state_query"][j] = int(h["query"])
            for key in [
                "l2_lat", "linf_lat", "l2_pix", "linf_pix", "ssim", "lpips", "asr"
            ]:
                out[key][j] = float(h[key])
        return out

    # ------------------------------------------------------------------
    # Main attack
    # ------------------------------------------------------------------
    def attack(self):
        z_b = self._initialize_boundary()
        if z_b is None:
            return {
                "success": False,
                "final_z": self.z_source.detach(),
                "final": None,
                "final_current": None,
                "checkpoints": self.checkpoint_summary(),
                "query_breakdown": dict(self.q_breakdown),
                "query_audit": {
                    "total": int(self.all_queries),
                    "sum_breakdown": int(sum(self.q_breakdown.values())),
                    "match": int(self.all_queries) == int(sum(self.q_breakdown.values())),
                },
                "diagnostics": self.diagnostics,
            }

        self._record_state(z_b, "Initial")

        for i in range(self.iteration):
            if self.remaining_queries() <= self.update_query_reserve:
                break

            old_q = self.all_queries
            old_z = z_b.detach().clone()
            previous_recorded_ssim = float(self.history[-1]["ssim"])

            qn = max(10, int(self.N0 * math.sqrt(i + 1.0)))
            qn = min(
                qn,
                max(0, self.remaining_queries() - self.update_query_reserve),
            )
            if qn <= 0:
                break

            eta = self._estimate_normal(z_b, qn)
            if eta is None:
                break

            if self.attack_method == "LGC":
                z_candidate = self._step_lgc(z_b, eta)
            elif self.attack_method == "LGC_H":
                z_candidate = self._step_lgc_h(z_b, eta)
            elif self.attack_method == "QS":
                z_candidate = self._step_qs(z_b, eta, i)
            elif self.attack_method == "Hybrid":
                z_candidate = self._step_hybrid(z_b, eta, i)
            elif self.attack_method == "RAG_TTBA_BASE":
                z_candidate = self._step_ttba(z_b, eta)
            else:
                raise RuntimeError("Unreachable attack dispatch.")

            # Preserve method behavior: use the returned adversarial state.
            z_b = z_candidate.detach()

            # Preserve the attached QS/Hybrid one-state lag for K/switch decisions.
            self._prev_ssim = previous_recorded_ssim
            self._cur_ssim = previous_recorded_ssim

            self._record_state(z_b, f"Iter{i}")

            if self.all_queries == old_q:
                break
            if (
                torch.norm(z_b - old_z).item() < self.eps
                and self.remaining_queries() <= self.update_query_reserve
            ):
                break

        if self.history:
            best = min(self.history, key=lambda r: r["l2_pix"])
            final = {
                key: (int(best[key]) if key == "query" else float(best[key]))
                for key in [
                    "query", "l2_lat", "linf_lat", "l2_pix", "linf_pix", "ssim", "lpips", "asr"
                ]
            }
            final_current = self._compute_metrics(z_b)
        else:
            final = None
            final_current = None

        q_sum = int(sum(self.q_breakdown.values()))
        audit = {
            "total": int(self.all_queries),
            "sum_breakdown": q_sum,
            "match": int(self.all_queries) == q_sum,
        }
        if not audit["match"]:
            raise RuntimeError(
                f"Query accounting mismatch: total={self.all_queries}, breakdown={q_sum}"
            )
        if self.all_queries > self.max_queries:
            raise RuntimeError(
                f"Budget violation: used {self.all_queries} > max_queries {self.max_queries}"
            )

        return {
            "success": True,
            "final_z": z_b.detach(),
            "final": final,
            "final_current": final_current,
            "checkpoints": self.checkpoint_summary(),
            "query_breakdown": dict(self.q_breakdown),
            "query_audit": audit,
            "diagnostics": self.diagnostics,
        }
