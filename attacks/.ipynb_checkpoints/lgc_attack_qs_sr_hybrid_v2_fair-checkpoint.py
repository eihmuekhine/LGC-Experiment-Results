"""
Fair hard-label comparison engine for the user's RAG/LGC ImageNet setting.

Compared methods:
  * RAG_TTBA_BASE                   : Original TtBA (Linear Interpolation)
  * RAG_TTBA_OPTIMIZED              : Linear TtBA + Adaptive Ratio + Tangent Extrapolation
  * RAG_TTBA_SLERP_OPTIMIZED        : SLERP (Angle-based) + Adaptive + SMART Tangent
  * RAG_TTBA_POLAR_BEZIER_OPTIMIZED : Rational Spherical Bezier + Adaptive + SMART Tangent
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _plane_basis(eta: torch.Tensor, v: torch.Tensor, eps: float = 1e-12):
    e1 = eta / (torch.norm(eta) + eps)
    proj = torch.sum(v * e1) * e1
    residual = v - proj
    nr = torch.norm(residual)
    if nr.item() < eps:
        flat = e1.reshape(-1)
        idx = int(torch.argmin(torch.abs(flat)).item())
        basis = torch.zeros_like(flat)
        basis[idx] = 1.0
        basis = basis.reshape_as(e1)
        residual = basis - torch.sum(basis * e1) * e1
        nr = torch.norm(residual)
    e2 = residual / (nr + eps)
    return e1, e2


def build_hard_label_target_seed(
    encoder,
    x_orig: torch.Tensor,
    x_rec_0: torch.Tensor,
    x_target_raw: torch.Tensor,
) -> torch.Tensor:
    pseudo_target = torch.clamp(x_target_raw - (x_orig - x_rec_0), 0.0, 1.0)
    with torch.no_grad():
        return encoder(pseudo_target).detach()


class FairLGCCompareAttack:
    SUPPORTED = {
        "LGC", "LGC_H", "QS", "QS_SR_V1", "QS_SR_V2",
        "Hybrid", "QS_SR_HYBRID_V1", "QS_SR_HYBRID_V2", 
        "RAG_TTBA_BASE", "RAG_TTBA_OPTIMIZED", 
        "RAG_TTBA_SLERP_OPTIMIZED", "RAG_TTBA_POLAR_BEZIER_OPTIMIZED"
    }
    NORMAL_METHODS = SUPPORTED
    CURRENT_STATE_SSIM_METHODS = {"QS_SR_V2", "QS_SR_HYBRID_V2"}

    def __init__(
        self,
        model,
        z_source: torch.Tensor,
        x_orig: torch.Tensor,
        source_label: int,
        loss_fn_vgg,
        ssim_fn,
        attack_method: str = "RAG_TTBA_POLAR_BEZIER_OPTIMIZED",
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
        ttba_bridge_tol: float = 1e-3,
        update_query_reserve: int = 18,
    ):
        if attack_method not in self.SUPPORTED:
            raise ValueError(f"Unsupported attack_method={attack_method}.")

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
        self.update_query_reserve = int(update_query_reserve)
        self.ttba_bridge_tol = float(ttba_bridge_tol)

        self.all_queries = 0
        self.q_breakdown = OrderedDict(
            init=0, normal=0, lgc_search=0, qs_search=0, 
            qs_sr_screen=0, ttba_bridge=0, tangent=0, refine=0,
        )
        self.history = []
        self.diagnostics = []

        self._set_seed(self.seed)

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
            return torch.where(labels != self.source_label, torch.ones_like(labels), -torch.ones_like(labels))
        return torch.where(labels == self.target_label, torch.ones_like(labels), -torch.ones_like(labels))

    def is_adversarial(self, z: torch.Tensor, category: str) -> int:
        labels = self._predict_labels(z, category)
        if labels.numel() == 0:
            return 0 
        return int(self._labels_to_status(labels)[0].item())

    def _find_initial_adversarial(self) -> Optional[torch.Tensor]:
        if self.remaining_queries() <= 0: return None
        if self.target_label is not None:
            delta = self.target_seed - self.z_source
            for factor in [1.0, 1.10, 1.25, 1.50, 2.0, 3.0]:
                if self.remaining_queries() <= 0: break
                cand = self.z_source + factor * delta
                if self.is_adversarial(cand, "init") == 1: return cand.detach()
            for j in range(1, 51):
                if self.remaining_queries() <= 0: break
                cand = self.target_seed + (0.01 * j) * torch.randn_like(self.target_seed)
                if self.is_adversarial(cand, "init") == 1: return cand.detach()
            return None
        for j in range(1, 501):
            if self.remaining_queries() <= 0: break
            cand = self.z_source + (0.02 * j) * torch.randn_like(self.z_source)
            if self.is_adversarial(cand, "init") == 1: return cand.detach()
        return None

    def _binary_to_boundary(self, z_clean: torch.Tensor, z_adv: torch.Tensor, category: str = "refine", max_steps: int = 100) -> torch.Tensor:
        clean = z_clean.detach().clone()
        adv = z_adv.detach().clone()
        for _ in range(max_steps):
            if self.remaining_queries() <= 0: break
            if torch.norm(adv - clean).item() <= self.tol: break
            mid = 0.5 * (clean + adv)
            status = self.is_adversarial(mid, category)
            if status == 0: break
            if status == 1: adv = mid
            else: clean = mid
        return adv.detach()

    def _initialize_boundary(self) -> Optional[torch.Tensor]:
        z_adv = self._find_initial_adversarial()
        if z_adv is None: return None
        return self._binary_to_boundary(self.z_source, z_adv, "init", max_steps=100)

    def _estimate_normal(self, z_boundary: torch.Tensor, q: int) -> Optional[torch.Tensor]:
        q = min(int(q), self.remaining_queries())
        if q <= 0: return None
        noises = torch.randn((q,) + tuple(self.z_source.shape[1:]), dtype=self.z_source.dtype, device=self.device)
        noisy = z_boundary + self.sigma * noises
        labels = self._predict_labels(noisy, "normal")
        if labels.numel() == 0: return None
        noises = noises[: labels.shape[0]]
        status = self._labels_to_status(labels).to(noises.dtype)
        shape = (status.shape[0],) + (1,) * (noises.ndim - 1)
        eta = (status.view(shape) * noises).mean(dim=0, keepdim=True)
        nrm = torch.norm(eta)
        if not torch.isfinite(nrm) or nrm.item() < self.eps: return None
        return eta / (nrm + self.eps)

    # ================= DIRECTIONS =================

    # 1. Linear Direction
    def _ttba_direction(self, z_b: torch.Tensor, eta: torch.Tensor, k: float) -> torch.Tensor:
        v = (z_b - self.z_source) / (torch.norm(z_b - self.z_source) + self.eps)
        n = eta / (torch.norm(eta) + self.eps)
        d = float(k) * n + (1.0 - float(k)) * v
        return d / (torch.norm(d) + self.eps)

    # 2. SLERP Direction
    def _ttba_slerp_direction(self, z_b: torch.Tensor, eta: torch.Tensor, k: float) -> torch.Tensor:
        v = (z_b - self.z_source) / (torch.norm(z_b - self.z_source) + self.eps)
        n = eta / (torch.norm(eta) + self.eps)
        dot_val = torch.clamp(torch.dot(n.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(dot_val)
        if theta.item() < 1e-5:
            d = float(k) * n + (1.0 - float(k)) * v
            return d / (torch.norm(d) + self.eps)
        alpha = float(k) * theta
        denom = torch.sin(theta)
        w1 = torch.sin(theta - alpha) / denom
        w2 = torch.sin(alpha) / denom
        d = w1 * v + w2 * n
        return d / (torch.norm(d) + self.eps)

    # 3. POLAR BEZIER Direction (New)
    def _ttba_polar_bezier_direction(self, z_b: torch.Tensor, eta: torch.Tensor, k: float, w: float = 1.2) -> torch.Tensor:
        v = (z_b - self.z_source) / (torch.norm(z_b - self.z_source) + self.eps)
        n = eta / (torch.norm(eta) + self.eps)
        
        # Calculate Pure Tangent Vector (T)
        dot_val = torch.clamp(torch.dot(n.reshape(-1), v.reshape(-1)), -1.0 + 1e-7, 1.0 - 1e-7)
        T = n - dot_val * v
        T = T / (torch.norm(T) + self.eps)
        
        # Control Points (Rational Bezier on Sphere)
        P0, P1, P2 = v, T, n
        k_f = float(k)
        
        b0 = (1.0 - k_f) ** 2
        b1 = 2.0 * k_f * (1.0 - k_f) * w
        b2 = k_f ** 2
        
        d = b0 * P0 + b1 * P1 + b2 * P2
        return d / (torch.norm(d) + self.eps)

    # ================= ATTACK STEPS =================

    # --- 1. BASE TtBA ---
    def _step_ttba(self, z_b: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        r = torch.norm(z_b - self.z_source)
        if r.item() < self.eps: return z_b.detach()
        L, R_b = 0.0, 1.0
        while (R_b - L) > self.ttba_bridge_tol and self.remaining_queries() > 0:
            mid = 0.5 * (L + R_b)
            d = self._ttba_direction(z_b, eta, mid)
            cand = self.z_source + r * d
            status = self.is_adversarial(cand, "ttba_bridge")
            if status == 0: break
            if status == -1: R_b = mid
            else: L = mid
        low, JUMP, high, DIVE = 0.10, 0.90, 0.20, 2.0 / 3.0
        kk = (DIVE * high - JUMP * low) / (high - low)
        if L <= low: k_app = JUMP * L
        elif L >= high: k_app = DIVE * L
        else: k_app = kk * (L - low) + JUMP * low

        d = self._ttba_direction(z_b, eta, k_app)
        adv_trial = self.z_source + r * d
        z_new = self._binary_to_boundary(self.z_source, adv_trial, "refine", max_steps=100)
        return z_new.detach()

    # --- 2. OPTIMIZED (Linear) ---
    def _step_ttba_optimized(self, z_b: torch.Tensor, eta: torch.Tensor, iter_idx: int) -> torch.Tensor:
        r = torch.norm(z_b - self.z_source)
        if r.item() < self.eps: return z_b.detach()
        L, R_b = 0.0, 1.0
        while (R_b - L) > self.ttba_bridge_tol and self.remaining_queries() > 0:
            mid = 0.5 * (L + R_b)
            d = self._ttba_direction(z_b, eta, mid)
            cand = self.z_source + r * d
            status = self.is_adversarial(cand, "ttba_bridge")
            if status == 0: break
            if status == -1: R_b = mid
            else: L = mid
        alpha = 0.666
        if len(self.history) >= 2:
            progress = (self.history[-2]['l2_lat'] - self.history[-1]['l2_lat']) / max(self.history[-2]['l2_lat'], 1e-6)
            alpha = 0.8 if progress > 0.05 else 0.5 if progress < 0.01 else 0.5 + 0.3 * ((progress - 0.01) / 0.04)
        k_app = alpha * L
        d = self._ttba_direction(z_b, eta, k_app)
        adv_trial = self.z_source + r * d
        z_new = self._binary_to_boundary(self.z_source, adv_trial, "refine", max_steps=100)
        
        r_new = torch.norm(z_new - self.z_source)
        if r_new.item() > self.eps and self.remaining_queries() > self.update_query_reserve:
            if L <= 0.15 or iter_idx > 15:
                v_new = (z_new - self.z_source) / r_new
                tangent = v_new - torch.dot(v_new.reshape(-1), eta.reshape(-1)) * eta
                tangent = tangent / (torch.norm(tangent) + self.eps)
                step_scale = 0.02 + 0.06 * min(iter_idx / 100.0, 1.0)
                z_ext = z_new + tangent * (step_scale * r_new)
                if self.is_adversarial(z_ext, "tangent") == 1:
                    z_ext_ref = self._binary_to_boundary(self.z_source, z_ext, "refine", max_steps=15)
                    if torch.norm(z_ext_ref - self.z_source).item() < r_new.item(): z_new = z_ext_ref
        return z_new.detach()

    # --- 3. SLERP OPTIMIZED ---
    def _step_ttba_slerp_optimized(self, z_b: torch.Tensor, eta: torch.Tensor, iter_idx: int) -> torch.Tensor:
        r = torch.norm(z_b - self.z_source)
        if r.item() < self.eps: return z_b.detach()
        L, R_b = 0.0, 1.0
        while (R_b - L) > self.ttba_bridge_tol and self.remaining_queries() > 0:
            mid = 0.5 * (L + R_b)
            d = self._ttba_slerp_direction(z_b, eta, mid)
            cand = self.z_source + r * d
            status = self.is_adversarial(cand, "ttba_bridge")
            if status == 0: break
            if status == -1: R_b = mid
            else: L = mid
        alpha = 0.666
        if len(self.history) >= 2:
            progress = (self.history[-2]['l2_lat'] - self.history[-1]['l2_lat']) / max(self.history[-2]['l2_lat'], 1e-6)
            alpha = 0.8 if progress > 0.05 else 0.5 if progress < 0.01 else 0.5 + 0.3 * ((progress - 0.01) / 0.04)
        k_app = alpha * L
        d = self._ttba_slerp_direction(z_b, eta, k_app)
        adv_trial = self.z_source + r * d
        z_new = self._binary_to_boundary(self.z_source, adv_trial, "refine", max_steps=100)
        
        r_new = torch.norm(z_new - self.z_source)
        if r_new.item() > self.eps and self.remaining_queries() > self.update_query_reserve:
            if L <= 0.15 or iter_idx > 15:
                v_new = (z_new - self.z_source) / r_new
                tangent = v_new - torch.dot(v_new.reshape(-1), eta.reshape(-1)) * eta
                tangent = tangent / (torch.norm(tangent) + self.eps)
                step_scale = 0.02 + 0.06 * min(iter_idx / 100.0, 1.0)
                z_ext = z_new + tangent * (step_scale * r_new)
                if self.is_adversarial(z_ext, "tangent") == 1:
                    z_ext_ref = self._binary_to_boundary(self.z_source, z_ext, "refine", max_steps=15)
                    if torch.norm(z_ext_ref - self.z_source).item() < r_new.item(): z_new = z_ext_ref
        return z_new.detach()

    # --- 4. POLAR BEZIER OPTIMIZED (New!) ---
    def _step_ttba_polar_bezier_optimized(self, z_b: torch.Tensor, eta: torch.Tensor, iter_idx: int) -> torch.Tensor:
        r = torch.norm(z_b - self.z_source)
        if r.item() < self.eps: return z_b.detach()

        # Dynamic Tension (w): Flat area needs more tangent pull
        w_tension = 1.0 + 0.8 * min(iter_idx / 100.0, 1.0) 

        L, R_b = 0.0, 1.0
        while (R_b - L) > self.ttba_bridge_tol and self.remaining_queries() > 0:
            mid = 0.5 * (L + R_b)
            d = self._ttba_polar_bezier_direction(z_b, eta, mid, w=w_tension)
            cand = self.z_source + r * d
            status = self.is_adversarial(cand, "ttba_bridge")
            if status == 0: break
            if status == -1: R_b = mid
            else: L = mid
        
        alpha = 0.666
        if len(self.history) >= 2:
            progress = (self.history[-2]['l2_lat'] - self.history[-1]['l2_lat']) / max(self.history[-2]['l2_lat'], 1e-6)
            alpha = 0.8 if progress > 0.05 else 0.5 if progress < 0.01 else 0.5 + 0.3 * ((progress - 0.01) / 0.04)
        
        k_app = alpha * L
        d = self._ttba_polar_bezier_direction(z_b, eta, k_app, w=w_tension)
        adv_trial = self.z_source + r * d
        z_new = self._binary_to_boundary(self.z_source, adv_trial, "refine", max_steps=100)
        
        r_new = torch.norm(z_new - self.z_source)
        if r_new.item() > self.eps and self.remaining_queries() > self.update_query_reserve:
            if L <= 0.15 or iter_idx > 15:
                v_new = (z_new - self.z_source) / r_new
                tangent = v_new - torch.dot(v_new.reshape(-1), eta.reshape(-1)) * eta
                tangent = tangent / (torch.norm(tangent) + self.eps)
                step_scale = 0.02 + 0.06 * min(iter_idx / 100.0, 1.0)
                z_ext = z_new + tangent * (step_scale * r_new)
                if self.is_adversarial(z_ext, "tangent") == 1:
                    z_ext_ref = self._binary_to_boundary(self.z_source, z_ext, "refine", max_steps=15)
                    if torch.norm(z_ext_ref - self.z_source).item() < r_new.item(): z_new = z_ext_ref
        return z_new.detach()

    # ------------------------------------------------------------------
    # Core Loop
    # ------------------------------------------------------------------
    def _compute_metrics(self, z_adv: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            x_adv = self.model.get_pixel_image(z_adv)
            l2_lat = float(torch.norm(z_adv - self.z_source, p=2).item())
            linf_lat = float(torch.max(torch.abs(z_adv - self.z_source)).item())
            delta = x_adv - self.x_orig
            l2_pix = float(torch.norm(delta, p=2).item())
            linf_pix = float(torch.max(torch.abs(delta)).item())
            ssim_val = float(self.ssim_fn(x_adv, self.x_orig, data_range=1.0).detach().item())
            lpips_val = float(self.loss_fn_vgg(x_adv * 2.0 - 1.0, self.x_orig * 2.0 - 1.0).detach().mean().item())
        return {
            "query": int(self.all_queries), "l2_lat": l2_lat, "linf_lat": linf_lat,
            "l2_pix": l2_pix, "linf_pix": linf_pix, "ssim": ssim_val, "lpips": lpips_val, "asr": 1.0,
        }

    def _record_state(self, z_adv: torch.Tensor, phase: str):
        row = self._compute_metrics(z_adv)
        row["phase"] = phase
        self.history.append(row)
        if self.verbose == "Yes":
            print(f"      -> [{phase:<9}] Q={row['query']:<5d} | PixL2={row['l2_pix']:.4f} | LatL2={row['l2_lat']:.4f} | SSIM={row['ssim']:.5f} | LPIPS={row['lpips']:.5f}")

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
            if not eligible: continue
            h = min(eligible, key=lambda r: r["l2_pix"])
            out["state_query"][j] = int(h["query"])
            for key in ["l2_lat", "linf_lat", "l2_pix", "linf_pix", "ssim", "lpips", "asr"]: out[key][j] = float(h[key])
        return out

    def attack(self):
        z_b = self._initialize_boundary()
        if z_b is None:
            return {
                "success": False, "final_z": self.z_source.detach(), "final": None, "final_current": None,
                "checkpoints": self.checkpoint_summary(), "query_breakdown": dict(self.q_breakdown),
                "query_audit": {"total": int(self.all_queries), "sum_breakdown": int(sum(self.q_breakdown.values())), "match": int(self.all_queries) == int(sum(self.q_breakdown.values()))},
                "diagnostics": self.diagnostics,
            }

        self._record_state(z_b, "Initial")
        for i in range(self.iteration):
            if self.remaining_queries() <= self.update_query_reserve: break
            old_q, old_z = self.all_queries, z_b.detach().clone()
            
            qn = max(10, int(self.N0 * math.sqrt(i + 1.0)))
            qn = min(qn, max(0, self.remaining_queries() - self.update_query_reserve))
            if qn <= 0: break

            eta = self._estimate_normal(z_b, qn)
            if eta is None: break

            if self.attack_method == "RAG_TTBA_BASE":
                z_candidate = self._step_ttba(z_b, eta)
            elif self.attack_method == "RAG_TTBA_OPTIMIZED":
                z_candidate = self._step_ttba_optimized(z_b, eta, i)
            elif self.attack_method == "RAG_TTBA_SLERP_OPTIMIZED":
                z_candidate = self._step_ttba_slerp_optimized(z_b, eta, i)
            elif self.attack_method == "RAG_TTBA_POLAR_BEZIER_OPTIMIZED":
                z_candidate = self._step_ttba_polar_bezier_optimized(z_b, eta, i)
            else:
                break

            z_b = z_candidate.detach()
            self._record_state(z_b, f"Iter{i}")

            if self.all_queries == old_q: break
            if torch.norm(z_b - old_z).item() < self.eps and self.remaining_queries() <= self.update_query_reserve: break

        if self.history:
            best = min(self.history, key=lambda r: r["l2_pix"])
            final = {k: (int(best[k]) if k == "query" else float(best[k])) for k in ["query", "l2_lat", "linf_lat", "l2_pix", "linf_pix", "ssim", "lpips", "asr"]}
            final_current = self._compute_metrics(z_b)
        else:
            final, final_current = None, None

        q_sum = int(sum(self.q_breakdown.values()))
        return {
            "success": True, "final_z": z_b.detach(), "final": final, "final_current": final_current,
            "checkpoints": self.checkpoint_summary(), "query_breakdown": dict(self.q_breakdown),
            "query_audit": {"total": int(self.all_queries), "sum_breakdown": q_sum, "match": int(self.all_queries) == q_sum},
            "diagnostics": self.diagnostics,
        }