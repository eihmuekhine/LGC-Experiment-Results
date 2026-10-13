#!/usr/bin/env python3
"""B4d T4(b) — CGBA (non_targeted) boundary-walk geometry trace (OBSERVATION ONLY).

Replicates go_to_boundary_CGBA (proposed_attack.py:151-167) verbatim but LOGS, for every walk:
  theta_t = acos(eta_hat . v_hat)   [rad + deg]
  for each candidate i (num_calls): psi_i as computed, cos(psi_i),
     r_i = ||x_q - x_s|| / ||x_b - x_s|| (normalised, unclipped x_q),
     r_i_clipped (the point actually queried), clip displacement ||x_q - clip(x_q)||, label, tracker best before/after.
  whether the 40-iteration cap fired; the BSSP result distance ratio ||x_b' - x_s|| / ||x_b - x_s||.

Then reports:
  * the FIRST walk where |r_i - cos(psi_i)| > 1e-4 (geometry deviation),
  * EVERY capped walk in which the tracker best did not decrease.

The returned boundary point and the query count are IDENTICAL to the official code (the count is the
all_queries diff, so the assert at proposed_attack.py:252 does not crash). NOTHING in the attack logic,
seeds or hyper-parameters is changed.

    python adapters/diag/cgba_trace.py [rows=3]
"""
from __future__ import annotations

import math
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
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
import torch  # noqa: E402
from proposed_attack import Proposed_attack  # noqa: E402
from utils import clip_image_values  # noqa: E402

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
BUDGET = 10000
ARCH = "resnet50"


def tof(x):
    if isinstance(x, (int, float)):
        return float(x)
    return float(torch.as_tensor(x).reshape(-1)[0].item())


class TraceAttack(Proposed_attack):
    """Logging subclass: identical geometry, identical returned boundary point + true query count."""

    def __init__(self, *args, tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
        self.walks = []
        self._walk_idx = 0

    def _best(self):
        return float(self.tracker.core.best_l2) if self.tracker is not None else float("nan")

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        # OBSERVATION-ONLY copy of proposed_attack.py:171-196 with NaN-source instrumentation.
        num_calls = 0
        norm_dis = torch.norm(x_b - x_0)
        boundary_dir = (x_b - x_0) / torch.norm(x_b - x_0)
        clean_dir = (p_near_boundary - x_0) / torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b
        clean = x_0
        self._nan = {"min_sum_norm": float("inf"), "max_acos_arg": -1.0, "where": None}
        while True:
            mid_dir = adv_dir + clean_dir
            sum_norm = torch.norm(mid_dir)
            if float(sum_norm) < self._nan["min_sum_norm"]:
                self._nan["min_sum_norm"] = float(sum_norm)
            mid_dir = mid_dir / torch.norm(mid_dir)
            acos_arg = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
                torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir))
            if float(acos_arg) > self._nan["max_acos_arg"]:
                self._nan["max_acos_arg"] = float(acos_arg)
            theta = torch.acos(acos_arg)
            d = torch.cos(theta) * norm_dis
            x_mid = x_0 + mid_dir * d
            num_calls += 1
            if torch.isnan(x_mid).any() and self._nan["where"] is None:
                self._nan["where"] = f"x_mid nan at num_calls={num_calls} (sum_norm={float(sum_norm):.3e} "
                f"acos_arg={float(acos_arg):.6f})"
            if self.is_adversarial(x_mid) == 1:
                adv_dir = mid_dir
                adv = x_mid
            else:
                clean_dir = mid_dir
                clean = x_mid
            if torch.norm(adv - clean).cpu().numpy() < self.tol:
                break
            if num_calls > 100:
                break
        return adv, num_calls

    def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o / torch.norm(eta_o)
        v = (x_b - x_s) / torch.norm(x_b - x_s)
        theta = torch.acos(torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0, 1.0))
        R = float(torch.norm(x_b - x_s))
        c0 = self.all_queries
        walk = {"iter": self._walk_idx, "theta_rad": float(theta), "theta_deg": math.degrees(float(theta)),
                "R": R, "candidates": [], "cap": False, "break_i": None}
        capped = False
        while True:
            psi = (math.pi / 2) * (1.0 - 1.0 / pow(2, num_calls))  # == 90 - 90/2^i degrees
            cospsi = math.cos(psi)
            m = (torch.sin(theta.cpu()) * torch.cos(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) /
                 torch.sin(torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) - torch.cos(theta.cpu())).item()
            zeta = (eta + m * v) / torch.norm(eta + m * v)
            p_near_boundary = x_s + zeta * torch.norm(x_b - x_s) * torch.dot(v.reshape(-1), zeta.reshape(-1))
            x_q = p_near_boundary
            clipped = clip_image_values(x_q, self.lb, self.ub)
            r_i = float(torch.norm(x_q - x_s)) / R if R > 0 else float("nan")
            r_i_clip = float(torch.norm(clipped - x_s)) / R if R > 0 else float("nan")
            clip_disp = float(torch.norm(clipped - x_q))
            best_before = self._best()
            label = self.is_adversarial(clipped)  # 1 or -1 (queries the CLIPPED point)
            best_after = self._best()
            walk["candidates"].append({
                "i": num_calls, "psi_deg": math.degrees(psi), "cospsi": cospsi,
                "r_i": r_i, "r_i_clip": r_i_clip, "clip_disp": clip_disp,
                "label": label, "best_before": best_before, "best_after": best_after})
            if label == -1:
                walk["break_i"] = num_calls
                break
            num_calls += 1
            if num_calls >= 40:
                walk["cap"] = True
                capped = True
                break
        if capped:
            out = x_b
            qs = self.all_queries - c0
            walk["bssp_ratio"] = float("nan")
        else:
            perturbed, n_calls = self.SemiCircular_boundary_search(x_s, x_b, clipped)
            out = perturbed
            qs = (self.all_queries - c0)
            walk["bssp_ratio"] = float(torch.norm(perturbed - x_s)) / R if R > 0 else float("nan")
            walk["bssp_nan_diag"] = dict(self._nan) if hasattr(self, "_nan") else None
        walk["returned_ratio"] = float(torch.norm(out - x_s)) / R if R > 0 else float("nan")
        walk["qs"] = int(qs)
        self.walks.append(walk)
        self._walk_idx += 1
        return out, qs


def run_image(r, idx):
    name = r["source"]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, name, "cuda")
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = A._faithful_model(victim)
    params = dict(A.PARAMS, attack_method="CGBA")
    ckpts = C.contract_checkpoints(C.load_contract())

    tracker = HardLabelTracker(raw, x0, y_src, None, ckpts, BUDGET,
                               to_pixel=C.to_pixel_unnormalize, init_adv_pixel=None)
    C.seed_all(idx)
    src = C.normalize(x0)
    lb = C.normalize(torch.zeros_like(x0))
    ub = C.normalize(torch.ones_like(x0))
    atk = TraceAttack(tracker, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=None,
                      dim_reduc_factor=params["dim_reduc_factor"], attack_method="CGBA",
                      iteration=params["iteration"], initial_query=params["initial_query"],
                      tol=params["tol"], sigma=params["sigma"], verbose_control="No", tracker=tracker)
    try:
        atk.Attack()
    except BudgetExhausted:
        pass
    print(f"\n########## IMAGE {C.stem_of(name)}  tracker.queries={tracker.queries} "
          f"best_pixel_l2@10k={tracker.core.recorded.get(10000)} walks={len(atk.walks)} ##########")

    first_geom = None
    capped_no_improve = []
    for w in atk.walks:
        best_start = w["candidates"][0]["best_before"] if w["candidates"] else float("nan")
        best_end = w["candidates"][-1]["best_after"] if w["candidates"] else best_start
        # geometry deviation
        for cand in w["candidates"]:
            if abs(cand["r_i"] - cand["cospsi"]) > 1e-4:
                if first_geom is None:
                    first_geom = (w["iter"], cand["i"], cand["r_i"], cand["cospsi"], cand["clip_disp"])
        if w["cap"] and best_end >= best_start - 1e-9:
            capped_no_improve.append((w["iter"], best_start, best_end, w["qs"]))

    print(f"  iter={len(atk.walks)}  first_geometry_deviation(|r_i-cospsi|>1e-4)={first_geom}")
    print(f"  capped_walks_without_best_decrease={len(capped_no_improve)}")
    for it, bs, be, qs in capped_no_improve:
        print(f"    iter={it} best {bs:.4f}->{be:.4f} qs={qs}")

    # compact summary of EVERY walk
    print("  --- per-walk summary (iter, theta_deg, R, cap, break_i, returned_ratio, qs) ---")
    for w in atk.walks:
        print(f"    iter={w['iter']:2d} theta={w['theta_deg']:>8.2f} R={w['R']:>10.3f} cap={str(w['cap']):5s} "
              f"break_i={str(w['break_i']):>3s} ret_ratio={w['returned_ratio']:>8.4f} qs={w['qs']}")

    # detail for the first 3 walks + every capped walk + any walk that returned NaN
    for w in atk.walks:
        show = (w["iter"] < 3) or w["cap"] or (w["returned_ratio"] != w["returned_ratio"])
        if not show:
            continue
        print(f"\n  walk iter={w['iter']} theta={w['theta_deg']:.2f}deg R={w['R']:.3f} cap={w['cap']} "
              f"bssp_ratio={w['bssp_ratio']:.4f} returned_ratio={w['returned_ratio']:.4f} qs={w['qs']}")
        if w.get("bssp_nan_diag"):
            d = w["bssp_nan_diag"]
            print(f"    BSSP nan-diag: {d}")
        for cand in w["candidates"]:
            print(f"    i={cand['i']:2d} psi={cand['psi_deg']:8.4f}deg cospsi={cand['cospsi']:.6f} "
                  f"r_i={cand['r_i']:.6f} r_i_clip={cand['r_i_clip']:.6f} clip_disp={cand['clip_disp']:.6f} "
                  f"label={cand['label']:+d} best={cand['best_before']:.4f}->{cand['best_after']:.4f}")
    return atk.walks


def main():
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_non_targeted.csv")[:nrows]
    for i, r in enumerate(rows):
        run_image(r, i)
    print("\nDONE cgba_trace")


if __name__ == "__main__":
    main()
