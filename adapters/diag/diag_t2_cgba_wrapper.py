#!/usr/bin/env python3
"""B3b T2 — CGBA wrapper fidelity: faithful (raw un-normalised net) vs adapter (UnnormalizeVictim).

Official drivers build ``net = torch_models.resnet50(pretrained=True)`` with NO normalisation layer and feed it an
ImageNet-normalised tensor (Non_targeted_attack.py:42,84-88 / Targeted_attack.py:43,94-98).  The faithful path is
therefore: the normalised tensor goes STRAIGHT into the un-normalised network, and the recorder maps to pixel space
only to compute L2 (to_pixel_unnormalize).

The adapter instead wraps the shared victim (build_ttba_github_victim, normalising [0,1] internally via
_Norm.forward ``self.m((x-mu)/sd)`` at baseline_protocol.py:364-365) in UnnormalizeVictim, so the model sees
``victim(x_n*sd+mu)`` -> ``net(((x_n*sd+mu)-mu)/sd)``, a float round-trip of x_n.

Both paths run the SAME official attack (with the boundary-walk count fix of T3 applied locally so both complete to
budget, isolating the model-object difference from the off-by-one crash).  Compares best-so-far pixel L2 at
1000/5000/10000 and the query count, and reports max|x_n - ((x_n*sd+mu)-mu)/sd| on x0.

Cells: row 1 non_targeted, rows 1-3 targeted (ResNet-50 V2), budget 10000.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))
sys.path.insert(0, "/root/autodl-tmp/baselines/CGBA")

import _common as C  # noqa: E402
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
import torch  # noqa: E402
from proposed_attack import Proposed_attack  # noqa: E402

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
CKPT = [1000, 5000, 10000]
BUDGET = 10000
ARCH = "resnet50"


class PatchedAttack(Proposed_attack):
    """Local copy of the T3 count fix: return the true query count of each boundary walk (diff of all_queries)."""

    def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
        c0 = self.all_queries
        out, n = super().go_to_boundary_CGBA(x_s, eta_o, x_b)
        return out, self.all_queries - c0

    def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):
        c0 = self.all_queries
        out, n = super().go_to_boundary_CGBA_H(x_s, eta_o, x_b)
        return out, self.all_queries - c0


def build_raw_net(device):
    import torchvision.models as tvm

    net = tvm.resnet50()
    net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "resnet50-11ad3fa6.pth"), map_location="cpu"))
    return net.to(device).eval()


def best_l2(tracker):
    tracker.core.carry_forward()
    return {b: tracker.core.recorded[b][0] for b in tracker.core.budgets}


def run(mode, rows):
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")
    raw = build_raw_net("cuda")
    import run_cgba as A

    for idx, r in enumerate(rows):
        name = r["source"]
        y_src = int(r["source_label"])
        y_tgt = int(r["target_label"]) if (mode == "targeted" and r.get("target_label")) else None
        x0 = C.load_image(DATA_DIR, name, "cuda")
        x_tgt = C.load_image(DATA_DIR, r["target"], "cuda") if y_tgt is not None else None
        params = dict(A.PARAMS, attack_method=("CGBA_H" if y_tgt is not None else "CGBA"))

        xn = C.normalize(x0)
        mu = torch.tensor(C.MEAN, device=xn.device, dtype=xn.dtype).view(1, 3, 1, 1)
        sd = torch.tensor(C.STD, device=xn.device, dtype=xn.dtype).view(1, 3, 1, 1)
        rterr = float((xn - ((xn * sd + mu - mu) / sd)).abs().max().item())

        def build_tracker(model):
            return HardLabelTracker(model, x0, y_src, y_tgt, CKPT, BUDGET,
                                    to_pixel=C.to_pixel_unnormalize,
                                    init_adv_pixel=x_tgt if y_tgt is not None else None)

        def run_attack(model):
            src = C.normalize(x0)
            tar = C.normalize(x_tgt) if x_tgt is not None else None
            lb = C.normalize(torch.zeros_like(x0))
            ub = C.normalize(torch.ones_like(x0))
            atk = PatchedAttack(model, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=tar,
                                dim_reduc_factor=params["dim_reduc_factor"],
                                attack_method=params["attack_method"], iteration=params["iteration"],
                                initial_query=params["initial_query"], tol=params["tol"],
                                sigma=params["sigma"], verbose_control=params["verbose_control"])
            atk.Attack()

        tr_adapt = build_tracker(C.UnnormalizeVictim(victim))
        C.seed_all(idx)
        try:
            run_attack(tr_adapt)
        except BudgetExhausted:
            pass
        ad = best_l2(tr_adapt)

        tr_faith = build_tracker(raw)
        C.seed_all(idx)
        try:
            run_attack(tr_faith)
        except BudgetExhausted:
            pass
        fa = best_l2(tr_faith)

        ratios = [float("nan") if (fa[b] == float("inf") or ad[b] == float("inf") or fa[b] == 0) else ad[b] / fa[b]
                  for b in CKPT]

        def fmt(v):
            return "inf" if v == float("inf") else f"{v:.6f}"

        print(f"RESULT T2 {mode} {C.stem_of(name)} rterr={rterr:.3e} "
              f"faith={fmt(fa[1000])},{fmt(fa[5000])},{fmt(fa[10000])} "
              f"adapt={fmt(ad[1000])},{fmt(ad[5000])},{fmt(ad[10000])} "
              f"ratio={ratios[0]:.6f},{ratios[1]:.6f},{ratios[2]:.6f} "
              f"q_faith={tr_faith.queries} q_adapt={tr_adapt.queries}", flush=True)
        del tr_adapt, tr_faith, x0, x_tgt


def main():
    rows_nt = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_non_targeted.csv")[:1]
    rows_t = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_targeted.csv")[:3]
    t0 = time.time()
    run("non_targeted", rows_nt)
    run("targeted", rows_t)
    print(f"DONE T2 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
