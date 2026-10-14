#!/usr/bin/env python3
"""B4f T6: domain / clipping audit + normalised<->pixel distance ratio + victim == raw_net(normalize(x)).

1. Prints the attack space, bounds, clamp site and to_pixel map per method (file:line references, verified by
   reading the code — see the report table for the exact quotes).
2. For the normalised-space method (CGBA / CGBA_H), computes d_n/d_p for 3 stored adversarial points; the ratio
   must lie in [1/sd_max, 1/sd_min] = [4.367, 4.444].
3. For all 6 victims, computes max|victim(x0) - raw_net((x0 - mu)/sd)|; it must be exactly 0.

Usage (GPU, D12 flags via _common.configure_precision at import):
    OMP_NUM_THREADS=8 python adapters/diag/diag_b4f_t6_domain.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import _common as C  # noqa: E402
import torch  # noqa: E402

MEAN = torch.tensor(C.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor(C.STD, dtype=torch.float32).view(1, 3, 1, 1)


def victim_mu_sd(victim):
    if hasattr(victim, "mu") and hasattr(victim, "sd"):
        return victim.mu, victim.sd
    if isinstance(victim, torch.nn.Sequential):
        kids = list(victim._modules.keys())
        if kids and kids[0] == "normalize":
            n = victim.normalize
            return n.mean, n.std
    return MEAN, STD


def first_source_of(rel: str) -> str:
    import csv
    with open(os.path.join(REPO_ROOT, rel), newline="", encoding="utf-8") as fh:
        return next(csv.DictReader(fh))["source"]


SPLIT_OF = {
    "vgg19": "splits/imagenet/ttba_github/vgg19_non_targeted.csv",
    "resnet50": "splits/imagenet/ttba_github/resnet50_non_targeted.csv",
    "vit_b_32": "splits/imagenet/ttba_github/vit_b_32_non_targeted.csv",
    "resnet50_places365": "splits/places365/resnet50_places365_non_targeted.csv",
    "densenet161_places365": "splits/places365/densenet161_places365_non_targeted.csv",
    "Salman2020Do_R50": "splits/robust/Salman2020Do_R50_non_targeted.csv",
}


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] GPU {torch.cuda.get_device_name(0)}  D12 matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} deterministic={torch.backends.cudnn.deterministic} "
          f"benchmark={torch.backends.cudnn.benchmark}")

    # ---- 1. attack-space summary (static; see report table for file:line quotes) ----
    print("\n## Attack space / bounds / clamp / to_pixel (per method)")
    print("| method | attack space | bounds (file:line) | clamp (file:line) | to_pixel |")
    print("|---|---|---|---|---|")
    print("| TtBA | pixel [0,1] | [0,1] via GeneralTorchModel.predict_label clamp (general_torch_model.py:53) | "
          "general_torch_model.py:53 (torch.clamp(images,0,1)); adapter ClampedVictim (_common.py:249-260) | to_pixel_clamp (_common.py:284-286) |")
    print("| CGBA / CGBA_H / CGBA_D8only | ImageNet-normalised | lb=normalize(0), ub=normalize(1) (run_cgba.py:178-179; drivers valid_bounds delta=255) | "
          "clip_image_values (utils.py:6-10) at proposed_attack.py:65,140,160,264 | to_pixel_unnormalize (_common.py:289-293) |")
    print("| ARS-OPT | pixel [0,1] | images+gg*xg unbounded, epsilon=10.0 gate only | "
          "no in-loop clamp; tracker clamps for the decision (to_pixel_clamp) | to_pixel_clamp (_common.py:284-286) |")
    print("| LGC_RC_EGC | latent (decoder -> pixel [0,1]) | ambient residual space, decoder clamps to [0,1] | "
          "space.to_image (lgc_bench, frozen) | identity (pixel space already) |")

    # ---- 2. d_n / d_p for normalised-space method (CGBA) ----
    print("\n## d_n / d_p for CGBA (normalised-space method), 3 stored points")
    points = [
        "Results/b4e2_smoke_cgba_resnet50/non_targeted/adv_store/resnet50/ILSVRC2012_val_00007450/CGBA_Q10000.npz",
        "Results/b4e2_smoke_cgba_resnet50/non_targeted/adv_store/resnet50/ILSVRC2012_val_00015433/CGBA_Q10000.npz",
        "Results/b4e2_smoke_cgba_resnet50/non_targeted/adv_store/resnet50/ILSVRC2012_val_00025151/CGBA_Q10000.npz",
    ]
    print("| point | d_p (pixel L2) | d_n (norm-space L2) | d_n/d_p | in [4.367,4.444]? |")
    print("|---|---|---|---|---|")
    for p in points:
        z = np.load(os.path.join(REPO_ROOT, p), allow_pickle=False)
        delta = z["delta"].astype(np.float32)
        sd = np.asarray(C.STD, dtype=np.float32).reshape(3, 1, 1)
        d_p = float(np.linalg.norm(delta.ravel()))
        d_n = float(np.linalg.norm((delta / sd).ravel()))
        r = d_n / d_p
        ok = 4.367 <= r <= 4.444
        print(f"| {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} | {d_p:.6f} | {d_n:.6f} | {r:.6f} | {ok} |")

    # ---- 3. victim(x0) == raw_net((x0-mu)/sd) for all 6 victims ----
    print("\n## max |victim(x0) - raw_net((x0-mu)/sd)| (must be 0)")
    print("| victim | raw net type | max abs diff |")
    print("|---|---|---|")
    all_zero = True
    for arch, split_rel in SPLIT_OF.items():
        name = first_source_of(split_rel)
        victim = C.load_victim_any(arch, device)
        raw = C.raw_net(victim)
        mu, sd = victim_mu_sd(victim)
        x0 = C.load_x0_any(arch, name, device).to(device)
        with torch.no_grad():
            y_wrapped = victim(x0)
            y_raw = raw((x0 - mu) / sd)
        diff = float((y_wrapped - y_raw).abs().max().item())
        all_zero = all_zero and (diff == 0.0)
        print(f"| {arch} | {type(raw).__module__}.{type(raw).__name__} | {diff:g} |")
    print(f"\n[*] victim == raw_net(normalize(x0)) for all 6 victims: {'YES (all 0)' if all_zero else 'NO'}")

    # 1/sd bounds for reference
    print(f"\n[*] 1/sd_min={1/0.229:.4f}  1/sd_max={1/0.225:.4f}  (sd=(0.229,0.224,0.225))")
    return 0 if all_zero else 1


if __name__ == "__main__":
    raise SystemExit(main())
