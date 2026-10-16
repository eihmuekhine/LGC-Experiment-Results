"""Evaluation-only metrics (never visible to the attacks)."""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch

_SSIM = None


def load_ssim():
    """Import torchmetrics lazily (its first import can be slow in a large conda env)."""
    global _SSIM
    if _SSIM is None:
        try:  # torchmetrics >= 1.0
            from torchmetrics.functional.image import structural_similarity_index_measure as fn
        except ImportError:  # older torchmetrics
            from torchmetrics.functional import structural_similarity_index_measure as fn
        _SSIM = fn
    return _SSIM


def build_lpips(device: torch.device):
    """LPIPS(VGG) on ``device``; falls back to CPU if the backend rejects it."""
    import warnings

    import lpips

    with warnings.catch_warnings():  # lpips still calls torchvision with pretrained=True
        warnings.simplefilter("ignore", UserWarning)
        net = lpips.LPIPS(net="vgg", verbose=False)
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        net = net.to(device).eval()
        with torch.no_grad():
            net(torch.zeros(1, 3, 64, 64, device=device), torch.zeros(1, 3, 64, 64, device=device))
        return net, device
    except Exception as err:  # pragma: no cover - backend specific
        print(f"[!] LPIPS on {device} failed ({err}); using CPU")
        return net.to("cpu").eval(), torch.device("cpu")


# Sub-LSB attacks (RMSE < ~0.5/255, common at 10k queries on easy images) are erased by 8-bit
# rounding even on a flat boundary; a 5% push cannot bring them back. Measured on ResNet50/Places365:
# every repair failure with the old ladder (<=1.05) had RMSE <= 0.64 LSB. Max scale 6 covers ~0.1 LSB.
REPAIR_SCALES = (1.0005, 1.001, 1.002, 1.005, 1.01, 1.02, 1.05,
                 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0)


@torch.no_grad()
def uint8_repair(
    z: torch.Tensor,
    origin: torch.Tensor,
    to_image: Callable[[torch.Tensor], torch.Tensor],
    project: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    label_fn: Callable[[torch.Tensor], int],
    is_adv_label: Callable[[int], bool],
    scales=REPAIR_SCALES,
) -> Dict[str, float]:
    """Make a boundary point survive 8-bit rounding by moving it slightly outward.

    Boundary points returned by binary search sit within the search tolerance
    of the decision boundary, so rounding to uint8 flips about half of them
    back. Here z is pushed along the ray origin -> z by the smallest factor in
    ``scales`` whose ROUNDED image is still adversarial. Every trial is one
    hard-label query of an actual 8-bit image; their number is reported so it
    can be added to the budget in the paper (<= len(scales)).
    """
    for n, s in enumerate(scales, start=1):
        img = to_image(project(origin + s * (z - origin)))
        q = torch.round(img * 255.0) / 255.0
        if is_adv_label(label_fn(q)):
            return {"repair_scale": s, "repair_queries": n, "adv_uint8_repaired": 1,
                    "pix_l2_uint8_repaired": float((q - x0).flatten().norm().item())}
    return {"repair_scale": float("nan"), "repair_queries": len(scales), "adv_uint8_repaired": 0,
            "pix_l2_uint8_repaired": float("nan")}


@torch.no_grad()
def image_metrics(
    img: torch.Tensor,
    x0: torch.Tensor,
    label_fn: Callable[[torch.Tensor], int],
    is_adv_label: Callable[[int], bool],
    lpips_net=None,
    lpips_device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Metrics of an adversarial IMAGE in [0,1] w.r.t. the clean image x0."""
    diff = img - x0
    n = diff.numel()
    l2 = float(diff.flatten().norm().item())
    out = {
        "pix_l2": l2,
        "pix_linf": float(diff.abs().max().item()),
        "rmse": l2 / math.sqrt(n),
    }
    img_c, x0_c = img.detach().float().cpu(), x0.detach().float().cpu()
    out["ssim"] = float(load_ssim()(img_c, x0_c, data_range=1.0).item())
    if lpips_net is not None:
        dev = lpips_device or img.device
        out["lpips"] = float(lpips_net(img.to(dev) * 2 - 1, x0.to(dev) * 2 - 1).item())
    else:
        out["lpips"] = float("nan")
    out["adv_float"] = int(is_adv_label(label_fn(img)))
    q = torch.round(img * 255.0) / 255.0  # what is actually saved as PNG / uint8
    out["adv_uint8"] = int(is_adv_label(label_fn(q)))
    out["pix_l2_uint8"] = float((q - x0).flatten().norm().item())
    return out
