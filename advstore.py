"""Evaluation-only store of adversarial images at query checkpoints.

Purpose: compute ANY image-quality metric later (SSIM, MS-SSIM, LPIPS-alex/vgg, DISTS, FSIM, PSNR, ...) without
re-running an attack.  The store never touches the victim model, never draws random numbers and never changes the
attack state; it only copies the image that the runner already evaluates at each checkpoint.

Layout (one directory per run_dir, i.e. per output folder and mode):

    <root>/<arch>/<image_stem>/x0.npy                  clean input, float32, shape (3, H, W), values in [0, 1]
    <root>/<arch>/<image_stem>/<method>_Q<budget>.npz  delta = x_adv - x0 (float16 by default) + metadata
    <root>/x0_mismatch.csv                             written only if two methods saw different clean inputs

Environment variables:
    ADV_STORE_DTYPE    float16 (default, ~0.3 MB per image and checkpoint) | float32 (exact, 2x size)
    ADV_STORE_BUDGETS  optional space-separated list; store only these checkpoints (default: all)
"""
from __future__ import annotations

import csv
import os
import re
from typing import Any, Optional

import numpy as np

_X0_TOL = 1e-5  # max |x0_a - x0_b| tolerated between two methods on the same image (preprocessing identity, F1)


def _to_chw(x: Any) -> np.ndarray:
    """Convert a torch tensor or array of shape (1,3,H,W) / (3,H,W) to float32 numpy (3,H,W)."""
    try:
        import torch  # noqa: PLC0415  (optional dependency)

        if isinstance(x, torch.Tensor):
            x = x.detach().to(dtype=torch.float32, device="cpu").numpy()
    except ImportError:
        pass
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise ValueError(f"advstore expects a single image, got batch shape {a.shape}")
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"advstore expects (3,H,W) or (1,3,H,W), got {a.shape}")
    return a


def _safe(name: str) -> str:
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def _atomic_npy(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _wanted(budget: int) -> bool:
    sel = os.environ.get("ADV_STORE_BUDGETS", "").split()
    return not sel or str(int(budget)) in sel


def store_adv(root: str, arch: str, image: str, method: str, budget: int, x_adv: Any, x0: Any,
              state_query: int = -1, adv_flag: Any = -1, mode: Optional[str] = None) -> Optional[str]:
    """Store the checkpoint image of one (image, method, budget).  Returns the written path (or None if skipped).

    Exact quantities (float32 L2 and Linf of the stored state) are saved as metadata so that the float16 delta
    can be audited against them later (eval_metrics_offline.py reports the relative difference).
    """
    if not _wanted(budget):
        return None
    d = os.path.join(root, str(arch), _safe(image))
    os.makedirs(d, exist_ok=True)
    x0n, xa = _to_chw(x0), _to_chw(x_adv)
    if x0n.shape != xa.shape:
        raise ValueError(f"advstore: x0 {x0n.shape} and x_adv {xa.shape} differ")
    p0 = os.path.join(d, "x0.npy")
    if not os.path.exists(p0):
        _atomic_npy(p0, x0n)
    else:
        ref = np.load(p0)
        diff = float(np.max(np.abs(ref - x0n))) if ref.shape == x0n.shape else float("inf")
        if diff > _X0_TOL:  # different preprocessing between methods -> keep both and flag it
            _atomic_npy(os.path.join(d, f"x0_{_safe(method)}.npy"), x0n)
            new = not os.path.exists(os.path.join(root, "x0_mismatch.csv"))
            with open(os.path.join(root, "x0_mismatch.csv"), "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["arch", "image", "method", "max_abs_diff"])
                w.writerow([arch, image, method, f"{diff:.3e}"])
    delta = xa - x0n
    try:
        adv_i = int(float(adv_flag))
    except (TypeError, ValueError):
        adv_i = -1
    dtype = np.float32 if os.environ.get("ADV_STORE_DTYPE", "float16") == "float32" else np.float16
    out = os.path.join(d, f"{_safe(method)}_Q{int(budget)}.npz")
    tmp = out + ".tmp.npz"
    np.savez_compressed(
        tmp, delta=delta.astype(dtype), budget=np.int64(budget), state_query=np.int64(int(state_query)),
        adv_flag=np.int64(adv_i),
        l2=np.float64(np.linalg.norm(delta.ravel())), linf=np.float64(np.max(np.abs(delta))),
        mode=np.array(str(mode or "")), method=np.array(str(method)), image=np.array(str(image)),
        arch=np.array(str(arch)),
    )
    os.replace(tmp, out)
    return out


def load_adv(npz_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (x_adv, x0, meta) for one stored file; x_adv is clipped to [0, 1] after float16 reconstruction."""
    d = os.path.dirname(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    meta = {k: z[k].item() if z[k].shape == () else z[k] for k in z.files if k != "delta"}
    method = str(meta.get("method", ""))
    alt = os.path.join(d, f"x0_{_safe(method)}.npy")
    x0 = np.load(alt if os.path.exists(alt) else os.path.join(d, "x0.npy")).astype(np.float32)
    x_adv = np.clip(x0 + z["delta"].astype(np.float32), 0.0, 1.0)
    return x_adv, x0, meta
