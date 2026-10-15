"""Shared infrastructure for the LGC-EGC baseline adapters (paper-runs branch, phase B2).

Every adapter (run_ttba_v2.py, run_cgba.py, run_hsja.py, run_arsopt.py, run_signopt.py) imports from this
module.  It centralises the pieces that MUST be identical for every method so that fairness rules F1-F12
(skill lgc-baselines) hold by construction:

  * the contract checkpoints/budget are read from ``contract/paper_contract.yaml`` at runtime (never hard-coded),
    overriding the evalkit defaults (``hardlabel_tracker`` budgets and ``adapter_template.BUDGETS``);
  * the shared victim loader (official TtBA weights, ImageNet normalisation inside, input [0,1] NCHW);
  * the shared image loader (TtBA protocol: Resize(256) -> CenterCrop(224) -> ToTensor), i.e. the same x0 for
    every method (F3);
  * the decision-only ``HardLabelTracker`` from ``tools/eval/hardlabel_tracker.py``;
  * the contract artefact writers (per_checkpoint CSV, runs CSV, adv_store, env JSON, run_manifest.json)
    and manifest-matched resume.

Nothing here draws random numbers or changes any attack algorithm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)  # advstore.py at repo root + lgc_bench package
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))  # hardlabel_tracker.py, advstore.py

from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402

import torch  # noqa: E402


def configure_precision() -> None:
    """D12 precision parity: every adapter runs the frozen-engine FP32 backend.

    ``lgc_bench/env.py:configure_backend`` sets ``cuda.matmul.allow_tf32 = False``, ``cudnn.allow_tf32 = False``,
    ``cudnn.deterministic = True``, ``cudnn.benchmark = False`` for the two EGC runners.  The baseline adapters
    historically inherited the PyTorch defaults (cudnn TF32 enabled, benchmark auto), which made the decision
    oracle depend on the GPU generation (D16: RTX 5090 vs RTX PRO 6000 vs RTX 4080 SUPER give different TF32
    results for several victims).  This function makes every adapter attack the same FP32 classifier so that
    results are reproducible across the GPU migration.  Called at import time, before any victim/model is built."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


configure_precision()


def backend_flags() -> dict:
    """The five precision-parity backend flags read from torch at run time (skill lgc-preflight §4).

    Written into every adapter env_*.json under the literal ``backend.*`` keys so that human_preflight section 1
    and the D12 parity audit see exactly what the attack ran with.  ``float32_matmul_precision`` is read via
    ``torch.get_float32_matmul_precision()`` (the ``torch.backends`` attribute of the same name does not exist
    in torch 2.7)."""
    return {
        "backend.cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "backend.cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "backend.cudnn_deterministic": torch.backends.cudnn.deterministic,
        "backend.cudnn_benchmark": torch.backends.cudnn.benchmark,
        "backend.float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

# --- E10 (Places365) + E11 (RobustBench) ---
PLACES365_MODEL_DIR = "/root/autodl-tmp/models/places365"
PLACES365_DATA_DIR = "/root/autodl-tmp/places365_data"
PLACES365_VAL_DIR = os.path.join(PLACES365_DATA_DIR, "val_256")
PLACES365_LABELS_PATH = os.path.join(PLACES365_DATA_DIR, "places365_val.txt")
# arch -> (weight file, sha256). sha256 from reports/E10_sources.md (official MIT Places365 weights).
PLACES365_WEIGHTS = {
    "resnet50_places365": ("resnet50_places365.pth.tar",
                           "46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5"),
    "densenet161_places365": ("densenet161_places365.pth.tar",
                              "b311cb0e8bea8ecf34dee58da317c6217894db94721005c54cadcb7f2f2f9598"),
}
ROBUST_MODEL_NAME = "Salman2020Do_R50"
ROBUST_MODEL_DIR = os.path.expanduser("~/.cache/robustbench/models")
ROBUST_WEIGHT_FILE = "Salman2020Do_R50.pt"
ROBUST_WEIGHT_SHA256 = "1835a5bbe5e31ef35915889dc949b96528bed513205bb9ec1cac80c32c84ce46"  # reports/E11_sources.md

CONTRACT_PATH = os.path.join(REPO_ROOT, "contract", "paper_contract.yaml")

CKPT_FIELDS = ["arch", "mode", "image", "method", "budget", "found", "state_query", "pix_l2", "pix_linf",
               "adv_float", "adv_uint8", "source_label", "target_label", "ssim", "lpips"]
RUN_FIELDS = ["arch", "mode", "image", "method", "queries", "nonfinite_queries", "status", "wall_time_s", "error"]


# --------------------------------------------------------------------------------------------- contract
def load_contract(path: Optional[str] = None) -> dict:
    import yaml

    p = path or CONTRACT_PATH
    if not os.path.exists(p):
        raise SystemExit(f"[!] contract not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def contract_checkpoints(contract: Optional[dict] = None) -> List[int]:
    c = contract or load_contract()
    return [int(x) for x in c["constants"]["checkpoints"]]


def contract_budget(contract: Optional[dict] = None) -> int:
    c = contract or load_contract()
    return int(c["constants"]["budget"])


# ----------------------------------------------------------------------------------------------- hashing
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------ shared victim
def load_victim(arch: str, ckpt_dir: str, device: str):
    """The official TtBA victim (torchvision constructor + load_state_dict from <ckpt_dir>/<file>).

    Identical weights to our runner and to TtBA's GeneralTorchModel.  Input [N,3,224,224] in [0,1]; ImageNet
    mean/std normalisation inside.  The weight file sha256 is asserted (U3)."""
    from lgc_bench.baseline_protocol import TTBA_GH_WEIGHTS, build_ttba_github_victim

    if arch not in TTBA_GH_WEIGHTS:
        raise SystemExit(f"[!] no official TtBA weight file for arch {arch!r} (known: {sorted(TTBA_GH_WEIGHTS)})")
    wname = TTBA_GH_WEIGHTS[arch]
    wpath = os.path.join(ckpt_dir, wname)
    if not os.path.exists(wpath):
        raise SystemExit(f"[!] victim weights {wpath} not found")
    victim = build_ttba_github_victim(arch, ckpt_dir, device)
    victim.victim_weight_file = wname
    victim.victim_weight_sha256 = sha256_file(wpath)
    return victim


def load_victim_any(arch: str, device: str, ckpt_dir: str = "/root/autodl-tmp/TtBA/model"):
    """Dispatch the shared victim loader to the E10 (Places365), E11 (RobustBench) or main-ImageNet victim.

    Every victim has the SAME interface: input [N,3,224,224] in [0,1] -> logits, with ImageNet normalisation
    inside the wrapper, and a ``victim_weight_file`` / ``victim_weight_sha256`` attribute."""
    if arch == "kat_linear":
        return build_kat_linear_victim(device)
    if arch in PLACES365_WEIGHTS:
        return build_places365_victim(arch, device)
    if arch == ROBUST_MODEL_NAME:
        return build_robust_victim(device)
    return load_victim(arch, ckpt_dir, device)


def raw_net(victim):
    """The un-normalised network an attack in ImageNet-normalised space must call (CGBA faithful path).

    The ImageNet TtBA victims wrap the raw net as ``_Norm.m`` (baseline_protocol.py), the Places365 victims as
    ``NormalizedVictim.net``, and the RobustBench victim is a Sequential whose first submodule is the
    ``normalize`` layer with the raw net as ``.model``."""
    m = getattr(victim, "m", None)
    if m is not None:
        return m
    net = getattr(victim, "net", None)
    if net is not None:
        return net
    if isinstance(victim, torch.nn.Sequential):
        kids = list(victim._modules.keys())
        if kids and kids[0] == "normalize":
            return victim.model
    return victim


def build_places365_victim(arch: str, device: str):
    """Official MIT Places365 victim (E10): torchvision constructor with ``num_classes=365`` + the official
    ``<base>_places365.pth.tar`` state dict (``module.`` prefix stripped).  ImageNet mean/std normalisation inside;
    input [N,3,224,224] in [0,1]; returns logits.  The weight file + its sha256 (reports/E10_sources.md) are stored
    on the module."""
    import torchvision.models as tvm

    if arch not in PLACES365_WEIGHTS:
        raise SystemExit(f"[!] no Places365 weight file for arch {arch!r} (known: {sorted(PLACES365_WEIGHTS)})")
    wname, wsha256 = PLACES365_WEIGHTS[arch]
    wpath = os.path.join(PLACES365_MODEL_DIR, wname)
    if not os.path.exists(wpath):
        raise SystemExit(f"[!] Places365 weights {wpath} not found")
    base = arch[: -len("_places365")]
    net = getattr(tvm, base)(num_classes=365)
    ck = torch.load(wpath, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    if base == "densenet161":
        # official 2018 checkpoint uses the old torchvision dense-layer names (norm.1/conv.1 ...);
        # current torchvision uses norm1/conv1 (missing num_batches_tracked is handled by _BatchNorm).
        sd = {k.replace("norm.1", "norm1").replace("norm.2", "norm2")
              .replace("conv.1", "conv1").replace("conv.2", "conv2"): v for k, v in sd.items()}
    net.load_state_dict(sd)
    victim = NormalizedVictim(net).to(device).eval()
    victim.victim_weight_file = wname
    victim.victim_weight_sha256 = wsha256
    return victim


def build_robust_victim(device: str):
    """RobustBench victim (E11): ``Salman2020Do_R50`` (imagenet, Linf).  The model zoo wraps it with an internal
    ImageNet ``normalize`` layer, so the input is [0,1] and logits are returned.  Weight file + sha256 stored."""
    import robustbench.utils as rbu

    model = rbu.load_model(ROBUST_MODEL_NAME, model_dir=ROBUST_MODEL_DIR, dataset="imagenet", threat_model="Linf")
    model = model.to(device).eval()
    model.victim_weight_file = ROBUST_WEIGHT_FILE
    model.victim_weight_sha256 = ROBUST_WEIGHT_SHA256
    return model


# ----------------------------------------------------------------------------------- kat_linear victim (B4g)
# Known-answer test victim: a linear classifier with K=10 classes over ImageNet-normalised features.
#   z = (x - mu)/sd ; logits_k = <W_k, z> + b_k
# W_k = bilinear upsampling of a (3,14,14) N(0,1) array to (3,224,224) (seed 0), b ~ N(0,1) (seed 0).
# The pixel-space gradient of logit k is g_k = W_k / sd (per channel), so every check of the B4g known-answer
# test is analytic.  The victim counts every forward pass (per image) and, when KAT_FORWARD_COUNT_FILE is set,
# dumps that count at process exit (used by adapters/diag/kat_linear.py check (d)).
KAT_LINEAR_K = 10
KAT_DATA_DIR = os.path.join(REPO_ROOT, "generated", "smoke_canary", "kat")


def kat_linear_w_b():
    """Deterministic W (K,3,224,224) float32 and b (K,) float32 for the kat_linear victim (seed 0)."""
    import torch.nn.functional as F

    rng = np.random.RandomState(0)
    A = rng.randn(KAT_LINEAR_K, 3, 14, 14).astype(np.float32)          # (K,3,14,14) N(0,1)
    b = rng.randn(KAT_LINEAR_K).astype(np.float32)                     # (K,) N(0,1)
    W = F.interpolate(torch.from_numpy(A), size=(IMG_SIZE, IMG_SIZE),
                      mode="bilinear", align_corners=False).numpy()    # (K,3,224,224)
    return W, b


class KatLinearRaw(torch.nn.Module):
    """Un-normalised linear map for the kat_linear victim: normalised input z -> logits = W vec(z) + b.

    This is the analogue of the torchvision net the real victims expose as ``_Norm.m`` / ``NormalizedVictim.net``:
    ``C.raw_net(victim)`` returns it, so the CGBA faithful path feeds the normalised tensor straight into this net
    (exactly as the official CGBA drivers feed the un-normalised ResNet-50).  It counts every forward pass (per
    image); when KAT_FORWARD_COUNT_FILE is set, the count is dumped at process exit (adapters/diag/kat_linear.py
    check (d))."""

    def __init__(self, W, b):
        super().__init__()
        self.register_buffer("W", torch.from_numpy(W))
        self.register_buffer("b", torch.from_numpy(b))
        self.forward_count = 0
        self._count_file = os.environ.get("KAT_FORWARD_COUNT_FILE")
        if self._count_file:
            import atexit
            atexit.register(lambda: self._dump_forward_count())

    def _dump_forward_count(self) -> None:
        try:
            with open(self._count_file, "w", encoding="utf-8") as fh:
                fh.write(str(self.forward_count))
        except Exception:  # noqa: BLE001
            pass

    def forward(self, z):
        self.forward_count += int(z.shape[0])
        return torch.einsum("kchw,bchw->bk", self.W, z) + self.b


class KatLinearVictim(torch.nn.Module):
    """Linear victim: input x in [0,1]^(3x224x224) -> logits (K,); z = (x - mu)/sd; logits = W vec(z) + b.

    The input is clamped to [0,1] (its stated domain) before normalisation, mirroring how the real victims are fed
    only in-range images (TtBA wraps them in ClampedVictim, CGBA clips in-loop).  The un-normalised linear map is
    exposed as ``.m`` so the CGBA faithful path (``C.raw_net``) receives the normalised tensor directly."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        W, b = kat_linear_w_b()
        self.m = KatLinearRaw(W, b)
        self.register_buffer("mu", torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1))
        wb = np.concatenate([W.ravel(), b.ravel()])
        self.victim_weight_file = "kat_linear_seed0"
        self.victim_weight_sha256 = sha256_bytes(wb.astype(np.float32).tobytes())

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        z = (x - self.mu) / self.sd
        return self.m(z)


def build_kat_linear_victim(device: str):
    return KatLinearVictim(device).to(device).eval()


def robust_victim_info(device: str = "cpu") -> dict:
    """Forward pass on a random (1,3,224,224) [0,1] batch; report the victim type, whether the ImageNet
    normalisation is inside the model, and a sample argmax (E11 report)."""
    victim = build_robust_victim(device)
    kids = list(victim._modules.keys()) if isinstance(victim, torch.nn.Sequential) else []
    internal_norm = bool(kids) and kids[0] == "normalize"
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        logits = victim(x)
    return {
        "type": f"{type(victim).__module__}.{type(victim).__name__}",
        "submodules": kids,
        "internal_normalisation": internal_norm,
        "input": f"rand(1,3,{IMG_SIZE},{IMG_SIZE}) in [0,1]",
        "logits_shape": list(logits.shape),
        "sample_argmax": int(logits.argmax(1).item()),
        "victim_weight_file": getattr(victim, "victim_weight_file", None),
        "victim_weight_sha256": getattr(victim, "victim_weight_sha256", None),
    }


class NormalizedVictim(torch.nn.Module):
    """Victim that applies ImageNet mean/std normalisation inside, then returns logits.  Input [0,1] NCHW."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("sd", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class ClampedVictim(torch.nn.Module):
    """Victim that clamps its [0,1] input before the underlying victim.

    Reproduces the clamping done by the official model wrappers (TtBA GeneralTorchModel.predict_label and
    attackbox PytorchModel.predict_label clamp to [0,1]); it is part of the model object, not the attack."""

    def __init__(self, victim):
        super().__init__()
        self.victim = victim

    def forward(self, x):
        return self.victim(x.clamp(0.0, 1.0))


class ClampedVictimNormalized(torch.nn.Module):
    """Victim that clamps a NORMALISED-space input channel-wise to ``[lb, ub]`` before the underlying raw net.

    This is the D19 oracle-domain clamp for CGBA / CGBA_H: the official ``bin_search`` (:77-85) and
    ``SemiCircular_boundary_search`` (:179-191) probe in ImageNet-normalised space without clipping, while the
    decision and the stored pixel image were computed on two different inputs (unclamped normalised vs clamped
    pixel).  Wrapping the raw net with per-channel bounds ``lb = normalize(0)`` and ``ub = normalize(1)`` makes the
    decision use the exact clamped [0,1] pixel image (``normalize`` and ``to_pixel_unnormalize`` are inverse up to
    the clamp), which is what ``to_pixel_unnormalize`` stores.  ``lb``/``ub`` are registered as buffers; they are
    built from ``x0`` on the attack's device (so no device/dtype transfer is relied upon)."""
    def __init__(self, victim, lb, ub):
        super().__init__()
        self.victim = victim
        self.register_buffer("lb", lb.clone().detach())
        self.register_buffer("ub", ub.clone().detach())

    def forward(self, x):
        return self.victim(torch.clamp(x, self.lb, self.ub))


class UnnormalizeVictim(torch.nn.Module):
    """Victim that accepts an ImageNet-normalised tensor and un-normalises it to [0,1] before the underlying
    victim.  CGBA works in normalised space (mean/std applied) but the shared victim expects [0,1].
    The mean/std constants are materialised on the input's device/dtype (the attack input may be on cuda while the
    wrapper is a plain nn.Module that was never moved)."""

    def __init__(self, victim):
        super().__init__()
        self.victim = victim
        self._cache: Dict[tuple, tuple] = {}

    def forward(self, x):
        key = (x.device, x.dtype)
        if key not in self._cache:
            mu = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            sd = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            self._cache[key] = (mu, sd)
        mu, sd = self._cache[key]
        return self.victim(x * sd + mu)


def to_pixel_clamp(x):
    """Identity in [0,1] pixel space, clamped (the image actually classified)."""
    return x.clamp(0.0, 1.0)


def to_pixel_unnormalize(x):
    """ImageNet-normalised -> [0,1] pixel space (for CGBA)."""
    mu = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    sd = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * sd + mu).clamp(0.0, 1.0)


def normalize(x):
    """[0,1] -> ImageNet-normalised (for feeding CGBA's normalised-space attack)."""
    mu = torch.tensor(MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    sd = torch.tensor(STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mu) / sd


# ------------------------------------------------------------------------------------------ non-finite guard
class NonfiniteGuard(torch.nn.Module):
    """Wraps the model the HardLabelTracker calls and counts queries whose input contains a non-finite element.

    Observation only (B4e T1c): it never alters the model's output, the query accounting or the attack trajectory.
    A "query" is one image of the input batch (the tracker counts batches element-wise); the guard counts, over the
    whole attack, how many of those images carried a NaN or +/-inf anywhere.  Written to runs*.csv as
    ``nonfinite_queries``.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nonfinite_queries = 0

    def forward(self, x):
        xb = x if x.dim() == 4 else x.unsqueeze(0)
        nonfinite = torch.isnan(xb).any(dim=(1, 2, 3)) | torch.isinf(xb).any(dim=(1, 2, 3))
        self.nonfinite_queries += int(nonfinite.sum().item())
        return self.model(x)


class _GuardedHardLabelTracker(HardLabelTracker):
    """HardLabelTracker that wraps its model in a NonfiniteGuard (B4e T1c).  Identical bookkeeping otherwise."""

    def __init__(self, model, *args, **kwargs):
        self._nf_guard = NonfiniteGuard(model)
        super().__init__(self._nf_guard, *args, **kwargs)

    @property
    def nonfinite_queries(self) -> int:
        return self._nf_guard.nonfinite_queries


# Every adapter constructs its tracker through C.HardLabelTracker, so rebinding it here applies the non-finite
# guard to all of them (run_cgba / run_arsopt / run_hsja / run_signopt use C.HardLabelTracker; run_ttba_v2.py is
# pointed at C.HardLabelTracker too).  hardlabel_tracker.HardLabelTracker itself is untouched (selftest tests it).
HardLabelTracker = _GuardedHardLabelTracker


# -------------------------------------------------------------------------------------------- shared images
def load_split_rows(split_csv: str) -> List[dict]:
    with open(split_csv, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_image(data_dir: str, name: str, device: str):
    """x0 as a (1,3,224,224) tensor in [0,1] using the TtBA protocol loader (Resize 256 -> CenterCrop 224)."""
    from lgc_bench.baseline_protocol import ttba_github_tensor

    path = os.path.join(data_dir, "images", name)
    x = ttba_github_tensor(path)  # (3,224,224) float32 in [0,1]
    return x.unsqueeze(0).to(device)


def stem_of(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def load_image_places365(name: str, device: str):
    """x0 as a (1,3,224,224) tensor in [0,1] using the Places365 protocol (Resize 256 -> CenterCrop 224 -> ToTensor)."""
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    path = os.path.join(PLACES365_VAL_DIR, name)
    with open(path, "rb") as fh:
        im = Image.open(fh)
        x = tf(im.convert("RGB"))
    return x.unsqueeze(0).to(device)


def load_places365_labels() -> Dict[str, int]:
    """{filename: class_index} parsed from places365_val.txt (class indices 0..364)."""
    labels: Dict[str, int] = {}
    with open(PLACES365_LABELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels[parts[0]] = int(parts[1])
    return labels


def load_x0_any(arch: str, name: str, device: str, data_dir: str = "/root/autodl-tmp/ttba_mini_flat"):
    """x0 (or a target image) as a (1,3,224,224) tensor in [0,1] for ANY victim, using the shared TtBA protocol.

    Places365 -> Places365 loader (Resize 256 -> CenterCrop 224, flat val_256); RobustBench -> the TtBA-mini
    flat ImageNet loader; the main ImageNet victims -> ``load_image(data_dir, ...)`` (unchanged)."""
    if arch == "kat_linear":
        return load_image(KAT_DATA_DIR, name, device)
    if arch in PLACES365_WEIGHTS:
        return load_image_places365(name, device)
    if arch == ROBUST_MODEL_NAME:
        return load_image("/root/autodl-tmp/ttba_mini_flat", name, device)
    return load_image(data_dir, name, device)


# -------------------------------------------------------------------------------------------------- seeding
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------------------------------ CSV
def append_csv(path: str, fields: Sequence[str], rows: Iterable[dict]) -> None:
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def done_images(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["image"] for r in csv.DictReader(fh) if r.get("status") in ("ok", "budget")}


# --------------------------------------------------------------------------------------------------- env + manifest
def git_commit(repo: str) -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def build_env(method: str, arch: str, mode: str, args, contract: dict, params: dict, hyper_sources: dict,
              split_csv_sha256: str, victim_weight_file: str, victim_weight_sha256: str,
              adapter_sha256: str, tracker_sha256: str, repo_commit: str, checkpoints: List[int]) -> dict:
    return {
        "method": method, "arch": arch, "mode": mode,
        "repo": os.path.abspath(args.repo), "repo_commit": repo_commit,
        "params": params, "hyper_parameter_sources": hyper_sources,
        "split_csv": os.path.abspath(args.split_csv), "split_csv_sha256": split_csv_sha256,
        "victim_weight_file": victim_weight_file, "victim_weight_sha256": victim_weight_sha256,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "budget": contract_budget(contract), "max_queries": contract_budget(contract),
        "checkpoints": checkpoints, "budgets": checkpoints,
        "adapter_sha256": adapter_sha256, "tracker_sha256": tracker_sha256,
        "precision": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        **backend_flags(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "adv_store_dtype": os.environ.get("ADV_STORE_DTYPE", "float16"),
        "ADV_STORE_BUDGETS": "unset" if not os.environ.get("ADV_STORE_BUDGETS") else os.environ["ADV_STORE_BUDGETS"],
    }


def build_manifest(env: dict, contract: dict) -> dict:
    m = {"method": env["method"], "arch": env["arch"], "mode": env["mode"],
         "split_csv": env["split_csv"], "split_csv_sha256": env["split_csv_sha256"],
         "victim_weight_file": env["victim_weight_file"], "victim_weight_sha256": env["victim_weight_sha256"],
         "repo": env["repo"], "repo_commit": env["repo_commit"],
         "adapter_sha256": env["adapter_sha256"], "tracker_sha256": env["tracker_sha256"],
         "contract_sha256": env["contract_sha256"], "hyper_parameters": env["params"],
         "hyper_parameter_sources": env["hyper_parameter_sources"],
         "torch": env["torch"], "cuda": env["cuda"], "gpu_name": env["gpu"],
         "budget": env["budget"], "checkpoints": env["checkpoints"]}
    for key in ("cgba_variant", "paper_faithful_fix", "oracle_domain"):
        if key in env:
            m[key] = env[key]
    return m


def check_manifest(run_dir: str, manifest: dict) -> None:
    """Refuse to resume when an existing run_manifest.json differs from the current configuration."""
    p = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as fh:
        old = json.load(fh)
    for key in manifest:
        if key in ("gpu_name", "torch", "cuda"):
            continue  # environment keys may legitimately differ across a migration; the split/commit/weights must not
        if old.get(key) != manifest[key]:
            raise SystemExit(f"[!] run_manifest.json mismatch on {key!r}: existing {old.get(key)!r} != {manifest[key]!r}; "
                             f"refusing to resume. Move/delete {run_dir} or re-run with identical configuration.")


# -------------------------------------------------------------------------------------------------- CLI + loop
def make_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True,
                    choices=["vgg19", "resnet50", "vit_b_32", "kat_linear",
                             "resnet50_places365", "densenet161_places365", "Salman2020Do_R50"])
    ap.add_argument("--mode", choices=["non_targeted", "targeted"], required=True)
    ap.add_argument("--cgba_variant", choices=["fixed", "d8only"], default="fixed",
                    help="CGBA adapter only: 'fixed' = D11 acos clip (method id CGBA / CGBA_H); "
                         "'d8only' = D8 count fix only, no clip (method id CGBA_D8only, non_targeted only).")
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/ttba_mini_flat")
    ap.add_argument("--ckpt_dir", default="/root/autodl-tmp/TtBA/model")
    ap.add_argument("--repo", required=True, help="official repository clone (git commit is recorded)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=200, help="attack the first N rows of the split (rows 1-200)")
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap


def run(method: str, description: str, params: dict, hyper_sources: dict,
        make_tracker: Callable[..., HardLabelTracker],
        run_official: Callable[..., None],
        adapter_path: str,
        extra_env: Optional[Callable[[dict, dict], dict]] = None) -> int:
    """Shared main loop.  make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget) -> tracker;
    run_official(tracker, x0, y_src, x_tgt, y_tgt, params)."""
    ap = make_parser(description)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))  # the official clone, imported unchanged by run_official

    contract = load_contract()
    checkpoints = contract_checkpoints(contract)
    budget = contract_budget(contract)

    adapter_sha256 = sha256_file(adapter_path)
    tracker_sha256 = sha256_file(os.path.join(REPO_ROOT, "tools", "eval", "hardlabel_tracker.py"))
    repo_commit = git_commit(args.repo)
    split_csv_sha256 = sha256_file(args.split_csv)

    run_dir = os.path.join(args.out_dir, args.mode)
    os.makedirs(run_dir, exist_ok=True)
    ck_path = os.path.join(run_dir, f"per_checkpoint_{method}.csv")
    run_path = os.path.join(run_dir, f"runs_{method}.csv")

    victim = load_victim_any(args.arch, args.device, args.ckpt_dir)
    env = build_env(method, args.arch, args.mode, args, contract, params, hyper_sources, split_csv_sha256,
                    victim.victim_weight_file, victim.victim_weight_sha256, adapter_sha256, tracker_sha256,
                    repo_commit, checkpoints)
    if extra_env:
        env.update(extra_env(args, contract))
    manifest = build_manifest(env, contract)
    check_manifest(run_dir, manifest)
    with open(os.path.join(run_dir, f"env_{method}.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=1)
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    rows = load_split_rows(args.split_csv)[: args.limit]
    done = done_images(run_path)

    for idx, r in enumerate(rows):
        name = r["source"]
        stem = stem_of(name)
        if stem in done:
            continue
        y_src = int(r["source_label"])
        y_tgt = int(r["target_label"]) if (args.mode == "targeted" and r.get("target_label") not in ("", None)) else None
        x0 = load_x0_any(args.arch, name, args.device, args.data_dir)
        x_tgt = load_x0_any(args.arch, r["target"], args.device, args.data_dir) if (args.mode == "targeted" and r.get("target")) else None

        seed = args.seed_base + idx
        seed_all(seed)

        tracker = make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget)
        t0, status, err = time.time(), "ok", ""
        try:
            run_official(tracker, x0, y_src, x_tgt, y_tgt, params)
        except BudgetExhausted:
            status = "budget"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            status, err = "error", f"{type(e).__name__}: {e}"

        if status == "error":
            # No per_checkpoint rows for an errored image: it will be re-attacked on resume (no duplicates).
            append_csv(run_path, RUN_FIELDS, [{"arch": args.arch, "mode": args.mode, "image": stem,
                                               "method": method, "queries": tracker.queries,
                                               "nonfinite_queries": tracker.nonfinite_queries, "status": status,
                                               "wall_time_s": round(time.time() - t0, 2), "error": err}])
            print(f"[{idx + 1}/{len(rows)}] {stem} {method} ERROR {err}", flush=True)
            continue

        ck_rows = tracker.finalize(os.path.join(run_dir, "adv_store"), args.arch, stem, method, args.mode,
                                   eval_model=victim)
        for row in ck_rows:
            row["source_label"] = y_src
            row["target_label"] = "" if y_tgt is None else y_tgt
        append_csv(ck_path, CKPT_FIELDS, ck_rows)
        tracker.save_trace(os.path.join(run_dir, "traces", args.arch, stem, f"{method}.npz"))
        append_csv(run_path, RUN_FIELDS, [{"arch": args.arch, "mode": args.mode, "image": stem, "method": method,
                                            "queries": tracker.queries,
                                            "nonfinite_queries": tracker.nonfinite_queries, "status": status,
                                            "wall_time_s": round(time.time() - t0, 2), "error": err}])
        last = ck_rows[-1]
        print(f"[{idx + 1}/{len(rows)}] {stem} {method} q={tracker.queries} "
              f"L2@{last['budget']}={last['pix_l2']} {status} {time.time() - t0:.1f}s", flush=True)
    return 0
