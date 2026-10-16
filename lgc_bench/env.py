"""Device selection, backend configuration, seeding and random streams.

NVIDIA / AutoDL notes
---------------------
* ``configure_backend`` disables TF32 (exact FP32, identical decisions to the
  CPU/MPS runs; on an RTX 4090 TF32 gives almost no speed-up for these models)
  and makes cuDNN deterministic. ``cudnn.benchmark`` stays off because the
  query batch size changes constantly.

Apple Silicon notes
-------------------
* ``PYTORCH_ENABLE_MPS_FALLBACK=1`` lets the few ops without an MPS kernel run
  on the CPU instead of raising. It must be set before ``import torch``.
* MPS has no float64. All device tensors are float32; float64 linear algebra
  (QR / eigh for the Jacobian subspace) is done explicitly on the CPU.
* All random numbers are drawn from CPU ``torch.Generator`` objects and then
  moved to the device. The random streams are therefore identical on MPS,
  CUDA and CPU, which keeps paired comparisons reproducible across machines.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import random  # noqa: E402
import warnings  # noqa: E402
from typing import Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def pick_device(preference: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for 'auto'."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: torch.device, deterministic: bool = True, allow_tf32: bool = False) -> None:
    """Numerically faithful defaults for hard-label decisions near the boundary."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
    # benign cuDNN v8 fallback message seen with torch 2.3 on some ConvTranspose shapes
    warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")
    warnings.filterwarnings("ignore", message=".*parameter 'pretrained' is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


class CPURNG:
    """Device-independent random stream (CPU generator, results moved to device)."""

    def __init__(self, seed: int, device: torch.device, dtype: torch.dtype = torch.float32):
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) % (2**63 - 1))
        self.device = device
        self.dtype = dtype

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator, dtype=self.dtype).to(self.device)

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, dtype=self.dtype).to(self.device)

    def choice(self, n: int, k: int) -> torch.Tensor:
        """k distinct integers from range(n) (CPU LongTensor)."""
        return torch.randperm(n, generator=self.generator)[:k]


def derive_seed(base: int, *parts: int) -> int:
    """Deterministic seed from a base seed and integer parts (e.g. image index)."""
    s = int(base)
    for p in parts:
        s = (s * 1_000_003 + int(p) * 7919 + 12345) % (2**31 - 1)
    return s


def describe_device(device: torch.device) -> str:
    info = [f"device={device}", f"torch={torch.__version__}"]
    if device.type == "mps":
        info.append(f"mps_built={torch.backends.mps.is_built()}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.append(f"gpu={props.name} ({props.total_memory / 2**30:.0f} GB)")
        info.append(f"tf32={torch.backends.cudnn.allow_tf32} deterministic={torch.backends.cudnn.deterministic}")
    return " | ".join(info)


def as_list(x: Sequence[int] | int) -> list[int]:
    return list(x) if isinstance(x, (list, tuple)) else [int(x)]
