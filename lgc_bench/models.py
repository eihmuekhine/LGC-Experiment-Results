"""Victim classifiers and the (attacker-owned) autoencoder.

Mac fix: the original ``models/builder.py`` from imagenet-autoencoder wraps the
model in ``nn.DataParallel(model).cuda()``, which crashes on Apple Silicon.
Here the VGG/ResNet autoencoder is constructed directly from the repo's
``models/vgg.py`` / ``models/resnet.py`` and moved to the chosen device.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PLACES365_URLS = {
    "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
    "densenet161": "http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar",
}


def _torch_load(path: str) -> dict:
    """torch.load that works on torch<2.4 (no weights_only) and >=2.6 (default True)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class NormalizedClassifier(nn.Module):
    """Takes images in [0, 1] and applies ImageNet normalisation internally."""

    def __init__(self, net: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def load_places365_classifier(arch: str, device: torch.device, cache_dir: str) -> NormalizedClassifier:
    if arch not in PLACES365_URLS:
        raise ValueError(f"Unknown Places365 arch '{arch}'. Choose from {sorted(PLACES365_URLS)}")
    if arch == "resnet50":
        net = tvm.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 365)
    else:
        net = tvm.densenet161(weights=None)
        net.classifier = nn.Linear(net.classifier.in_features, 365)

    os.makedirs(cache_dir, exist_ok=True)
    url = PLACES365_URLS[arch]
    fname = os.path.basename(url)
    path = os.path.join(cache_dir, fname)
    # Reuse a copy that torch.hub already downloaded (e.g. by run_places365_attack.py on AutoDL).
    hub_copy = os.path.join(torch.hub.get_dir(), "checkpoints", fname)
    if not os.path.exists(path) and os.path.exists(hub_copy):
        path = hub_copy
    if not os.path.exists(path):
        print(f"[*] Downloading {url}\n    (AutoDL: if this is slow or fails, run `source /etc/network_turbo` "
              f"first, or copy the file to {path})")
        tmp = path + ".part"
        torch.hub.download_url_to_file(url, tmp, progress=True)
        os.replace(tmp, path)
    ckpt = _torch_load(path)
    state = ckpt.get("state_dict", ckpt)
    fixed = {}
    for k, v in state.items():
        name = k.replace("module.", "")
        # Legacy DenseNet key names in the Places365 checkpoints.
        name = name.replace("norm.1.", "norm1.").replace("norm.2.", "norm2.")
        name = name.replace("conv.1.", "conv1.").replace("conv.2.", "conv2.")
        fixed[name] = v
    net.load_state_dict(fixed)
    return NormalizedClassifier(net).to(device).eval()


def load_imagenet_classifier(arch: str, device: torch.device, weights: str = "IMAGENET1K_V1") -> NormalizedClassifier:
    """torchvision ImageNet classifier (resnet50, densenet161, vgg16, vit_b_16, ...).

    weights: "IMAGENET1K_V1" (default; the legacy ``pretrained=True`` weights used by the CGBA / TtBA /
    HSJA code bases) or "DEFAULT" (newest torchvision recipe, e.g. resnet50 IMAGENET1K_V2) or any enum name.
    """
    net = tvm.get_model(arch, weights=weights)
    return NormalizedClassifier(net).to(device).eval()


def load_classifier(dataset: str, arch: str, device: torch.device, cache_dir: str,
                    imagenet_weights: str = "IMAGENET1K_V1") -> NormalizedClassifier:
    if dataset == "places365":
        clf = load_places365_classifier(arch, device, cache_dir)
    elif dataset == "imagenet":
        clf = load_imagenet_classifier(arch, device, imagenet_weights)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


def _disable_inplace(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "inplace"):
            m.inplace = False


def load_autoencoder(
    checkpoint: str,
    repo_root: str,
    device: torch.device,
    arch: str = "vgg16",
) -> Tuple[nn.Module, nn.Module]:
    """Return (encoder, decoder) in eval mode on ``device``.

    ``repo_root`` must contain ``models/vgg.py`` (and ``models/resnet.py`` for
    ResNet autoencoders) from the imagenet-autoencoder code base.
    """
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        if arch.startswith("vgg"):
            from models import vgg as ae_vgg  # type: ignore

            model = ae_vgg.VGGAutoEncoder(ae_vgg.get_configs(arch))
        elif arch.startswith("resnet"):
            from models import resnet as ae_resnet  # type: ignore

            configs, bottleneck = ae_resnet.get_configs(arch)
            model = ae_resnet.ResNetAutoEncoder(configs, bottleneck)
        else:
            raise ValueError(f"Unsupported autoencoder arch '{arch}'")
    except ImportError as err:
        raise ImportError(
            f"Could not import models/vgg.py from '{repo_root}'. Pass --ae_repo pointing to the "
            "Latent-Geometric-Chords repository root."
        ) from err

    sd = _torch_load(checkpoint)
    state = sd.get("state_dict", sd)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"AE checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    _disable_inplace(model)  # safer for autograd through the decoder
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.encoder, model.decoder
