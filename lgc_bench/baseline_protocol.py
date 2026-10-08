"""Image selection exactly as in the CGBA and TtBA evaluation scripts (build 34).

Why
---
Every method in the paper (LGC, LGC-PVRE and the CGBA / CGBA-H / TtBA baselines) must be scored on the same
(source, target) list, and that list must be the one the baseline scripts attack. This module reproduces the
two selection protocols line by line. The split CSV it writes is read by run_benchmark_33b.py (--split_csv).

The two protocols (the user's scripts, 2026-10-05)
--------------------------------------------------
CGBA (Non_targeted_attack.py / Targeted_attack.py; victim ``pretrained=True`` = IMAGENET1K_V1)
  Pre-processing: PIL open -> Resize((224, 224)) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet mean/std).
  Ground truth: line (k - 1) of val.txt for image ILSVRC2012_val_0000{k:04d}.JPEG (line index, not name lookup).
  Non-targeted: k = 1, 2, ..., 9999 in order; skip missing files; keep the image iff GT == prediction; first
      num_img (200).
  Targeted: np.random.seed(992) once at the top of the script; per attempt i = 1..999:
        idxs = np.random.choice(range(LO, HI), 2)       # with replacement; LO, HI = 1177, 1201 in the script
        if image_iter >= pair_num: break                 # NB: the draw happens BEFORE the stop test
        source = idxs[0], target = idxs[1]
        tar_label = prediction on the target image (not its GT)
        keep iff GT(source) == prediction(source) and prediction(source) != tar_label.
      The global numpy stream continues across the (model, method) loops of the script, so every run after
      the first sees different pairs. ``prior_runs`` emulates that.
TtBA (IEEE_ImageNet_TtBA_Non_Tar.py / _Tar.py; victim ``weights='DEFAULT'``)
  Pre-processing: PIL open .convert('RGB') -> Resize((224, 224)) -> CenterCrop(224) -> ToTensor; normalisation
      inside GeneralTorchModel.
  Ground truth: name lookup in val.txt.
  random.seed(992); all_files = sorted(glob(images/*.JPEG)); random.shuffle(all_files) once, before the model loop.
  Non-targeted: walk all_files; skip names without GT; keep iff prediction == GT; first 200.
  Targeted: as non-targeted; for each kept source, target_candidates = all_files.copy();
      random.shuffle(target_candidates); the first t with GT(t) != GT(source) and prediction(t) == GT(t) is the
      target (target label = GT(t)); sources without a valid target are skipped.
      Assumption: the TtBA Attacker does not draw from Python's ``random`` (else the shuffles after the first
      attack differ). Check this with make_baseline_splits.py --verify_npz (target_labels are in the npz).

Differences from our own seed splits (data.prepare_split), stated so that they are not mistaken for bugs
- Resize((224, 224)) distorts the aspect ratio; our ImageNet default is Resize(256) + CenterCrop(224).
- ResNet-50 weights differ between the two scripts: V1 (CGBA) vs DEFAULT = IMAGENET1K_V2 (TtBA). For
  densenet121 / vgg16 / vit_b_16, DEFAULT is V1, so the two coincide.

This file is torch-free at import time: the selection functions take ``exists(name)`` and ``predict(name)``
callables, so they can be unit-tested with a fake classifier (selftest_protocol_34.py). The torch helpers
(``baseline_tensor``, ``build_victim``, ``TorchPredictor``) import torch lazily.
"""
from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
CSV_FIELDS = ["source", "source_label", "target", "target_label", "order", "protocol", "victim_weights"]

Pair = Tuple[str, int, str, Optional[int]]  # (source, source_label, target or "", target_label or None)


def cgba_name(k: int) -> str:
    """'ILSVRC2012_val_0000' + str(k).zfill(4), as in both CGBA scripts (k < 10000)."""
    return "ILSVRC2012_val_0000" + str(int(k)).zfill(4) + ".JPEG"


# ----------------------------------------------------------------------------------------------- labels
def read_val_lines(path: str) -> List[str]:
    """val.txt split on '\\n' exactly like CGBA (``open(...).read().split('\\n')``)."""
    with open(path, "r") as f:
        return f.read().split("\n")


def read_val_dict(path: str) -> Dict[str, int]:
    """val.txt as {name: label} exactly like TtBA's load_val_txt (2-column lines only)."""
    out: Dict[str, int] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out


def cgba_gt(val_lines: Sequence[str], k: int) -> Tuple[int, str]:
    """GT of image k by LINE INDEX (CGBA). Returns (label, name written on that line) for a consistency check."""
    parts = val_lines[k - 1].split()
    return int(parts[1]), parts[0]


# ------------------------------------------------------------------------------------------- protocols
@dataclass
class SelectionLog:
    attempts: int = 0
    skipped_missing: int = 0
    skipped_misclassified: int = 0
    skipped_same_class: int = 0
    skipped_no_target: int = 0
    line_name_mismatch: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def cgba_nontargeted(exists: Callable[[str], bool], predict: Callable[[str], int], val_lines: Sequence[str],
                     num_img: int = 200, k_max: int = 10000, log: Optional[SelectionLog] = None) -> List[Pair]:
    """Non_targeted_attack.py: ``for image_iter1 in range(1, 10000)``; the selection is deterministic."""
    log = log or SelectionLog()
    pairs: List[Pair] = []
    for k in range(1, k_max):
        if len(pairs) >= num_img:
            break
        name = cgba_name(k)
        if not exists(name):
            log.skipped_missing += 1
            continue
        log.attempts += 1
        gt, line_name = cgba_gt(val_lines, k)
        if line_name != name:
            log.line_name_mismatch += 1
        if gt != int(predict(name)):
            log.skipped_misclassified += 1
            continue
        pairs.append((name, gt, "", None))
    return pairs


def cgba_targeted(exists: Callable[[str], bool], predict: Callable[[str], int], val_lines: Sequence[str],
                  rng: np.random.RandomState, pair_num: int = 200, idx_range: Tuple[int, int] = (1177, 1201),
                  attempts: int = 1000, log: Optional[SelectionLog] = None, strict_missing: bool = False) -> List[Pair]:
    """Targeted_attack.py, one (model, method) run, consuming ``rng`` exactly like the script.

    ``rng`` must be np.random.RandomState(992) for the first run of the script; pass the same object again to
    emulate the next run (the script never re-seeds). ``attempts`` = 1000 reproduces ``range(1, 1000)``
    (999 draws, plus the stop test after the draw)."""
    log = log or SelectionLog()
    pairs: List[Pair] = []
    population = range(int(idx_range[0]), int(idx_range[1]))
    for _ in range(1, attempts):
        idxs = rng.choice(population, 2)  # same call as np.random.choice(range(LO, HI), 2) on the global state
        if len(pairs) >= pair_num:
            break
        s, t = int(idxs[0]), int(idxs[1])
        s_name, t_name = cgba_name(s), cgba_name(t)
        if not (exists(s_name) and exists(t_name)):
            # the script has no existence check here and would raise in Image.open
            if strict_missing:
                raise FileNotFoundError(f"CGBA targeted draw needs {s_name if not exists(s_name) else t_name}")
            log.skipped_missing += 1
            continue
        log.attempts += 1
        gt, line_name = cgba_gt(val_lines, s)
        if line_name != s_name:
            log.line_name_mismatch += 1
        orig_label = int(predict(s_name))
        tar_label = int(predict(t_name))
        if gt != orig_label or orig_label == tar_label:
            if gt != orig_label:
                log.skipped_misclassified += 1
            else:
                log.skipped_same_class += 1
            continue
        pairs.append((s_name, gt, t_name, tar_label))
    return pairs


def ttba_file_order(img_dir: str, py_rng: random.Random) -> List[str]:
    """sorted(glob(images/*.JPEG)) then ONE random.shuffle (before the model loop). Returns base names."""
    all_files = sorted(glob.glob(os.path.join(img_dir, "*.JPEG")))
    py_rng.shuffle(all_files)
    return [os.path.basename(p) for p in all_files]


def ttba_nontargeted(order: Sequence[str], predict: Callable[[str], int], img2label: Dict[str, int],
                     target_count: int = 200, log: Optional[SelectionLog] = None) -> List[Pair]:
    log = log or SelectionLog()
    pairs: List[Pair] = []
    for name in order:
        if len(pairs) >= target_count:
            break
        gt = img2label.get(name)
        if gt is None:
            continue
        log.attempts += 1
        if int(predict(name)) != gt:
            log.skipped_misclassified += 1
            continue
        pairs.append((name, gt, "", None))
    return pairs


def ttba_targeted(order: Sequence[str], predict: Callable[[str], int], img2label: Dict[str, int],
                  py_rng: random.Random, target_count: int = 200, log: Optional[SelectionLog] = None) -> List[Pair]:
    """IEEE_ImageNet_TtBA_Tar.py, one model run; ``order`` is the already-shuffled all_files (base names, in the
    script these are full paths: the shuffle permutes the same sorted list, so base names give the same order)."""
    log = log or SelectionLog()
    pairs: List[Pair] = []
    for name in order:
        if len(pairs) >= target_count:
            break
        gt = img2label.get(name)
        if gt is None:
            continue
        log.attempts += 1
        if int(predict(name)) != gt:
            log.skipped_misclassified += 1
            continue
        cands = list(order)
        py_rng.shuffle(cands)
        target, t_lab = None, None
        for t in cands:
            tl = img2label.get(t)
            if tl is None or tl == gt:
                continue
            if int(predict(t)) == tl:
                target, t_lab = t, tl
                break
        if target is None:
            log.skipped_no_target += 1
            continue
        pairs.append((name, gt, target, t_lab))
    return pairs


# -------------------------------------------------------------------------------------------- split IO
def write_split(path: str, pairs: Iterable[Pair], protocol: str, weights: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, (s, sl, t, tl) in enumerate(pairs):
            w.writerow({"source": s, "source_label": int(sl), "target": t or "",
                        "target_label": "" if tl is None else int(tl), "order": i, "protocol": protocol,
                        "victim_weights": weights})


def read_split(path: str) -> List[Pair]:
    """Rows of a protocol split CSV as (source, source_label, target or '', target_label or None)."""
    out: List[Pair] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tl = r.get("target_label", "")
            out.append((r["source"], int(r["source_label"]), r.get("target", "") or "",
                        None if tl in ("", None) else int(tl)))
    return out


def split_meta(path: str) -> dict:
    """protocol / victim_weights recorded in the first row (empty dict if absent)."""
    with open(path, newline="") as f:
        r = next(csv.DictReader(f), None)
    return {} if r is None else {"protocol": r.get("protocol", ""), "victim_weights": r.get("victim_weights", "")}


def duplicate_sources(pairs: Sequence[Pair]) -> Dict[str, int]:
    seen: Dict[str, int] = {}
    for s, *_ in pairs:
        seen[s] = seen.get(s, 0) + 1
    return {k: v for k, v in seen.items() if v > 1}


# --------------------------------------------------------------------------------- torch helpers (lazy)
def baseline_tensor(path: str):
    """[3, 224, 224] float tensor in [0, 1]: Resize((224, 224)) -> CenterCrop(224) -> ToTensor (both scripts).

    convert('RGB') follows TtBA; CGBA opens without it and would fail on a grayscale / CMYK JPEG, so for every
    image CGBA can process the two are identical."""
    import torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    with Image.open(path) as im:
        return tf(im.convert("RGB"))


def build_victim(arch: str, weights: str, device):
    """Victim with ImageNet normalisation inside; input [N, 3, 224, 224] in [0, 1].

    arch: resnet50 | vgg16 | densenet121 | densenet161 | vit_b_16 (torchvision) | vit_timm (timm
    vit_base_patch16_224, as in CGBA Targeted_attack.py; that script feeds it ImageNet mean/std, which we copy).
    weights: 'V1' (= pretrained=True) or 'DEFAULT' (torchvision >= 0.13)."""
    import torch
    import torchvision.models as tvm
    if arch == "vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_224", pretrained=True)
    else:
        name = {"vit": "vit_b_16", "ViT": "vit_b_16", "vit_basic": "vit_b_16"}.get(arch, arch)
        try:
            enum = tvm.get_model_weights(name)
            w = enum.DEFAULT if weights.upper() == "DEFAULT" else enum.IMAGENET1K_V1
            net = tvm.get_model(name, weights=w)
        except AttributeError:  # torchvision < 0.14: legacy flag is V1
            if weights.upper() != "V1":
                raise
            net = getattr(tvm, name)(pretrained=True)

    class Norm(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mu", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("sd", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.m((x - self.mu) / self.sd)

    return Norm(net).to(device).eval()


# --------------------------------------------- build 38: the official TtBA repository (github.com/BUPTAIOC/TtBA)
# dataset.load_dataset_model('imagenet-*'): torchvision constructor WITHOUT weights, then load_state_dict from
# model/<file> (README links), GeneralTorchModel with ImageNet mean/std; images = ImageFolder(imagenet-mini/val) with
# Resize(256) -> CenterCrop(224) -> ToTensor.  resnet50-11ad3fa6.pth is torchvision's IMAGENET1K_V2.
TTBA_GH_DATASETS = {"imagenet-vgg": "vgg19", "imagenet-resnet50": "resnet50", "imagenet-vit": "vit_b_32",
                    "imagenet-inceptionv3": "inception_v3", "imagenet-efficient": "efficientnet_b0",
                    "imagenet-densenet": "densenet161"}
TTBA_GH_WEIGHTS = {"vgg19": "vgg19-dcbb9e9d.pth", "resnet50": "resnet50-11ad3fa6.pth",
                   "vit_b_32": "vit_b_32-d86f8d99.pth", "inception_v3": "inception_v3_google-0cc3c7bd.pth",
                   "efficientnet_b0": "efficientnet_b0_rwightman-7f5810bc.pth",
                   "densenet161": "densenet161-8d451a50.pth"}


def ttba_github_tensor(path: str):
    """[3, 224, 224] in [0, 1] exactly as the official repo's ImageFolder: pil_loader (convert('RGB')) ->
    Resize(256) -> CenterCrop(224) -> ToTensor."""
    import torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize(256), T.CenterCrop(IMG_SIZE), T.ToTensor()])
    with open(path, "rb") as f:
        im = Image.open(f)
        return tf(im.convert("RGB"))


def build_ttba_github_victim(arch: str, ckpt_dir: str, device):
    """The official repo's victim: ``getattr(torchvision.models, arch)()`` + load_state_dict(<ckpt_dir>/<file>).

    Constructed exactly like the repo (no weights enum), so e.g. inception_v3 keeps transform_input=False as there.
    ImageNet mean/std normalisation inside; input [N, 3, 224, 224] in [0, 1]."""
    import torch
    import torchvision.models as tvm
    if arch not in TTBA_GH_WEIGHTS:
        raise SystemExit(f"[!] --victim_ckpt_dir: no official TtBA weight file for arch {arch!r} "
                         f"(known: {sorted(TTBA_GH_WEIGHTS)})")
    path = os.path.join(ckpt_dir, TTBA_GH_WEIGHTS[arch])
    if not os.path.exists(path):
        raise SystemExit(f"[!] victim weights {path} not found (download the file linked in the TtBA README)")
    net = getattr(tvm, arch)()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return _Norm(net).to(device).eval()


def _norm_module():
    import torch

    class Norm(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("mu", torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer("sd", torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x):
            return self.m((x - self.mu) / self.sd)

    return Norm


def _Norm(net):  # noqa: N802  (factory, keeps torch out of import time)
    return _norm_module()(net)


class TorchPredictor:
    """predict(name) with a cache (TtBA's target scan revisits images)."""

    def __init__(self, net, img_dir: str, device):
        self.net, self.img_dir, self.device = net, img_dir, device
        self.cache: Dict[str, int] = {}

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.img_dir, name))

    def __call__(self, name: str) -> int:
        if name not in self.cache:
            import torch
            x = baseline_tensor(os.path.join(self.img_dir, name)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.cache[name] = int(self.net(x).argmax(1).item())
        return self.cache[name]
