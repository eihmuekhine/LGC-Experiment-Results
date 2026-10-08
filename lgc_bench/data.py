"""Image loading and deterministic, strictly-correct evaluation splits."""

from __future__ import annotations

import csv
import glob
import os
import random
from typing import Dict, List, Optional, Set, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

Pair = Tuple[str, int, Optional[str], Optional[int]]

# Pre-processing of the victim input (x in [0, 1], 3 x 224 x 224; normalisation is inside the classifier).
#   resize      : Resize((224, 224))                    - Places365 (as the LGC paper / earlier runs)
#   center_crop : Resize(256) + CenterCrop(224)         - standard ImageNet evaluation protocol
PREPROCESS = {
    "resize": T.Compose([T.Resize((224, 224)), T.ToTensor()]),
    "center_crop": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]),
}
DATASET_DEFAULTS = {
    "places365": {"data_dir": "./data/Places365", "labels_file": "adv_test_labels.txt", "preprocess": "resize"},
    "imagenet": {"data_dir": "./data/ImageNet", "labels_file": "val_labels.txt", "preprocess": "center_crop"},
}
_TF = PREPROCESS["resize"]


def set_preprocess(name: str) -> None:
    """Select the image pre-processing used by load_image (and therefore by the splits)."""
    global _TF
    if name not in PREPROCESS:
        raise ValueError(f"unknown preprocess {name!r}; choose from {sorted(PREPROCESS)}")
    _TF = PREPROCESS[name]


def apply_dataset_defaults(args) -> None:
    """Fill data_dir / labels_file / preprocess (None) from the dataset and activate the pre-processing."""
    d = DATASET_DEFAULTS[args.dataset]
    for k in ("data_dir", "labels_file", "preprocess"):
        if getattr(args, k, None) is None:
            setattr(args, k, d[k])
    set_preprocess(args.preprocess)


def load_image(path: str, device: torch.device) -> torch.Tensor:
    return _TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def read_labels(labels_txt: str) -> Dict[str, int]:
    """'<filename> <label>' per line (Places365 adv_test_labels.txt format)."""
    out: Dict[str, int] = {}
    with open(labels_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def read_categories(categories_txt: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not os.path.exists(categories_txt):
        return out
    with open(categories_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                out[int(parts[1])] = parts[0]
    return out


@torch.no_grad()
def _predict(classifier, path: str, device: torch.device) -> int:
    return int(classifier(load_image(path, device)).argmax(1).item())


def prepare_split(
    split_file: str,
    image_dir: str,
    img2label: Dict[str, int],
    classifier,
    device: torch.device,
    mode: str,
    num_samples: int,
    seed: int,
    regenerate: bool = False,
    extensions: Tuple[str, ...] = ("*.JPEG", "*.jpg", "*.jpeg", "*.png"),
    exclude: Optional[Set[str]] = None,
) -> List[Pair]:
    """Source (and target) images that the victim classifies CORRECTLY.

    Uses its own ``random.Random(seed)`` so the split does not depend on any
    other code having consumed global randomness. ``exclude`` (image file names)
    keeps a held-out test split disjoint from a development split.
    """
    if os.path.exists(split_file) and not regenerate:
        pairs: List[Pair] = []
        with open(split_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if mode == "targeted":
                    pairs.append((row[0], int(row[1]), row[2], int(row[3])))
                else:
                    pairs.append((row[0], int(row[1]), None, None))
        if len(pairs) >= num_samples:
            return pairs[:num_samples]
        print(f"[*] {split_file} has {len(pairs)} < {num_samples} pairs; regenerating")

    files: List[str] = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = sorted(set(files))
    if exclude:
        files = [f for f in files if os.path.basename(f) not in exclude]
    rng = random.Random(seed)
    rng.shuffle(files)

    pairs = []
    correct_cache: Dict[str, bool] = {}

    def is_correct(path: str) -> bool:
        name = os.path.basename(path)
        if name not in correct_cache:
            lbl = img2label.get(name)
            correct_cache[name] = lbl is not None and _predict(classifier, path, device) == lbl
        return correct_cache[name]

    for s_path in files:
        if len(pairs) >= num_samples:
            break
        s_name = os.path.basename(s_path)
        if not is_correct(s_path):
            continue
        s_lbl = img2label[s_name]
        if mode == "non_targeted":
            pairs.append((s_name, s_lbl, None, None))
            continue
        candidates = [f for f in files if f != s_path]
        rng.shuffle(candidates)
        for t_path in candidates:
            t_name = os.path.basename(t_path)
            t_lbl = img2label.get(t_name)
            if t_lbl is None or t_lbl == s_lbl:
                continue
            if is_correct(t_path):
                pairs.append((s_name, s_lbl, t_name, t_lbl))
                break

    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
    with open(split_file, "w", newline="") as f:
        w = csv.writer(f)
        if mode == "targeted":
            w.writerow(["source_image", "source_label", "target_image", "target_label"])
            for p in pairs:
                w.writerow([p[0], p[1], p[2], p[3]])
        else:
            w.writerow(["source_image", "source_label"])
            for p in pairs:
                w.writerow([p[0], p[1]])
    print(f"[*] Wrote split {split_file} ({len(pairs)} samples)")
    return pairs
