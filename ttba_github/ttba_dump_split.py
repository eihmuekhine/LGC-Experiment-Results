#!/usr/bin/env python3
"""Dump the (source, target) list that the OFFICIAL TtBA repository attacks, without attacking (build 38).

The script imports the repository's own ``dataset.load_dataset_model`` and replays ``main.py``'s selection loop line
by line:
  - seeds 42 (torch, cuda, numpy, random) are set inside load_dataset_model;
  - the victim is constructed and its weights loaded from ``model/<file>``;
  - ImageFolder(imagenet-mini/val) uses Resize(256) -> CenterCrop(224) -> ToTensor;
  - DataLoader(shuffle=True) gives the order;
  - an image is kept iff the prediction equals the label, until ``imgnum`` images are kept;
  - targeted: the target is drawn with ``random.randint`` until its label differs from the source label and the
    model classifies it correctly (same while-loop as main.py).
Nothing in that loop is changed.  The only addition is a subclass of the dataset class that remembers the index of
the last sample read, so that every kept image (and target) can be written by file name.

Why this list equals the list main.py attacks in the same environment
  - The DataLoader permutation is drawn when the loop starts, before any attack.
  - The official Attacker never calls Python's ``random``, so the targeted draws do not depend on the attacks.
  - Predictions are deterministic (cudnn.deterministic = True in load_dataset_model).
  ttba_run_official.py additionally re-checks every prediction.  NOTE: the list depends on the torch/torchvision
  version (model initialisation consumes the CPU RNG before the permutation is drawn), so it can differ from the
  list behind the paper's Table 1 (torch 2.3.0).  Table 1 values are therefore cited, unpaired.

Run from anywhere (the script changes into the repo because the repo uses relative paths):
  python ttba_github/ttba_dump_split.py --ttba_repo /root/autodl-tmp/TtBA --dataset imagenet-resnet50 --targeted 0 \
      --imgnum 1000 --out_splits $PWD/splits/imagenet/ttba_github --flat_dir /root/autodl-tmp/ttba_mini_flat
Writes
  <out_splits>/<arch>_<mode>.csv (+ .json meta)       the split for run_benchmark_33b.py --split_csv
  <flat_dir>/images/<name> -> symlinks to imagenet-mini/val/<wnid>/<name>
  <flat_dir>/ttba_mini_val_labels.txt                 '<name> <ImageFolder index>' for every val image
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ttba_common import DATASETS, SPLIT_FIELDS, WEIGHTS  # noqa: E402


def index_capture(ds) -> None:
    """Make ds remember the index of the last __getitem__ call (return value unchanged)."""
    base = ds.__class__

    def __getitem__(self, index):  # noqa: N807
        self._last_idx = int(index)
        return base.__getitem__(self, index)

    ds.__class__ = type("Indexed" + base.__name__, (base,), {"__getitem__": __getitem__})


def write_flat(ds, flat_dir: str) -> int:
    """Symlink every val image into <flat>/images (names are unique ILSVRC2012_val_*) + the labels file."""
    img_dir = os.path.join(flat_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    names = {}
    for path, lab in ds.samples:
        n = os.path.basename(path)
        if n in names and names[n][0] != path:
            raise SystemExit(f"[!] duplicate file name {n} in two class folders; cannot flatten")
        names[n] = (path, lab)
    for n, (path, _) in names.items():
        dst = os.path.join(img_dir, n)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(path), dst)
    with open(os.path.join(flat_dir, "ttba_mini_val_labels.txt"), "w") as f:
        for n in sorted(names):
            f.write(f"{n} {names[n][1]}\n")
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttba_repo", required=True, help="clone of github.com/BUPTAIOC/TtBA with model/ and data/")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--targeted", type=int, choices=[0, 1], required=True)
    ap.add_argument("--imgnum", type=int, default=1000)
    ap.add_argument("--beginIMG", type=int, default=0)
    ap.add_argument("--out_splits", required=True, help="absolute dir, e.g. <lgc repo>/splits/imagenet/ttba_github")
    ap.add_argument("--flat_dir", required=True, help="absolute dir for the flattened symlink copy")
    a = ap.parse_args()
    out_splits, flat = os.path.abspath(a.out_splits), os.path.abspath(a.flat_dir)
    repo = os.path.abspath(a.ttba_repo)
    arch = DATASETS[a.dataset]
    mode = "targeted" if a.targeted else "non_targeted"
    sys.path.insert(0, repo)
    os.chdir(repo)

    import torch
    import torchvision
    import dataset as ttds  # the official repository's module

    t0 = time.time()
    test_loader, torch_model = ttds.load_dataset_model(a.dataset)  # seeds 42 + model init + weights + loader
    ds = test_loader.dataset
    n_cls = len(getattr(ds, "classes", []))
    if n_cls != 1000:
        print(f"[!] ImageFolder has {n_cls} classes, not 1000: labels would not match the ImageNet victims")
    index_capture(ds)
    n_flat = write_flat(ds, flat)

    rows, tested, kept = [], 0, 0
    for imgi, (original_image, label) in enumerate(test_loader):  # == main.py
        if kept >= a.imgnum:
            break
        if imgi < a.beginIMG:
            continue
        tested += 1
        src_idx = ds._last_idx
        out_label = torch_model.predict_label(original_image.cuda()).cpu().item()
        real_label = label.item()
        if out_label != real_label:
            continue
        t_idx, t_lab = None, None
        if a.targeted == 1:  # main.py, verbatim logic
            random_index = random.randint(0, len(test_loader) - 1)
            tx, ty = test_loader.dataset[random_index]
            tf = torch_model.predict_label(tx.cuda()).cpu()
            while label.item() == ty or tf.item() != ty:
                random_index = random.randint(0, len(test_loader) - 1)
                tx, ty = test_loader.dataset[random_index]
                tf = torch_model.predict_label(tx.cuda()).cpu()
            t_idx, t_lab = random_index, int(ty)
        s_path = ds.samples[src_idx][0]
        t_path = ds.samples[t_idx][0] if t_idx is not None else ""
        rows.append({"source": os.path.basename(s_path), "source_label": int(real_label),
                     "target": os.path.basename(t_path) if t_path else "",
                     "target_label": "" if t_lab is None else t_lab, "order": kept,
                     "protocol": f"ttba_github:{a.dataset}:targeted{a.targeted}", "victim_weights": WEIGHTS[arch],
                     "loader_pos": imgi, "source_relpath": os.path.relpath(s_path, repo),
                     "target_relpath": os.path.relpath(t_path, repo) if t_path else ""})
        kept += 1

    os.makedirs(out_splits, exist_ok=True)
    out = os.path.join(out_splits, f"{arch}_{mode}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    srcs = [r["source"] for r in rows]
    meta = {"protocol": "official TtBA repository main.py selection (build 38)", "dataset": a.dataset,
            "arch": arch, "mode": mode, "victim_weights": WEIGHTS[arch], "imgnum": a.imgnum, "kept": kept,
            "tested": tested, "natural_accuracy": kept / max(tested, 1), "val_images": len(ds),
            "classes": n_cls, "flat_images": n_flat, "unique_sources": len(set(srcs)),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "first_sources": srcs[:5], "seconds": round(time.time() - t0, 1)}
    with open(out.replace(".csv", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    if kept < a.imgnum:
        print(f"[!] only {kept} correctly classified images in the val set (asked {a.imgnum})")
    if len(set(srcs)) != len(srcs):
        print(f"[!] {len(srcs) - len(set(srcs))} repeated sources (impossible with one pass of a permutation)")
    print(f"[*] wrote {out}")


if __name__ == "__main__":
    main()
