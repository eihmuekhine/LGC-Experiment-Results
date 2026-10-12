#!/usr/bin/env python3
"""Dump the RobustBench (E11) split CSVs: first 200 correctly classified images, seed 42, both modes.

Same selection protocol as ttba_dump_split.py, adapted to the RobustBench victim:
  - pool  = /root/autodl-tmp/ttba_mini_flat/images/ (3923 ILSVRC2012 val JPEGs, flat)
  - labels = /root/autodl-tmp/ttba_mini_flat/ttba_mini_val_labels.txt (ImageFolder index 0..999)
  - victim = Salman2020Do_R50 (RobustBench, imagenet, Linf; ImageNet norm inside the model)
  - seed 42: a seeded permutation of the pool is classified in order; the first 200 correct are kept
  - targeted: a target is drawn (seeded, separate RNG) until its label differs from the source and the victim
    classifies it correctly (same while-loop as ttba_dump_split.py / main.py)

Writes
  splits/robust/Salman2020Do_R50_{non_targeted,targeted}.csv (+ .json meta)
Run from the repo root (CPU only):
  python ttba_github/ttba_dump_split_robust.py
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
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from ttba_common import SPLIT_FIELDS  # noqa: E402
from adapters._common import (  # noqa: E402
    ROBUST_MODEL_NAME,
    ROBUST_WEIGHT_FILE,
    ROBUST_WEIGHT_SHA256,
    build_robust_victim,
    load_image,
    robust_victim_info,
)

FLAT_DIR = "/root/autodl-tmp/ttba_mini_flat"
LABELS_PATH = os.path.join(FLAT_DIR, "ttba_mini_val_labels.txt")


def classify(victim, x, device: str) -> int:
    import torch

    with torch.no_grad():
        return int(victim(x.to(device)).argmax(1).item())


def load_labels() -> dict:
    labels = {}
    with open(LABELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            labels[parts[0]] = int(parts[1])
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["non_targeted", "targeted"],
                    choices=["non_targeted", "targeted"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_splits", default=os.path.join(REPO_ROOT, "splits", "robust"))
    a = ap.parse_args()

    import torch

    torch.set_num_threads(1)

    print("[*] victim inspection (E11):")
    print(json.dumps(robust_victim_info(a.device), indent=1))

    labels = load_labels()
    pool = sorted(labels.keys())
    order_rng = random.Random(a.seed)
    order_rng.shuffle(pool)

    t0 = time.time()
    victim = build_robust_victim(a.device)

    # ---- source selection: classify the shuffled pool in order; keep the first n correct ----
    sources: list = []  # (name, label, loader_pos)
    tested = correct = 0
    for pos, name in enumerate(pool):
        x = load_image(FLAT_DIR, name, a.device)
        pred = classify(victim, x, a.device)
        tested += 1
        lab = labels[name]
        if pred == lab:
            correct += 1
            if len(sources) < a.n:
                sources.append((name, lab, pos))
        if len(sources) >= a.n:
            break

    acc = correct / max(tested, 1)
    print(f"[*] {ROBUST_MODEL_NAME}: kept={len(sources)} tested={tested} correct={correct} "
          f"natural_accuracy={acc:.4f} (RobustBench clean 0.6402)", flush=True)
    if len(sources) < a.n:
        print(f"[!] {ROBUST_MODEL_NAME}: only {len(sources)} correctly classified images (asked {a.n}); STOP")
        return 1

    for mode in a.modes:
        tgt_rng = random.Random(a.seed)
        cache: dict = {}
        rows = []
        for order, (name, lab, pos) in enumerate(sources):
            target, t_lab = "", ""
            if mode == "targeted":
                ri = tgt_rng.randint(0, len(pool) - 1)
                tname = pool[ri]
                t_lab = labels[tname]
                if tname not in cache:
                    cache[tname] = classify(victim, load_image(FLAT_DIR, tname, a.device), a.device)
                tpred = cache[tname]
                while lab == t_lab or tpred != t_lab:
                    ri = tgt_rng.randint(0, len(pool) - 1)
                    tname = pool[ri]
                    t_lab = labels[tname]
                    if tname not in cache:
                        cache[tname] = classify(victim, load_image(FLAT_DIR, tname, a.device), a.device)
                    tpred = cache[tname]
                target = tname
            rows.append({"source": name, "source_label": int(lab), "target": target,
                         "target_label": "" if t_lab is None else t_lab, "order": order,
                         "protocol": f"robustbench:{ROBUST_MODEL_NAME}:targeted{1 if mode == 'targeted' else 0}",
                         "victim_weights": ROBUST_WEIGHT_FILE, "loader_pos": pos,
                         "source_relpath": os.path.join("images", name),
                         "target_relpath": os.path.join("images", target) if target else ""})

        os.makedirs(a.out_splits, exist_ok=True)
        out = os.path.join(a.out_splits, f"{ROBUST_MODEL_NAME}_{mode}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        srcs = [r["source"] for r in rows]
        meta = {"protocol": "robustbench selection (E11, contract v1.5 D9)", "arch": ROBUST_MODEL_NAME,
                "mode": mode, "victim_weights": ROBUST_WEIGHT_FILE,
                "victim_weight_sha256": ROBUST_WEIGHT_SHA256,
                "n": a.n, "kept": len(rows), "tested": tested, "correct": correct,
                "natural_accuracy": acc, "val_images": len(pool), "classes": 1000,
                "unique_sources": len(set(srcs)), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "first_sources": srcs[:5], "seconds": round(time.time() - t0, 1)}
        with open(out.replace(".csv", ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[*] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
