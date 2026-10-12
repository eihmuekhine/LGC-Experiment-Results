#!/usr/bin/env python3
"""Dump the Places365 (E10) split CSVs: first 200 correctly classified images per victim, seed 42, both modes.

Same selection protocol as ttba_dump_split.py (official TtBA main.py logic), adapted to Places365-Standard:
  - pool  = /root/autodl-tmp/places365_data/val_256/ (36500 JPEGs, flat)
  - labels = /root/autodl-tmp/places365_data/places365_val.txt (class index 0..364)
  - victims = resnet50_places365, densenet161_places365 (official MIT weights, ImageNet norm inside)
  - seed 42: a seeded permutation of the pool is classified in order; the first 200 correct are kept
  - targeted: a target is drawn (seeded, separate RNG) until its label differs from the source and the victim
    classifies it correctly (same while-loop as ttba_dump_split.py / main.py)
  - accuracy gate: the natural accuracy over the evaluated prefix must be within +/-5pts of the reference
    (resnet50_places365 0.550, densenet161_places365 0.565); otherwise STOP and write nothing.

Writes
  splits/places365/<arch>_{non_targeted,targeted}.csv (+ .json meta)
Run from the repo root (CPU only):
  python ttba_github/ttba_dump_split_places365.py
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
    PLACES365_WEIGHTS,
    build_places365_victim,
    load_image_places365,
    load_places365_labels,
)

REF_ACCURACY = {"resnet50_places365": 0.550, "densenet161_places365": 0.565}
ACCURACY_TOL = 0.05


def classify(victim, x, device: str) -> int:
    import torch

    with torch.no_grad():
        return int(victim(x.to(device)).argmax(1).item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=["resnet50_places365", "densenet161_places365"],
                    choices=sorted(PLACES365_WEIGHTS))
    ap.add_argument("--modes", nargs="+", default=["non_targeted", "targeted"],
                    choices=["non_targeted", "targeted"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=1000, help="images evaluated for the accuracy gate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_splits", default=os.path.join(REPO_ROOT, "splits", "places365"))
    a = ap.parse_args()

    import torch

    torch.set_num_threads(1)

    labels = load_places365_labels()
    pool = sorted(labels.keys())
    order_rng = random.Random(a.seed)
    order_rng.shuffle(pool)

    t0 = time.time()
    for arch in a.archs:
        victim = build_places365_victim(arch, a.device)
        wname = PLACES365_WEIGHTS[arch][0]

        # ---- source selection: classify the shuffled pool in order; keep the first n correct ----
        sources: list = []  # (name, label, loader_pos)
        tested = correct = 0
        for pos, name in enumerate(pool):
            x = load_image_places365(name, a.device)
            pred = classify(victim, x, a.device)
            tested += 1
            lab = labels[name]
            if pred == lab:
                correct += 1
                if len(sources) < a.n:
                    sources.append((name, lab, pos))
            if tested >= a.n_eval and len(sources) >= a.n:
                break

        acc = correct / max(tested, 1)
        ref = REF_ACCURACY[arch]
        gate_ok = abs(acc - ref) <= ACCURACY_TOL
        print(f"[*] {arch}: kept={len(sources)} tested={tested} correct={correct} accuracy={acc:.4f} "
              f"(ref {ref:.3f} +/- {ACCURACY_TOL:.2f} -> {'OK' if gate_ok else 'STOP'})", flush=True)
        if len(sources) < a.n:
            print(f"[!] {arch}: only {len(sources)} correctly classified images in the pool (asked {a.n}); STOP")
            return 1
        if not gate_ok:
            print(f"[!] {arch}: natural accuracy {acc:.4f} outside {ref:.3f} +/- {ACCURACY_TOL:.2f}; STOP")
            return 1

        # ---- per mode: non_targeted (target = "") or targeted (draw a correct, differently-labelled target) ----
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
                        cache[tname] = classify(victim, load_image_places365(tname, a.device), a.device)
                    tpred = cache[tname]
                    while lab == t_lab or tpred != t_lab:
                        ri = tgt_rng.randint(0, len(pool) - 1)
                        tname = pool[ri]
                        t_lab = labels[tname]
                        if tname not in cache:
                            cache[tname] = classify(victim, load_image_places365(tname, a.device), a.device)
                        tpred = cache[tname]
                    target = tname
                rows.append({"source": name, "source_label": int(lab), "target": target,
                             "target_label": "" if t_lab is None else t_lab, "order": order,
                             "protocol": f"places365:{arch}:targeted{1 if mode == 'targeted' else 0}",
                             "victim_weights": wname, "loader_pos": pos,
                             "source_relpath": os.path.join("val_256", name),
                             "target_relpath": os.path.join("val_256", target) if target else ""})

            os.makedirs(a.out_splits, exist_ok=True)
            out = os.path.join(a.out_splits, f"{arch}_{mode}.csv")
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SPLIT_FIELDS)
                w.writeheader()
                w.writerows(rows)
            srcs = [r["source"] for r in rows]
            meta = {"protocol": "places365 selection (E10, contract v1.5 D9)", "arch": arch, "mode": mode,
                    "victim_weights": wname, "victim_weight_sha256": PLACES365_WEIGHTS[arch][1],
                    "n": a.n, "kept": len(rows), "tested": tested, "correct": correct,
                    "natural_accuracy": acc, "accuracy_gate": gate_ok, "accuracy_ref": ref,
                    "val_images": len(pool), "classes": 365, "unique_sources": len(set(srcs)),
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "first_sources": srcs[:5], "seconds": round(time.time() - t0, 1)}
            with open(out.replace(".csv", ".json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(f"[*] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
