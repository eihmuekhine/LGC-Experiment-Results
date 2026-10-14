#!/usr/bin/env python3
"""B4f T3: re-verify the 12 frozen split CSVs rows 1-200 on the GPU under D12 flags.

For every split CSV: the source image must be classified as source_label; for targeted files the target image
must be classified as target_label, target_label != source_label, and there must be no duplicate sources.
Read-only: never writes into result folders.  Prints one table per file and an overall PASS/FAIL.

Usage (from the worktree root):
    OMP_NUM_THREADS=8 python adapters/diag/diag_b4f_t3_pairs.py
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, REPO_ROOT)

import _common as C  # noqa: E402  (configure_precision runs at import -> D12 flags)
import torch  # noqa: E402

SPLITS = [
    ("splits/imagenet/ttba_github/vgg19_non_targeted.csv", "vgg19", False),
    ("splits/imagenet/ttba_github/vgg19_targeted.csv", "vgg19", True),
    ("splits/imagenet/ttba_github/resnet50_non_targeted.csv", "resnet50", False),
    ("splits/imagenet/ttba_github/resnet50_targeted.csv", "resnet50", True),
    ("splits/imagenet/ttba_github/vit_b_32_non_targeted.csv", "vit_b_32", False),
    ("splits/imagenet/ttba_github/vit_b_32_targeted.csv", "vit_b_32", True),
    ("splits/places365/resnet50_places365_non_targeted.csv", "resnet50_places365", False),
    ("splits/places365/resnet50_places365_targeted.csv", "resnet50_places365", True),
    ("splits/places365/densenet161_places365_non_targeted.csv", "densenet161_places365", False),
    ("splits/places365/densenet161_places365_targeted.csv", "densenet161_places365", True),
    ("splits/robust/Salman2020Do_R50_non_targeted.csv", "Salman2020Do_R50", False),
    ("splits/robust/Salman2020Do_R50_targeted.csv", "Salman2020Do_R50", True),
]


def classify_batch(victim, xs):
    xs = torch.stack(xs, dim=0).to(next(iter(victim.parameters())).device)
    with torch.no_grad():
        return victim(xs).argmax(dim=1).cpu().tolist()


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[!] no CUDA device; this check must run on the GPU under D12 flags")
        return 2
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] D12 flags: matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} "
          f"deterministic={torch.backends.cudnn.deterministic} "
          f"benchmark={torch.backends.cudnn.benchmark}")

    victims = {}
    for _, arch, _ in SPLITS:
        if arch not in victims:
            victims[arch] = C.load_victim_any(arch, device)

    overall_fail = 0
    print("\n| file | n | wrong source | wrong target | same-label | dups |")
    print("|---|---|---|---|---|---|")
    for split_rel, arch, targeted in SPLITS:
        split_path = os.path.join(REPO_ROOT, split_rel)
        rows = list(csv.DictReader(open(split_path, newline="", encoding="utf-8")))[:200]
        victim = victims[arch]
        wrong_src = wrong_tgt = same_label = 0
        seen = {}
        dup = 0
        for r in rows:
            name = r["source"]
            seen[name] = seen.get(name, 0) + 1
            x0 = C.load_x0_any(arch, name, device)
            pred = victim(x0).argmax(dim=1).item()
            src_lbl = int(r["source_label"])
            if pred != src_lbl:
                wrong_src += 1
            if targeted and r.get("target"):
                tgt_name = r["target"]
                tgt_lbl = int(r["target_label"])
                if tgt_lbl == src_lbl:
                    same_label += 1
                xt = C.load_x0_any(arch, tgt_name, device)
                tpred = victim(xt).argmax(dim=1).item()
                if tpred != tgt_lbl:
                    wrong_tgt += 1
        dup = sum(1 for v in seen.values() if v > 1)
        fail = wrong_src or wrong_tgt or same_label or dup
        overall_fail += int(fail)
        print(f"| {split_rel} | {len(rows)} | {wrong_src} | {wrong_tgt} | {same_label} | {dup} |")
    print(f"\n[*] files with any violation: {overall_fail}/12")
    return 1 if overall_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
