"""
Fair ImageNet comparison runner for:
  LGC, LGC_H, QS, QS_SR_V1, Hybrid, RAG_TTBA_BASE

This is a NEW runner. It does not modify the user's attached
run_imagenet_attack.py or lgc_attack.py.

In addition to the detailed fair-comparison outputs, this version writes
legacy-compatible NPZ files that can be read directly by the user's
Figure_LGC.py, SSIP_LPIPS.py, and Tables.py scripts.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from typing import Dict, List

import lpips
import numpy as np
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from models import builder
from attacks.lgc_attack_ttba_qs_sr_fair import (
    FairLGCCompareAttack,
    build_hard_label_target_seed,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=992):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def yes_no(s: str) -> bool:
    if isinstance(s, bool):
        return s
    x = str(s).strip().lower()
    if x in {"yes", "y", "true", "1"}:
        return True
    if x in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected Yes/No.")


def get_target_model(arch_name, device):
    if arch_name == "resnet50":
        return torch_models.resnet50(weights="DEFAULT").to(device).eval()
    if arch_name == "vgg16":
        return torch_models.vgg16(weights="DEFAULT").to(device).eval()
    if arch_name == "densenet121":
        return torch_models.densenet121(weights="DEFAULT").to(device).eval()
    if arch_name == "vit_b_16":
        return torch_models.vit_b_16(weights="DEFAULT").to(device).eval()
    raise ValueError(f"Unknown architecture: {arch_name}")


class LatentModelWrapper(torch.nn.Module):
    """
    Same RAG map as the attached runner:
      Phi(z) = clip(x_orig + decoder(z) - decoder(z_source), 0, 1)
    """

    def __init__(self, decoder, classifier, mean, std):
        super().__init__()
        self.decoder = decoder
        self.classifier = classifier
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.x_orig = None
        self.x_rec_0 = None

    def set_reference_images(self, x_orig, x_rec_0):
        self.x_orig = x_orig.detach()
        self.x_rec_0 = x_rec_0.detach()

    def get_pixel_image(self, z):
        delta = self.decoder(z) - self.x_rec_0
        return torch.clamp(self.x_orig + delta, 0.0, 1.0)

    def forward(self, z):
        x_final = self.get_pixel_image(z)
        return self.classifier((x_final - self.mean) / self.std)


def load_image(path):
    return transforms.ToTensor()(
        transforms.Resize((224, 224))(Image.open(path).convert("RGB"))
    ).unsqueeze(0).to(device)


def prepare_strict_splits(args, arch, image_dir, img2label, classifier, mean_t, std_t):
    """
    Persistent deterministic split. If an existing split is shorter than
    --num_samples, extend it instead of silently evaluating fewer images.
    """
    splits_dir = os.path.join("splits", "imagenet")
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, f"{arch}_{args.mode}.csv")

    pairs = []
    if os.path.exists(split_file):
        with open(split_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                if args.mode == "targeted" and len(row) >= 4:
                    pairs.append((row[0], int(row[1]), row[2], int(row[3])))
                elif args.mode == "non_targeted" and len(row) >= 2:
                    pairs.append((row[0], int(row[1]), None, None))

        if len(pairs) >= args.num_samples:
            print(f"[*] Loading split from {split_file} ({len(pairs)} available)")
            return pairs[: args.num_samples]
        print(f"[*] Existing split has {len(pairs)} samples; extending to {args.num_samples}.")

    all_files = sorted(glob.glob(os.path.join(image_dir, "*.JPEG")))
    if not all_files:
        raise RuntimeError(f"No .JPEG images found in {image_dir}")

    rng = random.Random(args.seed)
    shuffled_sources = list(all_files)
    rng.shuffle(shuffled_sources)
    existing_sources = {p[0] for p in pairs}

    for s_path in shuffled_sources:
        if len(pairs) >= args.num_samples:
            break
        s_name = os.path.basename(s_path)
        if s_name in existing_sources:
            continue
        s_true = img2label.get(s_name)
        if s_true is None:
            continue

        x_s = load_image(s_path)
        with torch.no_grad():
            pred_s = classifier((x_s - mean_t) / std_t).argmax(1).item()
        if pred_s != s_true:
            continue

        if args.mode == "non_targeted":
            pairs.append((s_name, s_true, None, None))
            existing_sources.add(s_name)
            continue

        targets = [p for p in all_files if p != s_path]
        rng.shuffle(targets)
        for t_path in targets:
            t_name = os.path.basename(t_path)
            t_true = img2label.get(t_name)
            if t_true is None or t_true == s_true:
                continue
            x_t = load_image(t_path)
            with torch.no_grad():
                pred_t = classifier((x_t - mean_t) / std_t).argmax(1).item()
            if pred_t == t_true:
                pairs.append((s_name, s_true, t_name, t_true))
                existing_sources.add(s_name)
                break

    if len(pairs) < args.num_samples:
        raise RuntimeError(
            f"Only {len(pairs)} valid pairs found, but --num_samples={args.num_samples}."
        )

    with open(split_file, "w", newline="") as f:
        writer = csv.writer(f)
        if args.mode == "targeted":
            writer.writerow(["source_image", "source_label", "target_image", "target_label"])
            for p in pairs:
                writer.writerow([p[0], p[1], p[2], p[3]])
        else:
            writer.writerow(["source_image", "source_label"])
            for p in pairs:
                writer.writerow([p[0], p[1]])

    print(f"[*] Saved strict split: {split_file} ({len(pairs)} samples)")
    return pairs[: args.num_samples]


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj



LEGACY_ARCH_NAME = {
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "densenet121": "DenseNet121",
    "vit_b_16": "ViT",
}

# The plotting/table scripts use the historical paper names PGDS / PGDS_H
# for LGC / LGC_H and TtBA for the TtBA baseline.
LEGACY_ATTACK_NAME = {
    "LGC": "PGDS",
    "LGC_H": "PGDS_H",
    "RAG_TTBA_BASE": "TtBA",
    "QS": "QS",
    "QS_SR_V1": "QS_SR_V1",
    "Hybrid": "Hybrid",
}


def _ragged_object_array(rows, key, dtype=np.float64):
    """Return a guaranteed 1-D object array; each element is one trajectory."""
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        hist = r.get("history", [])
        vals = [h[key] for h in hist if key in h]
        out[i] = np.asarray(vals, dtype=dtype)
    return out


def _ragged_string_array(rows, key):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        val = r.get(key, None)
        out[i] = "" if val is None else str(val)
    return out


def save_legacy_plot_npz(
    out_dir: str,
    arch: str,
    mode: str,
    method: str,
    rows: List[Dict],
):
    """
    Write the exact schema expected by the user's existing plotting/table code.

    Required by those scripts:
      * filename: ImageNet_<Model>_<Attack>_<Targeted|NonTargeted>.npz
      * all_queries (or query)
      * l2_pixel (or a compatible norm key)
      * ssim
      * lpips

    We intentionally store FULL attack trajectories, not only fixed checkpoints,
    because Figure_LGC.py evaluates arbitrary query budgets with q <= budget.
    """
    os.makedirs(out_dir, exist_ok=True)

    model_name = LEGACY_ARCH_NAME[arch]
    attack_name = LEGACY_ATTACK_NAME[method]
    target_str = "Targeted" if mode == "targeted" else "NonTargeted"
    filename = f"ImageNet_{model_name}_{attack_name}_{target_str}.npz"
    path = os.path.join(out_dir, filename)

    all_queries = _ragged_object_array(rows, "query", dtype=np.int64)
    l2_pixel = _ragged_object_array(rows, "l2_pix", dtype=np.float64)
    linf_pixel = _ragged_object_array(rows, "linf_pix", dtype=np.float64)
    l2_latent = _ragged_object_array(rows, "l2_lat", dtype=np.float64)
    linf_latent = _ragged_object_array(rows, "linf_lat", dtype=np.float64)
    ssim = _ragged_object_array(rows, "ssim", dtype=np.float64)
    lpips_vals = _ragged_object_array(rows, "lpips", dtype=np.float64)
    asr = _ragged_object_array(rows, "asr", dtype=np.float64)

    # Aliases are deliberate: they make the file compatible with all three
    # attached readers without requiring any change to those scripts.
    np.savez(
        path,
        all_queries=all_queries,
        query=all_queries,
        l2_pixel=l2_pixel,
        l2_pix=l2_pixel,
        all_norms=l2_pixel,
        norms=l2_pixel,
        l2_latent=l2_latent,
        linf_pixel=linf_pixel,
        linf_latent=linf_latent,
        ssim=ssim,
        lpips=lpips_vals,
        asr=asr,
        source_image=_ragged_string_array(rows, "source_image"),
        target_image=_ragged_string_array(rows, "target_image"),
        source_label=np.asarray([r.get("source_label", -1) for r in rows], dtype=np.int64),
        target_label=np.asarray([
            -1 if r.get("target_label", None) is None else r.get("target_label")
            for r in rows
        ], dtype=np.int64),
        success=np.asarray([bool(r.get("success", False)) for r in rows], dtype=np.bool_),
        used_queries=np.asarray([
            int(r.get("query_audit", {}).get("total", 0)) for r in rows
        ], dtype=np.int64),
        dataset=np.asarray("ImageNet"),
        model=np.asarray(model_name),
        attack=np.asarray(attack_name),
        implementation=np.asarray(method),
        mode=np.asarray(mode),
    )
    return path


def verify_legacy_plot_npz(path: str, expected_rows: int):
    """Fail fast if a saved file is not readable by the attached scripts."""
    data = np.load(path, allow_pickle=True)
    required = {"all_queries", "l2_pixel", "ssim", "lpips"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(f"Legacy NPZ {path} is missing keys: {missing}")

    n = len(data["all_queries"])
    if n != expected_rows:
        raise RuntimeError(
            f"Legacy NPZ row mismatch for {path}: {n} != {expected_rows}"
        )

    for key in ["l2_pixel", "ssim", "lpips"]:
        if len(data[key]) != n:
            raise RuntimeError(
                f"Legacy NPZ length mismatch: {key} has {len(data[key])}, queries has {n}"
            )

    # Check per-image trajectory alignment; empty rows are valid failures.
    for i in range(n):
        lens = [
            len(np.asarray(data[k][i]))
            for k in ["all_queries", "l2_pixel", "ssim", "lpips"]
        ]
        if len(set(lens)) != 1:
            raise RuntimeError(
                f"Legacy NPZ trajectory mismatch in row {i}: lengths={lens}"
            )
    return True


def save_method_numeric(method_dir: str, rows: List[Dict], checkpoints: List[int]):
    os.makedirs(method_dir, exist_ok=True)
    ncp = len(checkpoints)

    def stack(key, dtype=np.float64):
        if not rows:
            return np.empty((0, ncp), dtype=dtype)
        return np.stack([np.asarray(r["checkpoints"][key]) for r in rows], axis=0).astype(dtype)

    np.savez(
        os.path.join(method_dir, "metrics_numeric.npz"),
        checkpoints=np.asarray(checkpoints, dtype=np.int64),
        state_query=stack("state_query", np.int64),
        l2_lat=stack("l2_lat"),
        linf_lat=stack("linf_lat"),
        l2_pix=stack("l2_pix"),
        linf_pix=stack("linf_pix"),
        ssim=stack("ssim"),
        lpips=stack("lpips"),
        asr=stack("asr"),
        final_query=np.asarray([
            -1 if r["final"] is None else r["final"]["query"] for r in rows
        ], dtype=np.int64),
        final_l2_pix=np.asarray([
            np.nan if r["final"] is None else r["final"]["l2_pix"] for r in rows
        ], dtype=np.float64),
        final_l2_lat=np.asarray([
            np.nan if r["final"] is None else r["final"]["l2_lat"] for r in rows
        ], dtype=np.float64),
        final_ssim=np.asarray([
            np.nan if r["final"] is None else r["final"]["ssim"] for r in rows
        ], dtype=np.float64),
        final_lpips=np.asarray([
            np.nan if r["final"] is None else r["final"]["lpips"] for r in rows
        ], dtype=np.float64),
        total_queries=np.asarray([r["query_audit"]["total"] for r in rows], dtype=np.int64),
    )

    with open(os.path.join(method_dir, "sample_details.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "source_image", "target_image", "success",
            "best_query", "used_queries", "l2_lat", "l2_pix", "ssim", "lpips",
            "q_init", "q_normal", "q_lgc_search", "q_qs_search",
            "q_qs_sr_screen", "q_ttba_bridge", "q_tangent", "q_refine", "audit_match",
        ])
        for i, r in enumerate(rows):
            final = r["final"]
            qb = r["query_breakdown"]
            writer.writerow([
                i,
                r["source_image"],
                r["target_image"] or "",
                int(r["success"]),
                -1 if final is None else final["query"],
                r["query_audit"]["total"],
                np.nan if final is None else final["l2_lat"],
                np.nan if final is None else final["l2_pix"],
                np.nan if final is None else final["ssim"],
                np.nan if final is None else final["lpips"],
                qb.get("init", 0),
                qb.get("normal", 0),
                qb.get("lgc_search", 0),
                qb.get("qs_search", 0),
                qb.get("qs_sr_screen", 0),
                qb.get("ttba_bridge", 0),
                qb.get("tangent", 0),
                qb.get("refine", 0),
                bool(r["query_audit"]["match"]),
            ])


def summarize_method(method: str, rows: List[Dict]):
    good = [r for r in rows if r["success"] and r["final"] is not None]
    if not good:
        return {
            "method": method,
            "n": len(rows),
            "success_n": 0,
            "asr": 0.0,
            "median_best_query": np.nan,
            "median_used_query": np.nan,
            "median_l2_lat": np.nan,
            "median_l2_pix": np.nan,
            "mean_ssim": np.nan,
            "mean_lpips": np.nan,
        }
    return {
        "method": method,
        "n": len(rows),
        "success_n": len(good),
        "asr": len(good) / max(1, len(rows)),
        "median_best_query": float(np.median([r["final"]["query"] for r in good])),
        "median_used_query": float(np.median([r["query_audit"]["total"] for r in good])),
        "median_l2_lat": float(np.median([r["final"]["l2_lat"] for r in good])),
        "median_l2_pix": float(np.median([r["final"]["l2_pix"] for r in good])),
        "mean_ssim": float(np.mean([r["final"]["ssim"] for r in good])),
        "mean_lpips": float(np.mean([r["final"]["lpips"] for r in good])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["targeted", "non_targeted"], required=True)
    parser.add_argument(
        "--target_archs", nargs="+", default=["resnet50"],
        choices=["resnet50", "vgg16", "densenet121", "vit_b_16"],
    )
    parser.add_argument(
        "--attack_methods", nargs="+",
        default=["LGC", "LGC_H", "RAG_TTBA_BASE", "QS", "QS_SR_V1", "Hybrid"],
        choices=["LGC", "LGC_H", "RAG_TTBA_BASE", "QS", "QS_SR_V1", "Hybrid"],
    )
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--base_dir", default="./data/ImageNet")
    parser.add_argument("--legacy_npz_dir", default=".",
                        help="Directory for Figure_LGC.py / SSIP_LPIPS.py / Tables.py compatible NPZ files")
    parser.add_argument("--num_samples", type=int, default=5)

    parser.add_argument("--max_queries", type=int, default=10000)
    parser.add_argument(
        "--query_checkpoints", nargs="+", type=int,
        default=[500, 1000, 2000, 5000, 10000],
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--initial_query", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.0002)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=992)
    parser.add_argument("--verbose", choices=["Yes", "No"], default="Yes")
    parser.add_argument("--update_query_reserve", type=int, default=18)

    parser.add_argument("--ttba_bridge_tol", type=float, default=0.001)

    parser.add_argument("--switch_ssim", type=float, default=0.995)
    parser.add_argument("--qs_k_mode", choices=["fixed", "gap_based"], default="gap_based")
    parser.add_argument("--qs_fixed_k", type=int, default=10)
    parser.add_argument("--qs_k_min", type=int, default=4)
    parser.add_argument("--qs_k_max", type=int, default=16)
    parser.add_argument("--enable_progress_k", type=yes_no, default=False)

    # QS-SR V1 controls. Defaults correspond to the first clean ablation:
    # two cheap radial screening rounds, one final full refinement, no tangent.
    parser.add_argument("--qs_sr_screen_rounds", type=int, default=2)
    parser.add_argument("--qs_sr_refine_reserve", type=int, default=18)
    parser.add_argument("--qs_sr_monotone_safety", type=yes_no, default=True)

    args = parser.parse_args()
    set_seed(args.seed)

    if max(args.query_checkpoints) > args.max_queries:
        raise ValueError("Every query checkpoint must be <= --max_queries.")

    image_dir = os.path.join(args.base_dir, "images")
    labels_txt = os.path.join(args.base_dir, "val.txt")
    categories_txt = os.path.join(args.base_dir, "synset_words.txt")

    if not os.path.exists(labels_txt):
        raise FileNotFoundError(labels_txt)

    img2label = {}
    with open(labels_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img2label[parts[0]] = int(parts[1]) - 1

    idx2label = {}
    if os.path.exists(categories_txt):
        with open(categories_txt, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(" ", 1)
                idx2label[i] = parts[1] if len(parts) == 2 else line.strip()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)

    class MockArgs:
        arch, parallel, gpu = "vgg16", 0, 0

    ae_model = builder.BuildAutoEncoder(MockArgs())
    sd = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
    state_dict = sd["state_dict"] if "state_dict" in sd else sd
    aligned_sd = {("module." + k.replace("module.", "")): v for k, v in state_dict.items()}
    ae_model.load_state_dict(aligned_sd)
    ae_model = ae_model.to(device).eval()

    if isinstance(ae_model, torch.nn.DataParallel):
        encoder = ae_model.module.encoder
        decoder = ae_model.module.decoder
    else:
        encoder = ae_model.encoder
        decoder = ae_model.decoder

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    for arch in args.target_archs:
        print("\n" + "=" * 118)
        print(f"FAIR LGC / TtBA / QS COMPARISON — {arch.upper()} — {args.mode.upper()}")
        print("=" * 118)

        classifier = get_target_model(arch, device)
        wrapper = LatentModelWrapper(decoder, classifier, mean, std).to(device)

        pairs = prepare_strict_splits(
            args, arch, image_dir, img2label, classifier, mean_t, std_t
        )
        print(f"[*] Evaluating the SAME {len(pairs)} pairs for every method")

        root = os.path.join("Results", "IMAGENET_FAIR_TTBA_QS_SR", args.mode, arch)
        os.makedirs(root, exist_ok=True)
        rows_by_method = {m: [] for m in args.attack_methods}

        for sample_idx, (s_name, s_lbl, t_name, t_lbl) in enumerate(pairs, start=1):
            print(f"\n[{sample_idx}/{len(pairs)}] {s_name} -> {t_lbl if t_lbl is not None else 'Any'}")
            x_orig = load_image(os.path.join(image_dir, s_name))

            with torch.no_grad():
                z_source = encoder(x_orig)
                x_rec_0 = decoder(z_source)
            wrapper.set_reference_images(x_orig, x_rec_0)

            target_seed = None
            if args.mode == "targeted":
                x_target = load_image(os.path.join(image_dir, t_name))
                target_seed = build_hard_label_target_seed(
                    encoder, x_orig, x_rec_0, x_target
                )

            for method in args.attack_methods:
                # Reset all RNGs before EVERY method: same source gets the same
                # initialization/noise stream until method query paths diverge.
                set_seed(args.seed)
                print(f"\n --- {method} ---")

                attack = FairLGCCompareAttack(
                    model=wrapper,
                    z_source=z_source,
                    x_orig=x_orig,
                    source_label=s_lbl,
                    loss_fn_vgg=loss_fn_vgg,
                    ssim_fn=ssim_fn,
                    attack_method=method,
                    target_seed=target_seed,
                    target_label=t_lbl,
                    iteration=args.iterations,
                    initial_query=args.initial_query,
                    tol=args.tol,
                    sigma=args.sigma,
                    seed=args.seed,
                    query_checkpoints=args.query_checkpoints,
                    max_queries=args.max_queries,
                    verbose=args.verbose,
                    ttba_bridge_tol=args.ttba_bridge_tol,
                    switch_ssim=args.switch_ssim,
                    qs_k_mode=args.qs_k_mode,
                    qs_k_min=args.qs_k_min,
                    qs_k_max=args.qs_k_max,
                    qs_fixed_k=args.qs_fixed_k,
                    enable_progress_k=args.enable_progress_k,
                    qs_sr_screen_rounds=args.qs_sr_screen_rounds,
                    qs_sr_refine_reserve=args.qs_sr_refine_reserve,
                    qs_sr_monotone_safety=args.qs_sr_monotone_safety,
                    update_query_reserve=args.update_query_reserve,
                )
                result = attack.attack()
                # Keep the full per-iteration trajectory for legacy NPZ output.
                # This is post-hoc bookkeeping only; it does not alter the attack.
                result["history"] = [dict(h) for h in attack.history]
                result["source_image"] = s_name
                result["source_label"] = int(s_lbl)
                result["target_image"] = t_name
                result["target_label"] = None if t_lbl is None else int(t_lbl)
                rows_by_method[method].append(result)

                if result["final"] is None:
                    print(
                        f"   FAILED | used={result['query_audit']['total']} "
                        f"audit={result['query_audit']['match']}"
                    )
                else:
                    f = result["final"]
                    qb = result["query_breakdown"]
                    print(
                        f"   BEST Q={f['query']} (used={result['query_audit']['total']}) "
                        f"PixL2={f['l2_pix']:.4f} LatL2={f['l2_lat']:.4f} "
                        f"SSIM={f['ssim']:.5f} LPIPS={f['lpips']:.5f}"
                    )
                    print(
                        "   Q: "
                        + " ".join(f"{k}={v}" for k, v in qb.items())
                        + f" | sum={sum(qb.values())} match={result['query_audit']['match']}"
                    )

                method_dir = os.path.join(root, method)
                os.makedirs(method_dir, exist_ok=True)
                with open(
                    os.path.join(method_dir, f"sample_{sample_idx:03d}.json"),
                    "w",
                ) as fobj:
                    json.dump(json_safe(result), fobj, indent=2)

        # Per-method numeric files.
        summaries = []
        legacy_npz_paths = []
        for method, rows in rows_by_method.items():
            method_dir = os.path.join(root, method)
            save_method_numeric(method_dir, rows, args.query_checkpoints)

            legacy_path = save_legacy_plot_npz(
                args.legacy_npz_dir, arch, args.mode, method, rows
            )
            verify_legacy_plot_npz(legacy_path, expected_rows=len(rows))
            legacy_npz_paths.append(legacy_path)
            print(f"[*] Legacy-compatible NPZ: {legacy_path}")

            summaries.append(summarize_method(method, rows))

        # One cross-method final summary.
        final_csv = os.path.join(root, "comparison_final_summary.csv")
        with open(final_csv, "w", newline="") as f:
            fields = [
                "method", "n", "success_n", "asr", "median_best_query",
                "median_used_query", "median_l2_lat", "median_l2_pix",
                "mean_ssim", "mean_lpips",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

        # One cross-method checkpoint summary.
        checkpoint_csv = os.path.join(root, "comparison_checkpoint_summary.csv")
        with open(checkpoint_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method", "checkpoint", "n_success_at_checkpoint",
                "median_state_query", "median_l2_lat", "median_l2_pix",
                "mean_ssim", "mean_lpips", "mean_asr",
            ])
            for method, rows in rows_by_method.items():
                for j, cp in enumerate(args.query_checkpoints):
                    vals = []
                    for r in rows:
                        c = r["checkpoints"]
                        if int(c["state_query"][j]) >= 0 and np.isfinite(c["l2_pix"][j]):
                            vals.append(c)
                    if not vals:
                        writer.writerow([method, cp, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])
                        continue
                    writer.writerow([
                        method,
                        cp,
                        len(vals),
                        float(np.median([v["state_query"][j] for v in vals])),
                        float(np.median([v["l2_lat"][j] for v in vals])),
                        float(np.median([v["l2_pix"][j] for v in vals])),
                        float(np.mean([v["ssim"][j] for v in vals])),
                        float(np.mean([v["lpips"][j] for v in vals])),
                        float(np.mean([v["asr"][j] for v in vals])),
                    ])

        print("\n" + "=" * 118)
        print("FINAL FAIR COMPARISON")
        print("=" * 118)
        for row in summaries:
            print(
                f"{row['method']:<16} "
                f"ASR={100*row['asr']:.1f}% "
                f"PixL2med={row['median_l2_pix']:.4f} "
                f"LatL2med={row['median_l2_lat']:.4f} "
                f"SSIMmean={row['mean_ssim']:.5f} "
                f"LPIPSmean={row['mean_lpips']:.5f} "
                f"usedQmed={row['median_used_query']:.1f}"
            )
        print(f"[*] Results root: {root}")
        print(f"[*] Final summary: {final_csv}")
        print(f"[*] Checkpoint summary: {checkpoint_csv}")
        print(f"[*] Legacy NPZ directory: {os.path.abspath(args.legacy_npz_dir)}")
        print("[*] Existing Figure_LGC.py / SSIP_LPIPS.py / Tables.py read PGDS=LGC, PGDS_H=LGC_H, TtBA=RAG_TTBA_BASE.")
        if any(m in args.attack_methods for m in ["QS", "QS_SR_V1", "Hybrid"]):
            print("[*] QS/QS_SR_V1/Hybrid NPZ files are also saved. Existing plotting scripts need these names added to their attack lists to display them.")

        del classifier
        del wrapper
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
