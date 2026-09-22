"""
Fair ImageNet comparison runner for:
  RAG_TTBA_BASE, RAG_TTBA_OPTIMIZED, RAG_TTBA_SLERP_OPTIMIZED, RAG_TTBA_POLAR_BEZIER_OPTIMIZED
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
from attacks.lgc_attack_qs_sr_hybrid_v2_fair import (
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

def get_target_model(arch_name, device):
    if arch_name == "resnet50": return torch_models.resnet50(weights="DEFAULT").to(device).eval()
    if arch_name == "vgg16": return torch_models.vgg16(weights="DEFAULT").to(device).eval()
    if arch_name == "densenet121": return torch_models.densenet121(weights="DEFAULT").to(device).eval()
    if arch_name == "vit_b_16": return torch_models.vit_b_16(weights="DEFAULT").to(device).eval()
    raise ValueError(f"Unknown architecture: {arch_name}")

class LatentModelWrapper(torch.nn.Module):
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
    return transforms.ToTensor()(transforms.Resize((224, 224))(Image.open(path).convert("RGB"))).unsqueeze(0).to(device)

def prepare_strict_splits(args, arch, image_dir, img2label, classifier, mean_t, std_t):
    splits_dir = os.path.join("splits", "imagenet")
    os.makedirs(splits_dir, exist_ok=True)
    split_file = os.path.join(splits_dir, f"{arch}_{args.mode}.csv")
    pairs = []
    if os.path.exists(split_file):
        with open(split_file, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row: continue
                if args.mode == "targeted" and len(row) >= 4: pairs.append((row[0], int(row[1]), row[2], int(row[3])))
                elif args.mode == "non_targeted" and len(row) >= 2: pairs.append((row[0], int(row[1]), None, None))
        if len(pairs) >= args.num_samples: return pairs[: args.num_samples]

    all_files = sorted(glob.glob(os.path.join(image_dir, "*.JPEG")))
    rng = random.Random(args.seed)
    shuffled_sources = list(all_files)
    rng.shuffle(shuffled_sources)
    existing_sources = {p[0] for p in pairs}

    for s_path in shuffled_sources:
        if len(pairs) >= args.num_samples: break
        s_name = os.path.basename(s_path)
        if s_name in existing_sources or img2label.get(s_name) is None: continue
        x_s = load_image(s_path)
        with torch.no_grad():
            if classifier((x_s - mean_t) / std_t).argmax(1).item() != img2label[s_name]: continue

        if args.mode == "non_targeted":
            pairs.append((s_name, img2label[s_name], None, None))
            existing_sources.add(s_name)
            continue

        targets = [p for p in all_files if p != s_path]
        rng.shuffle(targets)
        for t_path in targets:
            t_name = os.path.basename(t_path)
            if img2label.get(t_name) is None or img2label[t_name] == img2label[s_name]: continue
            x_t = load_image(t_path)
            with torch.no_grad():
                if classifier((x_t - mean_t) / std_t).argmax(1).item() == img2label[t_name]:
                    pairs.append((s_name, img2label[s_name], t_name, img2label[t_name]))
                    existing_sources.add(s_name)
                    break

    with open(split_file, "w", newline="") as f:
        writer = csv.writer(f)
        if args.mode == "targeted":
            writer.writerow(["source_image", "source_label", "target_image", "target_label"])
            for p in pairs: writer.writerow([p[0], p[1], p[2], p[3]])
        else:
            writer.writerow(["source_image", "source_label"])
            for p in pairs: writer.writerow([p[0], p[1]])
    return pairs[: args.num_samples]

LEGACY_ARCH_NAME = {"resnet50": "ResNet50", "vgg16": "VGG16", "densenet121": "DenseNet121", "vit_b_16": "ViT"}
LEGACY_ATTACK_NAME = {
    "RAG_TTBA_BASE": "TtBA_Base",
    "RAG_TTBA_OPTIMIZED": "TtBA_Optimized",
    "RAG_TTBA_SLERP_OPTIMIZED": "TtBA_SLERP_Opt",
    "RAG_TTBA_POLAR_BEZIER_OPTIMIZED": "TtBA_Polar_Bezier_Opt"
}

def _ragged_object_array(rows, key, dtype=np.float64):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        vals = [h[key] for h in r.get("history", []) if key in h]
        out[i] = np.asarray(vals, dtype=dtype)
    return out

def _ragged_string_array(rows, key):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows): out[i] = "" if r.get(key) is None else str(r.get(key))
    return out

def save_legacy_plot_npz(out_dir: str, arch: str, mode: str, method: str, rows: List[Dict]):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"ImageNet_{LEGACY_ARCH_NAME[arch]}_{LEGACY_ATTACK_NAME[method]}_{'Targeted' if mode == 'targeted' else 'NonTargeted'}.npz"
    path = os.path.join(out_dir, filename)
    all_queries = _ragged_object_array(rows, "query", dtype=np.int64)
    l2_pixel = _ragged_object_array(rows, "l2_pix", dtype=np.float64)
    np.savez(
        path, all_queries=all_queries, query=all_queries, l2_pixel=l2_pixel, l2_pix=l2_pixel, all_norms=l2_pixel, norms=l2_pixel,
        l2_latent=_ragged_object_array(rows, "l2_lat", dtype=np.float64), linf_pixel=_ragged_object_array(rows, "linf_pix", dtype=np.float64),
        linf_latent=_ragged_object_array(rows, "linf_lat", dtype=np.float64), ssim=_ragged_object_array(rows, "ssim", dtype=np.float64),
        lpips=_ragged_object_array(rows, "lpips", dtype=np.float64), asr=_ragged_object_array(rows, "asr", dtype=np.float64),
        source_image=_ragged_string_array(rows, "source_image"), target_image=_ragged_string_array(rows, "target_image"),
        source_label=np.asarray([r.get("source_label", -1) for r in rows], dtype=np.int64),
        target_label=np.asarray([-1 if r.get("target_label", None) is None else r.get("target_label") for r in rows], dtype=np.int64),
        success=np.asarray([bool(r.get("success", False)) for r in rows], dtype=np.bool_),
        used_queries=np.asarray([int(r.get("query_audit", {}).get("total", 0)) for r in rows], dtype=np.int64),
        dataset=np.asarray("ImageNet"), model=np.asarray(LEGACY_ARCH_NAME[arch]), attack=np.asarray(LEGACY_ATTACK_NAME[method]),
        implementation=np.asarray(method), mode=np.asarray(mode),
    )
    return path

def save_method_numeric(method_dir: str, rows: List[Dict], checkpoints: List[int]):
    os.makedirs(method_dir, exist_ok=True)
    def stack(key, dtype=np.float64):
        return np.empty((0, len(checkpoints)), dtype=dtype) if not rows else np.stack([np.asarray(r["checkpoints"][key]) for r in rows], axis=0).astype(dtype)

    np.savez(
        os.path.join(method_dir, "metrics_numeric.npz"), checkpoints=np.asarray(checkpoints, dtype=np.int64),
        state_query=stack("state_query", np.int64), l2_lat=stack("l2_lat"), linf_lat=stack("linf_lat"), l2_pix=stack("l2_pix"),
        linf_pix=stack("linf_pix"), ssim=stack("ssim"), lpips=stack("lpips"), asr=stack("asr"),
        final_query=np.asarray([-1 if r["final"] is None else r["final"]["query"] for r in rows], dtype=np.int64),
        final_l2_pix=np.asarray([np.nan if r["final"] is None else r["final"]["l2_pix"] for r in rows], dtype=np.float64),
        total_queries=np.asarray([r["query_audit"]["total"] for r in rows], dtype=np.int64),
    )

def summarize_method(method: str, rows: List[Dict]):
    good = [r for r in rows if r["success"] and r["final"] is not None]
    if not good: return {"method": method, "n": len(rows), "success_n": 0, "asr": 0.0, "median_best_query": np.nan, "median_used_query": np.nan, "median_l2_lat": np.nan, "median_l2_pix": np.nan, "mean_ssim": np.nan, "mean_lpips": np.nan}
    return {
        "method": method, "n": len(rows), "success_n": len(good), "asr": len(good) / max(1, len(rows)),
        "median_best_query": float(np.median([r["final"]["query"] for r in good])), "median_used_query": float(np.median([r["query_audit"]["total"] for r in good])),
        "median_l2_lat": float(np.median([r["final"]["l2_lat"] for r in good])), "median_l2_pix": float(np.median([r["final"]["l2_pix"] for r in good])),
        "mean_ssim": float(np.mean([r["final"]["ssim"] for r in good])), "mean_lpips": float(np.mean([r["final"]["lpips"] for r in good])),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["targeted", "non_targeted"], required=True)
    parser.add_argument("--target_archs", nargs="+", default=["resnet50"], choices=["resnet50", "vgg16", "densenet121", "vit_b_16"])
    parser.add_argument(
        "--attack_methods", nargs="+", 
        default=[
            "RAG_TTBA_BASE", 
            "RAG_TTBA_SLERP_OPTIMIZED",
            "RAG_TTBA_POLAR_BEZIER_OPTIMIZED"
        ]
    )
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--base_dir", default="./data/ImageNet")
    parser.add_argument("--legacy_npz_dir", default=".")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=10000)
    parser.add_argument("--query_checkpoints", nargs="+", type=int, default=[500, 1000, 2000, 5000, 10000])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--initial_query", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.0002)
    parser.add_argument("--tol", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=992)
    parser.add_argument("--verbose", choices=["Yes", "No"], default="Yes")
    parser.add_argument("--update_query_reserve", type=int, default=18)
    parser.add_argument("--ttba_bridge_tol", type=float, default=0.001)

    args = parser.parse_args()
    set_seed(args.seed)

    image_dir = os.path.join(args.base_dir, "images")
    labels_txt = os.path.join(args.base_dir, "val.txt")

    img2label = {}
    with open(labels_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2: img2label[parts[0]] = int(parts[1]) - 1

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std_t = torch.tensor(std).view(1, 3, 1, 1).to(device)

    class MockArgs: arch, parallel, gpu = "vgg16", 0, 0
    ae_model = builder.BuildAutoEncoder(MockArgs())
    sd = torch.load(args.ae_checkpoint, map_location=device, weights_only=False)
    aligned_sd = {("module." + k.replace("module.", "")): v for k, v in (sd.get("state_dict", sd)).items()}
    ae_model.load_state_dict(aligned_sd)
    ae_model = ae_model.to(device).eval()
    encoder, decoder = (ae_model.module.encoder, ae_model.module.decoder) if isinstance(ae_model, torch.nn.DataParallel) else (ae_model.encoder, ae_model.decoder)
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    for arch in args.target_archs:
        print("\n" + "=" * 118)
        print(f"FAIR LGC / TtBA COMPARISON — {arch.upper()} — {args.mode.upper()}")
        print("=" * 118)

        classifier = get_target_model(arch, device)
        wrapper = LatentModelWrapper(decoder, classifier, mean, std).to(device)
        pairs = prepare_strict_splits(args, arch, image_dir, img2label, classifier, mean_t, std_t)

        root = os.path.join("Results", "IMAGENET_FAIR_QS_SR_HYBRID_V2", args.mode, arch)
        os.makedirs(root, exist_ok=True)
        rows_by_method = {m: [] for m in args.attack_methods}

        for sample_idx, (s_name, s_lbl, t_name, t_lbl) in enumerate(pairs, start=1):
            print(f"\n[{sample_idx}/{len(pairs)}] {s_name} -> {t_lbl if t_lbl is not None else 'Any'}")
            x_orig = load_image(os.path.join(image_dir, s_name))
            with torch.no_grad():
                z_source = encoder(x_orig)
                x_rec_0 = decoder(z_source)
            wrapper.set_reference_images(x_orig, x_rec_0)

            target_seed = build_hard_label_target_seed(encoder, x_orig, x_rec_0, load_image(os.path.join(image_dir, t_name))) if args.mode == "targeted" else None

            for method in args.attack_methods:
                set_seed(args.seed)
                print(f"\n --- {method} ---")
                attack = FairLGCCompareAttack(
                    model=wrapper, z_source=z_source, x_orig=x_orig, source_label=s_lbl,
                    loss_fn_vgg=loss_fn_vgg, ssim_fn=ssim_fn, attack_method=method,
                    target_seed=target_seed, target_label=t_lbl, iteration=args.iterations,
                    initial_query=args.initial_query, tol=args.tol, sigma=args.sigma,
                    seed=args.seed, query_checkpoints=args.query_checkpoints,
                    max_queries=args.max_queries, verbose=args.verbose, ttba_bridge_tol=args.ttba_bridge_tol,
                    update_query_reserve=args.update_query_reserve,
                )
                result = attack.attack()
                result["history"] = [dict(h) for h in attack.history]
                result["source_image"], result["source_label"] = s_name, int(s_lbl)
                result["target_image"], result["target_label"] = t_name, None if t_lbl is None else int(t_lbl)
                rows_by_method[method].append(result)

                if result["final"] is not None:
                    f = result["final"]
                    print(f"   BEST Q={f['query']} (used={result['query_audit']['total']}) PixL2={f['l2_pix']:.4f} LatL2={f['l2_lat']:.4f} SSIM={f['ssim']:.5f} LPIPS={f['lpips']:.5f}")

        summaries = []
        for method, rows in rows_by_method.items():
            method_dir = os.path.join(root, method)
            save_method_numeric(method_dir, rows, args.query_checkpoints)
            save_legacy_plot_npz(args.legacy_npz_dir, arch, args.mode, method, rows)
            summaries.append(summarize_method(method, rows))

        print("\n" + "=" * 118 + "\nFINAL FAIR COMPARISON\n" + "=" * 118)
        for row in summaries:
            print(f"{row['method']:<32} ASR={100*row['asr']:.1f}% PixL2med={row['median_l2_pix']:.4f} LatL2med={row['median_l2_lat']:.4f} SSIMmean={row['mean_ssim']:.5f} LPIPSmean={row['mean_lpips']:.5f}")

        del classifier, wrapper
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()