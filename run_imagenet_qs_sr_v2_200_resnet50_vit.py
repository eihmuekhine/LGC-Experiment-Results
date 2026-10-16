"""
Large-run ImageNet evaluator for QS-SR V2 on ResNet-50 and ViT-B/16.

Designed for 200-image experiments with:
  * strict hard-label query accounting;
  * the lag-corrected QS_SR_V2 method only;
  * architecture-specific deterministic ImageNet splits;
  * crash-safe per-sample JSON checkpoints;
  * resume support;
  * final and checkpoint CSV summaries;
  * legacy-compatible NPZ output for the user's plotting/table scripts.

This runner preserves the established experimental protocol used in the earlier
fair-comparison runner: direct resize to 224x224, ImageNet normalization,
seed=992, N0=30, sigma=2e-4, tol=1e-4, and the same QS-SR controls.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lpips
import numpy as np
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from models import builder
from attacks.lgc_attack_qs_sr_v2_200 import QSSRV2Attack, build_hard_label_target_seed


device = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = "QS_SR_V2"


def set_seed(seed: int = 992):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def yes_no(s) -> bool:
    if isinstance(s, bool):
        return s
    x = str(s).strip().lower()
    if x in {"yes", "y", "true", "1"}:
        return True
    if x in {"no", "n", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected Yes/No.")


def get_target_model(arch_name: str, device_: str):
    """Load the exact torchvision ImageNet classifiers used by the protocol."""
    if arch_name == "resnet50":
        weights = torch_models.ResNet50_Weights.DEFAULT
        model = torch_models.resnet50(weights=weights)
    elif arch_name == "vit_b_16":
        weights = torch_models.ViT_B_16_Weights.DEFAULT
        model = torch_models.vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")
    return model.to(device_).eval()


class LatentModelWrapper(torch.nn.Module):
    """
    RAG image map used by the existing experiments:
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


def load_image(path: str):
    # Deliberately preserves the earlier protocol rather than changing preprocessing.
    img = Image.open(path).convert("RGB")
    x = transforms.Resize((224, 224))(img)
    return transforms.ToTensor()(x).unsqueeze(0).to(device)


def atomic_json_dump(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(json_safe(obj), f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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


def prepare_strict_split(
    args,
    arch: str,
    image_dir: str,
    img2label: Dict[str, int],
    classifier,
    mean_t,
    std_t,
):
    """
    Architecture-specific persistent deterministic split.

    Existing split files are reused so the 200-image run remains compatible with
    prior experiments on the same architecture. If fewer than num_samples are
    present, the split is extended deterministically using seed=992.
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
            return pairs[: args.num_samples], split_file
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
        del x_s
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
            del x_t
            if pred_t == t_true:
                pairs.append((s_name, s_true, t_name, t_true))
                existing_sources.add(s_name)
                break

    if len(pairs) < args.num_samples:
        raise RuntimeError(
            f"Only {len(pairs)} valid pairs found for {arch}, "
            f"but --num_samples={args.num_samples}."
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
    return pairs[: args.num_samples], split_file


LEGACY_ARCH_NAME = {
    "resnet50": "ResNet50",
    "vit_b_16": "ViT",
}


def _ragged_object_array(rows, key, dtype=np.float64):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        hist = r.get("history", [])
        out[i] = np.asarray([h[key] for h in hist if key in h], dtype=dtype)
    return out


def _ragged_string_array(rows, key):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        val = r.get(key, None)
        out[i] = "" if val is None else str(val)
    return out


def save_legacy_npz(out_dir: str, arch: str, mode: str, rows: List[Dict], final: bool):
    os.makedirs(out_dir, exist_ok=True)
    model_name = LEGACY_ARCH_NAME[arch]
    target_str = "Targeted" if mode == "targeted" else "NonTargeted"
    stem = f"ImageNet_{model_name}_QS_SR_V2_{target_str}"
    filename = f"{stem}.npz" if final else f"{stem}_PARTIAL.npz"
    path = os.path.join(out_dir, filename)

    all_queries = _ragged_object_array(rows, "query", dtype=np.int64)
    l2_pixel = _ragged_object_array(rows, "l2_pix", dtype=np.float64)
    linf_pixel = _ragged_object_array(rows, "linf_pix", dtype=np.float64)
    l2_latent = _ragged_object_array(rows, "l2_lat", dtype=np.float64)
    linf_latent = _ragged_object_array(rows, "linf_lat", dtype=np.float64)
    ssim = _ragged_object_array(rows, "ssim", dtype=np.float64)
    lpips_vals = _ragged_object_array(rows, "lpips", dtype=np.float64)
    asr = _ragged_object_array(rows, "asr", dtype=np.float64)

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
        attack=np.asarray("QS_SR_V2"),
        implementation=np.asarray("QS_SR_V2"),
        mode=np.asarray(mode),
        completed_n=np.asarray(len(rows), dtype=np.int64),
    )
    return path


def verify_legacy_npz(path: str, expected_rows: int):
    data = np.load(path, allow_pickle=True)
    required = {"all_queries", "l2_pixel", "ssim", "lpips"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(f"Legacy NPZ {path} missing keys: {missing}")
    if len(data["all_queries"]) != expected_rows:
        raise RuntimeError(f"Legacy NPZ row mismatch: {len(data['all_queries'])} != {expected_rows}")
    for i in range(expected_rows):
        lens = [len(np.asarray(data[k][i])) for k in ["all_queries", "l2_pixel", "ssim", "lpips"]]
        if len(set(lens)) != 1:
            raise RuntimeError(f"Trajectory length mismatch in row {i}: {lens}")


def run_config(args, arch: str) -> Dict:
    return {
        "method": METHOD,
        "arch": arch,
        "mode": args.mode,
        "num_samples": int(args.num_samples),
        "max_queries": int(args.max_queries),
        "query_checkpoints": list(map(int, args.query_checkpoints)),
        "iterations": int(args.iterations),
        "initial_query": int(args.initial_query),
        "sigma": float(args.sigma),
        "tol": float(args.tol),
        "seed": int(args.seed),
        "update_query_reserve": int(args.update_query_reserve),
        "switch_ssim": float(args.switch_ssim),
        "qs_k_mode": str(args.qs_k_mode),
        "qs_fixed_k": int(args.qs_fixed_k),
        "qs_k_min": int(args.qs_k_min),
        "qs_k_max": int(args.qs_k_max),
        "enable_progress_k": bool(args.enable_progress_k),
        "qs_sr_screen_rounds": int(args.qs_sr_screen_rounds),
        "qs_sr_refine_reserve": int(args.qs_sr_refine_reserve),
        "qs_sr_monotone_safety": bool(args.qs_sr_monotone_safety),
        "preprocess": "Resize((224,224))+ToTensor+ImageNetMeanStd",
    }


def result_matches(result: Dict, cfg: Dict, pair: Tuple) -> bool:
    s_name, s_lbl, t_name, t_lbl = pair
    if result.get("method") != METHOD:
        return False
    if result.get("source_image") != s_name:
        return False
    if int(result.get("source_label", -999)) != int(s_lbl):
        return False
    if result.get("target_image") != t_name:
        return False
    got_t = result.get("target_label", None)
    if got_t is not None:
        got_t = int(got_t)
    if got_t != t_lbl:
        return False
    return result.get("run_config", {}) == cfg


def sample_json_path(root: str, sample_idx: int) -> str:
    return os.path.join(root, METHOD, f"sample_{sample_idx:03d}.json")


def load_completed_rows(root: str, pairs: List[Tuple], cfg: Dict) -> List[Dict]:
    rows = []
    for i, pair in enumerate(pairs, start=1):
        p = sample_json_path(root, i)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r") as f:
                r = json.load(f)
            if result_matches(r, cfg, pair):
                rows.append(r)
        except Exception:
            continue
    return rows


def save_summaries(root: str, rows: List[Dict], checkpoints: List[int], arch: str, args, complete: bool):
    method_dir = os.path.join(root, METHOD)
    os.makedirs(method_dir, exist_ok=True)
    ncp = len(checkpoints)

    def stack_checkpoint(key, dtype=np.float64):
        if not rows:
            return np.empty((0, ncp), dtype=dtype)
        return np.stack([np.asarray(r["checkpoints"][key]) for r in rows], axis=0).astype(dtype)

    np.savez(
        os.path.join(method_dir, "metrics_numeric.npz" if complete else "metrics_numeric_PARTIAL.npz"),
        checkpoints=np.asarray(checkpoints, dtype=np.int64),
        state_query=stack_checkpoint("state_query", np.int64),
        l2_lat=stack_checkpoint("l2_lat"),
        linf_lat=stack_checkpoint("linf_lat"),
        l2_pix=stack_checkpoint("l2_pix"),
        linf_pix=stack_checkpoint("linf_pix"),
        ssim=stack_checkpoint("ssim"),
        lpips=stack_checkpoint("lpips"),
        asr=stack_checkpoint("asr"),
        final_query=np.asarray([
            -1 if r.get("final") is None else r["final"]["query"] for r in rows
        ], dtype=np.int64),
        final_l2_pix=np.asarray([
            np.nan if r.get("final") is None else r["final"]["l2_pix"] for r in rows
        ], dtype=np.float64),
        final_l2_lat=np.asarray([
            np.nan if r.get("final") is None else r["final"]["l2_lat"] for r in rows
        ], dtype=np.float64),
        final_ssim=np.asarray([
            np.nan if r.get("final") is None else r["final"]["ssim"] for r in rows
        ], dtype=np.float64),
        final_lpips=np.asarray([
            np.nan if r.get("final") is None else r["final"]["lpips"] for r in rows
        ], dtype=np.float64),
        total_queries=np.asarray([
            r.get("query_audit", {}).get("total", 0) for r in rows
        ], dtype=np.int64),
    )

    details_path = os.path.join(method_dir, "sample_details.csv" if complete else "sample_details_PARTIAL.csv")
    with open(details_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "source_image", "target_image", "success",
            "best_query", "used_queries", "l2_lat", "l2_pix", "ssim", "lpips",
            "q_init", "q_normal", "q_qs_sr_screen", "q_refine", "audit_match",
        ])
        for r in rows:
            final = r.get("final")
            qb = r.get("query_breakdown", {})
            writer.writerow([
                r.get("sample_index", -1),
                r.get("source_image", ""),
                r.get("target_image") or "",
                int(bool(r.get("success", False))),
                -1 if final is None else final["query"],
                r.get("query_audit", {}).get("total", 0),
                np.nan if final is None else final["l2_lat"],
                np.nan if final is None else final["l2_pix"],
                np.nan if final is None else final["ssim"],
                np.nan if final is None else final["lpips"],
                qb.get("init", 0),
                qb.get("normal", 0),
                qb.get("qs_sr_screen", 0),
                qb.get("refine", 0),
                bool(r.get("query_audit", {}).get("match", False)),
            ])

    good = [r for r in rows if r.get("success") and r.get("final") is not None]
    final_csv = os.path.join(root, "comparison_final_summary.csv" if complete else "comparison_final_summary_PARTIAL.csv")
    with open(final_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "arch", "completed_n", "requested_n", "success_n", "asr",
            "median_best_query", "median_used_query", "median_l2_lat", "median_l2_pix",
            "mean_ssim", "mean_lpips",
        ])
        if good:
            writer.writerow([
                METHOD,
                arch,
                len(rows),
                args.num_samples,
                len(good),
                len(good) / max(1, len(rows)),
                float(np.median([r["final"]["query"] for r in good])),
                float(np.median([r["query_audit"]["total"] for r in good])),
                float(np.median([r["final"]["l2_lat"] for r in good])),
                float(np.median([r["final"]["l2_pix"] for r in good])),
                float(np.mean([r["final"]["ssim"] for r in good])),
                float(np.mean([r["final"]["lpips"] for r in good])),
            ])
        else:
            writer.writerow([METHOD, arch, len(rows), args.num_samples, 0, 0.0] + [np.nan] * 6)

    checkpoint_csv = os.path.join(root, "comparison_checkpoint_summary.csv" if complete else "comparison_checkpoint_summary_PARTIAL.csv")
    with open(checkpoint_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "arch", "checkpoint", "n_success_at_checkpoint",
            "median_state_query", "median_l2_lat", "median_l2_pix",
            "mean_ssim", "mean_lpips", "mean_asr",
        ])
        for j, cp in enumerate(checkpoints):
            vals = []
            for r in rows:
                c = r.get("checkpoints", {})
                sq = c.get("state_query", [])
                lp = c.get("l2_pix", [])
                if len(sq) > j and len(lp) > j and int(sq[j]) >= 0 and np.isfinite(lp[j]):
                    vals.append(c)
            if not vals:
                writer.writerow([METHOD, arch, cp, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])
            else:
                writer.writerow([
                    METHOD,
                    arch,
                    cp,
                    len(vals),
                    float(np.median([v["state_query"][j] for v in vals])),
                    float(np.median([v["l2_lat"][j] for v in vals])),
                    float(np.median([v["l2_pix"][j] for v in vals])),
                    float(np.mean([v["ssim"][j] for v in vals])),
                    float(np.mean([v["lpips"][j] for v in vals])),
                    float(np.mean([v["asr"][j] for v in vals])),
                ])

    legacy_path = save_legacy_npz(args.legacy_npz_dir, arch, args.mode, rows, final=complete)
    verify_legacy_npz(legacy_path, len(rows))
    return final_csv, checkpoint_csv, legacy_path


def load_autoencoder(checkpoint: str):
    class MockArgs:
        arch, parallel, gpu = "vgg16", 0, 0

    ae_model = builder.BuildAutoEncoder(MockArgs())
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    aligned_sd = {("module." + k.replace("module.", "")): v for k, v in state_dict.items()}
    ae_model.load_state_dict(aligned_sd)
    ae_model = ae_model.to(device).eval()

    if isinstance(ae_model, torch.nn.DataParallel):
        return ae_model, ae_model.module.encoder, ae_model.module.decoder
    return ae_model, ae_model.encoder, ae_model.decoder


def write_manifest(root: str, cfg: Dict, split_file: str, args):
    path = os.path.join(root, "run_manifest.json")
    manifest = {
        "config": cfg,
        "split_file": split_file,
        "base_dir": os.path.abspath(args.base_dir),
        "ae_checkpoint": os.path.abspath(args.ae_checkpoint),
    }
    if os.path.exists(path):
        with open(path, "r") as f:
            old = json.load(f)
        if old != manifest:
            raise RuntimeError(
                f"Run manifest mismatch at {path}. Use a new --results_tag, or remove "
                "the old result directory only if you intentionally want a fresh run."
            )
    else:
        atomic_json_dump(manifest, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["targeted", "non_targeted"], default="non_targeted")
    parser.add_argument(
        "--target_archs", nargs="+", default=["resnet50", "vit_b_16"],
        choices=["resnet50", "vit_b_16"],
    )
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--base_dir", default="./data/ImageNet")
    parser.add_argument("--legacy_npz_dir", default=".")
    parser.add_argument("--results_tag", default="IMAGENET_QS_SR_V2_200")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--resume", type=yes_no, default=True)
    parser.add_argument("--save_every", type=int, default=10)

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
    parser.add_argument("--verbose", choices=["Yes", "No"], default="No")
    parser.add_argument("--update_query_reserve", type=int, default=18)

    # V2 controller / QS-SR settings. Keep these frozen for the 200-image run.
    parser.add_argument("--switch_ssim", type=float, default=0.995)
    parser.add_argument("--qs_k_mode", choices=["fixed", "gap_based"], default="gap_based")
    parser.add_argument("--qs_fixed_k", type=int, default=10)
    parser.add_argument("--qs_k_min", type=int, default=4)
    parser.add_argument("--qs_k_max", type=int, default=16)
    parser.add_argument("--enable_progress_k", type=yes_no, default=False)
    parser.add_argument("--qs_sr_screen_rounds", type=int, default=2)
    parser.add_argument("--qs_sr_refine_reserve", type=int, default=18)
    parser.add_argument("--qs_sr_monotone_safety", type=yes_no, default=True)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be > 0")
    if max(args.query_checkpoints) > args.max_queries:
        raise ValueError("Every query checkpoint must be <= --max_queries.")

    image_dir = os.path.join(args.base_dir, "images")
    labels_txt = os.path.join(args.base_dir, "val.txt")
    categories_txt = os.path.join(args.base_dir, "synset_words.txt")
    if not os.path.exists(labels_txt):
        raise FileNotFoundError(labels_txt)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)

    img2label = {}
    with open(labels_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img2label[parts[0]] = int(parts[1]) - 1

    # Retain for compatibility/debugging even though names are not needed by the attack.
    idx2label = {}
    if os.path.exists(categories_txt):
        with open(categories_txt, "r") as f:
            for i, line in enumerate(f):
                parts = line.strip().split(" ", 1)
                idx2label[i] = parts[1] if len(parts) == 2 else line.strip()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    print(f"[*] Device: {device}")
    print(f"[*] Loading autoencoder: {args.ae_checkpoint}")
    ae_model, encoder, decoder = load_autoencoder(args.ae_checkpoint)
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    cross_arch_rows = []

    for arch in args.target_archs:
        print("\n" + "=" * 118)
        print(f"QS-SR V2 — 200-IMAGE RUN — {arch.upper()} — {args.mode.upper()}")
        print("=" * 118)

        classifier = get_target_model(arch, device)
        wrapper = LatentModelWrapper(decoder, classifier, mean, std).to(device)

        pairs, split_file = prepare_strict_split(
            args, arch, image_dir, img2label, classifier, mean_t, std_t
        )
        if len(pairs) != args.num_samples:
            raise RuntimeError(f"Split size {len(pairs)} != requested {args.num_samples}")

        root = os.path.join("Results", args.results_tag, args.mode, arch)
        os.makedirs(os.path.join(root, METHOD), exist_ok=True)
        cfg = run_config(args, arch)
        write_manifest(root, cfg, split_file, args)

        completed_before = load_completed_rows(root, pairs, cfg) if args.resume else []
        print(f"[*] Split: {split_file}")
        print(f"[*] Requested samples: {args.num_samples}")
        print(f"[*] Reusable completed samples: {len(completed_before)}")

        for sample_idx, pair in enumerate(pairs, start=1):
            s_name, s_lbl, t_name, t_lbl = pair
            out_json = sample_json_path(root, sample_idx)

            if args.resume and os.path.exists(out_json):
                try:
                    with open(out_json, "r") as f:
                        old = json.load(f)
                    if result_matches(old, cfg, pair):
                        if sample_idx % 10 == 0 or sample_idx == 1 or sample_idx == args.num_samples:
                            print(f"[{sample_idx:03d}/{args.num_samples}] resume: {s_name}")
                        continue
                except Exception:
                    pass

            print(f"[{sample_idx:03d}/{args.num_samples}] run: {s_name} -> {t_lbl if t_lbl is not None else 'Any'}")
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
                del x_target

            # Preserve the established seed protocol exactly.
            set_seed(args.seed)
            attack = QSSRV2Attack(
                model=wrapper,
                z_source=z_source,
                x_orig=x_orig,
                source_label=s_lbl,
                loss_fn_vgg=loss_fn_vgg,
                ssim_fn=ssim_fn,
                attack_method=METHOD,
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
            result["history"] = [dict(h) for h in attack.history]
            result["method"] = METHOD
            result["arch"] = arch
            result["mode"] = args.mode
            result["sample_index"] = sample_idx
            result["source_image"] = s_name
            result["source_label"] = int(s_lbl)
            result["target_image"] = t_name
            result["target_label"] = None if t_lbl is None else int(t_lbl)
            result["run_config"] = cfg
            atomic_json_dump(result, out_json)

            final = result.get("final")
            audit = result.get("query_audit", {})
            if final is None:
                print(f"    FAILED | usedQ={audit.get('total', 0)} audit={audit.get('match', False)}")
            else:
                print(
                    f"    BEST Q={final['query']} usedQ={audit.get('total', 0)} "
                    f"PixL2={final['l2_pix']:.4f} LatL2={final['l2_lat']:.4f} "
                    f"SSIM={final['ssim']:.5f} LPIPS={final['lpips']:.5f} "
                    f"audit={audit.get('match', False)}"
                )

            # Release per-sample tensors. The attack/result JSON no longer needs GPU tensors.
            del attack, result, target_seed, z_source, x_rec_0, x_orig
            if sample_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Crash-safe partial summaries.
            if sample_idx % args.save_every == 0 or sample_idx == args.num_samples:
                rows_now = load_completed_rows(root, pairs, cfg)
                complete_now = len(rows_now) == args.num_samples
                save_summaries(root, rows_now, args.query_checkpoints, arch, args, complete=complete_now)
                print(f"[*] Saved progress: {len(rows_now)}/{args.num_samples} completed for {arch}")

        rows = load_completed_rows(root, pairs, cfg)
        complete = len(rows) == args.num_samples
        final_csv, checkpoint_csv, legacy_path = save_summaries(
            root, rows, args.query_checkpoints, arch, args, complete=complete
        )

        good = [r for r in rows if r.get("success") and r.get("final") is not None]
        if good:
            cross_arch_rows.append({
                "arch": arch,
                "completed_n": len(rows),
                "success_n": len(good),
                "asr": len(good) / max(1, len(rows)),
                "median_l2_pix": float(np.median([r["final"]["l2_pix"] for r in good])),
                "median_l2_lat": float(np.median([r["final"]["l2_lat"] for r in good])),
                "mean_ssim": float(np.mean([r["final"]["ssim"] for r in good])),
                "mean_lpips": float(np.mean([r["final"]["lpips"] for r in good])),
                "median_used_queries": float(np.median([r["query_audit"]["total"] for r in good])),
            })

        print(f"[*] Completed: {len(rows)}/{args.num_samples} for {arch}")
        print(f"[*] Final summary: {final_csv}")
        print(f"[*] Checkpoint summary: {checkpoint_csv}")
        print(f"[*] Legacy NPZ: {legacy_path}")

        del classifier, wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-architecture summary if both models were run in this invocation.
    if cross_arch_rows:
        cross_root = os.path.join("Results", args.results_tag, args.mode)
        os.makedirs(cross_root, exist_ok=True)
        cross_csv = os.path.join(cross_root, "cross_architecture_summary.csv")
        with open(cross_csv, "w", newline="") as f:
            fields = [
                "arch", "completed_n", "success_n", "asr", "median_l2_pix",
                "median_l2_lat", "mean_ssim", "mean_lpips", "median_used_queries",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in cross_arch_rows:
                writer.writerow(row)
        print(f"\n[*] Cross-architecture summary: {cross_csv}")

    del loss_fn_vgg, ae_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
