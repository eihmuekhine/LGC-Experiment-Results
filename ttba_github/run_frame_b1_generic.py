#!/usr/bin/env python3
"""Run the FROZEN LGC-EGC engine (lgc_bench/engine.py) on the E10 (Places365) / E11 (RobustBench) victims.

This is the generic counterpart of run_benchmark_33b.py (which is ImageNet-specific).  It reuses the SAME engine
entry points, unchanged:

    GeometricAttack(spec, cfg, space, oracle, seed, target_seed=..., target_image=None)  (lgc_bench/engine.py:1097)
    HardLabelOracle(clf, space, s_lbl, t_lbl, max_queries, query_batch)                   (lgc_bench/spaces.py:164)

and the SAME spaces (LatentRAGSpace -> LatentResidualSpace -> AmbientResidualSpace for LGC_RC_EGC, space
``latent_ra``) and the SAME autoencoder (checkpoints/vgg16_imagenet.pth).  The only differences are the victim
(``adapters/_common.load_victim_any``: [0,1] -> logits, normalisation inside) and the image loader
(``load_x0_any``: TtBA protocol Resize 256 -> CenterCrop 224).  Never modifies lgc_bench/engine.py.

Writes the SAME artefacts as run_benchmark_33b.py (per_checkpoint.csv, runs.csv, adv_store, env_frame.txt) so that
tools/eval/check_contract.py --stage smoke validates the E10/E11 cells.

Usage (from the repo root, GPU):
    python ttba_github/run_frame_b1_generic.py --arch resnet50_places365 --mode non_targeted \
        --split_csv splits/places365/resnet50_places365_non_targeted.csv \
        --out_dir Results/b4b_smoke_LGC_RC_EGC_resnet50_places365 --num_samples 3 --save_adv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import _common as C  # noqa: E402
from lgc_bench.engine import REGISTRY, EngineConfig, GeometricAttack  # noqa: E402
from lgc_bench.spaces import HardLabelOracle, LatentRAGSpace, LatentResidualSpace  # noqa: E402
from lgc_bench.ambient import AmbientResidualSpace  # noqa: E402
from lgc_bench.models import load_autoencoder  # noqa: E402
from lgc_bench.metrics import build_lpips, image_metrics, uint8_repair  # noqa: E402
from lgc_bench.env import configure_backend, derive_seed, empty_cache, seed_everything  # noqa: E402
from advstore import store_adv  # noqa: E402

import torch  # noqa: E402

METHOD = "LGC_RC_EGC"
CKPT_FIELDS = ["dataset", "arch", "mode", "image", "source_label", "target_label", "method", "space", "budget",
               "found", "state_query", "pix_l2", "pix_linf", "rmse", "lat_l2", "ssim", "lpips",
               "adv_float", "adv_uint8", "pix_l2_uint8", "adv_uint8_repaired", "pix_l2_uint8_repaired",
               "repair_scale", "repair_queries", "seed"]
RUN_FIELDS = ["dataset", "arch", "mode", "image", "method", "space", "status", "total_queries", "iterations",
              "wall_time_s", "final_state_query", "error"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True,
                    choices=["vgg19", "resnet50", "vit_b_32", "kat_linear",
                             "resnet50_places365", "densenet161_places365", "Salman2020Do_R50"])
    ap.add_argument("--mode", choices=["non_targeted", "targeted"], required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/ttba_mini_flat",
                    help="ImageNet (TtBA-mini flat) data dir; Places365/RobustBench have their own loaders")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--max_queries", type=int, default=10000)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=None, help="default: contract checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--query_batch", type=int, default=64)
    ap.add_argument("--ae_checkpoint", default="checkpoints/vgg16_imagenet.pth")
    ap.add_argument("--ae_repo", default=".")
    ap.add_argument("--ae_arch", default="vgg16")
    ap.add_argument("--save_adv", action="store_true", help="store the adversarial image at every checkpoint")
    ap.add_argument("--save_traces", action="store_true")
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()

    # kit v5 provenance guards (T4): frozen engine, frozen autoencoder, ADV_STORE_BUDGETS unset.
    engine_sha = C.sha256_file(os.path.join(REPO_ROOT, "lgc_bench", "engine.py"))
    if not engine_sha.startswith("835f1933d9e5"):
        raise SystemExit(f"[!] lgc_bench/engine.py sha256 {engine_sha} does not start with 835f1933d9e5 (frozen engine)")
    ae_sha = C.sha256_file(a.ae_checkpoint)
    if not ae_sha.startswith("1d004c634f1b"):
        raise SystemExit(f"[!] autoencoder {a.ae_checkpoint} sha256 {ae_sha} does not start with 1d004c634f1b")
    if os.environ.get("ADV_STORE_BUDGETS"):
        raise SystemExit("[!] ADV_STORE_BUDGETS is set: unset it (contract I11)")

    contract = C.load_contract()
    checkpoints = a.checkpoints if a.checkpoints is not None else C.contract_checkpoints(contract)
    budget = a.max_queries

    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    configure_backend(device)
    seed_everything(a.seed)
    torch.set_grad_enabled(False)

    clf = C.load_victim_any(a.arch, device)  # [0,1] -> logits, normalisation inside
    encoder, decoder = load_autoencoder(a.ae_checkpoint, a.ae_repo, device, a.ae_arch)
    lpips_net, lpips_dev = (None, None) if a.no_lpips else build_lpips(device)

    spec = REGISTRY[METHOD]
    cfg = EngineConfig(max_queries=budget, checkpoints=tuple(checkpoints), query_batch=a.query_batch,
                       log_geometry=True, save_trace=a.save_traces, verbose=False)

    dataset = "kat" if a.arch == "kat_linear" else ("places365" if a.arch in C.PLACES365_WEIGHTS else "imagenet")
    run_dir = os.path.join(a.out_dir, a.mode)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_csv = os.path.join(run_dir, "per_checkpoint.csv")
    runs_csv = os.path.join(run_dir, "runs.csv")

    # env_frame.txt (ours) with the tokens check_contract.py I10/I11 read.
    split_sha = C.sha256_file(a.split_csv)
    adv_store_budgets = os.environ.get("ADV_STORE_BUDGETS", "unset")
    env_lines = [
        f"torch {torch.__version__}",
        f"cuda {torch.version.cuda}",
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        f"{engine_sha}  lgc_bench/engine.py",
        f"{split_sha}  {a.split_csv}",
        "checkpoints " + " ".join(str(c) for c in checkpoints),
        f"budget {budget}",
        f"backend.cuda_matmul_allow_tf32 {torch.backends.cuda.matmul.allow_tf32}",
        f"backend.cudnn_allow_tf32 {torch.backends.cudnn.allow_tf32}",
        f"backend.cudnn_deterministic {torch.backends.cudnn.deterministic}",
        f"backend.cudnn_benchmark {torch.backends.cudnn.benchmark}",
        f"backend.float32_matmul_precision {torch.get_float32_matmul_precision()}",
        f"victim_weight_file {getattr(clf, 'victim_weight_file', '')}",
        f"victim_weight_sha256 {getattr(clf, 'victim_weight_sha256', '')}",
        f"ADV_STORE_DTYPE {os.environ.get('ADV_STORE_DTYPE', 'float16')} ADV_STORE_BUDGETS {adv_store_budgets}",
        "",
    ]
    with open(os.path.join(a.out_dir, "env_frame.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(env_lines))

    rows = C.load_split_rows(a.split_csv)[: a.num_samples]
    done = C.done_images(runs_csv)
    t_all = time.time()

    for img_idx, r in enumerate(rows):
        s_name = r["source"]
        if s_name in done:
            continue
        s_lbl = int(r["source_label"])
        t_name = r["target"] if (a.mode == "targeted" and r.get("target")) else None
        t_lbl = int(r["target_label"]) if (a.mode == "targeted" and r.get("target_label") not in ("", None)) else None
        x0 = C.load_x0_any(a.arch, s_name, device, data_dir=a.data_dir)
        x_t = C.load_x0_any(a.arch, t_name, device, data_dir=a.data_dir) if t_name else None
        seed_img = derive_seed(a.seed, img_idx)
        stem = C.stem_of(s_name)

        z_s = encoder(x0)
        latent_space = LatentRAGSpace(z_s, x0, decoder)
        target_seed = None
        if x_t is not None:
            pseudo = torch.clamp(x_t - (x0 - latent_space.x_rec0), 0.0, 1.0)
            target_seed = encoder(pseudo)
        rspace = LatentResidualSpace(z_s, x0, decoder, encoder, x_t)
        space = AmbientResidualSpace(rspace)
        target = space.target_point

        oracle = HardLabelOracle(clf, space, s_lbl, t_lbl, budget, a.query_batch)
        t0 = time.time()
        attack = GeometricAttack(spec, cfg, space, oracle, seed_img, target_seed=target, target_image=None)
        res = attack.run()
        wall = time.time() - t0

        ckpt_rows = []
        for cp in checkpoints:
            row = {"dataset": dataset, "arch": a.arch, "mode": a.mode, "image": s_name,
                   "source_label": s_lbl, "target_label": "" if t_lbl is None else t_lbl,
                   "method": spec.name, "space": spec.space, "budget": cp, "seed": seed_img}
            if cp in res.checkpoint_states:
                z, sq = res.checkpoint_states[cp]
                sp = res.checkpoint_spaces.get(cp, space)
                row["space"] = sp.name
                img = sp.to_image(z)
                m = image_metrics(img, x0, oracle.label_uncounted, oracle.is_adv_label, lpips_net, lpips_dev)
                row.update(m)
                if m["adv_uint8"] == 1:
                    row.update({"adv_uint8_repaired": 1, "pix_l2_uint8_repaired": m["pix_l2_uint8"],
                                "repair_scale": 1.0, "repair_queries": 0})
                else:
                    row.update(uint8_repair(z, sp.origin, sp.to_image, sp.project, x0,
                                            oracle.label_uncounted, oracle.is_adv_label))
                lat = sp.latent_distance(z) if sp.name == "latent_ra" else float("nan")
                row.update({"found": 1, "state_query": sq, "lat_l2": lat})
                if a.save_adv:
                    store_adv(os.path.join(run_dir, "adv_store"), a.arch, stem, spec.name, cp, img, x0,
                              state_query=sq, adv_flag=m.get("adv_float", -1), mode=a.mode)
            else:
                row.update({"found": 0, "state_query": -1, "adv_float": 0, "adv_uint8": 0,
                            "adv_uint8_repaired": 0})
            ckpt_rows.append(row)
        C.append_csv(ckpt_csv, CKPT_FIELDS, ckpt_rows)

        C.append_csv(runs_csv, RUN_FIELDS, [{
            "dataset": dataset, "arch": a.arch, "mode": a.mode, "image": s_name, "method": spec.name,
            "space": spec.space, "status": res.status, "total_queries": res.queries,
            "iterations": res.iterations, "wall_time_s": round(res.wall_time, 2),
            "final_state_query": res.final_state_query, "error": ""}])

        last = ckpt_rows[-1]
        el = time.time() - t_all
        print(f"[{img_idx + 1}/{len(rows)}] {s_name} {spec.name} q={res.queries} status={res.status} "
              f"L2@{last['budget']}={last.get('pix_l2', float('nan')):.4f} {wall:.1f}s | ~{el / (img_idx + 1) * (len(rows) - img_idx - 1) / 3600:.2f}h",
              flush=True)
        empty_cache(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
