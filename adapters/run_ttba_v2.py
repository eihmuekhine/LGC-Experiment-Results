#!/usr/bin/env python3
"""Run the OFFICIAL TtBA attack (github.com/BUPTAIOC/TtBA, TtBA.py class Attacker) on the shared split through the
decision-only HardLabelTracker, store the adversarial images (advstore) and compute SSIM (extension of
ttba_github/ttba_run_official.py; paper-runs branch, phase B2).

What is official and unchanged
  - TtBA.Attacker: same defaults as main.py (dim_reduc_factor 4, tol 1e-4, sigma 3e-4, initial_query 30,
    iteration 100) and same args (budget, epsilon, early 0).
  - dataset.load_dataset_model: seeds 42, victim construction, weight file, GeneralTorchModel, transforms.
  - The exact RNG stream of main.py is preserved (--seed_mode exact): the tracker only counts queries and clamps
    the input exactly as GeneralTorchModel.predict_label does; it never draws random numbers.
What is added (observation / bookkeeping only; no extra victim query is charged beyond what the attack requests)
  - The model is wrapped in tools/eval HardLabelTracker: every predict_label counts one query, the budget 10000 is
    enforced (BudgetExhausted), the best-so-far adversarial (min pixel L2 among all queried adversarial inputs) is
    tracked and recorded at the CONTRACT checkpoints (read from contract/paper_contract.yaml).
  - SnapAttacker (unchanged from ttba_run_official.py) records the OFFICIAL iterate at each checkpoint
    (pix_l2_iter) for the G3/G5 Table 1 comparison (contract D2).
  - advstore writes the tracker-best image at every checkpoint; SSIM (torchmetrics, Gaussian 11, sigma 1.5,
    data_range 1) is computed on the stored image (evaluation only, uncounted).
  - env_TtBA.json + run_manifest.json (resume refuses on manifest mismatch).

Usage
  python adapters/run_ttba_v2.py --ttba_repo /root/autodl-tmp/TtBA --dataset imagenet-resnet50 --targeted 0 \
      --split_csv $PWD/splits/imagenet/ttba_github/resnet50_non_targeted.csv --out_dir $PWD/Results/ttba_v2_resnet50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # adapters/
sys.path.insert(0, REPO_ROOT)  # advstore.py + lgc_bench
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))  # hardlabel_tracker
sys.path.insert(0, os.path.join(REPO_ROOT, "ttba_github"))  # ttba_common

import _common as C  # noqa: E402
from hardlabel_tracker import BudgetExhausted  # noqa: E402
from ttba_common import DATASETS, WEIGHTS, CheckpointRecorder, HookList, done_images, read_split_rows, write_rows  # noqa: E402

METHOD = "TtBA"
CKPT_FIELDS = ["arch", "mode", "image", "method", "budget", "found", "state_query", "pix_l2", "pix_linf",
               "adv_float", "adv_uint8", "source_label", "target_label", "ssim", "pix_l2_iter", "seed_mode"]
RUN_FIELDS = ["arch", "mode", "image", "method", "queries", "total_queries", "nonfinite_queries", "status",
              "wall_time_s", "error", "success_eps", "x0_pred_ok", "seed_mode"]

PARAMS = {
    "dim_reduc_factor": 4,
    "tol": 1e-4,
    "sigma": 3e-4,
    "initial_query": 30,
    "iteration": 100,
}
HYPER_SOURCES = {
    "dim_reduc_factor": "TtBA.py:19 (Attacker default 4)",
    "tol": "TtBA.py:20 (default 1e-4)",
    "sigma": "TtBA.py:20 (default 3e-4)",
    "initial_query": "TtBA.py:20 (default 30)",
    "iteration": "TtBA.py:20 (default 100)",
    "early": "TtBA.py / main.py: early=0 (no success-threshold early stop)",
    "budget": "contract budget 10000",
}


def _ssim(xa, x0) -> float:
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

    return float(tm_ssim(xa.clamp(0.0, 1.0), x0.clamp(0.0, 1.0), data_range=1.0).item())


def _main_generic(a) -> int:
    """E10/E11 path: the SAME official TtBA.Attacker, with the shared Places365/RobustBench victim + loader.

    The only differences from the ImageNet path are the victim object (``C.load_victim_any``, [0,1] -> logits,
    normalisation inside) and the image loader (``C.load_x0_any``, TtBA protocol Resize 256 -> CenterCrop 224).
    The tracker, SnapAttacker and the artefact writers are identical; there is no official main.py RNG stream to
    replay for these datasets, so the per-image seed is 42 + row order."""
    contract = C.load_contract()
    checkpoints = C.contract_checkpoints(contract)
    budget = a.budget if a.budget is not None else C.contract_budget(contract)
    if a.seed_mode == "exact":
        a.seed_mode = "per_image"  # no official main.py stream for E10/E11
    arch, mode = a.arch, ("targeted" if a.targeted else "non_targeted")
    out_dir, split_csv = os.path.abspath(a.out_dir), os.path.abspath(a.split_csv)
    rows = read_split_rows(split_csv)
    if a.limit:
        rows = rows[: a.limit]
    if bool(a.targeted) != bool(rows and rows[0].get("target")):
        raise SystemExit(f"[!] --targeted {a.targeted} does not match the split {split_csv}")

    run_dir = os.path.join(out_dir, mode)
    os.makedirs(run_dir, exist_ok=True)
    ck_path = os.path.join(run_dir, f"per_checkpoint_{METHOD}.csv")
    run_path = os.path.join(run_dir, f"runs_{METHOD}.csv")
    done = done_images(run_path)

    repo = os.path.abspath(a.ttba_repo)
    sys.path.insert(0, repo)
    import torch

    import TtBA as ttba  # official
    from general_torch_model import GeneralTorchModel  # official

    device = "cuda" if torch.cuda.is_available() else "cpu"
    victim = C.load_victim_any(arch, device)
    n_cls = 365 if arch in C.PLACES365_WEIGHTS else 1000
    torch_model = GeneralTorchModel(victim, n_class=n_cls, im_mean=None, im_std=None)

    repo_commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
    env = {"method": METHOD, "arch": arch, "mode": mode, "repo": repo, "repo_commit": repo_commit,
           "params": PARAMS, "hyper_parameter_sources": HYPER_SOURCES,
           "split_csv": split_csv, "split_csv_sha256": C.sha256_file(split_csv),
           "victim_weight_file": getattr(victim, "victim_weight_file", "MISSING"),
           "victim_weight_sha256": getattr(victim, "victim_weight_sha256", "MISSING"),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "budget": budget, "max_queries": budget, "checkpoints": checkpoints, "budgets": checkpoints,
           "precision": {
               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
           },
           "adapter_sha256": C.sha256_file(os.path.abspath(__file__)),
           "tracker_sha256": C.sha256_file(os.path.join(REPO_ROOT, "tools", "eval", "hardlabel_tracker.py")),
           "contract_sha256": C.sha256_file(C.CONTRACT_PATH),
           "adv_store_dtype": os.environ.get("ADV_STORE_DTYPE", "float16"),
           "ADV_STORE_BUDGETS": "unset" if not os.environ.get("ADV_STORE_BUDGETS") else os.environ["ADV_STORE_BUDGETS"]}
    manifest = C.build_manifest(env, contract)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in manifest:
            if k in ("gpu_name", "torch", "cuda"):
                continue
            if old.get(k) != manifest[k]:
                raise SystemExit(f"[!] run_manifest mismatch on {k!r}; refusing to resume")
    else:
        json.dump(manifest, open(manifest_path, "w"), indent=1)
    json.dump(env, open(os.path.join(run_dir, f"env_{METHOD}.json"), "w"), indent=1)

    class SnapAttacker(ttba.Attacker):
        def __init__(self, *args, checkpoints=(), **kw):
            super().__init__(*args, **kw)
            self._last_bx = None
            self.recorder = CheckpointRecorder(checkpoints, self.L2_line[0][1])
            self.L2_line = HookList(self.L2_line, self)

        def bin_search_fast(self, adv_x, tol):
            out, l2, n = super().bin_search_fast(adv_x, tol)
            self._last_bx = out[1].detach().clone()
            return out, l2, n

    args = SimpleNamespace(budget=budget, epsilon=a.epsilon, early=0, remember=0, targeted=a.targeted, norm="TtBA")
    t_all = time.time()
    for order, r in enumerate(rows):
        name = r["source"]
        if name in done:
            continue
        torch.manual_seed(42 + order)
        torch.cuda.manual_seed_all(42 + order)
        x0 = C.load_x0_any(arch, name, device)
        pred = torch_model.predict_label(x0.cuda()).cpu().item()  # evaluation-only sanity check (raw model)
        tar_x, tar_y = None, None
        if a.targeted:
            tar_x = C.load_x0_any(arch, r["target"], device)[0]  # (3,224,224): the official Attacker unsqueezes it
            tar_y = int(r["target_label"])

        y_tgt = tar_y if a.targeted else None
        tracker = C.HardLabelTracker(C.ClampedVictim(torch_model), x0, int(r["source_label"]), y_tgt,
                                     checkpoints, budget, to_pixel=C.to_pixel_clamp,
                                     init_adv_pixel=tar_x if a.targeted else None)
        t0 = time.time()
        status, err = "ok", ""
        try:
            atk = SnapAttacker(args, tracker, x0, int(r.get("loader_pos", order) or order), "TtBA", tar_x, tar_y,
                               checkpoints=checkpoints)
            atk.attack()
        except BudgetExhausted:
            status = "budget"
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            status, err = "error", f"{type(e).__name__}: {e}"
        wall = time.time() - t0

        if status == "error":
            write_rows(run_path, RUN_FIELDS, [{"arch": arch, "mode": mode, "image": name, "method": METHOD,
                                               "queries": tracker.queries, "total_queries": -1,
                                               "nonfinite_queries": tracker.nonfinite_queries, "status": status,
                                               "wall_time_s": round(wall, 2), "error": err}])
            print(f"[!] {name} ERROR {err}", flush=True)
            continue

        official_snap = atk.recorder.finish()
        ck_rows = tracker.finalize(os.path.join(run_dir, "adv_store"), arch, C.stem_of(name), METHOD, mode,
                                   eval_model=torch_model)
        x0c = atk.x0
        for row in ck_rows:
            b = row["budget"]
            row["source_label"] = r["source_label"]
            row["target_label"] = r.get("target_label", "") or ""
            row["seed_mode"] = a.seed_mode
            row["pix_l2_iter"] = official_snap[b][1] if b in official_snap else float("nan")
            snap = tracker.snap.get(b)
            row["ssim"] = _ssim(snap, x0c) if snap is not None else float("nan")
        write_rows(ck_path, CKPT_FIELDS, ck_rows)
        tracker.save_trace(os.path.join(run_dir, "traces", arch, C.stem_of(name), f"{METHOD}.npz"))
        write_rows(run_path, RUN_FIELDS, [{
            "arch": arch, "mode": mode, "image": name, "method": METHOD, "queries": tracker.queries,
            "total_queries": atk.queries, "nonfinite_queries": tracker.nonfinite_queries, "status": status,
            "wall_time_s": round(wall, 2), "error": err,
            "success_eps": int(atk.success == 1), "x0_pred_ok": int(pred == int(r["source_label"])),
            "seed_mode": a.seed_mode}])
        last = ck_rows[-1]
        el = time.time() - t_all
        left = el / (order + 1) * (len(rows) - order - 1) / 3600
        print(f"[{order + 1}/{len(rows)}] {name} q={tracker.queries}/{atk.queries} "
              f"L2@{last['budget']}={last['pix_l2']} iter={last['pix_l2_iter']} {status} {wall:.1f}s | ~{left:.1f}h",
              flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttba_repo", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--arch", choices=["resnet50_places365", "densenet161_places365", "Salman2020Do_R50"],
                    help="E10/E11 victim (generic path); overrides --dataset")
    ap.add_argument("--targeted", type=int, choices=[0, 1], required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budget", type=int, default=None, help="default: contract budget")
    ap.add_argument("--epsilon", type=float, default=2.5)
    ap.add_argument("--seed_mode", choices=["exact", "per_image"], default="exact")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.arch is not None:
        return _main_generic(a)
    if a.dataset is None:
        raise SystemExit("[!] provide --dataset (ImageNet) or --arch (Places365/RobustBench)")

    contract = C.load_contract()
    checkpoints = C.contract_checkpoints(contract)
    budget = a.budget if a.budget is not None else C.contract_budget(contract)

    if a.seed_mode == "exact" and a.num_shards != 1:
        raise SystemExit("[!] --seed_mode exact replays main.py's single RNG stream; use --seed_mode per_image for shards")
    repo, out_dir = os.path.abspath(a.ttba_repo), os.path.abspath(a.out_dir)
    split_csv = os.path.abspath(a.split_csv)
    arch, mode = DATASETS[a.dataset], ("targeted" if a.targeted else "non_targeted")
    rows = read_split_rows(split_csv)
    if a.limit:
        rows = rows[: a.limit]
    if bool(a.targeted) != bool(rows and rows[0].get("target")):
        raise SystemExit(f"[!] --targeted {a.targeted} does not match the split {split_csv}")

    suffix = "" if a.num_shards == 1 else f"_shard{a.shard_index:02d}of{a.num_shards:02d}"
    run_dir = os.path.join(out_dir, mode)
    os.makedirs(run_dir, exist_ok=True)
    ck_path = os.path.join(run_dir, f"per_checkpoint_{METHOD}{suffix}.csv")
    run_path = os.path.join(run_dir, f"runs_{METHOD}{suffix}.csv")
    state_path = os.path.join(run_dir, f"ttba_v2_rng_state{suffix}.pt")
    done = done_images(run_path)

    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch

    import dataset as ttds  # official
    import TtBA as ttba  # official

    # ---- env + manifest (before any attack) -------------------------------------------------------
    repo_commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()
    wname = WEIGHTS[arch]
    wpath = os.path.join(repo, "model", wname)
    env = {"method": METHOD, "arch": arch, "mode": mode, "repo": repo, "repo_commit": repo_commit,
           "params": PARAMS, "hyper_parameter_sources": HYPER_SOURCES,
           "split_csv": split_csv, "split_csv_sha256": C.sha256_file(split_csv),
           "victim_weight_file": wname, "victim_weight_sha256": C.sha256_file(wpath) if os.path.exists(wpath) else "MISSING",
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "budget": budget, "max_queries": budget, "checkpoints": checkpoints, "budgets": checkpoints,
           "precision": {
               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark,
           },
           "adapter_sha256": C.sha256_file(os.path.abspath(__file__)),
           "tracker_sha256": C.sha256_file(os.path.join(REPO_ROOT, "tools", "eval", "hardlabel_tracker.py")),
           "contract_sha256": C.sha256_file(C.CONTRACT_PATH),
           "adv_store_dtype": os.environ.get("ADV_STORE_DTYPE", "float16"),
           "ADV_STORE_BUDGETS": "unset" if not os.environ.get("ADV_STORE_BUDGETS") else os.environ["ADV_STORE_BUDGETS"]}
    manifest = C.build_manifest(env, contract)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in manifest:
            if k in ("gpu_name", "torch", "cuda"):
                continue
            if old.get(k) != manifest[k]:
                raise SystemExit(f"[!] run_manifest mismatch on {k!r}; refusing to resume")
    else:
        C.check_manifest(run_dir, manifest)
        json.dump(manifest, open(manifest_path, "w"), indent=1)
    json.dump(env, open(os.path.join(run_dir, f"env_{METHOD}.json"), "w"), indent=1)

    class SnapAttacker(ttba.Attacker):
        """Official Attacker + observation hooks (unchanged from ttba_run_official.py)."""

        def __init__(self, *args, checkpoints=(), **kw):
            super().__init__(*args, **kw)
            self._last_bx = None
            self.recorder = CheckpointRecorder(checkpoints, self.L2_line[0][1])
            self.L2_line = HookList(self.L2_line, self)

        def bin_search_fast(self, adv_x, tol):
            out, l2, n = super().bin_search_fast(adv_x, tol)
            self._last_bx = out[1].detach().clone()
            return out, l2, n

    test_loader, torch_model = ttds.load_dataset_model(a.dataset)  # == main.py (seeds 42, init, weights)
    ds = test_loader.dataset
    it = iter(test_loader)  # == main.py's enumerate(test_loader)
    next(it)
    by_name = {os.path.basename(p): i for i, (p, _) in enumerate(ds.samples)}
    if a.seed_mode == "exact" and os.path.exists(state_path) and done:
        st = torch.load(state_path, weights_only=False)
        torch.set_rng_state(st["cpu"])
        torch.cuda.set_rng_state_all(st["cuda"])
        random.setstate(st["py"])
        print(f"[*] resumed after {len(done)} images (RNG state restored)")
    elif a.seed_mode == "exact" and done:
        raise SystemExit(f"[!] {run_path} has results but {state_path} is missing; cannot resume the exact stream")

    args = SimpleNamespace(budget=budget, epsilon=a.epsilon, early=0, remember=0, targeted=a.targeted, norm="TtBA")
    todo = list(enumerate(rows))[a.shard_index:: a.num_shards]
    t_all = time.time()
    for n_done, (order, r) in enumerate(todo):
        name = r["source"]
        if name in done:
            continue
        if a.seed_mode == "per_image":
            torch.manual_seed(42 + int(r["order"]))
            torch.cuda.manual_seed_all(42 + int(r["order"]))
        img, lab = ds[by_name[name]]
        if int(lab) != int(r["source_label"]):
            raise SystemExit(f"[!] {name}: label {lab} != split label {r['source_label']} (different dataset?)")
        x0 = img.unsqueeze(0)  # [1,3,224,224] in [0,1]
        pred = torch_model.predict_label(x0.cuda()).cpu().item()  # evaluation-only sanity check (raw model)
        tar_x, tar_y = None, None
        if a.targeted:
            tar_x, tar_y = ds[by_name[r["target"]]]
            tar_x, tar_y = tar_x.cuda(), int(tar_y)

        y_tgt = tar_y if a.targeted else None
        tracker = C.HardLabelTracker(C.ClampedVictim(torch_model), x0.cuda(), int(r["source_label"]), y_tgt,
                                     checkpoints, budget, to_pixel=C.to_pixel_clamp,
                                     init_adv_pixel=tar_x if a.targeted else None)
        t0 = time.time()
        status, err = "ok", ""
        try:
            atk = SnapAttacker(args, tracker, x0, int(r.get("loader_pos", order) or order), "TtBA", tar_x, tar_y,
                               checkpoints=checkpoints)
            atk.attack()
        except BudgetExhausted:
            status = "budget"
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            status, err = "error", f"{type(e).__name__}: {e}"
        wall = time.time() - t0

        if status == "error":
            write_rows(run_path, RUN_FIELDS, [{"arch": arch, "mode": mode, "image": name, "method": METHOD,
                                               "queries": tracker.queries, "total_queries": -1,
                                               "nonfinite_queries": tracker.nonfinite_queries, "status": status,
                                               "wall_time_s": round(wall, 2), "error": err}])
            print(f"[!] {name} ERROR {err}", flush=True)
            continue

        official_snap = atk.recorder.finish()  # budget -> (q, val, bx) official iterate
        ck_rows = tracker.finalize(os.path.join(run_dir, "adv_store"), arch, C.stem_of(name), METHOD, mode,
                                   eval_model=torch_model)
        x0c = atk.x0
        for row in ck_rows:
            b = row["budget"]
            row["source_label"] = r["source_label"]
            row["target_label"] = r.get("target_label", "") or ""
            row["seed_mode"] = a.seed_mode
            row["pix_l2_iter"] = official_snap[b][1] if b in official_snap else float("nan")
            snap = tracker.snap.get(b)
            row["ssim"] = _ssim(snap, x0c) if snap is not None else float("nan")
        write_rows(ck_path, CKPT_FIELDS, ck_rows)
        tracker.save_trace(os.path.join(run_dir, "traces", arch, C.stem_of(name), f"{METHOD}.npz"))
        write_rows(run_path, RUN_FIELDS, [{
            "arch": arch, "mode": mode, "image": name, "method": METHOD, "queries": tracker.queries,
            "total_queries": atk.queries, "nonfinite_queries": tracker.nonfinite_queries, "status": status,
            "wall_time_s": round(wall, 2), "error": err,
            "success_eps": int(atk.success == 1), "x0_pred_ok": int(pred == int(r["source_label"])),
            "seed_mode": a.seed_mode}])
        if a.seed_mode == "exact":
            torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "py": random.getstate()}, state_path)
        last = ck_rows[-1]
        el = time.time() - t_all
        left = el / (n_done + 1) * (len(todo) - n_done - 1) / 3600
        print(f"[{order + 1}/{len(rows)}] {name} q={tracker.queries}/{atk.queries} "
              f"L2@{last['budget']}={last['pix_l2']} iter={last['pix_l2_iter']} {status} {wall:.1f}s | ~{left:.1f}h",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
