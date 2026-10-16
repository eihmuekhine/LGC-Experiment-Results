#!/usr/bin/env python3
"""Run the OFFICIAL TtBA attack (github.com/BUPTAIOC/TtBA, file TtBA.py, class Attacker) on a dumped split and
record per-image checkpoints in our per_checkpoint format (build 38).

What is official and unchanged
  - TtBA.Attacker: same defaults as main.py (dim_reduc_factor 4, tol 1e-4, sigma 3e-4, initial_query 30,
    iteration 100) and same args (budget, epsilon, early 0).
  - dataset.load_dataset_model: seeds 42, victim construction, weight file, GeneralTorchModel, transforms.
  - Query counting is the Attacker's own counter.
What is added (observation only; no extra victim query is charged to the attack)
  - SnapAttacker overrides bin_search_fast to keep a clone of the boundary point it returns, and replaces
    self.L2_line by a list that, on each append, hands (queries, L2, boundary point) to a CheckpointRecorder.
  - After the attack, at each budget c, the state of the last iteration with queries <= c is scored. The repo's
    reported L2 (unclamped bisection distance) and the L2 / Linf of the clamped image are recorded. Adversarial
    status and LPIPS are evaluated on the clamped image, with evaluation-only calls.
RNG stream
  --seed_mode exact (default): one process per (model, mode), images in split order.  The torch CPU stream is
      consumed exactly as in main.py:
        - load_dataset_model seeds and initialises;
        - iter(loader) + one next() draw the loader seeds;
        - after that, only the Attacker draws, and only main.py's attacks happen in between.
      In the same environment this reproduces main.py's run on this list.  Resumable: the RNG state is saved after
      every image.
  --seed_mode per_image: torch.manual_seed(42 + order) before each image; allows --num_shards (a different but
      equally valid random stream; stated in the output column seed_mode).

Usage
  python ttba_github/ttba_run_official.py --ttba_repo /root/autodl-tmp/TtBA --dataset imagenet-resnet50 --targeted 0 \
     --split_csv $PWD/splits/imagenet/ttba_github/resnet50_non_targeted.csv --out_dir $PWD/Results/ttbagh_resnet50
Output  <out_dir>/<mode>/ttbagh_per_checkpoint[_shardXXofYY].csv, ttbagh_runs[...].csv
        (not named per_checkpoint*.csv on purpose: our runner's own report never reads them; confirm_ttba.py does)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ttba_common import (CKPT_FIELDS, DATASETS, METHOD, RUN_FIELDS, CheckpointRecorder, HookList,  # noqa: E402
                         done_images, read_split_rows, write_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttba_repo", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--targeted", type=int, choices=[0, 1], required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--epsilon", type=float, default=2.5, help="ASR threshold of the paper's Table 5")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 2500, 5000, 10000])
    ap.add_argument("--seed_mode", choices=["exact", "per_image"], default="exact")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="attack only the first N rows (smoke test)")
    ap.add_argument("--lpips_net", default="alex", help="alex | vgg | squeeze | none (use the net of our metrics.py)")
    a = ap.parse_args()
    if a.seed_mode == "exact" and a.num_shards != 1:
        raise SystemExit("[!] --seed_mode exact replays main.py's single RNG stream; use --seed_mode per_image "
                         "for shards")
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
    ck_path = os.path.join(run_dir, f"ttbagh_per_checkpoint{suffix}.csv")
    run_path = os.path.join(run_dir, f"ttbagh_runs{suffix}.csv")
    state_path = os.path.join(run_dir, f"ttbagh_rng_state{suffix}.pt")
    done = done_images(run_path)

    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch

    lp_net = None  # built BEFORE load_dataset_model: its construction consumes RNG, which load_dataset_model resets
    if a.lpips_net.lower() != "none":
        try:
            import lpips
            lp_net = lpips.LPIPS(net=a.lpips_net, verbose=False).cuda().eval()
        except Exception as err:  # noqa: BLE001
            print(f"[!] LPIPS disabled ({err}); pip install lpips")
    import dataset as ttds  # official
    import TtBA as ttba  # official

    class SnapAttacker(ttba.Attacker):
        """Official Attacker + observation hooks (see module docstring)."""

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
    it = iter(test_loader)  # == main.py's enumerate(test_loader): loader base seed + sampler seed draws
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
    args = SimpleNamespace(budget=a.budget, epsilon=a.epsilon, early=0, remember=0, targeted=a.targeted,
                           norm="TtBA")
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
        x0 = img.unsqueeze(0)  # main.py passes the loader batch [1, 3, 224, 224]
        pred = torch_model.predict_label(x0.cuda()).cpu().item()
        tar_x, tar_y = None, None
        if a.targeted:
            tar_x, tar_y = ds[by_name[r["target"]]]
            tar_x, tar_y = tar_x.cuda(), int(tar_y)
        t0 = time.time()
        atk = SnapAttacker(args, torch_model, x0, int(r.get("loader_pos", order) or order), "TtBA", tar_x, tar_y,
                           checkpoints=a.checkpoints)
        atk.attack()
        snap = atk.recorder.finish()
        wall = time.time() - t0
        x0c = atk.x0
        ck_rows = []
        for c in sorted(snap):
            q, val, bx = snap[c]
            row = {"dataset": "imagenet", "arch": arch, "mode": mode, "image": name,
                   "source_label": r["source_label"], "target_label": r.get("target_label", ""),
                   "method": METHOD, "space": "pixel", "budget": c, "state_query": q, "pix_l2_reported": val,
                   "ssim": float("nan"), "lpips_net": a.lpips_net if lp_net is not None else "none",
                   "seed_mode": a.seed_mode}
            if bx is None:  # no completed iteration yet: the repo's placeholder value
                row.update(found=0, pix_l2=val, pix_linf=float("nan"), lpips=float("nan"))
            else:
                xc = torch.clamp(bx.unsqueeze(0) if bx.dim() == 3 else bx, 0.0, 1.0)
                d = (xc - x0c).flatten()
                with torch.no_grad():  # evaluation only: not counted, no RNG
                    p = torch_model.model(torch_model.preprocess(xc)).argmax(1).item()
                    lp = float(lp_net(xc * 2 - 1, x0c * 2 - 1).item()) if lp_net is not None else float("nan")
                adv = (p == tar_y) if a.targeted else (p != atk.x0_label)
                row.update(found=int(adv), pix_l2=float(d.norm()), pix_linf=float(d.abs().max()), lpips=lp)
            ck_rows.append(row)
        write_rows(ck_path, CKPT_FIELDS, ck_rows)
        write_rows(run_path, RUN_FIELDS, [{
            "dataset": "imagenet", "arch": arch, "mode": mode, "image": name, "method": METHOD, "order": order,
            "status": "ok", "total_queries": atk.queries, "iterations": len(atk.L2_line) - 1,
            "wall_time_s": round(wall, 2), "success_eps": int(atk.success == 1), "acc_query": atk.ACCquery,
            "final_l2_reported": atk.old_best_adv.Reall2, "x0_pred_ok": int(pred == int(r["source_label"])),
            "seed_mode": a.seed_mode}])
        if a.seed_mode == "exact":
            torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "py": random.getstate()}, state_path)
        last = snap[max(snap)]
        el = time.time() - t_all
        left = el / (n_done + 1) * (len(todo) - n_done - 1) / 3600
        print(f"\n[{order + 1}/{len(rows)}] {name} q={atk.queries} L2@{max(snap)}={last[1]:.4f} "
              f"{wall:.1f}s | ~{left:.1f} h left", flush=True)
        if pred != int(r["source_label"]):
            print(f"[!] {name}: the victim misclassifies x0 here ({pred}); the split came from another environment")


if __name__ == "__main__":
    main()
