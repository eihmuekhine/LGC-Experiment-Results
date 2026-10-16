#!/usr/bin/env python3
"""B4g T1 — known-answer test on the ``kat_linear`` victim (linear classifier, K = 10).

Builds a linear victim and runs EVERY method through the normal loader, tracker and adv store, budget 10000 and
the contract checkpoints, then verifies the analytic known answer on the stored npz files.

Victim (also built by ``adapters/_common.build_kat_linear_victim``, the exact same W/b):
    input x in [0,1]^(3x224x224); z = (x - mu)/sd (ImageNet mu/sd inside); logits = W vec(z) + b, K = 10.
    W_k = bilinear upsampling of a (3,14,14) N(0,1) array to (3,224,224) (seed 0); b ~ N(0,1) (seed 0).
    The pixel-space gradient of logit k is g_k = W_k / sd (per channel).

Images: 256x256 RGB PNGs (values k/255 so Resize(256) is a no-op and CenterCrop(224) is exact); candidates
0.5 + 0.08*N(0,1) clipped to [0.3, 0.7] (seed i); keep the first 5 whose analytic non-targeted distance
d* = min_{j != y} (l_y - l_j) / ||g_y - g_j||_2 lies in [1, 10] and whose minimiser
x0 - d* (g_y - g_j*)/||g_y - g_j*|| lies inside [0,1].  Targeted: t = second-largest logit; target image = a
candidate classified as t; LB_t = (l_y - l_t)/||g_y - g_t||_2 is a lower bound on any targeted L2.

Checks per (method, mode, image, checkpoint with found = 1), using the stored npz (delta float16, u = 2^-11):
  (a) adversarial: margin m(x0 + delta) <= max_j ||g_ref - g_j|| * u * ||delta||
      (NT ref = y, m = l_y - max_{j != y} l_j; T ref = t, m = max_{j != t} l_j - l_t);
  (b) lower bound: ||delta|| >= d* (1 - u)  (NT)  /  ||delta|| >= LB_t (1 - u)  (T);
  (c) pix_l2 in per_checkpoint equals ||delta|| (relative error < 1e-3);
  (d) queries == 10000 for baselines, <= 10000 for ours; tracker count == forward count at the linear victim
      minus the listed evaluation-only calls;
  (e) descriptive: pix_l2@10000 / d* (NT) or / LB_t (T) per method and image (no pass/fail).

Exit code 0 iff every (a)-(d) check passes.  Any FAIL in (a)-(d) => STOP (B5 must not start).

Usage (from the worktree root):
    OMP_NUM_THREADS=8 python adapters/diag/kat_linear.py
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

import _common as C  # noqa: E402  (configure_precision runs at import -> D12 flags)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

MEAN = C.MEAN
STD = C.STD
K = C.KAT_LINEAR_K
U = 2.0 ** -11
N_IMAGES = 5
BUDGET = 10000

KAT_DATA_DIR = C.KAT_DATA_DIR
IMAGES_DIR = os.path.join(KAT_DATA_DIR, "images")
SPLIT_DIR = os.path.join(REPO_ROOT, "splits", "kat")
RUNS_DIR = os.path.join(REPO_ROOT, "generated", "smoke_canary", "kat_runs")

# (method_id, mode, runner module, arg builder)
RUN_SPECS = [
    ("LGC_RC_EGC", "non_targeted"),
    ("LGC_RC_EGC", "targeted"),
    ("TtBA", "non_targeted"),
    ("TtBA", "targeted"),
    ("CGBA", "non_targeted"),
    ("CGBA_H", "targeted"),
    ("ARS_OPT", "non_targeted"),
    ("ARS_OPT", "targeted"),
]

SPLIT_FIELDS = ["source", "source_label", "target", "target_label", "order", "protocol", "victim_weights",
                "loader_pos", "source_relpath", "target_relpath"]


# --------------------------------------------------------------------------------------------- victim helpers
def build_victim(device: str):
    return C.build_kat_linear_victim(device)


def victim_grad(victim):
    """g_k = W_k / sd (per channel) -> (K, 3, 224, 224)."""
    return victim.m.W / victim.sd.view(1, 3, 1, 1)


def logits_of(victim, x0):
    """x0 (1,3,224,224) [0,1] -> (K,) logits (exact linear map)."""
    with torch.no_grad():
        return victim(x0)[0]


# -------------------------------------------------------------------------------------------- image generation
def save_png(arr: np.ndarray, path: str) -> None:
    u = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(u.transpose(1, 2, 0), mode="RGB").save(path)


def load_x0_cpu(name: str) -> torch.Tensor:
    """(1,3,224,224) [0,1] via the SAME loader the attacks use (ttba_github_tensor)."""
    return C.load_image(KAT_DATA_DIR, name, "cpu")


def make_candidate(rng) -> np.ndarray:
    """(3,256,256) float32 candidate in [0.3,0.7].

    B4g deviation (documented in the report): the victim W is a bilinear upsampling of a (3,14,14) array, so it is
    band-limited to ~14x14 and its decisions depend only on the LOW-frequency content of the image.  White-noise
    candidates (0.5 + 0.08*N(0,1) per pixel) therefore have a constant low-frequency content (their mean is
    0.5 + O(0.0002)), so EVERY candidate is classified as the same class (verified: 2000/2000 -> class 1) and the
    targeted split (a candidate classified as the second-largest logit t) cannot be built.  To make the test
    well-posed we keep the spec's mean (0.5) and per-pixel std (0.08) but draw the N(0,1) on a (3,14,14) grid and
    bilinearly upsample it to (3,256,256), i.e. the noise carries the same low-frequency structure the victim sees.
    """
    low = rng.standard_normal((3, 14, 14)).astype(np.float32)
    sm = torch.nn.functional.interpolate(torch.from_numpy(low[None]), size=(256, 256),
                                         mode="bilinear", align_corners=False)[0].numpy()
    sm = sm / (float(sm.std()) + 1e-9) * 0.08
    return np.clip(0.5 + sm, 0.3, 0.7).astype(np.float32)


def generate_images(victim, device: str, n_images: int = N_IMAGES, max_candidates: int = 2000):
    """Generate candidates, select the first n_images with d* in [1,10] and minimiser in [0,1], return source info.

    Returns dict: stem -> {name, x0 (1,3,224,224 cpu), x0_cuda, y, t, dstar, lb_t, target, target_label}.
    """
    if os.path.isdir(IMAGES_DIR):
        shutil.rmtree(IMAGES_DIR)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    g = victim_grad(victim)                      # (K,3,224,224) on device
    gflat = g.flatten(1)                         # (K, D)
    src_info: dict = {}
    pool: dict = {}                              # stem -> label (argmax)
    i = 0
    while i < max_candidates:
        cand = make_candidate(np.random.RandomState(i))
        name = f"kat_{i}.png"
        save_png(cand, os.path.join(IMAGES_DIR, name))
        x0 = load_x0_cpu(name).to(device)        # (1,3,224,224)
        l = logits_of(victim, x0)                # (K,)
        y = int(l.argmax().item())
        stem = C.stem_of(name)
        pool[stem] = y
        # analytic NT distance d* = min_{j != y} (l_y - l_j) / ||g_y - g_j||_2
        best, jstar = float("inf"), -1
        for j in range(K):
            if j == y:
                continue
            denom = float((gflat[y] - gflat[j]).norm().item())
            dj = float((l[y] - l[j]).item()) / denom
            if dj < best:
                best, jstar = dj, j
        dstar = best
        unit = (g[y] - g[jstar]) / (g[y] - g[jstar]).flatten().norm()
        minimiser = x0 - dstar * unit
        inside = bool(((minimiser >= 0.0) & (minimiser <= 1.0)).all().item())
        if 1.0 <= dstar <= 10.0 and inside and len(src_info) < n_images:
            lv = l.tolist()
            t = int(np.argsort(lv)[-2]) if K > 1 else y
            denom_t = float((gflat[y] - gflat[t]).norm().item())
            lb_t = float((l[y] - l[t]).item()) / denom_t
            src_info[stem] = {"name": name, "x0": x0.detach().cpu(), "x0_cuda": x0.detach(),
                              "y": y, "t": t, "dstar": dstar, "lb_t": lb_t}
        i += 1
        # stop once every source is found AND every target class is represented in the pool
        if len(src_info) >= n_images:
            needed = {v["t"] for v in src_info.values()}
            if needed <= set(pool.values()):
                break
    if len(src_info) < n_images:
        raise SystemExit(f"[!] only {len(src_info)}/{n_images} valid sources found in {max_candidates} candidates")
    # targets: for each source, the first candidate (seed order) classified as t
    for stem in src_info:
        info = src_info[stem]
        t = info["t"]
        target = None
        for j in range(i):
            if pool.get(f"kat_{j}") == t:
                target = f"kat_{j}.png"
                break
        if target is None:
            raise SystemExit(f"[!] no candidate classified as class {t} found (source {stem})")
        info["target"] = target
        info["target_label"] = t
    return src_info


def write_splits(src_info: dict) -> None:
    os.makedirs(SPLIT_DIR, exist_ok=True)
    nt_path = os.path.join(SPLIT_DIR, "kat_linear_non_targeted.csv")
    t_path = os.path.join(SPLIT_DIR, "kat_linear_targeted.csv")
    order = list(src_info.keys())
    with open(nt_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_FIELDS)
        w.writeheader()
        for i, stem in enumerate(order):
            info = src_info[stem]
            w.writerow({"source": info["name"], "source_label": info["y"], "target": "", "target_label": "",
                        "order": i, "protocol": "kat_linear:linear:targeted0", "victim_weights": "kat_linear_seed0",
                        "loader_pos": i, "source_relpath": f"images/{info['name']}", "target_relpath": ""})
    with open(t_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_FIELDS)
        w.writeheader()
        for i, stem in enumerate(order):
            info = src_info[stem]
            w.writerow({"source": info["name"], "source_label": info["y"], "target": info["target"],
                        "target_label": info["target_label"], "order": i,
                        "protocol": "kat_linear:linear:targeted1", "victim_weights": "kat_linear_seed0",
                        "loader_pos": i, "source_relpath": f"images/{info['name']}",
                        "target_relpath": f"images/{info['target']}"})


# --------------------------------------------------------------------------------------------- runner commands
def run_command(method: str, mode: str, split_csv: str, out_dir: str) -> list:
    py = sys.executable
    if method == "LGC_RC_EGC":
        return [py, "ttba_github/run_frame_b1_generic.py", "--arch", "kat_linear", "--mode", mode,
                "--split_csv", split_csv, "--out_dir", out_dir, "--data_dir", KAT_DATA_DIR,
                "--num_samples", str(N_IMAGES), "--save_adv"]
    if method == "TtBA":
        return [py, "adapters/diag/kat_ttba_wrapper.py", "--targeted", "1" if mode == "targeted" else "0",
                "--split_csv", split_csv, "--out_dir", out_dir, "--limit", str(N_IMAGES)]
    if method in ("CGBA", "CGBA_H"):
        return [py, "adapters/run_cgba.py", "--arch", "kat_linear", "--mode", mode, "--split_csv", split_csv,
                "--out_dir", out_dir, "--repo", "/root/autodl-tmp/baselines/CGBA", "--limit", str(N_IMAGES)]
    if method == "ARS_OPT":
        return [py, "adapters/run_arsopt.py", "--arch", "kat_linear", "--mode", mode, "--split_csv", split_csv,
                "--out_dir", out_dir, "--repo", "/root/autodl-tmp/baselines/ARS-OPT", "--limit", str(N_IMAGES)]
    raise SystemExit(f"[!] unknown method {method}")


def per_checkpoint_path(out_dir: str, mode: str, method: str) -> str:
    run_dir = os.path.join(out_dir, mode)
    if method == "LGC_RC_EGC":
        return os.path.join(run_dir, "per_checkpoint.csv")
    return os.path.join(run_dir, f"per_checkpoint_{method}.csv")


def runs_path(out_dir: str, mode: str, method: str) -> str:
    run_dir = os.path.join(out_dir, mode)
    if method == "LGC_RC_EGC":
        return os.path.join(run_dir, "runs.csv")
    return os.path.join(run_dir, f"runs_{method}.csv")


def run_all(run_dir_base: str) -> None:
    os.makedirs(run_dir_base, exist_ok=True)
    for method, mode in RUN_SPECS:
        split_csv = os.path.join(SPLIT_DIR, f"kat_linear_{mode}.csv")
        out_dir = os.path.join(run_dir_base, method)
        cmd = run_command(method, mode, split_csv, out_dir)
        fc_file = os.path.join(run_dir_base, f"forward_count_{method}_{mode}.txt")
        env = dict(os.environ)
        env["KAT_FORWARD_COUNT_FILE"] = fc_file
        env["OMP_NUM_THREADS"] = "8"
        env["MKL_NUM_THREADS"] = "8"
        print(f"\n===== {method} {mode} =====", flush=True)
        print("  " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if r.returncode != 0:
            print(f"[!] {method} {mode} exited {r.returncode}", flush=True)
        else:
            print(f"[+] {method} {mode} done", flush=True)


# --------------------------------------------------------------------------------------------------- the checks
def read_ckpt_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_runs(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def queries_of(runs_rows: list, method: str, image_stem: str) -> int:
    """Per-image query count.  ours: total_queries (res.queries); baselines: queries (tracker)."""
    for r in runs_rows:
        stem = C.stem_of(r.get("image", ""))
        if stem != image_stem:
            continue
        if method == "LGC_RC_EGC":
            return int(r.get("total_queries", r.get("queries", -1)))
        return int(r.get("queries", -1))
    return -1


def eval_only_calls(method: str, ckpt_rows: list, image_stem: str) -> int:
    """The listed evaluation-only forward passes through the raw victim for one image."""
    n_found = sum(1 for r in ckpt_rows if C.stem_of(r.get("image", "")) == image_stem
                  and int(r.get("found", 0)) == 1)
    if method == "LGC_RC_EGC":
        tot = 0
        for r in ckpt_rows:
            if C.stem_of(r.get("image", "")) != image_stem or int(r.get("found", 0)) != 1:
                continue
            repair = int(float(r.get("repair_queries", 0) or 0))
            tot += 2 + repair  # image_metrics adv_float + adv_uint8 (2) + uint8_repair trials
        return tot
    if method == "TtBA":
        return 1 + n_found  # x0_pred_ok (1) + finalize adv_uint8 per found checkpoint
    return n_found  # CGBA / CGBA_H / ARS_OPT: finalize adv_uint8 per found checkpoint


def run_checks(method: str, mode: str, src_info: dict, run_dir_base: str, victim, gflat) -> dict:
    """Run checks (a)-(e) for one (method, mode).  Returns a summary dict."""
    out_dir = os.path.join(run_dir_base, method)
    ck = read_ckpt_rows(per_checkpoint_path(out_dir, mode, method))
    runs = read_runs(runs_path(out_dir, mode, method))
    adv_store = os.path.join(out_dir, mode, "adv_store", "kat_linear")

    g = victim_grad(victim)
    counts = {"a": [0, 0], "b": [0, 0], "c": [0, 0], "d": [0, 0]}
    ratios = []  # (e) at budget 10000
    n_checked = 0
    detail = []

    # (d2) forward count == sum(queries) + sum(eval_only)
    fc_file = os.path.join(run_dir_base, f"forward_count_{method}_{mode}.txt")
    forward_count = None
    if os.path.exists(fc_file):
        with open(fc_file, encoding="utf-8") as fh:
            forward_count = int(fh.read().strip())
    sum_queries = 0
    sum_eval = 0
    images = list(src_info.keys())
    d1_pass = 0
    for stem in images:
        q = queries_of(runs, method, stem)
        eo = eval_only_calls(method, ck, stem)
        sum_queries += max(q, 0)
        sum_eval += eo
        ok_d1 = (q == BUDGET) if method != "LGC_RC_EGC" else (0 <= q <= BUDGET)
        if ok_d1:
            d1_pass += 1
    residual = (forward_count - sum_queries - sum_eval) if forward_count is not None else None
    counts["d"] = [d1_pass, len(images)]

    for stem in images:
        info = src_info[stem]
        y, t = info["y"], info["t"]
        dstar, lb_t = info["dstar"], info["lb_t"]
        x0 = info["x0"].to(victim.m.W.device)  # (1,3,224,224)
        l_x0 = logits_of(victim, x0)
        lip = float((gflat - gflat[y if mode == "non_targeted" else t]).norm(dim=1).max().item())
        for r in ck:
            if C.stem_of(r.get("image", "")) != stem:
                continue
            if int(r.get("found", 0)) != 1:
                continue
            budget = int(r["budget"])
            pix_l2 = float(r["pix_l2"])
            npz = os.path.join(adv_store, stem, f"{method}_Q{budget}.npz")
            if not os.path.exists(npz):
                detail.append((stem, budget, "npz_missing"))
                continue
            z = np.load(npz, allow_pickle=False)
            delta16 = z["delta"].astype(np.float32)          # (3,224,224)
            norm_delta = float(np.linalg.norm(delta16.ravel()))
            n_checked += 1
            # (a) adversarial
            xadv = (x0 + torch.from_numpy(delta16).to(victim.m.W.device)).float()
            l_adv = logits_of(victim, xadv)
            if mode == "non_targeted":
                m = float((l_adv[y] - torch.cat([l_adv[:y], l_adv[y + 1:]]).max()).item())
            else:
                m = float((torch.cat([l_adv[:t], l_adv[t + 1:]]).max() - l_adv[t]).item())
            ok_a = m <= lip * U * norm_delta + 1e-6
            # (b) lower bound
            bound = dstar if mode == "non_targeted" else lb_t
            ok_b = norm_delta >= bound * (1.0 - U) - 1e-6
            # (c) pix_l2 == ||delta||
            rel = abs(pix_l2 - norm_delta) / pix_l2 if pix_l2 else float("inf")
            ok_c = rel < 1e-3
            counts["a"][0] += int(ok_a)
            counts["b"][0] += int(ok_b)
            counts["c"][0] += int(ok_c)
            if not (ok_a and ok_b and ok_c):
                detail.append((stem, budget, f"a={m:.3e}>{lip * U * norm_delta:.3e}" if not ok_a else
                               f"b={norm_delta:.4f}<{bound * (1 - U):.4f}" if not ok_b else
                               f"c rel={rel:.3e}"))
            if budget == BUDGET:
                ratios.append(pix_l2 / (dstar if mode == "non_targeted" else lb_t))
    return {"method": method, "mode": mode, "n": n_checked,
            "a": counts["a"], "b": counts["b"], "c": counts["c"], "d": counts["d"],
            "d2_forward_count": forward_count, "d2_sum_queries": sum_queries, "d2_sum_eval": sum_eval,
            "d2_residual": residual, "ratios": ratios, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check_only", action="store_true", help="skip the attack runs (results must exist)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[!] no CUDA device; the known-answer test must run on the study GPU")
        return 2
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[*] D12 flags: matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} deterministic={torch.backends.cudnn.deterministic} "
          f"benchmark={torch.backends.cudnn.benchmark}")

    victim = build_victim(device)
    src_info = generate_images(victim, device)
    write_splits(src_info)
    print(f"[*] selected {len(src_info)} sources: "
          + ", ".join(f"{s}(y={v['y']},d*={v['dstar']:.3f})" for s, v in src_info.items()))

    if not a.check_only:
        if os.path.isdir(RUNS_DIR):
            shutil.rmtree(RUNS_DIR)
        run_all(RUNS_DIR)

    gflat = victim_grad(victim).flatten(1)
    results = []
    all_pass = True
    print("\n| method | mode | n | (a) | (b) | (c) | (d) queries | fwd-count residual | median ratio | max ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for method, mode in RUN_SPECS:
        res = run_checks(method, mode, src_info, RUNS_DIR, victim, gflat)
        results.append(res)
        r = res["ratios"]
        med = float(np.median(r)) if r else float("nan")
        mx = float(np.max(r)) if r else float("nan")
        ok = (res["a"][0] == res["n"] and res["b"][0] == res["n"] and res["c"][0] == res["n"]
              and res["d"][0] == res["d"][1] and res["d2_residual"] == 0)
        all_pass &= ok
        fwd = "MISSING" if res["d2_forward_count"] is None else res["d2_residual"]
        print(f"| {res['method']} | {res['mode']} | {res['n']} | {res['a'][0]}/{res['n']} | "
              f"{res['b'][0]}/{res['n']} | {res['c'][0]}/{res['n']} | {res['d'][0]}/{res['d'][1]} | "
              f"{fwd} | {med:.3f} | {mx:.3f} |")
        for d in res["detail"]:
            print(f"    FAIL {res['method']} {res['mode']} {d}", flush=True)

    print()
    if all_pass:
        print("[+] T1 known-answer test: all (a)-(d) PASS")
        return 0
    print("[!] T1 known-answer test: FAIL in (a)-(d) -> STOP")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
