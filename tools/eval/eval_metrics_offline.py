"""Offline image-quality metrics for every stored adversarial example (no attack is re-run).

Reads the adv_store folders written by advstore.py (any number of result folders) and computes, with ONE
implementation for every method:

    l2, linf, psnr (data range 1)                     pixel-level, descriptive
    ssim      torchmetrics SSIM (Gaussian 11, sigma 1.5, data_range 1, RGB mean)   - used by our gate: disclose
    ms_ssim   piq.multi_scale_ssim (data_range 1)
    lpips_alex, lpips_vgg   lpips package (inputs scaled to [-1, 1])
    dists     piq.DISTS
    fsim      piq.fsim (data_range 1)
    (optional NR: brisque, clip_iqa via --nr)

Usage on AutoDL (after the runs):
    pip install piq lpips torchmetrics            # once; pre-download weights with --selftest (see README)
    python eval_metrics_offline.py --selftest
    python eval_metrics_offline.py --roots Results/b1_resnet50 Results/b1_vgg19 Results/b1_vit_b_32 \
        --out metrics_all.csv --device cuda --batch 32
Resume-safe: rows already in --out are skipped.  An integrity column l2_relerr compares the recomputed L2 with the
exact float32 L2 stored at run time (float16 deltas: expect < 1e-3).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advstore import load_adv, store_adv  # noqa: E402

FR_DEFAULT = ["l2", "linf", "psnr", "ssim", "ms_ssim", "lpips_alex", "lpips_vgg", "dists", "fsim"]
KEY = ("root", "arch", "mode", "image", "method", "budget")


def find_files(roots: List[str]) -> List[tuple]:
    """Return (root, npz_path) for every stored checkpoint under each root (searches for adv_store folders)."""
    out = []
    for r in roots:
        stores = [r] if os.path.basename(os.path.normpath(r)) == "adv_store" else \
            sorted(glob.glob(os.path.join(r, "**", "adv_store"), recursive=True))
        for s in stores:
            for p in sorted(glob.glob(os.path.join(s, "*", "*", "*_Q*.npz"))):
                out.append((r, p))
    return out


def build_metrics(names: List[str], device: str) -> Dict[str, Callable]:
    """Metric callables on batches x, y of shape (B,3,H,W) in [0,1] -> tensor (B,)."""
    import torch

    fns: Dict[str, Callable] = {}
    flat = lambda t: t.flatten(1)  # noqa: E731
    if "l2" in names:
        fns["l2"] = lambda x, y: flat(x - y).norm(dim=1)
    if "linf" in names:
        fns["linf"] = lambda x, y: flat(x - y).abs().amax(dim=1)
    if "psnr" in names:
        fns["psnr"] = lambda x, y: 10 * torch.log10(1.0 / flat(x - y).pow(2).mean(dim=1).clamp_min(1e-20))
    if "ssim" in names:
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

        fns["ssim"] = lambda x, y: tm_ssim(x, y, data_range=1.0, reduction="none")
    if "ms_ssim" in names or "dists" in names or "fsim" in names:
        import piq
    if "ms_ssim" in names:
        fns["ms_ssim"] = lambda x, y: piq.multi_scale_ssim(x, y, data_range=1.0, reduction="none")
    if "fsim" in names:
        fns["fsim"] = lambda x, y: piq.fsim(x, y, data_range=1.0, reduction="none")
    if "dists" in names:
        dists = piq.DISTS(reduction="none").to(device)
        fns["dists"] = lambda x, y: dists(x, y)
    for net in ("alex", "vgg"):
        if f"lpips_{net}" in names:
            import lpips

            model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
            fns[f"lpips_{net}"] = (lambda mdl: (lambda x, y: mdl(x * 2 - 1, y * 2 - 1).flatten()))(model)
    return fns


def build_nr(names: List[str], device: str) -> Dict[str, Callable]:
    import piq

    fns: Dict[str, Callable] = {}
    if "brisque" in names:
        fns["brisque"] = lambda x: piq.brisque(x, data_range=1.0, reduction="none")
    if "clip_iqa" in names:
        clip = piq.CLIPIQA(data_range=1.0).to(device)
        fns["clip_iqa"] = lambda x: clip(x).flatten()
    return fns


def evaluate(files: List[tuple], out_csv: str, names: List[str], nr: List[str], device: str, batch: int) -> None:
    import torch

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(tuple(r[k] for k in KEY))
    fr, nrf = build_metrics(names, device), (build_nr(nr, device) if nr else {})
    cols = list(KEY) + ["state_query", "adv_flag", "l2_meta", "l2_relerr"] + list(fr) + [f"{k}_adv" for k in nrf]
    new = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()
    todo = []
    for root, p in files:
        z = np.load(p, allow_pickle=False)
        key = (root, str(z["arch"]), str(z["mode"]), str(z["image"]), str(z["method"]), str(int(z["budget"])))
        if key not in done:
            todo.append((key, p))
    print(f"[*] {len(files)} stored states, {len(todo)} to evaluate, metrics {list(fr) + list(nrf)}")
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(todo), batch):
            chunk = todo[b: b + batch]
            xs, ys, metas = [], [], []
            for _, p in chunk:
                xa, x0, meta = load_adv(p)
                xs.append(xa), ys.append(x0), metas.append(meta)
            shapes = {x.shape for x in xs}
            groups = [list(range(len(chunk)))] if len(shapes) == 1 else [[i] for i in range(len(chunk))]
            res: Dict[int, dict] = {i: {} for i in range(len(chunk))}
            for g in groups:
                x = torch.from_numpy(np.stack([xs[i] for i in g])).to(device)
                y = torch.from_numpy(np.stack([ys[i] for i in g])).to(device)
                for name, fn in fr.items():
                    try:
                        v = fn(x, y).detach().float().cpu().numpy().reshape(-1)
                    except Exception as err:  # noqa: BLE001  (e.g. MS-SSIM on images that are too small)
                        print(f"[!] {name}: {err}")
                        v = np.full(len(g), np.nan)
                    for j, i in enumerate(g):
                        res[i][name] = float(v[j])
                for name, fn in nrf.items():
                    v = fn(x).detach().float().cpu().numpy().reshape(-1)
                    for j, i in enumerate(g):
                        res[i][f"{name}_adv"] = float(v[j])
            for i, (key, _) in enumerate(chunk):
                l2m = float(metas[i].get("l2", float("nan")))
                row = dict(zip(KEY, key))
                row.update({"state_query": int(metas[i].get("state_query", -1)),
                            "adv_flag": int(metas[i].get("adv_flag", -1)), "l2_meta": l2m})
                l2r = res[i].get("l2", float("nan"))
                row["l2_relerr"] = abs(l2r - l2m) / l2m if l2m and not math.isnan(l2m) else float("nan")
                row.update({k: (round(v, 7) if isinstance(v, float) else v) for k, v in res[i].items()})
                w.writerow(row)
            fh.flush()
            print(f"    {min(b + batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s")
    fh.close()
    print(f"[+] wrote {out_csv}")


def selftest(device: str) -> int:
    """Synthetic checks of the store round trip and of every metric's identity / direction (downloads weights)."""
    import tempfile

    import torch

    rng = np.random.default_rng(0)
    h = w = 224
    yy, xx = np.mgrid[0:h, 0:w] / h
    x0 = np.stack([0.5 + 0.3 * np.sin(6 * xx), 0.5 + 0.3 * np.cos(5 * yy), 0.4 + 0.2 * xx * yy]).astype(np.float32)
    shift = np.clip(x0 + 0.01, 0, 1)
    noise = np.clip(x0 + 0.01 * rng.standard_normal(x0.shape).astype(np.float32), 0, 1)
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for name, xa in (("same", x0), ("shift", shift), ("noise", noise)):
            store_adv(os.path.join(d, "adv_store"), "toy", f"img_{name}.png", "M", 1000, torch.from_numpy(xa)[None],
                      torch.from_numpy(x0)[None], state_query=999, adv_flag=1.0, mode="non_targeted")
        files = find_files([d])
        out = os.path.join(d, "m.csv")
        evaluate(files, out, FR_DEFAULT, [], device, 8)
        rows = {r["image"]: r for r in csv.DictReader(open(out))}
    same, shift_r, noise_r = rows["img_same.png"], rows["img_shift.png"], rows["img_noise.png"]
    checks = [
        ("identity l2 == 0", float(same["l2"]) < 1e-6),
        ("identity ssim ~ 1", float(same["ssim"]) > 0.9999),
        ("identity lpips_alex ~ 0", float(same["lpips_alex"]) < 1e-4),
        ("identity dists ~ 0", float(same["dists"]) < 1e-3),
        ("shift psnr ~ 40 dB", 39.5 < float(shift_r["psnr"]) < 40.5),
        ("float16 l2 relerr < 1e-3", float(noise_r["l2_relerr"]) < 1e-3),
        ("ssim: shift > noise (texture masking)", float(shift_r["ssim"]) > float(noise_r["ssim"])),
        ("lpips_alex: noise > shift", float(noise_r["lpips_alex"]) > float(shift_r["lpips_alex"])),
    ]
    for name, cond in checks:
        print(f"   {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)
    print("[+] SELFTEST PASS" if ok else "[!] SELFTEST FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--out", default="metrics_all.csv")
    ap.add_argument("--metrics", nargs="+", default=FR_DEFAULT)
    ap.add_argument("--nr", nargs="*", default=[], help="optional no-reference metrics: brisque clip_iqa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.device)
    if not a.roots:
        ap.error("--roots is required (result folders containing adv_store/)")
    evaluate(find_files(a.roots), a.out, a.metrics, a.nr, a.device, a.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
