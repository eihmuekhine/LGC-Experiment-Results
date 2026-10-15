#!/usr/bin/env python3
"""Read-only pre-flight report for a HUMAN reviewer of LGC-EGC result folders.

Written independently of the coding agent so that the human does not have to trust the
agent's own summaries. It never writes into the result folders.

Usage (from the worktree root):
    python tools/eval/human_preflight.py --roots Results/b4e_smoke_* --out reports/human_preflight_b4e

Outputs:
    <out>.md                      tables for sections 1-5 and 7 (read every FLAG line)
    <out>_grid_<arch>_<mode>.png  section 6: clean | adversarial per method | 0.5 + amp * delta

Sections:
    1 environment   GPU name, driver, hostname, torch, backend flags, ADV_STORE_BUDGETS, split sha256 per env file
                    (--require_host: FLAG roots that record no driver/hostname; use it on B5 roots)
    2 x0 identity   one x0.npy hash per (arch, image) across every root/method
    3 query budget  runs*.csv: queries vs budget, status counts, non-finite queries
    4 L2 recompute  ||delta|| from the npz vs pix_l2 in per_checkpoint and the npz l2 field
    5 curves        found rate and median pix_l2 per budget; flat best-so-far (stall) count
    6 visual grid   images a human must look at
    7 degeneracy    identical trajectories across methods, constant outputs, found rate 0, one x0 for many images
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
GPU_RE = re.compile(r"((?:NVIDIA\s+)?(?:GeForce\s+)?RTX[^\n\"]*|NVIDIA[^\n\"]*)")
TORCH_RE = re.compile(r"torch[\s=:\"]+(\d+\.\d+\.\d+\S*)")
ADV_RE = re.compile(r"ADV_STORE_BUDGETS[=\s\"':]+([^\s,\"'}]+)")
# host drift (LESSONS L03): every RTX 5090 host prints the same GPU name, so the GPU column alone cannot see a host change
DRIVER_RE = re.compile(r"driver(?:[ _]?version)?[\s\"':=|]+(\d{3}\.\d{1,3}(?:\.\d{1,3})?)", re.I)
HOST_RE = re.compile(r"(?<![A-Za-z_])hostname[\s\"':=]+([A-Za-z0-9][A-Za-z0-9._-]{2,})", re.I)


def stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mode_dirs(root: str) -> List[str]:
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and (glob.glob(os.path.join(d, "per_checkpoint*.csv")) or glob.glob(os.path.join(d, "runs*.csv"))):
            out.append(d)
    return out


# ---------------------------------------------------------------------------------------------- 1
CANON = (("matmul_tf32", re.compile(r"matmul[._]?allow[._]?tf32[\s\"':=]+(\w+)", re.I)),
         ("cudnn_tf32", re.compile(r"(?:cudnn[._]?allow[._]?tf32|(?<![\w.])allow_tf32|(?<![\w.])tf32)[\s\"':=]+(\w+)", re.I)),
         ("deterministic", re.compile(r"(?<![a-zA-Z])deterministic[\s\"':=]+(\w+)", re.I)),
         ("benchmark", re.compile(r"(?<![a-zA-Z])benchmark[\s\"':=]+(\w+)", re.I)))


def canon_flags(txt: str) -> Dict[str, str]:
    """Canonical backend flags found in an env/config text (JSON or plain)."""
    out = {}
    for key, rx in CANON:
        m = rx.search(txt)
        if m:
            out[key] = {"1": "True", "0": "False"}.get(m.group(1), m.group(1).capitalize())
    return out


def section_env(roots: List[str], require_host: bool = False) -> Tuple[List[str], int]:
    lines = ["## 1. Environment (one GPU name, one torch, one driver, one host, identical backend flags, "
             "ADV_STORE_BUDGETS unset)", "",
             "| root | env file | GPU | driver | host | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |",
             "|---|---|---|---|---|---|---|---|---|"]
    gpus, torches, drivers, hosts = Counter(), Counter(), Counter(), Counter()
    flag_values: Dict[str, Counter] = defaultdict(Counter)
    flags_out, n_files = 0, 0
    for root in roots:
        files = sorted(set(glob.glob(os.path.join(root, "env_frame.txt")) + glob.glob(os.path.join(root, "*", "env_frame.txt"))
                           + glob.glob(os.path.join(root, "*", "env_*.json")) + glob.glob(os.path.join(root, "env_*.json"))
                           + glob.glob(os.path.join(root, "*", "config_*.json"))))
        root_flags: Dict[str, str] = {}
        root_has_host = False
        for f in files:
            n_files += 1
            txt = open(f, encoding="utf-8", errors="replace").read()
            gpu, torch_v, adv, shas, drv, host = "?", "?", "MISSING", [], "?", "?"
            if f.endswith(".json"):
                try:
                    d = json.loads(txt)
                except json.JSONDecodeError:
                    d = {}
                if not isinstance(d, dict):
                    d = {}
                gpu = str(d.get("gpu") or d.get("gpu_name") or "?")
                torch_v = str(d.get("torch") or "?")
                drv = str(d.get("driver") or d.get("driver_version") or d.get("nvidia_driver") or "?")
                host = str(d.get("hostname") or d.get("host") or "?")
                if "ADV_STORE_BUDGETS" in d:
                    adv = str(d["ADV_STORE_BUDGETS"])
                if d.get("split_csv_sha256"):
                    shas.append(str(d["split_csv_sha256"]))
            m = GPU_RE.search(txt) if gpu == "?" else None
            if m:
                gpu = m.group(1).strip()
            m = TORCH_RE.search(txt) if torch_v == "?" else None
            if m:
                torch_v = m.group(1)
            m = ADV_RE.search(txt) if adv == "MISSING" else None
            if m:
                adv = m.group(1)
            m = DRIVER_RE.search(txt) if drv == "?" else None
            if m:
                drv = m.group(1)
            m = HOST_RE.search(txt) if host == "?" else None
            if m:
                host = m.group(1)
            if drv != "?":
                drivers[drv] += 1
            if host != "?":
                hosts[host] += 1
            root_has_host = root_has_host or (drv != "?" and host != "?")
            fl = canon_flags(txt)
            for k, v in fl.items():
                root_flags.setdefault(k, v)
                flag_values[k][v] += 1
            shas += [h for h in HEX64.findall(txt) if h not in shas]
            if gpu != "?":
                gpus[gpu] += 1
            if torch_v != "?":
                torches[torch_v] += 1
            flag_str = "; ".join(f"{k}={v}" for k, v in sorted(fl.items())) or "-"
            lines.append(f"| {os.path.basename(root)} | {os.path.relpath(f, root)} | {gpu} | {drv} | {host} | {torch_v} | "
                         f"{flag_str} | {adv} | {', '.join(s[:12] for s in shas[:3])} |")
        missing = [k for k in ("cudnn_tf32", "deterministic") if k not in root_flags]
        if files and missing:
            lines.append(f"| {os.path.basename(root)} | **FLAG** | | | | | not recorded in any env/config file: {missing} | | |")
            flags_out += 1
        if files and require_host and not root_has_host:
            lines.append(f"| {os.path.basename(root)} | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |")
            flags_out += 1
    lines.append("")
    for label, counter in (("GPU names", gpus), ("torch versions", torches), ("drivers", drivers), ("hosts", hosts)):
        lines.append(f"- {label}: {dict(counter)}")
        if len(counter) > 1:
            lines.append(f"  - **FLAG: more than one value across env files** (drivers/hosts: expected only if roots from "
                         f"earlier hosts are included on purpose; B5 roots must show one value)")
            flags_out += 1
    if not drivers and not hosts:
        lines.append("- note: no driver or hostname recorded in any env file; host drift (L03) cannot be checked from "
                     "these folders, use `nvidia-smi` and the B5 runner host log")
    for k, c in sorted(flag_values.items()):
        lines.append(f"- backend {k}: {dict(c)}")
        if len(c) > 1:
            lines.append(f"  - **FLAG: {k} differs across env files (D12 parity broken)**")
            flags_out += 1
    lines.append(f"- env files read: {n_files}")
    return lines + [""], flags_out


# ---------------------------------------------------------------------------------------------- 2
def section_x0(roots: List[str]) -> Tuple[List[str], int]:
    lines = ["## 2. x0 identity (every method must attack the same clean tensor)", ""]
    hashes: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for root in roots:
        for p in glob.glob(os.path.join(root, "*", "adv_store", "*", "*", "x0.npy")):
            parts = p.split(os.sep)
            arch, image = parts[-3], parts[-2]
            hashes[(arch, image)][sha256_file(p)].append(os.path.relpath(p))
    bad = {k: v for k, v in hashes.items() if len(v) > 1}
    lines.append(f"- (arch, image) keys: {len(hashes)}; with more than one x0 hash: **{len(bad)}**")
    for (arch, image), v in list(bad.items())[:10]:
        lines.append(f"  - FLAG {arch}/{image}: " + " | ".join(f"{h[:12]}: {paths[0]}" for h, paths in v.items()))
    single = sum(1 for v in hashes.values() if sum(len(x) for x in v.values()) == 1)
    lines.append(f"- keys seen in only one root (cannot be cross-checked): {single}")
    return lines + [""], len(bad)


# ---------------------------------------------------------------------------------------------- 3
def read_csvs(pattern_dir: str, prefix: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(pattern_dir, f"{prefix}*.csv"))):
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if "method" not in df.columns:
            m = re.match(rf"{prefix}_?(.*)\.csv$", os.path.basename(f))
            df["method"] = m.group(1) if m and m.group(1) else "?"
        df["_file"] = f
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def section_budget(roots: List[str], budget: int) -> Tuple[List[str], int]:
    lines = ["## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)", "",
             "| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |",
             "|---|---|---|---|---|---|---|---|"]
    flags_out = 0
    for root in roots:
        for md in mode_dirs(root):
            runs = read_csvs(md, "runs")
            if runs.empty:
                continue
            qcol = "queries" if "queries" in runs.columns else ("total_queries" if "total_queries" in runs.columns else None)
            for meth, g in runs.groupby("method"):
                status = dict(Counter(g["status"].astype(str))) if "status" in g.columns else {}
                if qcol:
                    q = pd.to_numeric(g[qcol], errors="coerce")
                    eq, lo, hi = int((q == budget).sum()), int((q < budget).sum()), int((q > budget).sum())
                else:
                    eq = lo = hi = -1
                nf = int(pd.to_numeric(g["nonfinite_queries"], errors="coerce").fillna(0).sum()) if "nonfinite_queries" in g.columns else "n/a"
                lines.append(f"| {os.path.basename(root)}/{os.path.basename(md)} | {meth} | {len(g)} | {status} | {eq} | {lo} | {hi} | {nf} |")
                if hi > 0 or (isinstance(nf, int) and nf > 0) or "error" in status:
                    lines.append(f"|  | **FLAG** | | over-budget, non-finite or error rows | | | | |")
                    flags_out += 1
    lines += ["", "Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash."]
    return lines + [""], flags_out


# ---------------------------------------------------------------------------------------------- 4
def section_l2(roots: List[str], samples: int, seed: int) -> Tuple[List[str], int]:
    lines = ["## 4. L2 recomputation from stored images (relative error must be < 1e-3)", "",
             "| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |", "|---|---|---|---|---|---|---|---|---|"]
    rng = random.Random(seed)
    flags_out, done = 0, 0
    for root in roots:
        for md in mode_dirs(root):
            ck = read_csvs(md, "per_checkpoint")
            if ck.empty or "found" not in ck.columns:
                continue
            rows = ck[pd.to_numeric(ck["found"], errors="coerce") == 1]
            idx = list(rows.index)
            rng.shuffle(idx)
            for i in idx[:samples]:
                r = rows.loc[i]
                arch = str(r.get("arch", "")) if "arch" in rows.columns else ""
                cands = glob.glob(os.path.join(md, "adv_store", arch or "*", stem(r["image"]), f"{r['method']}_Q{int(r['budget'])}.npz"))
                if not cands:
                    lines.append(f"| {os.path.basename(root)}/{os.path.basename(md)} | {r['method']} | {arch} | {stem(r['image'])} | {int(r['budget'])} | {r['pix_l2']} | NPZ MISSING | | **FLAG** |")
                    flags_out += 1
                    continue
                z = np.load(cands[0], allow_pickle=False)
                dn = float(np.linalg.norm(z["delta"].astype(np.float64).ravel()))
                l2n = float(z["l2"]) if "l2" in z.files else float("nan")
                ref = float(r["pix_l2"])
                rel = max(abs(dn - ref), abs(l2n - ref)) / max(ref, 1e-12)
                mark = "**FLAG**" if not rel < 1e-3 else f"{rel:.1e}"
                flags_out += int(not rel < 1e-3)
                done += 1
                lines.append(f"| {os.path.basename(root)}/{os.path.basename(md)} | {r['method']} | {arch or os.path.basename(os.path.dirname(os.path.dirname(cands[0])))} | "
                             f"{stem(r['image'])} | {int(r['budget'])} | {ref:.5f} | {dn:.5f} | {l2n:.5f} | {mark} |")
    lines.append(f"\n- rows checked: {done}")
    return lines + [""], flags_out


# ---------------------------------------------------------------------------------------------- 5
def section_curves(roots: List[str]) -> Tuple[List[str], int]:
    lines = ["## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)", ""]
    flags_out = 0
    for root in roots:
        for md in mode_dirs(root):
            ck = read_csvs(md, "per_checkpoint")
            if ck.empty or not {"budget", "found", "pix_l2"} <= set(ck.columns):
                continue
            ck["budget"] = pd.to_numeric(ck["budget"], errors="coerce")
            ck["found"] = pd.to_numeric(ck["found"], errors="coerce")
            ck["pix_l2"] = pd.to_numeric(ck["pix_l2"], errors="coerce")
            ck["img"] = ck["image"].map(stem)
            for meth, g in ck.groupby("method"):
                budgets = sorted(g["budget"].dropna().unique())
                cells = []
                for b in budgets:
                    gb = g[g["budget"] == b]
                    fr = gb["found"].mean()
                    med = gb.loc[gb["found"] == 1, "pix_l2"].median()
                    cells.append(f"{int(b)}: {fr:.2f} / {med:.3g}")
                piv = g[g["found"] == 1].pivot_table(index="img", columns="budget", values="pix_l2", aggfunc="first")
                flat = 0
                if 2000 in piv.columns and 10000 in piv.columns:
                    both = piv[[2000, 10000]].dropna()
                    flat = int((both[2000] == both[10000]).sum())
                nonmono = 0
                if piv.shape[1] > 1:
                    arr = piv.to_numpy(dtype=float)
                    diffs = np.diff(arr, axis=1)
                    nonmono = int(np.nansum(diffs > 1e-9 * np.nan_to_num(arr[:, :-1], nan=1.0), axis=1).astype(bool).sum())
                lines.append(f"- **{os.path.basename(root)}/{os.path.basename(md)} {meth}** (images {piv.shape[0]}): found rate / median pix_l2 per budget: "
                             + "; ".join(cells))
                lines.append(f"  - flat from 2000 to 10000 (possible stall): {flat}; images with an L2 increase between budgets: {nonmono}"
                             + ("  **FLAG**" if flat or nonmono else ""))
                flags_out += int(bool(flat or nonmono))
    lines.append("\nAn L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. "
                 "Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.")
    return lines + [""], flags_out


# ---------------------------------------------------------------------------------------------- 6
def section_grid(roots: List[str], out: str, n_images: int, budget: int, amp: float) -> List[str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return ["## 6. Visual grid", "", "PIL not installed; skipped.", ""]
    lines = ["## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)", ""]
    store: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    x0s: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for root in roots:
        for md in mode_dirs(root):
            mode = os.path.basename(md)
            for p in glob.glob(os.path.join(md, "adv_store", "*", "*", f"*_Q{budget}.npz")):
                parts = p.split(os.sep)
                arch, image = parts[-3], parts[-2]
                meth = os.path.basename(p)[: -len(f"_Q{budget}.npz")]
                store[(arch, mode)][image][meth] = p
                x0 = os.path.join(os.path.dirname(p), "x0.npy")
                if os.path.exists(x0):
                    x0s[(arch, mode)][image] = x0
    tile = 160
    for (arch, mode), imgs in sorted(store.items()):
        methods = sorted({m for v in imgs.values() for m in v})
        names = sorted(imgs)[:n_images]
        if not names:
            continue
        canvas = Image.new("RGB", (tile * (1 + 2 * len(methods)), (tile + 14) * len(names) + 14), "white")
        draw = ImageDraw.Draw(canvas)
        heads = ["clean"] + [f"{m}" for m in methods] + [f"d x{amp:g} {m}" for m in methods]
        for j, h in enumerate(heads):
            draw.text((j * tile + 2, 0), h[:24], fill="black")

        def to_img(a: np.ndarray):
            a = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)
            return Image.fromarray((a.transpose(1, 2, 0) * 255).round().astype(np.uint8)).resize((tile, tile))

        for i, name in enumerate(names):
            y = 14 + i * (tile + 14)
            x0p = x0s[(arch, mode)].get(name)
            if not x0p:
                continue
            x0 = np.load(x0p).astype(np.float32)
            canvas.paste(to_img(x0), (0, y))
            for j, m in enumerate(methods):
                p = imgs[name].get(m)
                if not p:
                    continue
                z = np.load(p, allow_pickle=False)
                d = z["delta"].astype(np.float32)
                canvas.paste(to_img(x0 + d), ((1 + j) * tile, y))
                canvas.paste(to_img(0.5 + amp * d), ((1 + len(methods) + j) * tile, y))
                l2 = float(np.linalg.norm(d.ravel()))
                nan_note = " NaN!" if not np.isfinite(d).all() else ""
                draw.text(((1 + j) * tile + 2, y + tile), f"L2={l2:.2f}{nan_note}", fill="red" if nan_note else "black")
            draw.text((2, y + tile), name[-22:], fill="black")
        path = f"{out}_grid_{arch}_{mode}.png"
        canvas.save(path)
        lines.append(f"- {path}: {len(names)} images x {len(methods)} methods at budget {budget}")
    return lines + [""]


# ---------------------------------------------------------------------------------------------- 7
def section_degeneracy(roots: List[str], expect_identical: List[Tuple[str, str]]) -> Tuple[List[str], int]:
    """Silent-collapse checks (cf. AutoResearchClaw DEGENERATE_METRICS / its T10 case study).

    (a) two different methods with bit-identical pix_l2 trajectories on the same image (wiring or copy bug),
        except declared pairs such as CGBA vs CGBA_D8only, which are identical by design under D12;
    (b) one method with zero spread of pix_l2 across >= 3 images at its last budget (collapse to a constant);
    (c) one method that never finds an adversarial example at its last budget (found rate 0);
    (d) two different images with the same x0 hash (loader feeds one image for many rows).
    """
    lines = ["## 7. Degeneracy (silent collapse: identical methods, constant outputs, never found, duplicated inputs)", ""]
    allowed = {frozenset(p) for p in expect_identical}
    flags_out = 0
    frames = []
    for root in roots:
        for md in mode_dirs(root):
            ck = read_csvs(md, "per_checkpoint")
            if ck.empty or not {"budget", "found", "pix_l2", "image"} <= set(ck.columns):
                continue
            ck = ck.copy()
            ck["mode_dir"] = os.path.basename(md)
            ck["arch_"] = ck["arch"].astype(str) if "arch" in ck.columns else "?"
            frames.append(ck)
    if not frames:
        return lines + ["- no per_checkpoint files", ""], 0
    ck = pd.concat(frames, ignore_index=True)
    for c in ("budget", "found", "pix_l2"):
        ck[c] = pd.to_numeric(ck[c], errors="coerce")
    ck["img"] = ck["image"].map(stem)
    for (arch, mode), g in ck.groupby(["arch_", "mode_dir"]):
        traj: Dict[str, Dict[str, tuple]] = defaultdict(dict)
        for (meth, img), h in g.groupby(["method", "img"]):
            h = h.sort_values("budget")
            traj[meth][img] = tuple(zip(h["budget"].tolist(), h["found"].tolist(), h["pix_l2"].round(12).tolist()))
        meths = sorted(traj)
        for i, m1 in enumerate(meths):
            for m2 in meths[i + 1:]:
                common = set(traj[m1]) & set(traj[m2])
                same = [im for im in common if traj[m1][im] == traj[m2][im] and any(f == 1 for _, f, _ in traj[m1][im])]
                if same:
                    ok = frozenset((m1, m2)) in allowed
                    lines.append(f"- {arch}/{mode}: {m1} and {m2} identical on {len(same)}/{len(common)} images"
                                 + ("  (declared identical by design)" if ok else "  **FLAG**"))
                    flags_out += int(not ok)
        for meth, h in g.groupby("method"):
            bmax = h["budget"].max()
            last = h[h["budget"] == bmax]
            found = last[last["found"] == 1]["pix_l2"].dropna()
            if len(last) >= 3 and len(found) == 0:
                lines.append(f"- {arch}/{mode} {meth}: found rate 0 at budget {int(bmax)} on {len(last)} images  **FLAG**")
                flags_out += 1
            elif len(found) >= 3 and float(found.std(ddof=0)) == 0.0:
                lines.append(f"- {arch}/{mode} {meth}: pix_l2 identical on all {len(found)} images at budget {int(bmax)}  **FLAG**")
                flags_out += 1
    xh: Dict[Tuple[str, str], set] = defaultdict(set)
    for root in roots:
        for p in glob.glob(os.path.join(root, "*", "adv_store", "*", "*", "x0.npy")):
            parts = p.split(os.sep)
            xh[(parts[-3], sha256_file(p))].add(parts[-2])
    dup = {k: v for k, v in xh.items() if len(v) > 1}
    for (arch, h), imgs in sorted(dup.items()):
        lines.append(f"- {arch}: one x0 tensor ({h[:12]}) stored for {len(imgs)} different images {sorted(imgs)[:4]}  **FLAG**")
    flags_out += len(dup)
    if flags_out == 0:
        lines.append("- no degenerate pattern found")
    return lines + [""], flags_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. reports/human_preflight_b4e")
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--l2_samples", type=int, default=10, help="rows per root/mode for section 4")
    ap.add_argument("--grid_images", type=int, default=3)
    ap.add_argument("--grid_budget", type=int, default=10000)
    ap.add_argument("--amp", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect_identical", nargs="*", default=["CGBA:CGBA_D8only"],
                    help="method pairs that are identical by design (A:B); others identical => FLAG")
    ap.add_argument("--require_host", action="store_true",
                    help="FLAG every root whose env files record no driver + hostname (use for B5 roots)")
    a = ap.parse_args()
    pairs = [tuple(x.split(":", 1)) for x in a.expect_identical if ":" in x]
    roots = [r for r in a.roots if os.path.isdir(r)]
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    for r in roots:
        if os.path.abspath(out_dir).startswith(os.path.abspath(r)):
            raise SystemExit(f"[!] --out must not be inside a result root ({r})")
    report, total = [f"# Human pre-flight report", "", f"roots: {roots}", ""], 0
    for fn in (lambda: section_env(roots, a.require_host), lambda: section_x0(roots), lambda: section_budget(roots, a.budget),
               lambda: section_l2(roots, a.l2_samples, a.seed), lambda: section_curves(roots),
               lambda: section_degeneracy(roots, pairs)):
        lines, n = fn()
        report += lines
        total += n
    report += section_grid(roots, a.out, a.grid_images, a.grid_budget, a.amp)
    report.insert(3, f"**FLAG lines: {total}** (0 is required before a GPU run; every FLAG needs a written explanation)\n")
    with open(a.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"[*] wrote {a.out}.md ; FLAG lines = {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
