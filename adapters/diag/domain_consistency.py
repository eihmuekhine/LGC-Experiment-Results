#!/usr/bin/env python3
"""B4h T1 — domain-consistency measurement of stored adversarial images (evaluation-only, uncounted).

For every (method, cell, image, contract checkpoint) with ``found == 1`` in the per_checkpoint CSV of each root, this
reads the stored float16 delta and clean x0 from the adv_store and, for ``s`` in ``{1, 1 + 2^-9, 1 + 2^-6}``, evaluates
the victim on ``clamp(x0 + s * float32(delta), 0, 1)``.  The victim is the raw shared victim (``C.load_victim_any``)
under the D12 flags (``_common.configure_precision`` runs at import); the clamp is applied HERE, before the victim, so
the measurement is independent of any adapter clamp.

Per row the decision and margin are recorded:

  * adversarial: NT ``argmax != y``; T ``argmax == t``;
  * margin m: NT ``l_y - max_{j!=y} l_j``; T ``max_{j!=t} l_j - l_t``;
  * class: ``C`` if adversarial at s=1; ``Q9`` if not at s=1 but adversarial at s=1+2^-9; ``Q6`` if the first
    adversarial point is s=1+2^-6; ``X`` otherwise.

Q9/Q6 are float16-storage effects (||fl16(d) - d|| <= 2^-11 ||d||), not defects.  After the B4h-T3 D19 fix,
``X > 0`` for ANY method (ARS_OPT included) is a STOP condition (E-OTHER): every method must now decide on the
exact clamped [0,1] image that is stored and scored.

Usage (from the worktree root, GPU, D12 flags):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python adapters/diag/domain_consistency.py \
        --roots $(cat reports/B4e2_smoke_roots.txt) --out reports/raw/B4h/t1_rows.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "adapters"))
sys.path.insert(0, REPO_ROOT)

import _common as C  # noqa: E402  (configure_precision runs at import -> D12 flags)
import torch  # noqa: E402

SCALES = (("1", 1.0), ("1+2^-9", 1.0 + 2.0 ** -9), ("1+2^-6", 1.0 + 2.0 ** -6))

CSV_FIELDS = ["method", "arch", "mode", "image", "budget", "source_label", "target_label",
              "class", "m_s1", "m_s9", "m_s6", "adv_s1", "adv_s9", "adv_s6", "adv_uint8"]


def mode_dirs(root: str):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and (glob.glob(os.path.join(d, "per_checkpoint*.csv"))
                                 or glob.glob(os.path.join(d, "runs*.csv"))):
            out.append(d)
    return out


def read_ckpt(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def advstore_path(md: str, arch: str, stem: str, method: str, budget: int):
    return os.path.join(md, "adv_store", str(arch), stem, f"{method}_Q{int(budget)}.npz")


def x0_path(md: str, arch: str, stem: str, method: str):
    d = os.path.join(md, "adv_store", str(arch), stem)
    alt = os.path.join(d, f"x0_{method}.npy")
    return alt if os.path.exists(alt) else os.path.join(d, "x0.npy")


def margins(logits, y, t):
    l = logits[0]
    if t is None:
        idx = [j for j in range(l.numel()) if j != y]
        return (l[y] - l[idx].max()).item()
    idx = [j for j in range(l.numel()) if j != t]
    return (l[idx].max() - l[t]).item()


def evaluate_one(victim, x0_t, delta_t, y, t):
    out = []
    for _name, s in SCALES:
        x = torch.clamp(x0_t + s * delta_t, 0.0, 1.0)
        with torch.no_grad():
            logits = victim(x)
        lab = int(logits.argmax(1).item())
        adv = (lab == t) if t is not None else (lab != y)
        m = margins(logits, y, t)
        out.append((adv, m))
    return out


def classify(adv):
    adv_s1, adv_s9, adv_s6 = adv
    if adv_s1:
        return "C"
    if adv_s9:
        return "Q9"
    if adv_s6:
        return "Q6"
    return "X"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[!] no CUDA device; this measurement must run on the study GPU")
        return 2
    print(f"[*] GPU {torch.cuda.get_device_name(0)}  D12 matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} deterministic={torch.backends.cudnn.deterministic} "
          f"benchmark={torch.backends.cudnn.benchmark}")

    victims = {}
    roots = [r for r in a.roots if os.path.isdir(r)]
    rows = []
    for root in roots:
        for md in mode_dirs(root):
            mode = os.path.basename(md)
            for ck_path in sorted(glob.glob(os.path.join(md, "per_checkpoint*.csv"))):
                ck = read_ckpt(ck_path)
                for r in ck:
                    if str(r.get("found", "0")) != "1":
                        continue
                    method = r["method"]
                    arch = r["arch"]
                    image = r["image"]
                    stem = C.stem_of(image)
                    budget = int(r["budget"])
                    y = int(r["source_label"])
                    t_raw = (r.get("target_label") or "").strip()
                    t = int(t_raw) if t_raw not in ("", None) else None
                    adv_uint8 = r.get("adv_uint8", "-1")

                    npz = advstore_path(md, arch, stem, method, budget)
                    if not os.path.exists(npz):
                        print(f"[!] missing npz: {npz}", flush=True)
                        continue
                    x0p = x0_path(md, arch, stem, method)
                    if not os.path.exists(x0p):
                        print(f"[!] missing x0: {x0p}", flush=True)
                        continue

                    if arch not in victims:
                        victims[arch] = C.load_victim_any(arch, device)
                    victim = victims[arch]

                    z = np.load(npz, allow_pickle=False)
                    delta = z["delta"].astype(np.float32)          # (3,224,224)
                    x0 = np.load(x0p).astype(np.float32)           # (3,224,224)
                    x0_t = torch.from_numpy(x0).unsqueeze(0).to(device)
                    delta_t = torch.from_numpy(delta).unsqueeze(0).to(device)

                    res = evaluate_one(victim, x0_t, delta_t, y, t)
                    adv = [r_[0] for r_ in res]
                    ms = [r_[1] for r_ in res]
                    cls = classify(adv)
                    rows.append({"method": method, "arch": arch, "mode": mode, "image": stem, "budget": budget,
                                 "source_label": y, "target_label": "" if t is None else t, "class": cls,
                                 "m_s1": ms[0], "m_s9": ms[1], "m_s6": ms[2],
                                 "adv_s1": int(adv[0]), "adv_s9": int(adv[1]), "adv_s6": int(adv[2]),
                                 "adv_uint8": adv_uint8})

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- aggregate table per (method, cell) ----
    from collections import OrderedDict

    table = OrderedDict()
    for r in rows:
        key = (r["method"], r["arch"], r["mode"])
        table.setdefault(key, {"C": 0, "Q9": 0, "Q6": 0, "X": 0, "m_s1_nonC": [], "adv_uint8_ok": 0, "n": 0})
        e = table[key]
        e["n"] += 1
        e[r["class"]] += 1
        if r["class"] != "C":
            e["m_s1_nonC"].append(r["m_s1"])
        if str(r["adv_uint8"]) == "1":
            e["adv_uint8_ok"] += 1

    print("\n| method | cell (arch/mode) | rows | C | Q9 | Q6 | X | median m(s=1) non-C | max m(s=1) non-C | adv_uint8==1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    any_X = False
    for (method, arch, mode), e in table.items():
        med = float(np.median(e["m_s1_nonC"])) if e["m_s1_nonC"] else float("nan")
        mx = float(np.max(e["m_s1_nonC"])) if e["m_s1_nonC"] else float("nan")
        print(f"| {method} | {arch}/{mode} | {e['n']} | {e['C']} | {e['Q9']} | {e['Q6']} | {e['X']} | "
              f"{med:.3g} | {mx:.3g} | {e['adv_uint8_ok']} |")
        if e["X"] > 0:
            any_X = True

    print(f"\n[*] wrote {a.out} ({len(rows)} rows)")
    if any_X:
        print("[!] E-OTHER: X > 0 for a method (post-fix: every method must decide on the stored clamped image) -> STOP")
        return 3
    print("[+] no X>0 for any method (ARS_OPT and CGBA/CGBA_H both clamp correctly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
