#!/usr/bin/env python3
"""Domain consistency of the face store with the rule of the main study (L, 2026-11-01).

For every stored state (method, task, pair, budget) of <face_root>/store_main: d = x - x_src (float32 as stored);
the victim decides on clamp(x_src + s*d, 0, 1) for s in {1, 1 + 2^-9, 1 + 2^-6}. Targeted margin
m = max_{j != t} l_j - l_t (negative: adversarial). Class C: adversarial at s = 1; Q9: not at 1 but at 1 + 2^-9;
Q6: first at 1 + 2^-6; X: at none (the main study's classes, claude/b4h_amendment_and_review_2026-10-15.md).
Also the decision on the 8-bit image (round half to even). Evaluation only: no query is counted, no store is changed.

out: <out>/domain_rows.csv, domain_summary.json, domain_summary.md
usage (experiment environment, kit dir on sys.path):
  python face_domain.py --face_root /root/autodl-tmp/face --kit /root/autodl-tmp/face/kit --out <dir> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

SCALES = (1.0, 1.0 + 2.0 ** -9, 1.0 + 2.0 ** -6)
METHODS = ("LGC_RC_EGC", "LatentHSJA", "HSJA")


def safe(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)


def find_npz(store: str, arch: str, image: str, method: str, budget: int) -> str:
    stem = safe(os.path.splitext(image)[0])
    p = os.path.join(store, arch, stem, f"{method}_Q{budget}.npz")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(store, "**", stem, f"{method}_Q{budget}.npz"), recursive=True)
    if len(hits) != 1:
        raise SystemExit(f"[!] {method} {image} Q{budget}: {len(hits)} npz files under {store}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_root", default="/root/autodl-tmp/face")
    ap.add_argument("--kit", default="/root/autodl-tmp/face/kit")
    ap.add_argument("--kind", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    import face_common as fc  # noqa: E402
    sys.path.insert(0, os.path.realpath(fc.FROZEN))
    import numpy as np
    import torch
    from advstore import load_adv  # frozen worktree

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    inp = fc.load_inputs(a.face_root)
    root = os.path.join(a.face_root, f"store_{a.kind}")
    store, index = os.path.join(root, "adv_store"), os.path.join(root, "store_index.csv")
    rows_in = list(csv.DictReader(open(index)))
    os.makedirs(a.out, exist_ok=True)
    out_rows, summary = [], {}
    for task in sorted(fc.TASKS):
        w = inp["weights"][task]
        clf = fc.build_face_victim(w["path"], fc.TASKS[task], dev, expected_sha=w["sha256"])
        img_dir = os.path.join(inp["data"][task]["flat"], "images")
        pairs = fc.read_pairs(inp["pairs"][task][a.kind]["path"])
        tgt = {s: int(tl) for s, _, _, tl in pairs}
        arch = fc.ARCH[task]
        cache = {}
        for r in [r for r in rows_in if r["task"] == task]:
            s_name, m, b = r["image"], r["method"], int(r["budget"])
            if s_name not in cache:
                cache[s_name] = fc.load_face_image(os.path.join(img_dir, s_name))[0].numpy().astype(np.float32)
            x0 = cache[s_name]
            x = load_adv(find_npz(store, arch, s_name, m, b))[0].astype(np.float32)
            d = x - x0
            t = tgt[s_name]
            batch = [np.clip(x0 + np.float32(s) * d, 0.0, 1.0) for s in SCALES]
            batch.append((np.clip(np.round(x.astype(np.float64) * 255.0), 0, 255) / 255.0).astype(np.float32))
            with torch.no_grad():
                lg = clf(torch.from_numpy(np.stack(batch)).to(dev)).double().cpu().numpy()
            other = np.delete(lg, t, axis=1).max(axis=1)
            marg = other - lg[:, t]
            adv = [int(lg[i].argmax() == t) for i in range(4)]
            cls = "C" if adv[0] else "Q9" if adv[1] else "Q6" if adv[2] else "X"
            out_rows.append([task, m, s_name, b, r["held_start"], r["adv_flag"], cls, adv[3],
                             "%.6g" % marg[0], "%.6g" % marg[1], "%.6g" % marg[2]])
            k = summary.setdefault(task, {}).setdefault(m, {"rows": 0, "C": 0, "Q9": 0, "Q6": 0, "X": 0,
                                                             "uint8_adv": 0, "max_margin_nonC": None})
            k["rows"] += 1
            k[cls] += 1
            k["uint8_adv"] += adv[3]
            if cls != "C":
                k["max_margin_nonC"] = max(k["max_margin_nonC"] or -1e9, float(marg[0]))
        del clf
    with open(os.path.join(a.out, "domain_rows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["task", "method", "image", "budget", "held_start", "adv_flag", "class", "adv_uint8",
                     "m_s1", "m_s9", "m_s6"])
        wr.writerows(out_rows)
    json.dump({"scales": list(SCALES), "summary": summary}, open(os.path.join(a.out, "domain_summary.json"), "w"),
              indent=1)
    lines = ["# Face store: domain consistency (classes of the main study; evaluation only)", "",
             "| task | method | rows | C | Q9 | Q6 | X | 8-bit adversarial | max margin of non-C rows |", "|---|---|---|---|---|---|---|---|---|"]
    for task in sorted(summary):
        for m in METHODS:
            k = summary[task].get(m)
            if k:
                lines.append("| %s | %s | %d | %d | %d | %d | %d | %d | %s |" % (
                    task, m, k["rows"], k["C"], k["Q9"], k["Q6"], k["X"], k["uint8_adv"],
                    "-" if k["max_margin_nonC"] is None else "%.3g" % k["max_margin_nonC"]))
    open(os.path.join(a.out, "domain_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
