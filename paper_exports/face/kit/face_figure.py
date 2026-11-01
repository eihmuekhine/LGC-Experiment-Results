#!/usr/bin/env python3
"""Rule-based face comparison images for the paper (L, 2026-11-01).

Rule (fixed before looking at any image): for each task, the first pair in the order of the pairs file whose stored
state at budget B is of class C (adversarial when re-evaluated, face_domain.py) for all three methods. For that pair:
source x_src, target x_trg, and per method the stored image x and the perturbation shown as 0.5 + k (x - x_src),
with one factor k per row: k = 0.5 / (largest 99.5th percentile of |x - x_src| over the three methods).
Writes 8-bit PNG cells <out>/<task>_<col>.png and <out>/selection.json; the paper sets them in a LaTeX tabular.

usage: python face_figure.py --face_root /root/autodl-tmp/face --kit /root/autodl-tmp/face/kit \
         --domain <dir with domain_rows.csv> --budget 1000 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

COLS = (("LGC_RC_EGC", "egc"), ("LatentHSJA", "lhsja"), ("HSJA", "hsja"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_root", default="/root/autodl-tmp/face")
    ap.add_argument("--kit", default="/root/autodl-tmp/face/kit")
    ap.add_argument("--kind", default="main")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    import face_common as fc  # noqa: E402
    sys.path.insert(0, os.path.realpath(fc.FROZEN))
    import numpy as np
    from PIL import Image
    from advstore import load_adv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_domain import find_npz

    cls = {}
    for r in csv.DictReader(open(os.path.join(a.domain, "domain_rows.csv"))):
        cls[(r["task"], r["method"], r["image"], int(r["budget"]))] = r["class"]
    inp = fc.load_inputs(a.face_root)
    store = os.path.join(a.face_root, f"store_{a.kind}", "adv_store")
    os.makedirs(a.out, exist_ok=True)

    def save(x, path):
        Image.fromarray((np.clip(np.round(x * 255.0), 0, 255)).astype(np.uint8).transpose(1, 2, 0)).save(path)

    sel = {"rule": __doc__.split("\n\n")[1].strip(), "budget": a.budget, "rows": {}}
    for task in sorted(fc.TASKS):
        pairs = fc.read_pairs(inp["pairs"][task][a.kind]["path"])
        img_dir = os.path.join(inp["data"][task]["flat"], "images")
        pick = None
        for k, (s_name, s_lbl, t_name, t_lbl) in enumerate(pairs):
            if all(cls.get((task, m, s_name, a.budget)) == "C" for m, _ in COLS):
                pick = (k, s_name, s_lbl, t_name, t_lbl)
                break
        if pick is None:
            raise SystemExit(f"[!] {task}: no pair with class C for all methods at {a.budget}")
        k, s_name, s_lbl, t_name, t_lbl = pick
        x0 = fc.load_face_image(os.path.join(img_dir, s_name))[0].numpy().astype(np.float32)
        xt = fc.load_face_image(os.path.join(img_dir, t_name))[0].numpy().astype(np.float32)
        xs = {tag: load_adv(find_npz(store, fc.ARCH[task], s_name, m, a.budget))[0].astype(np.float32)
              for m, tag in COLS}
        amp = max(float(np.percentile(np.abs(x - x0), 99.5)) for x in xs.values())
        kf = 0.5 / max(amp, 1e-8)
        save(x0, os.path.join(a.out, f"{task}_src.png"))
        save(xt, os.path.join(a.out, f"{task}_tgt.png"))
        for tag, x in xs.items():
            save(x, os.path.join(a.out, f"{task}_{tag}.png"))
            save(np.clip(0.5 + kf * (x - x0), 0.0, 1.0), os.path.join(a.out, f"{task}_{tag}_p.png"))
        sel["rows"][task] = {"pair_index": k, "source": s_name, "source_label": int(s_lbl), "target": t_name,
                             "target_label": int(t_lbl), "k": kf, "pairs_skipped": k}
        print(f"[fig] {task}: pair {k} {s_name} -> {t_name}; k = {kf:.3g}")
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
