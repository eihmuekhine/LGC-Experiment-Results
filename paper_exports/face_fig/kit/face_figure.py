#!/usr/bin/env python3
"""Rule-based face comparison images for the paper (L, 2026-11-01; budget sweep added 2026-11-03).

Rule (fixed before looking at any image, unchanged): for each task, the first pair in the order of the pairs file
whose stored state at the reference budget --budget is of class C (adversarial when re-evaluated, face_domain.py)
for all three methods. For that pair: source x_src, target x_trg, and per method and per budget in --budgets the
stored image x, plus the perturbation of the reference budget shown as 0.5 + k (x - x_src), with one factor k per
row: k = 0.5 / (largest 99.5th percentile of |x - x_src| over the three methods at the reference budget).

Writes 8-bit PNG cells into <out>:
    <task>_src.png, <task>_tgt.png                       the pair
    <task>_<col>.png, <task>_<col>_p.png                 the reference budget and its perturbation
    <task>_<col>_q<B>.png                                every budget B of --budgets
and <out>/selection.json, which records the pair, the factor k and, for every (method, budget) cell, the domain
class of the stored state, so that a cell that is not adversarial under re-evaluation can be marked in the figure.
The paper sets the cells in a LaTeX tabular (tools/make_face_tables.py); this script draws nothing itself.

usage: python face_figure.py --face_root /root/autodl-tmp/face --kit /root/autodl-tmp/face/kit \
         --domain <dir with domain_rows.csv> --budget 1000 --budgets 2000 5000 10000 --out <dir>
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
    ap.add_argument("--budget", type=int, default=1000,
                    help="reference budget: selects the pair and scales the perturbation")
    ap.add_argument("--budgets", type=int, nargs="*", default=[],
                    help="extra budgets to export as a sweep, e.g. 2000 5000 10000")
    ap.add_argument("--pick", action="append", default=[], metavar="TASK=IMAGE",
                    help="use this source image for TASK instead of the rule; repeatable. Recorded in "
                         "selection.json as an author choice, so the figure caption can say so.")
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
    budgets = sorted({a.budget, *a.budgets})

    def save(x, path):
        Image.fromarray((np.clip(np.round(x * 255.0), 0, 255)).astype(np.uint8).transpose(1, 2, 0)).save(path)

    picks = {}
    for item in a.pick:
        if "=" not in item:
            raise SystemExit("[!] --pick needs TASK=IMAGE, got %r" % item)
        t, img = item.split("=", 1)
        picks[t.strip()] = img.strip()

    # Per-cell measurements, read from the frozen metric tables; nothing is recomputed here.
    mdir = os.path.join(a.face_root, "metrics")
    b5 = []
    b5path = os.path.join(mdir, "b5_metrics.csv")
    if os.path.exists(b5path):
        b5 = list(csv.DictReader(open(b5path)))
    PREFIX = {"LGC_RC_EGC": "lgcegc", "LatentHSJA": "lhsja", "HSJA": "hsja"}
    simrows = {}
    for m, pre in PREFIX.items():
        for task in sorted(fc.TASKS):
            q = os.path.join(mdir, "%s_%s.csv" % (pre, task))
            if os.path.exists(q):
                for r in csv.DictReader(open(q)):
                    simrows[(task, m, r.get("source"), str(r.get("budget")))] = r

    def cell_metrics(task, method, image, budget):
        """SSIM, LPIPS (VGG and AlexNet), DISTS, l2 from b5_metrics.csv; SIM and the 8-bit flag from the
        per-method table; the domain class from domain_rows.csv. Every value is read, never computed."""
        out = {}
        for r in b5:
            if (r.get("image") == image and r.get("method") == method
                    and str(r.get("budget")) == str(budget) and task in str(r.get("arch", ""))):
                for k in ("ssim", "lpips_vgg", "lpips_alex", "dists", "l2", "psnr", "adv_flag", "state_query"):
                    if r.get(k) not in (None, ""):
                        out[k] = r[k]
                break
        r = simrows.get((task, method, image, str(budget)))
        if r:
            for k in ("sim_mtcnn", "adv_uint8"):
                if r.get(k) not in (None, ""):
                    out[k] = r[k]
        out["class"] = cls.get((task, method, image, int(budget)), "?")
        return out

    sel = {"rule": __doc__.split("\n\n")[1].strip(), "budget": a.budget, "budgets": budgets,
           "selection": "authors" if picks else "rule", "picked": picks, "rows": {}}
    cells = []
    for task in sorted(fc.TASKS):
        pairs = fc.read_pairs(inp["pairs"][task][a.kind]["path"])
        img_dir = os.path.join(inp["data"][task]["flat"], "images")
        pick = None
        if task in picks:
            for k, (s_name, s_lbl, t_name, t_lbl) in enumerate(pairs):
                if s_name == picks[task]:
                    pick = (k, s_name, s_lbl, t_name, t_lbl)
                    break
            if pick is None:
                raise SystemExit(f"[!] {task}: --pick image {picks[task]} is not a source of the pairs file")
            print(f"[fig] {task}: pair chosen by the author ({picks[task]}), not by the rule")
        else:
            for k, (s_name, s_lbl, t_name, t_lbl) in enumerate(pairs):
                if all(cls.get((task, m, s_name, a.budget)) == "C" for m, _ in COLS):
                    pick = (k, s_name, s_lbl, t_name, t_lbl)
                    break
        if pick is None:
            raise SystemExit(f"[!] {task}: no pair with class C for all methods at {a.budget}")
        k, s_name, s_lbl, t_name, t_lbl = pick
        x0 = fc.load_face_image(os.path.join(img_dir, s_name))[0].numpy().astype(np.float32)
        xt = fc.load_face_image(os.path.join(img_dir, t_name))[0].numpy().astype(np.float32)
        save(x0, os.path.join(a.out, f"{task}_src.png"))
        save(xt, os.path.join(a.out, f"{task}_tgt.png"))

        xs_ref = {}
        classes = {}
        for m, tag in COLS:
            for b in budgets:
                x = load_adv(find_npz(store, fc.ARCH[task], s_name, m, b))[0].astype(np.float32)
                save(x, os.path.join(a.out, f"{task}_{tag}_q{b}.png"))
                classes[f"{tag}_q{b}"] = cls.get((task, m, s_name, b), "?")
                met = cell_metrics(task, m, s_name, b)
                for key, val in met.items():
                    cells.append({"task": task, "method": m, "col": tag, "image": s_name,
                                  "budget": b, "metric": key, "value": val})
                if b == a.budget:
                    xs_ref[tag] = x
        amp = max(float(np.percentile(np.abs(x - x0), 99.5)) for x in xs_ref.values())
        kf = 0.5 / max(amp, 1e-8)
        for tag, x in xs_ref.items():
            save(x, os.path.join(a.out, f"{task}_{tag}.png"))
            save(np.clip(0.5 + kf * (x - x0), 0.0, 1.0), os.path.join(a.out, f"{task}_{tag}_p.png"))
        sel["rows"][task] = {"pair_index": k, "source": s_name, "source_label": int(s_lbl), "target": t_name,
                             "target_label": int(t_lbl), "k": kf, "pairs_skipped": k, "class": classes}
        print(f"[fig] {task}: pair {k} {s_name} -> {t_name}; k = {kf:.3g}; budgets {budgets}")
        off = sorted(key for key, c in classes.items() if c != "C")
        if off:
            print(f"[fig] {task}: cells not of class C: {', '.join(off)}")
    sel["cells"] = cells
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=1)
    with open(os.path.join(a.out, "cell_metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "method", "col", "image", "budget", "metric", "value"])
        w.writeheader()
        for row in cells:
            w.writerow(row)
    print(f"[fig] wrote {len(cells)} measured cell values to cell_metrics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
