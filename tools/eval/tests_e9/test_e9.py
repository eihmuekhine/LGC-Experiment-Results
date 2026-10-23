"""Synthetic tests of claims_e9.py and e9_gate.py (no GPU, no real result). L, 2026-10-23.

    python -m unittest discover -s tools/eval/tests_e9 -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)
import claims_e9  # noqa: E402
import e9_gate  # noqa: E402

BUDGETS = [100, 200, 500, 1000, 2000, 2500, 5000, 10000]
ARCHS = ["vgg19", "resnet50", "vit_b_32"]
MODES = {"non_targeted": "NT", "targeted": "T"}
N = 200

# multiplicative L2 effect of each variant relative to EGC (> 1 = variant worse) and additive LPIPS effect
L2_EFFECT = {"LGC_RC": {"NT": 2.0, "T": 1.0}, "LGC_RC_EGCI": {"NT": 0.95, "T": 1.08},
             "LGC_RC_HT1000": {"NT": 0.9, "T": 1.1}}
LP_EFFECT = {"LGC_RC": 0.0, "LGC_RC_EGCI": 0.0, "LGC_RC_HT1000": 0.05}


def write_cell(root_cells, root_metrics, cell, method, arch, mode, rng, base_l2, base_lp):
    m = MODES[mode]
    l2f = 1.0 if method == "LGC_RC_EGC" else L2_EFFECT[method][m]
    lpd = 0.0 if method == "LGC_RC_EGC" else (LP_EFFECT[method] if m == "T" else 0.0)
    d = os.path.join(root_cells, cell, mode)
    os.makedirs(d, exist_ok=True)
    pc, met, runs = [], [], []
    for i in range(N):
        img = f"ILSVRC2012_val_{i:08d}"
        noise = rng.lognormal(0, 0.05)
        runs.append({"image": img + ".JPEG", "method": method, "status": "ok"})
        for b in BUDGETS:
            l2 = base_l2[i] * (10000 / b) ** 0.3 * l2f * noise
            pc.append({"image": img + ".JPEG", "method": method, "budget": b, "pix_l2": l2, "adv_float": 1,
                       "found": 1, "seed": 1000 + i})
            lp = max(1e-4, base_lp[i] * (10000 / b) ** 0.3 + lpd + rng.normal(0, 0.002))
            met.append({"image": img, "method": method, "budget": b, "lpips_vgg": lp, "lpips_alex": lp * 0.8,
                        "dists": lp * 0.5, "ssim": 1 - lp})
    pd.DataFrame(pc).to_csv(os.path.join(d, "per_checkpoint.csv"), index=False)
    pd.DataFrame(runs).to_csv(os.path.join(d, "runs.csv"), index=False)
    with open(os.path.join(d, "config_single.json"), "w") as fh:
        json.dump({"methods": [method], "num_samples": N, "max_queries": 10000, "seed": 42, "checkpoints": BUDGETS,
                   "exclude_seeds": [992, 2027, 3141, 2718], "save_adv": True, "torch": "2.7.1+cu128"}, fh)
    os.makedirs(os.path.join(root_metrics, cell), exist_ok=True)
    pd.DataFrame(met).to_csv(os.path.join(root_metrics, cell, "metrics.csv"), index=False)


class E9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e9test_")
        t = cls.tmp
        os.makedirs(os.path.join(t, "contract"))
        with open(os.path.join(t, "contract", "paper_contract.yaml"), "w") as fh:
            fh.write("constants:\n  auc_budgets: [500, 1000, 2500, 5000, 10000]\n  n_main: 200\n"
                     "  stats: {bootstrap_B: 400}\n")
        with open(os.path.join(t, "prereg_build38_amendment3.txt"), "w") as fh:
            fh.write("test amendment\n")
        rng = np.random.default_rng(1)
        dev = {}
        for arch in ARCHS:
            for mode, m in MODES.items():
                base_l2 = rng.lognormal(0, 0.5, N)
                base_lp = rng.uniform(0.01, 0.2, N)
                write_cell(os.path.join(t, "b5", "cells"), os.path.join(t, "b5", "metrics"),
                           f"b5_main_LGC_RC_EGC_{arch}_{m}", "LGC_RC_EGC", arch, mode, rng, base_l2, base_lp)
                for v in claims_e9.VARIANTS:
                    write_cell(os.path.join(t, "Results"), os.path.join(t, "e9", "metrics"),
                               f"e9_{v}_{arch}_{m}", v, arch, mode, rng, base_l2, base_lp)
                dev[f"{arch}_{mode}"] = [f"ILSVRC2012_val_{i:08d}" for i in range(5)]
        with open(os.path.join(t, "dev.json"), "w") as fh:
            json.dump({"dev_overlap_rows_1_200": dev}, fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_claims(self, extra=()):
        t = self.tmp
        args = ["--b5-cells", f"{t}/b5/cells", "--b5-metrics", f"{t}/b5/metrics", "--e9-cells", f"{t}/Results",
                "--e9-metrics", f"{t}/e9/metrics", "--contract", f"{t}/contract/paper_contract.yaml",
                "--amendment", f"{t}/prereg_build38_amendment3.txt", "--dev-overlap", f"{t}/dev.json",
                "--out", f"{t}/claims", *extra]
        self.assertEqual(claims_e9.main(args), 0)
        name = "claims_no_devoverlap.json" if extra else "claims.json"
        with open(os.path.join(t, "claims", name)) as fh:
            return json.load(fh)

    def test_families_and_decisions(self):
        out = self.run_claims()
        rows = {r["claim_id"]: r for r in out["rows"]}
        self.assertEqual(out["meta"]["families"], {"A1": 3, "A2": 3, "A3": 3, "A4": 3})
        for arch in ARCHS:
            self.assertEqual(rows[f"A1_{arch}"]["decision"], "confirmed")      # RC NT 2x worse
            self.assertAlmostEqual(rows[f"A1_{arch}"]["estimate"], 0.5, delta=0.02)
            self.assertEqual(rows[f"A2_{arch}"]["decision"], "confirmed")      # HT1000 T LPIPS +0.05
            self.assertEqual(rows[f"A3_{arch}"]["decision"], "confirmed")      # EGCI T 8% worse
            self.assertEqual(rows[f"A4_{arch}"]["decision"], "confirmed")      # RC T parity, bound <= 1.03
            self.assertIsNone(rows[f"A4_{arch}"]["p_holm"])
            for f in ("A1", "A2", "A3"):
                self.assertIsNotNone(rows[f"{f}_{arch}"]["p_holm"])
            d = rows[f"D_HT1000_l2k_{arch}_NT"]                               # HT1000 better in NT: descriptive
            self.assertEqual(d["decision"], "descriptive")
            self.assertGreater(d["estimate"], 1.0)
            self.assertIsNone(d["p_one_sided"])
        self.assertEqual(sum(r["family"] is None for r in out["rows"]), 3 * 2 * 3 * 6 - 12)
        self.assertTrue(all(r["n_pairs"] == N and r["n_excluded"] == 0 for r in out["rows"]))
        lev = pd.read_csv(os.path.join(self.tmp, "claims", "levels_e9.csv"))
        self.assertEqual(len(lev), 4 * 3 * 2 * 6 * 2)

    def test_no_devoverlap(self):
        out = self.run_claims(("--no-devoverlap",))
        self.assertTrue(all(r["decision"] == "descriptive" for r in out["rows"]))
        self.assertTrue(all(r["n_pairs"] == N - 5 for r in out["rows"]))

    def test_missing_cell_fails(self):
        t = self.tmp
        p = os.path.join(t, "e9", "metrics", "e9_LGC_RC_vgg19_NT", "metrics.csv")
        os.rename(p, p + ".x")
        try:
            args = ["--b5-cells", f"{t}/b5/cells", "--b5-metrics", f"{t}/b5/metrics", "--e9-cells",
                    f"{t}/Results", "--e9-metrics", f"{t}/e9/metrics", "--contract",
                    f"{t}/contract/paper_contract.yaml", "--out", f"{t}/claims_x"]
            self.assertEqual(claims_e9.main(args), 1)
        finally:
            os.rename(p + ".x", p)

    def test_gate_pass_and_fail(self):
        t = self.tmp
        self.assertEqual(e9_gate.main(["e9_LGC_RC_vgg19_NT", "--results_root", f"{t}/Results",
                                       "--b5_cells", f"{t}/b5/cells"]), 0)
        self.assertEqual(e9_gate.main(["--all", "--results_root", f"{t}/Results", "--b5_cells", f"{t}/b5/cells"]), 0)
        p = os.path.join(t, "Results", "e9_LGC_RC_HT1000_vit_b_32_T", "targeted", "per_checkpoint.csv")
        df = pd.read_csv(p)
        df.loc[df.index[0], "seed"] = 7
        df.to_csv(p + ".bad", index=False)
        os.rename(p, p + ".orig")
        os.rename(p + ".bad", p)
        try:
            self.assertEqual(e9_gate.main(["e9_LGC_RC_HT1000_vit_b_32_T", "--results_root", f"{t}/Results",
                                           "--b5_cells", f"{t}/b5/cells"]), 1)
        finally:
            os.replace(p + ".orig", p)

    def test_check_inputs(self):
        import hashlib
        r = tempfile.mkdtemp(prefix="e9in_")
        try:
            def sha(b):
                return hashlib.sha256(b).hexdigest()
            data, vic = os.path.join(r, "data"), os.path.join(r, "vic")
            os.makedirs(os.path.join(data, "images"))
            os.makedirs(vic)
            os.makedirs(os.path.join(r, "checkpoints"))
            ae = b"autoencoder"
            open(os.path.join(r, "checkpoints", "vgg16_imagenet.pth"), "wb").write(ae)
            vb = b"victim"
            vf = f"net-{sha(vb)[:8]}.pth"
            open(os.path.join(vic, vf), "wb").write(vb)
            open(os.path.join(data, "ttba_mini_val_labels.txt"), "w").write("a.JPEG 1\n")
            split = {}
            os.makedirs(os.path.join(r, "splits", "imagenet", "ttba_github"))
            for arch in ARCHS:
                for mode, m in MODES.items():
                    rows = "source,source_label,target,target_label\n" + "".join(
                        f"s{i}.JPEG,1,t{i}.JPEG,2\n" for i in range(3))
                    sp = os.path.join(r, "splits", "imagenet", "ttba_github", f"{arch}_{mode}.csv")
                    open(sp, "w").write(rows)
                    split[f"{arch}_{mode}"] = sha(rows.encode())
                    cell = f"b5_main_LGC_RC_EGC_{arch}_{m}"
                    os.makedirs(os.path.join(r, "paper_exports/b5a2/cells", cell, mode))
                    open(os.path.join(r, "paper_exports/b5a2/cells", cell, mode, "per_checkpoint.csv"), "w").write("x\n")
                    os.makedirs(os.path.join(r, "paper_exports/b5a2/metrics", cell))
                    open(os.path.join(r, "paper_exports/b5a2/metrics", cell, "metrics.csv"), "w").write("x\n")
            for i in range(3):
                for n in (f"s{i}.JPEG", f"t{i}.JPEG"):
                    open(os.path.join(data, "images", n), "w").write("img")
            os.makedirs(os.path.join(r, "contract"))
            with open(os.path.join(r, "contract", "paper_contract.yaml"), "w") as fh:
                fh.write("constants:\n  split_sha256:\n" + "".join(f"    {k}: {v}\n" for k, v in split.items()))
            kw = dict(repo=r, data_dir=data, victim_dir=vic, victim_files=[vf], ae_prefix=sha(ae)[:12])
            self.assertEqual(e9_gate.check_inputs(**kw), [])
            os.remove(os.path.join(data, "images", "t1.JPEG"))
            self.assertEqual(len(e9_gate.check_inputs(**kw)), 1)
            kw["ae_prefix"] = "000000000000"
            self.assertEqual(len(e9_gate.check_inputs(**kw)), 2)
        finally:
            shutil.rmtree(r)

    def test_order_is_lpt(self):
        order = [c[0] for c in e9_gate.cells()]
        self.assertEqual(len(order), 18)
        self.assertEqual(order[0], "e9_LGC_RC_EGCI_vgg19_T")
        self.assertTrue(order[-1].startswith("e9_LGC_RC_HT1000_"))


if __name__ == "__main__":
    unittest.main()
