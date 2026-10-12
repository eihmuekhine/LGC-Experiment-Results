#!/usr/bin/env python3
"""Run the OFFICIAL ARS-OPT (prior-free mode) on the shared split through the hard-label tracker.

Official code: github.com/machanic/hard_label_attacks (arXiv 2512.21241), ``PARS_OPT/pars_opt_l2_norm_attack.py``
(class ``ParsOptL2Norm``), prior-free mode ``method='ARS'`` (no surrogate: pars_opt_l2_norm_attack.py:21, :317-327).

Allowed changes (documented in env_ARS_OPT.json / run_manifest.json):
  * model object: the decision-only HardLabelTracker.  The tracker carries ``arch`` only so the official
    constructor's CLIP gate (DataLoaderMaker.get_test_attacked_data) can run unchanged; it is never used by the
    prior-free algorithm (``self.model(x).max(1)[1]`` reads only the top-1 decision);
  * data input: our image x0 ([0,1]) and, for targeted mode, the CSV target image (the ``target_class_image``
    the official ``targeted_attack`` starts from), bypassing the repo's own ImageNet dataset
    (select_random_image_of_target_class / attack_all_images are not used);
  * import shims (adapter-side, no algorithm change): a no-op ``glog`` logger, a minimal ``bidict``, and a stub
    ``dataset.npz_dataset`` (this file is MISSING from the official clone).  None of these code paths is executed
    by the prior-free ARS loop -- they exist only so the official module imports;
  * maximum_queries = 10000 (contract budget F4; the official driver forces 20000 for ImageNet targeted at
    attack.py:127-128, but our tracker budget is 10000 for every method).

Hyper-parameters (ImageNet l2, source file:line) are the official defaults; NOTHING in the attack loop is changed.
"""
from __future__ import annotations

import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import torch  # noqa: E402

METHOD = "ARS_OPT"

BASE_PARAMS = {
    "epsilon": 10.0,
    "clip_grad_max_norm": 1.0,
    "tol": 1e-4,
    "k": 200,
    "beta": 0.001,
    "iterations": 1000,
    "maximum_queries": 10000,
    "dim_reduce_factor": 4,
}

HYPER_SOURCES = {
    "epsilon": "configures/PARS_OPT.json: ImageNet.l2.{un,targeted}.epsilon = 10.0",
    "clip_grad_max_norm": "configures/PARS_OPT.json: ImageNet.l2.clip_grad_max_norm = 1.0",
    "tol": "configures/PARS_OPT.json: ImageNet.l2.tol = 1e-4",
    "lr": "configures/PARS_OPT.json: ImageNet.l2.lr = 0.4 (untargeted) / 0.2 (targeted)",
    "k (est_grad_samples)": "PARS_OPT/attack.py:62 (default 200); pars_opt_l2_norm_attack.py:17 (q=k)",
    "beta": "pars_opt_l2_norm_attack.py:17 (class default 0.001)",
    "iterations": "pars_opt_l2_norm_attack.py:17 (class default 1000)",
    "maximum_queries": "contract budget 10000 (class default 10000; official driver forces 20000 for ImageNet targeted)",
    "dim_reduce_factor": "PARS_OPT/attack.py:98 (default 4); pars_opt_l2_norm_attack.py:18",
    "method": "pars_opt_l2_norm_attack.py:21 (surrogate only if 'PARS'; 'ARS' is prior-free)",
}

_ARCH = "resnet50"
_TARGETED = False


def _install_shims() -> None:
    """Adapter-side import shims for dependencies missing from the official clone (never used by the ARS loop)."""
    if "glog" not in sys.modules:
        g = types.ModuleType("glog")
        for fn in ("info", "warn", "warning", "error", "fatal", "flush", "check", "check_eq"):
            setattr(g, fn, lambda *a, **k: None)
        sys.modules["glog"] = g
    if "bidict" not in sys.modules:
        class _bidict(dict):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.inv = {v: kk for kk, v in self.items()}
        b = types.ModuleType("bidict")
        b.bidict = _bidict
        sys.modules["bidict"] = b
    if "dataset.npz_dataset" not in sys.modules:
        n = types.ModuleType("dataset.npz_dataset")

        class _Npz(torch.utils.data.Dataset):
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.zeros(3, 224, 224), torch.tensor(0)

        n.NpzDataset = _Npz
        n.NpzExtraDataset = _Npz
        n.NpzClipDataset = _Npz
        sys.modules["dataset.npz_dataset"] = n


def make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget):
    tracker = C.HardLabelTracker(victim, x0, y_src, y_tgt, checkpoints, budget,
                                 to_pixel=C.to_pixel_clamp,
                                 init_adv_pixel=x_tgt if y_tgt is not None else None)
    tracker.arch = _ARCH  # only gates the constructor's CLIP dataset branch; unused by the ARS loop
    return tracker


def run_official(tracker, x0, y_src, x_tgt, y_tgt, params):
    _install_shims()
    from PARS_OPT.pars_opt_l2_norm_attack import ParsOptL2Norm

    attacker = ParsOptL2Norm(
        tracker, [], "ImageNet", params["epsilon"], _TARGETED, batch_size=1,
        k=params["k"], alpha=params["lr"], beta=params["beta"], iterations=params["iterations"],
        maximum_queries=params["maximum_queries"], method="ARS",
        clip_grad_max_norm=params["clip_grad_max_norm"], tol=params["tol"],
        grad_binary_search_tol=None, best_initial_target_sample=False, resized=False,
        dim_reduce_factor=params["dim_reduce_factor"], line_search=False)
    true_labels = torch.tensor([int(y_src)], device=x0.device)
    if _TARGETED:
        target_labels = torch.tensor([int(y_tgt)], device=x0.device)
        attacker.targeted_attack(0, x0, target_labels, x_tgt)
    else:
        attacker.untargeted_attack(0, x0, true_labels)


def main() -> int:
    global _ARCH, _TARGETED
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--arch", default="resnet50")
    peek.add_argument("--mode", choices=["non_targeted", "targeted"], default="non_targeted")
    ns, _ = peek.parse_known_args()
    _ARCH = ns.arch
    _TARGETED = ns.mode == "targeted"
    params = dict(BASE_PARAMS, lr=(0.2 if _TARGETED else 0.4))
    return C.run(METHOD, __doc__, params, HYPER_SOURCES, make_tracker, run_official, os.path.abspath(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
