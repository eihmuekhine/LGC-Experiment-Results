#!/usr/bin/env python3
"""B4d T4(e) — machanic/hard_label_attacks CGBA cross-check (CROSS-CHECK ONLY, never reported as CGBA).

Run the machanic CGBA reimplementation (``CGBA/attack_gpu.py`` @ commit ab71cb88, pin) with the SAME victim,
images, decision-only HardLabelTracker and flags (dim_reduc_factor=4, iterations=93, initial_query=30,
tol=1e-4, sigma=2e-4, budget 10000, init_rnd_adv=True) on resnet50 NT rows 1-3.  Report the tracker best
pix_l2 @2000/5000/10000 and stall yes/no, next to the OFFICIAL (Farhamdur) CGBA numbers for the same rows.

The machanic code works in [0,1] pixel space (clip_min=0, clip_max=1); its dataset loader is stubbed
(DataLoaderMaker.get_test_attacked_data -> dummy len-1 loader) because we feed our own images via
``CGBA.attack(batch_index, images, None, true_labels, None)``.  It is NOT used as a primary baseline.

    python adapters/diag/diag_b4d_t4e_machanic.py
"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
sys.path.insert(0, ADAPTERS)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

AR = "/root/autodl-tmp/baselines/ARS-OPT"
sys.path.insert(0, AR)

import _common as C  # noqa: E402
from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402
import torch  # noqa: E402


def _install_shims():
    """Adapter-side import shims for deps missing from the machanic clone (never used by the CGBA loop)."""
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
    # models.standard_model / models.defensive_model pull in pretrainedmodels/timm/clip (not installed, and unused
    # by the prior-free attack() loop) -> stub them so `from models.standard_model import StandardModel` resolves.
    if "models.standard_model" not in sys.modules:
        std = types.ModuleType("models.standard_model")

        class StandardModel:
            input_size = (1, 224, 224)
        std.StandardModel = StandardModel
        sys.modules["models.standard_model"] = std
    if "models.defensive_model" not in sys.modules:
        dm = types.ModuleType("models.defensive_model")

        class DefensiveModel:
            pass
        dm.DefensiveModel = DefensiveModel
        sys.modules["models.defensive_model"] = dm


_install_shims()

CKPT_DIR = "/root/autodl-tmp/TtBA/model"
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
BUDGET = 10000
ARCH = "resnet50"
BUDGETS = [2000, 5000, 10000]

# official (Farhamdur) CGBA numbers for the same 3 rows, from the B4 smoke cell
OFFICIAL = {
    "ILSVRC2012_val_00007450": {2000: 85.078, 5000: 85.078, 10000: 85.078},
    "ILSVRC2012_val_00015433": {2000: 74.569, 5000: 74.569, 10000: 74.569},
    "ILSVRC2012_val_00025151": {2000: 54.705, 5000: 54.705, 10000: 54.705},
}


def run_image(r, idx):
    name = r["source"]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, name, "cuda")
    victim = C.load_victim(ARCH, CKPT_DIR, "cuda")  # [0,1] -> logits (normalisation inside)
    ckpts = C.contract_checkpoints(C.load_contract())
    tracker = HardLabelTracker(victim, x0, y_src, None, ckpts, BUDGET, to_pixel=C.to_pixel_clamp)
    tracker.arch = "resnet50"  # only gates the CLIP dataset branch; unused by the ARS loop

    # stub the machanic ImageNet dataset loader (we feed our own image)
    import CGBA.attack_gpu as m  # noqa: E402
    from dataset.dataset_loader_maker import DataLoaderMaker  # noqa: E402

    _orig = DataLoaderMaker.get_test_attacked_data

    def _dummy(dataset, batch_size, arch):
        return types.SimpleNamespace(dataset=[0])

    DataLoaderMaker.get_test_attacked_data = _dummy

    cgba = m.CGBA(tracker, "ImageNet", clip_min=0.0, clip_max=1.0, height=224, width=224, channels=3,
                  norm="l2", epsilon=0.0, load_random_class_image=False, tol=1e-4, sigma=2e-4,
                  init_rnd_adv=True, dim_reduce_factor=4, attack_method="CGBA",
                  iterations=93, initial_query=30, grad_estimator_batch_size=40,
                  maximum_queries=BUDGET, batch_size=1)
    DataLoaderMaker.get_test_attacked_data = _orig

    true_labels = torch.tensor([int(y_src)], device=x0.device)
    C.seed_all(idx)
    status = "ok"
    try:
        cgba.attack(0, x0, None, true_labels, None)
    except BudgetExhausted:
        status = "budget"

    rec = {b: tracker.core.recorded.get(b) for b in ckpts}
    l2 = {b: (rec[b][0] if rec[b] is not None else float("nan")) for b in BUDGETS}
    stalled = l2[2000] == l2[10000]
    off = OFFICIAL.get(C.stem_of(name), {})
    print(f"RESULT T4e {C.stem_of(name)} status={status} queries={tracker.queries} "
          f"2k={l2[2000]:.3f} 5k={l2[5000]:.3f} 10k={l2[10000]:.3f} stalled={stalled} "
          f"official_2k/5k/10k={off.get(2000, float('nan')):.3f}/{off.get(5000, float('nan')):.3f}/"
          f"{off.get(10000, float('nan')):.3f}")


def main():
    rows = C.load_split_rows(f"splits/imagenet/ttba_github/{ARCH}_non_targeted.csv")[:3]
    for i, r in enumerate(rows):
        run_image(r, i)
    print("DONE T4e")


if __name__ == "__main__":
    main()
