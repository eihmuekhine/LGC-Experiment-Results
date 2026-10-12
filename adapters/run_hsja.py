#!/usr/bin/env python3
"""Run the OFFICIAL HSJA (HopSkipJumpAttack, IEEE S&P 2020) on the shared split through the hard-label tracker.

Official code: github.com/Jianbo-Lab/HSJA, ``hsja.py`` (numpy core).  The Keras ``main.py`` / ``build_model.py``
parts are not used.  Only the model object and the data are adapted (skill Section 4).

Allowed changes (documented in env_HSJA.json / run_manifest.json):
  * model object: an object with ``predict(x)`` returning an argmax-able one-hot array, backed by the
    decision-only HardLabelTracker (no logits/probabilities reach the attack);
  * data input: our image x0 ([0,1], HWC for the numpy core) and, for targeted mode, the CSV target image;
  * num_iterations = 64 (main.py:124 default) so that, with max_num_evals = 1e4, the tracker's 10000-query
    budget binds (the repo has no total counter -- B1).

Everything else (constraint, gamma, stepsize_search, max_num_evals, init_num_evals, clip_min/clip_max, the whole
hsja() algorithm) is the official default, unchanged.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

METHOD = "HSJA"

PARAMS = {
    "constraint": "l2",
    "gamma": 1.0,
    "stepsize_search": "geometric_progression",
    "max_num_evals": 1e4,
    "init_num_evals": 100,
    "num_iterations": 64,
    "clip_min": 0.0,
    "clip_max": 1.0,
}

HYPER_SOURCES = {
    "constraint": "hsja.py:6 (default)",
    "gamma": "hsja.py:10 (default)",
    "stepsize_search": "hsja.py:13 (default)",
    "max_num_evals": "hsja.py:14 (default)",
    "init_num_evals": "hsja.py:15 (default)",
    "num_iterations": "main.py:124 (default 64; raised so the 10000-query tracker budget binds -- B1)",
    "clip_min": "hsja.py:7 / main.py:86",
    "clip_max": "hsja.py:6 / main.py:85",
}


class HSJAPredictor:
    """``predict`` view of the tracker for the numpy HSJA core.  Accepts (H,W,C) or (N,H,W,C) numpy arrays in
    [0,1] and returns one-hot scores (argmax identical; no other information leaks)."""

    def __init__(self, tracker, device):
        self.tracker = tracker
        self.device = device

    def predict(self, x):
        a = np.ascontiguousarray(np.asarray(x))
        single = a.ndim == 3
        if single:
            a = a[None]
        a = np.transpose(a, (0, 3, 1, 2))  # (N,H,W,C) -> (N,C,H,W)
        t = C.torch.from_numpy(a).float().to(self.device)
        out = self.tracker(t).cpu().numpy()
        return out.reshape(-1) if single else out


def make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget):
    return C.HardLabelTracker(C.ClampedVictim(victim), x0, y_src, y_tgt, checkpoints, budget,
                              to_pixel=C.to_pixel_clamp,
                              init_adv_pixel=x_tgt if y_tgt is not None else None)


def run_official(tracker, x0, y_src, x_tgt, y_tgt, params):
    import hsja

    device = x0.device
    model = HSJAPredictor(tracker, device)
    sample = x0[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H,W,C) in [0,1]
    target_image = x_tgt[0].permute(1, 2, 0).cpu().numpy().astype(np.float32) if x_tgt is not None else None
    hsja.hsja(model, sample,
              clip_max=params["clip_max"], clip_min=params["clip_min"],
              constraint=params["constraint"], num_iterations=params["num_iterations"],
              gamma=params["gamma"], target_label=y_tgt, target_image=target_image,
              stepsize_search=params["stepsize_search"], max_num_evals=params["max_num_evals"],
              init_num_evals=params["init_num_evals"], verbose=False)


def main() -> int:
    return C.run(METHOD, __doc__, PARAMS, HYPER_SOURCES, make_tracker, run_official, os.path.abspath(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
