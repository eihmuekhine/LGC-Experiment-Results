#!/usr/bin/env python3
"""Run the OFFICIAL Sign-OPT (ICLR 2020) on the shared split through the hard-label tracker.

Official code: github.com/cmhcbb/attackbox, ``attack/Sign_OPT.py`` (class ``OPT_attack_sign_SGD``).  The module is
loaded directly (the package ``attack/__init__.py`` imports many unrelated attacks, some with extra deps).

Allowed changes (documented in env_Sign_OPT.json / run_manifest.json):
  * model object: the decision-only HardLabelTracker (``predict_label`` returns the top-1 label only);
  * data input: our image x0 ([0,1]) and its label;
  * query_limit = 10000 (contract budget; the official ``attack_untargeted`` default is 20000, the ``__call__``
    default is 4000, the shell script passes 40000) so the tracker's 10000-query budget binds.

NOT supported: targeted mode.  The official ``__call__(targeted=True)`` raises ``NotImplementedError``
(``attack/Sign_OPT.py:370``) and ``attack_targeted`` is absent -- Sign-OPT is untargeted-only (B1; reported to the
threat-model table).  The adapter refuses ``--mode targeted`` with a clear error.
"""
from __future__ import annotations

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

METHOD = "Sign_OPT"

PARAMS = {
    "k": 200,
    "alpha": 0.2,
    "beta": 0.001,
    "iterations": 1000,
    "query_limit": 10000,
    "svm": False,
    "momentum": 0.0,
    "stopping": 0.0001,
}

HYPER_SOURCES = {
    "k": "attack/Sign_OPT.py:43 (class default 200 gradient directions)",
    "alpha": "attack/Sign_OPT.py:53 (attack_untargeted default)",
    "beta": "attack/Sign_OPT.py:53 (attack_untargeted default)",
    "iterations": "attack/Sign_OPT.py:53 (attack_untargeted default 1000)",
    "query_limit": "contract budget 10000 (official attack_untargeted default 20000 / __call__ 4000 / signopt-l2.sh 40000)",
    "svm": "attack/Sign_OPT.py:53 (default False)",
    "momentum": "attack/Sign_OPT.py:53 (default 0.0)",
    "stopping": "attack/Sign_OPT.py:53 (default 0.0001)",
}

def _find_repo() -> str:
    """Locate the attackbox clone on sys.path (``_common.run`` inserts ``--repo`` at sys.path[0])."""
    for p in sys.path:
        if p and os.path.isfile(os.path.join(p, "attack", "Sign_OPT.py")):
            return p
    raise SystemExit("[!] attack/Sign_OPT.py not found on sys.path; pass --repo <attackbox clone>")


def _load_sign_opt():
    """Load attack/Sign_OPT.py directly (skip attack/__init__.py which imports unrelated attacks)."""
    path = os.path.join(_find_repo(), "attack", "Sign_OPT.py")
    spec = importlib.util.spec_from_file_location("_attackbox_sign_opt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OPT_attack_sign_SGD


def make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget):
    if y_tgt is not None:
        raise SystemExit("[!] Sign-OPT has no targeted mode (attack/Sign_OPT.py:370 NotImplementedError); "
                         "run --mode non_targeted only.")
    return C.HardLabelTracker(C.ClampedVictim(victim), x0, y_src, None, checkpoints, budget,
                              to_pixel=C.to_pixel_clamp)


class _SignOPTModel:
    """``predict_label`` view of the tracker that returns a 0-dim scalar tensor, matching the official
    ``PytorchModel.predict_label`` (``models.py:45`` returns ``predict[0]``).  The Sign-OPT loop compares the
    label against an int and formats it with ``{:d}``, so a (B,) tensor would break those sites."""

    def __init__(self, tracker):
        self.tracker = tracker

    def predict_label(self, x):
        lab = self.tracker.predict_label(x)
        return lab[0] if lab.numel() > 1 else lab.view(())


def run_official(tracker, x0, y_src, x_tgt, y_tgt, params):
    OPT_attack_sign_SGD = _load_sign_opt()
    attacker = OPT_attack_sign_SGD(_SignOPTModel(tracker), k=params["k"])
    attacker.attack_untargeted(x0, [int(y_src)], alpha=params["alpha"], beta=params["beta"],
                               iterations=params["iterations"], query_limit=params["query_limit"],
                               distortion=None, svm=params["svm"], momentum=params["momentum"],
                               stopping=params["stopping"])


def main() -> int:
    return C.run(METHOD, __doc__, PARAMS, HYPER_SOURCES, make_tracker, run_official, os.path.abspath(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
