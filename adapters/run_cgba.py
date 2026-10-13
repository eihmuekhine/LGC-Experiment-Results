#!/usr/bin/env python3
"""Run the OFFICIAL CGBA / CGBA-H (ICCV 2023) on the shared split through the hard-label tracker.

Official code: github.com/Farhamdur/CGBA, ``proposed_attack.py`` (class ``Proposed_attack``).  CGBA is run for
non_targeted (``attack_method='CGBA'``, contract method id ``CGBA``) and CGBA-H for targeted
(``attack_method='CGBA_H'``, contract method id ``CGBA_H``); ``go_to_boundary_CGBA`` /
``go_to_boundary_CGBA_H`` are the official boundary walks at proposed_attack.py:151 / :131.

Allowed changes (documented in env_<method>.json / run_manifest.json):
  * model object (FAITHFUL path, B3b T2): the normalised tensor goes STRAIGHT into the UN-normalised network
    exactly as the official drivers feed ``net = torch_models.resnet50(pretrained=True)`` with no normalisation
    layer (Non_targeted_attack.py:42,84-88 / Targeted_attack.py:43,94-98).  The raw net is ``_Norm.m``
    (baseline_protocol.py:354-367); the tracker maps to pixel space only for L2 (``to_pixel_unnormalize``).  The
    earlier ``UnnormalizeVictim`` wrapper un-normalised then let the victim re-normalise (a float round-trip of
    max 2.384e-07); B3b T2 showed this round-trip flips a decision at query ~102 and diverges the trajectory, so
    the faithful path is used;
  * data input: our image x0 and (targeted) the CSV target image, both converted to ImageNet-normalised space
    (the attack's own input space) with the SAME mean/std as the official drivers (Non_targeted_attack.py:35-36 /
    Targeted_attack.py:36-37); lb/ub = normalise(0)/normalise(1) exactly as the drivers' valid_bounds(delta=255)
    (Non_targeted_attack.py:80-95 / Targeted_attack.py:89-104);
  * the legacy driver preprocessing ``Resize((224,224))`` is NOT used: images come from the shared TtBA loader
    (Resize 256 -> CenterCrop 224) per F3/F4 (B1);
  * boundary-walk query count (B3b T3): ``PatchedProposedAttack`` overrides ``go_to_boundary_CGBA`` /
    ``go_to_boundary_CGBA_H`` to return the true query count (``all_queries`` diff) -- the official code returns
    ``num_calls==40`` after only 39 ``is_adversarial`` calls at the 40-iteration cap, which overshoots ``q_num`` by
    one and trips ``assert self.all_queries == q_num`` (proposed_attack.py:252).  The returned boundary point is
    unchanged (the cap branch returns ``x_b``), so the trajectory is unchanged; only the count is corrected;
  * verbose_control='No' (print-only; the algorithm is unchanged).

`--cgba_variant fixed` (default) additionally overrides ``SemiCircular_boundary_search`` with a verbatim copy of
proposed_attack.py:171-196 whose only change wraps the acos argument at :182 in ``torch.clamp(..., -1.0, 1.0)``
(D11_outcome: the un-clipped argument reaches 1 + 2^-23 and ``torch.acos`` returns NaN, freezing the best-so-far).
``--cgba_variant d8only`` keeps exactly the D8 count fix (no clip) and is the E17 appendix variant
(method id ``CGBA_D8only``, non_targeted only).

Everything else (dim_reduc_factor, iteration, initial_query, tol, sigma, the batch gradient estimator size 40 and
the whole Attack() loop) is the official default, unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import torch  # noqa: E402

PARAMS = {
    "dim_reduc_factor": 4,
    "iteration": 93,
    "initial_query": 30,
    "tol": 1e-4,
    "sigma": 2e-4,
    "verbose_control": "No",
}

HYPER_SOURCES = {
    "dim_reduc_factor": "Non_targeted_attack.py:33 / Targeted_attack.py:31",
    "iteration": "Non_targeted_attack.py:30 / Targeted_attack.py:33",
    "initial_query": "proposed_attack.py:15 (class default 30)",
    "tol": "proposed_attack.py:15 (class default 1e-4)",
    "sigma": "proposed_attack.py:15 (class default 2e-4)",
    "mean/std": "Non_targeted_attack.py:35-36 / Targeted_attack.py:36-37 (ImageNet)",
    "attack_method": "go_to_boundary_CGBA proposed_attack.py:151 (NT) / go_to_boundary_CGBA_H :131 (T)",
    "grad_estimator_batch_size": "proposed_attack.py:32 (class default 40)",
    "verbose_control": "print-only; set 'No' to reduce log noise (allowed logging change)",
}


def _faithful_model(victim):
    """The UN-normalised network the official drivers feed the normalised tensor into.

    ``build_ttba_github_victim`` wraps the torchvision net in ``_Norm`` (baseline_protocol.py:354-367) and stores the
    raw net as ``_Norm.m``; ``net = torch_models.resnet50(pretrained=True)`` has no normalisation layer, so the
    normalised tensor must go straight into that raw net.  The E10/E11 victims expose the same raw net as
    ``NormalizedVictim.net`` (Places365) / ``Sequential.model`` (RobustBench); ``C.raw_net`` extracts it.  A victim
    without such a wrapper (unit-test dummy) is returned unchanged."""
    return C.raw_net(victim)


def make_tracker(victim, x0, y_src, y_tgt, x_tgt, checkpoints, budget):
    return C.HardLabelTracker(_faithful_model(victim), x0, y_src, y_tgt, checkpoints, budget,
                              to_pixel=C.to_pixel_unnormalize,
                              init_adv_pixel=x_tgt if y_tgt is not None else None)


PAPER_FAITHFUL_FIX = "clip acos arg, proposed_attack.py:182, D11_outcome"

_VARIANT = "fixed"

# D11 observation: per-image clip-event counts (acos argument outside [-1,1] before the clamp), in C.run
# processing order.  Collected by run_official and merged into runs_<method>.csv after C.run returns.
_CLIP_EVENTS: list = []


def _patched_attack_class(variant: str = "fixed"):
    """``Proposed_attack`` subclass with the D8 count fix and (``variant == "fixed"``) the D11 acos clip.

    The official walks start ``num_calls = 1`` and return ``num_calls == 40`` after only 39 ``is_adversarial`` calls
    (the 40-iteration cap branch returns ``x_b`` unchanged), so ``q_num`` overshoots ``all_queries`` by one and trips
    ``assert self.all_queries == q_num`` (proposed_attack.py:252).  The returned boundary point is left untouched
    (trajectory identical); only the count is corrected.  A call whose corrected count differs from the official one
    is logged.  Defined lazily because ``proposed_attack`` lives in the official clone, which is only on sys.path
    after the adapter's main loop adds it.

    For ``variant == "fixed"`` (D11_outcome), ``SemiCircular_boundary_search`` is additionally overridden with a
    verbatim copy of proposed_attack.py:171-196 whose ONLY change wraps the ``acos`` argument at :182 in
    ``torch.clamp(..., -1.0, 1.0)`` (as the machanic reimplementation does at attack_gpu.py:263-264).  The un-clipped
    argument reaches 1 + 2^-23 in float32 when the bisection directions coincide, giving ``acos`` -> NaN and a
    frozen best-so-far (D11 verdict b).  ``variant == "d8only"`` keeps exactly the current adapter (D8 count fix, no
    clip) and is the E17 appendix variant (method id ``CGBA_D8only``)."""
    from proposed_attack import Proposed_attack

    class PatchedProposedAttack(Proposed_attack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cap_fix_log = []
            self.clip_events_182 = 0  # D11 observation: acos arg outside [-1,1] before the clamp (per image)

        def _counted(self, c0, n, name):
            new = self.all_queries - c0
            if new != n:
                self._cap_fix_log.append((name, int(n), int(new)))
                print(f"[cap-fix] {name}: official count {n} != actual {new}", flush=True)
            return new

        def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
            c0 = self.all_queries
            out, n = super().go_to_boundary_CGBA(x_s, eta_o, x_b)
            return out, self._counted(c0, n, "CGBA")

        def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):
            c0 = self.all_queries
            out, n = super().go_to_boundary_CGBA_H(x_s, eta_o, x_b)
            return out, self._counted(c0, n, "CGBA_H")

        if variant == "fixed":
            def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
                num_calls = 0
                norm_dis = torch.norm(x_b-x_0)
                boundary_dir = (x_b-x_0)/torch.norm(x_b-x_0)
                clean_dir = (p_near_boundary - x_0)/torch.norm(p_near_boundary - x_0)
                adv_dir = boundary_dir
                adv = x_b
                clean = x_0
                while True:
                    mid_dir = adv_dir + clean_dir
                    mid_dir = mid_dir/torch.norm(mid_dir)
                    arg = torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))/ (torch.linalg.norm(boundary_dir)*torch.linalg.norm(mid_dir))
                    if arg < -1.0 or arg > 1.0:
                        self.clip_events_182 += 1
                    theta = torch.acos(torch.clamp(arg, -1.0, 1.0))
                    d = torch.cos(theta)*norm_dis
                    x_mid = x_0 + mid_dir*d
                    num_calls +=1
                    if self.is_adversarial(x_mid)==1:
                        adv_dir = mid_dir
                        adv = x_mid  
                    else:
                        clean_dir = mid_dir  
                        clean = x_mid                             
                    if torch.norm(adv-clean).cpu().numpy()<self.tol:
                        break
                    if num_calls >100:
                        break      
                return adv, num_calls

    return PatchedProposedAttack


def run_official(tracker, x0, y_src, x_tgt, y_tgt, params):
    PatchedProposedAttack = _patched_attack_class(_VARIANT)
    attack = None
    try:
        src = C.normalize(x0)
        tar = C.normalize(x_tgt) if x_tgt is not None else None
        lb = C.normalize(torch.zeros_like(x0))
        ub = C.normalize(torch.ones_like(x0))
        attack = PatchedProposedAttack(tracker, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=tar,
                                       dim_reduc_factor=params["dim_reduc_factor"],
                                       attack_method=params["attack_method"], iteration=params["iteration"],
                                       initial_query=params["initial_query"], tol=params["tol"],
                                       sigma=params["sigma"], verbose_control=params["verbose_control"])
        attack.Attack()
    finally:
        # observation-only: append exactly one entry per image so the C.run runs_*.csv rows stay aligned
        _CLIP_EVENTS.append(0 if attack is None else int(getattr(attack, "clip_events_182", 0)))


def _write_clip_events(method: str, out_dir: str, mode: str) -> None:
    """Merge the per-image clip-event counts (in C.run processing order) into runs_<method>.csv."""
    import csv as _csv

    run_path = os.path.join(out_dir, mode, f"runs_{method}.csv")
    if not os.path.exists(run_path):
        return
    with open(run_path, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    fields = list(rows[0].keys()) if rows else []
    if "clip_events_182" not in fields:
        fields.append("clip_events_182")
    for i, r in enumerate(rows):
        r["clip_events_182"] = _CLIP_EVENTS[i] if i < len(_CLIP_EVENTS) else ""
    with open(run_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    global _VARIANT
    _CLIP_EVENTS.clear()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--mode", choices=["non_targeted", "targeted"], default="non_targeted")
    peek.add_argument("--cgba_variant", choices=["fixed", "d8only"], default="fixed")
    peek.add_argument("--out_dir", default=None)
    ns, _ = peek.parse_known_args()
    _VARIANT = ns.cgba_variant
    targeted = ns.mode == "targeted"
    if _VARIANT == "d8only" and targeted:
        raise SystemExit("[!] --cgba_variant d8only is non_targeted only (E17 appendix); use fixed for targeted mode")
    if _VARIANT == "d8only":
        method = "CGBA_D8only"
    else:
        method = "CGBA_H" if targeted else "CGBA"
    params = dict(PARAMS, attack_method=("CGBA_H" if targeted else "CGBA"))

    def extra_env(args, contract):
        return {"cgba_variant": _VARIANT, "paper_faithful_fix": PAPER_FAITHFUL_FIX}

    rc = C.run(method, __doc__, params, HYPER_SOURCES, make_tracker, run_official,
               os.path.abspath(__file__), extra_env=extra_env)
    if ns.out_dir:
        _write_clip_events(method, ns.out_dir, ns.mode)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
