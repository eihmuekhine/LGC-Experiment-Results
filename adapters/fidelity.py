#!/usr/bin/env python3
"""B3 fidelity + hard-label audit for the four non-TtBA baselines (CGBA/CGBA-H, HSJA, ARS-OPT, Sign-OPT).

Two gates:

G2 (hard-label audit, 1 image per method, non_targeted, ResNet-50 V2):
    * the attack receives only decisions (tracker returns one-hot / labels, never logits);
    * tracker query count vs the official code's own counter (where one exists);
    * no query is issued after the 10 000 budget (BudgetExhausted);
    * uint8 validity (adv_uint8) is recorded.

G3 (fidelity, split rows 1-3, both modes):
    * run the OFFICIAL attack code with a logits-returning recorder (LogitRecorder) and with the decision-only
      tracker, on the same tensors and seed; compare the best-so-far pixel L2 at 1000/5000/10000.  Because every
      official attack reads only argmax/predict_label, the ratio must be 1.0 (band [0.90, 1.10]).

Usage:
    python adapters/fidelity.py --gate g2 [--method CGBA]     # one method, or all four
    python adapters/fidelity.py --gate g3 [--method CGBA]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)          # adapters/
sys.path.insert(0, REPO_ROOT)     # advstore.py + lgc_bench
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))  # hardlabel_tracker

import _common as C  # noqa: E402
from hardlabel_tracker import BudgetExhausted, HardLabelTracker, TrackerCore  # noqa: E402

METHODS = ("CGBA", "HSJA", "ARS-OPT", "Sign-OPT")
BASELINES = {
    "CGBA": "/root/autodl-tmp/baselines/CGBA",
    "HSJA": "/root/autodl-tmp/baselines/HSJA",
    "ARS-OPT": "/root/autodl-tmp/baselines/ARS-OPT",
    "Sign-OPT": "/root/autodl-tmp/baselines/Sign-OPT",
}
CKPT = [1000, 5000, 10000]
DATA_DIR = "/root/autodl-tmp/ttba_mini_flat"
CKPT_DIR = "/root/autodl-tmp/TtBA/model"

# top-level module names that collide between the official clones (utils/config/dataset/...): cleared between methods
_COLLIDE = {"utils", "config", "dataset", "proposed_attack", "hsja", "PARS_OPT", "glog", "bidict"}


def clear_official_modules() -> None:
    for m in list(sys.modules):
        if m.split(".")[0] in _COLLIDE:
            del sys.modules[m]


class LogitRecorder:
    """Logits-returning twin of HardLabelTracker (evaluation-only, for the fidelity comparison)."""

    def __init__(self, model, x0_pixel, y_src, y_tgt=None, budgets=(1000, 5000, 10000), max_queries=10000,
                 to_pixel=None, init_adv_pixel=None):
        self.model = model
        self.x0 = x0_pixel if x0_pixel.dim() == 4 else x0_pixel[None]
        self.y_src, self.y_tgt = int(y_src), (None if y_tgt is None else int(y_tgt))
        self.to_pixel = to_pixel or (lambda x: x)
        self.core = TrackerCore(budgets, max_queries)
        self.best = None
        if init_adv_pixel is not None:
            x = init_adv_pixel if init_adv_pixel.dim() == 4 else init_adv_pixel[None]
            self.best = x.detach().float().clone()
            self.core.set_initial(float((self.best - self.x0).flatten().norm()))

    @property
    def queries(self) -> int:
        return self.core.queries

    def _query(self, x):
        squeeze = x.dim() == 3
        xb = x[None] if squeeze else x
        n_req = xb.shape[0]
        n_ok = min(n_req, self.core.remaining)
        if n_ok == 0:
            raise BudgetExhausted(f"budget {self.core.max_queries} spent")
        xb = xb[:n_ok]
        with torch.no_grad():
            logits = self.model(xb)
            lab = logits.argmax(dim=1)
            pix = self.to_pixel(xb).float()
            l2 = (pix - self.x0.to(pix.device)).flatten(1).norm(dim=1)
            adv = (lab == self.y_tgt) if self.y_tgt is not None else (lab != self.y_src)
        _, events = self.core.process(l2.cpu().numpy(), adv.cpu().numpy())
        for kind, v in events:
            if kind == "improve":
                self.best = pix[v: v + 1].detach().clone()
        if n_ok < n_req:
            raise BudgetExhausted(f"budget {self.core.max_queries} spent inside a batch of {n_req}")
        return logits, lab, squeeze

    def __call__(self, x):
        logits, _, squeeze = self._query(x)
        return logits[0:1] if squeeze else logits

    forward = __call__

    def predict_label(self, x):
        _, lab, squeeze = self._query(x)
        return lab[0:1] if squeeze else lab


def best_l2(model) -> dict:
    model.core.carry_forward()
    return {b: model.core.recorded[b][0] for b in model.core.budgets}


def wrapped_victim(method, victim):
    if method == "CGBA":
        return C.UnnormalizeVictim(victim)
    if method == "ARS-OPT":
        return victim
    return C.ClampedVictim(victim)


def to_pixel_fn(method):
    return C.to_pixel_unnormalize if method == "CGBA" else C.to_pixel_clamp


def run_attack(method, model, x0, y_src, x_tgt, y_tgt, params) -> Optional[int]:
    """Run the OFFICIAL attack code with `model`; return the official code's own query counter (or None)."""
    if method == "CGBA":
        from proposed_attack import Proposed_attack

        src = C.normalize(x0)
        tar = C.normalize(x_tgt) if x_tgt is not None else None
        lb = C.normalize(torch.zeros_like(x0))
        ub = C.normalize(torch.ones_like(x0))
        attack = Proposed_attack(model, src, list(C.MEAN), list(C.STD), lb, ub, tar_img=tar,
                                 dim_reduc_factor=params["dim_reduc_factor"],
                                 attack_method=params["attack_method"], iteration=params["iteration"],
                                 initial_query=params["initial_query"], tol=params["tol"],
                                 sigma=params["sigma"], verbose_control=params["verbose_control"])
        attack.Attack()
        return int(attack.all_queries)

    if method == "HSJA":
        import hsja

        from run_hsja import HSJAPredictor

        pred = HSJAPredictor(model, x0.device)
        sample = x0[0].permute(1, 2, 0).cpu().numpy().astype("float32")
        target_image = x_tgt[0].permute(1, 2, 0).cpu().numpy().astype("float32") if x_tgt is not None else None
        hsja.hsja(pred, sample, clip_max=params["clip_max"], clip_min=params["clip_min"],
                  constraint=params["constraint"], num_iterations=params["num_iterations"],
                  gamma=params["gamma"], target_label=y_tgt, target_image=target_image,
                  stepsize_search=params["stepsize_search"], max_num_evals=params["max_num_evals"],
                  init_num_evals=params["init_num_evals"], verbose=False)
        return None  # HSJA has no internal total counter

    if method == "ARS-OPT":
        from run_arsopt import _install_shims

        _install_shims()
        from PARS_OPT.pars_opt_l2_norm_attack import ParsOptL2Norm

        targeted = y_tgt is not None
        model.arch = "resnet50"
        attacker = ParsOptL2Norm(model, [], "ImageNet", params["epsilon"], targeted, batch_size=1,
                                 k=params["k"], alpha=params["lr"], beta=params["beta"],
                                 iterations=params["iterations"], maximum_queries=params["maximum_queries"],
                                 method="ARS", clip_grad_max_norm=params["clip_grad_max_norm"], tol=params["tol"],
                                 grad_binary_search_tol=None, best_initial_target_sample=False, resized=False,
                                 dim_reduce_factor=params["dim_reduce_factor"], line_search=False)
        true_labels = torch.tensor([int(y_src)], device=x0.device)
        if targeted:
            target_labels = torch.tensor([int(y_tgt)], device=x0.device)
            out = attacker.targeted_attack(0, x0, target_labels, x_tgt)
        else:
            out = attacker.untargeted_attack(0, x0, true_labels)
        return int(out[1][0].item())

    if method == "Sign-OPT":
        from run_signopt import _SignOPTModel, _load_sign_opt

        OPT_attack_sign_SGD = _load_sign_opt()
        attacker = OPT_attack_sign_SGD(_SignOPTModel(model), k=params["k"])
        out = attacker.attack_untargeted(x0, [int(y_src)], alpha=params["alpha"], beta=params["beta"],
                                         iterations=params["iterations"], query_limit=params["query_limit"],
                                         distortion=None, svm=params["svm"], momentum=params["momentum"],
                                         stopping=params["stopping"])
        return int(out[3])

    raise SystemExit(f"unknown method {method}")


def params_for(method, targeted):
    if method == "CGBA":
        import run_cgba as A
        return dict(A.PARAMS, attack_method=("CGBA_H" if targeted else "CGBA"))
    if method == "HSJA":
        import run_hsja as A
        return dict(A.PARAMS)
    if method == "ARS-OPT":
        import run_arsopt as A
        return dict(A.BASE_PARAMS, lr=(0.2 if targeted else 0.4))
    if method == "Sign-OPT":
        import run_signopt as A
        return dict(A.PARAMS)
    raise SystemExit(f"unknown method {method}")


def build_models(method, victim, x0, y_src, y_tgt, x_tgt, budget):
    wrapped = wrapped_victim(method, victim)
    tp = to_pixel_fn(method)
    init = x_tgt if y_tgt is not None else None
    recorder = LogitRecorder(wrapped, x0, y_src, y_tgt, CKPT, budget, to_pixel=tp, init_adv_pixel=init)
    tracker = HardLabelTracker(wrapped, x0, y_src, y_tgt, CKPT, budget, to_pixel=tp, init_adv_pixel=init)
    return recorder, tracker


def onehot_probe() -> bool:
    """Structural probe on a fresh tracker (does not touch the attack budget): __call__ returns a one-hot."""
    class Dummy(torch.nn.Module):
        def forward(self, x):
            return x.flatten(1)[:, :3].repeat(1, 1).view(x.shape[0], 3) @ torch.eye(3, device=x.device)

    model = Dummy()
    x0 = torch.zeros(1, 3, 8, 8, device="cuda")
    with torch.no_grad():
        y_src = int(model(x0).argmax(1).item())
    tr = HardLabelTracker(model, x0, y_src, None, budgets=[4, 8], max_queries=8)
    out = tr(torch.rand(2, 3, 8, 8, device="cuda"))
    vals = set(np.round(out.detach().cpu().numpy().reshape(-1), 6).tolist())
    return bool(vals <= {0.0, 1.0}) and bool((out.sum(1) == 1).all())


def short_params(method, targeted=False):
    """Reduced-iteration hyper-parameters so the attack completes normally (official counter is returned)."""
    p = params_for(method, targeted)
    if method == "CGBA":
        return dict(p, iteration=3, initial_query=10)
    if method == "HSJA":
        return dict(p, num_iterations=2)
    if method == "ARS-OPT":
        return dict(p, iterations=3, k=10)
    if method == "Sign-OPT":
        return dict(p, iterations=3, k=10, query_limit=3000)
    return p


def g2(method: str) -> None:
    contract = C.load_contract()
    budget = C.contract_budget(contract)
    ckpts = C.contract_checkpoints(contract)
    arch, mode = "resnet50", "non_targeted"
    method_id = {"CGBA": "CGBA", "HSJA": "HSJA", "ARS-OPT": "ARS_OPT", "Sign-OPT": "Sign_OPT"}[method]
    sys.path.insert(0, BASELINES[method])
    clear_official_modules()
    victim = C.load_victim(arch, CKPT_DIR, "cuda")
    r = C.load_split_rows(f"splits/imagenet/ttba_github/{arch}_{mode}.csv")[0]
    name = r["source"]
    y_src = int(r["source_label"])
    x0 = C.load_image(DATA_DIR, name, "cuda")

    # full-budget run (tracker): verify budget enforced, no query after 10000, adv_uint8
    params = params_for(method, False)
    _, tracker = build_models(method, victim, x0, y_src, None, None, budget)
    C.seed_all(0)
    status = "budget"
    try:
        run_attack(method, tracker, x0, y_src, None, None, params)
        status = "ok"
    except BudgetExhausted:
        pass
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"RESULT {method} ERROR {type(e).__name__}: {e}", flush=True)
        return
    tq = tracker.queries
    rows = tracker.finalize(None, arch, C.stem_of(name), method_id, mode, eval_model=victim)
    u8 = ",".join(f"{r['budget']}:{r['adv_uint8']}" for r in rows)

    # short run (completes normally): compare the official code's own counter to the recorder count
    recorder = LogitRecorder(wrapped_victim(method, victim), x0, y_src, None, CKPT, budget,
                             to_pixel=to_pixel_fn(method))
    C.seed_all(0)
    official_q = None
    try:
        official_q = run_attack(method, recorder, x0, y_src, None, None, short_params(method))
    except BudgetExhausted:
        official_q = "budget"
    except Exception as e:  # noqa: BLE001
        official_q = f"err:{type(e).__name__}"
    rec_q = recorder.queries
    is_num = isinstance(official_q, int) or (isinstance(official_q, float))
    match = "PASS" if (is_num and int(official_q) == rec_q) else ("n/a" if not is_num else "FAIL")

    print(f"RESULT {method} onehot={'PASS' if onehot_probe() else 'FAIL'} tracker_q={tq} "
          f"budget_ok={'PASS' if tq <= budget else 'FAIL'} status={status} adv_uint8={u8} "
          f"official_q={official_q} recorder_q={rec_q} match={match}", flush=True)


def g3(method: str) -> None:
    contract = C.load_contract()
    budget = C.contract_budget(contract)
    arch = "resnet50"
    sys.path.insert(0, BASELINES[method])
    clear_official_modules()
    victim = C.load_victim(arch, CKPT_DIR, "cuda")
    for mode in ("non_targeted", "targeted"):
        if method == "Sign-OPT" and mode == "targeted":
            continue
        rows = C.load_split_rows(f"splits/imagenet/ttba_github/{arch}_{mode}.csv")[:3]
        for idx, r in enumerate(rows):
            name = r["source"]
            y_src = int(r["source_label"])
            y_tgt = int(r["target_label"]) if (mode == "targeted" and r.get("target_label")) else None
            x0 = C.load_image(DATA_DIR, name, "cuda")
            x_tgt = C.load_image(DATA_DIR, r["target"], "cuda") if y_tgt is not None else None
            params = params_for(method, mode == "targeted")
            recorder, tracker = build_models(method, victim, x0, y_src, y_tgt, x_tgt, budget)

            C.seed_all(idx)
            off_q = None
            try:
                off_q = run_attack(method, recorder, x0, y_src, x_tgt, y_tgt, params)
            except BudgetExhausted:
                pass
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"RESULT {method} {mode} {C.stem_of(name)} OFFICIAL_ERROR {type(e).__name__}: {e}", flush=True)
                continue
            off_l2 = best_l2(recorder)

            C.seed_all(idx)
            try:
                run_attack(method, tracker, x0, y_src, x_tgt, y_tgt, params)
            except BudgetExhausted:
                pass
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"RESULT {method} {mode} {C.stem_of(name)} ADAPTER_ERROR {type(e).__name__}: {e}", flush=True)
                continue
            ad_l2 = best_l2(tracker)

            ratios = []
            for b in CKPT:
                o, a = off_l2[b], ad_l2[b]
                ratios.append(float("nan") if (o == float("inf") or a == float("inf") or o == 0) else a / o)
            def fmt(v):
                return "inf" if v == float("inf") else f"{v:.3f}"
            print(f"RESULT {method} {mode} {C.stem_of(name)} "
                  f"off={fmt(off_l2[1000])},{fmt(off_l2[5000])},{fmt(off_l2[10000])} "
                  f"ad={fmt(ad_l2[1000])},{fmt(ad_l2[5000])},{fmt(ad_l2[10000])} "
                  f"ratio={ratios[0]:.4f},{ratios[1]:.4f},{ratios[2]:.4f} off_q={off_q} trk_q={tracker.queries}", flush=True)
            del recorder, tracker, x0, x_tgt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g2", "g3"], required=True)
    ap.add_argument("--method", choices=list(METHODS) + [None], default=None)
    a = ap.parse_args()
    methods = [a.method] if a.method else list(METHODS)
    for m in methods:
        (g2 if a.gate == "g2" else g3)(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
