"""Latent -> pixel hybrid (final LGC method, L2 operating point).

Stage 1 runs the latent method (RAG latent + residual coordinate, update PVR-E). The latent space
reaches low distances quickly but has a floor d* / ||P_V n||: its reachable set is a translate
of the decoder manifold. Stage 2 therefore continues in pixel space (TtBA_PVRE), warm-started from
the stage-1 image, with the remaining queries and the probe schedule continued
(n0 sqrt(t + t1 + 1), where t1 is the number of stage-1 iterations).

Switch rules (MethodSpec.switch):
  "q<N>" : after exactly N queries in stage 1 (e.g. "q1000", "q2500");
  "auto" : when the local decay exponent of the PIXEL L2 of the stage-1 state,
               s(q) = log(L(q / w) / L(q)) / log(w),   w = EngineConfig.switch_window (1.5),
           falls below EngineConfig.switch_slope (0.45), not before switch_min_q queries and not later
           than switch_max_frac of the budget. The rule reads only the attacker's own state and
           costs no queries. Pixel attacks decay with exponent ~0.4-0.9 over 1k-10k queries in the
           ResNet-50/Places365 pilot, whereas the latent exponent falls to ~0.2 at 5k-10k.

Every checkpoint state is tagged with the space it lives in (AttackResult.checkpoint_spaces), so the
runner evaluates stage-1 checkpoints in latent coordinates and stage-2 checkpoints in pixel space.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import torch

from .engine import REGISTRY, AttackResult, EngineConfig, GeometricAttack, MethodSpec
from .spaces import AttackSpace, HardLabelOracle, PixelSpace


def _pixel_l2(space: AttackSpace, z: torch.Tensor) -> float:
    return float((space.to_image(z) - space.x0).flatten().norm().item())


def make_switch_rule(spec: MethodSpec, cfg: EngineConfig, space: AttackSpace
                     ) -> Tuple[Optional[Callable[[int, torch.Tensor], bool]], int]:
    """Returns (stop_rule, stage-1 budget)."""
    sw = (spec.switch or "").lower()
    if sw.startswith("q"):
        q_s = min(int(sw[1:]), cfg.max_queries)
        return None, q_s
    if sw != "auto":
        raise ValueError(f"unknown switch rule {spec.switch!r}")
    hist: List[Tuple[int, float]] = []
    q_max = int(cfg.switch_max_frac * cfg.max_queries)

    def rule(q: int, state: torch.Tensor) -> bool:
        hist.append((q, _pixel_l2(space, state)))
        if q >= q_max:
            return True
        if q < cfg.switch_min_q:
            return False
        q_ref = q / cfg.switch_window
        past = [h for h in hist if h[0] <= q_ref]
        if not past:
            return False
        q0, l0 = past[-1]
        l1 = hist[-1][1]
        if q0 <= 0 or l1 <= 0 or l0 <= 0 or q <= q0:
            return False
        slope = math.log(l0 / l1) / math.log(q / q0)
        return slope < cfg.switch_slope

    return rule, cfg.max_queries


def run_hybrid(spec: MethodSpec, cfg: EngineConfig, latent_space: AttackSpace, pixel_space: PixelSpace,
               classifier: torch.nn.Module, source_label: int, target_label: Optional[int], seed: int,
               query_batch: int = 64, target_seed: Optional[torch.Tensor] = None,
               target_image: Optional[torch.Tensor] = None) -> Tuple[AttackResult, dict]:
    """Two-stage attack with a shared query budget cfg.max_queries. Returns (result, info)."""
    t0 = time.time()
    B = cfg.max_queries
    rule, q_stage1 = make_switch_rule(spec, cfg, latent_space)

    # ---------------- stage 1: latent
    cps1 = tuple(c for c in cfg.checkpoints if c <= q_stage1)
    cfg1 = replace(cfg, max_queries=q_stage1, checkpoints=cps1)
    spec1 = replace(spec, switch=None)
    o1 = HardLabelOracle(classifier, latent_space, source_label, target_label, q_stage1, query_batch)
    a1 = GeometricAttack(spec1, cfg1, latent_space, o1, seed, target_seed=target_seed, stop_rule=rule)
    r1 = a1.run()
    q1 = o1.queries

    info = {"switch_q": q1, "stage1_status": r1.status, "stage1_iterations": r1.iterations,
            "stopped_by_rule": bool(r1.diag.get("stopped_at"))}
    states = {}
    spaces = {}
    if r1.status == "ok":
        for cp, (z, sq) in r1.checkpoint_states.items():
            if cp <= q1:
                states[cp], spaces[cp] = (z, sq), latent_space

    # ---------------- stage 2: pixel, warm start from the stage-1 image
    remaining = B - q1
    pix_spec = REGISTRY[spec.pixel_stage]
    cps2 = tuple(c - q1 for c in cfg.checkpoints if c > q1)
    r2 = None
    latent_ok = r1.status == "ok" and r1.final_state is not None
    if remaining > cfg.update_reserve + 2 and cps2:
        cfg2 = replace(cfg, max_queries=remaining, checkpoints=cps2)
        o2 = HardLabelOracle(classifier, pixel_space, source_label, target_label, remaining, query_batch)
        warm = latent_space.to_image(r1.final_state) if latent_ok else None
        a2 = GeometricAttack(pix_spec, cfg2, pixel_space, o2, seed + 7_000_003,
                             target_image=target_image, init_point=warm,
                             it_offset=r1.iterations if warm is not None else 0)
        r2 = a2.run()
        info["stage2_status"] = r2.status
        info["warm_start"] = warm is not None

    # checkpoints after the switch: the better (pixel L2) of the stage-2 state and the final stage-1
    # state; both are verified adversarial and both were reached with <= cp queries
    lat_l2 = _pixel_l2(latent_space, r1.final_state) if latent_ok else float("inf")
    for cp in cfg.checkpoints:
        if cp <= q1:
            continue
        cand = None
        if r2 is not None and r2.status == "ok" and (cp - q1) in r2.checkpoint_states:
            x, sq2 = r2.checkpoint_states[cp - q1]
            cand = ((x, sq2 + q1), pixel_space, _pixel_l2(pixel_space, x))
        if latent_ok and (cand is None or lat_l2 < cand[2]):
            cand = ((r1.final_state.detach().clone(), r1.final_state_query), latent_space, lat_l2)
        if cand is not None:
            states[cp], spaces[cp] = cand[0], cand[1]

    by_tag: Counter = Counter(r1.by_tag)
    diag = {k: list(v) for k, v in r1.diag.items()}
    geom = [dict(g, stage="latent") for g in r1.geom]
    final_state, final_q, final_space = r1.final_state, r1.final_state_query, latent_space
    iterations = r1.iterations
    total_q = q1
    if r2 is not None:
        by_tag.update(r2.by_tag)
        for k, v in r2.diag.items():
            diag.setdefault(k, []).extend(v)
        geom += [dict(g, q=g["q"] + q1, it=g["it"] + r1.iterations, step=f"px_{g['step']}", stage="pixel")
                 for g in r2.geom]
        iterations += r2.iterations
        total_q += r2.queries
        if r2.status == "ok" and r2.final_state is not None and _pixel_l2(pixel_space, r2.final_state) <= lat_l2:
            final_state, final_q, final_space = r2.final_state, r2.final_state_query + q1, pixel_space
    diag["switch_q"] = [q1]

    status = "ok" if states else (r1.status if r2 is None else r2.status)
    res = AttackResult(
        method=spec.name, status=status, checkpoint_states=states, final_state=final_state,
        final_state_query=final_q, queries=total_q, by_tag=dict(by_tag), iterations=iterations,
        wall_time=time.time() - t0, trace=[], diag=diag, geom=geom, checkpoint_spaces=spaces,
    )
    info["final_space"] = final_space.name
    return res, info
