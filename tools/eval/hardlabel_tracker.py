"""Uniform hard-label victim wrapper for running OFFICIAL baseline code unchanged.

Wrap the victim model that an official attack implementation calls.  The wrapper
  * counts every image the victim classifies (one query per image, batches included),
  * enforces the query budget (raises BudgetExhausted once the budget is spent),
  * returns ONLY the top-1 decision (one-hot "logits" / integer labels) -> the attack cannot use scores,
  * tracks the best adversarial example seen so far (minimum pixel L2 to x0 among queried adversarial inputs),
    which is the "L2 of the best adversarial example so far" convention of the TtBA protocol,
  * records that best example at every query checkpoint and writes it with advstore (any metric later),
  * keeps a compact trace (query index, best L2) for L2-vs-query curves at any budget.

The bookkeeping lives in TrackerCore (pure Python / numpy, unit-tested in selftest_evalkit.py); the torch layer
only computes labels and L2 on the device and clones a tensor when the best example improves.

Typical use inside an adapter (see adapters/adapter_template.py):

    tracker = HardLabelTracker(victim, x0_pixel, y_src, y_tgt=None, budgets=(500, 1000, 2000, 2500, 5000, 10000),
                               max_queries=10000, to_pixel=lambda x: x)  # x already in [0, 1]
    try:
        official_attack(model=tracker, x=x0, ...)          # official code, unchanged
    except BudgetExhausted:
        pass
    rows = tracker.finalize(store_root, arch, image, method, mode, eval_model=victim)
"""
from __future__ import annotations

import bisect
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BudgetExhausted(RuntimeError):
    """Raised when the attack asks for a query beyond the budget (after the allowed part was processed)."""


@dataclass
class TrackerCore:
    """Backend-free bookkeeping: query counter, best-so-far adversarial, checkpoint snapshots, trace.

    process(l2, adv) consumes one batch of already-classified queries (in order) and returns
    (n_allowed, events), events being ("improve", j) for batch position j that became the new best and
    ("checkpoint", budget) when the counter reaches a budget.  The caller applies events in order.
    """

    budgets: Sequence[int]
    max_queries: int
    queries: int = 0
    best_l2: float = float("inf")
    best_query: int = -1
    trace: List[Tuple[int, float]] = field(default_factory=list)
    recorded: Dict[int, Tuple[float, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.budgets = sorted(int(b) for b in self.budgets if 0 < int(b) <= int(self.max_queries))

    @property
    def remaining(self) -> int:
        return max(0, int(self.max_queries) - self.queries)

    def set_initial(self, l2: float) -> None:
        """A known adversarial start obtained WITHOUT a query (e.g. the target image in targeted mode)."""
        if l2 < self.best_l2:
            self.best_l2, self.best_query = float(l2), 0
            self.trace.append((0, float(l2)))

    def process(self, l2: Iterable[float], adv: Iterable[bool]) -> Tuple[int, List[Tuple[str, int]]]:
        l2 = np.asarray(list(l2), dtype=np.float64)
        adv = np.asarray(list(adv), dtype=bool)
        n = min(len(l2), self.remaining)
        events: List[Tuple[str, int]] = []
        for j in range(n):
            self.queries += 1
            if adv[j] and l2[j] < self.best_l2:
                self.best_l2, self.best_query = float(l2[j]), self.queries
                self.trace.append((self.queries, float(l2[j])))
                events.append(("improve", j))
            i = bisect.bisect_left(self.budgets, self.queries)
            if i < len(self.budgets) and self.budgets[i] == self.queries:
                self.recorded[self.queries] = (self.best_l2, self.best_query)
                events.append(("checkpoint", self.queries))
        return n, events

    def carry_forward(self) -> List[int]:
        """Budgets above the final query count inherit the final best (the attack stopped early: still valid).
        Budgets at or below the count were recorded when reached; they are never back-filled."""
        added = []
        for b in self.budgets:
            if b not in self.recorded and b > self.queries:
                self.recorded[b] = (self.best_l2, self.best_query)
                added.append(b)
        return added


class HardLabelTracker:
    """torch layer around TrackerCore.  Behaves like the wrapped model but leaks only the top-1 decision."""

    def __init__(self, model: Any, x0_pixel: Any, y_src: int, y_tgt: Optional[int] = None,
                 budgets: Sequence[int] = (500, 1000, 2000, 2500, 5000, 10000), max_queries: int = 10000,
                 to_pixel: Optional[Callable[[Any], Any]] = None, init_adv_pixel: Any = None) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.x0 = x0_pixel if x0_pixel.dim() == 4 else x0_pixel[None]
        self.y_src, self.y_tgt = int(y_src), (None if y_tgt is None else int(y_tgt))
        self.to_pixel = to_pixel or (lambda x: x)
        self.core = TrackerCore(budgets, max_queries)
        self.best: Optional[Any] = None  # pixel tensor (1,C,H,W) of the current best adversarial example
        self.snap: Dict[int, Optional[Any]] = {}
        self._ncls = 1000
        if init_adv_pixel is not None:
            x = init_adv_pixel if init_adv_pixel.dim() == 4 else init_adv_pixel[None]
            self.best = x.detach().float().clone()
            self.core.set_initial(float((self.best - self.x0).flatten().norm()))

    # ---- core query -------------------------------------------------------------------------------------
    def labels(self, x: Any) -> Any:
        """Counted query of a batch in the attack's own input space; returns int64 labels (B,)."""
        torch = self.torch
        squeeze = x.dim() == 3
        xb = x[None] if squeeze else x
        n_req = xb.shape[0]
        n_ok = min(n_req, self.core.remaining)
        if n_ok == 0:
            raise BudgetExhausted(f"budget {self.core.max_queries} spent")
        xb = xb[:n_ok]
        with torch.no_grad():
            logits = self.model(xb)
            self._ncls = int(logits.shape[1])
            lab = logits.argmax(dim=1)
            pix = self.to_pixel(xb).float()
            l2 = (pix - self.x0.to(pix.device)).flatten(1).norm(dim=1)
            adv = (lab == self.y_tgt) if self.y_tgt is not None else (lab != self.y_src)
        _, events = self.core.process(l2.cpu().numpy(), adv.cpu().numpy())
        for kind, v in events:
            if kind == "improve":
                self.best = pix[v: v + 1].detach().clone()
            else:
                self.snap[v] = self.best
        if n_ok < n_req:
            raise BudgetExhausted(f"budget {self.core.max_queries} spent inside a batch of {n_req}")
        return lab[0:1] if squeeze else lab

    # ---- views for different official code bases --------------------------------------------------------
    def __call__(self, x: Any) -> Any:
        """torch-module style: returns one-hot float scores (argmax identical, nothing else leaks)."""
        torch = self.torch
        lab = self.labels(x)
        out = torch.zeros(lab.shape[0], self._ncls, device=lab.device)
        out[torch.arange(lab.shape[0]), lab] = 1.0
        return out

    forward = __call__

    def predict_label(self, x: Any) -> Any:
        """Sign-OPT / attackbox style API."""
        return self.labels(x)

    def predict_numpy(self, x_np: np.ndarray, channels_last: bool = False) -> np.ndarray:
        """numpy style (e.g. HSJA reference code): input (B,H,W,C) or (B,C,H,W) array -> one-hot numpy."""
        torch = self.torch
        dev = self.x0.device
        x = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(dev)
        if channels_last:
            x = x.permute(0, 3, 1, 2)
        return self(x).cpu().numpy()

    @property
    def queries(self) -> int:
        return self.core.queries

    # ---- results ------------------------------------------------------------------------------------------
    def finalize(self, store_root: Optional[str], arch: str, image: str, method: str, mode: str,
                 eval_model: Any = None) -> List[dict]:
        """Rows for per_checkpoint CSV + advstore files.

        eval_model: optional callable on PIXEL tensors in [0, 1] returning logits (UNCOUNTED, evaluation only).
        It checks whether the uint8-rounded state is still adversarial; the attack never sees it."""
        torch = self.torch
        for b in self.core.carry_forward():
            self.snap[b] = self.best
        rows = []
        for b in self.core.budgets:
            l2, q = self.core.recorded.get(b, (float("inf"), -1))
            x = self.snap.get(b)
            row = {"arch": arch, "mode": mode, "image": image, "method": method, "budget": b,
                   "found": int(x is not None), "state_query": q, "pix_l2": l2 if x is not None else float("nan"),
                   "adv_float": int(x is not None), "adv_uint8": -1}
            if x is not None:
                row["pix_linf"] = float((x - self.x0.to(x.device)).abs().max())
                if eval_model is not None:
                    with torch.no_grad():
                        lab = int(eval_model(torch.round(x * 255.0) / 255.0).argmax(1))
                    row["adv_uint8"] = int(lab == self.y_tgt if self.y_tgt is not None else lab != self.y_src)
                if store_root:
                    from advstore import store_adv

                    store_adv(store_root, arch, image, method, b, x, self.x0, state_query=q, adv_flag=1, mode=mode)
            rows.append(row)
        return rows

    def save_trace(self, path: str) -> None:
        """Compact best-L2 trace for curves at any budget: arrays q (int) and l2 (float)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        t = np.asarray(self.core.trace, dtype=np.float64).reshape(-1, 2)
        np.savez_compressed(path, q=t[:, 0].astype(np.int64), l2=t[:, 1], queries=np.int64(self.core.queries),
                            max_queries=np.int64(self.core.max_queries))
