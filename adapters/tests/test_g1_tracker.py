"""G1 gate: HardLabelTracker bookkeeping through the torch layer, with a dummy linear model (CPU, no GPU).

Covers the G1 criteria of skill lgc-baselines Section 5:
  * one-hot output only (no logits/probabilities leak);
  * every query counted (batches element-wise);
  * budget enforced inside a batch -> BudgetExhausted;
  * checkpoint = best adversarial seen AT that query (no back-fill);
  * carry-forward when the attack stops early (no back-fill of smaller budgets);
  * targeted start at query 0 (the target image is the best at query 0);
  * finalize() writes contract-shaped rows + adv_store npz.

Run:  python adapters/tests/test_g1_tracker.py   (exit 0 iff all pass)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "eval"))

from hardlabel_tracker import BudgetExhausted, HardLabelTracker  # noqa: E402

RESULTS = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


class SumModel(torch.nn.Module):
    """Decision = argmax([s, -s, 0]) with s = sum of pixels -> label 0 (s>0), 1 (s<0)."""

    def forward(self, x):
        s = x.flatten(1).sum(dim=1, keepdim=True)
        return torch.cat([s, -s, torch.zeros_like(s)], dim=1)


def l2(x, x0):
    return float((x - x0).flatten().norm().item())


def test_onehot_and_counting() -> None:
    print("[1] one-hot output, query counting, budget inside a batch, BudgetExhausted")
    model = SumModel()
    x0 = torch.full((1, 3, 2, 2), 0.5)  # s=6 -> label 0
    with torch.no_grad():
        y_src = int(model(x0).argmax(1).item())
    tr = HardLabelTracker(model, x0, y_src, None, budgets=[3, 5, 10], max_queries=10)

    out = tr(torch.rand(4, 3, 2, 2))
    vals = set(np.round(out.detach().cpu().numpy().reshape(-1), 6).tolist())
    check("one-hot output: values in {0,1}", vals <= {0.0, 1.0})
    check("one-hot output: row sum == 1", bool((out.sum(1) == 1).all()))
    check("queries counted (4)", tr.queries == 4)
    lab = tr.predict_label(torch.rand(1, 3, 2, 2))
    check("predict_label returns (B,) int labels", lab.shape == (1,) and lab.dtype in (torch.int64, torch.long))

    raised = False
    try:
        tr(torch.rand(20, 3, 2, 2))  # 20 asked, 6 remaining
    except BudgetExhausted:
        raised = True
    check("budget enforced inside a batch -> BudgetExhausted", raised)
    check("queries capped at budget 10", tr.queries == 10)

    tr2 = HardLabelTracker(model, x0, y_src, None, budgets=[3], max_queries=3)
    tr2(torch.rand(3, 3, 2, 2))
    raised = False
    try:
        tr2(torch.rand(1, 3, 2, 2))
    except BudgetExhausted:
        raised = True
    check("BudgetExhausted when no budget remains", raised)


def test_checkpoint_best_and_carry_forward() -> None:
    print("[2] checkpoint = best at that query; carry-forward; no back-fill")
    model = SumModel()
    x0 = torch.full((1, 3, 2, 2), 0.5)  # label 0
    with torch.no_grad():
        y_src = int(model(x0).argmax(1).item())
    # points: c<0 are adversarial (label 1); L2 = sqrt(12)*(0.5-c), smaller c -> larger L2
    pts = [0.30, -0.40, -0.10, -0.05]  # adv only at indices 1,2,3
    tr = HardLabelTracker(model, x0, y_src, None, budgets=[1, 2, 3, 4], max_queries=4)
    for c in pts:
        try:
            tr(torch.full((1, 3, 2, 2), c))
        except BudgetExhausted:
            break
    rec = tr.core.recorded
    check("checkpoint 1 has no adversarial (found=0 -> inf)", rec[1][0] == float("inf"))
    check("checkpoint 2 = best at q2 (c=-0.4)", abs(rec[2][0] - l2(torch.full((1, 3, 2, 2), -0.40), x0)) < 1e-6)
    check("checkpoint 3 = best at q3 (c=-0.1), not the earlier larger L2",
          abs(rec[3][0] - l2(torch.full((1, 3, 2, 2), -0.10), x0)) < 1e-6)
    check("checkpoint 4 = best at q4 (c=-0.05)",
          abs(rec[4][0] - l2(torch.full((1, 3, 2, 2), -0.05), x0)) < 1e-6)

    # early stop + carry-forward
    tr2 = HardLabelTracker(model, x0, y_src, None, budgets=[2, 4, 8], max_queries=8)
    for c in [0.30, -0.40, -0.10]:  # stops after 3 queries
        tr2(torch.full((1, 3, 2, 2), c))
    rows = tr2.finalize(None, "toy", "img", "M", "non_targeted")
    by_b = {r["budget"]: r for r in rows}
    check("carry-forward: budget 4 and 8 inherit the final best (q3)",
          by_b[4]["pix_l2"] == by_b[8]["pix_l2"] == rec[3][0])
    check("no back-fill: budget 2 keeps q2 (c=-0.4)", by_b[2]["pix_l2"] == rec[2][0])
    check("finalize state_query <= budget", all(r["state_query"] <= r["budget"] for r in rows))
    bs = sorted(by_b)
    check("finalize pix_l2 monotone non-increasing over budgets",
          all(by_b[bs[i]]["pix_l2"] >= by_b[bs[i + 1]]["pix_l2"] for i in range(len(bs) - 1)
              if not np.isnan(by_b[bs[i]]["pix_l2"]) and not np.isnan(by_b[bs[i + 1]]["pix_l2"])))


def test_targeted_start_and_store() -> None:
    print("[3] targeted start at query 0; adv_store npz written at every checkpoint")
    model = SumModel()
    x0 = torch.full((1, 3, 2, 2), 0.5)  # label 0
    x_tgt = torch.full((1, 3, 2, 2), -0.30)  # label 1 (adversarial / target)
    with torch.no_grad():
        y_src = int(model(x0).argmax(1).item())
        y_tgt = int(model(x_tgt).argmax(1).item())
    assert y_tgt != y_src
    tr = HardLabelTracker(model, x0, y_src, y_tgt, budgets=[1, 2, 3], max_queries=3,
                          init_adv_pixel=x_tgt)
    check("targeted start: queries == 0 before any attack", tr.queries == 0)
    check("targeted start: best_l2 == ||xt - x0|| at query 0", abs(tr.core.best_l2 - l2(x_tgt, x0)) < 1e-6)
    check("targeted start: best_query == 0", tr.core.best_query == 0)

    tr(torch.full((1, 3, 2, 2), 0.30))  # label 0, not target -> not adversarial
    with tempfile.TemporaryDirectory() as d:
        rows = tr.finalize(os.path.join(d, "adv_store"), "toy", "img", "M", "targeted")
        for r in rows:
            npz = os.path.join(d, "adv_store", "toy", "img", f"M_Q{r['budget']}.npz")
            check(f"adv_store npz written at Q{r['budget']} (found={r['found']})",
                  r["found"] == 1 and os.path.exists(npz))
    check("found=1 at every checkpoint (targeted start carries forward)", all(r["found"] == 1 for r in rows))


def main() -> int:
    test_onehot_and_counting()
    test_checkpoint_best_and_carry_forward()
    test_targeted_start_and_store()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
