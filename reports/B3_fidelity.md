# B3 — Hard-label audit (G2) + fidelity (G3)

Status: G2 PASS, G3 fidelity PASS for TtBA / HSJA / ARS-OPT / Sign-OPT and for the CGBA cells that run;
**BLOCKED on CGBA non_targeted** by a pre-existing crash in the official code (see §4). Worktree
`/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD `f255b7cbb949161290373343b3430e64b42fe7bd`.
Victim: ResNet-50 IMAGENET1K_V2, `resnet50-11ad3fa6.pth` sha256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
All runs GPU: RTX 5090 D (32 GB), driver 595.71.05.

## 1. G2 — hard-label audit (1 image per method, non_targeted, split row 1)

Harness `adapters/fidelity.py --gate g2` (LogitRecorder + one-hot probe + short-run official counter).

| method | one-hot only | tracker_q (full) | official_q (short) | recorder_q (short) | budget ≤10000 | adv_uint8 recorded |
|---|---|---|---|---|---|---|
| CGBA | PASS | 10000 | 223 | 224 | PASS | 1000:0, 5000:1, 10000:0 |
| HSJA | PASS | 10000 | — (no counter) | 333 | PASS | 1000:0, 5000:0, 10000:1 |
| ARS-OPT | PASS | 10000 | 501 | 501 | PASS | 1000:0, 5000:0, 10000:0 |
| Sign-OPT | PASS | 10000 | 492 | 506 | PASS | 1000:0, 5000:0, 10000:0 |
| TtBA | PASS | 10000 | — | — | PASS | (via adapter `per_checkpoint` adv_uint8) |

**No query is ever issued after 10 000** (the tracker raises `BudgetExhausted` at exactly 10 000; every
`state_query ≤ 10000`). The attack receives only the top-1 decision (tracker returns one-hot / labels; the probe
confirms values ∈ {0,1} and row-sum 1).

**Tracker count vs official counter (documented differences):**
- **ARS-OPT** — exact match (official `query` counter == recorder count = 501). ✅
- **CGBA** — tracker = official `all_queries` **+ 1**: the constructor's `src_lbl` evaluation
  (`proposed_attack.py:19`; +`tar_lbl` at `:22` in targeted) is a victim query the official counter never adds.
- **HSJA** — no internal total counter (`hsja.py:43-46`); the tracker is the authoritative count.
- **Sign-OPT** — tracker = official `query_count` **+ 14**: the init check (`Sign_OPT.py:65`), the final target
  check (`:183`) and the `fine_grained_binary_search` early-return branch (`:347-349`, where `nquery += 1` sits
  inside the `== y0` arm) are victim queries the official counter drops.
- **TtBA** — tracker = official `Attacker.queries` + ~140-190: the `__init__` `x0_label` (`TtBA.py:23`), the final
  `out_label` (`:245`) and the interrupted final iteration (the official checks the budget only after an iteration,
  so it overshoots to ~10100; our tracker stops at exactly 10000). The official counter misses these; the tracker
  is the faithful "every evaluation is one query" count.

All differences are the *hard-label accounting rule* working as intended (rule 2, U7): the tracker counts every
victim evaluation, the official counters skip some.

## 2. G3 — fidelity (split rows 1-3, both modes)

Harness `adapters/fidelity.py --gate g3`. For each cell the OFFICIAL attack runs twice on the same tensors and
seed: once with a logits-returning `LogitRecorder` (official path), once with the decision-only tracker (adapter).
Because every official attack reads only argmax / `predict_label`, a correct adapter gives ratio **1.0000**.

### 2a. HSJA / ARS-OPT / Sign-OPT (all ratio 1.0000 — adapter bit-faithful)

HSJA (off == ad at 1k/5k/10k; `off_q=None`, `trk_q=10000`):
| mode | image | L2 @1k | L2 @5k | L2 @10k |
|---|---|---|---|---|
| NT | ...00007450 | 93.093 / 93.093 | 93.093 / 93.093 | 92.416 / 92.416 |
| NT | ...00015433 | 90.846 / 90.846 | 90.846 / 90.846 | 90.846 / 90.846 |
| NT | ...00025151 | 89.833 / 89.833 | 89.833 / 89.833 | 89.833 / 89.833 |
| T  | ...00007450 | 45.394 / 45.394 | 45.394 / 45.394 | 45.394 / 45.394 |
| T  | ...00015433 | 103.352 / 103.352 | 101.646 / 101.646 | 100.269 / 100.269 |
| T  | ...00025151 | 49.351 / 49.351 | 47.249 / 47.249 | 46.841 / 46.841 |

ARS-OPT (off == ad):
| mode | image | L2 @1k | L2 @5k | L2 @10k |
|---|---|---|---|---|
| NT | ...00007450 | 97.735 | 40.670 | 27.351 |
| NT | ...00015433 | 59.500 | 16.670 | 15.131 |
| NT | ...00025151 | 88.687 | 50.527 | 36.048 |
| T  | ...00007450 | 38.156 | 26.396 | 22.188 |
| T  | ...00015433 | 96.038 | 58.028 | 36.169 |
| T  | ...00025151 | 42.866 | 26.150 | 19.753 |

Sign-OPT (untargeted only; off == ad):
| mode | image | L2 @1k | L2 @5k | L2 @10k |
|---|---|---|---|---|
| NT | ...00007450 | 87.444 | 45.572 | 28.895 |
| NT | ...00015433 | 73.783 | 7.363 | 4.738 |
| NT | ...00025151 | 89.218 | 40.612 | 29.119 |

### 2b. CGBA (targeted 3/3 pass; non_targeted 2/3 crash — see §4)

Targeted (ratio 1.0000): ...00007450 `42.265 / 40.080 / 39.607`; ...00015433 `95.397 / 88.499 / 83.769`;
...00025151 `43.864 / 37.738 / 36.059`.
Non_targeted ...00007450: `79.480 / 67.022 / 57.802` (ratio 1.0000). ...00015433 and ...00025151: **AssertionError** (§4).

### 2c. TtBA (iterate bit-identical; best-so-far ratio within band)

The adapter records both the official iterate (`pix_l2_iter`, == official `pix_l2_reported` to <1e-6, all 6 cells)
and the tracker best-so-far (`pix_l2`). `official L2` = official clamped iterate, `adapter L2` = tracker best-so-far.

| mode | image | off@1k / ad@1k (r) | off@5k / ad@5k (r) | off@10k / ad@10k (r) |
|---|---|---|---|---|
| NT | ...00007450 | 50.92 / 50.79 (0.997) | 22.96 / 22.96 (1.000) | 11.83 / 11.83 (1.000) |
| NT | ...00015433 | 53.22 / 52.00 (0.977) | 27.38 / 27.38 (1.000) | 13.24 / 13.24 (1.000) |
| NT | ...00025151 | 39.88 / 35.56 (0.892) | 12.93 / 12.93 (1.000) | 7.76 / 7.76 (1.000) |
| T  | ...00007450 | 36.19 / 36.19 (1.000) | 24.61 / 24.61 (1.000) | 16.42 / 16.42 (1.000) |
| T  | ...00015433 | 89.77 / 89.77 (1.000) | 72.01 / 72.01 (1.000) | 56.30 / 56.30 (1.000) |
| T  | ...00025151 | 32.48 / 32.48 (1.000) | 16.41 / 16.40 (1.000) | 10.25 / 10.25 (1.000) |

Median ratio per budget: NT {1k: 0.977, 5k: 1.000, 10k: 1.000}, T {1.000, 1.000, 1.000} — all inside [0.90, 1.10].
The one cell at 0.892 (NT ...00025151 @1k) is the documented D2 convention: the tracker records the best-so-far
(min L2 over all adversarial queries), which is ≤ the official *iterate* and oscillates more at 1k. It is not a
fidelity defect — the adapter's `pix_l2_iter` equals the official iterate bit-for-bit.

## 3. G3 verdict

- **TtBA, HSJA, ARS-OPT, Sign-OPT: PASS.** Every ratio = 1.0000 (median), TtBA iterate bit-identical.
- **CGBA targeted: PASS** (3/3 ratio 1.0000).
- **CGBA non_targeted: BLOCKED** — 1/3 ratio 1.0000, 2/3 crash (§4). Not a fidelity-ratio failure: the adapter
  reproduces the official crash identically.

## 4. Blocking finding — pre-existing crash in official CGBA code

`proposed_attack.py` `go_to_boundary_CGBA` (and the sibling `go_to_boundary_CGBA_H`) return the wrong query count
on the 40-iteration cap:

```python
def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
    num_calls = 1
    while True:
        ...
        if self.is_adversarial(p_near_boundary) == -1:   # <-- each call = +1 query
            break
        num_calls += 1
        if num_calls >= 40:
            return x_b, num_calls                        # <-- returns 40 after only 39 is_adversarial calls
```

`num_calls` starts at 1, so the cap is reached after **39** `is_adversarial` calls but the early return reports
**40**; the official sanity check `assert self.all_queries == q_num` (`proposed_attack.py:252`) then fails
(observed at iteration 15: `all_queries=1903, q_num=1904`). The boundary walk returns `x_b` unchanged in this case,
so the *attack trajectory / L2 is unaffected* — only the assert fires, crashing the run on ~2 of 3 non_targeted
images in our sample (large initial boundary → the 40-cap is hit). The official drivers
(`Non_targeted_attack.py`/`Targeted_attack.py`) run both `attack_methods = ['CGBA_H', 'CGBA']`, so they would crash
identically; this is not caused by our tracker (the tracker and the logits recorder crash at the same assert).

**This is a "STOP and report" per skill lgc-baselines §4** (official code cannot run unchanged). Options for the
user (do NOT proceed to B4/B5 until decided):
1. Run CGBA with asserts disabled (`python -O`) — the algorithm/L2 is unaffected (the cap branch returns `x_b`
   unchanged); document it as an evaluation-only deviation.
2. Accept CGBA non_targeted failures on the affected images and report them (D6).
3. Switch the non_targeted cell to `CGBA_H` (also buggy on the cap, but may be hit less often) — changes the method.

## 5. Files created this phase

| file | sha256 |
|---|---|
| `adapters/fidelity.py` (G2/G3 harness, LogitRecorder) | `76433a43622709a6433cb64d35d732160ecd859d56b7d50aa62c53bc96c78f86` |
| `Results/b3_ttba_official/{mode}/ttbagh_per_checkpoint.csv`, `ttbagh_runs.csv` | official TtBA, rows 1-3 |
| `Results/b3_ttba_adapter/{mode}/per_checkpoint_TtBA.csv`, `runs_TtBA.csv`, `adv_store/`, `traces/` | adapter TtBA, rows 1-3 |

G2/G3 for CGBA/HSJA/ARS-OPT/Sign-OPT were run in-memory (no result files written; the fidelity harness is
evaluation-only). No adapter or `tools/eval` file changed in this phase.

## 6. Estimated cost for next phase

B4 smoke (3 images × 3 victims × 2 modes per method, seconds/image + peak memory). CGBA is blocked by §4 until the
user decides. Measured solo seconds/image on RTX 5090 / ResNet-50 at 10 000 queries: CGBA ≈ 40 s, HSJA ≈ 96 s,
ARS-OPT ≈ 33 s, Sign-OPT ≈ 125 s, TtBA ≈ 47 s.

---
**Plain-English summary (5 lines).**
1. Ran the G2 hard-label audit: every baseline receives only the top-1 decision, no query is issued past 10 000,
   and uint8 validity is recorded for all five methods.
2. Ran the G3 fidelity gate on the first three images, both modes: TtBA, HSJA, ARS-OPT and Sign-OPT reproduce the
   official attack bit-for-bit (L2 ratio 1.0000), and CGBA targeted passes 3/3.
3. Found that official CGBA crashes on 2 of 3 non-targeted images with an assertion, caused by an off-by-one query
   count in its own boundary-walk cap — a pre-existing bug in the published code, not our wrapper.
4. The tracker's query counts differ from the official counters by small, explainable amounts (CGBA +1, Sign-OPT
   +14, TtBA ~+150) because the tracker counts every victim evaluation the official code skips.
5. CGBA non-targeted cannot proceed to the smoke/full runs until you choose how to handle that official-code bug
   (disable asserts, accept failures, or switch the method).
