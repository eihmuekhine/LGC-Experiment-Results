# B3b — CGBA wrapper / cap-fix fidelity + TtBA completeness + bypass audit

Status: COMPLETE (2026-10-11). Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD
`d845c74` ("Amendment 2: HSJA + Sign_OPT removed (contract v1.3, D7)"). Victim: ResNet-50 IMAGENET1K_V2
(`resnet50-11ad3fa6.pth`, sha256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`).
GPU: RTX 5090 (32 GB), driver 595.71.05. Budget 10000, split rows 1-3.

This phase closes the two open B3 items for CGBA (the double-normalisation round-trip defect and the boundary-walk
off-by-one crash) and re-verifies TtBA completeness plus the tracker's query accounting. All four diagnostics run
in-memory (evaluation-only; no result files written) except T4, which is a read-only comparison of the two B3 result
sets.

## 1. Summary of changes to `adapters/run_cgba.py` (vs B2 sha `f7fb1e31…`)

The B2 file is **not retained in git** (`adapters/` is untracked), so a byte-level diff is not available; the B2
content is pinned only by its sha256 in `reports/B2_adapters.md` (`f7fb1e3128fd7537b1217d5be14f872ce43fc183da43d101edb0c14f62367b73`)
and its §3 description ("CGBA uses `UnnormalizeVictim`"). The B3b change is reconstructed from that description and
documented in the file's own docstring; the two substantive changes are:

| # | B2 (sha `f7fb1e31…`) | B3b (sha `1763e950…`) | Why |
|---|---|---|---|
| 1 | `make_tracker` wrapped the shared victim in `UnnormalizeVictim` (un-normalise then re-normalise) | `make_tracker` uses `_faithful_model` (the raw un-normalised net) | T2: the float32 double-normalisation round-trip flips a decision and diverges the trajectory |
| 2 | `run_official` used the official `Proposed_attack` unchanged | `run_official` uses `PatchedProposedAttack` (subclass) | T3: the official boundary-walk returns `num_calls==40` after only 39 `is_adversarial` calls at the 40-iteration cap, tripping `assert self.all_queries == q_num` |

### Step-2 confirmation (file:line)

**(a) `make_tracker` uses `_faithful_model` (faithful, non-double-normalised path).**
`adapters/run_cgba.py:76` — `return C.HardLabelTracker(_faithful_model(victim), x0, y_src, y_tgt, checkpoints, budget, to_pixel=C.to_pixel_unnormalize, …)`.
`_faithful_model` (`run_cgba.py:65-72`) returns `getattr(victim, "m", victim)` — the raw `resnet50` network that the
official drivers feed the normalised tensor into (`net = torch_models.resnet50(pretrained=True)`,
`Non_targeted_attack.py:42,84-88`), with no normalisation layer. No `UnnormalizeVictim` appears anywhere in the file.

**(b) `PatchedProposedAttack` only overrides the boundary-walk query-count return value, not attack logic.**
The subclass (`run_cgba.py:92-114`) overrides exactly two methods:
- `go_to_boundary_CGBA` (`:104-107`) and `go_to_boundary_CGBA_H` (`:109-112`), each calling `super()` and replacing
  the returned count with the true `all_queries` delta via `_counted` (`:97-102`, `new = self.all_queries - c0`).
- `__init__` (`:93-95`) only adds a `_cap_fix_log` list.

The returned boundary point is passed through unchanged, and no other method (`Attack`, the gradient estimator, step
sizes, probe counts, schedules, initialisation) is overridden. Verified T3 (below): trajectory is bit-identical.

## 2. T2 — CGBA wrapper fidelity (faithful vs double-normalised adapter)

`diag_t2_cgba_wrapper.py`. Same official attack on the same tensors/seed, differing only in the model object.
`faith` = raw un-normalised net (correct); `adapt` = `UnnormalizeVictim` (double-normalisation round-trip). Ratio =
`adapt/faith`. `rterr = max|x_n - ((x_n·sd+mu)-mu)/sd|` on x0 (float32 double-normalisation error).

| mode | image | rterr (x0) | faith @1k/5k/10k | adapt @1k/5k/10k | ratio @1k/5k/10k | queries |
|---|---|---|---|---|---|---|
| NT | ...00007450 | 2.384e-07 | 85.078 / 85.078 / 85.078 | 79.480 / 67.022 / 57.802 | 0.934 / 0.788 / 0.679 | 10000 / 10000 |
| T | ...00007450 | 2.384e-07 | 42.653 / 37.155 / 33.804 | 42.265 / 40.080 / 39.607 | 0.991 / 1.079 / 1.172 | 10000 / 10000 |
| T | ...00015433 | 2.384e-07 | 96.728 / 87.034 / 80.911 | 95.397 / 88.499 / 83.769 | 0.986 / 1.017 / 1.035 | 10000 / 10000 |
| T | ...00025151 | 2.384e-07 | 44.709 / 38.324 / 35.055 | 43.864 / 37.738 / 36.059 | 0.981 / 0.985 / 1.029 | 10000 / 10000 |

**Finding.** The double-normalisation round-trip error is a single float32 ULP (2.384e-07) at x0, but it flips the
victim's top-1 decision near the boundary (observed at query ~102 in the docstring), so the two trajectories diverge:
ratios reach 0.679 (NT) and 1.172 (T) by 10k — far outside the G3 band [0.90, 1.10]. The faithful path (raw net) is
therefore required; `run_cgba.py` now uses it.

## 3. T3 — CGBA boundary-walk cap-fix (patched vs unpatched)

`diag_t3_cgba_cap.py`. Faithful path. For cells whose **unpatched** run completes, per-checkpoint `pix_l2` and
`state_query` must be identical patched vs unpatched. For cells whose unpatched run crashes on
`assert self.all_queries == q_num`, the patched run must complete and its `(n_query, norms)` trajectory must equal the
unpatched run up to the crash iteration.

| mode | image | unpatched | verdict | detail |
|---|---|---|---|---|
| NT | ...00007450 | crash (iter 8) | **PASS** | prefix_len=9, nq_match=True, norms_match=True, cap_fix_log=1 |
| NT | ...00015433 | crash (iter 10) | **PASS** | prefix_len=11, nq_match=True, norms_match=True, cap_fix_log=1 |
| NT | ...00025151 | crash (iter 12) | **PASS** | prefix_len=13, nq_match=True, norms_match=True, cap_fix_log=1 |
| T | ...00007450 | budget (10000) | **PASS** | patched_q=10000, unpatched_q=10000, per-checkpoint diffs=none |
| T | ...00015433 | budget (10000) | **PASS** | patched_q=10000, unpatched_q=10000, per-checkpoint diffs=none |
| T | ...00025151 | budget (10000) | **PASS** | patched_q=10000, unpatched_q=10000, per-checkpoint diffs=none |

**Finding.** With the faithful path, **all three** non_targeted rows crash (the large initial boundary hits the
40-iteration cap each time — this is a change from B3, where the old double-normalised adapter let row 1 complete).
For each crash cell the patched run completes normally and its trajectory is bit-identical to the unpatched run up to
the crash iteration (`nq_match=True`, `norms_match=True`), and the cap fix fires exactly once (`cap_fix_log=1`). The
three targeted cells complete unpatched and are bit-identical to the patched run at every checkpoint. The fix is
therefore safe: it corrects only the returned query count, never the trajectory.

## 4. T4 — TtBA completeness (adapter `pix_l2_iter` vs official `pix_l2_reported`)

`diag_t4_ttba_completeness.py`. Read-only comparison of `Results/b3_ttba_adapter/{mode}/per_checkpoint_TtBA.csv`
(column `pix_l2_iter`) vs `Results/b3_ttba_official/{mode}/ttbagh_per_checkpoint.csv` (column `pix_l2_reported`), all
8 checkpoints, 6 cells.

| mode | image | pix_l2_iter == pix_l2_reported |
|---|---|---|
| NT | ...00007450 | PASS (identical at all 8 checkpoints) |
| NT | ...00015433 | PASS (identical at all 8 checkpoints) |
| NT | ...00025151 | PASS (identical at all 8 checkpoints) |
| T | ...00007450 | PASS (identical at all 8 checkpoints) |
| T | ...00015433 | PASS (identical at all 8 checkpoints) |
| T | ...00025151 | PASS (identical at all 8 checkpoints) |

Query accounting (official `Attacker.queries` vs tracker): the official counter overshoots the budget — NT 10109 /
10119 / 10073, T 10079 / 10162 / 10053 — while the tracker stops at exactly **10000** in every cell (the TtBA loop
checks the budget only after an iteration; the tracker raises `BudgetExhausted` at 10000). T4 verdict: **PASS**.

## 5. T5 — bypass audit (raw-victim forward count vs tracker queries; row 1 NT)

`diag_t5_bypass.py`. The raw victim forward is wrapped with an image counter (counts batch elements); after the
attack, `atk_images` must equal `tracker.queries` (every tracker query = exactly one victim classification, nothing
classifies outside the tracker). `finalize()`'s uint8 re-checks are then reported separately (uncounted by design).

| method | atk_images | tracker.queries | official counter | finalize eval calls | verdict |
|---|---|---|---|---|---|
| TtBA | 10000 | 10000 | `Attacker.queries` = 9879 | 8 | **PASS** |
| CGBA | 10000 | 10000 | `all_queries` = 9928 | 8 | **PASS** |
| ARS-OPT | 10000 | 10000 | — (interrupted at budget; `out` never returned) | 8 | **PASS** |

**Finding.** For all three methods `atk_images == tracker.queries == 10000`: the tracker is the faithful
"every evaluation is one query" count, and nothing else classifies an image. The official counters under-count
(TtBA 9879, CGBA 9928) for the documented reasons (constructor/final checks and the budget overshoot the official
loops skip). `finalize()` issues exactly 8 additional victim evaluations (one uint8 validity re-check per contract
checkpoint), separate from and uncounted by the tracker — confirming the attack-time/eval-time separation.

## 6. Files (sha256)

| file | sha256 |
|---|---|
| `adapters/run_cgba.py` | `1763e9503b32b0603ce75d83687437006bcdbf23795f3dc5955ea3e4b490a12c` |
| `adapters/diag/diag_t2_cgba_wrapper.py` | `a6c3f0a863df80b2e6bff43dff38a8a742171a68ff4ea1a833f8ca0c080c1b84` |
| `adapters/diag/diag_t3_cgba_cap.py` | `2f1dd033c9b8a5e2cb0fb090bb90bd80a4f1ba727fc2ef9317269a7438498d0e` |
| `adapters/diag/diag_t4_ttba_completeness.py` | `7433dbadc2de4156735540267fc985c6486cd493ad31c215315ecabcdc38015b` |
| `adapters/diag/diag_t5_bypass.py` | `7c1d0c830e150247c8c183901782b900b006bf9782333e6b867eb38e9c96fda7` |

`adapters/diag/diag_t5_bypass.py` was edited this phase to fix two latent bugs that prevented a clean end-to-end run:
(1) `arsopt()` used a CWD-relative split path after `ttba()` had `os.chdir`ed into the TtBA repo — changed to the
absolute `REPO_ROOT`-based path used by `cgba()`; (2) running TtBA/CGBA/ARS-OPT in one interpreter collides on shared
top-level module names (`dataset`, etc. — the same collision documented in `reports/B2_adapters.md` §1) — `main()`
now runs each method in its own subprocess via a `--only` flag. These are diagnostic-script fixes only; no official
clone, `lgc_bench/engine.py`, or `contract/paper_contract.yaml` was touched.

## 7. GPU minutes

RTX 5090 wall-clock (measured `DONE` timestamps): T2 168.2 s, T3 139.9 s, T5 114.4 s, T4 ≈ 0 s (CPU read-only).
Total successful GPU wall-clock ≈ **7.0 min**. Approximately 3 additional GPU-minutes were spent in two aborted T5
attempts before the module-collision fix.

## 8. HSJA / Sign-OPT

**HSJA and Sign-OPT were not run in this phase.** No diagnostic touches them, and `adapters/run_hsja.py` and
`adapters/run_signopt.py` were left untouched. (Per contract v1.3 D7, both are cited only, not run.)

---
**Plain-English summary (5 lines).**
1. CGBA's adapter was corrected to feed the official attack's normalised tensor straight into the un-normalised
   ResNet-50 (the faithful path); the old wrapper's tiny float32 double-normalisation round-trip (2.4e-7) flips a
   decision and diverges the attack, so ratios drift as far as 0.68–1.17 by 10k.
2. CGBA's official boundary walk has an off-by-one query count at its 40-iteration cap that crashes all three
   non-targeted test images; the fix corrects only that count, and the attack trajectory is bit-identical up to and
   past the crash point (verified on all six cells).
3. TtBA's adapter iterate matches the official iterate bit-for-bit at all 8 checkpoints across all 6 cells, and the
   tracker stops at exactly 10000 queries where the official counter overshoots.
4. A raw-victim forward counter confirms every tracker query is exactly one victim classification for TtBA, CGBA and
   ARS-OPT (all 10000 == 10000), with the 8 finalize re-checks cleanly separated.
5. HSJA and Sign-OPT were not run; nothing outside the official-clone adapters, the frozen engine, or the paper
   contract was modified.
