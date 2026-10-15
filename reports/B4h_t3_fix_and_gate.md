# B4h-T3 — D19 oracle-domain fix (ARS_OPT + CGBA/CGBA_H), D16 TtBA done_images L4 fix, KAT regression & domain-consistency gate

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, base HEAD `e55dea1f`. Contract
v1.9. No B5/T6 runs started. Frozen engine `835f1933d9e5…` untouched; contract, prereg and official clones untouched.

## 1. Scope and prompt hash

- Prompt (saved copy by A): `reports/raw/B4h/prompt_t3.txt`, sha256
  `722affe82e4b33e82bf28c31671101e1cebb04aa8a1dbe601daf562e0584d795`.
- Contract `contract/paper_contract.yaml` **v1.9** sha256 (printed at start):
  `fc04498315dec51b0f663568d76a7804d46f63aa57eb2950a2126d157451ca32` (matches the expected value).
- Worktree HEAD: `e55dea1f32d4a31889bddf77d8ff38fcd1777572`.

## 2. Environment block

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (see Deviations §1: contract D16 names RTX PRO 6000 as the study GPU; this instance still reports 5090) |
| driver | 580.105.08 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `float32_matmul_precision=highest` |
| worktree HEAD | `e55dea1f32d4a31889bddf77d8ff38fcd1777572` |
| contract sha256 | `fc04498315dec51b0f663568d76a7804d46f63aa57eb2950a2126d157451ca32` |
| OMP/MKL/OPENBLAS | 8 (set for every GPU command) |

## 3. Step 0 — direct verification of `eval_metrics_offline.py` (closes B4h-T1b item 1)

Command: `sha256sum tools/eval/eval_metrics_offline.py && sed -n '1,140p' tools/eval/eval_metrics_offline.py`
(raw saved to `reports/raw/B4h/t3_eval_metrics_offline_head.txt`).

- `eval_metrics_offline.py` sha256: `4f83267714d2ea07e6f31c1fc3a8415ec163382614d81aec82933ad420c2bbe3`.
- Confirmed verbatim at `eval_metrics_offline.py:152`:
  `"adv_flag": int(metas[i].get("adv_flag", -1)),`
- The `evaluate()` loop only `np.load`s stored npz files and computes metrics with `torch` (SSIM/LPIPS/DISTS/etc.);
  no victim model is instantiated or queried anywhere in the file. **Verdict: PASS.**

## 4. Step 1 — D19 oracle-domain fixes + D16 L4 resume fix

### (a) `adapters/_common.py`
- Added `ClampedVictimNormalized(torch.nn.Module)` next to `ClampedVictim`: registers `lb.clone().detach()` and
  `ub.clone().detach()` as buffers; `forward(x) = self.victim(torch.clamp(x, self.lb, self.ub))`.
- `build_manifest` now forwards `"oracle_domain"` (added to the `("cgba_variant", "paper_faithful_fix", "oracle_domain")`
  tuple) so it lands in `run_manifest.json`.

### (b) `adapters/run_cgba.py`
- `_faithful_model(victim, lb, ub)` returns `C.ClampedVictimNormalized(C.raw_net(victim), lb, ub)`.
- `make_tracker` computes `lb = C.normalize(torch.zeros_like(x0))`, `ub = C.normalize(torch.ones_like(x0))` and threads
  them into `_faithful_model`. (No other diagnostic call sites of `_faithful_model` exist — grepped.)
- `extra_env` records `"oracle_domain": "clamp_normalized"`.

### (c) `adapters/run_arsopt.py`
- `make_tracker` wraps `C.ClampedVictim(victim)`.
- `extra_env` records `"oracle_domain": "clamp01"`.

### (d) `ttba_github/ttba_common.py:83` (`done_images`)
- Applied the 1-line L4 fix: `return {r["image"] for r in csv.DictReader(f) if r.get("status") in ("ok", "budget")}`.

## 5. Step 2 — CPU unit tests & selftests

New unit test `adapters/tests/test_g1_clamped_victim.py` verifies that `ClampedVictimNormalized`:
1. clamps out-of-box normalised inputs channel-wise to `[lb, ub]` (raw net receives `torch.clamp(x, lb, ub)`);
2. agrees with `C.normalize(C.to_pixel_unnormalize(x))` (the D19 identity, allclose within FP round-off);
3. passes in-range inputs through unchanged.

| command | result |
|---|---|
| `python adapters/tests/test_g1_clamped_victim.py` | ALL PASS (6/6) |
| `python tools/eval/selftest_evalkit.py` | ALL PASS (20/20) |
| `python tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21) |
| `python adapters/tests/test_g1_helpers.py` | ALL PASS (9/9) |
| `python adapters/tests/test_g1_tracker.py` | ALL PASS (22/22) |
| `python adapters/tests/test_g1_adapters.py` | ALL PASS (8/8) |

## 6. Step 3 — KAT regression

Command: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python adapters/diag/kat_linear.py`
(log `reports/raw/B4h/t3_kat_linear.log`).

| method | mode | n | (a) adversarial | (b) lower bound | (c) pix_l2==‖δ‖ | (d) queries | fwd residual | median ratio | max ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| LGC_RC_EGC | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.654 | 1.710 |
| LGC_RC_EGC | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.399 | 1.414 |
| TtBA | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.782 | 1.948 |
| TtBA | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.680 | 1.696 |
| CGBA | non_targeted | 35 | 35/35 | 35/35 | 35/35 | 5/5 | 0 | 1.649 | **2.187** |
| CGBA_H | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.709 | 1.733 |
| ARS_OPT | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 7.770 | 9.024 |
| ARS_OPT | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 5.522 | 5.918 |

**Verdict: PASS** — every (a)–(d) check passes; forward-count residual = 0. Only CGBA NT changed vs B4g (max ratio
2.186 → **2.187**): the D19 clamp altered the trajectory by one step on `kat_linear`, i.e. unclamped `bin_search` /
`SemiCircular_boundary_search` probes DID occur even here but their effect was tiny and invisible to check (a)
(L23). LGC_RC_EGC, TtBA, CGBA_H and ARS_OPT are bitwise identical to B4g (ARS_OPT is unchanged because its victim
already clamped, and `ClampedVictim` is idempotent on top of it).

## 7. Step 3 — 3-image smoke tests (18 cells)

Driven by `reports/raw/B4h/t3_smoke.log` (driver script `/tmp/opencode/t3_smoke_driver.sh`), one process at a time
(D14). Outputs in `Results/b4h_t3_smoke_<method>_<arch>` (18 cells = 3 ARS_OPT archs × 2 modes + 6 CGBA/CGBA_H
victims × 2 modes; 3 images each, rows 1–3, budget 10000).

Per-cell runs CSV summary (all 18 cells identical in shape):

| check | value |
|---|---|
| cells | 18 (6 ARS_OPT + 12 CGBA/CGBA_H) |
| images per cell | 3 |
| status | `budget` (BudgetExhausted) × 54/54; **0 error** |
| max queries | 10000 (never exceeded) |
| max nonfinite_queries | 0 |

`oracle_domain` recording (env + run_manifest): 6 × `"clamp01"` (ARS_OPT), 12 × `"clamp_normalized"`
(CGBA/CGBA_H), in both `env_*.json` and `run_manifest.json` — verified for all 18 cells.

## 8. Step 3 — domain-consistency gate

`adapters/diag/domain_consistency.py` (post-fix: X>0 for ANY method is a STOP) on all 9 new smoke roots
(log `reports/raw/B4h/t3_domain_consistency.log`, rows `reports/raw/B4h/t3_domain_consistency_rows.csv`, 432 rows):

| method | cell (arch/mode) | rows | C | Q9 | Q6 | X |
|---|---|---|---|---|---|---|
| ARS_OPT | resnet50/{nt,t} | 24+24 | 15/11 | 9/13 | 0/0 | **0/0** |
| ARS_OPT | vgg19/{nt,t} | 24+24 | 23/15 | 1/9 | 0/0 | **0/0** |
| ARS_OPT | vit_b_32/{nt,t} | 24+24 | 15/17 | 9/7 | 0/0 | **0/0** |
| CGBA/CGBA_H | resnet50/{nt,t} | 24+24 | 19/17 | 5/7 | 0/0 | **0/0** |
| CGBA/CGBA_H | vgg19/{nt,t} | 24+24 | 24/16 | 0/8 | 0/0 | **0/0** |
| CGBA/CGBA_H | vit_b_32/{nt,t} | 24+24 | 23/22 | 1/2 | 0/0 | **0/0** |
| CGBA/CGBA_H | resnet50_places365/{nt,t} | 24+24 | 22/17 | 2/7 | 0/0 | **0/0** |
| CGBA/CGBA_H | densenet161_places365/{nt,t} | 24+24 | 21/23 | 3/1 | 0/0 | **0/0** |
| CGBA/CGBA_H | Salman2020Do_R50/{nt,t} | 24+24 | 16/15 | 8/9 | 0/0 | **0/0** |

**Verdict: PASS — X = 0 for every (method, cell)**, i.e. no stored-image re-decision mismatch for ARS_OPT **or**
CGBA/CGBA_H. The Q9 (and a few Q6) rows are float16-storage effects (‖fl16(d) − d‖ ≤ 2⁻¹¹‖d‖), not defects. `adv_uint8`
validity is reported separately and is not part of this gate (uint8 rounding near the boundary flips some states, as
expected for best-so-far points).

## 9. Deviations from the prompt

1. **GPU name.** The instance reports `NVIDIA GeForce RTX 5090` (driver 580.105.08), while contract v1.9 D16 names
   `RTX PRO 6000` as the study GPU. This phase is smoke/regression tool-check level (D13: n<30 is never a gate) and
   the B4e2/B4g comparison baselines were themselves produced on RTX 5090, so I proceeded and recorded the actual GPU
   in every env file. The B5 full runs (which must run on the D16 study GPU) are not started. This is the L03
   situation (GPU changes between sessions without notice) and is flagged for the human.
2. `domain_consistency.py` was edited (allowed list `adapters/diag/*`) so that X>0 for ANY method — ARS_OPT included —
   is a STOP, reflecting the post-fix invariant; before the fix it only stopped on non-ARS_OPT X>0.
3. The prompt's Step-3 acceptance note ("Do NOT expect CGBA bitwise IDENTICAL to B4e smoke") is confirmed in the KAT
   regression as well: CGBA NT differs from B4g by one trajectory step (max ratio 2.186 → 2.187).
4. No other deviations.

## 10. sha256 of created/changed files

| file | sha256 |
|---|---|
| `adapters/_common.py` (edited) | `1167ed7b97cc53eff899ff373f83950734e6333a211e1cf7e9b337cf796c16c9` |
| `adapters/run_arsopt.py` (edited) | `0f454a78c615712b226075cf4f8bdd9c9de9ae10129740bc0f83f481967f9c6a` |
| `adapters/run_cgba.py` (edited) | `7596d79806ffc27ddd78a998b784cba0d79822ba7aec436b402b32051ae33487` |
| `ttba_github/ttba_common.py` (edited) | `efd6ab9785712fef8731ab4d45a9ceb4ca8bad363fe2ec7c6e82f1a2ceb039c6` |
| `adapters/tests/test_g1_clamped_victim.py` (new) | `6a4d97d9d1c296779a27611579d085ed40526bb1f0fe91da736566574f26cea0` |
| `adapters/diag/domain_consistency.py` (new/edited) | `28defc0cab95799f3c07d6bd80376bf5843416706f675c7fb9d2623228584d58` |
| `reports/B4g_kat_canary.md` (edited, L23 caveat) | `7ddb5eabfc59296870b9b25724fc34a19d8c607e0deeb64e8ef74bd7a08be2e7` |
| `reports/raw/B4h/prompt_t3.txt` (new) | `722affe82e4b33e82bf28c31671101e1cebb04aa8a1dbe601daf562e0584d795` |
| `reports/raw/B4h/t3_eval_metrics_offline_head.txt` | `469d577762d13b52abc547b721a2f2e202cb71fd1ff1bb57f2a60c594e63be19` |
| `reports/raw/B4h/t3_kat_linear.log` | `823d2d95ebbb0665a02c902e5a0c510169b164c52ad6edab410c959e9f12b527` |
| `reports/raw/B4h/t3_smoke.log` | `2e4cbcc3f05db38065a3f484c570b1f2220e79af679f5275e17bd2b0e2f7f3aa` |
| `reports/raw/B4h/t3_domain_consistency.log` | `63d8b2e6b7fb429aee084f7e455c474844ef178fc6ee34e0ffcd374b71c3d685` |
| `reports/raw/B4h/t3_domain_consistency_rows.csv` | `72db2cb55e5b271138263ee18c1c23db45907b3d0d2906acacc2fc61c69563b4` |

GPU minutes: KAT regression 12.2 min (40 images) + smoke 28.4 min (54 images) + domain-consistency ≈ 2 min ≈ **43 min**
total. Frozen engine untouched.

## 11. Proposed LESSONS L22 / L23 (for H to add to `docs/LESSONS.md`)

| id | date | area | severity | lesson | evidence | mitigation | status |
|---|---|---|---|---|---|---|---|
| L22 | 2026-10-16 | oracle | critical | The decision oracle and the stored/scored image must be the same tensor. ARS_OPT queried unclamped [0,1] pixels and CGBA/CGBA_H queried unclamped ImageNet-normalised points (bin_search / SemiCircular_boundary_search never clip), while the tracker stored the clamped pixel projection — so an out-of-range probe's decision (unclamped) disagreed with the stored image (clamped). | B4h-T3 D19; domain_consistency X>0 pre-fix | ClampedVictim (clamp01) / ClampedVictimNormalized (clamp_normalized); `oracle_domain` in env+manifest; domain_consistency gate (X=0) | proposed |
| L23 | 2026-10-16 | protocol | major | The B4g kat_linear known-answer test did NOT catch D19: check (a) used a victim that clamps its own [0,1] input, and the CGBA faithful path fed a linear map directly, so the unclamped bin_search/SemiCircular_boundary_search queries emitted against real victims were never exercised — the defect is invisible on a victim that masks it. A known-answer test must use a non-clamping victim, or assert the adapter's query inputs stay in range. | B4h-T3 KAT regression (CGBA NT 2.186→2.187 only) | domain_consistency.py on real victims as the gate; KAT caveat recorded in B4g | proposed |

## 12. Summary (plain English)

1. Verified the offline metric script reads `adv_flag` from stored metadata without touching any victim (Step 0).
2. Implemented the D19 oracle-domain fix for both methods — CGBA/CGBA_H now clamps every normalised-space query to the
   exact [0,1] pixel domain, ARS_OPT clamps every pixel query to [0,1] — and recorded `oracle_domain` in every env
   file and manifest; applied the 1-line TtBA `done_images` L4 resume fix (D16).
3. Added a CPU unit test for the new clamp wrapper; all test suites pass (selftest 20/20, contract 21/21, adapters).
4. Re-ran the KAT known-answer test (all (a)–(d) PASS; only CGBA NT moved one step, matching the L23 caveat) and ran
   18 three-image smoke cells (ARS_OPT 6, CGBA/CGBA_H 12) with zero errors, zero non-finite queries, budget honoured.
5. The domain-consistency gate is green: X = 0 for every method and cell — ARS_OPT and CGBA/CGBA_H now decide on the
   exact image that is stored and scored. Committed; no B5/T6 started.
