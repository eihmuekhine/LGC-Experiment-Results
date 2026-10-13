# B4e2 follow-up — address the `checker_review.md` findings (F1–F3)

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`. Base for the review was
`e9ff7da`, the B4e2 close HEAD was `aee5027`; this follow-up is committed as the **new HEAD** on top of `aee5027`
(not `e9ff7da`).

## 1. Scope

Phase B4e2 follow-up: fix the T6 resume-misalignment (`F1`), correct `reports/B4e2_close.md` (`F2`), and re-verify
the diagnostics and the smoke gate (`F3`). Findings source: `reports/raw/B4e2/checker_review.md` (the independent
B4e2 review).

- Prompt hash (sha256 of `reports/raw/B4e2/prompt_followup.txt`):
  `4dd0b63642e6b5af1c4459e274a9f5e99fd54d1b987808c6acb3d869039a1f44`
- Contract `paper_contract.yaml` (v1.8) sha256:
  `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`
- Engine sha256 (frozen, untouched): `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a`

Allowed-list compliance: only `adapters/run_cgba.py`, `adapters/diag/*` and `reports/*` were edited. No forbidden
file (`lgc_bench/*`, `run_benchmark_33b.py`, `run_frame_b1.sh`, official clones, the contract, prereg) was touched.

## 2. Environment

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU) |
| driver | 580.105.08 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| OMP_NUM_THREADS | 8 (every command; also `MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`) |

## 3. F1 — fix the T6 resume-misalignment (`adapters/run_cgba.py`)

The bug (`checker_review.md` finding 3): `_write_clip_events` merged `_CLIP_EVENTS` by row index, so on a resumed
CGBA/CGBA_H run the done images (skipped without appending to `_CLIP_EVENTS`) shifted the counts onto the wrong rows.

Fix (observation-only, column name and counter semantics unchanged):
- Added `_existing_stems(out_dir, mode, method)` — snapshots the image stems already present in `runs_<method>.csv`
  **before** `C.run`.
- `main()` now captures `pre_existing = _existing_stems(...)` before `C.run` and passes it to `_write_clip_events`.
- `_write_clip_events(..., pre_existing)` assigns the `_CLIP_EVENTS` entries **only to the NEW rows** (rows whose
  `image` stem is not in `pre_existing`), in CSV/processing order; pre-existing rows' `clip_events_182` are left
  untouched. No attack code, counter semantics, or diag assertion changed.

### F1(a) — fresh re-run CGBA resnet50 NT rows 1–3

Command (log `reports/raw/B4e2/F1.log`):
```
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python adapters/run_cgba.py \
  --arch resnet50 --mode non_targeted --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
  --repo /root/autodl-tmp/baselines/CGBA --out_dir Results/b4e2_f1_cgba_resnet50 --limit 3 \
  --seed_base 0 --cgba_variant fixed
```

Result vs `Results/b4e2_smoke_cgba_resnet50/non_targeted`:

| check | result |
|---|---|
| `per_checkpoint_CGBA.csv` rows | 24/24 (identical) |
| numeric diffs (`budget,found,state_query,pix_l2,pix_linf`) | 0 |
| any column diff across all 24 rows | 0 |
| npz `delta` files | 24/24, bit-equal |
| `clip_events_182` (runs_CGBA.csv) | 0, 0, 0 |

**VERDICT: PASS** — numerics IDENTICAL and `clip_events_182 = 0,0,0`.

### F1(b) — CPU sandbox test (`adapters/diag/diag_b4e2_t6_resume.py`, new)

Builds a `runs_CGBA.csv` with one pre-existing row (image A, status ok, no `clip_events_182`), snapshots via
`_existing_stems`, appends two new rows (B, C), merges counts `[7, 13]`, and asserts A is untouched and B/C receive
the counts in order.

```
OMP_NUM_THREADS=8 python adapters/diag/diag_b4e2_t6_resume.py
```

Result: 5/5 PASS (A untouched, B=7, C=13, snapshot={A}, column name preserved). **VERDICT: PASS.**

## 4. F2 — `reports/B4e2_close.md` edits

- (i) T6: replaced the unsupported "Printed counts: …" with the exact evidence path
  `Results/b4e2_t6_cgba_resnet50/non_targeted/runs_CGBA.csv` (column `clip_events_182`, values 0,0,0).
- (ii) Deviations: added deviation §6 — `adapters/_common.py:421-425` `done_images` still filters
  `status in ("ok", "budget")` for CGBA/CGBA_H/CGBA_D8only and ARS_OPT, while T5 fixed
  `ttba_github/ttba_common.py:83`; the two resume policies diverge and the binding policy for B5 is a **user
  decision** (contract v1.9).
- (iii) Copied `/tmp/b4e2_gate.md` → `reports/raw/B4e2/T3_gate.md` (committed); committed
  `reports/human_preflight_b4e2.md` (FLAG lines = 7) and referenced both in §1.
- (iv) §5: added the sha256 of every `reports/raw/B4e2/` log; stated the report is committed at the new HEAD
  (`aee5027` close → follow-up), not `e9ff7da`.
- (v) T6: noted `CGBA_D8only`'s `clip_events_182` is constant-0 by design (fixed-only counter); the E17 NaN-share
  measurement is a separate B4g/B5 item.

## 5. F3 — re-verify

| command | result |
|---|---|
| `python adapters/diag/diag_b4e_t1_clip_source.py` | PASS (clamp = only value-changing edit; d8only inherits official) |
| `python adapters/diag/diag_b4e_t5_done_images.py` | PASS (5/5) |
| `python adapters/diag/diag_b4e2_t6_resume.py` | PASS (5/5) |
| `python tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21) |
| `python tools/eval/selftest_evalkit.py` | ALL PASS (20/20) |
| `python tools/eval/check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --roots $(cat reports/B4e2_smoke_roots.txt) --report /tmp/b4e2_followup_gate.md` | **exit 0** (I01–I12 + DISK all PASS) |

## 6. Findings table (fixed / deferred / evidence)

| # | checker_review finding | status | evidence |
|---|---|---|---|
| 1 | T6 "Printed counts" not present in T6.log | **fixed** | B4e2_close.md T6 now cites `runs_CGBA.csv` (0/0/0); values verified in `Results/b4e2_t6_cgba_resnet50` |
| 2 | `done_images` policy divergence (`_common.py` vs `ttba_common.py`) | **deferred** (user decision, contract v1.9) | B4e2_close.md deviation §6 |
| 3a | `_write_clip_events` positional merge misaligned after resume | **fixed** | `run_cgba.py` stem-keyed merge; `diag_b4e2_t6_resume.py` PASS; F1(a) re-run identical (24/24 rows, 24 npz bit-equal) |
| 3b | `CGBA_D8only` `clip_events_182` constant-0 | **noted** (by design) | B4e2_close.md T6 note; E17 NaN-share is B4g/B5 |
| 4 | T3 gate report + `human_preflight_b4e2.md` uncommitted | **fixed** | `reports/raw/B4e2/T3_gate.md` committed; `human_preflight_b4e2.md` committed (FLAG=7) |
| 5 | §5 missing log sha256; stale HEAD `e9ff7da` | **fixed** | log sha256 table added; HEAD note added |

## 7. sha256 of created / changed files; GPU minutes

| file | change | sha256 |
|---|---|---|
| `adapters/run_cgba.py` | F1: resume-safe stem-keyed clip_events_182 merge | `d3c83758e7cb18b73b7d169d84806a26b8d4b4421413cb8571d3e010277670a3` |
| `adapters/diag/diag_b4e2_t6_resume.py` | new (F1(b) sandbox test) | `ad4fc9f26ff047ba7b18869d54f9437a6091b192530dda0d55de26252481ab92` |
| `reports/B4e2_close.md` | F2 edits (i–v) | `10cda7cd23de65a8f94ed7c7df614978f34c58e875af1d7a6ed98ed072a8e4b8` |
| `reports/B4e2_followup.md` | this report | (see commit) |
| `reports/human_preflight_b4e2.md` | committed (FLAG=7) | `53ee56641abde7a9eca7dd80f9dc0b19a3e87a52a0637753e3d92c7c3d8ca007` |
| `reports/raw/B4e2/T3_gate.md` | copied from `/tmp/b4e2_gate.md` | `031fbe9f1dd15d4822d7a9e71304ac8395eba8989094e22892bb68bc0b904880` |
| `reports/raw/B4e2/F1.log` | F1(a) GPU log | `1450dc525bf19ce8ec8752be5c50b2603ae182ebaeec76faa211351280f8094c` |
| `reports/raw/B4e2/prompt_followup.txt` | this prompt | `4dd0b63642e6b5af1c4459e274a9f5e99fd54d1b987808c6acb3d869039a1f44` |
| `reports/raw/B4e2/checker_review.md` | findings source | `822a7bbedbbc02fed33428a74caf479ddee44689b7e0eeb5056d88a3bc6f923d` |

GPU minutes: F1(a) CGBA resnet50 NT rows 1–3 ≈ 74 s of attack time (three ~25 s images) plus model load; F3 is
CPU-only except the smoke gate (read-only). Frozen engine untouched (`835f1933d9e5…`).

## 8. Summary (plain English)

1. F1: fixed `_write_clip_events` to attach `clip_events_182` to the NEW rows by image stem (snapshot before `C.run`),
   so a resumed CGBA run no longer misaligns the column; a CPU sandbox test and a bit-identical fresh re-run
   (24/24 rows, 24 npz) confirm the fix with `clip_events_182 = 0,0,0`.
2. F2: corrected `B4e2_close.md` — cited the actual `runs_CGBA.csv` evidence, documented the `_common.py` vs
   `ttba_common.py` resume-policy divergence as a user decision (contract v1.9), committed the T3 gate report and
   `human_preflight_b4e2.md` (FLAG=7), added every log sha256, and noted the new HEAD and the CGBA_D8only
   constant-0 counter.
3. F3: all diagnostics PASS (clip-source, done_images, resume-merge) and the contract smoke gate exits 0 with
   I01–I12 + DISK all PASS.
4. No forbidden file was edited; the frozen engine (`835f1933d9e5…`) and the contract (v1.8) are unchanged.
5. Nothing else changed; B5 was not started. STOP.
