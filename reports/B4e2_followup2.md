# B4e2 follow-up-2 — address the `checker_review_followup.md` findings (G1–G3)

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`. Base for this phase is the
B4e2 follow-up HEAD `a67ee7e`; this follow-up-2 is committed as the **new HEAD** on top of `a67ee7e`.

## 1. Scope

Phase B4e2 follow-up-2: fix the residual error-row-resume misalignment in the T6 `clip_events_182` merge (`G1`),
apply the required report corrections to `reports/B4e2_followup.md` (`G2`), and re-verify the diagnostics and the
smoke gate (`G3`). Findings source: `reports/raw/B4e2/checker_review_followup.md` (the independent re-review of
the B4e2 follow-up).

- Prompt hash (sha256 of `reports/raw/B4e2/prompt_followup2.txt`):
  `1d7525b95ae0197160b7cb1c35a4a1c8b1a0e8dc8e95e690379e3380e4a06271`
- Contract `paper_contract.yaml` (v1.8) sha256:
  `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`
- Engine sha256 (`lgc_bench/engine.py`, frozen, untouched):
  `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a`

Allowed-list compliance: only `adapters/run_cgba.py`, `adapters/diag/*` and `reports/*` were edited. No forbidden
file (`lgc_bench/*`, `run_benchmark_33b.py`, `run_frame_b1.sh`, official clones, the contract, prereg) was touched.

## 2. Environment

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU) |
| driver | 580.105.08 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 (CUDA 12.8) |
| backend flags (D12, recorded in the G1(a) env file) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| OMP_NUM_THREADS | 8 (every command; also `MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`) |
| CGBA official repo | `/root/autodl-tmp/baselines/CGBA` @ `73d3e9c94423b1503053258494a54ad4d2892efa` |

## 3. G1 — fix the residual error-row-resume misalignment (`adapters/run_cgba.py`)

The residual bug (`checker_review_followup.md` finding 1): `_write_clip_events` keyed on image **stem** only, but
`_common.py:425` `done_images` treats only `status in ("ok","budget")` as done, so an errored image is re-attacked
and a SECOND row with the same stem is appended; the stem-keyed merge then skipped that new row and shifted its
count onto the next new image.

Fix (observation-only, column name and counter semantics unchanged; done-image behaviour F1 unchanged):
- Replaced `_existing_stems` (stem snapshot) with `_pre_existing_count` (number of rows present in
  `runs_<method>.csv` before `C.run`).
- `_write_clip_events(..., pre_existing)` now assigns the `_CLIP_EVENTS` entries to the rows **appended this run
  only** — every row at index `>= pre_existing` — in processing order.  Pre-existing rows (done `ok`/`budget`
  **and** old `error` rows) keep their `clip_events_182` untouched.  A re-attacked error image's new row is
  exactly the row at index `>= pre_existing`, so it receives its own count while the old error row stays empty.

Why row position, not the prompt's option 1 (snapshot only `ok`/`budget` stems): filtering the stem snapshot
alone is insufficient — the old `error` row remains in the CSV and would otherwise consume the first
`_CLIP_EVENTS` entry (the row has no `ok`/`budget` status to be skipped by).  Row position is equivalent to the
prompt's option 2 ("merge by occurrence for new rows only") because `C.run` appends new rows in processing order,
which matches `_CLIP_EVENTS` order, and does not require passing the stem into `run_official` (which would need a
forbidden `_common.py` edit).

### G1(a) — fresh re-run CGBA resnet50 NT rows 1–3

Command (log `reports/raw/B4e2/G1a.log`, throwaway dir `Results/b4e2_g1a_cgba_resnet50`):
```
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python adapters/run_cgba.py \
  --arch resnet50 --mode non_targeted --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
  --repo /root/autodl-tmp/baselines/CGBA --out_dir Results/b4e2_g1a_cgba_resnet50 --limit 3 \
  --seed_base 0 --cgba_variant fixed
```

Result vs `Results/b4e2_smoke_cgba_resnet50/non_targeted`:

| check | result |
|---|---|
| `per_checkpoint_CGBA.csv` rows | 24/24 (identical) |
| any field diff across all 24 rows | 0 |
| npz `delta` files | 24/24, bit-equal |
| `x0.npy` files | 3/3, bit-equal (27/27 files total) |
| `clip_events_182` (runs_CGBA.csv) | `0, 0, 0` |

**VERDICT: PASS** — numerics IDENTICAL and `clip_events_182 = 0,0,0`. Wall times 26.5/24.7/24.0 s (≈75 s attack).

### G1(b) — error-row-resume sandbox scenario (extended `diag_b4e2_t6_resume.py`)

New scenario (b): pre-existing error row `X`, then the re-attacked `X` (new row, same stem) and new row `Y`;
counts `[5, 9]`.  Asserts the old error `X` stays empty, `X`-new gets `5`, `Y` gets `9` (the reviewer's demo now
passes).

### G1(c) — done-image-resume scenario (unchanged)

Scenario (c): pre-existing ok-row `A`, new rows `B`/`C`, counts `[7, 13]`.  Asserts `A` untouched, `B=7`, `C=13`.

```
OMP_NUM_THREADS=8 python adapters/diag/diag_b4e2_t6_resume.py
```

Result: 5/5 PASS (c) + 5/5 PASS (b), exit 0. **VERDICT: PASS.**

## 4. G2 — `reports/B4e2_followup.md` corrections

- (i) The summary's first point (formerly §8.1, now §9.1 after the Deviations section was inserted): the claim is
  scoped to "so a resume over done (ok/budget) images no longer misaligns the column; an error-row resume is
  fixed in G1 above".
- (ii) Added the required **"Deviations from the prompt"** section (new §7), listing the one deviation
  (the over-qualified §8.1 claim, `checker_review_followup.md` finding 1) with the G1 evidence.
- (iii) `checker_review_followup.md` finding 1 and the G1 evidence are cited in §3 (G1 note), §7 (Deviations) and
  §9.1 (Summary).

## 5. G3 — re-verify

| command (rerun, `OMP_NUM_THREADS=8`) | result |
|---|---|
| `python adapters/diag/diag_b4e_t1_clip_source.py` | PASS (clamp = only value-changing edit; d8only inherits official), exit 0 |
| `python adapters/diag/diag_b4e_t5_done_images.py` | PASS (5/5), exit 0 |
| `python adapters/diag/diag_b4e2_t6_resume.py` | PASS (5/5 (c) + 5/5 (b)), exit 0 |
| `python tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21), exit 0 |
| `python tools/eval/selftest_evalkit.py` | ALL PASS (20/20), exit 0 |
| `python tools/eval/check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --roots $(cat reports/B4e2_smoke_roots.txt) --report /tmp/b4e2_followup2_gate.md` | **exit 0** (I01–I12 + DISK all PASS) |

Tracked `reports/contract_status.md` untouched (`git status` clean for it).

## 6. Findings table (fixed / deferred / evidence)

| # | checker_review_followup finding | status | evidence |
|---|---|---|---|
| 1 | Residual error-row-resume misalignment (stem-keyed merge skips the re-attacked error row) | **fixed** | `_pre_existing_count` row-position merge in `run_cgba.py`; `diag_b4e2_t6_resume.py` scenario (b) 5/5 (X-new=5, Y=9); F1(a) bit-identical (24/24 rows, 24 npz) |
| 2 | §9.1 summary claim over-qualified | **fixed** | §9.1 scoped to done (ok/budget) resumes; error-row case recorded as G1 |
| 3 | Missing "Deviations from the prompt" section | **fixed** | §7 Deviations section added to `B4e2_followup.md` |

No deferred items: both findings are resolved in this phase.

## 7. Deviations from the prompt

None. Every instruction of `prompt_followup2.txt` (G1 fix + (a)/(b)/(c) verification, G2 report corrections,
G3 re-verify, write `B4e2_followup2.md`, commit) was executed as specified.  One design note (not a deviation):
G1 implemented the prompt's option 2 (merge by occurrence for new rows only) via row position instead of the
option 1 stem filter, because option 1 alone would leave the old `error` row consuming a `_CLIP_EVENTS` entry
(see §3); the prompt explicitly offered both options.

## 8. sha256 of created / changed files; GPU minutes

| file | change | sha256 |
|---|---|---|
| `adapters/run_cgba.py` | G1: `_existing_stems` -> `_pre_existing_count` (row-position merge) | `c2779755a1de88ab2830cd61ba0f9edc676a341595de5da0219d85a2c8f91b51` |
| `adapters/diag/diag_b4e2_t6_resume.py` | G1(b): added error-row scenario (b) | `e5a1d8feced63a50c3d6f370a261e42c184605b838bd6b75864f6da78a4090a9` |
| `reports/B4e2_followup.md` | G2 edits (i–iii) | `ff1f0ba378e1a4a19768b48e315e10b89ca95cc1b00b6a93d89dbff69555f1cb` |
| `reports/B4e2_followup2.md` | this report | (see commit) |
| `reports/raw/B4e2/prompt_followup2.txt` | this prompt | `1d7525b95ae0197160b7cb1c35a4a1c8b1a0e8dc8e95e690379e3380e4a06271` |
| `reports/raw/B4e2/checker_review_followup.md` | findings source (committed here) | `403d9d219155de48047c1222b9a374be24a4f68481f529782c39cb609b510172` |
| `reports/raw/B4e2/G1a.log` | G1(a) GPU log | `dfa947f4c29f9b2ccacfab29ab4afaf2a2b45a88867175eb29ba5e94ce238b83` |

GPU minutes: G1(a) CGBA resnet50 NT rows 1–3 ≈ 75 s of attack time plus model load. G3 is CPU-only except the
smoke gate (read-only). Frozen engine untouched (`835f1933d9e5…`).

Note: the untracked `reports/raw/B4e2/.ipynb_checkpoints/` directory (Jupyter dirt) is **not** committed.

## 9. Summary (plain English)

1. G1: replaced the stem-keyed `clip_events_182` merge with a row-position merge (`_pre_existing_count`) so a
   re-attacked error image's new row gets its own count and nothing shifts; the reviewer's error-row demo now
   passes (X-new=5, Y=9) and the done-image case is unchanged.
2. G1 verification: the fresh CGBA resnet50 NT re-run is bit-identical to the smoke result (24/24 rows, 24 npz
   delta bit-equal, `clip_events_182` = 0,0,0), and the diag passes both scenarios.
3. G2: `B4e2_followup.md` now carries the scoped §8.1 claim, a Deviations section, and the finding-1 citation.
4. G3: all three diags, tests_contract (21/21), selftest (20/20) and the smoke gate (exit 0) pass.
5. No forbidden file was edited; frozen engine and contract v1.8 unchanged; B5 not started. STOP.
