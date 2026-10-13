# B4e2 — Close B4e on the study GPU (RTX 5090): T4–T8

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`. The T0–T8 runs below used base
`e9ff7da83c71205f0884fbbf7b7f49cb3f8090a8` (tag `build38-amend2-17-ge9ff7da`); this report is **committed at the
new HEAD** `aee5027` (B4e2 close) and amended by the B4e2 follow-up commit (`reports/B4e2_followup.md`) — **not**
`e9ff7da`.

## 1. Scope

Phase B4e2 closes B4e on the study GPU. This report covers **T4–T8** (T0–T3 are closed: T0/T1/T2 environment,
frozen-engine B0' and the final smoke set; T3 gates — `check_contract --stage smoke` exit 0 (I01–I12 + DISK all
PASS, one GPU name in I10) and `human_preflight` FLAG = 7, documented as a deviation in §4).

- Prompt hash (sha256 of `reports/raw/B4e2/prompt.txt`):
  `ce4f18615a9cd3f72dd16aa7144e739b1b97c50ef1c02de4b4572fc3cb5820a5`
- Contract `paper_contract.yaml` (v1.8) sha256:
  `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`
- T3 raw evidence (committed): `reports/raw/B4e2/T3_gate.md` (the `check_contract --stage smoke` gate report,
  exit 0) and `reports/human_preflight_b4e2.md` (FLAG lines = 7, characterised in §4 deviation 1).

## 2. Environment

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU) |
| driver | 580.105.08 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False` |
| HEAD | `e9ff7da83c71205f0884fbbf7b7f49cb3f8090a8` (base; report committed at `aee5027` → follow-up commit) |
| engine sha | `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` |
| OMP_NUM_THREADS | 8 (every command) |

Every GPU command of this phase was run with
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` (plus `LPIPS_NET=vgg` for the EGC runner).

## 3. Tasks

### T4 — Row-split invariance (needed to split a cell across GPUs in B5)

For `LGC_RC_EGC` resnet50 NT and `TtBA` resnet50 NT, rows 2–3 were run alone into throwaway dirs and compared
with rows 2–3 of T2 (`found`, `state_query`, `pix_l2`, npz `delta`).

Commands (logs `reports/raw/B4e2/T4.log`, `T4_ttba.log`):

```
# EGC rows 2-3 (throwaway 2-row split /tmp/opencode/b4e2_t4/resnet50_non_targeted.csv)
python run_benchmark_33b.py --dataset imagenet --mode non_targeted --archs resnet50 --methods LGC_RC_EGC \
  --data_dir /root/autodl-tmp/ttba_mini_flat --labels_file ttba_mini_val_labels.txt \
  --split_csv "/tmp/opencode/b4e2_t4/{arch}_{mode}.csv" --image_loader ttba_github \
  --victim_ckpt_dir /root/autodl-tmp/TtBA/model --num_samples 2 --max_queries 10000 \
  --checkpoints 100 200 500 1000 2000 2500 5000 10000 --seed 42 \
  --exclude_seeds 992 2027 3141 2718 --out_dir Results/b4e2_t4_egc_resnet50 --save_adv

# TtBA rows 2-3 (exact seed mode)
python adapters/run_ttba_v2.py --ttba_repo /root/autodl-tmp/TtBA --dataset imagenet-resnet50 --targeted 0 \
  --split_csv /tmp/opencode/b4e2_t4/resnet50_non_targeted.csv --out_dir Results/b4e2_t4_ttba_resnet50 --limit 2
```

**Result: NOT IDENTICAL for both runners.** First difference in each case:

| runner | image | budget | field | T2 (rows 1–3) | T4 (rows 2–3 alone) |
|---|---|---|---|---|---|
| LGC_RC_EGC | ILSVRC2012_val_00015433.JPEG | 100 | pix_l2 | 32.866226 | 27.180103 |
| TtBA | ILSVRC2012_val_00015433.JPEG | 100 | state_query | 100 | 94 |
| TtBA | ILSVRC2012_val_00015433.JPEG | 100 | pix_l2 | 84.149429 | 72.871010 |

(`found` is 1 in both at that checkpoint; for TtBA the state_query flips first, and pix_l2 differs as well.)

**How each runner seeds per image (file:line):**

- **EGC** — `run_benchmark_33b.py:1325` `seed_img = derive_seed(args.seed, img_idx)`, where `img_idx` is the
  **0-based position** in the `pairs` list (`pairs = pairs[: args.num_samples]`, `:667`), enumerated at `:1315`
  (`indexed = list(enumerate(pairs))[args.shard_index :: args.num_shards]`, `:1312`). `derive_seed` is
  `lgc_bench/env.py:106-111` (multiplicative hash of `(base, img_idx, ...)`). Running rows 2–3 as a 2-row list
  relabels their `img_idx` 1,2 → 0,1, so seeds change from `derive_seed(42,1)=42020390` / `derive_seed(42,2)=42028309`
  to `42012471` / `42020390` (the `seed` column of `per_checkpoint.csv` records exactly this).
- **TtBA** — `adapters/run_ttba_v2.py` (exact mode, default) replays **main.py's single continuous RNG stream**
  (docstring `:9-11`); there is no per-image re-seed. `:258-259` refuses `--seed_mode exact` with `--num_shards != 1`
  ("replays main.py's single RNG stream; use --seed_mode per_image for shards"). `:356-358` shows the per-image
  alternative (`torch.manual_seed(42 + int(r["order"]))`, keyed on the CSV `order` field, not position).

Do not fix anything (per prompt). Consequence for B5: **row-splitting a cell is NOT invariant** under the T2
configuration for either runner — EGC's seed is position-based (truncation relabels rows) and TtBA-exact uses a
continuous stream. EGC *would* be invariant via its built-in sharding (`--num_shards/--shard_index`, which
preserves `img_idx`); TtBA would need `--seed_mode per_image`. This drives T8 (no row splitting).

### T5 — `ttba_github/ttba_common.py:83 done_images` resume fix (RESUME_FIX_APPROVED = yes)

Changed `done_images` so an image with **ANY** row in `runs_*.csv` is done, whatever its status (contract D6:
failures are never re-run selectively). Diff: removed the `if r.get("status") == "ok"` filter (`:87`).

```python
# before
return {r["image"] for r in csv.DictReader(f) if r.get("status") == "ok"}
# after
return {r["image"] for r in csv.DictReader(f)}
```

Sandbox unit test `adapters/diag/diag_b4e_t5_done_images.py` — **PASS** (5/5): `ok`, `budget` and `error` rows all
skipped; a missing image (with and without the file) is not done.

Fresh-run invariance (row 1 into new dirs → IDENTICAL to T2):

| runner | out dir | row 1 vs T2 row 1 (found, state_query, pix_l2 @ all 8 checkpoints) |
|---|---|---|
| LGC_RC_EGC resnet50 NT | `Results/b4e2_t5_egc_resnet50` | IDENTICAL (8/8 budgets, 0 diffs) |
| TtBA resnet50 NT | `Results/b4e2_t5_ttba_resnet50` | IDENTICAL (8/8 budgets incl. `pix_l2_iter`, 0 diffs) |

### T6 — CGBA clip-event counter (observation only)

- `adapters/run_cgba.py` `PatchedProposedAttack.__init__` initialises `self.clip_events_182 = 0`; the `fixed`
  `SemiCircular_boundary_search` hoists the acos argument into `arg` (verbatim copy of the official expression),
  increments `self.clip_events_182` when `arg < -1.0 or arg > 1.0`, then computes
  `theta = torch.acos(torch.clamp(arg, -1.0, 1.0))`. `run_official` appends the per-image count (try/finally) and
  `main()` merges it into `runs_<method>.csv` as column `clip_events_182`.
- `adapters/diag/diag_b4e_t1_clip_source.py` now asserts the clamp is the **only value-changing edit**: the
  `arg` line is the official acos argument verbatim, the theta line is `torch.acos(torch.clamp(arg, -1.0, 1.0))`,
  the two counter lines are the only other additions, and every other line is byte-identical. **PASS** (also
  `d8only` still inherits the official method).
- Re-run CGBA resnet50 NT rows 1–3 into `Results/b4e2_t6_cgba_resnet50` (log `T6.log`): **numerics IDENTICAL**
  to T2 (24/24 `per_checkpoint` rows equal; npz `delta` bit-equal). The counts are recorded in
  `Results/b4e2_t6_cgba_resnet50/non_targeted/runs_CGBA.csv` column `clip_events_182` (values 0,0,0 for the three
  images `ILSVRC2012_val_00007450`, `_00015433`, `_00025151`). Under D12 precision parity the NaN event does not
  occur for this cell, consistent with B4e §T2.
- **CGBA_D8only note:** `clip_events_182` is constant-0 by design for `CGBA_D8only` (the counter lives only in the
  `fixed` `SemiCircular_boundary_search` override; `d8only` inherits the official method and never increments it).
  The E17 NaN-share measurement is a separate B4g/B5 item, not this counter.

### T7 — `tools/eval/check_contract.py` status-file placement

The checker used to hardcode the status file to `reports/contract_status.md`. Now `render_report` writes it next
to `--report` as `<report stem>_status.md` unless `--status_out` is given (`main()` adds `--status_out`).

```
python tools/eval/tests_contract/test_contract.py   -> ALL PASS (21/21)
python tools/eval/selftest_evalkit.py               -> ALL PASS (20/20)
```

Added one test (T18 in `test_contract.py`): default status written to `<report stem>_status.md`; the tracked
`reports/contract_status.md` is **not** rewritten when `--report` points elsewhere; `--status_out` is honoured.

### T8 — Cost on STUDY_GPU

`s/img` = mean `wall_time_s` over the 3 smoke images from T2 (`b4e2_smoke_*`). GPU-h per cell = 200 × s/img / 3600
(main, E10/E11); E17 = 30 × s/img / 3600 (d8only, not re-run in T2 → same runtime as CGBA NT, the clip is
negligible). Full per-cell table is in `reports/raw/B4e2/T8.log`.

| scope | cells | GPU-h |
|---|---|---|
| main ImageNet | 24 | 45.27 |
| E10/E11 (Places365 + RobustBench) | 18 | 40.63 |
| E17 appendix (CGBA D8-only, 6 CGBA NT cells × 30) | 6 | 1.29 |
| **total** | **48** | **87.18** |

Makespan (48 cells, longest-cell-first greedy, one process per GPU, **no row splitting** — T4 is NOT IDENTICAL):

| GPUs | makespan |
|---|---|
| N = 4 | 21.86 h |
| N = 8 | 11.29 h |
| N = 12 | 7.40 h |

The two largest cells are TtBA / densenet161_places365 (5.88 and 5.83 GPU-h, ≈105 s/img); they dominate the
E10/E11 total.

## 4. Deviations from the prompt

1. **FLAG = 7 from `human_preflight` (T3), recorded as a documented deviation for B4f T7.** The 7 flags are
   **6 EGC E10/E11 roots** whose `env_frame.txt` (written by the generic runner `ttba_github/run_frame_b1_generic.py`)
   lacks `cudnn_tf32`/`deterministic`, **plus 1 GPU-name mismatch** from the frozen `config_single.json` device
   string (run_frame_b1.sh records the precision flags inside `config_single.json`, not `env_frame.txt`, and the
   device string there differs from `NVIDIA GeForce RTX 5090`). These are environment-*recording* gaps in the
   frozen/generic EGC runners — a preflight recording gap, not a data-validity issue — and are explicitly the
   B4f T7 task (the B4e2 allowed-list excludes `run_frame_b1_generic.py` and `human_preflight.py`). The paper's
   Methods will state "single RTX 5090, TF32 off, cuDNN deterministic". All other `human_preflight` sections
   (2–5) are clean: `stall = 0`, `nonfinite_queries = 0`, `error = 0` in every cell.
2. **T4/T5/T6 throwaway dirs** `Results/b4e2_t4_*`, `Results/b4e2_t5_*`, `Results/b4e2_t6_cgba_resnet50` were
   created in the worktree as directed (not under `reports/raw/`).
3. **T4/T5 throwaway splits** live in `/tmp/opencode/b4e2_t4|b4e2_t5` (not committed); their content is the
   header + the exact CSV rows of the frozen split (protocol/victim_weights columns preserved).
4. **T4 log split into two files** (`T4.log` for EGC, `T4_ttba.log` for TtBA) because T4 runs two runners.
5. **E17 s/img source**: E17 d8only was not re-run in T2 and has no separate STUDY_GPU smoke; its `s/img` is
   taken from the T2 CGBA NT cells (identical algorithm, the clip adds no measurable time).
6. **Resume policy divergence (unresolved, user decision for B5).** T5 changed `ttba_github/ttba_common.py:83`
   (`done_images` now treats ANY row as done, whatever its status). The baseline adapters that go through `C.run`
   (`adapters/_common.py:421-425`) still filter `status in ("ok", "budget")`, so CGBA / CGBA_H / CGBA_D8only and
   ARS_OPT re-run `error` rows on resume. The two resume policies therefore diverge, and the binding policy for B5
   is a **user decision** (to be recorded in contract v1.9).

## 5. sha256 of created / changed files; GPU minutes

| file | change | sha256 |
|---|---|---|
| `ttba_github/ttba_common.py` | done_images: any row = done (T5) | `53d518b3bf8c57d5ff40183c8ab4bdaee08daba92afffea136af5e7c415a64e9` |
| `adapters/run_cgba.py` | clip_events_182 counter (T6) | `99f5c11ff856f6749766c5b52feff9b7f1047e2c30bac1091d83635b13b7b35e` |
| `adapters/diag/diag_b4e_t1_clip_source.py` | clamp = only value-changing edit (T6) | `0a8695b3f5709e38bb2482b2ade20f4fed21a96eddb58dd7dde74d3f828e9e90` |
| `adapters/diag/diag_b4e_t5_done_images.py` | new sandbox unit test (T5) | `219d1b53885ebd661c4aaf031eb7a9adb1f0d2271551c73403fe47b600eb58bb` |
| `tools/eval/check_contract.py` | status next to --report / --status_out (T7) | `ce2cf71cb722dcf1c5f5d0ed3c8446b26be8574f2dda66b92e1812be1bedd1ab` |
| `tools/eval/tests_contract/test_contract.py` | +T18 status-file test (T7) | `6ab5deaa107f3d2a8972462f98803babba770d9ea822a66c4013e8396ff4eaaa` |

Log files under `reports/raw/B4e2/` (skill lgc-preflight §5 — sha256 of every created/changed file):

| file | sha256 |
|---|---|
| `T1_A.log` | `ea4b1fa7d09beb3bee39b818797a05e5495728d41a2d46818186503e842cc272` |
| `T1_A2.log` | `ec6b4e39984dcc1e578cde38507eefd8c1a80adbb2e6d84eb0869e4e09545d91` |
| `T1_B.log` | `2d97b1591bca554669e432446b631e16e6f68e415d0dfb1768b5ccf127da9327` |
| `T2_arsopt_main.log` | `edf7560e03e490bc7117789d5ec538e33a964da27455400df78ce648348ebf1d` |
| `T2_cgba_e10e11.log` | `c7b24c6bfb323f7df1de6d84bd3d41d06987a135c4d91de6389dbd7cc15b2982` |
| `T2_cgba_main.log` | `e342709e96d61be050441004c908cc82a30894f2ce7e72c468a61073a78aa8df` |
| `T2_egc_e10e11.log` | `e2f4492d481026b071f2eb75a45f33ba4e8e8a25ef31ed4da68de1666f5b52d6` |
| `T2_egc_main.log` | `b17da8ed0da9cccaa72d81aafcd8ff67e2c1ca11a0d9d2822c8d3685cb216911` |
| `T2_ttba_e10e11.log` | `0d729150141e1b10f0e3ba6f0116b762a8de37160558586ff5bed66e9b406349` |
| `T2_ttba_main.log` | `cf8c2f055eadfcb089ddc06ca4a9800dfc05233119269bd74a74653bc2ed81a5` |
| `T4.log` | `bfd7c2f6eea1ae8f78c20688903d0f797a9b2b933c6beba546c17d0d19492155` |
| `T4_ttba.log` | `02c96587a7871758f7aa2283ca606d23a4484b67415d8748663dab3be08d209e` |
| `T5_egc.log` | `32a1770b919c940241141ade4fb9a06fa03f8cb8ab9b4d663b7866d6b7cad02b` |
| `T5_ttba.log` | `78f44658fab8ba47a8c9ff8caf728fdcbb4648b0f90d9fcb06def85978ea3000` |
| `T6.log` | `95a834204189f96f968ba0bf42f6ed7156a858bb95ee54d574ca5e76bb252dd9` |
| `T8.log` | `f142c3ddcb0e102836d67c7d820d681b44ec21c98733360f9d2366d9457f3de7` |

GPU minutes this session (T4–T8): ≈ 5 min of attack time (T4 EGC ≈46 s + T4 TtBA ≈105 s + T5 EGC ≈26 s +
T5 TtBA ≈53 s + T6 CGBA ≈70 s) plus model-load overhead; T7/T8 are CPU-only. Frozen engine untouched
(`835f1933d9e5…`).

## 6. Summary (plain English)

1. T4: running rows 2–3 of a cell in isolation is **NOT bit-identical** to those rows in the full T2 run, for both
   the EGC runner (position-based per-image seed `derive_seed(seed, img_idx)`) and TtBA (continuous main.py RNG
   stream in exact mode) — so B5 must not row-split cells under the current configuration; T8 assumes one process
   per GPU.
2. T5: fixed `done_images` so a row with any status (`ok`/`budget`/`error`) counts as done; the sandbox test passes
   and a fresh row-1 run reproduces T2 bit-identically for both EGC and TtBA.
3. T6: added an observation-only `clip_events_182` counter to CGBA; the re-run is numerically identical to T2 and
   the counts are 0 (the NaN event is absent under D12 precision parity); the diag asserts the clamp is the only
   value-changing edit.
4. T7: the contract checker now writes its status file next to `--report` (or `--status_out`); tests 21/21 and
   evalkit selftest 20/20.
5. T8: total cost ≈ 87.2 GPU-h (main 45.3 + E10/E11 40.6 + E17 1.3); makespan 21.9 / 11.3 / 7.4 h on 4 / 8 / 12
   GPUs (longest cell first, no row splitting). STOP — no B5 work started.
