# B5h2 — host probe for B5 attempt 2 (one GPU)

Status: **COMPLETE**. Verdict: **HOST-IDENTICAL** — T2, T3 and T7a all bitwise IDENTICAL to the B4e2/B4h-t3
reference roots, and T8 (the launcher's per-cell gate end-to-end) exits 0 on every tested cell; T5/T5b n/a
(one GPU). The T6 contract gate, which STOPped twice (E-ENV shim, then E-SPEC junk dirs), now exits 0 after H's
decisions (sitecustomize shim approval; the two `.ipynb_checkpoints` junk dirs moved to
`/root/autodl-tmp/_junk_ipynb`) — documented as a deviation below. B5 not started, no `Results/b5_*` or
`Results/appendix_cgba_d8only_*` directory created.

Lessons ledger rows L01–L27 read at phase start; the minimum set (L03 L04 L06 L21 L23 L24 L26 L27) is active and
applies. None contradicted; L26/L24 dictate the X1–X6 execution rules used throughout; L03 motivates T0's device
check; L22/L23 motivate the domain_consistency gate used in T3/T7a; L02/L23 are enforced in code (D11 clip,
domain gate).

## 1. Scope and prompt

- Prompt (executed byte-for-byte): `reports/raw/B5h2/prompt.txt`
- Prompt sha256: `0be547fb96096f968d817ca6af47c7f9d3c42520c6edc3a4f1cd9a7c6def2395`
- Contract `contract/paper_contract.yaml` **v1.9-rev7** sha256: `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`
- `git rev-parse HEAD`: `5a1fa5a1ef02226905f61a2e6c6dea9eca832f49` (branch paper-runs)
- `run_b5_parallel.sh` sha256: `50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb`
- Every GPU command ran with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` and was tee'd to `reports/raw/B5h2/<step>.log`.
- `lgc-baselines` skill is NOT installed in `.opencode/skills/` (only lgc-paper-contract, lgc-preflight).

## 2. Environment block (T0)

| field | value | verdict |
|---|---|---|
| hostname | `autodl-container-20bc47a853-4e77a3dc` | — |
| GPU count / name | 1 x `NVIDIA GeForce RTX 5090` (reports/raw/B5h2/nvidia_smi_L.txt; direct `nvidia-smi` is denied by the OpenCode guard, per H's note) | count 1 == NGPU; name == D16 study GPU → OK |
| GPU name (2nd source) | `torch.cuda.get_device_name(0)` = `NVIDIA GeForce RTX 5090` | OK |
| driver | `580.105.08` (reports/raw/B5h2/nvidia_smi_query.txt) | equals the B4e2/attempt-1 host driver; differs from env_build38.txt `595.71.05` → finding (contract D16 rev7 records the attempt-2 driver at T0) |
| torch / torchvision / cuDNN | `2.7.1+cu128` / `0.22.1+cu128` / `90701` | exactly as required → OK |
| backend flags (D12, as `adapters/_common.py` sets them) | matmul_tf32=False, cudnn_tf32=False, deterministic=True, benchmark=False | OK |
| CPU quota | `/sys/fs/cgroup/cpu.max` = `2500000 100000` → Q = 25 | Q = 25 ≥ 8 x NGPU = 8 → no finding |
| `env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc` | `208` | — |
| RAM | `free -g`: 754 GB total, 568 GB available | OK |
| disk | `df -h /root/autodl-tmp`: 50 G total, 3.8 G used, 47 G free | ≥ 30 GB free → OK |
| TMUX | `TMUX=/tmp/tmux-0/default,1593,0` (non-empty) | finding: OpenCode was started inside a tmux session; no tmux started by the agent |
| engine sha | `835f1933d9e5…` (frozen) | OK |

Dry run `NGPU=1 bash run_b5_parallel.sh --dry_run` (tee `reports/raw/B5h2/T0_dryrun.log`, exit 0). Guard lines:

```
[env] torch=2.7.1+cu128 (expected 2.7.1+cu128)  torchvision=0.22.1+cu128 (expected 0.22.1+cu128)  cuDNN=90701 (expected 90701)
[threads] Q=25.0 NGPU=1 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8
[dry_run] NGPU=1 RESULTS_ROOT=Results LOG_DIR=reports/raw/B5a2 (no GPU process started)
```

First 3 queued cells: `e10_TtBA_densenet161_places365_NT`, `e10_TtBA_densenet161_places365_T`, `main_TtBA_vgg19_T`.

## 3. T1 — B0' determinism on this host (resnet50 NT rows 1–3)

| run | out_dir | command | result |
|---|---|---|---|
| A | `Results/b5h2_A` | `run_benchmark_33b.py … --checkpoints 500 1000 2000 2500 5000 10000` (6 registered budgets, no `--save_adv`) | 3/3 ok |
| A2 | `Results/b5h2_A2` | identical command | 3/3 ok |
| B | `Results/b5h2_B_resnet50` | `MODES=0 N=3 OUT_PREFIX=Results/b5h2_B bash ttba_github/run_frame_b1.sh resnet50` (8 contract budgets, `--save_adv`) | 3/3 ok |

compare_roots last 3 lines:

```
A vs A2:  checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
          npz files: shared 0, only in A 0, only in B 0; differences: 0
          VERDICT: IDENTICAL
A vs B (registered):  checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
          npz files: shared 0, only in A 0, only in B 0; differences: 0
          VERDICT: IDENTICAL
```

**PASS** (A == A2 bitwise at every budget; A == B at the 6 registered budgets).

## 4. T2 — cross-host identity, LGC_RC_EGC resnet50 NT (information)

compare_roots `Results/b4e2_smoke_LGC_RC_EGC_resnet50` vs `Results/b5h2_B_resnet50` (last 3 lines):

```
checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
npz files: shared 24, only in A 24, only in B 0; differences: 0
VERDICT: IDENTICAL
```

24/24 shared non_targeted checkpoints identical, 24/24 npz identical; the "only in A" rows are the reference root's
targeted-mode rows (B ran non_targeted only). **Verdict: IDENTICAL — hosts may be pooled bitwise for LGC_RC_EGC
(pre-registered stack, torch 2.7.1+cu128 / cuDNN 90701).**

## 5. T3 — TtBA / CGBA (fixed) / ARS_OPT on resnet50 NT rows 1–3

Runs into `Results/b5h2_{TtBA,CGBA,ARS_OPT}_resnet50` with the current adapters (D19 domains, `--cgba_variant fixed`,
TtBA default `--seed_mode exact`). compare_roots vs the reference roots — last 3 lines each:

```
TtBA    vs b4e2_smoke_ttba_resnet50:     checkpoint differences: 0 … npz files: shared 24, only in A 24, only in B 0; differences: 0 → VERDICT: IDENTICAL
CGBA    vs b4h_t3_smoke_CGBA_resnet50:   checkpoint differences: 0 … npz files: shared 24, only in A 24, only in B 0; differences: 0 → VERDICT: IDENTICAL
ARS_OPT vs b4h_t3_smoke_ARS_OPT_resnet50: checkpoint differences: 0 … npz files: shared 24, only in A 24, only in B 0; differences: 0 → VERDICT: IDENTICAL
```

oob_queries / nonfinite_queries (new runs, of budget 10000; reference values do not exist — the counter postdates the
reference roots):

| method | 07450 / 15433 / 25151 oob_queries | nonfinite | status |
|---|---|---|---|
| TtBA | 8435 / 8546 / 8457 | 0 | budget x3 |
| CGBA | 9800 / 9369 / 9828 | 0 | budget x3 |
| ARS_OPT | 10000 / 10000 / 10000 | 0 | budget x3 |

domain_consistency on the new CGBA and ARS_OPT roots (`t3_domain_consistency.csv`, 48 rows):

```
| CGBA | resnet50/non_targeted | 24 | 19 | 5 | 0 | 0 | …
| ARS_OPT | resnet50/non_targeted | 24 | 15 | 9 | 0 | 0 | …
[+] no X>0 for any method
```

**X = 0 by the column named `class` for both methods. PASS.**

## 6. T4 — s/img (mean wall_time_s of rows 1–3, non_targeted)

| method | ref s/img | this host s/img | ratio |
|---|---|---|---|
| LGC_RC_EGC | 23.34 | 23.58 | 1.01 |
| TtBA | 52.76 | 54.65 | 1.04 |
| CGBA | 26.22 | 23.63 | 0.90 |
| ARS_OPT | 34.55 | 33.69 | 0.98 |

Throughput comparable; no slowdown finding.

## 7. T5 / T5b

**T5 n/a (one GPU)** (per the prompt/H note: NGPU = 1). **T5b n/a (launcher threads = 8)** — the T0 dry-run
`[threads]` line is `Q=25.0 NGPU=1 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8`, i.e. the launcher's own value is 8, not < 8.

## 8. T6 — ARS_OPT on Salman2020Do_R50 + Places365 victims — runs PASS, contract gate PASS after H's junk move

H's decision on the earlier STOP (`reports/stop/B5h2_T6_20261020-0242.md`, E-ENV): create the attempt-1
`site-packages/sitecustomize.py` timm pre-import shim on this instance (approval under AGENTS.md rule 10). Applied
this session (deviation 8):

- `/root/miniconda3/lib/python3.12/site-packages/sitecustomize.py` (content exactly as approved) sha256
  `d0161cad705ad8b7cbdd82650fda548452065f099c04e4759617126142178458`
- `python -c "import sys, timm; print(timm.__version__, timm.__file__)"` → `1.0.28
  /root/miniconda3/lib/python3.12/site-packages/timm/__init__.py` (tee
  `reports/raw/B5h2/sitecustomize_timm_check.txt`)
- The old repro (`sys.path.insert(0, ARS-OPT repo); import robustbench`) now passes (tee
  `reports/raw/B5h2/sitecustomize_repro_check.txt`).

All six T6 cells then completed (each exit 0, 3/3 rows, tee'd to `reports/raw/B5h2/T6b_*.log`):

| victim | mode | rows | q | L2@10k per image | s/img |
|---|---|---|---|---|---|
| Salman2020Do_R50 | non_targeted | 3/3 | 10000 | 13.028959 / 15.318589 / 0.842405 | 33.9 / 33.9 / 33.2 |
| Salman2020Do_R50 | targeted | 3/3 | 10000 | 69.431328 / 89.484467 / 82.281891 | 38.2 / 37.9 / 38.8 |
| resnet50_places365 | non_targeted | 3/3 | 10000 | 11.373825 / 7.177472 / 0.580240 | 34.1 / 33.2 / 33.4 |
| resnet50_places365 | targeted | 3/3 | 10000 | 40.435055 / 27.547689 / 30.049971 | 37.9 / 37.4 / 37.4 |
| densenet161_places365 | non_targeted | 3/3 | 10000 | 5.063626 / 6.313108 / 1.210177 | 57.6 / 57.0 / 55.2 |
| densenet161_places365 | targeted | 3/3 | 10000 | 8.811350 / 19.929216 / 5.897661 | 51.7 / 50.7 / 50.1 |

The empty `Results/b5h2_ARS_OPT_Salman2020Do_R50/{non_targeted,targeted}` dirs left by the failed attempt did NOT
block anything (check_manifest passes on empty dirs, `adapters/_common.py:631-635`) — nothing removed.

T6 acceptance checks:

- **queries = 10000** for all 18 rows (`runs_ARS_OPT.csv`); nonfinite_queries = 0 everywhere.
- **domain_consistency X = 0**: `adapters/diag/domain_consistency.py` over the three new roots → 144 rows, X=0 for
  all 6 cells (tee `reports/raw/B5h2/T6_domain_consistency.log`, csv `t6_domain_consistency.csv`).
- **human_preflight FLAG 0**: `tools/eval/human_preflight.py` over the three new roots →
  `FLAG lines: 0` (`reports/raw/B5h2/t6_human_preflight.md` + 6 grid PNGs).
- **per checkpoint found / pix_l2**: `found = 1` at every budget for all 18 rows; pix_l2 monotonically decreasing
  (full table in `reports/raw/B5h2/T6_per_checkpoint_summary.txt`).

Root lists: replaced exactly the three `Results/b5host_ARS_OPT_*` lines by
`Results/b5h2_ARS_OPT_Salman2020Do_R50`, `Results/b5h2_ARS_OPT_resnet50_places365`,
`Results/b5h2_ARS_OPT_densenet161_places365` in `reports/B4e2_smoke_roots.txt` and `reports/B5_postfix_roots.txt`
(nothing else changed; diff tee'd to `reports/raw/B5h2/T6_root_lists_diff.txt`).

**Contract gate — first re-run FAILed (STOP bundle `reports/stop/B5h2_T6b_20261020-0310.md`, E-SPEC), then PASSed
after H's junk move (§9).**

```
python tools/eval/check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust \
    --roots $(cat reports/B4e2_smoke_roots.txt) --report reports/raw/B5h2/contract_smoke.md \
    --status_out reports/raw/B5h2/contract_smoke_status.md
→ FAIL  I01 (FAIL) count=48 … [b4h_t3_smoke_ARS_OPT_resnet50 … .ipynb_checkpoints/per_checkpoint_ARS_OPT-checkpoint.csv …
   duplicate (image, method, budget)] … exit code 1
```

All 48 I01 duplicates came from TWO pre-existing junk dirs left by the attempt-1 host's notebook sessions
(`Results/b4h_t3_smoke_ARS_OPT_{resnet50,vgg19}/non_targeted/.ipynb_checkpoints/`, created 2026-10-16/17;
`check_contract.py:173-180` does not prune dot-dirs). Every other invariant (I01–I12 + DISK, and I02/I10 once
`--split_dir` is passed) PASSed, including over the three new b5h2 roots — the gate would exit 0 with only the two
junk dirs removed. Deleting files in `Results/` was outside the allowed edits, so a STOP bundle was written
(`reports/stop/B5h2_T6b_20261020-0310.md`, E-SPEC; deviation 10) and H then moved the two junk dirs to
`/root/autodl-tmp/_junk_ipynb` (verified at T6c: `find Results -name .ipynb_checkpoints -type d | wc -l` = 0). The
gate re-run (T6c) exits 0.

## 9. T6c — contract smoke gate re-run after H's junk move — PASS

Verified first (before any other command): `find Results -name .ipynb_checkpoints -type d | wc -l` → **0**;
the two moved dirs are in `/root/autodl-tmp/_junk_ipynb/`
(`Results_b4h_t3_smoke_ARS_OPT_{resnet50,vgg19}_non_targeted_.ipynb_checkpoints`); `df -h /root/autodl-tmp` → 47 G
free. Then the T6 command re-run exactly as before (same `--split_dir`), writing NEW files
`reports/raw/B5h2/T6c_*` (the T6 logs were not overwritten, per H's instruction):

```
python tools/eval/check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust \
    --roots $(cat reports/B4e2_smoke_roots.txt) --report reports/raw/B5h2/T6c_contract_smoke.md \
    --status_out reports/raw/B5h2/T6c_contract_smoke_status.md   (tee reports/raw/B5h2/T6c_check_contract_smoke.log)

    PASS  I01   (FAIL)  count=0
    PASS  I02   (FAIL)  count=0
    PASS  I03   (FAIL)  count=0
    PASS  I04   (FAIL)  count=0
    PASS  I05   (FAIL)  count=0
    PASS  I06   (FAIL)  count=0
    PASS  I07   (FAIL)  count=0
    PASS  I08   (FAIL)  count=0
    PASS  I09   (WARN)  count=0
    PASS  I10   (FAIL)  count=0
    PASS  I11   (FAIL)  count=0
    PASS  I12   (FAIL)  count=0
    PASS  DISK  (FAIL)  count=0
[*] report: reports/raw/B5h2/T6c_contract_smoke.md; exit code 0
```

**Exit 0; every invariant PASS. T6 gate: PASS.** `reports/contract_status.md` was not rewritten
(`git status --porcelain reports/contract_status.md` is empty; `--status_out` redirected the status write).

## 10. T7a — LGC_RC_EGC / TtBA / CGBA-CGBA_H on both Places365 victims, both modes, rows 1–3

12 cells into `Results/b5h2_*` (each command tee'd to `reports/raw/B5h2/T7a_*.log`, exit 0, 3/3 rows):

| method | cell | command family | wall time (3 imgs) |
|---|---|---|---|
| LGC_RC_EGC | resnet50_places365 NT / T | `ttba_github/run_frame_b1_generic.py --arch resnet50_places365 --mode … --split_csv splits/places365/… --out_dir Results/b5h2_LGC_RC_EGC_resnet50_places365_{non,targeted} --num_samples 3 --save_adv` | 24.3/22.2/20.4 s; 26.2/24.4/24.5 s |
| LGC_RC_EGC | densenet161_places365 NT / T | same with `--arch densenet161_places365`, `--out_dir Results/b5h2_LGC_RC_EGC_densenet161_places365_{non,targeted}` | 35.4/35.8/31.3 s; 38.7/35.9/33.1 s |
| TtBA | resnet50_places365 NT + T | `adapters/run_ttba_v2.py --arch resnet50_places365 --targeted 0/1 --split_csv splits/places365/… --out_dir Results/b5h2_TtBA_resnet50_places365 --limit 3` | 51.7/51.0/51.1 s; 48.1/48.2/48.6 s |
| TtBA | densenet161_places365 NT + T | same, `--out_dir Results/b5h2_TtBA_densenet161_places365` | 107.8/108.0/108.3 s; 108.1/108.0/107.4 s |
| CGBA / CGBA_H | resnet50_places365 NT (CGBA) / T (CGBA_H) | `adapters/run_cgba.py --arch resnet50_places365 --mode non_targeted/targeted --split_csv splits/places365/… --repo /root/autodl-tmp/baselines/CGBA --out_dir Results/b5h2_CGBA_resnet50_places365 --limit 3 --cgba_variant fixed` | 24.4/21.5/21.5 s; 24.2/23.0/22.9 s |
| CGBA / CGBA_H | densenet161_places365 NT (CGBA) / T (CGBA_H) | same, `--out_dir Results/b5h2_CGBA_densenet161_places365` | 38.6/37.2/35.2 s; 43.3/41.6/41.0 s |

compare_roots vs the matching roots of `reports/B4e2_smoke_roots.txt` (CGBA: the b4h_t3 roots) — last 3 lines of
each of the 8 comparisons (tee `reports/raw/B5h2/T7a_compare_roots.log`):

```
1. LGC_RC_EGC resnet50_places365 NT  (b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 24, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
2. LGC_RC_EGC resnet50_places365 T  (b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 24, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
3. LGC_RC_EGC densenet161_places365 NT  (b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 24, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
4. LGC_RC_EGC densenet161_places365 T  (b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 24, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
5. TtBA resnet50_places365  (b4e2_smoke_ttba_resnet50_places365):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 48, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
6. TtBA densenet161_places365  (b4e2_smoke_ttba_densenet161_places365):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 48, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
7. CGBA/CGBA_H resnet50_places365  (b4h_t3_smoke_CGBA_resnet50_places365):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 48, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
8. CGBA/CGBA_H densenet161_places365  (b4h_t3_smoke_CGBA_densenet161_places365):
   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
   npz files: shared 48, only in A 0, only in B 0; differences: 0
   VERDICT: IDENTICAL
```

domain_consistency on the two new CGBA/CGBA_H roots (tee `reports/raw/B5h2/T7a_domain_consistency.log`,
`t7a_domain_consistency.csv`, 96 rows) — **X = 0 for CGBA and CGBA_H** (`[+] no X>0 for any method`):

```
| CGBA   | resnet50_places365/non_targeted   | 24 | 22 | 2 | 0 | 0 | …
| CGBA_H | resnet50_places365/targeted       | 24 | 17 | 7 | 0 | 0 | …
| CGBA   | densenet161_places365/non_targeted| 24 | 21 | 3 | 0 | 0 | …
| CGBA_H | densenet161_places365/targeted    | 24 | 23 | 1 | 0 | 0 | …
```

**T7a: PASS — all 8 bitwise comparisons IDENTICAL, X = 0 for CGBA / CGBA_H.**

## 11. T8 — gate end-to-end (the launcher's own gate tool on real GPU outputs) — PASS

`python tools/eval/b5_gate_cell.py <cell> --root <root>` (plus `--ttba_root` where named) with
`--log_dir reports/raw/B5h2/gates_e2e`, one tee'd log per cell (`reports/raw/B5h2/T8_gate_*.log`):

| cell | root (this probe) | partial | g5 | domain | exit |
|---|---|---|---|---|---|
| `main_LGC_RC_EGC_resnet50_NT` | `Results/b5h2_B_resnet50` (T1 B) | PASS | N/A (not a G5 method) | PASS, X=0 (reported, not a stop rule) | 0 |
| `main_TtBA_resnet50_NT` | `Results/b5h2_TtBA_resnet50` | PASS | PENDING (tool check only, n=3) | SKIP (TtBA not in the domain gate) | 0 |
| `main_CGBA_resnet50_NT` | `Results/b5h2_CGBA_resnet50` (`--ttba_root Results/b5h2_TtBA_resnet50`) | PASS | PENDING (tool check only, n=3) | PASS, X=0 | 0 |
| `main_ARS_OPT_resnet50_NT` | `Results/b5h2_ARS_OPT_resnet50` | PASS | N/A | PASS, X=0 | 0 |
| `e10_CGBA_H_resnet50_places365_T` | `Results/b5h2_CGBA_resnet50_places365` (T7a root) | PASS | N/A (G5 n/a, no published reference) | PASS, X=0 | 0 |
| `e11_ARS_OPT_Salman2020Do_R50_NT` | `Results/b5h2_ARS_OPT_Salman2020Do_R50` (T6 root) | PASS | N/A | PASS, X=0 | 0 |

Requirements met: **exit 0 for each cell**; **partial PASS** everywhere; **G5 PENDING** ("tool check only",
3 images) for the two main-study G5 cells and **n/a** for the rest — never PASS; **domain X = 0** for
CGBA / CGBA_H / ARS_OPT, and the X count recorded (X=0) for LGC_RC_EGC. No `reports/stop/B5a2_*` file was written
by these runs (`ls reports/stop/ | grep B5a2` → nothing). Summary (pasted from
`reports/raw/B5h2/gates_e2e/gates/summary.csv`):

```
cell,scope,method,arch,mode,partial,g5,g5_code,domain_x,domain,stop
main_LGC_RC_EGC_resnet50_NT,main,LGC_RC_EGC,resnet50,non_targeted,PASS,N/A,n/a,0,PASS,0
main_TtBA_resnet50_NT,main,TtBA,resnet50,non_targeted,PASS,PENDING,pending,,SKIP,0
main_CGBA_resnet50_NT,main,CGBA,resnet50,non_targeted,PASS,PENDING,pending,0,PASS,0
main_ARS_OPT_resnet50_NT,main,ARS_OPT,resnet50,non_targeted,PASS,N/A,n/a,0,PASS,0
e10_CGBA_H_resnet50_places365_T,e10,CGBA_H,resnet50_places365,targeted,PASS,N/A,n/a,0,PASS,0
e11_ARS_OPT_Salman2020Do_R50_NT,e11,ARS_OPT,Salman2020Do_R50,non_targeted,PASS,N/A,n/a,0,PASS,0
```

**T8: PASS.**

## 12. Deviations from the prompt

1. Direct `nvidia-smi` calls are denied by the OpenCode guard; GPU count, name and driver are taken from H's saved
   outputs `reports/raw/B5h2/nvidia_smi_L.txt` / `nvidia_smi_query.txt` (quoted above) with
   `torch.cuda.get_device_name(0)` as a second source (H's note to T0).
2. T1 B lands in `Results/b5h2_B_resnet50` (`run_frame_b1.sh` appends `_<arch>`; the script is not editable), same
   convention as attempt 1.
3. Driver is `580.105.08`, not env_build38.txt's `595.71.05` (equals the B4e2/attempt-1 host driver; recorded here as
   the attempt-2 T0 driver per contract D16 rev7). GPU name matches D16 → no STOP.
4. `TMUX` is non-empty (OpenCode inside a tmux session) — reported as a finding; no tmux was started by the agent.
5. T6 STOP (E-ENV) as described in the earlier STOP bundle `reports/stop/B5h2_T6_20261020-0242.md`; RESOLVED by H's
   approved shim (deviation 8) — all six T6 cells then completed.
6. `Results/b5h2_*` result dirs are data, not committed by H; the RESULT block lists the report/raw files (§13) and
   this report.
7. The Salman targeted attempt (first T6 session) was chained after the NT attempt; the NT failure's pipeline exit
   (tail) still launched it, so both modes were attempted and both failed identically — two failures of the same
   cause, zero fix attempts (X4: no fix within the allowed edits).
8. **H-approved file outside the repo** (AGENTS.md rule 10, H's decision on the E-ENV STOP): created exactly
   `/root/miniconda3/lib/python3.12/site-packages/sitecustomize.py` with exactly the approved content (timm
   pre-import try/except). Same shim as the attempt-1 host. sha256
   `d0161cad705ad8b7cbdd82650fda548452065f099c04e4759617126142178458`; timm check `1.0.28` under site-packages.
   Nothing else outside `reports/*` was changed.
9. The first `check_contract.py --stage smoke` invocation omitted `--split_dir`, showing I02 WARN (count=6); the
   second invocation with `--split_dir splits/imagenet/ttba_github splits/places365 splits/robust` cleared I02 and
   I10, leaving only the I01 junk-dir FAIL (STOP bundle `reports/stop/B5h2_T6b_20261020-0310.md`).
10. **T6 gate STOP history (documented; now resolved):** the T6 contract gate FAILed with exit 1 on 48 I01
    duplicates caused solely by two pre-existing `.ipynb_checkpoints` junk dirs in the b4h_t3 reference roots;
    deleting files in `Results/` was outside my allowed edits, so STOP bundle
    `reports/stop/B5h2_T6b_20261020-0310.md` (E-SPEC) was written and T7a/T8 were not run in that session. H then
    moved the two junk dirs to `/root/autodl-tmp/_junk_ipynb`; after resuming
    (`find Results -name .ipynb_checkpoints -type d | wc -l` = 0), the gate re-ran as **T6c with exit 0**
    (all invariants PASS) and the new gate outputs were written to NEW `T6c_*` files (the T6 logs were not
    overwritten, per H's instruction).
11. T7a adds two CGBA/CGBA_H cells in each CGBA root (CGBA in `non_targeted`, CGBA_H in `targeted`), matching the
    b4h_t3 reference root layout; the CGBA roots' targeted subdir holds `env_CGBA_H.json` /
    `per_checkpoint_CGBA_H.csv` / `runs_CGBA_H.csv`, exactly like the references.

## 13. sha256 of created/changed files

Files created this session (all under `reports/raw/B5h2/` unless stated):

| file | sha256 |
|---|---|
| `reports/raw/B5h2/T6c_check_contract_smoke.log` | `015f49d5839006949df867311876550c8ad515c06f791a9c1197d95cff45d448` |
| `reports/raw/B5h2/T6c_contract_smoke.md` | `d45443250af9e854fb9a7b626751d9ebdc33195107c3ab73fc20c6e682cfabed` |
| `reports/raw/B5h2/T6c_contract_smoke_status.md` | `dc679c62243934756999e441b692796e07e36df93aca085b2dc6946382a9f581` |
| `reports/raw/B5h2/T7a_EGC_r50p_NT.log` | `12cbe1df3458b00833bb3d97332eb0ceedc7a05142bc63a027f220bcd893a616` |
| `reports/raw/B5h2/T7a_EGC_r50p_T.log` | `115ee9669b3bfc1f1d8ba4ea9b8e2565242c8447ca3cf907e68e3f24d2f257e5` |
| `reports/raw/B5h2/T7a_EGC_d161p_NT.log` | `c569cd028e614f381a41eece4b18e41c4be4fcbe5096793c7e20e7643ec1dcae` |
| `reports/raw/B5h2/T7a_EGC_d161p_T.log` | `ae415468b1a7bdcef879ad0882bd86caa42b9691b55949efa9493a9ac63ef2d6` |
| `reports/raw/B5h2/T7a_TtBA_r50p_NT.log` | `6e61c2a304d086ae3ae433f06297f18f9560744e749db1ea138ee9bb3f431e35` |
| `reports/raw/B5h2/T7a_TtBA_r50p_T.log` | `726be83bfa187e5eb41f8c343820e032fed52bcd19bc319748ed7de4d53d433b` |
| `reports/raw/B5h2/T7a_TtBA_d161p_NT.log` | `db4e936c151014df6d49cd033281274afacdd0bbcaae80246977a8b7bf88780c` |
| `reports/raw/B5h2/T7a_TtBA_d161p_T.log` | `977200ad1b6b624ff2a089c543e01e8b475c79f8bb0a23dfb5a1af3897bd01b4` |
| `reports/raw/B5h2/T7a_CGBA_r50p_NT.log` | `8e0886e33e9e111fd8686351071cea312161516485bf085f70668f62ff719aa1` |
| `reports/raw/B5h2/T7a_CGBA_H_r50p_T.log` | `6a261db1fb921961ca4b21a1b13ad7a24c9ea422090c296418ccf1d18f6bcf6d` |
| `reports/raw/B5h2/T7a_CGBA_d161p_NT.log` | `30e6ee334d346fbad87d72b37b3675c79f6d26061d3022f3fdd4f5912148ece7` |
| `reports/raw/B5h2/T7a_CGBA_H_d161p_T.log` | `25dfa0f2df5e95c2e0539d24332069660e1a111754dc66ac18064dd5e47488b6` |
| `reports/raw/B5h2/T7a_compare_roots.log` | `6853d28678fecca399d0ca53e8ec979b2a9103e0bfe4cdf92f518ef8b5f2333f` |
| `reports/raw/B5h2/T7a_domain_consistency.log` | `b502d74c7aa403cec234e9e6f2116d3ef2d047ef77b511b65cfdf010979ebbcf` |
| `reports/raw/B5h2/t7a_domain_consistency.csv` | `538c94ebd717c97a1cf427620ccf12ec20ca80074ed8c15b5afdc72d1d4307bc` |
| `reports/raw/B5h2/T8_gate_main_LGC_RC_EGC_resnet50_NT.log` | `10c3d11b28d6c54a1719f1a5fed9558aa5b81eebabfaa5638b1c88d0201b2ed6` |
| `reports/raw/B5h2/T8_gate_main_TtBA_resnet50_NT.log` | `d413780c50a0f49a8a4677d61990294868092863692714a5e1f3ba0159908ae1` |
| `reports/raw/B5h2/T8_gate_main_CGBA_resnet50_NT.log` | `bf735d6d769cd781c6daa8ab7b7435f8b7f385a1a7e6f5f0ca052acc1354f112` |
| `reports/raw/B5h2/T8_gate_main_ARS_OPT_resnet50_NT.log` | `684892f5b5b839db6796f39f3f6414c3efb57b02d1f983c021214f0fd3997971` |
| `reports/raw/B5h2/T8_gate_e10_CGBA_H_resnet50_places365_T.log` | `94da53b9e3ac7b8e12ef67de83d05b1799d67e7e3f0f9d8b7cdbf303fdce4e9c` |
| `reports/raw/B5h2/T8_gate_e11_ARS_OPT_Salman2020Do_R50_NT.log` | `684892f5b5b839db6796f39f3f6414c3efb57b02d1f983c021214f0fd3997971` |
| `reports/raw/B5h2/gates_e2e/gates/summary.csv` | `ac2a00272342299a1beafabc20419e8ec65e56bee1dde0466c2939d4ee9bca7c` |
| `reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_partial.md` | `605698490fa6fa53c15ab2998fc0453f621508e5f593ae3579a9734c05848681` |
| `reports/raw/B5h2/gates_e2e/gates/main_TtBA_resnet50_NT_partial.md` | `b0092b7669e0b93e0a9d5a3ccc53f2408259c1d4c2a8e4e098c489b933086c83` |
| `reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_partial.md` | `42bee95709dbe0e83b0982a06dd27fe02717cc011151abc181cdd3a1dae9644d` |
| `reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_partial.md` | `16b619948180be3eec7214cffd15b7a6afd7b03b2b9820a9babc42b43552a92d` |
| `reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_partial.md` | `1efc6c55da108aa11855dcd7dc6bc2e808cda1ddf586541684bcdc19f70960d5` |
| `reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_partial.md` | `64534d2a68944e149c4927270eed48c8f93a80b05f44dd2dd70b665a41a85963` |
| `reports/raw/B5h2/gates_e2e/gates/*_status.md` (6 files, identical content) | `c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4` |
| `reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_domain.csv` | `1799c424d79195df228839dd674b8868bb6393fba98a75134eebacc06f2374e2` |
| `reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_domain.csv` | `270f8867635562a12e14cedc872af5badd0de2a77e7f2014730ff3c62a0df4da` |
| `reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_domain.csv` | `71bd3458412921c31dc765e019514b5db0575caad13c6ef69c538b148bacfbd7` |
| `reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_domain.csv` | `7e5f733e1af881820e7a0eb96464ba74ccb84cf219e789b618d7c75ffb447df5` |
| `reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_domain.csv` | `ab1ce6fca11a96c0eb0c6a903aa384cf54bc8bc93f4f38f455ecfb84c164fb3a` |
| `reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_domain.log` | `f415aa1d65608eb9f61d704f55fb0bdd6ae052c4dc7f8eee07dd7823b1768808` |
| `reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_domain.log` | `7ca80e7c4aa6194cacb5afefd50a7749a8dee8bf36197c2dddac24e935eb8218` |
| `reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_domain.log` | `c62df7fce576b1c6873b79006a359a76f8c79c71f30672653ebc2ae2d8db7eb6` |
| `reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_domain.log` | `45fc128cb8bd314f8887c9ee5441b3d5a45b328d142bd7ff28febb69e3e85fe6` |
| `reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_domain.log` | `8859ebdc13e677e7e4eee58804ee390fea652f78865a55788d90b7f50733a188` |
| `reports/raw/B5h2/agent_log.md` (updated) | `c34b27b992c7d8e789faf76367d6d86fa848495784c95704208f23ad05409330` |

Earlier files of this phase (from the first two sessions; unchanged since): the sha256 table of the previous report
edition (`T0_dryrun.log`, `T1_*.log`, `T3_*.log`, `T6_*.log`, `T6b_*.log`, `t3/t6_domain_consistency.*`,
`t6_human_preflight*`, `T6_per_checkpoint_summary.txt`, `T6_root_lists_diff.txt`, `T6_check_contract_smoke.log`,
`contract_smoke*.md`, the STOP bundles, the two root lists, the sitecustomize shim and its checks) — the sha256s
there are still current (nothing rewritten).

Result dirs created (data, not committed): `Results/b5h2_A`, `Results/b5h2_A2`, `Results/b5h2_B_resnet50`,
`Results/b5h2_TtBA_resnet50`, `Results/b5h2_CGBA_resnet50`, `Results/b5h2_ARS_OPT_resnet50`,
`Results/b5h2_ARS_OPT_Salman2020Do_R50`, `Results/b5h2_ARS_OPT_resnet50_places365`,
`Results/b5h2_ARS_OPT_densenet161_places365` (from the earlier sessions), plus this session:
`Results/b5h2_LGC_RC_EGC_resnet50_places365_non_targeted`, `Results/b5h2_LGC_RC_EGC_resnet50_places365_targeted`,
`Results/b5h2_LGC_RC_EGC_densenet161_places365_non_targeted`,
`Results/b5h2_LGC_RC_EGC_densenet161_places365_targeted`, `Results/b5h2_TtBA_resnet50_places365`,
`Results/b5h2_TtBA_densenet161_places365`, `Results/b5h2_CGBA_resnet50_places365`,
`Results/b5h2_CGBA_densenet161_places365`.

GPU minutes: T1 ≈ 5.5, T3 ≈ 7.7 + domain_consistency ≈ 1, T6 ≈ 12.6 (6 ARS_OPT cells) + domain_consistency ≈ 1 +
human_preflight grids ≈ 1, failed attempts + diagnostics ≈ 2 (earlier sessions ≈ 31), T7a ≈ 27.9 (12 cells) +
domain_consistency ≈ 0.2, T8 ≈ 1.5 (6 gate cells incl. 5 domain gates) → **≈ 61 GPU-minutes total**.

## 14. Summary (plain English)

1. The host is a single RTX 5090 (driver 580.105.08) with the pre-registered stack (torch 2.7.1+cu128, cuDNN 90701)
   and the correct D12 FP32 flags; quota Q=25 CPUs, 47 GB free disk; OpenCode itself runs inside a tmux session.
2. B0' passes: A == A2 bitwise and A == B at the six registered budgets (compare_roots IDENTICAL).
3. Every cross-host comparison is IDENTICAL: LGC_RC_EGC (T2), TtBA/CGBA/ARS_OPT (T3) and all 12 T7a Places365
   cells (8 compare_roots verdicts, 24 or 48 npz each, zero differences); domain_consistency X = 0 everywhere it
   is required (T3, T6, T7a); T5/T5b n/a with one GPU.
4. The T6 ARS_OPT cells (Salman + Places365) all pass their checks, and the T6 contract smoke gate now exits 0
   (T6c re-run, every invariant PASS) after H moved the two pre-existing `.ipynb_checkpoints` junk dirs out of
   `Results/`; the two T6 STOP bundles remain documented as the deviation history.
5. T8 passes end-to-end: the launcher's own `b5_gate_cell.py` exits 0 on all six tested cells (partial PASS,
   G5 PENDING/n/a, domain X = 0) with no B5a2 STOP bundle written → **verdict HOST-IDENTICAL**, B5 not started.

## RESULT

```
RESULT
phase: B5h2
head: 5a1fa5a1ef02226905f61a2e6c6dea9eca832f49
contract_sha256: ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
prompt_sha256: 0be547fb96096f968d817ca6af47c7f9d3c42520c6edc3a4f1cd9a7c6def2395
tasks: T0=PASS T1=PASS T2=PASS T3=PASS T4=PASS T5=n/a T5b=n/a T6=PASS T7a=PASS T8=PASS
files: "reports/raw/B5h2/agent_log.md c34b27b992c7d8e789faf76367d6d86fa848495784c95704208f23ad05409330"
       "reports/raw/B5h2/T6c_check_contract_smoke.log 015f49d5839006949df867311876550c8ad515c06f791a9c1197d95cff45d448"
       "reports/raw/B5h2/T6c_contract_smoke.md d45443250af9e854fb9a7b626751d9ebdc33195107c3ab73fc20c6e682cfabed"
       "reports/raw/B5h2/T6c_contract_smoke_status.md dc679c62243934756999e441b692796e07e36df93aca085b2dc6946382a9f581"
       "reports/raw/B5h2/T7a_EGC_r50p_NT.log 12cbe1df3458b00833bb3d97332eb0ceedc7a05142bc63a027f220bcd893a616"
       "reports/raw/B5h2/T7a_EGC_r50p_T.log 115ee9669b3bfc1f1d8ba4ea9b8e2565242c8447ca3cf907e68e3f24d2f257e5"
       "reports/raw/B5h2/T7a_EGC_d161p_NT.log c569cd028e614f381a41eece4b18e41c4be4fcbe5096793c7e20e7643ec1dcae"
       "reports/raw/B5h2/T7a_EGC_d161p_T.log ae415468b1a7bdcef879ad0882bd86caa42b9691b55949efa9493a9ac63ef2d6"
       "reports/raw/B5h2/T7a_TtBA_r50p_NT.log 6e61c2a304d086ae3ae433f06297f18f9560744e749db1ea138ee9bb3f431e35"
       "reports/raw/B5h2/T7a_TtBA_r50p_T.log 726be83bfa187e5eb41f8c343820e032fed52bcd19bc319748ed7de4d53d433b"
       "reports/raw/B5h2/T7a_TtBA_d161p_NT.log db4e936c151014df6d49cd033281274afacdd0bbcaae80246977a8b7bf88780c"
       "reports/raw/B5h2/T7a_TtBA_d161p_T.log 977200ad1b6b624ff2a089c543e01e8b475c79f8bb0a23dfb5a1af3897bd01b4"
       "reports/raw/B5h2/T7a_CGBA_r50p_NT.log 8e0886e33e9e111fd8686351071cea312161516485bf085f70668f62ff719aa1"
       "reports/raw/B5h2/T7a_CGBA_H_r50p_T.log 6a261db1fb921961ca4b21a1b13ad7a24c9ea422090c296418ccf1d18f6bcf6d"
       "reports/raw/B5h2/T7a_CGBA_d161p_NT.log 30e6ee334d346fbad87d72b37b3675c79f6d26061d3022f3fdd4f5912148ece7"
       "reports/raw/B5h2/T7a_CGBA_H_d161p_T.log 25dfa0f2df5e95c2e0539d24332069660e1a111754dc66ac18064dd5e47488b6"
       "reports/raw/B5h2/T7a_compare_roots.log 6853d28678fecca399d0ca53e8ec979b2a9103e0bfe4cdf92f518ef8b5f2333f"
       "reports/raw/B5h2/T7a_domain_consistency.log b502d74c7aa403cec234e9e6f2116d3ef2d047ef77b511b65cfdf010979ebbcf"
       "reports/raw/B5h2/t7a_domain_consistency.csv 538c94ebd717c97a1cf427620ccf12ec20ca80074ed8c15b5afdc72d1d4307bc"
       "reports/raw/B5h2/T8_gate_main_LGC_RC_EGC_resnet50_NT.log 10c3d11b28d6c54a1719f1a5fed9558aa5b81eebabfaa5638b1c88d0201b2ed6"
       "reports/raw/B5h2/T8_gate_main_TtBA_resnet50_NT.log d413780c50a0f49a8a4677d61990294868092863692714a5e1f3ba0159908ae1"
       "reports/raw/B5h2/T8_gate_main_CGBA_resnet50_NT.log bf735d6d769cd781c6daa8ab7b7435f8b7f385a1a7e6f5f0ca052acc1354f112"
       "reports/raw/B5h2/T8_gate_main_ARS_OPT_resnet50_NT.log 684892f5b5b839db6796f39f3f6414c3efb57b02d1f983c021214f0fd3997971"
       "reports/raw/B5h2/T8_gate_e10_CGBA_H_resnet50_places365_T.log 94da53b9e3ac7b8e12ef67de83d05b1799d67e7e3f0f9d8b7cdbf303fdce4e9c"
       "reports/raw/B5h2/T8_gate_e11_ARS_OPT_Salman2020Do_R50_NT.log 684892f5b5b839db6796f39f3f6414c3efb57b02d1f983c021214f0fd3997971"
       "reports/raw/B5h2/gates_e2e/gates/summary.csv ac2a00272342299a1beafabc20419e8ec65e56bee1dde0466c2939d4ee9bca7c"
       "reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_partial.md 605698490fa6fa53c15ab2998fc0453f621508e5f593ae3579a9734c05848681"
       "reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_partial.md 42bee95709dbe0e83b0982a06dd27fe02717cc011151abc181cdd3a1dae9644d"
       "reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_partial.md 16b619948180be3eec7214cffd15b7a6afd7b03b2b9820a9babc42b43552a92d"
       "reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_partial.md 1efc6c55da108aa11855dcd7dc6bc2e808cda1ddf586541684bcdc19f70960d5"
       "reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_partial.md 64534d2a68944e149c4927270eed48c8f93a80b05f44dd2dd70b665a41a85963"
       "reports/raw/B5h2/gates_e2e/gates/main_TtBA_resnet50_NT_partial.md b0092b7669e0b93e0a9d5a3ccc53f2408259c1d4c2a8e4e098c489b933086c83"
       "reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/main_TtBA_resnet50_NT_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
       "reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_domain.csv 1799c424d79195df228839dd674b8868bb6393fba98a75134eebacc06f2374e2"
       "reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_domain.csv 270f8867635562a12e14cedc872af5badd0de2a77e7f2014730ff3c62a0df4da"
       "reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_domain.csv 71bd3458412921c31dc765e019514b5db0575caad13c6ef69c538b148bacfbd7"
       "reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_domain.csv 7e5f733e1af881820e7a0eb96464ba74ccb84cf219e789b618d7c75ffb447df5"
       "reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_domain.csv ab1ce6fca11a96c0eb0c6a903aa384cf54bc8bc93f4f38f455ecfb84c164fb3a"
       "reports/raw/B5h2/gates_e2e/gates/main_LGC_RC_EGC_resnet50_NT_domain.log f415aa1d65608eb9f61d704f55fb0bdd6ae052c4dc7f8eee07dd7823b1768808"
       "reports/raw/B5h2/gates_e2e/gates/main_CGBA_resnet50_NT_domain.log 7ca80e7c4aa6194cacb5afefd50a7749a8dee8bf36197c2dddac24e935eb8218"
       "reports/raw/B5h2/gates_e2e/gates/main_ARS_OPT_resnet50_NT_domain.log c62df7fce576b1c6873b79006a359a76f8c79c71f30672653ebc2ae2d8db7eb6"
       "reports/raw/B5h2/gates_e2e/gates/e10_CGBA_H_resnet50_places365_T_domain.log 45fc128cb8bd314f8887c9ee5441b3d5a45b328d142bd7ff28febb69e3e85fe6"
       "reports/raw/B5h2/gates_e2e/gates/e11_ARS_OPT_Salman2020Do_R50_NT_domain.log 8859ebdc13e677e7e4eee58804ee390fea652f78865a55788d90b7f50733a188"
deviation_note: T6 STOP history (E-ENV shim, E-SPEC junk dirs) kept as documented deviations 5/8/9/10; resolved by H's decisions
report_sha256: given in the closing message (the report's own sha256, computed after the final write)
deviations: 11 (listed in the report)
stop_bundles: reports/stop/B5h2_T6_20261020-0242.md reports/stop/B5h2_T6b_20261020-0310.md
verdict: READY-FOR-REVIEW
```

Verdict line: **HOST-IDENTICAL** — T2, T3 and T7a all IDENTICAL (bitwise, every method / victim / mode on the
pre-registered stack torch 2.7.1+cu128 / cuDNN 90701), T5/T5b n/a at NGPU = 1, and T8 PASS (gate end-to-end exit 0
on every tested cell). B5 not started; no `Results/b5_*` or `Results/appendix_cgba_d8only_*` directory created.
STOP.
