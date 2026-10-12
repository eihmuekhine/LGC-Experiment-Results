# B4 — Smoke timing + cost table (gate G4)

Status: **COMPLETE (2026-10-11/12)**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD
`f17f15db873e32420d327dca19310c23bba2f3cc`. Engine sha `835f1933d9e5` (frozen). GPU: RTX 5090 (32 GB),
driver 595.71.05, torch 2.7.1+cu128. Contract v1.4. Budget 10000, split rows 1–3 (smoke), `n = 3`.

## 1. Roster (contract v1.4, D7)

Run: `LGC_RC_EGC` (ours), `TtBA`, `CGBA` (NT) / `CGBA_H` (T), `ARS_OPT`. **HSJA and Sign-OPT are NOT run**
(cited only, D7). One attack process per GPU. Cells = method × {vgg19, resnet50, vit_b_32} × {NT, T}:
4 methods × 6 = 24 cells (CGBA split NT-only, CGBA_H T-only), 72 images total.

## 2. Smoke timing (rows 1–3, one process, 10 000 queries)

`s/img` = mean per-image `wall_time_s` from each cell's `runs*.csv` (3 images). `peak MB` = max
`nvidia-smi` memory.used sampled every 0.5 s during the run. All 24 cells completed with **zero errors**
(status `ok` for ours, `budget` = `BudgetExhausted` for baselines; no `error` row, no query > 10 000).

| method | arch | mode | s/img | peak MB | GPU-h (200) |
|---|---|---|---|---|---|
| LGC_RC_EGC | resnet50 | non_targeted | 22.86 | 9486 | 1.270 |
| LGC_RC_EGC | resnet50 | targeted | 24.68 | 9492 | 1.371 |
| LGC_RC_EGC | vgg19 | non_targeted | 24.84 | 9940 | 1.380 |
| LGC_RC_EGC | vgg19 | targeted | 29.23 | 9940 | 1.624 |
| LGC_RC_EGC | vit_b_32 | non_targeted | 21.54 | 9756 | 1.197 |
| LGC_RC_EGC | vit_b_32 | targeted | 23.77 | 9756 | 1.321 |
| TtBA | resnet50 | non_targeted | 46.21 | 4550 | 2.567 |
| TtBA | resnet50 | targeted | 46.16 | 4554 | 2.565 |
| TtBA | vgg19 | non_targeted | 29.48 | 5150 | 1.638 |
| TtBA | vgg19 | targeted | 29.03 | 5020 | 1.613 |
| TtBA | vit_b_32 | non_targeted | 42.29 | 4808 | 2.349 |
| TtBA | vit_b_32 | targeted | 41.99 | 4812 | 2.333 |
| CGBA | resnet50 | non_targeted | 21.18 | 1878 | 1.176 |
| CGBA_H | resnet50 | targeted | 21.20 | 1878 | 1.178 |
| CGBA | vgg19 | non_targeted | 22.58 | 4232 | 1.255 |
| CGBA_H | vgg19 | targeted | 21.12 | 4230 | 1.174 |
| CGBA | vit_b_32 | non_targeted | 21.14 | 1222 | 1.175 |
| CGBA_H | vit_b_32 | targeted | 21.91 | 1222 | 1.217 |
| ARS_OPT | resnet50 | non_targeted | 30.94 | 4094 | 1.719 |
| ARS_OPT | resnet50 | targeted | 30.85 | 4094 | 1.714 |
| ARS_OPT | vgg19 | non_targeted | 31.30 | 11930 | 1.739 |
| ARS_OPT | vgg19 | targeted | 31.00 | 11928 | 1.722 |
| ARS_OPT | vit_b_32 | non_targeted | 32.19 | 2314 | 1.788 |
| ARS_OPT | vit_b_32 | targeted | 32.32 | 2314 | 1.796 |

Notes:
- Peak memory is dominated by the victim + attack workspace; the max across all cells is **11.9 GB**
  (ARS_OPT / vgg19), comfortably inside the 32 GB card. Ours peaks ~9.5–9.9 GB, TtBA ~4.5–5.2 GB,
  CGBA ~1.2–4.2 GB, ARS_OPT ~2.3–11.9 GB.
- For LGC_RC_EGC, vgg19 and vit_b_32 were run as a single per-arch invocation (both modes in one process),
  so the two modes of each arch share one peak-memory reading (9940 / 9756 MB); resnet50 was timed per mode.

## 3. Two-process concurrency test (once)

Cell: TtBA / resnet50 / non_targeted, rows 1–3, `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`,
two independent processes on the same images (separate out dirs).

| metric | solo | 2 concurrent |
|---|---|---|
| per-image wall time | 46.21 s/img | 53.62 s/img (per process) |
| throughput (images/s) | 0.0216 | 0.0373 (combined) |
| **throughput ratio** | 1.00× | **1.72×** |
| wall-clock for 6 images | 277 s | 168 s (1.65×) |
| peak GPU memory | 4550 MB | 9093 MB |

**Verdict: 2 processes give ≈ 1.72× throughput (≥ 1.6× threshold), so running 2 processes is permitted, but
only marginally so.** Peak memory doubles to ~9.1 GB (still far below 32 GB). Recommendation: for the main runs,
use 2 processes per GPU only for the slowest methods (TtBA, ARS_OPT) if wall-clock is the binding constraint;
otherwise stay at 1 process to keep per-process `s/img` at the solo numbers the cost table is based on. (GPU-h
is computed from solo `s/img`; concurrency reduces wall-clock, not GPU-h.)

## 4. Contract smoke gate

```
python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage smoke --n 3 \
  --column_map contract/column_map.yaml \
  --roots Results/b4_smoke_egc_{resnet50,vgg19,vit_b_32} Results/b4_smoke_ttba_{...} \
          Results/b4_smoke_cgba_{...} Results/b4_smoke_arsopt_{...} \
  --report reports/contract_smoke.md
```

**Exit code 0 — PASS for every method × victim × mode.** I01–I12 all PASS; I09 WARN (informational: targeted
`found=0` at budgets 100/200 is `0/12` in every cell — all images are already adversarial at the low checkpoints);
DISK PASS (38.9 GB free ≥ 25 GB).

`reports/contract_smoke.md` (regenerated) is the authoritative record; 576 per_checkpoint rows (24 cells × 3
images × 8 checkpoints), 72 runs rows, 576 stored npz, 21 env files.

### 4a. Checker fixes required by this gate (documented, not check-weakening)

The first smoke run FAILed on I01/I06/I08/I10. All five were **checker defects**, not data defects; the data
was already correct (I02/I03/I04/I05/I07/I11/I12 passed on the same inputs). Fixes in `tools/eval/check_contract.py`
(sha256 after: `76be8e0c82a9129e8b6f62fc2a737140afe39dc39ce7badef7b6ed999a144219`):

1. **I01/I05/I06/I08 omitted `mode` from their cell keys.** The same 3 source images legitimately appear in both
   the NT and T split CSVs, so NT and T rows collided as "duplicates", and the npz index resolved a T row to the NT
   npz (hence the spurious I06 L2 mismatches and I08 monotonicity violations). `mode` is now part of the
   `(arch, mode, image, method, budget)` key everywhere (`load_npz_index`, I01, I05, I06, I08, and
   `completed_pairs` for the `partial` stage). This is *stricter*, matching how I02/I03 already keyed by
   `(arch, mode)`.
2. **I10 GPU-name normalisation.** `GPU_NAME_RE` was non-greedy (matched `"GeForce R"`) and was compared against
   the full `torch.cuda.get_device_name` string (`"NVIDIA GeForce RTX 5090"`), so a single physical GPU was seen as
   two names. The regex is now greedy and a `_norm_gpu` helper strips the vendor prefix / normalises case+whitespace
   before comparison.

Both suites still pass after the fixes: `tools/eval/tests_contract/test_contract.py` **15/15**, and
`tools/eval/selftest_evalkit.py` **20/20** (AGENTS.md rule 8). No contract YAML, result folder, or adapter file was
edited; only the checker was corrected to implement its own documented invariants.

## 5. Disk

`df -h /root/autodl-tmp`: 50 G total, 12 G used, **39 G avail** — satisfies the ≥ 25 GB requirement (contract
`artefacts.disk_estimate`).

## 6. Cost table — totals per method (full roster)

GPU-h per cell = 200 × s/img / 3600. Total = sum over the method's cells.

| method | cells | GPU-h |
|---|---|---|
| LGC_RC_EGC (ours) | 6 (all victims × both modes) | 8.16 |
| TtBA | 6 | 13.07 |
| CGBA (NT) | 3 | 3.61 |
| CGBA_H (T) | 3 | 3.57 |
| ARS_OPT | 6 | 10.48 |
| **full roster** | 24 | **38.88** |

## 7. Plan in priority order (skill lgc-baselines §6)

Ours (`LGC_RC_EGC`) is always run (every comparison needs it). Baselines are then added in the priority order
`TtBA → CGBA/CGBA-H → ARS-OPT`, stopping when the approved GPU-h budget is used.

| step | add | cumulative GPU-h | includes |
|---|---|---|---|
| 0 | LGC_RC_EGC (ours) | 8.16 | all 6 cells |
| 1 | + TtBA | 21.23 | ours + TtBA |
| 2 | + CGBA (NT) + CGBA_H (T) | 28.40 | ours + TtBA + CGBA/CGBA-H |
| 3 | + ARS_OPT | 38.88 | full roster |

For a budget of `X` GPU-h:
- `X < 8.16` → not enough for the full ours-only study; either reduce `n` or cut victims (not recommended).
- `8.16 ≤ X < 21.23` → run ours (all 6) + TtBA partially (TtBA cells in order resnet50 → vit_b_32 → vgg19, since
  vgg19 is cheapest; see per-cell table §2).
- `21.23 ≤ X < 28.40` → ours + TtBA + CGBA/CGBA-H partially.
- `X ≥ 38.88` → full roster.

Wall-clock: GPU-h above is sequential solo time. With 2 processes (1.72× throughput, §3), wall-clock ≈
`GPU-h / 1.72` (e.g. full roster ≈ 22.6 h instead of ≈ 38.9 h).

## 8. Next

**STOP.** No B5 work was started. The user must approve this cost table and choose the GPU-h budget before any
n = 200 run begins.
