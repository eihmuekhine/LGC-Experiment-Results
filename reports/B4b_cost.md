# B4b — Smoke timing + cost table for E10 (Places365) / E11 (RobustBench) (gate G4, reduced roster)

Status: **COMPLETE (2026-10-12)**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`. Engine sha
`835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` (frozen, unchanged). GPU: RTX 5090 (32 GB),
torch 2.7.1+cu128. Contract **v1.5** (D9: E10/E11 REQUIRED). Budget 10000, split rows 1–3 (smoke), `n = 3`.

Reduced roster (contract D9): `LGC_RC_EGC`, `TtBA`, `CGBA` (non-targeted only), `CGBA_H` (targeted only).
**HSJA, Sign-OPT and ARS_OPT were NOT run here.**

## 1. Infrastructure (Step 1 & 2)

- `adapters/_common.py`:
  - `load_victim_any(arch, device, ckpt_dir)` dispatches `resnet50_places365`/`densenet161_places365` ->
    `build_places365_victim`, `Salman2020Do_R50` -> `build_robust_victim`, else the main-ImageNet `load_victim`.
  - `load_x0_any(arch, name, device, data_dir)` dispatches the Places365 / ttba-mini-flat / ImageNet loaders
    (all use the TtBA protocol Resize 256 -> CenterCrop 224).
  - `raw_net(victim)` extracts the un-normalised network CGBA feeds the normalised tensor into
    (`_Norm.m` / `NormalizedVictim.net` / RobustBench `Sequential.model`).
  - `make_parser --arch` accepts the three new victims; `run()` now uses `load_victim_any` + `load_x0_any`.
- `adapters/run_cgba.py`: `_faithful_model` now delegates to `C.raw_net` (so the E10/E11 raw nets are used).
- `adapters/run_ttba_v2.py`: new `--arch` + `_main_generic` path runs the SAME official `TtBA.Attacker` with the
  E10/E11 victim (wrapped in `GeneralTorchModel`, `im_mean=None`) and loader.
- `ttba_github/run_frame_b1_generic.py` (new): runs the frozen engine on E10/E11.

## 2. Engine entry point reused for EGC on E10/E11 (unchanged, frozen)

| component | file:line |
|---|---|
| `GeometricAttack` (constructor) | `lgc_bench/engine.py:1097` |
| `GeometricAttack.run()` | `lgc_bench/engine.py:5442` |
| `HardLabelOracle` (counted top-1 oracle) | `lgc_bench/spaces.py:164` |
| `EngineConfig` (frozen defaults) | `lgc_bench/engine.py:795` |
| `REGISTRY["LGC_RC_EGC"]` (`space="latent_ra"`) | `lgc_bench/engine.py:656` |
| spaces (`LatentRAGSpace` -> `LatentResidualSpace` -> `AmbientResidualSpace`) | `lgc_bench/spaces.py:56/:73`, `lgc_bench/ambient.py:21` |

`run_frame_b1_generic.py` builds the same `EngineConfig(max_queries, checkpoints, query_batch=64)` (all other
fields = dataclass defaults), the same autoencoder (`checkpoints/vgg16_imagenet.pth`, arch `vgg16`), the same
`latent_ra` space chain, then calls `GeometricAttack(...).run()`. It never modifies `lgc_bench/engine.py`.

## 3. Smoke timing (rows 1–3, one process, 10 000 queries)

`s/img` = mean per-image `wall_time_s` from each cell's `runs*.csv` (3 images). `peak MB` = max `nvidia-smi`
memory.used sampled every 0.5 s. All 18 cells completed with **zero errors** (status `ok` for ours, `budget` =
`BudgetExhausted` for baselines; no `error` row, no query > 10 000, no STOP event).

| method | arch | mode | s/img | peak MB | GPU-h (200) |
|---|---|---|---|---|---|
| LGC_RC_EGC | resnet50_places365 | non_targeted | 21.7 | 9480 | 1.206 |
| LGC_RC_EGC | resnet50_places365 | targeted | 24.4 | 9486 | 1.356 |
| LGC_RC_EGC | densenet161_places365 | non_targeted | 32.8 | 9498 | 1.822 |
| LGC_RC_EGC | densenet161_places365 | targeted | 34.5 | 9502 | 1.917 |
| LGC_RC_EGC | Salman2020Do_R50 | non_targeted | 22.1 | 9470 | 1.228 |
| LGC_RC_EGC | Salman2020Do_R50 | targeted | 24.4 | 9476 | 1.356 |
| TtBA | resnet50_places365 | non_targeted | 39.5 | 4546 | 2.194 |
| TtBA | resnet50_places365 | targeted | 40.3 | 4546 | 2.239 |
| TtBA | densenet161_places365 | non_targeted | 107.3 | 4568 | 5.961 |
| TtBA | densenet161_places365 | targeted | 108.8 | 4570 | 6.044 |
| TtBA | Salman2020Do_R50 | non_targeted | 41.6 | 4550 | 2.311 |
| TtBA | Salman2020Do_R50 | targeted | 40.5 | 4550 | 2.250 |
| CGBA | resnet50_places365 | non_targeted | 20.3 | 1880 | 1.128 |
| CGBA | densenet161_places365 | non_targeted | 34.5 | 2256 | 1.917 |
| CGBA | Salman2020Do_R50 | non_targeted | 21.3 | 1878 | 1.183 |
| CGBA_H | resnet50_places365 | targeted | 21.7 | 1878 | 1.206 |
| CGBA_H | densenet161_places365 | targeted | 36.4 | 2254 | 2.022 |
| CGBA_H | Salman2020Do_R50 | targeted | 21.6 | 1878 | 1.200 |

Notes:
- Peak memory max **9.5 GB** (LGC_RC_EGC / densenet161), comfortably inside the 32 GB card. Ours ~9.5 GB,
  TtBA ~4.5–4.6 GB, CGBA/CGBA_H ~1.9–2.3 GB.
- **TtBA on densenet161 is the outlier** (~107–109 s/img, ≈ 6 GPU-h per cell; densenet161's single-image forward
  is much slower than resnet50). This dominates the E10 cost.

## 4. Updated cost table (main + E10 + E11)

GPU-h per cell = 200 × s/img / 3600. Main study from `reports/B4_cost.md` (24 cells); E10/E11 measured here.

| scope | cells | GPU-h |
|---|---|---|
| main ImageNet (B4, v1.4 roster incl. ARS_OPT) | 24 | 38.88 |
| E10 Places365 (2 victims × 2 modes × 3 methods) | 12 | 29.01 |
| E11 RobustBench (1 victim × 2 modes × 3 methods) | 6 | 9.53 |
| **total (main + E10 + E11)** | **42** | **77.42** |

Per-method breakdown (E10 + E11, reduced roster):

| method | E10 cells | E11 cells | E10+E11 GPU-h |
|---|---|---|---|
| LGC_RC_EGC (ours) | 4 | 2 | 8.88 |
| TtBA | 4 | 2 | 21.00 |
| CGBA (NT) | 2 | 1 | 4.23 |
| CGBA_H (T) | 2 | 1 | 4.43 |
| **E10+E11 total** | 12 | 6 | **38.54** |

(The main 24-cell roster uses 4 methods incl. ARS_OPT; E10/E11 use the reduced 3-method roster per contract D9.)

## 5. Contract smoke gate (Step 4)

```
python tools/eval/check_contract.py --stage smoke --n 3 \
  --column_map contract/column_map.yaml \
  --split_dir splits/imagenet/ttba_github splits/places365 splits/robust \
  --roots Results/b4b_smoke_* \
  --report reports/contract_smoke_b4b.md
```

**Exit code 0 — PASS.** 432 per_checkpoint rows (18 cells × 3 images × 8 checkpoints), 54 runs rows, 432 stored
npz, 15 env files.

| check | verdict |
|---|---|
| I01–I08 | PASS |
| I09 | WARN (informational: targeted `found=0` at budget 100/200 = `0/9` per victim — the initial binary search consumes > 200 queries in targeted mode) |
| I10–I12 | PASS |
| DISK | PASS (37.8 GB free ≥ 25 GB) |

`reports/contract_smoke_b4b.md` is the authoritative record (regenerated by the checker).

## 6. Errors / STOP events

**None.** All 18 cells exited 0; statuses are `ok` (LGC_RC_EGC, 18) and `budget` (TtBA 18, CGBA 9, CGBA_H 9);
no `error` row and no query count above 10 000.

## 7. sha256 of new/changed files

| file | change | SHA256 |
|---|---|---|
| `adapters/_common.py` | `load_victim_any`, `raw_net`, `load_x0_any`, `make_parser` archs, `run()` dispatch | `99062a673c502c862ae5d178684a42e8412d1013c7c06a03eac42a6480031f83` |
| `adapters/run_cgba.py` | `_faithful_model` -> `C.raw_net` | `4c3fdf714ea7cfbd91224346cf8239fcb5b6e569acbf31c0547cd1f08d3b9a08` |
| `adapters/run_ttba_v2.py` | `--arch` + `_main_generic` E10/E11 path | `d97292408ff1637825af653cc81ee3c4afa0ef8abbf73fec21aa12a894b0ac13` |
| `ttba_github/run_frame_b1_generic.py` | new generic EGC runner | `ce7b8a59cf9b21592ca99a6e9e6967d4fe329e7d0ed76063b83cf1e60e6aa181` |
| `reports/contract_smoke_b4b.md` | generated smoke gate report | `83e66b8beb65dc709b743dd6faeff9a9bf0d37d5a9f71a8d2f17bda7888f6569` |

Frozen engine unchanged: `sha256sum lgc_bench/engine.py` = `835f1933d9e5…` (verified; not edited).

Tests still green (AGENTS.md rule 8): `tools/eval/tests_contract/test_contract.py` **15/15**,
`tools/eval/selftest_evalkit.py` **20/20**.

## 8. Next

**STOP.** No B5 work was started. The user must approve this cost table (main 38.88 + E10 29.01 + E11 9.53 =
77.42 GPU-h, or any subset) before any n = 200 run begins.
