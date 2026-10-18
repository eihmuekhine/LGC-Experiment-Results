# B5-host — host identity probe before the b5-go tag

Status: **COMPLETE** (probe). Verdict: **HOST-DIFFERENT** — T2, T3 and T7a all DIFFERENT bitwise against the B4e2/B4h
smoke roots; T5 (intra-host, per-GPU) is IDENTICAL. Worktree `/root/autodl-tmp/LGC_frozen`, HEAD
`fdda03a79794029b0264acd8cdb81a28b439ed36`. B5 not started.

Active LESSONS rows read at phase start (skill §7): **L01** (encoded: D12 backend flags), **L02** (CGBA acos clip),
**L03** (GPU/host drift between sessions), **L06** (prompt archived + sha256), **L22 / L23** (D19 oracle domain,
domain_consistency gate). None contradicted.

## 1. Scope and prompt

- Prompt hash (sha256 of `reports/raw/B5-host/prompt.txt`):
  `7e73a26f1fb45f805d0d5a679e9fb2373b9fc559cda7f793bdc5cc53c956e79f`
- Contract `contract/paper_contract.yaml` **v1.9-rev5** sha256:
  `5b8af155b463dfb8cc6cf65d499a0806ae048ba5ee6b0ac79d616309219e204a`
- `git rev-parse HEAD`: `fdda03a79794029b0264acd8cdb81a28b439ed36`
- Every GPU command ran with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` and was tee'd to `reports/raw/B5-host/<step>.log`.
- STOP bundles written this phase: `reports/stop/B5-host_T1_20261018-0808.md` (missing packages; resolved by H's
  permission to install), `reports/stop/B5-host_T6_20261018-0848.md` (E-SPEC; still open, see T6 below).

## 2. Environment block (T0)

| field | value |
|---|---|
| hostname | `autodl-container-30d64fa2f6-699e3b58` |
| GPU | 5 x `NVIDIA GeForce RTX 5090` (driver 580.105.08, CUDA 12.8, 32607 MiB, capability (12, 0), 170 SM per GPU) |
| torch | **2.8.0+cu128** (reference B4e2/B4h roots: **2.7.1+cu128**) |
| cudnn | **91002** (reference: 90701) |
| CPU | Intel Xeon Platinum 8470Q; **`nproc` = 8** (container cgroup limit; 208 host CPUs) |
| RAM | 754 GB total, ~670 GB available |
| backend flags | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `float32_matmul_precision=highest` (D12; identical in old and new runs) |
| HEAD | `fdda03a79794029b0264acd8cdb81a28b439ed36` |
| engine sha | `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` (frozen) |

Reference roots' recorded environment (from their `env_*.json` / `env_frame.txt`; raw table in
`reports/raw/B5-host/t0_host.log`):

| root (abbreviated) | GPU | driver | cudnn | torch |
|---|---|---|---|---|
| b4e2_smoke_LGC_RC_EGC_{resnet50,vgg19,vit_b_32} | RTX 5090 | 580.105.08 | 90701 | 2.7.1+cu128 |
| b4e2_smoke_LGC_RC_EGC_{places365 x4, Salman x2} | RTX 5090 | not recorded | not recorded | 2.7.1+cu128 |
| b4e2_smoke_ttba_* (6) | RTX 5090 | not recorded | not recorded | 2.7.1+cu128 |
| b4h_t3_smoke_CGBA_* (6) | RTX 5090 | not recorded | not recorded | 2.7.1+cu128 |
| b4h_t3_smoke_ARS_OPT_* (3) | RTX 5090 | not recorded | not recorded | 2.7.1+cu128 |

## 3. T1 — B0' on this host (resnet50 NT, rows 1–3)

| run | out_dir | command | result |
|---|---|---|---|
| A | `Results/b5host_A` | `run_benchmark_33b.py … --checkpoints 500 1000 2000 2500 5000 10000` (registered budgets, no `--save_adv`) | 3/3 ok |
| A2 | `Results/b5host_A2` | identical command | 3/3 ok |
| B | `Results/b5host_B_resnet50` | `MODES=0 N=3 OUT_PREFIX=Results/b5host_B bash ttba_github/run_frame_b1.sh resnet50` (contract budgets 100…10000, `--save_adv`) | 3/3 ok |

`run_frame_b1.sh` hardcodes contract checkpoints + `--save_adv`, so A/A2 ran the identical
`run_benchmark_33b.py` command line that the script builds (script not editable); B ran the script itself, whose
`OUT` is `OUT_PREFIX_<arch>` (hence the `_resnet50` suffix; also matches the naming of the B4e2 smoke roots).

compare_roots verdicts (last 3 lines each):

```
A vs A2:  checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
          npz files: shared 0, only in A 0, only in B 0; differences: 0
          VERDICT: IDENTICAL
A vs B (registered budgets):  checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
          npz files: shared 0, only in A 0, only in B 0; differences: 0
          VERDICT: IDENTICAL
```

**PASS** — A == A2 bitwise at every budget; A == B at the 6 registered budgets (B's extra 100/200 rows are
only-in-B, not differences).

## 4. T2 — cross-host identity, LGC_RC_EGC resnet50 NT (information)

compare_roots `Results/b4e2_smoke_LGC_RC_EGC_resnet50` vs `Results/b5host_B_resnet50` (last 3 lines):

```
checkpoint differences: 24; max |pix_l2 A - B| = 6.309e-01
npz files: shared 24, only in A 24, only in B 0; differences: 24
VERDICT: DIFFERENT (first: ('non_targeted', 'resnet50', 'LGC_RC_EGC', 'ILSVRC2012_val_00007450', 100):
A(found=1, state_query=68, pix_l2=49.81208038330078) B(found=1, state_query=68, pix_l2=49.81200408935547))
```

First divergence: image `ILSVRC2012_val_00007450.JPEG`, budget **100** (pix_l2 differs by 7.6e-5; found and
state_query still equal there). All 24 shared non_targeted checkpoints differ in pix_l2, growing with budget
(≈0.63 at 10000); every npz delta differs. `found` and `state_query` stay aligned at low budgets and drift from
budget 2500. Configs are identical except the build (tf32=False, deterministic=True, same seed/exclude/seeds,
same checkpoints; device lines: `torch=2.7.1+cu128` vs `torch=2.8.0+cu128`). The trajectory divergence is
attributable to the torch/cuDNN build change (2.7.1/cudnn 90701 → 2.8.0/cudnn 91002) on identical hardware.

**Verdict: DIFFERENT — hosts may NOT be pooled bitwise for LGC_RC_EGC.**

## 5. T3 — cross-host identity, TtBA / CGBA (fixed) / ARS_OPT, resnet50 NT rows 1–3 (D19 domains)

Runs: `Results/b5host_TtBA_resnet50` (adapter sha `c05876ac…`, exact seed mode, limit 3),
`Results/b5host_CGBA_resnet50` (`--cgba_variant fixed`, adapter sha `7596d798…` == reference),
`Results/b5host_ARS_OPT_resnet50` (adapter sha `0f454a78…` == reference); tracker sha `d5ca5098…` == reference.
Current `_common.py` differs from the B4h-t3 reference only in observation-only ways (oob_queries column,
`raw_net` strictness — resnet50 exposes `.m`, so no behaviour change).

compare_roots first divergences:

| method | ref root | first diverging (mode, image, budget) | A | B |
|---|---|---|---|---|
| TtBA | b4e2_smoke_ttba_resnet50 | non_targeted, ILSVRC2012_val_00007450, **500** (budgets 100/200 identical) | sq=461, pix_l2=59.87986755371094 | sq=461, pix_l2=59.41632843017578 |
| CGBA | b4h_t3_smoke_CGBA_resnet50 | non_targeted, ILSVRC2012_val_00007450, **200** | sq=190, pix_l2=98.52855682373047 | sq=191, pix_l2=98.52857208251953 |
| ARS_OPT | b4h_t3_smoke_ARS_OPT_resnet50 | non_targeted, ILSVRC2012_val_00007450, **500** | sq=479, pix_l2=103.20997619628906 | sq=479, pix_l2=103.20999908447266 |

Final verdict lines: `VERDICT: DIFFERENT` for all three (TtBA: 21/24 checkpoint rows differ, 22 npz; CGBA: all
rows/npz; ARS_OPT: all rows/npz).

**oob_queries** (new observation column; no reference values exist — the counter postdates the reference runs):

| method | image | oob_queries (of 10000) | nonfinite | status |
|---|---|---|---|---|
| TtBA | 07450 / 15433 / 25151 | 8435 / 8547 / 8458 | 0 | budget x3 |
| CGBA | 07450 / 15433 / 25151 | 9799 / 7697 / 9828 | 0 | budget x3 |
| ARS_OPT | 07450 / 15433 / 25151 | 10000 / 10000 / 10000 | 0 | budget x3 |

domain_consistency on the three new roots (log `t3_domain_consistency.log`, 72 rows):
**X = 0 for TtBA, CGBA and ARS_OPT** (`[+] no X>0 for any method`). Gate PASS.

**Verdict: DIFFERENT for all three methods — hosts may NOT be pooled bitwise.**

## 6. T4 — s/img on this host vs the reference roots (resnet50 NT, rows 1–3, mean of 3)

| method | ref s/img (B4e2/B4h) | this host s/img | ratio |
|---|---|---|---|
| LGC_RC_EGC | 23.34 | 20.94 | 0.90 |
| TtBA | 52.76 | 49.97 | 0.95 |
| CGBA | 26.22 | 22.19 | 0.85 |
| ARS_OPT | 34.55 | 33.87 | 0.98 |

Throughput is comparable (0.85–0.98x); no slowdown finding.

## 7. T5 — per-GPU concurrency identity (row 1, 5 GPUs x {EGC, TtBA} launched together)

All 10 processes completed; `nproc = 8` (finding: < 40 = 8 x 5), `free -g`: 754 GB total / ~670 available,
`OMP_NUM_THREADS = MKL_NUM_THREADS = 8`.

compare_roots GPU i vs GPU 0 (last line of every comparison):

| method | gpu1 vs gpu0 | gpu2 vs gpu0 | gpu3 vs gpu0 | gpu4 vs gpu0 |
|---|---|---|---|---|
| LGC_RC_EGC | **IDENTICAL** (0 diffs, 8/8 npz) | IDENTICAL | IDENTICAL | IDENTICAL |
| TtBA | **IDENTICAL** | IDENTICAL | IDENTICAL | IDENTICAL |

Timing under 10-way concurrency vs T4 solo:

| GPU index | EGC s/img under concurrency | TtBA s/img under concurrency |
|---|---|---|
| 0 | 28.90 | 69.46 |
| 1 | 29.06 | 69.72 |
| 2 | 28.88 | 69.68 |
| 3 | 29.11 | 69.72 |
| 4 | 29.31 | 69.69 |

| method | solo s/img (T4) | concurrency s/img (mean) | ratio |
|---|---|---|---|
| LGC_RC_EGC | 20.94 | 29.05 | 1.39 |
| TtBA | 49.97 | 69.65 | 1.39 |

**Intra-host verdict: all 5 GPUs bitwise IDENTICAL to GPU 0 for both methods.** Concurrency costs ~1.4x per image
at nproc=8 (10 processes x OMP=8 on 8 cores — oversubscribed; B5 with 5 processes would be less severe but still
CPU-limited: the 40-core requirement of the prompt is not met).

## 8. T6 — ARS_OPT on Salman2020Do_R50 (correctness check, rows 1–3, both modes)

Victim construction: `adapters/_common.py:224` `build_robust_victim` →
`rbu.load_model(ROBUST_MODEL_NAME, model_dir=ROBUST_MODEL_DIR, dataset="imagenet", threat_model="Linf")` (:229),
dispatched from `load_victim_any` (:170). **Same code path as every other E11 method**: EGC
(`run_frame_b1_generic.py:99`), TtBA/CGBA/ARS_OPT (`_common.run` → `load_victim_any` :693). Observed victim:
`torch.nn.Sequential(['normalize','model'])` — internal ImageNet normalisation, as documented.

| mode | image | queries | oob | status | s/img | pix_l2 @ {100, 1k, 10k} |
|---|---|---|---|---|---|---|
| non_targeted | ILSVRC2012_val_00010608 | 10000 | 10000 | ok | 33.7 | 33.3 / 27.0 / 12.7 |
| non_targeted | ILSVRC2012_val_00046453 | 10000 | 10000 | budget | 33.5 | 34.4 / 27.6 / 15.0 |
| non_targeted | ILSVRC2012_val_00024509 | 10000 | 10000 | budget | 33.1 | 6.0 / 3.4 / 1.1 |
| targeted | ILSVRC2012_val_00010608 | 10000 | 9932 | budget | 35.0 | 99.5 / 76.8 / 69.4 |
| targeted | ILSVRC2012_val_00046453 | 10000 | 9737 | budget | 34.0 | 119.2 / 110.7 / 89.9 |
| targeted | ILSVRC2012_val_00024509 | 10000 | 9959 | budget | 34.4 | 135.5 / 121.5 / 82.5 |

`found` = 1 at **every** checkpoint for **every** image/mode; pix_l2 monotonically decreasing; queries = 10000
everywhere. domain_consistency: **X = 0** for both modes (`t6_domain_consistency.log`). human_preflight:
**FLAG lines = 0** (`t6_human_preflight.md`). All T6 acceptance checks PASS **except the contract gate**:

- Appended `Results/b5host_ARS_OPT_Salman2020Do_R50` to `reports/B4e2_smoke_roots.txt` (also appended the two
  T7b roots) and re-ran `check_contract --stage smoke` → **exit 1**, three independent causes (STOP bundle
  `reports/stop/B5-host_T6_20261018-0848.md`, code E-SPEC):
  1. **I10 FAIL — contract regression**: v1.9-rev5 `constants.split_sha256` lacks the 4 Places365 keys (removed in
     the rev2/3 "ImageNet-only" edits; rev5 restored E10 to REQUIRED without restoring them). Fails on any host.
  2. **I10 FAIL — torch heterogeneity**: the new root records 2.8.0+cu128 vs 2.7.1+cu128 in every reference root;
     this is exactly the HOST-DIFFERENT evidence of T2/T3/T7a.
  3. **I01 FAIL — junk dirs**: `.ipynb_checkpoints/` checkpoint CSVs inside
     `Results/b4h_t3_smoke_ARS_OPT_{resnet50,vgg19}` (left by the previous host's notebook sessions) duplicate
     rows. Cleaning Results/ is outside my allowed edits.
  My first invocation also omitted `--split_dir splits/imagenet/ttba_github splits/places365 splits/robust`
  (I02 WARN); re-running with those args does not change the three FAILs. **Awaiting a human decision** (contract
  fix per contract-skill rule 2, Results cleanup, and the torch-heterogeneity policy).

## 9. T7a — ImageNet victims on Places365 (rows 1–3, both modes, LGC_RC_EGC / TtBA / CGBA-CGBA_H)

12 new cells in `Results/b5host_<method>_<victim>[_<mode>]`. First divergence per cell (compare_roots):

| method | cell | first diverging (image, budget) | A | B |
|---|---|---|---|---|
| LGC_RC_EGC | resnet50_places365 NT | Places365_val_00002172, 100 | 27.14217185974121 | 27.142179489135742 |
| LGC_RC_EGC | resnet50_places365 T | Places365_val_00002172, 100 | 81.71736907958984 | 81.71753692626953 |
| LGC_RC_EGC | densenet161_places365 NT | Places365_val_00002172, 100 | 20.221662521362305 | 20.221670150756836 |
| LGC_RC_EGC | densenet161_places365 T | Places365_val_00002172, 100 | 62.85285568237305 | 62.85285186767578 |
| TtBA | resnet50_places365 (NT first) | Places365_val_00002172, 100 | 28.24930191040039 | 28.24921226501465 |
| TtBA | densenet161_places365 (NT first) | Places365_val_00002172, **2000** (identical through 1000) | sq=1962, 8.775924682617188 | sq=1958, 8.998336791992188 |
| CGBA | resnet50_places365 NT | Places365_val_00002172, 100 | 32.159358978271484 | 32.159366607666016 |
| CGBA_H | resnet50_places365 T | (after NT) | — | — |
| CGBA | densenet161_places365 NT | Places365_val_00002172, 100 | sq=98, 36.700904846191406 | sq=100, 36.700889587402344 |
| CGBA_H | densenet161_places365 T | (after NT) | — | — |

All 12 cells: `VERDICT: DIFFERENT` (every npz delta differs as well). domain_consistency on the 8 new roots
(288 rows, `t7a_domain_consistency.log`): **X = 0 for all 12 cells** (LGC_RC_EGC 4, TtBA 4, CGBA/CGBA_H 4).

## 10. T7b — ARS_OPT on Places365 (correctness only; no bitwise reference)

| cell | queries | oob | found@10k | pix_l2@10k | domain_consistency X |
|---|---|---|---|---|---|
| resnet50_places365 NT | 10000 x3 | 10000 x3 | 1/1/1 | 12.10 / 6.67 / 0.58 | 0 |
| resnet50_places365 T | 10000 x3 | 8435 / 9960 / 9739 | 1/1/1 | 41.53 / 26.46 / 30.27 | 0 |
| densenet161_places365 NT | 10000 x3 | 10000 x3 | 1/1/1 | 5.04 / 10.19 / 1.22 | 0 |
| densenet161_places365 T | 10000 x3 | 9948 / 9854 / 9961 | 1/1/1 | 9.34 / 21.01 / 5.93 | 0 |

Roots appended to `reports/B4e2_smoke_roots.txt`: `Results/b5host_ARS_OPT_resnet50_places365`,
`Results/b5host_ARS_OPT_densenet161_places365` (plus the T6 Salman root).

## 11. Deviations from the prompt

1. **Environment bootstrap** (H approved "install the missing dependencies"): installed `ripgrep` (apt), `pandas
   scipy lpips torchmetrics timm transformers robustbench` (pip, Aliyun mirror only; GitHub unreachable).
   `autoattack` is not on the mirror → inert import shim `site-packages/autoattack/{__init__,state}.py` (only makes
   `robustbench.__init__` importable; never executed by the victim path) and `site-packages/sitecustomize.py`
   pre-imports the real `timm` so the ARS-OPT clone's partial vendored `timm` cannot shadow it. Neither changes any
   attack code; both are outside the repo.
2. The Salman2020Do_R50 weight file was absent from `~/.cache/robustbench/models` and Google Drive is unreachable;
   copied it from the repo's `models/imagenet/Linf/Salman2020Do_R50.pt` (sha256 verified `1835a5bbe5…` == the
   contract/`_common.py` constant) into the robustbench cache dir.
3. `run_frame_b1.sh` cannot express T1-A/A2 (hardcoded contract checkpoints + `--save_adv`; not editable): A/A2 ran
   the identical `run_benchmark_33b.py` line the script builds; B ran the script, producing
   `Results/b5host_B_resnet50` (the script appends `_<arch>`).
4. **T6 contract gate did NOT reach exit 0** — STOP bundle `reports/stop/B5-host_T6_20261018-0848.md` (E-SPEC):
   contract v1.9-rev5 missing the 4 Places365 `split_sha256` constants, torch 2.7.1/2.8.0 heterogeneity (inherent to
   the HOST-DIFFERENT verdict), and `.ipynb_checkpoints` junk in two old roots. Nothing was patched (contract,
   checker and Results are outside the allowed edits).
5. TtBA/CGBA/ARS_OPT `oob_queries` are large (see §5/§8/§10). The counter is observation-only and postdates the
   reference roots, so no reference comparison exists; the binding gate (domain_consistency) is X = 0 everywhere.
6. `nproc = 8` (container CPU limit): the prompt's B5 requirement "at least 40 CPU cores" is **not met**; recorded
   as a finding (T5 ratios ~1.4 at 10 processes; TtBA/CGBA/ARS_OPT oob figures unaffected).
7. The new roots record torch 2.8.0+cu128 / cudnn 91002 in their env files (the reference roots record 2.7.1+cu128 /
   90701) — this is the measured host difference, not an error.

## 12. Verdict

**HOST-DIFFERENT.** T2, T3 and T7a: every method, victim and mode diverges bitwise from the B4e2/B4h reference
roots on identical GPU hardware, with identical D12 backend flags, seeds, splits and adapters; the only environment
difference is the torch build (2.7.1+cu128/cudnn 90701 → 2.8.0+cu128/cudnn 91002). T1 (A==A2==B) and T5 (all GPUs
IDENTICAL to GPU 0) pass, so the host itself is internally consistent and deterministic; the divergence is between
hosts. Consequences for B5 (human decision, not started): the old smoke roots cannot serve as bitwise references on
this host, results from the two hosts must not be pooled, and the T6 smoke gate cannot reach exit 0 until the
contract's Places365 constants, the junk dirs, and the torch-heterogeneity policy are resolved
(STOP bundle `reports/stop/B5-host_T6_20261018-0848.md`).

## 13. sha256 of created/changed files (all GPU logs under reports/raw/B5-host/)

| file | sha256 |
|---|---|
| `reports/raw/B5-host/prompt.txt` | `7e73a26f1fb45f805d0d5a679e9fb2373b9fc559cda7f793bdc5cc53c956e79f` |
| `reports/B5_host_probe.md` (this file) | see commit |
| `reports/B4e2_smoke_roots.txt` (3 roots appended) | see commit |
| `reports/stop/B5-host_T1_20261018-0808.md` | see commit |
| `reports/stop/B5-host_T6_20261018-0848.md` | see commit |

GPU minutes: T1 ≈ 5 min (9 images), T3 ≈ 7 min (9 images) + domain-consistency ≈ 1 min, T5 ≈ 8 min (10 row-1
images under concurrency), T6 ≈ 4 min (6 images) + gates ≈ 2 min, T7a ≈ 27 min (36 images) + domain-consistency ≈
3 min, T7b ≈ 12 min (12 images) + domain-consistency ≈ 1 min, victim/import probes ≈ 3 min → **≈ 73 GPU-minutes**.

## 14. Summary (plain English)

1. The host is 5 x RTX 5090 with the same driver, but a **fresh container**: torch 2.8.0+cu128 / cudnn 91002 instead
   of the reference 2.7.1+cu128 / 90701, and only 8 usable CPU cores (B5 wants 40). Missing packages were installed
   with the human's permission (plus two inert env shims: `autoattack`, timm pre-import).
2. The B0' probe passes: A == A2 bitwise, and A == B at the six registered budgets.
3. Every cross-host comparison is DIFFERENT (EGC, TtBA, CGBA, ARS_OPT; main, Places365 and Salman victims), always
   as tiny floating-point divergences in pix_l2 that amplify along the trajectory — found/state_query stay aligned
   at low budgets. The only changed variable is the torch/cuDNN build.
4. Within this host everything is bitwise reproducible: five GPUs under 10-way concurrency produce byte-identical
   results to GPU 0 (~1.4x slowdown from CPU oversubscription), and all D19/domain-consistency gates are green
   (X = 0 everywhere; queries = 10000; no non-finite queries; human_preflight FLAG 0 on the new E11 cell).
5. The T6 contract smoke gate cannot reach exit 0 without human action: the contract v1.9-rev5 is missing the four
   Places365 split constants (a rev2/3 cleanup regression), two old roots contain `.ipynb_checkpoints` junk, and the
   new roots record torch 2.8.0 — the last being exactly the HOST-DIFFERENT finding. STOP bundle written; B5 not
   started.

## Deviations (post-report)

| Date | Deviation | Authorization | Notes |
|------|-----------|---------------|-------|
| 2026-10-18 | Contract rev6 edit (commit 173e73a) | H (Decision Point 2) | Restored 4 Places365 split_sha256 constants |
| 2026-10-18 | prompt.txt update (rev5->rev6, OMP=8) | H | sha256=2eb8698e3ac874196b518386c310702add8be63509689ceae2300ec89b905afb |
| 2026-10-18 | nproc/cpu.max investigation | H | Real nproc=208 (OMP artifact gave 8); cpu.max=125 cores; OMP=8 chosen (40 threads << 125 quota) |
| 2026-10-18 | Results/ junk-dir cleanup (.ipynb_checkpoints) | H (authorized) | Removed at 09:37 |
| 2026-10-18 | run_b5_parallel.sh edit (commit 1170dc1) | H (authorized) | Thread config at 09:54 |
| 2026-10-18 | run_b5_parallel.sh OMP=1->8 (revert 1170dc1) | H (authorized) | Real nproc=208; OMP=8 x 5 = 40 threads << 125 quota; prompt.txt consistency |
| 2026-10-18 | run_b5_parallel.sh OMP=1->8 (revert 1170dc1) | H (authorized) | Real nproc=208; OMP=8 x 5 = 40 threads << 125 quota; prompt.txt consistency |
