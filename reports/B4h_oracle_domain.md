# B4h — Oracle domain + E8 low-query tool (T1–T8, consolidated close-out)

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, base HEAD `2d7cde4`. Contract v1.9.
No B5/T6 run started. Frozen engine `835f1933d9e5…` untouched; contract, prereg and official clones untouched.
T1/T3/T5/T6 (D19 oracle-domain fix, KAT regression, 18-cell smoke, domain-consistency gate) were completed in
`5258b3a` and are reported in **`reports/B4h_t3_fix_and_gate.md`**; this phase (`B4h-Close`) completes the remaining
tasks **T2, T4, T7, T8** and consolidates T1–T8.

## 1. Scope and prompt hash

- Prompt (this phase) saved at `reports/raw/B4h/prompt_close.txt`; sha256:
  `f7553bf64ae713fe332d8b5d8c986fddb1142731b0f242347d70be783d77fed4`.
- Contract `contract/paper_contract.yaml` **v1.9** sha256 (printed at start):
  `fc04498315dec51b0f663568d76a7804d46f63aa57eb2950a2126d157451ca32` (matches expected).
- Worktree HEAD: `2d7cde4b5dbb36fc75dc17f0fd4633cdc07452f5`.

## 2. Environment block

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU for this phase; contract D16 names RTX PRO 6000 for the B5 full runs, not started here) |
| driver / CUDA | 595.71.05 / 13.0 (nvidia-smi); torch cuda 12.8 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `float32_matmul_precision=highest` |
| worktree HEAD | `2d7cde4b5dbb36fc75dc17f0fd4633cdc07452f5` |
| contract sha256 | `fc04498315dec51b0f663568d76a7804d46f63aa57eb2950a2126d157451ca32` |
| OMP/MKL/OPENBLAS | 8 (set for every GPU command) |

**Active LESSONS applying to this phase** (read from `docs/LESSONS.md` at start): L02 (CGBA acos stall, encoded), L03
(GPU drift, STOP device check), L04 (evidence rules), L05/L06 (reviewer independence, prompt archiving — this phase
saved `prompt_close.txt`), L07 (foreign AGENTS/skills), L10 (resume `done_images`), L11 (check_contract status race),
L12 (one `build_env`), L14 (check specs vs prereg/contract/code), L19 (metrics selftest needs GPU), L20/L21 (empty
replies / degeneracy), L22 (D19 oracle domain — encoded by the ClampedVictim/ClampedVictimNormalized fix), L23 (KAT
caveat — recorded in B4g).

## 3. T2 — `oob_queries` observation-only counter

`adapters/_common.py` (`NonfiniteGuard` / `_GuardedHardLabelTracker`):

- `NonfiniteGuard` now takes `normalized_space: bool` and keeps a per-image counter `oob_queries` in addition to
  `nonfinite_queries`.  In `forward`, after the non-finite check, it maps the raw batch input `xb` to PIXEL space
  **without clamping** and counts the images with any coordinate `< -1e-6` or `> 1 + 1e-6`:
  - `normalized_space=True` (tracker `to_pixel is to_pixel_unnormalize` OR the wrapped model is
    `ClampedVictimNormalized`): `px = xb * sd + mu` (per-channel ImageNet mean/std materialised on `xb`'s device);
  - otherwise (pixel-space `ClampedVictim` / `to_pixel_clamp`): `px = xb` directly, checked BEFORE `ClampedVictim`
    clamps it — so out-of-box probes are still observed even though the decision is computed on the clamped image.
- `_GuardedHardLabelTracker.__init__` computes the `normalized_space` flag from `kwargs["to_pixel"]` and the model
  type and passes it to the guard; an `oob_queries` property exposes the counter.
- `"oob_queries"` added to `RUN_FIELDS` in `adapters/_common.py` and to the two `runs_*.csv` writer calls in
  `C.run` (error + success rows), and to `RUN_FIELDS` + all four `write_rows` calls in `adapters/run_ttba_v2.py`
  (generic E10/E11 + ImageNet paths), so every future run writes `oob_queries` per image.

CPU unit test added to `adapters/tests/test_g1_clamped_victim.py` (section `[2]`): a `_GuardedHardLabelTracker`
around `ClampedVictim` (pixel space) and one around `ClampedVictimNormalized` (normalised space) are queried with
in-range and out-of-box inputs; `oob_queries` counts `0/1/2` correctly and in-range queries give `0`.

| command | result |
|---|---|
| `python adapters/tests/test_g1_clamped_victim.py` | ALL PASS (12/12, incl. 6 new oob_queries checks) |
| `python adapters/tests/test_g1_adapters.py` | ALL PASS (8/8) |

**Verdict: PASS.**

## 4. T4 — Strict `C.raw_net(victim)` + B4f T6 raw-net check

- `adapters/_common.py::raw_net` now raises
  `ValueError(f"raw_net: victim {type(victim).__name__} exposes neither .m, .net, nor Sequential.normalize")`
  instead of silently returning `victim` unchanged (the silent fallback double-normalised and silently diverged).
- `adapters/tests/test_g1_adapters.py` exposes `.m = victim` on its `DummyVictim` (the linear dummy *is* its own raw
  net) so `run_cgba.make_tracker → C.raw_net` still resolves; `test_g1_adapters.py` remains 8/8 PASS.
- B4f T6 raw-net check re-run (`adapters/diag/diag_b4f_t6_domain.py`) across all 6 victims, log
  `reports/raw/B4h/t4_rawnet.log`:

| victim | raw net | max abs diff |
|---|---|---|
| vgg19 | torchvision VGG | 0 |
| resnet50 | torchvision ResNet | 0 |
| vit_b_32 | torchvision VisionTransformer | 0 |
| resnet50_places365 | torchvision ResNet | 0 |
| densenet161_places365 | torchvision DenseNet | 0 |
| Salman2020Do_R50 | torchvision ResNet | 0 |

`victim == raw_net(normalize(x0))` for all 6 victims: **YES (all 0)**. d_n/d_p for the 3 stored CGBA points stays in
[4.367, 4.444]. **Verdict: PASS (diff = 0).**

## 5. T7 — E8 low-query table tool + pipeline canary

### (a) `tools/eval/make_low_query_table.py` (new)

Implements paper element `E8_low_query_table` exactly per `contract/paper_contract.yaml`: budgets `100, 200, 500,
1000`; per method and budget — found rate (%), ASR (%) at RMSE `0.1/0.05/0.01` with `RMSE = pix_l2 / sqrt(3·224·224)`
(an image with `found == 0` or `adv_flag == 0` counts as a failure / L2 = inf, contract D6), median `pix_l2` (over
found rows), median `lpips_vgg` (from the offline metrics CSV).  It normalises image names to stems (the EGC runner
writes `.JPEG` names, the baselines write stems), reads `found`/`pix_l2` from `per_checkpoint*.csv` and `lpips_vgg`/
`adv_flag` from the metrics CSV, prints `n` per method, orders ours (`LGC_RC_EGC`) first, and applies direction-aware
bolding (highest found rate / ASR, lowest L2 / LPIPS).

### (b) `tools/eval/tests_lowq/test_lowq.py` (new)

26 CPU checks on synthetic CSVs with known exact answers: found rate, ASR at the three RMSE thresholds (including the
10.0-pixel RMSE 0.01 boundary, the 10000-pixel all-fail case), median pix_l2, median lpips_vgg, `.JPEG`-stem loading,
`adv_flag == 0` failure handling, ours-first ordering, and direction-aware bolding.  Exit 0 / ALL PASS.

### (c) Pipeline re-run on the updated 24 smoke roots

1. `eval_metrics_offline.py` re-run on `$(cat reports/B4e2_smoke_roots.txt)` → `generated/smoke_canary/metrics.csv`
   (old sha256 `6871e5f7…` regenerated; the 9 D19-fixed roots — 6 CGBA/CGBA_H + 3 ARS_OPT — now reflected, no stale
   `b4e2_smoke_{arsopt,cgba}` roots).  **1008 rows, 24 roots, 6 archs, 5 methods, 0 NaN.**
2. `make_low_query_table.py` → **12** `generated/smoke_canary/tab_lowq_<arch>_<mode>.tex` (6 archs × 2 modes), ours
   first, `n = 3` per method, no `--`/NaN in the printed cells for the main ImageNet cells.
3. `human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out reports/human_preflight_b4h` → **FLAG = 7**,
   identical set to B4e2/B4f: the 6 EGC E10/E11 generic-runner `env_frame.txt` (historical, missing
   `cudnn_tf32`/`deterministic`) and the 1 GPU-name mismatch from the frozen `config_single.json` device string
   (`NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True`).  All 9 new D19-fixed roots record all five
   D12 flags correctly; no new FLAG.  (These 7 are pre-existing, documented in `reports/B4f_audit.md` T7/Deviations.)

**Verdict: PASS.**

## 6. T8 — Errata & final report

- `reports/B4f_audit.md`: added an **Erratum (B4h, contract v1.9 D19)** note after the T6 verdict, stating that the
  ARS-OPT and CGBA/CGBA_H rows of the T6 domain/clipping table describe the pre-fix state and are superseded by the
  D19 oracle-domain fix (`ClampedVictim` / `ClampedVictimNormalized`), with the domain-consistency gate X = 0.  The
  original table/verdict text is unchanged (no history rewrite).
- `reports/B4g_kat_canary.md`: the **Erratum (B4h-T3, D19)** for deviation 2 and the L23 caveat was already added and
  committed in `5258b3a`; verified present, no further edit needed.

## 7. Test summary (all re-run this phase)

| command | result |
|---|---|
| `python tools/eval/selftest_evalkit.py` | ALL PASS (20/20) |
| `python tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21) |
| `python adapters/tests/test_g1_clamped_victim.py` | ALL PASS (12/12) |
| `python adapters/tests/test_g1_adapters.py` | ALL PASS (8/8) |
| `python tools/eval/tests_lowq/test_lowq.py` | ALL PASS (26/26) |

## 8. Deviations from the prompt

1. **GPU name.** This instance reports `NVIDIA GeForce RTX 5090` (driver 595.71.05), while contract v1.9 D16 names
   `RTX PRO 6000` as the study GPU for the B5 full runs.  This phase is smoke/tool-check level (n = 3, D13: n < 30 is
   never a gate); no B5 run is started.  Recorded in every env file (L03).
2. `adapters/run_cgba.py`'s `_faithful_model` docstring still says "A victim without such a wrapper (unit-test dummy)
   is returned unchanged", which is now inaccurate after T4 (the fallback raises).  `run_cgba.py` is outside this
   phase's allowed-edit list, so the stale docstring is left as-is and flagged here.
3. `diag_b4f_t6_domain.py` still reads 3 stored CGBA points from the *pre-D19* path `Results/b4e2_smoke_cgba_resnet50`
   (hard-coded) for the d_n/d_p ratio; the ratio is unaffected by D19 (it is a property of the ImageNet std, not the
   clamp) and the raw-net check — the T4 acceptance criterion — covers all 6 victims with diff 0.  No edit was made to
   the diag script beyond running it.
4. No other deviations.

## 9. sha256 of created/changed files; GPU minutes

| file | sha256 |
|---|---|
| `adapters/_common.py` (edited, T2 + T4) | `6af35821686f4e23175becf4fb787081d3b839cf973fd725d51f55ed3f1ff781` |
| `adapters/run_ttba_v2.py` (edited, T2 oob_queries) | `c05876acee65cacda908a0e5e266e6251b7139ee1c21e0f70b85e6147c584fd2` |
| `adapters/tests/test_g1_clamped_victim.py` (edited, T2 oob_queries test) | `0df7ebb7b1c259ba565206ee22468ed11a58d0b9f7ec2b6bbdffa0853ebff9e1` |
| `adapters/tests/test_g1_adapters.py` (edited, T4 dummy `.m`) | `47b36d6c2e78f77eac1edee740953b461a07b132f0cb18c130b6665548c07cc9` |
| `tools/eval/make_low_query_table.py` (new, T7) | `026c4bdd4cb2b7cfd8944d8a108b1e7ecdfa611b17ba9fe246cdfeef46ccae36` |
| `tools/eval/tests_lowq/test_lowq.py` (new, T7) | `e2778932b4e95e5b6de5c084560da3acd77b351a393a7f81c406878f4cb26f48` |
| `reports/B4f_audit.md` (edited, T8 erratum) | `c6f6a1a1ec44371d6062d4ef5f35bc7f8a6826e0adecc43a35c1e1b688c7f793` |
| `reports/raw/B4h/prompt_close.txt` (new) | `f7553bf64ae713fe332d8b5d8c986fddb1142731b0f242347d70be783d77fed4` |
| `reports/raw/B4h/t4_rawnet.log` (new) | `453696c862097a0351512df1cf6a51c9df8385cf2e6fb68dd1d811e9feb5dbd5` |
| `generated/smoke_canary/metrics.csv` (regenerated, T7) | `2e54f37dedd717acf2134ed553d3ee380981b875a2c511c319ecd491cd6fa981` |

Also produced (not hashed individually): 12 `generated/smoke_canary/tab_lowq_<arch>_<mode>.tex` and
`reports/human_preflight_b4h.md` + 12 grid PNGs.

GPU minutes: ≈ **2 min** (T4 raw-net check across 6 victims + d_n/d_p, and T7 `eval_metrics_offline.py` ~7 s over 1008
states).  No attack was re-run (the 18 smoke cells and KAT regression were completed in `5258b3a`).  Frozen engine
untouched.

## 10. Summary (plain English)

1. Added an observation-only `oob_queries` counter to the shared guard (pixel- and normalised-space aware, before any
   clamp) and wrote it to every future `runs_*.csv`, with a CPU unit test (12/12).
2. Made `raw_net(victim)` strict (raises on a victim without `.m`/`.net`/`Sequential.normalize`), fixed the dummy in
   `test_g1_adapters`, and re-ran the B4f T6 raw-net check: diff = 0 on all 6 victims.
3. Built the E8 low-query table tool exactly per the contract (found rate, RMSE-ASR 0.1/0.05/0.01, median L2/LPIPS at
   100/200/500/1000, ours first, direction-aware bolding) with 26 CPU tests; regenerated the smoke metrics CSV over the
   24 updated roots and produced 12 `tab_lowq_*.tex`; `human_preflight` FLAG = 7 (the same pre-existing set, no new).
4. Added the B4h errata to the B4f audit (B4g already carried its D19/L23 errata) without rewriting history.
5. All five test suites pass (20/20, 21/21, 12/12, 8/8, 26/26).  Committed; no B5 started.
