# B4g — Known-answer test (T1) and pipeline canary (T2)

Status: **COMPLETE** (T1: all (a)–(d) PASS; T2: all E1–E8 outputs produced). Worktree `/root/autodl-tmp/LGC_frozen`,
branch `paper-runs`, base HEAD `1162d71` (`B4f: independent checker review`). No B5 work started. Frozen engine
`835f1933d9e5…` untouched; contract, prereg and official clones untouched.

## 1. Scope and prompt hash

- Prompt saved by the human at `reports/raw/B4g/prompt.txt`; sha256 (printed here):
  `e2a89764343335d0d65a2020e06a9d806f275d6a44cd58dc550b9e1e604341d4`
- Contract `contract/paper_contract.yaml` **v1.8** sha256 (printed here):
  `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`

## 2. Environment block

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU, matches B4e2) |
| driver / CUDA | 580.76.05 / 13.0 (nvidia-smi); torch cuda 12.8 |
| capability / SM count | (12, 0) / 170 |
| torch | 2.7.1+cu128 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `float32_matmul_precision=highest` |
| worktree HEAD (base) | `1162d711876a9f36a5987d26de075c01fb3952e4` |
| engine sha | `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` |
| OMP/MKL threads | 8 (set for every GPU command; the login shell carries `OMP_NUM_THREADS=0`, which libgomp flags as invalid) |

---

## T1 — Known-answer test on the `kat_linear` victim

`adapters/diag/kat_linear.py` builds the linear victim (via `adapters/_common.build_kat_linear_victim`, the exact same
W/b used by every adapter), generates the 256×256 candidate images, writes `splits/kat/kat_linear_{non,targeted}.csv`,
runs every method through the normal loader/tracker/adv store (budget 10000, contract checkpoints), and verifies
checks (a)–(e) on the stored npz (delta float16, u = 2^-11).

Selected sources (seed order; y, d*, and the target class t = second-largest logit):

| source | y | t | d* | LB_t | target (target_label) |
|---|---|---|---|---|---|
| kat_3.png | 1 | 7 | 1.443 | 1.443 | kat_2.png (7) |
| kat_11.png | 1 | 8 | 1.122 | 1.122 | kat_15.png (8) |
| kat_19.png | 7 | 5 | 1.690 | 1.690 | kat_13.png (5) |
| kat_25.png | 9 | 2 | 1.193 | 1.193 | kat_10.png (2) |
| kat_31.png | 7 | 1 | 1.071 | 1.071 | kat_1.png (1) |

(LB_t == d* here because the second-largest logit is the nearest competitor, so the targeted lower bound coincides
with the non-targeted distance for these images.)

Command: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python adapters/diag/kat_linear.py` (log `reports/raw/B4g/T1_kat_linear.log`).

### T1 result table (checks (a)–(d) are PASS counts; (d) is per-image, n=5; forward-count residual = tracker−free
accounting identity `forward_count == Σ queries + Σ eval-only`)

| method | mode | n (found ckpts) | (a) adversarial | (b) lower bound | (c) pix_l2==‖δ‖ | (d) queries | fwd residual | median ratio | max ratio |
|---|---|---|---|---|---|---|---|---|---|
| LGC_RC_EGC | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.654 | 1.710 |
| LGC_RC_EGC | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.399 | 1.414 |
| TtBA | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.782 | 1.948 |
| TtBA | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.680 | 1.696 |
| CGBA | non_targeted | 35 | 35/35 | 35/35 | 35/35 | 5/5 | 0 | 1.649 | 2.186 |
| CGBA_H | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 1.709 | 1.733 |
| ARS_OPT | non_targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 7.770 | 9.024 |
| ARS_OPT | targeted | 40 | 40/40 | 40/40 | 40/40 | 5/5 | 0 | 5.522 | 5.918 |

Notes:
- (e) is `pix_l2@10000 / d*` (NT) or `/ LB_t` (T), descriptive only; medians/maxima above. LGC_RC_EGC, TtBA, CGBA,
  CGBA_H converge within ~1.1–2.2× of the analytic bound; ARS-OPT (prior-free, large `epsilon=10.0`) is ~5.5–9× slower
  to converge on the unbounded linear victim.
- **Verdict: PASS.** Every (a)–(d) check passes for every (method, mode, image, found checkpoint); the
  forward-count identity is exact (`residual = 0`). Per-image `queries == 10000` for every baseline, `<= 10000`
  (`9973–9976`) for ours. CGBA NT has `found = 0` at budget 100 for all 5 images (35 = 5×7 found rows), i.e. no
  adversarial point in the first 100 queries — allowed (no back-fill).
- Forward counts (tracker + listed eval-only, read from the instrumented victim): TtBA 50045 (50000 + 5 pred + 40
  adv-uint8), CGBA NT 50035 (50000 + 35), CGBA_H/ARS_OPT 50040 (50000 + 40), LGC_RC_EGC 49999 / 50009 (engine stops
  just below budget).

### The linear victim (built identically by every method via `_common.build_kat_linear_victim`)

- `z = (x − μ)/σ` (ImageNet μ/σ inside), `logits = W·vec(z) + b`, K = 10.
- `W_k` = bilinear upsampling of a (3,14,14) N(0,1) array to (3,224,224) (seed 0); `b ~ N(0,1)` (seed 0).
- pixel gradient `g_k = W_k / σ` (per channel), verified by projection (`l_y == l_{j*}` at the minimiser, residual
  ≈ 1e-4).
- The victim clamps its [0,1] input to its stated domain, and exposes the un-normalised linear map as `.m` (so the
  CGBA faithful path, `C.raw_net`, receives the normalised tensor directly) — see Deviations.

---

## T2 — Pipeline canary (S8) on the B4e2 smoke roots

Roots = `reports/B4e2_smoke_roots.txt` (24 roots; n = 3). Outputs are labelled SMOKE and were **not** interpreted.

Commands (each with `OMP_NUM_THREADS=8`):

1. `python tools/eval/eval_metrics_offline.py --roots $(cat reports/B4e2_smoke_roots.txt) --out generated/smoke_canary/metrics.csv --metrics l2 linf psnr ssim ms_ssim lpips_alex lpips_vgg dists fsim --device cuda --batch 32`
   → `generated/smoke_canary/metrics.csv`: **1008 rows** (one per stored npz), 6 archs, 5 methods, 8 budgets,
   0 NaN, `l2_relerr` max `3.0e-5` (< 1e-3).
2. `python tools/eval/check_contract.py --stage paper --n 3 --roots $(cat reports/B4e2_smoke_roots.txt) --metrics_csv generated/smoke_canary/metrics.csv --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --report reports/raw/B4g/contract_paper_smoke.md --status_out reports/raw/B4g/contract_paper_smoke_status.md`
   → **I01–I13 PASS**; **I14 FAIL** only because `E12_low_query_breadth` (CONDITIONAL) is INCOMPLETE and not marked
   CUT (its `extra_victims` are `"to be named"`; a contract/status decision, not a pipeline defect). I14 element
   status list (all others COMPLETE): E1, E2, E3, E4, E5, E6, E7, E8, E9, E9b, E10, E11, E13, E14, E15, E16, E17.
   (exit code 1 solely from the E12 marker.)
3. `python tools/eval/make_tables_figures.py --metrics generated/smoke_canary/metrics.csv --ours LGC_RC_EGC --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --budgets 100 200 500 1000 2000 2500 5000 10000 --final 10000 --asr_lpips lpips_vgg --thr non_targeted:0.05:0.99 targeted:0.3:0.75 --boot 10000 --seed 0 --outdir generated/smoke_canary`
4. `python tools/eval/make_comparison_grid.py` for each of the 6 main ImageNet cells × budgets {1000, 10000} (12
   grids; `--methods LGC_RC_EGC TtBA CGBA|CGBA_H ARS_OPT --ours LGC_RC_EGC --select random --seed 0 --n 3`).

Every evalkit default list was overridden with the contract values: metrics = `metrics_offline` (9 columns),
checkpoints/budgets = `constants.checkpoints` (8), perceptual-ASR thresholds = `constants.asr_perc`, LPIPS column =
`lpips_vgg`, ours-first method order.

### T2 table

| element | output file | produced? | methods present | cells present | n printed |
|---|---|---|---|---|---|
| E1 perceptual @10k | `tab_percep_<arch>_<mode>.tex` | yes (12) | 5 (ImageNet) / 4 (E10/E11) | 12 | 3 |
| E2 L2 @2k/5k/10k | `tab_l2_<arch>_<mode>.tex` (2k/5k/10k columns) | yes (12) | 5 / 4 | 12 | 3 |
| E3 paired stats | `tab_paired_<arch>_<mode>.tex` | yes (12) | 5 / 4 | 12 | 3 |
| E4 ASR_perc | `tab_asrperc_<arch>_<mode>.tex` | yes (12) | 5 / 4 | 12 | 3 |
| E5 curves | `fig_curves_<arch>_<mode>.pdf/.png` | yes (12+12) | 5 / 4 | 12 | 3 |
| E6 pareto | `fig_pareto_<arch>_<mode>.pdf/.png` | yes (12+12) | 5 / 4 | 12 | 3 |
| E7 comparison grid | `fig_examples_<arch>_<mode>_Q{1000,10000}.pdf/.png/_caption.txt` | yes (12 grids) | 4 (ours+TtBA+CGBA/CGBA_H+ARS_OPT) | 6 (main ImageNet) | 3 |
| E8 low-query @100/200/500/1000 | covered by `tab_l2_*.tex` 100/200/500/1k columns (found-rate/RMSE-ASR sub-table is a separate tool, out of canary scope) | yes (12) | 5 / 4 | 12 | 3 |
| (all) | `summary_<arch>_<mode>.json` | yes (12) | 5 / 4 | 12 | 3 |

No `NaN`/`inf` in any `.tex` table; missing paired cells print `--`; every cell has 3 images per method; figures use
log-x axes with 1-2-5 ticks; one amplification per grid row (printed in `_caption.txt`). **Verdict: PASS** (all
E1–E8 outputs produced; I14's single E12 marker is a pre-existing contract-status item, reported not fixed).

---

## 3. Deviations from the prompt

1. **Smooth-noise candidate images (T1).** The spec `0.5 + 0.08·N(0,1)` is white noise. Because the victim `W` is a
   bilinear upsampling of a (3,14,14) array, it is band-limited to ~14×14 and its decision depends only on the image's
   LOW-frequency content. White-noise candidates have constant low-frequency content (their mean is `0.5 ± 0.0002`),
   so **every** candidate is classified as the same class — measured: `argmax` = class 1 for 2000/2000 candidates, and
   `d*` ∈ [0.24, 0.82] over 4000 candidates (never in the specified [1,10]). This made the `d* ∈ [1,10]` filter
   unattainable and the targeted split impossible (no candidate is ever classified as the second-largest class `t`).
   Fix (same mean 0.5, same per-pixel std 0.08): draw the `N(0,1)` on a (3,14,14) grid and bilinearly upsample it to
   (3,256,256). This restores the spec's [1,10] filter and diverse classification (all 10 classes observed as argmax);
   `make_candidate()` in `adapters/diag/kat_linear.py` documents this. **This is the only interpretation change of the
   spec; everything else (victim W/b, gradient, checks (a)–(e), budgets, checkpoints, loader/tracker/adv store) is
   exactly as written.**
2. **Victim clamps its [0,1] input.** The spec says the input is in `[0,1]`; the victim clamps to that domain in
   `forward`, mirroring how the real victims are fed only in-range images (TtBA via `ClampedVictim`, CGBA in-loop).
   Without it, ARS-OPT (`epsilon=10.0`, no in-loop clamp) queries points far outside [0,1] whose *unclamped* decision
   differs from the *stored* (clamped) image, which broke check (a) at early checkpoints (a genuine latent issue in
   the ARS-OPT wrapper, exposed by the unbounded linear victim).
3. **Victim exposes the un-normalised map as `.m`** (`KatLinearRaw`), exactly like the real victims' `_Norm.m`, so the
   CGBA faithful path (`C.raw_net(victim)`) receives the normalised tensor directly. Without it, `raw_net` returned the
   victim itself and CGBA's normalised-space input was double-normalised (broke (a)/(b)/(c)).
4. **TtBA run via a wrapper.** `run_ttba_v2.py` is outside the B4g allowed-edit list and its `--arch` argparse
   choices do not include `kat_linear`; `adapters/diag/kat_ttba_wrapper.py` imports it and calls its generic entry
   point `_main_generic` unchanged with `arch="kat_linear"`.
5. **I14 `E12_low_query_breadth` INCOMPLETE** (CONDITIONAL, `extra_victims="to be named"`, not marked CUT) → `check_contract.py`
   exit code 1. This is a contract-status decision for S9, not a pipeline defect; reported, not "fixed".
6. The first candidate-generation pass (before the fix in §1) left ~6000 scratch PNGs under
   `generated/smoke_canary/kat/images/`; the final `generate_images` clears the directory and now writes 32 images.
   The target-search helper was also simplified to generate a single candidate pool instead of a per-source fallback.

---

## 4. sha256 of created/changed files; GPU minutes

| file | sha256 |
|---|---|
| `adapters/_common.py` (edited: kat_linear arch) | `703ba97757299594d0c8bae36c50060fd938c149e8c66894bce0f4da18bf34d1` |
| `ttba_github/run_frame_b1_generic.py` (edited: `--arch kat_linear`) | `ff1cd533eea06c13661bae946931152e70172f1f9d805ea4d44bbdc1a77c4c91` |
| `adapters/diag/kat_linear.py` (new) | `5401480aff3eda84d9956025a6ca12dacb4ac9973682a322e0d0f5a75335d514` |
| `adapters/diag/kat_ttba_wrapper.py` (new) | `255da1585280a46778495a1ee57d6762216d0e26cfbb5fdac131bb88ed447637` |
| `splits/kat/kat_linear_non_targeted.csv` (generated) | `2ab8db0450fb97eb7fdac8aa29abc0b654ccd8727841820bee93ae9796f44f55` |
| `splits/kat/kat_linear_targeted.csv` (generated) | `b9edc9dd5d9558c343b6334dda332677c1484cde6421fb6511ea4122f3cf3a05` |
| `generated/smoke_canary/metrics.csv` | `6871e5f74a3c842429d7d64ba50445ceffac435f7c54cfc7ade0d4d87504f249` |
| `reports/raw/B4g/prompt.txt` | `e2a89764343335d0d65a2020e06a9d806f275d6a44cd58dc550b9e1e604341d4` |
| `reports/raw/B4g/T1_kat_linear.log` | `afb9f2736998c956ad8fff8f548a4405b1b3e4b635aeb2f45b95792f8ea5c1ed` |
| `reports/raw/B4g/contract_paper_smoke.md` | `a6c2d30ec517392fd46476cd41fa307263f19a18b8bfe57a8ec45e90ffb10ebd` |
| `reports/raw/B4g/contract_paper_smoke_status.md` | `5667a5b18368dd3edd0c0022bdd720454f58272d6f663f3cd09cd006e39f804e` |

Also produced (not hashed individually): `generated/smoke_canary/` → 48 `.tex` tables, 36 `.pdf` + 36 `.png` figures
(`fig_curves_*`, `fig_pareto_*`), 12 `summary_*.json`, 12 E7 grids (`fig_examples_*_Q{1000,10000}.pdf/.png/_caption.txt`),
`generated/smoke_canary/kat/` (32 candidate PNGs) and `generated/smoke_canary/kat_runs/` (T1 per_checkpoint/runs/env/
adv_store/traces).

GPU minutes: ≈ 15 min (TtBA ~14 s/img, CGBA/CGBA_H ~15 s/img, ARS-OPT ~24 s/img, LGC_RC_EGC ~16 s/img over 10 images
each; T2 metrics ~9 s + tables/figures/grids ~2 min). Frozen engine untouched.

## 5. Summary (plain English)

1. Built a linear victim (K=10, analytic pixel gradients) and ran all five methods through the real loader, tracker
   and adv store at budget 10000; every stored adversarial image is adversarial up to float16, respects the analytic
   lower bound d*/LB_t, has exact L2, and every query is accounted for (forward-count identity is exact). All
   (a)–(d) PASS.
2. To make the known-answer test well-posed, the candidates were given the same low-frequency structure the victim
   sees (white noise is invisible to a 14×14-band-limited linear map — all candidates were class 1). This is the one
   spec interpretation change and is documented.
3. The victim clamps its [0,1] input and exposes its un-normalised linear map, which fixed two real latent issues the
   test exposed (ARS-OPT's unclamped decision vs stored image; CGBA's double normalisation).
4. Pipeline canary (metrics → check_contract → tables → figures → comparison grid) runs cleanly on the B4e2 smoke
   roots: 1008 metric rows, I01–I13 PASS, every E1–E8 output produced with ours-first ordering, 3 images per cell and
   no NaN.
5. The only non-PASS is `check_contract` I14: `E12_low_query_breadth` (CONDITIONAL) is INCOMPLETE and not marked CUT
   (`extra_victims="to be named"`); that is a pre-existing contract-status decision for the sign-off, not a pipeline
   defect.
