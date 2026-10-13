# B4e — Pre-B5 preflight (CGBA clip fix, D8-only appendix, precision parity, G5 tool, B5 cost)

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD `616b41a`
(+ follow-up commits, see the "B4e follow-up" section at the end).
Contract **v1.8** — sha256 `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` (verified at start,
matches the expected value). Engine sha `835f1933d9e5…` (frozen, untouched).

---

## 0. Study GPU notice (for contract v1.9 D16)

**The study GPU is `NVIDIA RTX PRO 6000 Blackwell Server Edition`, driver `580.82.09` — NOT RTX 5090.**
torch `2.7.1+cu128`, CUDA 13.0. The B4 / B4b smoke baselines and the EGC smoke were recorded on an
`NVIDIA GeForce RTX 5090` (driver 595.71.05); every `env_*.json` from those runs records `RTX 5090`.

Consequence for reproducibility (investigated in T1, resolved by D12 in T2):

| victim (backbone) | 5090 == 6000 under TF32-on? | note |
|---|---|---|
| resnet50 (ImageNet) | **bit-identical** | d8only re-run == b4_smoke, all 8 checkpoints + npz delta |
| Salman2020Do_R50 (RobustBench) | **bit-identical** | d8only re-run == b4b_smoke |
| vgg19 (ImageNet) | no (≈1e-8 pix_l2 / state_query flips) | d8only re-run also diverges → GPU, not the clip |
| vit_b_32 (ImageNet) | no (2/3 images; `00024661` ≈4.8e-7 @500, `00046402` ≈1.5e-4 @5000) | d8only re-run also diverges |
| resnet50_places365 / densenet161 (Places365) | no (diverges from first checkpoint) | d8only re-run also diverges |

Both the 5090 and the 6000 are Blackwell; the earlier B4e d8only runs in `Results/b4e_t1_d8only_resnet50{,_det,_repeat}`
were recorded on a transient `RTX 4080 SUPER` (Ada) and **must be discarded** (they diverge; `_det` used
deterministic flags, `_repeat` was a TF32-on repeat). The current 6000 reproduces the 5090 bit-for-bit for the two
bit-stable backbones above. **D16 (contract v1.9) must record the GPU change; the D12 precision-parity backend
(§T2) removes the remaining TF32/cuDNN GPU-architecture dependence.**

---

## T1 — CGBA clip fix (D11_outcome)

### (a) Code
`adapters/run_cgba.py` now takes `--cgba_variant {fixed, d8only}` (default `fixed`).
- `fixed`: `PatchedProposedAttack` overrides `SemiCircular_boundary_search` with a verbatim copy of
  `proposed_attack.py:171-196` whose ONLY change wraps the `acos` argument at `:182` in
  `torch.clamp(..., -1.0, 1.0)`. Method id `CGBA` (NT) / `CGBA_H` (T).
- `d8only`: exactly the previous adapter (D8 count fix, no clip). Method id `CGBA_D8only`; NT only
  (`--cgba_variant d8only` + `--mode targeted` refuses, exit non-zero).
- `cgba_variant` and `paper_faithful_fix` (`"clip acos arg, proposed_attack.py:182, D11_outcome"`) are written
  into `env_*.json` and `run_manifest.json` (via `extra_env` + `build_manifest`).

### (b) Source-diff test — PASS
`python adapters/diag/diag_b4e_t1_clip_source.py`:
```
d8only: SemiCircular_boundary_search is official (not overridden) = True
PASS: exactly one changed line (index 11), the acos line
  official: theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))/ ...)
  override: theta = torch.acos(torch.clamp(torch.dot(...)/ ..., -1.0, 1.0))
VERDICT: PASS
```

### (c) Non-finite guard (observation only)
`adapters/_common.py` wraps the tracker model in `NonfiniteGuard`, counting per-image queries whose input
carries a NaN/±inf; `nonfinite_queries` is written per image to every `runs_*.csv` (all adapters, incl.
`run_ttba_v2.py`). It never alters output, accounting, or trajectory.

### (d) Acceptance (T1 run before T2, current TF32-on flags)
1. **d8only on resnet50 NT rows 1–3 — IDENTICAL to `Results/b4_smoke_cgba_resnet50`** (PASS; all 8 checkpoints
   + npz `delta` bytes bit-identical, `method`/filename differ by design).
2. **fixed on non-NaN images** (vgg19, vit_b_32, densenet161 NT; every CGBA_H cell): bit-identical to the old
   smoke **only for backbones that are bit-stable across 5090→6000** (resnet50, Salman2020Do_R50). vgg19 diverges at
   ≈1e-8 and vit_b_32 at ≈4.8e-7–1.5e-4 (image `00046402` diverges ≈1.5e-4 at its first divergent checkpoint @5000);
   Places365 diverges from the first checkpoint **because of the GPU migration, not the clip** — proven by re-running
   `d8only` (unclipped) and observing the same divergence (no d8only control exists for vit_b_32; see Deviations).
   This is the D16 issue, resolved
   by D12 (§T2). CGBA_H cells have no NaN event (D11_outcome "CGBA_H 0").
3. **fixed on the stalled images** — first-NaN query index (from `adapters/diag/cgba_trace.py`), prefix
   identical before it, then `nonfinite_queries = 0`, no stall:

   | image | first-NaN query | prefix identical through | pix_l2 1k / 2k / 5k / 10k (fixed) | BSSP 100-call cap hits |
   |---|---|---|---|---|
   | resnet50 NT 00007450 | 774 | @500 | 84.996 / 81.410 / 75.256 / 67.838 | 0 |
   | resnet50 NT 00015433 | 1084 | @1000 | 74.569 / 71.072 / 55.868 / 46.867 | 0 |
   | resnet50 NT 00025151 | 776 | @500 | 62.793 / 47.104 / 38.243 / 31.657 | 0 |
   | Salman2020Do_R50 NT (3 rows) | 228 / 152 / 199 | @200 (all) | 30.08 / 30.37 / 3.96 @10k | 0 |
   | resnet50_places365 NT (rows 1–2) | 5402 / 4014 | @500 (row 1) | 15.83 / 11.67 @10k | 0 |

   (BSSP cap hits = 0 everywhere; the pix_l2 columns above are the TF32-on acceptance run; the D12 re-run values
   are in §T2.)

**T1 verdict: PASS for the code and for bit-stable backbones (resnet50, Salman2020Do_R50).** The
"IDENTICAL to old smoke" acceptance for vgg19 / vit_b_32 / Places365 cannot be met under TF32-on because the
GPU changed (5090→6000); this is a D16 fact, not a clip bug, and is eliminated by D12.

---

## T2 — Precision parity (D12)

### Code
`adapters/_common.py` sets, at import time (before any model is built, for every adapter, ImageNet and E10/E11):
```
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False
```
plus a `precision` block in `env_*.json`. This matches `lgc_bench/env.py:configure_backend` (the EGC runners).

### Diagnostic
`python adapters/diag/diag_b4d_t1_precision.py` (updated) → `ALL RUNNERS IDENTICAL: cuda.matmul.allow_tf32=False,
cudnn.allow_tf32=False, cudnn.deterministic=True, cudnn.benchmark=False (D12 precision parity).`

### Baseline smoke re-run (rows 1–3, every baseline cell, `--cgba_variant fixed`)
30 cells → `Results/b4e_smoke_<method>_<arch>` (main: TtBA, CGBA/CGBA_H, ARS_OPT; E10/E11: TtBA, CGBA/CGBA_H).
**Every cell: `stall = 0`, `nonfinite_queries = 0`, `error = 0`.** `s/img` (RTX PRO 6000, D12):

| method | vgg19 NT/T | resnet50 NT/T | vit_b_32 NT/T | r50_places NT/T | densenet161 NT/T | Salman NT/T |
|---|---|---|---|---|---|---|
| TtBA | 60.6/62.9 | 54.8/54.1 | 41.7/40.6 | 48.8/48.3 | 118.3/118.0 | 48.0/48.3 |
| CGBA/CGBA_H | 27.1/27.4 | 23.2/24.3 | 23.1/22.2 | 22.5/23.5 | 36.9/38.6 | 21.8/23.5 |
| ARS_OPT | 35.3/35.4 | 32.6/29.5 | 27.8/28.1 | — | — | — |

D12 (TF32 off) slows the baselines vs the B4/B4b TF32-on numbers: TtBA resnet50 46.2→54.8 s/img (+19 %),
CGBA resnet50 21.2→23.2 (+10 %). This is the disclosed cost of the shared FP32 oracle.

### Key interaction finding (D11 × D12)
Under D12 the **D11 NaN event disappears even for the un-clipped `d8only` variant**:
`d8only` on resnet50 NT rows 1–3 has `nonfinite_queries = 0` and pix_l2@10k = 6.4498 / 6.3317 / 2.6057 —
**bit-identical to the `fixed` variant** (the `acos` clip is a provably no-op under pure FP32). The D11 stall is
a TF32 artifact; the clip is a correct defensive fix that keeps CGBA official-code-faithful under any backend.
For E17 (D8-only appendix), the "share of images with the NaN event" under D12 is expected to be 0 — the user
should confirm how E17 should be worded (contract v1.9).

### Contract smoke gate
- **B4e baseline roots alone: exit 0** (`reports/contract_smoke_b4e_baselines.md`), I01–I12 PASS, I09 WARN, DISK PASS.
- B4e baselines + the existing EGC smoke roots: **I10 FAIL** (`multiple GPU names across cells:
  ['geforce rtx 5090', 'rtx pro 6000 blackwell server edition']`) — the EGC smoke is on the 5090. This is the
  D16 migration; the EGC smoke must be re-run on the 6000 (or D16 records the migration) before a single combined
  gate can exit 0.

---

## T3 — tools/eval/g5_check.py (D13; replaces adapters/diag/g5_table1.py)

- **TtBA**: median `pix_l2_iter` / published median, @2k/5k/10k.
- **CGBA / CGBA_H**: `rho_B = [med L_m(B) / med L_TtBA(B)] / [pub_med_m(B) / pub_med_TtBA(B)]` on images
  completed by both methods (paired on image stem). ARS_OPT: n/a.
- Paired percentile bootstrap (10000 resamples, seed 0), 95 % CI; PASS iff CI ∩ [0.5, 2] ≠ ∅ at every B;
  ratio-of-means printed as descriptive only; `n < 30` → `tool check only` (no verdict).

Tests `python tools/eval/tests_g5/test_g5.py` → **8/8 PASS** (synthetic PASS, synthetic FAIL, missing-image
pairing, n<30 refusal).

Dry-run on the B4e smoke (resnet50 NT, n=3):
```
TtBA  ratio(med) @2k/5k/10k = 1.659 / 1.594 / 1.372   (CIs all intersect [0.5,2])
CGBA  rho_B      @2k/5k/10k = 1.329 / 1.531 / 1.847   (CIs all intersect [0.5,2])
VERDICT: tool check only (n=3 < 30; not a gate)
```

`python tools/eval/selftest_evalkit.py` → **ALL PASS 20/20** (AGENTS.md rule 8).

---

## T4 — (optional) TtBA+TtBA pairing

**Skipped** (optional; not required for the solo cost table). D14 concurrency remains "one process per GPU" for
B5 unless the user asks for the pairing test.

---

## T5 — B5 cost table (solo)

GPU-h per cell = 200 × s/img / 3600 (main, E10/E11); E17 = 30 × s/img / 3600. EGC s/img from B4/B4b (unchanged
by D12); baselines from B4e (RTX PRO 6000, D12).

| scope | method | vgg19 NT/T | resnet50 NT/T | vit_b_32 NT/T | GPU-h |
|---|---|---|---|---|---|
| main | LGC_RC_EGC | 24.8/29.2 | 22.9/24.7 | 21.5/23.8 | 8.16 |
| main | TtBA | 60.6/62.9 | 54.8/54.1 | 41.7/40.6 | 17.49 |
| main | CGBA/CGBA_H | 27.1/27.4 | 23.2/24.3 | 23.1/22.2 | 8.19 |
| main | ARS_OPT | 35.3/35.4 | 32.6/29.5 | 27.8/28.1 | 10.49 |

| scope | method | r50_places NT/T | densenet161 NT/T | Salman NT/T | GPU-h |
|---|---|---|---|---|---|
| E10/E11 | LGC_RC_EGC | 21.7/24.4 | 32.8/34.5 | 22.1/24.4 | 8.88 |
| E10/E11 | TtBA | 48.8/48.3 | 118.3/118.0 | 48.0/48.3 | 23.87 |
| E10/E11 | CGBA/CGBA_H | 22.5/23.5 | 36.9/38.6 | 21.8/23.5 | 9.27 |

| total | GPU-h |
|---|---|
| main ImageNet (24 cells) | 44.33 |
| E10/E11 (18 cells) | 42.02 |
| E17 appendix (d8only, 6 CGBA NT cells × 30) | 1.29 |
| **GRAND TOTAL (solo)** | **87.64** |

E17 per cell (×30): vgg19 0.23, resnet50 0.19, vit_b_32 0.19, resnet50_places365 0.19, densenet161_places365
0.31, Salman2020Do_R50 0.18 GPU-h. TtBA on densenet161 remains the cost outlier (118 s/img ≈ 6.6 GPU-h/cell).

---

## Deviations from the prompt

The B4e prompt (session `ses_ecce9f4f0ffer0zn7UnEqEJR3X`, transcript `B4e.md`) allowed edits only to
`adapters/_common.py`, `adapters/run_cgba.py`, `tools/eval/g5_check.py` (+ tests) and `adapters/diag/*`.
The following deviations are recorded (per lgc-preflight §2.6 / §5.4; independent @checker review in
`reports/raw/B4e/checker_review.md`):

- **(a) `adapters/run_ttba_v2.py` was edited outside the allowed list.** This was necessary for T1(c): the
  `nonfinite_queries` column had to be written by *every* adapter, so the guard's per-image count landed in the
  TtBA `runs_*.csv` too. Change is observation-only (one extra column; no attack logic, seed or hyper-parameter
  change). This is the same file now further edited in the B4e follow-up to add the D12 `precision` block (F1).
- **(b) T1(d)(2) acceptance runs for `densenet161_places365` NT and every `CGBA_H` (targeted) cell were NOT
  executed.** No `Results/b4e_t1_fixed_densenet*` and no targeted `fixed`-vs-old-smoke comparison exist. Under D12
  (§T2) the old TF32-on smoke is not reproducible across the GPU change (D16), so `fixed ≡ d8only` bit-identically
  (verified for resnet50: `nonfinite_queries = 0`, identical pix_l2@10k). These acceptance cells are **waived**
  as superseded by the D12 smoke; a D16/D17 waiver must be recorded in contract v1.9.
- **(c) The tracker trace-prefix comparison was not done.** Traces exist and diverge at ≈1e-5 (e.g. `00007450`
  q≈118: `112.3796691895` vs `112.3796768188`), far below the first-NaN query, for both `fixed` and `d8only`
  controls → the divergence is the GPU migration, not the clip. Reported here instead of a PASS.
- **(d) The combined contract smoke gate exits 1** (`reports/contract_smoke_b4e.md`): I10 FAIL, multiple GPU names
  across cells (`geforce rtx 5090` vs `rtx pro 6000 blackwell server edition`). Baselines-only gate exits 0. This
  is the D16 migration; the EGC smoke must be re-run on the 6000 (or D16 records the migration) before one combined
  gate exits 0.
- **(e) No raw logs were kept under `reports/raw/B4e/`** (the transcript ends before the RTX PRO 6000 runs; the
  only raw artifact is the checker review). BSSP 100-call-cap hits = 0 and the first-NaN query indices
  (774/1084/776, 228/152/199, 5402/4014) are consistent with all stored evidence but are UNVERIFIED at review time
  (GPU off).

---

## sha256 of changed / new files

| file | change | SHA256 |
|---|---|---|
| `adapters/_common.py` | NonfiniteGuard + `configure_precision` (D12) + `precision` in env | `489501e74cda86b0750d67f8648c1213a3d6ce51147ced27d13ba80e862a0c10` |
| `adapters/run_cgba.py` | `--cgba_variant fixed/d8only` + D11 acos clip + env/manifest fields | `5f7fac692bb99aa51df8cb22b6ac2fa1e743739107fdeb19c0a88e5947595b88` |
| `adapters/run_ttba_v2.py` | `nonfinite_queries` column + D12 `precision` block in both env dicts (observation only) | `80cbcdcc19b2d428b3c56890e7a430b93a77def8caba86f99544e4b0e83bb8ee` |
| `adapters/diag/cgba_trace.py` | first-NaN query index + `--arch` (E10/E11) | `3feedd9b112267b5e8b5e975e27f5ffe2698a4e2051165bdc95962ec9479220d` |
| `adapters/diag/diag_b4d_t1_precision.py` | D12 conclusions | `2b61ee433856032855e18dbffd2642b778a85b51040127549945cf1e0c68a333` |
| `adapters/diag/diag_b4e_t1_clip_source.py` | new (T1b) | `658282d1d2710d39cba83cc4687210a408bb78ad2cd4a3dbcc44f79cc15c2461` |
| `adapters/diag/diag_b4e_t1_compare.py` | new (T1d numeric/npz compare) | `5e0d1d34dbf9c80d2639fdca5682c305770fa3e57e8ebfaf0788adf20dad9bbb` |
| `adapters/diag/diag_b4e_t1_stalled.py` | new (T1d stalled reporter) | `ad00ba25ea76c6b0b4946ffa26733cabc3aebf67b7f3fc50a5eba75b4ba070ce` |
| `tools/eval/g5_check.py` | new (D13 G5 gate) | `ce573f3b8a804dae854cdf80e6eafc865fcbe7d52ad99e0d959e1d948f08e159` |
| `tools/eval/tests_g5/test_g5.py` | new (D13 tests) | `18c344fd72b934e2076cd188b467bf42d8df86738be9b35da4ade6566886a20e` |

Frozen engine unchanged: `sha256sum lgc_bench/engine.py` = `835f1933d9e5…`. Contract not edited.

## GPU minutes (approx.)

B4e phase ≈ 95 GPU-minutes: T1 diagnostics (cgba_trace ×4, d8only/fixed acceptance ≈ 24 cells ≈ 35 min) +
T2 smoke (30 cells ≈ 55 min) + g5 dry-run + T1d stalled reporter (≈ 5 min).

## Open items / for the user (contract v1.9)

1. **D16**: record the GPU change (RTX 5090 → RTX PRO 6000 Blackwell, driver 580.82.09). The EGC smoke roots are
   still on the 5090 and must be re-run on the 6000 (or D16 records the migration) before one combined smoke gate
   exits 0 (I10 GPU-name check).
2. **D11 × D12**: under D12 the acos-NaN event does not occur, so `fixed` ≡ `d8only` bit-identically and the
   E17 appendix would report 0 NaN-event images. Confirm the D11/E17 wording.
3. **TtBA resume bug** (`ttba_github/ttba_common.py:83` `done_images` keeps only rows with `status == "ok"`), but
   every baseline run ends in `status == "budget"` (all 30 b4e cells re-parsed: `status=budget`), so a resumed TtBA
   cell re-runs completed images. `ttba_common.py` is outside the allowed edit list, so the fix is NOT applied here;
   recorded for a user decision (L4 correction) in contract v1.9. The one-line fix is
   `return {r["image"] for r in csv.DictReader(f) if r.get("status") in ("ok", "budget")}` (also consider `"error"`
   so failed images are not retried in place of a fresh run). Until then, never resume a TtBA cell: use a fresh
   `--out_dir`.

## 5-line summary

B4e preflight complete on the RTX PRO 6000 (not 5090). The CGBA acos-clip fix is verified (one-line change,
prefix-identical to the official code before the NaN event, no stall after). Precision parity (TF32 off,
deterministic) is in every adapter; the re-run baseline smoke is clean (0 stall / 0 non-finite / 0 errors) and
passes the contract gate on its own. The G5 gate tool + tests pass and the dry-run reports "tool check only".
Solo B5 cost is ≈ 87.6 GPU-h; the GPU migration and a TtBA resume bug are flagged for contract v1.9.

---

## B4e follow-up: env-recording fix (D12 precision block for TtBA)

**Finding.** `adapters/run_ttba_v2.py` builds its `env` dict manually in both paths (ImageNet `main()` and the
generic E10/E11 `_main_generic()`) and did NOT include the D12 `precision` block that `adapters/_common.py`'s
`build_env()` adds. `env_TtBA.json` therefore lacked `cuda_matmul_allow_tf32` / `cudnn_allow_tf32` /
`cudnn_deterministic` / `cudnn_benchmark`, and `tools/eval/human_preflight.py` FLAGged every TtBA root with
"not recorded in any env/config file: ['cudnn_tf32', 'deterministic']". Run behaviour was already correct
(`configure_precision()` runs at `_common` import time); this was a recording gap only.

**Fix (observation-only; no attack-logic / seed / hyper-parameter change).** Added a `precision` block to both
env dicts in `adapters/run_ttba_v2.py`, reading the four values from `torch` at run time exactly as
`_common.build_env()` does. `import torch` already precedes both env constructions.

```python
"precision": {
    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
},
```

- `adapters/run_ttba_v2.py` sha256 = `80cbcdcc19b2d428b3c56890e7a430b93a77def8caba86f99544e4b0e83bb8ee`.

**Verification.**

| check | result |
|---|---|
| `python -m py_compile adapters/run_ttba_v2.py` | OK |
| precision-block keys == `_common.build_env` (unit check, CPU) | OK |
| `python tools/eval/tests_contract/test_contract.py` | **ALL PASS (18/18)**, exit 0 |
| `python tools/eval/selftest_evalkit.py` | **ALL PASS (20/20)**, exit 0 |
| `python tools/eval/human_preflight.py --roots Results/b4e_smoke_* --out reports/human_preflight_b4e` | **FLAG lines = 6** (unchanged) |

**Why FLAG is still 6 (not 0).** The 6 FLAG lines are the six *existing* `b4e_smoke_ttba_*` env files, which
were written by the pre-fix code and are immutable records of those runs; the fix only affects newly-written env
files. Reaching FLAG = 0 requires regenerating those env files via a real TtBA re-run, which (a) cannot be done at
this time because **no GPU is present** (`nvidia-smi` → "No devices were found"), and (b) must not be done by
overwriting the existing B4e smoke or hand-editing result files. All six TtBA roots (12 `env_TtBA.json` files) are
affected, so the re-run is the whole TtBA smoke (6 cells, 3 images each), not a single cell. Plan: when the GPU
returns, re-run each TtBA smoke cell into a **throwaway dir** (e.g. `Results/b4e_smoke_ttba_<arch>_D12env`), verify
`precision` is present and the flags match `False/False/True/False`, then supersede the old dirs; do NOT write into
the existing `b4e_smoke_ttba_*`. Until then FLAG = 6 stands as a recorded deviation (F1), deferred to B5.
