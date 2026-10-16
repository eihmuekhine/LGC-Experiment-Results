# Checker review — phase B4e (review mode, lgc-preflight §6)

- **Reviewer:** independent checker (@checker), read-only. No files edited except this review.
- **Base:** `729cf19` → **HEAD:** `616b41a` ("B4e preflight: CGBA acos-clip fix (D11), precision parity (D12),
  g5_check gate (D13), B5 cost; GPU notice D16"), branch `paper-runs`, worktree `/root/autodl-tmp/LGC_frozen`.
- **Spec:** `reports/raw/B4e/prompt.txt` **does not exist** (searched both repos; `ls reports/raw/` fails in both).
  The prompt text was taken from the session transcript `/root/autodl-tmp/Latent-Geometric-Chords/B4e.md`
  (lines 11–71). The transcript covers only the first part of the session (it ends at the transient
  RTX 4080 SUPER blocker); the RTX PRO 6000 work is evidenced only by the worktree artifacts.
- **Checker environment:** no GPU visible at review time (`nvidia-smi` → "No devices were found"), so all
  GPU-requiring re-runs (cgba_trace, stalled reporter) were impossible; everything else was re-run on CPU.
- **Author's report:** `reports/B4e_preflight.md` (committed in 616b41a).

## 0. Environment and frozen-file checks — PASS

| check | evidence | verdict |
|---|---|---|
| contract v1.8 sha256 = `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` | `sha256sum contract/paper_contract.yaml` (re-run) | PASS |
| engine frozen, sha `835f1933d9e5…` | `sha256sum lgc_bench/engine.py` = `835f1933d9e57107…`; not in diff | PASS |
| no forbidden file touched | `git diff --name-only 729cf19..HEAD` = 13 files, all in `adapters/`, `tools/eval/`, `reports/`; no `lgc_bench/*`, no `run_benchmark_33b.py`, no `ttba_github/run_frame_b1.sh`, no contract, no prereg | PASS |
| allowed-list compliance | spec allowed `_common.py`, `run_cgba.py`, `tools/eval/g5_check.py`+tests, `adapters/diag/*`. **`adapters/run_ttba_v2.py` was also edited** (needed for T1c "every adapter" + `nonfinite_queries` column) — a deviation the report never lists | PARTIAL |

## T1(a) — `--cgba_variant` code — PASS

- `adapters/run_cgba.py`: `--cgba_variant {fixed,d8only}` (default `fixed`); `fixed` adds the
  `SemiCircular_boundary_search` override; `d8only` inherits the official method (verified by test, see T1b).
- Method ids: `CGBA`/`CGBA_H` (fixed), `CGBA_D8only` (d8only). Refusal of `d8only --mode targeted` re-run:
  `python adapters/run_cgba.py --mode targeted --cgba_variant d8only` → `[!] --cgba_variant d8only is
  non_targeted only …`, **exit 1**. PASS.
- `cgba_variant` + `paper_faithful_fix` ("clip acos arg, proposed_attack.py:182, D11_outcome") verified in
  `env_CGBA.json`, `env_CGBA_H.json`, `env_CGBA_D8only.json` and `run_manifest.json` (re-read by me).
- Manual source check: the override is a verbatim copy of the official
  `/root/autodl-tmp/baselines/CGBA/proposed_attack.py:171-196` (including original spacing quirks); the only
  change wraps the `:182` acos argument in `torch.clamp(..., -1.0, 1.0)`. PASS.

## T1(b) — source-diff test — PASS (reproduced)

`python adapters/diag/diag_b4e_t1_clip_source.py` →
`d8only: … not overridden = True`; `PASS: exactly one changed line (index 11), the acos line`; `VERDICT: PASS`,
exit 0. Matches the report.

## T1(c) — non-finite guard — PASS

- `adapters/_common.py`: `NonfiniteGuard` (observation-only; counts per-image NaN/±inf inputs inside the
  tracker's model), `_GuardedHardLabelTracker`, and the rebind `HardLabelTracker = _GuardedHardLabelTracker`.
  Runtime check: `import _common` → `C.HardLabelTracker` is `_GuardedHardLabelTracker`. All adapters
  (`run_cgba`, `run_arsopt`, `run_hsja`, `run_signopt`, `run_ttba_v2`) route through it (grep verified).
- `nonfinite_queries` column present in every `runs_*.csv` of all 30 b4e smoke cells and all `b4e_t1_*` dirs
  (re-parsed all of them). The uint8 validity check uses a separate `eval_model`
  (`tools/eval/hardlabel_tracker.py:198-200`), so evaluation-only calls are not counted. PASS.

## T1(d) — acceptance — MIXED (see per-item verdicts)

Re-run by me (CPU, from stored CSVs/npz/traces) with `adapters/diag/diag_b4e_t1_compare.py`:

1. **d8only resnet50 NT rows 1–3 vs `Results/b4_smoke_cgba_resnet50`** — IDENTICAL, 3/3 images, 8 npz deltas
   bitwise equal each. **PASS** (matches report).
2. **fixed on non-NaN images** —
   - vgg19 NT: fixed AND d8only (6000) diverge from the 5090 smoke at identical points (e.g.
     `00002148 @2500 pix_l2 0.2392140030860901` vs `0.23921401798725128`; `00037360 @500 state_query 497 vs
     499`) → GPU-caused, not clip-caused. Consistent with the report. **Cannot be "IDENTICAL" as the spec
     demanded — FAIL as written, cause disclosed (D16).**
   - vit_b_32 NT: fixed diverges on 2/3 images (`00024661 @500` ≈4.8e-7; `00046402 @5000` ≈1.5e-4). **FAIL
     as written (disclosed).** Note: **no d8only control run exists for vit_b_32** (`Results/` has d8only
     controls only for resnet50, vgg19, r50places, Salman), so the report's "proven by re-running d8only …
     same divergence" is not supported for vit.
   - **densenet161_places365 NT: NO fixed acceptance run exists** (no `Results/b4e_t1_fixed_densenet*`).
   - **every CGBA_H smoke cell: NO acceptance run exists** (all `b4e_t1_*` dirs are `non_targeted` only; no
     targeted fixed-vs-old-smoke comparison anywhere). The report's T1(d)2 sentence implies these cells were
     assessed; they were not run. **FAIL (not executed).**
3. **fixed on the stalled images** —
   - resnet50 NT 1–3: prefix identical through @500/@1000/@500 (re-run, `--cmp prefix`), consistent with
     first-NaN 774/1084/776; pix_l2 1k/2k/5k/10k in the report table match the CSVs to all printed digits
     (84.996/81.410/75.256/67.838 etc.). `nonfinite_queries=0`, no stall. **PASS.**
   - Salman NT 1–3: prefix identical through @200 (all 3); pix_l2 @10k = 30.08/30.37/3.96 ✓; Salman
     d8only-6000 == old b4b smoke bit-identical (re-run: 3/3 IDENTICAL, 8 npz each). **PASS.**
   - resnet50_places365 NT: row 1 prefix identical only through @500 although first-NaN = 5402 (budgets
     1000/2000/5000 < 5402 diverge: `00002172 @1000 state_query 975 vs 997`); row 2 diverges at @200
     (pix_l2 2.786 vs 2.742) with first-NaN 4014; row 3 diverges at @100. **FAIL as written** (GPU-caused;
     the §0 table discloses Places365 instability, but the T1d table row does not flag the miss).
   - **trace prefix:** the spec asked for "the tracker trace prefix, if traces exist". Traces exist, and the
     prefix is **NOT identical**: fixed-6000 vs old-5090 traces diverge at trace index 11/9/12 (e.g.
     `00007450` q=118: l2 112.3796691895 vs 112.3796768188, ≈7.6e-6) — far below the first-NaN query. The
     d8only-6000 traces diverge from old-5090 at the same points, and fixed ≡ d8only on the 6000 up to the
     NaN event (first divergence None / None / idx 72), so the cause is the GPU migration, not the clip. The
     report does not mention the trace comparison at all. **FAIL as written (undisclosed).**
   - BSSP 100-call cap hits = 0 and the first-NaN indices: **no raw logs stored** (`reports/raw/B4e/` does not
     exist; the transcript ends before these runs). The first-NaN values are *consistent* with all stored
     evidence (checkpoint-prefix divergence points; d8only trace last-improvement queries 761/892/1144 ≤
     reported first-NaN; post-NaN freeze), but cannot be independently re-verified without a GPU.
     UNVERIFIED (consistent).

**T1 overall:** code PASS; acceptance PASS only for the bit-stable backbones (resnet50, Salman). The spec's
"Any failure: STOP" was not followed literally — the author continued to T2 with the (documented, and in my
view correct) d8only-control argument that the divergences are GPU-caused. This is a deviation that the
report should have listed.

## T2 — precision parity — PASS on substance, one gate FAIL as written

- `adapters/_common.py:configure_precision()` called at import, before any model build; runtime re-check:
  `matmul.allow_tf32=False, cudnn.allow_tf32=False, cudnn.deterministic=True, cudnn.benchmark=False` after
  `import _common`. PASS.
- `python adapters/diag/diag_b4d_t1_precision.py` re-run → prints "ALL RUNNERS IDENTICAL … (D12 precision
  parity)". Note this is a static source-reading diagnostic (by design), which is why I also did the runtime
  import check above. PASS.
- Baseline smoke re-run: **30 cells on disk** (`b4e_smoke_{ttba,cgba,arsopt}_*`; 12 TtBA + 12 CGBA/CGBA_H +
  6 ARS_OPT), all `env_*.json` = `NVIDIA RTX PRO 6000 Blackwell Server Edition`, torch 2.7.1+cu128, contract
  sha v1.8, CGBA cells `cgba_variant=fixed`. Re-parsed every `runs_*.csv`: **stall=0, nonfinite=0, error=0,
  status=budget, queries=10000** in all 30 cells. s/img recomputed from `wall_time_s`: matches the report
  table (e.g. TtBA vgg19 60.6/62.9, densenet161 118.3/118.0; CGBA resnet50 23.2/24.3; ARS 32.6/29.5).
  PASS.
- Old-vs-new s/img: only two cells are given in the report (TtBA resnet50 46.2→54.8 = +19 %, CGBA resnet50
  21.2→23.2 = +10 %); both verified against `B4_cost.md` (46.21, 21.18). The spec asked for old vs new **per
  cell**; the other 28 old values are only in B4/B4b reports. PARTIAL.
- D11×D12 key finding re-verified: `Results/b4e_t1_d8only_r50_d12` has `nonfinite_queries=0` (runs CSV) and is
  **bit-identical to the fixed variant** (`b4e_smoke_cgba_resnet50`) at all 8 checkpoints + npz deltas
  (compare tool: 3/3 IDENTICAL); pix_l2@10k = 6.4498/6.3317/2.6057 as reported. PASS.
- Contract smoke gate: baselines-only → exit 0 (`reports/contract_smoke_b4e_baselines.md`, I01–I12 PASS).
  **Combined (B4e + EGC roots) → exit 1, I10 FAIL** (`multiple GPU names across cells: ['geforce rtx 5090',
  'rtx pro 6000 blackwell server edition']`, `reports/contract_smoke_b4e.md`). The spec demanded exit 0 →
  **FAIL as written**, disclosed by the author as the D16 migration.
- `env_*.json` precision block: present in CGBA/ARS_OPT env files, **absent from all 12 TtBA env files**
  (`'precision' not in env_TtBA.json`, re-checked). The report's "plus a `precision` block in `env_*.json`"
  is therefore only true for 18/30 cells; `reports/human_preflight_b4e.md` FLAGs this 6 times ("not recorded
  in any env/config file: ['cudnn_tf32', 'deterministic']") and the B4e report neither mentions nor explains
  these FLAGs. PARTIAL FAIL (recording gap; the flags themselves are set at runtime — verified).

## T3 — g5_check.py — PASS

- `tools/eval/g5_check.py` implements the spec: TtBA median `pix_l2_iter` vs published median @2k/5k/10k;
  CGBA/CGBA_H `rho_B` on images completed by both; ARS_OPT n/a; paired percentile bootstrap (10000, seed 0),
  95 % CI, PASS iff CI ∩ [0.5,2] ≠ ∅ at every B; ratio-of-means descriptive only; n<30 → "tool check only",
  no verdict (code read + dry-run).
- `python tools/eval/tests_g5/test_g5.py` re-run → **8/8 PASS, exit 0** (synthetic PASS, synthetic FAIL,
  missing-image pairing, n<30 refusal). Matches the report.
- Dry-run re-executed: `g5_check.py --arch resnet50 --mode non_targeted --ttba_root
  Results/b4e_smoke_ttba_resnet50 --cgba_root Results/b4e_smoke_cgba_resnet50` → TtBA 1.659/1.594/1.372,
  CGBA 1.329/1.531/1.847, "VERDICT: tool check only (n=3 < 30; not a gate)", exit 0 — **exactly the reported
  numbers**. PASS.
- AGENTS.md rule 8: `python tools/eval/selftest_evalkit.py` re-run → **ALL PASS (20/20)**. PASS.

## T4 — optional TtBA pairing — SKIPPED (allowed; report says so). PASS.

## T5 — B5 cost table — PASS

- All GPU-h arithmetic re-derived from the s/img values (which I re-derived from `runs_*.csv`): main
  8.16+17.49+8.19+10.49 = 44.33; E10/E11 8.88+23.87+9.27 = 42.02; E17 six cells ×30 images = 1.29
  (0.23+0.19+0.19+0.19+0.31+0.18); grand total 87.64. All match.
- EGC s/img traced to `B4_cost.md` (24.84/29.23, 22.86/24.68, 21.54/23.77) and `B4b_cost.md`
  (21.7/24.4, 32.8/34.5, 22.1/24.4) — correctly rounded in the report. E17 uses the b4e (fixed, D12) CGBA
  s/img; under D12 fixed ≡ d8only bit-identically (verified above), so the substitution is sound.

## Report hygiene

- sha256 table: all 10 values re-computed and **match**. Commit 616b41a contains exactly the 13 diff files
  (code + report). PASS.
- Report says "HEAD `729cf19`" — stale: the report is committed in `616b41a`. Minor inaccuracy.
- **No "Deviations from the prompt" section** (required by lgc-preflight §2.6/§5.4); no prompt hash; no raw
  logs under `reports/raw/B4e/` (§2.5). The spec file itself (`reports/raw/B4e/prompt.txt`) was never created.
- Unreported artifact: `Results/b4e_test_arsopt_r50` (ARS_OPT NT, 6000) exists but is not mentioned in the
  report.
- §0 imprecisions: "vit_b_32 … ≈1e-7" — one of the two diverging images differs by ≈1.5e-4 at its first
  divergent checkpoint (`00046402 @5000`); "resnet50 … bit-identical" holds at checkpoint/npz granularity but
  not at trace granularity (≈1e-5 differences, see T1(d)).
- TtBA resume bug (open item 3): code verified — `ttba_github/ttba_common.py:83` `done_images` keeps only
  `status == "ok"`, while every baseline run ends `status == "budget"` (all 30 cells re-parsed), so a resumed
  TtBA cell re-runs completed images. Current `b4e_smoke_ttba_vgg19` runs CSVs are clean (3 rows per mode),
  consistent with "fixed by a fresh re-run". The historical duplicate-row observation itself is UNVERIFIABLE
  (no log kept).

## Claims in the author's report that the evidence does NOT support

1. **"fixed on non-NaN images (vgg19, vit_b_32, densenet161 NT; every CGBA_H cell): bit-identical to the old
   smoke only for backbones that are bit-stable … (resnet50, Salman2020Do_R50)"** — no fixed acceptance run
   exists for densenet161 NT or for ANY CGBA_H (targeted) cell; for the bit-stable CGBA_H backbones
   (resnet50-T, Salman-T) the claim has no run evidence at all. The sentence reads as if these cells were
   assessed.
2. **"proven by re-running `d8only` (unclipped) and observing the same divergence"** — true for vgg19 (and
   r50places/robust), but there is **no d8only control run for vit_b_32**; its divergence attribution rests
   on inference, not on a run.
3. **"plus a `precision` block in `env_*.json`"** — false for all 12 TtBA env files (6 unexplained FLAGs in
   `human_preflight_b4e.md`).
4. **T1(d)3 as a whole is reported as satisfied** ("prefix identical before it") — the **tracker trace
   prefix** (explicitly requested "if traces exist") is not identical and was never compared in the report;
   the r50_places prefix-identity also fails below the first-NaN budget.
5. **"HEAD `729cf19`"** — stale in the committed report (actual HEAD 616b41a).
6. **"vit_b_32 … ≈1e-7"** — understates the observed 1.5e-4 divergence on one image.
7. **BSSP 100-call cap hits = 0 and the first-NaN query indices (774/1084/776, 228/152/199, 5402/4014)** —
   no raw logs stored; consistent with all stored evidence but UNVERIFIABLE at review time (GPU off).
8. **"≈ 95 GPU-minutes"** — plausible from the stored wall times but not audited (no run log).

## Deviations from the review prompt (checker's own)

1. `reports/raw/B4e/prompt.txt` does not exist; the spec was taken from the session transcript `B4e.md`.
2. No GPU at review time: `cgba_trace.py` and `diag_b4e_t1_stalled.py` could not be re-run; all other
   verification commands were re-run and are cited above.
3. This review file is the only file written.

## Verdict

- **Code (T1a/b/c, T2 flags, T3 tool+tests): PASS** — faithfully implements the spec; frozen files
  untouched; selftest 20/20.
- **T1d acceptance: FAIL as written** for vgg19/vit (GPU-migration divergence, disclosed), **NOT EXECUTED**
  for densenet161 NT and all CGBA_H cells; trace-prefix sub-criterion FAILS and is undisclosed.
- **T2: PASS on substance** (30 clean cells, D12 verified, D11×D12 finding reproduced), but the combined
  contract gate exits 1 (I10, disclosed) and TtBA env files lack the precision record.
- **T3/T5: PASS.**
- Before B5: (i) record the precision block in TtBA env files or explain the 6 FLAGs; (ii) run or explicitly
  waive the missing T1d acceptance cells (densenet161 NT, CGBA_H) under D12; (iii) fix the TtBA resume bug
  (`done_images`) before relying on resume; (iv) resolve D16 (re-run EGC smoke on the 6000 or record the
  migration in contract v1.9) so the combined gate can exit 0.
