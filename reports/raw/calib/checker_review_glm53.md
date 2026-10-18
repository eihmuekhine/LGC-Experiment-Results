# Checker review — calibration, phase B4d (review mode, lgc-preflight §6)

- **Reviewer:** independent checker (@checker, GLM 5.3), read-only. Only this file was written.
- **Repository under review:** `/root/autodl-tmp/calib_b4d`, clean at commit **`88a7443`** ("B4d preflight: T1-T6
  diagnostics + g5_table1 gate tool + D11 CGBA stall verdict (b)"). The B4d phase work is the single commit
  `4551440..88a7443` (9 files: 8 new `adapters/diag/*` scripts + `reports/B4d_preflight.md`, 1359 insertions,
  0 deletions). "Base commit: 88a7443" is read as "review the commit 88a7443 against its parent 4551440".
- **Spec:** `reports/raw/B4d/prompt.txt` does **not** exist (no `reports/raw/` at all in the repo). The transcript
  `/root/autodl-tmp/Latent-Geometric-Chords/B4d.md` contains **no user message** (it starts mid-session with the
  assistant's thinking); the spec was **reconstructed from fragments quoted in the assistant's thinking**
  (B4d.md lines 44, 56, 1476–1480, 2122–2126, 2792–2800, 3094–3101, 6453, 8732–8753, 4294/9515). Recovered items:
  T1 precision parity ("Print, for every runner … and where each is set (file:line)"; "Report any place our runners
  DISABLE TF32"), T2 TtBA DataParallel diagnosis (grep clone; `diag_ttba_profile.py` on resnet50 NT row 1; explain
  46 s vs 19 s), T3 concurrency ("Solo: run ONE already-completed image … This is the reference"; concurrent pair
  "TtBA resnet50 NT + LGC_RC_EGC resnet50 NT, OMP=8"; "throughput … over >= 5 images each. Verdict: pass if >=
  1.6x"; "Confirm cudnn.benchmark == False in every runner"), T4(a)–(f) CGBA stall (scan / trace with per-candidate
  logging / unpatched crash / grep TtBA / machanic cross-check "same victim, images, tracker and flags … never
  reported as CGBA" / D11 verdict), T5 `ttba_table1.json` register (commit with a specific message; verify 72 cells
  within 0.01), T6 G5 gate tool (`g5_table1.py`, "PASS iff both ratios in [0.5, 2]"). Allowed edits:
  `adapters/diag/*`, `run_frame_b1_generic.py` (guards/env only), `tools/eval/check_contract.py` + tests,
  `ttba_github/ttba_table1.json` (commit only). "GPU ~5 min" (T3). The reconstruction may be incomplete.
- **Checker environment:** no GPU (`nvidia-smi` → "No devices were found"); **`Results/` is absent from the review
  repo**, so every result-dependent number is UNVERIFIABLE here; all verification below is static/CPU.
- Contract v1.7 sha256 re-verified: `05923ff049397f1912ee5677c0d77eb0b85351cf25823fae0dd2505e02e6753f` (matches the
  report). Engine `835f1933d9e57107…` (frozen, not in the diff). Diff is entirely within the allowed edit list; no
  `lgc_bench/*`, no `run_benchmark_33b.py`, no contract, no prereg, no official clone touched.

## Per-T verdicts (with my own evidence)

### T1 — precision parity: **PASS on the verifiable facts; the fairness judgment is unsupported (see §3.1)**

Re-derived independently, all confirmed:
- torch 2.7.1+cu128 fresh-process defaults (re-run by me): `matmul.allow_tf32=False, cudnn.allow_tf32=True,
  cudnn.deterministic=False, cudnn.benchmark=False, float32_matmul_precision=highest` — matches the report.
- `lgc_bench/env.py:46-52` `configure_backend` is the only setter among our runners; called at
  `run_benchmark_33b.py:1157` (flags `--allow_tf32`/`--nondeterministic` at `:460-461`, both `store_true`;
  `run_frame_b1.sh` passes neither — grep empty, verified) and `run_frame_b1_generic.py:95`.
- No precision setting anywhere in `adapters/` outside `diag/` (grep empty; only legacy `.ipynb_checkpoints/`
  hits, which are not runners). `cudnn.benchmark=False` everywhere (statically true for all five runners).
- TtBA ImageNet: official `TtBA/dataset.py:220` sets `cudnn.deterministic=True` (read by me; clone @ `ea6d1649`).
- **Gap found:** `adapters/run_ttba_v2.py` calls `load_dataset_model` **only in the ImageNet `main()` (line 321)**;
  the E10/E11 `_main_generic` (line 78) never sets deterministic → **TtBA E10/E11 cells run `deterministic=False`**.
  The report's single TtBA row ("deterministic True") is ImageNet-only and hides this. The transcript proves the
  author noticed ("TtBA E10/E11: deterministic=False (default) … a discrepancy worth noting in T1", B4d.md
  ~line 1515) — the report omits it.

### T2 — TtBA DataParallel: **PASS** (citations verified; run numbers UNVERIFIED)

- Every grep citation re-verified in the official clone @ `ea6d1649`: `dataset.py:220`
  (`torch.backends.cudnn.deterministic = True`), `dataset.py:290-293` (`resnet50().cuda()` → `load_state_dict` →
  `nn.DataParallel(model, device_ids=[0])`), `dataset.py:337` (`return test_loader, torch_model.cuda()`),
  `TtBA.py:22` and `general_torch_model.py:31/33` (`.cuda()`).
- `diag_ttba_profile.py` read in full: implements exactly the spec (chain printout, DataParallel-in-chain check,
  per-query forward median over the first 200 + all, and a sync'd 200-call control `DataParallel.forward` vs raw
  `.module`); `TimingVictim` is observation-only. Arithmetic in the report is consistent (0.456/2.902 = 15.7 %;
  10 000 × ~3.6 ms ≈ 36 s vs the measured 36.2 s).
- The measured values (2.589/2.603/2.902/2.446 ms, 36.2 s) and the "19 s/img is an unverified estimate"
  characterization: UNVERIFIED (no GPU/Results); the 19 s figure is the skill's dev timing, so the
  characterization is fair.

### T3 — concurrency: **PASS on design; numbers UNVERIFIED; GPU budget exceeded**

- `diag_b4d_t3_compare.py` read: implements the spec's identity check exactly (found/state_query/pix_l2 exact +
  npz `delta` sha256 per checkpoint, verdict IDENTICAL or first diverging checkpoint). The pair
  (TtBA resnet50 NT + LGC_RC_EGC resnet50 NT, OMP=8, same image row 1) matches the spec.
- Throughput arithmetic re-checked: 10/290 = 0.0345 vs 10/296.5 = 0.0337 → 1.022× ✓; verdict "FAIL the 1.6×
  threshold → B5 solo" follows.
- The spec budgeted "GPU ~5 min" for T3; the report discloses T3 ≈ 18 min (incl. two aborted/redone attempts),
  total ≈ 25 min — an overrun never flagged against the spec (each launch was smoke-scale, so AGENTS.md rule 10
  is not violated, but the prompt's budget is).

### T4 — CGBA stall / D11: **PASS on substance; evidence gaps listed in §1/§2**

- **(a)** `diag_b4d_t4a_stall.py` read: implements the spec (14 cells = B4 6 + B4b 6 + T1-rerun 2; pix_l2 at
  1k/2k/5k/10k; `stalled = pix_l2@2000 == pix_l2@10000`). The raw output preserved in the transcript (B4d.md
  ~5100–5150) matches the report's summary **exactly** (7 unique stalled: r50 NT 761/892/1144, Salman 155/672/450,
  r50places 1344; CGBA_H never; vgg19/vit/densenet161 never). The script's raw count is **8** (row-1 r50 appears
  in both the B4 cell and the T1-rerun cell); the report says "7 unique" without explaining the dedup.
- **(b)** `cgba_trace.py` read in full: logs everything the spec demanded (theta_t, per-candidate psi_i/cos(psi_i)/
  r_i/clip displacement/label/best before-after, cap fired, BSSP distance ratio) and its BSSP copy is faithful
  (unclamped acos, read-only instrumentation). **But the docstring's "Replicates go_to_boundary_CGBA
  (proposed_attack.py:151-167) verbatim" is false:** `cgba_trace.py:109` wraps the walk-theta acos in
  `torch.clamp(…, -1.0, 1.0)` where the official `:155` is unclamped (read by me) — the instrument silently guards
  one of the two unclamped acos sites, so a NaN at `:155` in the official code is unobservable in the trace. (The
  m/zeta computation, including the `.cpu()`/`.item()` quirks, IS verbatim.) Run outputs (first geometry deviation
  None; 45/43/41 capped walks; acos arg 1.0000001): UNVERIFIED.
- **(c)** `diag_b4d_t4c_unpatched.py` read: implements the spec. `proposed_attack.py:252`
  `assert self.all_queries == q_num` verified in the official clone. Crash table: UNVERIFIED (consistent with the
  hard-coded OFFICIAL values in t4e and with the later B4e-era data).
- **(d)** Re-ran `grep -rni cgba /root/autodl-tmp/TtBA` myself → **no hits**. PASS.
- **(e)** `diag_b4d_t4e_machanic.py` read: the "same flags" claim is **verified** — dim_reduc_factor=4,
  iterations=93, initial_query=30, tol=1e-4, sigma=2e-4 match `run_cgba.py` PARAMS (calib `adapters/run_cgba.py:43-50`,
  sourced to `Non_targeted_attack.py:30/33`, `proposed_attack.py:15` in B1_sources.md), and `initial_query=30`
  deliberately overrides the machanic default 100 (`attack_gpu.py:31`). Same victim loader (`C.load_victim`) and
  tracker class. `attack_gpu.py:261-264` verified: unclamped acos commented out, clamped active (clone @
  `ab71cb88`). Caveats: the official-side comparison numbers are **hard-coded** (`:95-99`), and the machanic run
  needs import shims (glog/bidict/npz_dataset/StandardModel stubs + a `DataLoaderMaker` monkey-patch — disclosed
  in the docstring). Run numbers: UNVERIFIED.
- **(f)** Verdict **(b)**: the reasoning is sound within the contract v1.7 tree (read by me): not (a) — B3b
  bit-identity cited; not (c) — the machanic cross-check keeps improving; therefore (b) at `:182`, which I verified
  is unclamped in the official code. Note the classification tension: T4(b) itself proves the walk formula is
  Alg.-1-consistent, and the defect is numerical robustness (NaN), not a formula deviation — the tree has no
  "numerical defect" category, and (b) is the closest fit leading to the same action (clamp, m=3, disclose). The
  proposed minimal fix matches the machanic guard exactly.

### T5 — ttba_table1.json register: **PASS (fully verified)**

- sha256 of the worktree copy `2cb224f682f9f5735c219a3adeb8b513cbec033c903d4be14c864af682d9059e` = the main-repo
  source copy (recomputed both) → "bit-identical" ✓. Top-level keys `["format","source","table1"]` ✓; victims
  `["efficientnet_b0","inception_v3","resnet50","vgg19","vit_b_32"]` ✓; format string ✓; **300 value-cells** and
  the **72-cell G5 subset** (CGBA-NT, CGBA-H-T, TtBA-NT, TtBA-T × 3 victims × 3 budgets × 2 stats) both recomputed
  by me ✓. `diag_t3a_magnitude.py` reads the worktree copy first (`REPO = /root/autodl-tmp/LGC_frozen`, lookup
  order worktree → main repo) ✓. File already committed in `c823018`; no new commit (empty commit forbidden) —
  disclosed. Note: "verified against the source" means against the main-repo JSON (a copy check), not the paper
  PDF; paper fidelity rests on B4c's transcription.

### T6 — G5 gate tool: **PASS on spec compliance; design weaknesses in §3.3**

- `g5_table1.py` read in full: implements the spec exactly (mean/median pix_l2 @2k/5k/10k over `found==1`, method
  from the file name, ratio(mean)+ratio(median), PASS iff every ratio ∈ [0.5,2], exit 0/1/2, writes nothing).
- Weaknesses: `REPO = "/root/autodl-tmp/LGC_frozen"` hard-coded (breaks portability/reviewer package); dead
  `CELL_METHODS` dict; **no small-n guard** (verdicts printed at any n — the report's smoke section presents n=3
  PASS/FLAG verdicts without qualification); unpaired comparison against a published table from a different
  image list/environment (the JSON's own format string says "NOT paired with our runs"). Smoke outputs: UNVERIFIED.

### Report hygiene

- sha256 table: **all 9 values re-verified, all match.**
- "HEAD `4551440`" is stale — the report is committed in `88a7443` (same pattern the B4e review flagged).
- **No raw logs under `reports/raw/B4d/`** (the directory does not exist): the T4(a) per-image table survives only
  in the session transcript; the T2/T3/T4(b)/(c)/(e) outputs are not archived. **No "Deviations from the prompt"
  section** (lgc-preflight §5.4). "No B5 work started" ✓ (diff contains only diagnostics + report).

## (1) Where the implementation does not meet the spec

1. **T5 commit not created** as specified ("B4d: register ttba_table1.json …") — the file was already committed in
   `c823018`; an empty commit is forbidden. Disclosed in the report; benign.
2. **T1 table incomplete for `run_ttba_v2.py`**: the spec asked for the values "for every runner"; the TtBA
   adapter has two paths with different backends (ImageNet `deterministic=True` via official `dataset.py:220`;
   E10/E11 `_main_generic` `deterministic=False`) and only the ImageNet values are reported. The transcript shows
   the author identified this and the report dropped it.
3. **T3 GPU budget**: spec "~5 min"; actual T3 ≈ 18 min, phase total ≈ 25 min. Disclosed numerically, never flagged
   as a spec overrun.
4. **`cgba_trace.py` is not a verbatim replica** (clamped walk-theta at `:109` vs official unclamped `:155`),
   contradicting its own docstring and the report's framing — the T4(b) trace cannot observe a NaN at the `:155`
   site.
5. **No raw logs under `reports/raw/B4d/`** (skill §2.5) and no "Deviations from the prompt" section (§5.4).
6. **T4(e) hard-codes the official comparison numbers** (`diag_b4d_t4e_machanic.py:95-99`) instead of reading them
   from the result CSVs — the table cannot be regenerated or checked from files.

## (2) Report claims the evidence does not support

1. **"…required for the frozen engine's bit-identical reproducibility (B0/B4c prove the engine is bit-stable) and is
   a property of the frozen engine, not a fairness gap."** B0/B4c prove the engine is bit-stable *under its own
   configuration*; they say nothing about the fairness of comparing it against baselines that run a numerically
   different oracle (cudnn TF32 on). The "not a fairness gap" conclusion is an unflagged judgment presented as
   fact — and it was later reversed by the user's D12 decision (contract v1.8, precision parity for every adapter).
2. **"Replicates go_to_boundary_CGBA … verbatim"** (T4(b) framing) — false at `cgba_trace.py:109` (see §1.4).
3. **The T1 row "run_ttba_v2.py … deterministic True"** — true only for the ImageNet path (see §1.2).
4. **"7 unique stalled images"** — the script's raw count is 8 (one image counted in two cells); the report does
   not explain the dedup (the underlying per-cell numbers are right per the transcript raw output).
5. **All run-dependent numbers** (T2 timings, T3 IDENTICAL verdicts and 1.02×, T4(b) trace findings, T4(c) crash
   table, T4(e) machanic table, T6 smoke verdicts) — **UNVERIFIED**: no GPU at review time and no `Results/` in the
   review repo; the only surviving raw evidence is the session transcript. They are consistent with the (verified)
   code and internally consistent.
6. **"HEAD `4551440`"** — stale; the report is committed in `88a7443`.

## (3) Methodological choices that would make the baseline comparison unfair

1. **Backend-flag asymmetry left in place and dismissed.** At B4d the comparison runs EGC with TF32-off +
   deterministic while CGBA/ARS_OPT run cudnn TF32-on + non-deterministic (and TtBA ImageNet TF32-on +
   deterministic): the victim oracle is numerically different across the methods being compared. The
   lgc-baselines fairness contract treats backend flags as task-layer identity ("identical for all methods,
   implemented in shared code"); the report's unilateral "not a fairness gap" pre-empted the user decision that
   later became D12. The correct B4d action was to flag it for a contract decision, not to dismiss it.
2. **TtBA intra-method inconsistency**: ImageNet cells deterministic=True vs E10/E11 cells deterministic=False —
   TtBA's own cells across victims are produced under different backend configurations (hidden by §1.2).
3. **The B5 G5 gate as designed** (`g5_table1.py`): unpaired mean/median ratio vs a published table from a
   different image list and environment, hard [0.5, 2] PASS/FAIL, no small-n guard, no CI — at small n it can pass
   or fail arbitrarily, and the report's smoke section already presents n=3 verdicts without qualification.
   (Later replaced by D13's paired-bootstrap `g5_check.py` with an n<30 refusal.)
4. **The machanic cross-check's wrapper parameters** (`norm='l2'`, `epsilon=0.0`, `init_rnd_adv=True`,
   `grad_estimator_batch_size=40`) have no documented mapping to the official CGBA setting (the five listed
   hyper-parameters do match — verified), and the run depends on import shims; as a never-reported cross-check this
   is acceptable, but "does not stall" is attributed to the clamp without excluding these differences.

## Deviations from the review prompt (checker's own)

1. `reports/raw/B4d/prompt.txt` absent and no prompt was pasted; the spec was reconstructed from quoted fragments
   in the `B4d.md` transcript (no user message survives there). Reconstruction may be incomplete.
2. "Read files only there" was relaxed in a controlled, disclosed way: I read (a) the `B4d.md` transcript in the
   main repo (spec reconstruction — unavoidable), (b) the pinned official clones `TtBA` @ `ea6d1649`,
   `baselines/CGBA` @ `73d3e9c`, `baselines/ARS-OPT` @ `ab71cb88` (only the cited files/lines, to verify the
   report's file:line citations), and (c) the main repo's `ttba_github/ttba_table1.json` (sha256 only, for the T5
   identity claim). I did **not** read LGC_frozen's code, reports or `Results/` (the post-B4d state); consequently
   every result-dependent claim is UNVERIFIED rather than re-checked.
3. No GPU at review time; all verification is static/CPU (source reads, sha256 recomputation, JSON structure
   checks, grep re-runs, arithmetic re-checks).
4. This review file is the only file written.

## Verdict

- **T1 PASS on facts / FAIL on judgment** (the fairness dismissal, §3.1; the E10/E11 omission, §1.2);
  **T2 PASS; T3 PASS on design** (numbers unverified; GPU budget exceeded); **T4 PASS on substance** (the D11(b)
  verdict is sound and its key code citations — unclamped `:182`, machanic clamp `:261-264`, `:252` assert — are
  all verified by direct reads), with the `cgba_trace.py` "verbatim" claim false; **T5 PASS (fully verified)**;
  **T6 PASS on spec compliance** with gate-design weaknesses.
- The B4d deliverables that matter for B5 (D11 verdict + minimal fix, solo-concurrency decision, table1 register)
  are supported by the verifiable evidence. The material risks carried forward were the unflagged backend-flag
  asymmetry (later reversed by D12), the TtBA E10/E11 flag gap, and a G5 gate that was later judged insufficient
  (D13) — all three are visible in the B4d report itself to a careful reader.
