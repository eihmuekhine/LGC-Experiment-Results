# B4f checker review (independent, review mode)

Independence: this session contains **no author work** — the reviewer only read the committed B4f
changes (base `69a763c` → HEAD `94ff027bd5e057c994149d371a9416d6139bfe70`) and ran read-only
re-verification. No files edited except this review; no git checkout/reset/commit;
`check_contract.py` run only with `--report /tmp/opencode/b4f_checker_contract.md`.

Scope hashes (recomputed): prompt `reports/raw/B4f/prompt.txt`
`66c94fbdc213483b235c16744e883719d1489b694fb5b43285d4aac15cdf9387`; contract v1.8
`95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`; engine
`835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` (untouched; the commit touches
no `lgc_bench/*`, `run_benchmark_33b.py`, `run_frame_b1.sh`, no official clone). GPU this session:
RTX 5090, 0 MiB used, driver 580.76.05 / CUDA 13.0 — matches the report's §2.

## Verdicts per T-item (reviewer's own evidence)

**T1 Loader audit — PASS.** Verified `ttba_github_tensor` (`lgc_bench/baseline_protocol.py:325-333`),
`load_image_places365` / `load_x0_any` (`adapters/_common.py:379-390`, `:405-415`), `load_x`
(`run_benchmark_33b.py:696-718`, `ref` discarded as claimed), TtBA ImageFolder loader
(`TtBA/dataset.py:173-182`; `run_ttba_v2.py:356` `img, lab = ds[by_name[name]]`). My own
`rg "Resize|CenterCrop|ToTensor|Image.open|read_image|interpolate"` over all executed adapter /
wrapper / runner / clone paths found no hit the author's table misses and no resize/crop/interpolate
after x0: ARS `F.interpolate` sites are gated (`:303-304` only via `get_prior_grad`, PARS-only at
`:317-327`; `:332`/`:350` by `resized=False`; `:709` by `best_initial_target_sample=False` at `:698`;
`:844` only in `attack_all :832`, never called); CGBA drivers never imported (`run_cgba.py:113`
imports only `proposed_attack`); TtBA uses `DataTools.cosine_similarity` only (`TtBA.py:185`).

**T2 Normalisation — PASS.** Runtime print: `VGG19_Weights.IMAGENET1K_V1` /
`ResNet50_Weights.IMAGENET1K_V2` / `ViT_B_32_Weights.IMAGENET1K_V1` transforms = mean
(0.485,0.456,0.406), std (0.229,0.224,0.225) = `baseline_protocol.py:56-57` and `_common.py:77-78`.
robustbench `model_zoo/imagenet.py:16-17` mu/sigma identical. TtBA `dataset.py:278-279,295-296,313-314`
identical. Places365 constants corroborated by `reports/E10_sources.md:18-19` and by the T6 bit-identity
check (below). One source quote is UNVERIFIED in the worktree (P8).

**T3 x0 identity and pairs — PASS.** Reran `OMP_NUM_THREADS=8 python adapters/diag/diag_b4f_t3_pairs.py`
on the RTX 5090 under D12 flags: output byte-identical to `reports/raw/B4f/T3_pairs.log`; all 12 CSVs
rows 1-200: 0 wrong source, 0 wrong target, 0 same-label, 0 dups (0/12 violations). Reran
`tools/eval/human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out /tmp/opencode/b4f_checker/hp`:
identical to `reports/raw/B4f/human_preflight.md` except the §6 output-path prefix; §2 = 18 keys,
0 hash mismatches, 0 single-root keys; FLAG lines = 7 (6 generic-runner `env_frame.txt` missing
cudnn_tf32/deterministic + 1 GPU-name string mismatch), exactly as the author reports.

**T4 Hyper-parameter audit — PASS.** Reran the ast extraction (`hparam_audit.sig_defaults`) myself:
TtBA `Attacker.__init__` = {attack_method 'TtBA', dim_reduc_factor 4, iteration 100, initial_query 30,
tol 1e-4, sigma 3e-4} (`TtBA.py:18-20`); CGBA = {dim_reduc_factor 4, attack_method 'CGBA_H',
iteration 93, initial_query 30, tol 1e-4, sigma 2e-4, verbose_control 'Yes'} +
`grad_estimator_batch_size 40` (`proposed_attack.py:13-16`, `:32`); ARS `ParsOptL2Norm.__init__` =
{batch_size 1, k 100, alpha 0.2, beta 0.001, iterations 1000, maximum_queries 10000, method 'ARS',
clip_grad_max_norm 1.0, best_initial_target_sample/resized/line_search False, dim_reduce_factor 4}
(`:16-19`); `PARS_OPT.json` ImageNet.l2 = eps 10.0, clip 1.0, tol 1e-4, lr 0.4/0.2. Adapter values
verified at `run_ttba_v2.py:55-59,162,242,349`, `run_cgba.py:50-57,128-136,150-153,178-184,255`,
`run_arsopt.py:36-44,50,63-94 (shims),98-121,132`; official driver overrides verified
(`attack.py:62` est-grad-samples 200; `attack.py:127-128` forces 20000 for ImageNet targeted, bypassed
by calling `targeted_attack` directly; `main.py:19-24,70`). The only L2 DIFFERS (TtBA epsilon 2.5) is
correctly classified: with `early=0` (`TtBA.py:242-243`) it only sets the recorded success flag
(`:239-241`), never the trajectory.

**T5 Oracle and query audit — PASS with two numeric errors (P1, P2) and one incomplete explanation
(P6).** (a) All victim-eval sites verified in `TtBA.py` (:23,65,78,126,164,245,260,296,308),
`proposed_attack.py` (:19,22,44,106), `pars_opt_l2_norm_attack.py` (:92,100,111,120,137,145,156,165,
181,194,207,220,254-273,372,376,596,667,683,711,802); each adapter passes the tracker as the model
(`run_ttba_v2.py:178/184` and `:370/376`; `run_cgba.py:180-184`; `run_arsopt.py:98-102,109-115`); in
`TtBA.py` the only non-`self.model` model use is the ctor call that stores it. (b) Termination sites
verified (`TtBA.py:204,237-238,242-243,275,293,162,306`; `proposed_attack.py:233,83,144,164,194,61`;
`pars_opt:593,618-619,750-751,198,224`); `early=0` and `maximum_queries=10000` verified. (c) Eval-only
`predict_label` on the raw model at `run_ttba_v2.py:171/363`, written to `x0_pred_ok` only at `:221/:413`
after `atk.attack()`; CGBA/ARS uint8 validity via `tracker.finalize(..., eval_model=victim)` (`_common.py:610-611`)
after the attack returns (`hardlabel_tracker.py:198-200`). (d) Reran `diag_b4f_t5_queries.py`: output identical
to the author's log; all baselines 3/3 at exactly 10000, 0 over budget, 0 non-finite; LGC 3/3 `ok` below
budget — but the report's quoted LGC query range is wrong (P1) and the cell count "9 + 9" is wrong (P2).
(e) TtBA tracker 10000 vs official 9779-9999, offsets 1-221 — verified from the log; mechanism consistent
with `TtBA.py:126-127` (batch call, then `self.queries += q_max`, so a mid-batch `BudgetExhausted` is
never added to the official counter) plus the ctor query at `:23` (tracker-only, +1) — the report's
explanation omits the ctor term (P6).

**T6 Domain and clipping — PASS on all measurements; the stated interval is wrong (P3).** Reran
`diag_b4f_t6_domain.py` on GPU: byte-identical to `reports/raw/B4f/T6_domain.log`. Independent numpy
recomputation of d_n/d_p from the three stored CGBA npz: 4.424224 / 4.424193 / 4.424066 — inside the
prompt's [1/0.229, 1/0.224] = [4.3668, 4.4643]. `victim(x0) == raw_net((x0-mu)/sd)` max abs diff 0
for all 6 victims (rerun). Clamp sites verified: `general_torch_model.py:53`, `_common.py:266-277`
(ClampedVictim), `:301-303`/`:306-310` (to_pixel maps), `run_cgba.py:178-179`, `utils.py:6-10`,
`proposed_attack.py:65,140,160,264`; ARS epsilon is a gate not a clamp (`pars_opt:666,672,813`); LGC
decoder clamps (`lgc_bench/spaces.py:69-70,121-126`).

**T7 Env recording — PASS with documented deviations; no explicit verdict line in the report (P7).**
Diff is env-recording only, within the prompt's allowed edits. `C.backend_flags()` returns exactly the
five keys the author quotes; `build_env` emits them (`_common.py:61-77`, `:472-478`);
`run_frame_b1_generic.py` now writes the five `backend.*` lines (`:126-130`); `py_compile` OK (rerun).
Frozen runner: `describe_device` (`lgc_bench/env.py:113-123`, tf32 token `:122`) written into
`config_single.json` (`run_benchmark_33b.py:1265-1267`); example string matches the report verbatim.
Historical B4e2 env files verified: `env_TtBA.json` / `env_CGBA.json` / `env_ARS_OPT.json` carry 4
precision flags, no `backend.*` keys, no `float32_matmul_precision` (Deviation 1 confirmed;
`run_ttba_v2.py:126-131,296-301` writes env inline); the 6 generic-runner `env_frame.txt` have no
backend lines (pre-fix). human_preflight §1 rerun: FLAG = 7, unchanged from B4e2
(`reports/B4e2_close.md:19`). `torch.backends.float32_matmul_precision` does not exist in torch 2.7.1
(`hasattr` → False), so Deviation 4 is correct. Post-edit smoke gate rerun:
`check_contract.py --stage smoke` exit 0, I01-I12 + DISK all PASS
(`--report /tmp/opencode/b4f_checker_contract.md`).

**Report hygiene — PASS.** All 12 sha256 in §5 recomputed: identical. §1 prompt/contract hashes
identical. "No B5 work" confirmed (diff touches only allowed paths). GPU minutes ≈3 plausible
(T3 rerun ≈3 min).

## Claims the evidence does not support

- **P1** `reports/B4f_audit.md:191` — "3/3 `ok` (9973–9976)": actual `total_queries` over the 36 LGC
  smoke rows is **9962–9979** (ImageNet-only 9965–9977; `cut -d, -f8
  Results/b4e2_smoke_LGC_RC_EGC_*/{non_targeted,targeted}/runs.csv`). Fix: re-quote 9962–9979.
- **P2** `reports/B4f_audit.md:191` — "(ours; 9 + 9 cells)": the T5 log lists **12** LGC cells
  (6 NT + 6 T from 9 roots), not 9+9. Fix: "12 cells (9 roots)".
- **P3** `reports/B4f_audit.md:217,236` + `adapters/diag/diag_b4f_t6_domain.py:93,116` —
  "[1/sd_max, 1/sd_min] = [4.367, 4.444]" is wrong: 1/sd_min = 1/0.224 = **4.4643** (prompt:
  [4.37, 4.46]); 4.444 = 1/0.225; the script's `1/sd_min`/`1/sd_max` labels are swapped (0.229 is
  sd_max). Measured 4.424 passes either interval, so the verdict stands. Fix: use [4.3668, 4.4643]
  and correct the labels.
- **P4** stale `_common.py` line numbers throughout the report (pre-dating the author's own
  `backend_flags()` insertion): report :60-61 → actual :77-78; :249-260 → :266-277; :284-286 →
  :301-303; :289-293 → :306-310; :362-372 → :379-390; :370 → :387; :388-397 → :405-415; :591-592 →
  :610-611. The functions exist and behave as claimed (each verified), but the citations do not
  resolve at HEAD. Fix: re-quote at HEAD.
- **P5** `reports/B4f_audit.md:22` (§2) — undisclosed driver drift: the audited B4e2 artefacts were
  produced under driver **580.105.08** (`reports/B4e2_close.md:26`,
  `Results/b4e2_smoke_LGC_RC_EGC_resnet50/env_frame.txt`), while B4f T3/T6 ran under **580.76.05**.
  T3 re-verified all labels under the new driver (0/12), so no invalidation, but the drift belongs in
  the environment block / Deviations. Fix: add a Deviations line.
- **P6** `reports/B4f_audit.md:198-202` (T5(e)) — the offset explanation omits the ctor query
  (`TtBA.py:23`, tracker-counted, never added to `atk.queries`), which is why the minimum offset is 1,
  not 0. Fix: "offset = 1 (ctor) + final partial batch".
- **P7** `reports/B4f_audit.md:240-275` (T7) — no explicit PASS/FAIL verdict, unlike T1-T6. Fix: add
  "Verdict: PASS with deviations (TtBA `float32_matmul_precision` gap; historical env files)".
- **P8** `reports/B4f_audit.md:294-296` (Deviation 5) — the claim that the official Places365 demo
  loader uses `Resize((224,224))` is UNVERIFIED in the worktree (no saved copy of
  `run_placesCNN_unified.py`; `reports/E10_sources.md:18-19` documents only the constants). Fix: mark
  UNVERIFIED or commit the fetched demo source.
- **P9** `reports/B4f_audit.md:28` — "OMP/MKL/OPENBLAS threads | 8 (every GPU command)" cites no log
  evidence (no output records thread counts). Fix: cite the command lines or drop the row.

## Other observations (no action in B4f scope)

- The worktree is not clean: 3 tracked files modified but uncommitted
  (`.opencode/skills/lgc-preflight/SKILL.md`, `docs/PREFLIGHT_RUNBOOK.md`,
  `reports/HUMAN_SIGNOFF_B5_TEMPLATE.md`) plus untracked files incl.
  `reports/raw/B4f/.ipynb_checkpoints/` — outside the reviewed commit; the human should be aware
  before B5.
- The frozen runner's `env_frame.txt` (ImageNet roots) has no `torch`-prefixed line, so
  human_preflight §1 shows torch "?" for those rows; pre-existing B4e2 format, not a B4f defect.
- ARS-OPT smoke covers only the 3 ImageNet victims (6 cells) — a B4e2 scope fact faithfully
  reflected by T5(d); no places365/robust ARS smoke exists to audit.

## Bottom line

All seven T-items reproduce under the reviewer's own commands (T3/T5/T6 logs byte-identical; smoke
contract gate exit 0; x0 identical; 0/12 pair violations; no query > budget; engine untouched; no
forbidden file edited). The nine findings above are citation/number accuracy defects (P1-P4, P6-P9)
and one undisclosed environment drift (P5); none changes a PASS verdict, but P1/P3 are factual errors
in numbers a reviewer would check first. Fix the report text before B5.
