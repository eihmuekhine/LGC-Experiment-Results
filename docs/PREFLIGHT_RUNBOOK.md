# PREFLIGHT RUNBOOK — from smoke to the first n = 200 run (LGC-EGC, TMM/PR)

Version 1.3, 2026-10-14 (v1.3: LESSONS.md ledger, STOP bundle, fast path B4e2 -> B4f -> B4g, human_preflight §7 degeneracy; v1.2: reviewer as its own agent in a new session, GLM 5.3, B4e2, calibration workspace). Binding order of work before B5. The contract (`contract/paper_contract.yaml`) wins over this
file; this file wins over older plans. Principle: **fairness = identical task, not identical algorithm** (layers L1-L4 in
skill `lgc-preflight` §1).

## 0. Roles, models and where they run

| id | role | who / model | where | may | must not |
|---|---|---|---|---|---|
| H | decides, signs, spot-checks | you | AutoDL shell, Mac, this Claude Project chat | everything in §5 "human-only" | delegate a §5 decision |
| L | lead analyst: specs, prompts, review of reports + raw evidence, root cause, contract drafts | **Claude Opus 5.5** (this Project chat; API id `claude-opus-5-5` if you call it locally) | chat (you paste logs) | write prompts, review, propose | run code on AutoDL, edit the contract |
| A | author/executor | OpenCode agent `coder`, **DeepSeek V4 Pro** (`deepseek/deepseek-v4-pro` in `opencode.json`) | AutoDL worktree `/root/autodl-tmp/LGC_frozen` | code within the prompt's allowed list, run, report | judge its own work as final, edit the contract/engine/clones |
| R | independent reviewer | OpenCode primary agent `reviewer`, model pinned to **GLM 5.3** (`openrouter/z-ai/glm-5.3`; adopted 2026-10-13 on the B4e reviews). Optional `reviewer2` = Qwen3.8 Max 0902 (`openrouter/qwen/qwen3.8-max-0902`) only after a clean calibration (§1b) | a NEW session started by H (`/new`, then Tab to `reviewer`); never inside A's session | read, rerun cheap commands, PASS/FAIL per criterion, write only `*checker_review*.md` | edit any other file; run `git checkout/reset/commit` |
| U | utility | DeepSeek Flash (`deepseek/deepseek-flash`) | OpenCode | mechanical edits, formatting | produce verdicts or summaries used for decisions |

Anthropic, OpenAI and Google models are not served in mainland China; do not route them through relays. Never use any
`deepseek/...` model (any version, incl. Flash) as R. `reviewer2` reviews only the three critical files (tracker,
metrics, checker), the G5/statistics code and any L4 correction; it never replaces `reviewer`. Model ids and prices were
read from openrouter.ai on 2026-10-13; confirm with OpenCode `/models` before use. Never use a moving alias
(`~z-ai/glm-latest`, `~moonshotai/kimi-latest`): the reviewer id must be a fixed slug, recorded in every review.

Three rules for every AI step:
1. **Author != reviewer.** A writes, R and L review.
2. **Evidence before summary.** Reviewers read the prompt, `git diff`, raw outputs under `reports/raw/<phase>/`, and
   only then the author's report. Past summaries contained a false "MISSING", a wrong "nothing changed" (TF32) and a
   wrong throughput metric.
3. **Fresh context.** Switching the model inside A's session is NOT a review: the model inherits A's whole context and
   A's edit rights (this happened on 2026-10-13: every "review" turn was `Coder · GLM 5.3`). A valid review shows the
   header `Reviewer · <model>` in a session that contains nothing but the review prompt.
4. **H saves every prompt** to `reports/raw/<phase>/prompt.txt` before A starts; A prints its sha256. Without the file
   a reviewer has to reconstruct the spec from A's own transcript, which biases the review toward A.

## 1. Files this runbook relies on

| file | purpose | installed by |
|---|---|---|
| `contract/paper_contract.yaml` (v1.8, then v1.9) | single source of truth | H |
| `.opencode/skills/lgc-preflight/SKILL.md` | audit rules for A and R | H (from this kit) |
| `.opencode/skills/{lgc-baselines,lgc-reviewer-eval,lgc-paper-contract}/SKILL.md` | existing | already installed |
| `tools/eval/human_preflight.py` | read-only report for H (written independently of A) | H (from this kit) |
| `tools/eval/check_contract.py` + `tools/eval/tests_contract/test_contract.py`, `tools/eval/g5_check.py` + `tools/eval/tests_g5/` | contract checker, G5 tool | exist (C1, B4c, B4e) |
| `reports/HUMAN_SIGNOFF_B5.md` | the go/no-go record | H fills the template from this kit |
| `docs/PREFLIGHT_RUNBOOK.md` | this file | H |
| `docs/LESSONS.md` | one ledger of lessons; `active` rows are read at every phase start (skill §7) | H (from this kit); rows added by H |

Install (AutoDL):
```bash
cd /root/autodl-tmp/LGC_frozen
unzip -o /path/to/lgc_preflight_kit_v1.3.zip -d .
cp -r .opencode/skills/lgc-preflight /root/autodl-tmp/Latent-Geometric-Chords/.opencode/skills/   # where OpenCode reads skills
python tools/eval/human_preflight.py --help
git add docs/PREFLIGHT_RUNBOOK.md docs/LESSONS.md tools/eval/human_preflight.py reports/HUMAN_SIGNOFF_B5_TEMPLATE.md .opencode/skills/lgc-preflight calib/
git commit -m "preflight kit v1.3 (runbook, LESSONS ledger, lgc-preflight skill with STOP bundle, human_preflight §7, sign-off template, calibration workspace)"
```

Reviewer configuration (once). Merge into the `opencode.json` that OpenCode loads (the one in the directory where you
start `opencode`; keys only via `/connect`, never in this file, `.env` or git):
```json
{
  "agent": {
    "reviewer": {
      "description": "Independent reviewer, skill lgc-preflight section 6. Fresh session only. Never edits code or data.",
      "mode": "primary",
      "model": "openrouter/z-ai/glm-5.3",
      "permission": {
        "edit": { "*": "deny", "*checker_review*.md": "allow" },
        "webfetch": "deny",
        "bash": {
          "*": "ask",
          "git diff*": "allow", "git log*": "allow", "git show*": "allow", "git rev-parse*": "allow", "git status*": "allow",
          "ls*": "allow", "cat *": "allow", "head *": "allow", "tail *": "allow", "sed -n *": "allow",
          "grep *": "allow", "rg *": "allow", "wc *": "allow", "sha256sum *": "allow", "nvidia-smi*": "allow",
          "rm *": "deny", "git push*": "deny", "git reset*": "deny", "git checkout*": "deny", "git commit*": "deny",
          "git add*": "deny", "git stash*": "deny"
        }
      }
    },
    "reviewer2": {
      "description": "Second reviewer: tracker, metrics, checker, G5 and L4 corrections only. Fresh session only.",
      "mode": "primary",
      "model": "openrouter/qwen/qwen3.8-max-0902",
      "permission": { "edit": { "*": "deny", "*checker_review*.md": "allow" }, "webfetch": "deny", "bash": "ask" }
    }
  },
  "provider": {
    "openrouter": {
      "models": {
        "z-ai/glm-5.3": { "options": { "provider": { "order": ["<first-party provider name>"], "allow_fallbacks": false } } },
        "qwen/qwen3.8-max-0902": { "options": { "provider": { "order": ["<first-party provider name>"], "allow_fallbacks": false } } }
      }
    }
  }
}
```
How to run a review: `/new` (new session) -> press Tab until the agent shown is `reviewer` -> paste the review prompt
(§4). Check the first reply header: it must read `Reviewer · GLM 5.3`. `Coder · ...` means the review is invalid.
`<first-party provider name>`: copy it from the "Providers" tab of the model page on openrouter.ai; UNVERIFIED spelling,
do not guess. A wrong name makes every request fail at once (an empty reply in 1-4 s, as Qwen and Kimi gave on
2026-10-13); read the error in the OpenCode log directory (usually `~/.local/share/opencode/log/`). In OpenRouter privacy
settings, exclude providers that may train on inputs. Permission rules: the last matching rule wins, so `"*"` comes first.
Known tool side effect: `tools/eval/check_contract.py` rewrites the tracked `reports/contract_status.md` (fixed in
B4e2 T7); until then a reviewer runs it with `--report /tmp/...` and H restores the file.

### 1b. Reviewer calibration (optional; only before adding `reviewer2`)
Status 2026-10-13: GLM 5.3 reviewed B4e twice (accurate to the printed digits, own commands, own deviations listed) and
B4d once; the B4d run is not a clean calibration (same session as the author, later decisions D12/D13 visible, spec not
supplied). Qwen3.8 Max and Kimi K3 produced no review (empty replies; Qwen then reused GLM's file). GLM 5.3 is adopted as
`reviewer`; L (Claude) stays the second reviewer. Run the clean calibration below only if you want `reviewer2`.
1. H: `bash calib/make_calib_ws.sh` (creates `/root/autodl-tmp/calib_b4d` at 88a7443 and an isolated workspace
   `/root/autodl-tmp/calib_ws` with the skill, the B4d transcript and the reconstructed spec; it has no access path to the
   post-B4d state). Then `cd /root/autodl-tmp/calib_ws && opencode`.
2. Per candidate model: `/new`, Tab to the calibration agent of that model (`cal_qwen`, `cal_kimi`, `cal_glm`), paste:
```
Calibration review, review mode (skill lgc-preflight section 6). Repository under review: /root/autodl-tmp/calib_b4d
(the B4d work is commit 88a7443, parent 4551440). Spec: spec_B4d.txt in this directory (reconstructed; treat it as the
spec). Author transcript: B4d_transcript.md (the author's own reasoning; evidence of what was done, not of what was
asked). Read the spec, then git -C /root/autodl-tmp/calib_b4d diff 4551440..88a7443, then the files, then
reports/B4d_preflight.md in that repository. Do not read /root/autodl-tmp/LGC_frozen or
/root/autodl-tmp/Latent-Geometric-Chords. For each T-item: PASS/FAIL with your own evidence (file:line or command
output). Then list (1) every place where the implementation does not meet the spec, (2) every claim in the report that
the evidence does not support, (3) every methodological choice that would make the baseline comparison unfair.
Write only out/checker_review_calib_<your model>.md.
```
3. H: `grep -n "LGC_frozen\|Latent-Geometric-Chords" <exported session>`; any hit outside the prompt text invalidates
   the run. Score with the answer key kept in the Claude Project (not on AutoDL), record the result in
   `reports/reviewer_calibration.md` (model slug, provider, date, score), then
   `git -C /root/autodl-tmp/LGC_frozen worktree remove /root/autodl-tmp/calib_b4d`.

## 2. Stage map

| stage | covers | phase / prompt | who | GPU | gate |
|---|---|---|---|---|---|
| S0 environment lock | study GPU, backend flags, B0' determinism, final smoke set on one GPU, row-split invariance | **B4e2** (§4) | A runs, R reviews, H signs §0-§1 | ~1-1.5 h | B0' IDENTICAL, combined smoke gate exit 0, FLAG lines 0 |
| S1 protocol audit (L1) | single loader, no second resize, normalisation constants, x0 identity, target pairs | **B4f** T1-T3 | A, R, H spot-checks | minutes | x0 mismatches 0, pairs valid |
| S2 hyper-parameter audit (L2/L3) | every argument vs official default `file:line` | **B4f** T4 | A builds table, R re-reads official files, H reads every DIFFERS row | 0 | every difference has an L3 reason |
| S3 oracle and query audit | all victim call sites through the tracker, early stops, eval-only calls, budget binding | **B4f** T5 | A, R | 0 | no bypass, queries == budget for baselines |
| S4 domain and clipping | attack space, bounds, clamp, to_pixel, L2 ratio | **B4f** T6 | A, R | minutes | raw-net diff 0, ratio in range |
| S5 known-answer test | analytic lower bound on a linear victim through the real loader, tracker and stores | **B4g** T1 | A, R, H reads table | ~15 min | checks (a)-(e) PASS |
| S6 invariance | determinism, precision flags, CGBA fixed vs d8only, row split, resume | B4e T1(d), B4e2 T1/T4/T5 | A, R | in B4e2 | IDENTICAL |
| S7 smoke + checker | artefacts of every cell | B4e2 T2-T3 | A, H runs `human_preflight.py` | in B4e2 | checker exit 0, FLAG lines 0 |
| S8 pipeline canary | metrics -> tables -> figures on smoke data | **B4g** T2 | A, H opens every output | ~10 min | every E1-E8 output produced |
| S9 sign-off | contract v1.9, amendment note, `HUMAN_SIGNOFF_B5.md` | H (+ L drafts) | H | 0 | signed, committed, tagged |
| S10 in-run gates | partial checker every 10 images, G5 at 30, daily human report | B5 prompts | A runs gates, H decides at each STOP | B5 | per D13/D14 |

### 2b. Fast path (default from v1.3)
B4e2, B4f and B4g run back to back in ONE GPU session. H saves the three prompts first. A starts the next phase only if
every gate of the previous one passed (B4e2: B0' IDENTICAL, combined smoke gate exit 0, human_preflight FLAG lines 0;
B4f: x0 mismatches 0, pairs valid, every DIFFERS row has an L3 reason, raw-net diff 0, ratio in range; B4g: KAT (a)-(d)
PASS, every canary output produced). Any failed gate -> STOP bundle (skill §7) and wait. At the end: one review in a new
session (agent `reviewer`) covering all three phases, one L review in the Claude chat (R and L verdicts side by side,
disagreements listed for H), one H sign-off. H's decisions are concentrated at six points: contract changes; B5 go/no-go;
every STOP except E-INFRA (incl. G5 FAIL); result-panel verdicts; manuscript claim wording; submission.

## 3. Per-stage instructions

### S0 environment lock (B4e2)
- H: decide the study GPU (contract D16) and whether the resume fix is approved (D18); fill the two variables at the top
  of prompt B4e2; save it as `reports/raw/B4e2/prompt.txt`.
- A: run prompt B4e2 (§4).
- H, after A reports:
  1. `nvidia-smi` and `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`.
  2. `python tools/eval/human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out reports/human_preflight_b4e2`
     and read section 1: one GPU name, one torch, the same backend flags in every root, `ADV_STORE_BUDGETS unset`,
     FLAG lines 0. Open the grid PNGs.
  3. Read the T1 (B0') and T4 (row split) tables yourself: IDENTICAL.
- R: new session, agent `reviewer`, review prompt (§4) with phase B4e2.
- L: paste `reports/B4e2_close.md`, the review and `reports/human_preflight_b4e2.md` into the Claude chat.

### S1-S4 protocol, hyper-parameter, oracle and domain audits (B4f)
- A: run prompt B4f (§4).
- H spot-checks (about 45 min):
  1. x0: section 2 of `human_preflight` = 0 mismatches. Open one grid PNG; pick one row, open the original JPEG
     (`splits/.../<arch>_<mode>.csv` -> `source_relpath`) and confirm it is the same scene, centre-cropped.
  2. Hyper-parameters: read every row marked DIFFERS in `reports/B4f_hparam_audit.md`. For two random rows open the
     official file at the cited line yourself, e.g. `sed -n 25,40p /root/autodl-tmp/baselines/CGBA/Non_targeted_attack.py`.
  3. Oracle: read the call-site table; for one method grep the official file yourself
     (`grep -n "model\|predict" <official file>`) and check every hit appears in the table.
  4. Domain: the raw-net table must show 0.0; the pixel/normalised L2 ratio must lie in [4.37, 4.46].
- R: new session, agent `reviewer`, review prompt (§4) with phase B4f.

### S5 + S8 known-answer test and pipeline canary (B4g)
- A: run prompt B4g (§4).
- H: read the KAT table (every (a)-(e) PASS; look at the convergence ratio column, it is descriptive). Open every file
  in `generated/smoke_canary/` (tables, figures, grid): ours first, every method and cell present, n printed, no NaN,
  log-x axes with ticks 1-2-5, one amplification per grid row. Do not interpret any number (n = 3).

### S9 sign-off
1. L drafts contract v1.9 (D15 precision finding, D16 study GPU, D17 KAT + canary as gates, decisions on E18,
   main-protocol ablation and same-prior control) as an apply script; H runs it dry, reads the diff, applies it, runs
   `python tools/eval/tests_contract/test_contract.py`, commits.
2. H writes the amendment-2 deviation notes (GPU, D11-D17) and commits them before any n = 200 run.
3. H fills `reports/HUMAN_SIGNOFF_B5.md` from the template; every line needs evidence (file + sha or command output).
4. `git tag b5-go-<date>`; only then the B5 prompt.

### S10 during B5
- A (in the B5 prompt): checker `--stage partial` every 10 images per cell; `tools/eval/g5_check.py` at 30 images for
  TtBA / CGBA / CGBA_H cells; STOP the cell on FAIL.
- H once per day: `python tools/eval/human_preflight.py --roots Results/b5_* --out reports/human_preflight_b5_<date>`;
  read section 1 (same GPU and flags), 3 (budget), 5 (stalls), open the grids. Never look at ours-vs-baseline tables.
- On any STOP: A writes the STOP bundle (skill §7: error code, signature, evidence, env, scope, candidates, proposed
  lesson); only E-INFRA is handled without H; otherwise R reviews, L proposes, H decides (fix with a commit and re-run
  the whole cell, or keep); never re-run single images selectively.
- H once per day also reads section 7 of the human_preflight report (degeneracy: identical methods, constant outputs,
  found rate 0, one x0 for many images).

## 4. Prompts

### Prompt B4e2 — close B4e on the study GPU (S0, S6, S7)
```
Load skills lgc-preflight, lgc-baselines, lgc-paper-contract. Phase B4e2 (close B4e on the study GPU). GPU ON.
Worktree /root/autodl-tmp/LGC_frozen, contract v1.8 (print its sha256). The human saved this prompt as
reports/raw/B4e2/prompt.txt; print its sha256. EXECUTE step by step. Do not start B5.
Never edit lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh, official clones, the contract or prereg files.
Allowed: adapters/run_cgba.py (T6 only), ttba_github/ttba_common.py (T5 only), tools/eval/check_contract.py and
tools/eval/tests_contract/* (T7 only), adapters/diag/*, reports/*. Tee the stdout/stderr of every GPU command to
reports/raw/B4e2/<step>.log.

STUDY_GPU = "<exact torch.cuda.get_device_name(0) string>"
RESUME_FIX_APPROVED = <yes|no>
export OMP_NUM_THREADS=8 for every command of this phase (it becomes the B5 value).

T0 Environment: nvidia-smi (name, driver, CUDA), torch.cuda.get_device_name(0), get_device_capability(0),
 multi_processor_count, torch.__version__, lscpu model name, OMP_NUM_THREADS. Device name != STUDY_GPU: STOP before
 any GPU run.
T1 Frozen engine B0' on STUDY_GPU: ttba_github/run_frame_b1.sh, resnet50 NT rows 1-3, three out_dirs:
 (A) the 6 registered budgets without --save_adv; (A2) A repeated; (B) the 8 contract budgets with --save_adv.
 A == A2 bitwise (found, state_query, pix_l2 at every budget); A == B at the 6 registered budgets. Any difference: STOP.
 If STUDY_GPU is the RTX 5090, also compare B with Results/b4_smoke_egc_resnet50 (IDENTICAL or the first diverging
 image/budget; information only).
T2 Final smoke set on STUDY_GPU (rows 1-3, D12 flags) into fresh roots Results/b4e2_smoke_<method>_<arch>:
 - LGC_RC_EGC: all 12 cells (main via run_frame_b1.sh, E10/E11 via run_frame_b1_generic.py; one out_dir per (arch, mode)).
 - Baselines: if every env file under Results/b4e_smoke_* records STUDY_GPU, re-run only the 12 TtBA cells (their env
   files lack the D12 precision block); otherwise all 30 baseline cells (TtBA, CGBA/CGBA_H --cgba_variant fixed,
   ARS_OPT; E10/E11 TtBA, CGBA/CGBA_H).
 Write reports/B4e2_smoke_roots.txt: the exact roots of the final smoke reference (EGC + every baseline cell, one GPU).
T3 Gates on exactly that list: check_contract.py --stage smoke --n 3 --report /tmp/b4e2_gate.md -> exit 0 (one GPU name
 in I10); python tools/eval/human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out
 reports/human_preflight_b4e2 -> FLAG lines = 0; stall 0, nonfinite_queries 0, errors 0 in every cell. Paste section 1
 and the FLAG count.
T4 Row-split invariance (needed to split a cell across GPUs in B5): for LGC_RC_EGC resnet50 NT and TtBA resnet50 NT, run
 rows 2-3 alone into a throwaway dir; compare with rows 2-3 of T2 (found, state_query, pix_l2, npz delta). Report
 IDENTICAL or the first difference, and quote how each runner seeds per image (file:line). Do not fix anything.
T5 (only if RESUME_FIX_APPROVED = yes) ttba_github/ttba_common.py:83 done_images: an image with ANY row in runs_*.csv is
 done, whatever its status (contract D6: failures are never re-run selectively). Sandbox unit test (ok, budget and
 error rows skipped; a missing image runs). Fresh-run invariance: EGC resnet50 NT row 1 and TtBA resnet50 NT row 1 into
 new dirs -> IDENTICAL to T2.
T6 CGBA clip-event counter (observation only; adapters/run_cgba.py fixed override): per image, count the calls in which
 the acos argument at proposed_attack.py:182 lies outside [-1, 1] before the clamp; write clip_events_182 to runs_*.csv
 for CGBA/CGBA_H. Update adapters/diag/diag_b4e_t1_clip_source.py so it asserts that the clamp is the only
 value-changing edit. Re-run CGBA resnet50 NT rows 1-3 into a throwaway dir: numerics IDENTICAL to T2; print the counts.
T7 tools/eval/check_contract.py rewrites the tracked reports/contract_status.md even when --report points elsewhere.
 Write the status file next to --report (<report stem>_status.md) unless --status_out is given. tests_contract all
 PASS (add one test for this), selftest_evalkit 20/20.
T8 Cost on STUDY_GPU (s/img from T2; B4e for cells not re-run): GPU-h per cell = 200 x s/img / 3600; totals main,
 E10/E11, E17; makespan for N = 4, 8, 12 GPUs, longest cell first, one process per GPU (row splitting only if T4 is
 IDENTICAL).
Write reports/B4e2_close.md (skill lgc-preflight §5, with "Deviations from the prompt"); commit code, report and
reports/raw/B4e2/*. STOP.
```

### Prompt B4f — protocol, hyper-parameter, oracle and domain audits (S1-S4)
```
Load skills lgc-preflight, lgc-baselines, lgc-reviewer-eval and lgc-paper-contract. Phase B4f (audit). Mostly CPU;
GPU only where stated. Worktree /root/autodl-tmp/LGC_frozen, contract v1.8 (print sha256). Study GPU: the one in
reports/B4e2_close.md (print and compare; STOP if different). The human saved this prompt as reports/raw/B4f/prompt.txt;
print its sha256.
EXECUTE step by step. Do not start B5. Never edit lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh, official
clones, the contract or prereg files. Allowed: adapters/diag/*, adapters/_common.py (env recording only, T7),
reports/*. Raw outputs go to reports/raw/B4f/.

T1 Loader audit (L1). For each dataset (ImageNet ttba_github, Places365, RobustBench pool) quote the single function
 that turns a file into x0 (file:line): decoder, resize, crop, ToTensor, dtype. Then grep every adapter, wrapper,
 runner and official clone path that is executed for: Resize, resize, interpolate, CenterCrop, transforms, ToTensor,
 PIL.Image.open, read_image. Table: hit | file:line | executed for which method | before or after x0 is produced.
 Any resize/crop applied after x0 is produced = FAIL.
T2 Normalisation constants. For each of the 6 victims print mean/std used inside the victim wrapper (file:line) and the
 values from weights.transforms() / the Places365 or RobustBench source; must be equal.
T3 x0 identity and pairs. Run python tools/eval/human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out
 reports/raw/B4f/human_preflight (paste section 2). For every split CSV (12 files) rows 1-200: re-verify on the study
 GPU with the D12 flags that the source is classified as source_label and (targeted) the target as target_label,
 target_label != source_label, no duplicate sources. Table per file: n, wrong source, wrong target, same-label, dups.
T4 Hyper-parameter audit (L2/L3). Write adapters/diag/hparam_audit.py that, for each method x mode, lists EVERY argument
 the adapter passes to the official code and every official default it relies on: name | adapter value | adapter
 file:line | official default value | official file:line | layer (L2 or L3) | DIFFERS? | L3 reason. Read the
 official values from the clones (not from memory, not from READMEs). Output reports/B4f_hparam_audit.md.
T5 Oracle and query audit. For each method: (a) every place the official code evaluates the victim (file:line), and
 proof that it reaches the victim only through the tracker; (b) every termination / early-stop / iteration-cap
 condition (file:line) and how the adapter neutralises it (L3) or why it cannot fire before the budget; (c) every
 evaluation-only victim call made by the adapter (file:line) and proof the attack cannot see its result; (d) from the
 B4e smoke runs: per cell, images with queries == 10000, < 10000 (with status), > 10000; (e) official counter vs
 tracker count per cell with the documented offsets.
T6 Domain and clipping. Per method: attack space (pixel or normalised), lower/upper bounds used (file:line), where
 clamping happens (file:line), to_pixel map; for normalised-space methods print d_n/d_p for 3 stored points (must lie
 in [1/sd_max, 1/sd_min] = [4.37, 4.46]); max|victim(x0) - raw_net((x0 - mu)/sd)| for all 6 victims (must be 0).
T7 Env recording. Every adapter env_*.json and the generic EGC runner's env_frame.txt must record
 backend.cuda_matmul_allow_tf32, backend.cudnn_allow_tf32, backend.cudnn_deterministic, backend.cudnn_benchmark,
 backend.float32_matmul_precision read from torch at run time (skill lgc-preflight §4). For run_frame_b1.sh (frozen)
 show where config_single.json records "tf32=... deterministic=..." instead. Re-run human_preflight section 1.

Write reports/B4f_audit.md (skill lgc-preflight §5), commit diag scripts + reports. STOP.
```

### Prompt B4g — known-answer test and pipeline canary (S5, S8)
```
Load skills lgc-preflight, lgc-baselines, lgc-reviewer-eval and lgc-paper-contract. Phase B4g. GPU ON (study GPU,
STOP if different), D12 flags. Worktree /root/autodl-tmp/LGC_frozen, contract v1.8. The human saved this prompt as
reports/raw/B4g/prompt.txt; print its sha256. Do not start B5. Never edit lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh,
official clones, the contract or prereg files. Allowed: adapters/_common.py (new arch kat_linear only),
ttba_github/run_frame_b1_generic.py (--arch kat_linear only), adapters/diag/*, generated/smoke_canary/*.

T1 Known-answer test. adapters/diag/kat_linear.py builds a linear victim and runs EVERY method (LGC_RC_EGC via
 run_frame_b1_generic.py; TtBA, CGBA/CGBA_H (fixed), ARS_OPT via their adapters) through the normal loader, tracker
 and adv store, budget 10000, contract checkpoints.
 Victim: input x in [0,1]^(3x224x224); z = (x - mu)/sd with ImageNet mu/sd inside; logits = W vec(z) + b, K = 10.
 W_k = bilinear upsampling of a (3,14,14) N(0,1) array to (3,224,224) (seed 0), b ~ N(0,1) (seed 0). The pixel-space
 gradient of logit k is g_k = W_k / sd (per channel).
 Images: 256x256 RGB PNGs with values k/255, so Resize(256) is a no-op and CenterCrop(224) is exact; generate
 candidates 0.5 + 0.08*N(0,1) clipped to [0.3, 0.7] (seed i) and keep the first 5 whose analytic non-targeted
 distance d* = min_{j != y} (l_y - l_j) / ||g_y - g_j||_2 (computed on the loaded 224x224 x0) lies in [1, 10] and whose
 minimiser x0 - d* (g_y - g_j*)/||g_y - g_j*|| lies inside [0,1]. Targeted: target class t = the second-largest logit;
 target image = a candidate classified as t; LB_t = (l_y - l_t)/||g_y - g_t||_2 is a lower bound on any targeted L2.
 Write splits/kat/kat_linear_{non_targeted,targeted}.csv in the contract split format.
 Checks per (method, mode, image, checkpoint with found = 1), using the stored npz (delta is float16, u = 2^-11):
  (a) adversarial: margin m(x0 + delta) <= max_j ||g_y - g_j|| * u * ||delta|| (NT: m = l_y - max_{j != y} l_j;
      T: m = max_{j != t} l_j - l_t);
  (b) lower bound: ||delta|| >= d* (1 - u) (NT) or >= LB_t (1 - u) (T);
  (c) pix_l2 in per_checkpoint equals ||delta|| (relative error < 1e-3);
  (d) queries == 10000 for baselines, <= 10000 for ours; tracker count == forward count at the linear victim minus
      the listed evaluation-only calls;
  (e) descriptive: pix_l2@10000 / d* per method and image (no pass/fail).
 Table per method x mode: n checked, (a)-(d) PASS counts, median and max ratio (e). Any FAIL in (a)-(d): STOP.
T2 Pipeline canary on the roots in reports/B4e2_smoke_roots.txt (n = 3; outputs labelled SMOKE, never interpreted):
 tools/eval/eval_metrics_offline.py -> generated/smoke_canary/metrics.csv (every contract metric, one row per npz);
 tools/eval/check_contract.py --stage paper --n 3 --metrics_csv generated/smoke_canary/metrics.csv (report I13 and the
 I14 status list); tools/eval/make_tables_figures.py with the contract budgets -> generated/smoke_canary/ (E1-E8
 tables and figures); tools/eval/make_comparison_grid.py -> E7 grid. Table: element | output file | produced? |
 methods present | cells present | n printed. Override every evalkit default list with the contract values.

Write reports/B4g_kat_canary.md (skill lgc-preflight §5), commit. STOP.
```

### Review prompt for R (H: `/new`, Tab to `reviewer`, paste), after every A phase
```
Review phase <B4e2|B4f|B4g> in review mode (skill lgc-preflight §6). Worktree /root/autodl-tmp/LGC_frozen.
Base commit: <sha before the phase>.
Spec: reports/raw/<phase>/prompt.txt. Read the spec, then `git diff <base>..HEAD`, then reports/raw/<phase>/, then the
author's report. For each T-item and acceptance criterion: PASS/FAIL with your own evidence (rerun cheap commands).
List every claim in the author's report that the evidence does not support. Do not edit files; do not run
git checkout/reset/commit. Run tools/eval/check_contract.py only with --report /tmp/<name>.md.
Write your result to reports/raw/<phase>/checker_review.md.
```

### What to paste into the Claude chat (L)
1. The phase report `reports/<phase>.md`.
2. `reports/raw/<phase>/checker_review.md`.
3. The newest `reports/human_preflight_*.md` (and say whether you opened the PNGs).
4. The OpenCode session log if anything STOPPED.
L answers with: verified items (with numbers), problems ranked, the next prompt or a contract draft.

## 5. Decision rights (who signs what)

| decision | H | L | A | R |
|---|---|---|---|---|
| claims, pre-registration, amendments | decides | drafts | - | - |
| contract edits | applies | drafts script | never | - |
| baseline roster, exclusions, threat model | decides | advises | - | - |
| any baseline-code change beyond L3 (L4) | decides | root cause + proposal | implements after approval | reviews diff |
| study GPU, splits, victims | decides | advises | reports | - |
| GPU spend > 1 h | approves cost table | checks table | produces table | - |
| go/no-go for B5 | signs | reviews | - | - |
| G5 FAIL / any STOP | decides | proposes | evidence | review |
| paper numbers | traces each to CSV | checks | generates | checks |

## 6. Known failure modes this runbook guards against (from this project)
UnnormalizeVictim round trip flipped decisions (B3b) -> S4; CGBA count off-by-one (D8) and NaN acos (D11) -> S3, S6;
TF32 on baselines only (D12) -> S0, T7; GPU changed between sessions -> S0, S10; env overwrite between modes (B4b) ->
S0; agent marked an existing file MISSING (B4c) -> skill §2; naive G5 statistic (B4d) -> D13 tool; wrong throughput
metric (B4d) -> D14 weighted speedup.
