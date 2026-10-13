# PREFLIGHT RUNBOOK — from smoke to the first n = 200 run (LGC-EGC, TMM/PR)

Version 1.1, 2026-10-13 (v1.1: reviewer models reachable from AutoDL/China; reviewer calibration). Binding order of work before B5. The contract (`contract/paper_contract.yaml`) wins over this
file; this file wins over older plans. Principle: **fairness = identical task, not identical algorithm** (layers L1-L4 in
skill `lgc-preflight` §1).

## 0. Roles, models and where they run

| id | role | who / model | where | may | must not |
|---|---|---|---|---|---|
| H | decides, signs, spot-checks | you | AutoDL shell, Mac, this Claude Project chat | everything in §5 "human-only" | delegate a §5 decision |
| L | lead analyst: specs, prompts, review of reports + raw evidence, root cause, contract drafts | **Claude Opus 5.5** (this Project chat; API id `claude-opus-5-5` if you call it locally) | chat (you paste logs) | write prompts, review, propose | run code on AutoDL, edit the contract |
| A | author/executor | OpenCode agent `coder`, **DeepSeek V4 Pro** (`deepseek/deepseek-v4-pro` in `opencode.json`) | AutoDL worktree `/root/autodl-tmp/LGC_frozen` | code within the prompt's allowed list, run, report | judge its own work as final, edit the contract/engine/clones |
| R | independent reviewer | OpenCode subagent `@checker`, a lab different from A and reachable from AutoDL (China): **GLM 5.3** via OpenRouter (`openrouter/z-ai/glm-5.3`); second reviewer `@checker2` **Qwen3.8 Max 0902** (`openrouter/qwen/qwen3.8-max-0902`); alternative Kimi K3 (`openrouter/moonshotai/kimi-k3`). Final choice by the calibration in §1b | same session as A | read, rerun cheap commands, PASS/FAIL per criterion, write only `*checker_review*.md` | edit any other file |
| U | utility | DeepSeek Flash (`deepseek/deepseek-flash`) | OpenCode | mechanical edits, formatting | produce verdicts or summaries used for decisions |

Anthropic, OpenAI and Google models are not served in mainland China; do not route them through relays. Never use any
`deepseek/...` model (any version, incl. Flash) as R. `@checker2` reviews only the three critical files (tracker,
metrics, checker), the G5/statistics code and any L4 correction; it never replaces `@checker`. Model ids and prices were
read from openrouter.ai on 2026-10-13; confirm with OpenCode `/models` before use. Never use a moving alias
(`~z-ai/glm-latest`, `~moonshotai/kimi-latest`): the reviewer id must be a fixed slug, recorded in every review.

Two rules for every AI step:
1. **Author != reviewer.** A writes, R and L review.
2. **Evidence before summary.** Reviewers read the prompt, `git diff`, raw outputs under `reports/raw/<phase>/`, and
   only then the author's report. Past summaries contained a false "MISSING", a wrong "nothing changed" (TF32) and a
   wrong throughput metric.

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

Install (AutoDL):
```bash
cd /root/autodl-tmp/LGC_frozen
unzip -o /path/to/lgc_preflight_kit_v1.1.zip -d .
cp -r .opencode/skills/lgc-preflight /root/autodl-tmp/Latent-Geometric-Chords/.opencode/skills/   # where OpenCode reads skills
python tools/eval/human_preflight.py --help
git add docs/PREFLIGHT_RUNBOOK.md tools/eval/human_preflight.py reports/HUMAN_SIGNOFF_B5_TEMPLATE.md .opencode/skills/lgc-preflight
git commit -m "preflight kit v1.1 (runbook, lgc-preflight skill, human_preflight.py, sign-off template)"
```

Check the reviewer configuration once (the `checker` subagent already exists in `opencode.json`):
```bash
cd /root/autodl-tmp/Latent-Geometric-Chords
python - <<'EOF'
import json; c = json.load(open("opencode.json"))
for name in ("coder", "checker", "auditor"):
    a = c.get("agent", {}).get(name, {})
    print(name, "| model:", a.get("model", c.get("model")), "| mode:", a.get("mode"), "| tools:", a.get("tools"), "| permission:", a.get("permission"))
EOF
```
Required: `checker` model is NOT a `deepseek/...` id (author != reviewer lab), its mode is `subagent`, it can write only
its own review file, and its OpenRouter provider is pinned without fallbacks. Merge into `opencode.json` (keys only via
`/connect`, never in this file, `.env` or git):
```json
{
  "agent": {
    "checker": {
      "description": "Independent reviewer, skill lgc-preflight section 6. Never edits code or data.",
      "mode": "subagent",
      "model": "openrouter/z-ai/glm-5.3",
      "permission": {
        "edit": { "*": "deny", "*checker_review*.md": "allow" },
        "webfetch": "deny",
        "bash": {
          "*": "ask",
          "git diff*": "allow", "git log*": "allow", "git show*": "allow", "git rev-parse*": "allow",
          "ls*": "allow", "cat *": "allow", "head *": "allow", "tail *": "allow", "sed -n *": "allow",
          "grep *": "allow", "rg *": "allow", "wc *": "allow", "sha256sum *": "allow", "nvidia-smi*": "allow",
          "rm *": "deny", "git push*": "deny", "git reset*": "deny", "git checkout*": "deny", "git commit*": "deny"
        }
      }
    },
    "checker2": {
      "description": "Second independent reviewer for tracker, metrics, checker, G5 and L4 corrections only.",
      "mode": "subagent",
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
`<first-party provider name>`: copy it from the "Providers" tab of the model page on openrouter.ai (Z.ai for GLM, Alibaba
for Qwen); UNVERIFIED spelling, do not guess. Pinning keeps the weights/precision fixed across reviews and makes a refused
request visible instead of silently re-routed. In OpenRouter privacy settings, exclude providers that may train on
inputs (unpublished code and results). Permission rules: the last matching rule wins, so `"*"` comes first.
If OpenCode asks for `external_directory` access to `/root/autodl-tmp/LGC_frozen`, allow read for the checkers.

### 1b. Reviewer calibration (once, before B4f; about 1 h of H time, a few USD of API)
A reviewer is trusted only after it finds known defects. B4d (HEAD 88a7443) has three documented defects that the
author's report did not flag: (i) the G5 tool did not meet its spec (unpaired, PASS on mean AND median ratio, no CI,
tracker-best for TtBA, no tests); (ii) the concurrency throughput was wall-clock on unequal workloads instead of a
per-process weighted speedup; (iii) baselines ran with cuDNN TF32 on while ours ran FP32, dismissed as harmless.
1. H: `git -C /root/autodl-tmp/LGC_frozen worktree add /root/autodl-tmp/calib_b4d 88a7443` (read-only use).
2. H: in a fresh OpenCode session, for each candidate (`@checker` = GLM 5.3, `@checker2` = Qwen3.8 Max, optionally
   Kimi K3 by switching the `checker` model), send the prompt below. Do not show it the Claude review of B4d.
3. Score each model: defects found (0-3, with correct evidence), false claims (a FAIL whose evidence is wrong), and cost
   (OpenRouter activity page). Keep the model with the most defects found and no false claims as `@checker`; the
   next best as `@checker2`. Record the result in `reports/reviewer_calibration.md` (model slug, provider, date, score).
4. H: `git -C /root/autodl-tmp/LGC_frozen worktree remove /root/autodl-tmp/calib_b4d`.
```
@checker Calibration review, review mode (skill lgc-preflight section 6). Repository: /root/autodl-tmp/calib_b4d
(commit 88a7443); use git -C /root/autodl-tmp/calib_b4d for every git command and read files only there.
Base commit: <sha of the commit before B4d>. Spec: reports/raw/B4d/prompt.txt if present, otherwise the B4d prompt
pasted below. Read the spec, then the diff base..88a7443, then the raw outputs, then the B4d report.
For each T-item: PASS/FAIL with your own evidence (file:line or command output). Then list (1) every place where the
implementation does not meet the spec, (2) every claim in the report that the evidence does not support, (3) every
methodological choice that would make the baseline comparison unfair. Do not edit files except your output:
/root/autodl-tmp/LGC_frozen/reports/raw/calib/checker_review_<model>.md
<paste the B4d prompt here if prompt.txt is absent>
```

## 2. Stage map

| stage | covers | phase / prompt | who | GPU | gate |
|---|---|---|---|---|---|
| S0 environment lock | study GPU, backend flags, B0' determinism and observation-only checkpoints | B4e (running) | A runs, R reviews, H signs §1 | ~2 h | T0, T0a, flags identical |
| S1 protocol audit (L1) | single loader, no second resize, normalisation constants, x0 identity, target pairs | **B4f** T1-T3 | A, R, H spot-checks | minutes | x0 mismatches 0, pairs valid |
| S2 hyper-parameter audit (L2/L3) | every argument vs official default `file:line` | **B4f** T4 | A builds table, R re-reads official files, H reads every DIFFERS row | 0 | every difference has an L3 reason |
| S3 oracle and query audit | all victim call sites through the tracker, early stops, eval-only calls, budget binding | **B4f** T5 | A, R | 0 | no bypass, queries == budget for baselines |
| S4 domain and clipping | attack space, bounds, clamp, to_pixel, L2 ratio | **B4f** T6 | A, R | minutes | raw-net diff 0, ratio in range |
| S5 known-answer test | analytic lower bound on a linear victim through the real loader, tracker and stores | **B4g** T1 | A, R, H reads table | ~15 min | checks (a)-(e) PASS |
| S6 invariance | determinism, concurrency, precision flags, CGBA fixed vs d8only | B4e T0a/T1(d)/T4 | A, R | in B4e | IDENTICAL |
| S7 smoke + checker | artefacts of every cell | B4e T2 | A, H runs `human_preflight.py` | in B4e | checker exit 0, FLAG lines 0 or explained |
| S8 pipeline canary | metrics -> tables -> figures on smoke data | **B4g** T2 | A, H opens every output | ~10 min | every E1-E8 output produced |
| S9 sign-off | contract v1.9, amendment note, `HUMAN_SIGNOFF_B5.md` | H (+ L drafts) | H | 0 | signed, committed, tagged |
| S10 in-run gates | partial checker every 10 images, G5 at 30, daily human report | B5 prompts | A runs gates, H decides at each STOP | B5 | per D13/D14 |

## 3. Per-stage instructions

### S0 environment lock (B4e, in progress)
- A: run the B4e resume prompt (`claude/b4e_gpu_decision_2026-10-13.md`).
- H, after A reports:
  1. `nvidia-smi` and `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`.
  2. `python tools/eval/human_preflight.py --roots Results/b4e_smoke_* --out reports/human_preflight_b4e`
     and read section 1: one GPU name, one torch, the same backend flags in every root, `ADV_STORE_BUDGETS unset`.
  3. Read the T0a table yourself: A vs A2 and A vs B must say IDENTICAL.
- R: `@checker Review phase B4e per skill lgc-preflight §6; base commit <sha before B4e>; criteria = T0-T5 in reports/raw/B4e/prompt.txt.`
- L: paste the B4e report, the @checker output and `reports/human_preflight_b4e.md` into the Claude chat.

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
- R: same command as S0 with phase B4f.

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
- On any STOP: A writes the evidence; R reviews; L proposes; H decides (fix with a commit and re-run the whole cell,
  or keep); never re-run single images selectively.

## 4. Prompts

### Prompt B4f — protocol, hyper-parameter, oracle and domain audits (S1-S4)
```
Load skills lgc-preflight, lgc-baselines, lgc-reviewer-eval and lgc-paper-contract. Phase B4f (audit). Mostly CPU;
GPU only where stated. Worktree /root/autodl-tmp/LGC_frozen, contract v1.8 (print sha256). Study GPU: the one in
reports/B4e_preflight.md (print and compare; STOP if different). Save this prompt verbatim to reports/raw/B4f/prompt.txt.
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
T3 x0 identity and pairs. Run python tools/eval/human_preflight.py --roots Results/b4e_smoke_* --out
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
STOP if different), D12 flags. Worktree /root/autodl-tmp/LGC_frozen, contract v1.8. Save this prompt verbatim to
reports/raw/B4g/prompt.txt. Do not start B5. Never edit lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh,
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
T2 Pipeline canary on the B4e smoke roots (n = 3; outputs labelled SMOKE, never interpreted):
 tools/eval/eval_metrics_offline.py -> generated/smoke_canary/metrics.csv (every contract metric, one row per npz);
 tools/eval/check_contract.py --stage paper --n 3 --metrics_csv generated/smoke_canary/metrics.csv (report I13 and the
 I14 status list); tools/eval/make_tables_figures.py with the contract budgets -> generated/smoke_canary/ (E1-E8
 tables and figures); tools/eval/make_comparison_grid.py -> E7 grid. Table: element | output file | produced? |
 methods present | cells present | n printed. Override every evalkit default list with the contract values.

Write reports/B4g_kat_canary.md (skill lgc-preflight §5), commit. STOP.
```

### Review prompt for `@checker` (R), after every A phase
```
@checker Review phase <B4e|B4f|B4g> in review mode (skill lgc-preflight §6). Base commit: <sha before the phase>.
Spec: reports/raw/<phase>/prompt.txt. Read the spec, then `git diff <base>..HEAD`, then reports/raw/<phase>/, then the
author's report. For each T-item and acceptance criterion: PASS/FAIL with your own evidence (rerun cheap commands).
List every claim in the author's report that the evidence does not support. Do not edit files.
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
