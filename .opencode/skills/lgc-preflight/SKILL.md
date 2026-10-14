---
name: lgc-preflight
description: Pre-GPU audit rules for LGC-EGC baseline evaluation. Load for every audit, known-answer, canary or review phase (B4f, B4g, B5 gates) together with lgc-baselines and lgc-paper-contract.
---

# Skill: lgc-preflight

Purpose: no GPU hour is spent on a run that a reviewer could invalidate. You produce EVIDENCE; the human and a
second model decide. `contract/paper_contract.yaml` wins over this skill; `docs/PREFLIGHT_RUNBOOK.md` gives the order.

## 1. Fairness = identical task, not identical algorithm
Every change you make or propose must be classified first:

| layer | what | rule |
|---|---|---|
| L1 task/protocol | image rows and pairs; the x0 tensor; victim weights and backend flags (D12); decision-only oracle; element-wise query counting; budget 10000 and the contract checkpoints; pixel domain [0,1]; success definitions; the one offline metric code; GPU type and torch build; failure policy | identical for all methods, implemented in shared code (`adapters/_common.py`, tracker, checker), never per adapter |
| L2 method internals | step sizes, probe counts, sigma, tolerances, dimension reduction, schedules, non-targeted init, internal batch size | official ImageNet/224 default with `file:line`; NEVER copied from our method, NEVER tuned, NEVER "harmonised" |
| L3 interface | model object, data feeding, iteration caps raised so the budget binds, early stops disabled, CSV target image as targeted start | allowed; each listed in `env_<method>.json` |
| L4 correction | bug fixes in official code (D8, D11) | only with a user contract decision; never on your own |

Classification test: (1) changes the problem, the cost or the success measure -> L1; (2) the baseline's authors
would object -> L2, keep the default; (3) needed only to run inside the harness -> L3. If unsure: STOP and ask.

## 2. Evidence rules (violations make the report invalid)
1. Every claim cites a command output, a `file:line`, or a file path + sha256. "Verified" without evidence = not verified.
2. Never write MISSING / UNVERIFIED before searching the worktree AND `/root/autodl-tmp/Latent-Geometric-Chords`
   (`ls`, `git ls-files`, `find -name`). Print the search command.
3. Never conclude "nothing to change" for a check that the prompt asked you to perform: report the measured values
   and let the human decide.
4. Report metrics exactly as specified in the prompt (formula, n, statistic). If you think the spec is wrong, compute
   the specified one AND explain the alternative separately.
5. Paste raw numbers (tables), not adjectives. Keep raw logs under `reports/raw/<phase>/`.
6. Report every deviation from the prompt in a section "Deviations from the prompt" (empty if none).

## 3. Hard STOP conditions (stop, write the report, ask)
- GPU name differs from the study GPU in the contract / prompt; backend flags differ between runners.
- Any per-cell `queries > budget`; a baseline with `queries < budget` without a documented reason.
- x0 hash differs between methods for the same (arch, image).
- pix_l2 recomputed from the npz differs by >= 1e-3 relative; an L2 below the analytic lower bound (known-answer test).
- A non-finite query (`nonfinite_queries > 0`) with the fixed variants.
- Any need to edit lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh, an official clone, the contract or a prereg file.
- Any L2 parameter that differs from the official default without an L3 reason.

## 4. Shared-code requirements (L1 in one place)
- `adapters/_common.py` sets the backend flags before any model is built and writes them into every env file as
  `backend.cuda_matmul_allow_tf32`, `backend.cudnn_allow_tf32`, `backend.cudnn_deterministic`, `backend.cudnn_benchmark`,
  `backend.float32_matmul_precision`.
- One image loader per dataset; no resize/interpolate anywhere after it (wrappers only normalise).
- The tracker is the only path from an attack to the victim; evaluation-only calls are uncounted and invisible.

## 5. Report format (`reports/<phase>.md`)
1. Scope and the exact prompt hash (sha256 of the prompt text you received, if available).
2. Environment block: GPU name, capability, SM count, torch, driver, backend flags, HEAD, contract sha256.
3. One section per task: commands run, raw tables, verdict PASS/FAIL/INFO per acceptance criterion.
4. Deviations from the prompt.
5. sha256 of every created or changed file; GPU minutes.
6. Five-line plain-English summary. STOP.

## 6. Review mode (agent `reviewer`, always in a new session)
You did not write the code. If this session already contains the author's work (earlier turns by another agent or
model), say so in the first line and stop: that is not an independent review. Read, in this order: the prompt/spec
(`reports/raw/<phase>/prompt.txt`), `git diff <base>..HEAD`, the raw outputs under `reports/raw/<phase>/`, and only then
the author's report. For each acceptance criterion give PASS/FAIL with your own evidence (rerun cheap commands
yourself). List every claim in the author's report that the evidence does not support. Never edit files except your
review file; never run git checkout/reset/commit/add; run tools/eval/check_contract.py only with --report /tmp/<name>.md
(it can rewrite reports/contract_status.md). If the spec file is missing, write that as the first finding.

## 7. Lessons ledger and STOP bundle
- At the start of every phase read `docs/LESSONS.md` and list, in the report's first section, the ids of the `active`
  rows that apply to this phase. Never contradict an active row; if a prompt seems to, STOP and ask.
- On any STOP write `reports/stop/<phase>_<cell>_<YYYYmmdd-HHMM>.md` before anything else, with exactly these fields:
  1. `code`: E-ENV (GPU, driver, flags) | E-INFRA (OOM, disk, killed process, network; no row written) | E-BUDGET |
     E-NAN | E-STALL | E-X0 | E-DEGEN (human_preflight §7) | E-G5 | E-SPEC (prompt unclear or needs a file outside
     the allowed list) | E-OTHER;
  2. `signature`: the exact failing line or check output;
  3. `evidence`: file:line or command, and the last 50 log lines;
  4. `env`: GPU name, driver, torch, backend flags, HEAD, contract sha256;
  5. `scope`: cells and images affected; rows already completed;
  6. `candidates`: possible causes with what would distinguish them (no fix applied);
  7. `proposed lesson`: one row in the LESSONS.md format, or "none".
- Only E-INFRA may be handled without the human: resume the same cell if B4e2 T4 (row-split invariance) was IDENTICAL
  and the D18 resume rule is in force, otherwise restart the whole cell into a fresh directory; at most 2 attempts per
  cell, each recorded in the STOP bundle. Every other code waits for the human. Never re-run single images selectively.
