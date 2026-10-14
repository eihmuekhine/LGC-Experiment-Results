# HUMAN SIGN-OFF for B5 (go / no-go) — template v1.0

Copy to `reports/HUMAN_SIGNOFF_B5.md`, fill it in yourself (H), commit, then `git tag b5-go-<date>`.
Rules: every line needs evidence (a file path + sha256, or a command and its output pasted in the evidence column).
A box that cannot be ticked with evidence is NO-GO. An AI may draft the evidence column; only H ticks the boxes and
signs. Do not tick from an AI summary: open the file or run the command yourself.

Quick sha256: `sha256sum <file>`; HEAD: `git rev-parse HEAD`.

## Header

| item | value |
|---|---|
| date (UTC) | |
| worktree HEAD at sign-off | |
| contract version / sha256 | v1.9 / |
| study GPU (`nvidia-smi --query-gpu=name,driver_version --format=csv`) | |
| torch / CUDA (`python -c "import torch;print(torch.__version__, torch.version.cuda)"`) | |
| human_preflight report used | `reports/human_preflight_<...>.md` sha256: |

## S0 environment lock (B4e2)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 0.1 | Study GPU equals the GPU in the contract (D16) and in every root of `reports/B4e2_smoke_roots.txt` | `human_preflight` §1: one GPU name; compare with `nvidia-smi` | | [ ] |
| 0.2 | Backend flags identical for every runner: matmul TF32 False, cuDNN TF32 False, deterministic True, benchmark False | `human_preflight` §1: zero FLAG lines in the flag table | | [ ] |
| 0.3 | `ADV_STORE_BUDGETS` unset in every root | `human_preflight` §1 | | [ ] |
| 0.4 | B0' frozen-engine A == A2 (determinism) on the study GPU | read the B4e2 T1 table in `reports/B4e2_close.md` | | [ ] |
| 0.5 | Observation-only checkpoints: A == B at the 6 registered budgets | read the B4e2 T1 table yourself | | [ ] |
| 0.6 | Reviewer (new session, header `Reviewer · <model>`) B4e2 review: every criterion PASS | `reports/raw/B4e2/checker_review.md` | | [ ] |
| 0.7 | Combined smoke gate on `reports/B4e2_smoke_roots.txt` exits 0 (one GPU name); row-split invariance (B4e2 T4) IDENTICAL or B5 does not split cells | B4e2 T3/T4 | | [ ] |

## S1 protocol audit (B4f T1-T3)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 1.1 | One loader per dataset; no resize/crop after x0 | B4f T1 table: zero rows "after x0" | | [ ] |
| 1.2 | Normalisation mean/std equal to the weights' source for all 6 victims | B4f T2 table | | [ ] |
| 1.3 | x0 sha256 identical across methods for every (arch, image) | `human_preflight` §2: mismatches = 0 | | [ ] |
| 1.4 | Visual identity: one grid row matches the original JPEG (same scene, centre-cropped) | open one `*_grid_*.png` and the JPEG from `source_relpath` | file names: | [ ] |
| 1.5 | 12 split CSVs, rows 1-200: wrong source 0, wrong target 0, same-label 0, duplicates 0 | B4f T3 table | | [ ] |

## S2 hyper-parameter audit (B4f T4)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 2.1 | Every argument listed with adapter `file:line` and official `file:line` | `reports/B4f_hparam_audit.md` | number of rows: | [ ] |
| 2.2 | Every DIFFERS row has an L3 reason (or an approved L4 decision D8/D11) | read every DIFFERS row | DIFFERS rows: | [ ] |
| 2.3 | Two random rows re-read in the official clone by H | `sed -n <a>,<b>p <official file>` | rows + output: | [ ] |
| 2.4 | No value copied from our method into a baseline; nothing tuned | DIFFERS reasons | | [ ] |

## S3 oracle and query audit (B4f T5)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 3.1 | Every victim call site of every method goes through the tracker | call-site table; H greps one official file: `grep -n "model\|predict" <file>` and finds each hit in the table | method checked: | [ ] |
| 3.2 | Early stops / iteration caps neutralised (L3) or cannot fire before the budget | T5(b) table | | [ ] |
| 3.3 | Evaluation-only calls are uncounted and invisible to the attack | T5(c) | | [ ] |
| 3.4 | Baselines: queries == 10000 on every smoke image; no cell > 10000 | `human_preflight` §3 and T5(d) | | [ ] |
| 3.5 | Official counter vs tracker equal up to the documented offsets | T5(e) | | [ ] |

## S4 domain and clipping (B4f T6, T7)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 4.1 | max \|victim(x0) - raw_net((x0-mu)/sd)\| = 0 for all 6 victims | T6 table | | [ ] |
| 4.2 | Normalised-space methods: d_n/d_p in [4.37, 4.46] | T6 table | | [ ] |
| 4.3 | Every env file records the 5 `backend.*` keys read at run time | T7 + `human_preflight` §1 rerun | | [ ] |
| 4.4 | Stored-delta L2 equals pix_l2 (relative error < 1e-3) | `human_preflight` §4: FLAG = 0 | | [ ] |

## S5 known-answer test (B4g T1)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 5.1 | (a) adversarial within the float16 tolerance: all PASS, every method x mode | KAT table | | [ ] |
| 5.2 | (b) no L2 below the analytic bound d* (NT) / LB_t (T) | KAT table | | [ ] |
| 5.3 | (c) pix_l2 equals the npz norm | KAT table | | [ ] |
| 5.4 | (d) queries == 10000 for baselines (<= 10000 ours); tracker == forward count minus listed eval calls | KAT table | | [ ] |
| 5.5 | (e) convergence ratio read (descriptive only; note anything > 10) | KAT table | | [ ] |

## S6 invariance (B4e T1(d), B4e2 T1/T4/T5)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 6.1 | CGBA fixed vs d8only bit-identical on trajectories without the NaN event | B4e T1(d) | | [ ] |
| 6.2 | `nonfinite_queries` = 0 for every fixed-variant cell | `human_preflight` §3 | | [ ] |
| 6.3 | One attack process per GPU (D14/D16); all B5 GPUs of the study GPU type, same driver and torch | B5 launch script + `nvidia-smi` per host | | [ ] |
| 6.4 | Resume rule: any image with a row is done (D6/D18), or B5 never resumes | B4e2 T5 or contract note | | [ ] |

## S7 smoke artefacts and checker

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 7.1 | checker exit code 0 on the smoke roots: `python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage smoke --n 3 --roots <smoke roots> --report reports/contract_smoke.md; echo $?` | command output + report sha256 | | [ ] |
| 7.2 | `human_preflight` FLAG lines = 0, or each one explained in writing below | count `FLAG` in the report | count: | [ ] |
| 7.3 | Curves: no non-monotone best-so-far; every flat 2000->10000 curve explained | `human_preflight` §5 | | [ ] |
| 7.4 | Grids opened: perturbations look like attacks (no blank, no full-image noise, no NaN label) | open every `*_grid_*.png` | | [ ] |

## S8 pipeline canary (B4g T2)

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 8.1 | Every contract element E1-E8 produced (labelled SMOKE) | open every file in `generated/smoke_canary/` | | [ ] |
| 8.2 | Ours listed first; every method and cell present; n printed; no NaN | same | | [ ] |
| 8.3 | Log-x axes with 1-2-5 ticks; one amplification per grid row | same | | [ ] |
| 8.4 | No number interpreted (n = 3) | — | | [ ] |

## S9 contract, amendments, cost

| # | check | how H checks | evidence | OK |
|---|---|---|---|---|
| 9.1 | Contract v1.9 applied by H; checker tests still pass | `python tools/eval/tests_contract/test_contract.py` (all PASS) and `python tools/eval/tests_g5/test_g5.py`; contract sha256 | | [ ] |
| 9.2 | Amendment-2 deviation notes committed BEFORE any n = 200 run (GPU change, D11-D17) | `git log --oneline -- <notes file>` | | [ ] |
| 9.3 | Confirmation seeds untouched; no tuning on them | statement + `git log` of seed files | | [ ] |
| 9.4 | B5 cost table approved (GPU hours per cell, total, price) | cost table file | total h: | [ ] |
| 9.5 | Security: no `kaggle.json`, `.env` or API key tracked | `git ls-files | grep -Ei "kaggle|\.env|key"` = empty | | [ ] |
| 9.6 | `docs/LESSONS.md`: every lesson proposed in B4e2-B4g STOP bundles or reviews accepted or rejected by H; active rows still hold | `git log -- docs/LESSONS.md` | | [ ] |

## Open items / explanations of FLAG lines

(write here; each item: what, why acceptable, who reviewed)

## Decision

- [ ] GO for B5 under contract v1.9 (sha256 above)
- [ ] NO-GO; blocking items:

Signed (H): ____________________   Date: __________

Reviewed (L, Claude Opus 5.5) — paste the reply id/date of the chat review: __________
Reviewed (R, `reviewer` agent, model slug + OpenRouter provider): __________
