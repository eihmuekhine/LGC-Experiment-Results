# checker_review — B5a2-finish (A + B), independent review

Fresh session; no author work in this session (first line per prompt). Review prompt:
`reports/raw/B5a2/finish/finish_review.txt`, sha256
`bc47363c63e25ae99685e8696cad6ba9bdb7588c10138252fee10e777d0d2e31`. Skills lgc-preflight (§6 review mode) and
lgc-paper-contract loaded. CPU only; no GPU process started (nvidia-smi read once before the rule was shown to me;
no computation on GPU). Probe files only under `/tmp/b5finish_review/`. Nothing written outside this file.

Read in order: finish_A.txt, finish_B.txt, agent_log_A.md, agent_log_B.md, reports/B5a2_stop_decisions.md,
`reports/raw/B5a2/export/*`, `paper_exports/b5a2/COMMANDS.md`, then reports/B5a2_finish_A.md and
reports/B5a2_finish_B.md (and, for the close phase context, `reports/raw/B5a2/close/agent_log.md` and
reports/B5_close.md).

## Criteria table

| criterion | verdict | evidence (my own reruns) |
|---|---|---|
| V1 stop decisions | PASS | Own csv-module script (`/tmp/b5finish_review/v1_recompute.py`) recomputed, from `reports/raw/B5a2/gates/<cell>_domain.csv`, the C/Q9/Q6/X counts and the STORAGE/DEFECT/OTHER label of every X row of all 9 cells by the rule quoted in finish_A.txt A1: **16 X rows**, identical images/budgets/labels (values equal at full precision) and identical cell decisions (6 INCLUDED, 3 EXCLUDED-FROM-CLAIMS: main_CGBA_H_vgg19_T, e11_CGBA_H_Salman2020Do_R50_T, main_CGBA_resnet50_NT) — equals reports/B5a2_stop_decisions.md line by line. Sign-convention quote verified: `adapters/diag/domain_consistency.py:12-15` (docstring), `:73-79` (`margins`), `:95-103` (`classify`); m > 0 = decision NOT adversarial. agent_log_A.md A1 rows (08:35:16 read_domain, 08:35:38 classify_domain) print only domain classes/labels/decisions — no outcome metric computed or printed during A1. |
| V1b G5 alias | PASS | Own script (`/tmp/b5finish_review/v1b_alias.py`): `reports/raw/B5a2/close/g5/ttba_table1_alias.json` differs from `ttba_github/ttba_table1.json` by **30 added `CGBA_H` leaves only** (table1/<arch>/CGBA_H/<mode>/<budget>), each equal to that arch's `CGBA-H` value; 0 removed keys, 0 changed keys; sha256 shipped `2cb224f682f9…9059e`, alias `5532b173f84d…1e558` — both equal the part-A record. `reports/raw/B5a2/close/agent_log.md` C2 rows 12-14 show the three main CGBA_H cells run with `--table1 …/ttba_table1_alias.json` (NT cells with the shipped table). I reran the resnet50 targeted C2 command myself: `VERDICT: PASS (all CIs intersect [0.5, 2] at every budget)`, exit 0 — equals the recorded VERDICT line (log `/tmp/b5finish_review/v1b_g5_rerun.log`). |
| V2 metrics run | PASS | `metrics_commands.sh`: **48** non-comment lines, every line exactly `python tools/eval/eval_metrics_offline.py --roots <dir> --out paper_exports/b5a2/metrics/<dirname>/metrics.csv --device cuda:0 --batch 32`, no `; & | < > \` $` in any line. One line per main/E10/E11 cell of the inventory (48/48; no missing/extra/duplicate roots; the 4 appendix dirs excluded). One output per line exists (48 `metrics.csv`). Output row count equals the inventory `adv_store` count — checked for 5 random cells and in fact for **all 48** (1600/1599/1594/1574 as in the inventory; 0 mismatches). `metrics_run.log`: 48 `[+] wrote` lines, `grep -c Traceback` = 0, `grep -ci error` = 0. |
| V3 cell files | PASS | `paper_exports/b5a2/cells/`: 52 directories, **196** files, `*.npz` count 0, files > 50 MB count 0. Per-directory counts (runs/per_checkpoint/env/manifest) equal the inventory for all 52 (`/tmp/b5finish_review/v3_cells.py`). 10 random files: sha256 equals the source under `Results/` 10/10. |
| V4 derived outputs | PASS | Tool sha256 in COMMANDS.md equal mine: make_tables_figures `86c07e26099c…42b0e3`, make_low_query_table `026c4bdd…46ccae36`, make_comparison_grid `e28773459…46bd6ee1`. Rerun 1 (full make_low_query_table.py command, `--outdir /tmp/b5finish_review/make_low_query_table`): all 6 `tab_lowq_*.tex` **byte-equal**. Rerun 2 (make_comparison_grid.py resnet50 NT Q1000, `--out /tmp/b5finish_review/…`): `.png` and `_caption.txt` **byte-equal**; `.pdf` identical except `/CreationDate` (D:20261023101958 vs D:20261023095042), `pdftotext` outputs identical — allowed ("equal numbers for files with a time stamp"). All 3 make_tables_figures.py and all 12 make_comparison_grid.py command lines carry **exactly** the three display-name mappings `LGC_RC_EGC=LGC-EGC (ours)`, `CGBA_H=CGBA-H`, `ARS_OPT=ARS-OPT` (verified programmatically; no extra, no missing). pdftotext `grep -c -E 'LGC_RC_EGC\|ARS_OPT\|CGBA_H'` = **0 on every PDF** written by the two tools (24 + 12 = 36/36). |
| V5 claims | PASS (R3) | claims_spec.md blocks are **verbatim** (byte-equal) `prereg_build38.txt` §6 and `prereg_build38_amendment2.txt` §7 and §2; prereg sha256 `0f2e9039…51ed27`, amendment `178809f98…9fd2c58` match. **`tools/eval/claims_registered.py` was written after the runs** (agent_log_B.md rows 10:06-10:09, session B; file created by finish_B B3c) — stated here as required; I found **no mismatch** with the quoted text: one-sided Wilcoxon signed-rank (`alternative="less"`, d = log-ratio for L2, difference for LPIPS/DISTS; lines 107-111), Holm within each family only (T1/P1/P2; lines 321-323), family membership and m = T1 3 / P1 6 / P2 16 = 18 − 2 EXCLUDED cells, CI rules T1 <1 / P1,P2 <0 / T2 ≤1.03 / N1 ≤1.15 / supporting <0 (lines 334-348), AUC definition per contract (`auc_budgets` [500…10000], log pix_l2 mean / metric mean / ASR indicator with contract `asr_perc` tau; lines 90-104, 168-176), bootstrap B = 10000 read from `contract.constants.stats.bootstrap_B` (line 193), seed `--seed 0` with **no contract seed constant** (verified; author's deviation 5). Rerun into `/tmp/b5finish_review/claims`: claims.json reproduced **exactly** (meta + all 82 rows + excluded_comparisons equal). Every row has exactly the 15 required keys; null p-values exactly on the 57 CI-only rows (family = null); decision values {confirmed, not_confirmed, descriptive} with counts 58/8/16 as in the RESULT block; `excluded_comparisons` = exactly the 2 main EXCLUDED-FROM-CLAIMS cell comparisons (CGBA_H vgg19 targeted, CGBA resnet50 non_targeted, "domain gate failed"). The 34-row claims table in reports/B5a2_finish_B.md matches claims.json row by row. |
| V6 E9 + erratum | PASS | E9 search commands reproduce: `find . -maxdepth 4 \( -iname '*confirm_37*' -o -iname '*seed2718*' \)` → only `./confirm_37.py`; `ls -d Results/*confirm*` → none; `find / -maxdepth 6 (…confirm_37.csv…)` → none (matches reports/E9_sources.md). `git diff --numstat reports/B1_sources.md` → **46 0** (additions only). Erratum heading and body equal finish_B.txt B6 verbatim. Quoted passages match the files: `adapters/run_arsopt.py:111` (`attacker = ParsOptL2Norm(`), `:114` (`method="ARS"`), `:117` (constructor end), `:121` (`attacker.targeted_attack…`), `:123` (`attacker.untargeted_attack…`); official clone HEAD `ab71cb88058ad0ca778e0048e5cf1b60921bb673` (`git rev-parse HEAD` in `/root/autodl-tmp/baselines/ARS-OPT`); `PARS_OPT/pars_opt_l2_norm_attack.py:21` (`self.surrogate_models = … else []`), `:317` (`def get_prior_grad`), `:577` (`def untargeted_attack`), `:675` (`def targeted_attack`). |
| V7 manifest + lists | PASS | `cd paper_exports/b5a2 && sha256sum -c MANIFEST.sha256`: **0** non-OK lines (408 entries). `grep -c 'autodl-tmp\|LGC_frozen\|\.\.'` on MANIFEST.sha256 = 0. `add_force.txt` = `paper_exports/b5a2` (1 line). `add_normal.txt` names **all 35** required paths (12 stop bundles, 13 `reports/human_preflight_b5a2*` files, the 7 reports, `reports/raw/B5a2`, `reports/raw/B5a2_triage`, `tools/eval/claims_registered.py`); all exist; none missing, none extra; **no file > 50 MB** in any listed path (checked by walk over all 36 paths). |
| V8 scope | PASS (R1, R2) | `sha256sum -c /root/autodl-tmp/B5a2_manifest_20261021.sha256 | grep -vc ': OK$'` → **0**. `git status --short | grep -v '^??'` → only ` M reports/B1_sources.md` (E9_sources.md is untracked/new, allowed). No `git add/commit/stash/checkout/reset/tag` in any of the three agent logs (grep exit 1). All *tracked* changes and all untracked paths attributable to the finish sessions are within the allowed lists (see R1 for 16 pre-existing untracked files outside the lists). Deviation 4 verified: tags `build38-prereg`/`build38-amend2` → "unknown revision"; prereg commit `9e2822e 2026-10-08 08:39:28 +0800`, amendment commit `d845c74 2026-10-11 20:00:47 +0800` — as reported. |
| V9 unsupported claims | 3 entries | R1, R2, R3 below. Every other numeric claim I could test in the two reports (A0 values, 52 job lines, exitval seq `26 28 30 31 36 39 40 41 44`, 12 stop bundles, bundle-pair codes E-G5/E-OTHER, inventory counts, 196 cell files, 36 PDFs, 408 manifest entries, 48 metrics lines, claims counts 58/8/16, families m 3/6/16, tool sha256s, contract sha256, MANIFEST sha256 `523fbc15…`, `inputs/ttba_table1.json` sha `2cb224f6…`, P2_selection_audit NOT FOUND, E9 NOT FOUND, lgc-baselines skill absent, close verdict READY-FOR-REVIEW) verifies against my own reruns. |

## Recall entries

R1 | V8 | MINOR | `git status --short` lists 16 untracked files not named by any allowed write list and not in
`add_normal.txt`/`add_force.txt`: `1140NN.md 140NN.md 1509N.md 1518N.md 1641N.md 1919NN.md 216NN.md 245NN.md
248NNreviewer.md 312NN.md 325NN.md 414NN.md 430NN.md 509NN.md 536NN.md` (mtime Oct 19-20, i.e. before the
finish_A/finish_B sessions of Oct 23 08:34-10:15; no command in agent_log_A.md / agent_log_B.md /
close/agent_log.md writes them) — pre-existing worktree leftovers, not attributable to the finish author, but they
make `git status` non-conforming to the literal criterion and would be swept up by a `git add -A`. |
scope: H decides (move/remove the files or leave untracked; not a finish-session defect) |
acceptance: `git status --short | grep '^??' | grep -vE 'reports/|paper_exports/|tools/eval/claims_registered.py'` prints nothing.

R2 | V8/X6 | MINOR | The RESULT-block sha256 values for the three self-referential files are stale: recorded
`reports/B5a2_finish_A.md e6f6592a…` vs actual `1b8cae3f…`; recorded `reports/B5a2_finish_B.md 5319b545…` (copied
from the 10:15:10 sha256sum in agent_log_B.md) vs actual `3dd2516e…`; recorded
`reports/raw/B5a2/finish/agent_log_B.md ae00189b…` vs actual `828fcdfd…` (the 10:15:10 log row appended after the
sha was taken). Inherent to the X6 self-hash convention: the values were copied faithfully from command output at
the time, then the RESULT block / final log row changed the files. All other recorded sha256 values (13 checked)
match the current files byte-exactly. | scope: none (bookkeeping note for H; no data affected) |
acceptance: none required; or re-record final sha256 after the last write in future sessions.

R3 | V9 | MINOR | reports/B5a2_finish_B.md §B3(d) line "(plus 6 N1/T2 rows and 40 supporting CI-only rows in
claims.json …)" is not supported: claims.json has **48** supporting CI-only rows (16 LPIPS-alex + 16 DISTS + 16
perceptual ASR; verified by count), and all 9 T2/N1 rows are already inside the 34-row claims table (34 + 48 = 82;
nothing extra). The JSON, the RESULT-block counts (58/8/16, 82 rows) and the decision counts are correct. |
scope: reports/B5a2_finish_B.md §B3(d) parenthetical only (never the JSON) |
acceptance: fix the parenthetical to "plus 48 supporting CI-only rows"; `python -c "import json;d=json.load(open('paper_exports/b5a2/claims/claims.json'));print(sum(1 for r in d['rows'] if r['claim_id'].startswith('SUP_')))"` prints 48.

## Verdict

All nine criteria PASS on my own reproduced evidence. Three MINOR recall entries (R1-R3); no BLOCKER, no MAJOR.

VERDICT: PASS-WITH-NOTES