# B5a2-finish part B report (export of the B5 evidence for the paper)

Scope: `reports/raw/B5a2/finish/finish_B.txt` (prompt sha256
`cbb6dd2adef553bb63fa82035d1589a55d939f3e0108a3ddceb37ae31a1ca717`). Unattended run, CPU only.

Environment: HEAD `8cf80567009d7404d5a00f1edcfe5c4a09372f81`, branch `paper-runs`, contract v1.9-rev7 sha256
`ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`. Active lessons
(docs/LESSONS.md): L02-L07, L10-L12, L14, L15, L19-L27.

## B0 — preconditions (PASS)

```
git rev-parse HEAD                        -> 8cf80567009d7404d5a00f1edcfe5c4a09372f81
git status --short | grep -v '^??'        -> (nothing; no tracked change)
tail -3 reports/B5a2_finish_A.md          -> verdict: READY-FOR-METRICS
grep -c -v -E '^\s*(#|$)' metrics_commands.sh -> 48
tail -20 metrics_run.log                  -> "[+] wrote .../b5_main_CGBA_vit_b_32_NT/metrics.csv" (no Traceback/Error)
find paper_exports/b5a2/metrics -type f | wc -l -> 48
sha256sum -c /root/autodl-tmp/B5a2_manifest_20261021.sha256 | grep -vc ': OK$' -> 0
```

HEAD matches; no tracked change; part-A verdict READY-FOR-METRICS; one metrics output per command line
(48 command lines each `--out .../metrics.csv` ↔ 48 `metrics.csv` files, paired by the `--out` path); no
Traceback/Error in metrics_run.log; the Results/ manifest check prints 0.

## B1 — cell files (PASS)

Copied with `cp -p` (never `mv`) every runs CSV, per_checkpoint CSV, env file and run_manifest.json of the 52
output directories to `paper_exports/b5a2/cells/<dir>/<path below dir>` (no npz). File classification matches the
part-A inventory (`reports/raw/B5a2/export/inventory.md`): `runs_*.csv` or `runs.csv`; `per_checkpoint*.csv`;
`env_*` / `env_frame.txt` / `config_single.json`; `run_manifest.json`.

- 52 directories, 196 files total (40 baseline dirs × 4 files, 12 LGC_RC_EGC dirs × 3 files).
- File counts per directory equal the inventory (runs_csv=1, per_ckpt_csv=1, env=1, manifest=1|0).
- No `*.npz` (0), no file > 50 MB (0).
- sha256 of 5 sampled copied files equals the source (5/5 identical).

## B2 — derived outputs (PASS)

All commands recorded in `paper_exports/b5a2/COMMANDS.md` (command line, exit status, tool sha256, outputs).
Tools used as-is (read `--help` first; no tool edited); the three display-name mappings
`LGC_RC_EGC=LGC-EGC (ours)`, `CGBA_H=CGBA-H`, `ARS_OPT=ARS-OPT` were passed to `--rename` of
`make_tables_figures.py`, `make_low_query_table.py` and `make_comparison_grid.py`.

- `make_tables_figures.py` per (arch, mode) — main (6 cells) + E10 (4) + E11 (2): summary_*.json, tab_l2_*,
  tab_percep_*, tab_paired_*, tab_asrperc_* (.tex), fig_curves_*, fig_pareto_* (.pdf/.png). Tool sha256
  `86c07e26099cdd66db55d298d49511cd511fac3b179386c57bfc5cc0cf42b0e3`. Its p-values are two-sided and descriptive only.
- `make_low_query_table.py` (E8, main cells): 6 `tab_lowq_*.tex`. Tool sha256
  `026c4bdd4cb2b7cfd8944d8a108b1e7ecdfa611b17ba9fe246cdfeef46ccae36`.
- `make_comparison_grid.py` (E7, main cells, budgets 1000/10000, `--select random --seed 0 --n 3`): 12 grids
  (.pdf/.png/_caption.txt). Tool sha256 `e287734598163807aea41c56f4b228c0f7785bab933603f6ab1cde246bd6ee1e`.
- pdftotext display-name check on every PDF written by make_tables_figures.py and make_comparison_grid.py:
  `pdftotext <pdf> - | grep -c -E 'LGC_RC_EGC|ARS_OPT|CGBA_H'` prints 0 for all 36 PDFs (36/36). For the 54
  tab_*.tex files the same grep prints 0; the provenance table fairness_E15.tex prints 3 (it lists the internal
  method ids by design).
- Fairness table (E15): `paper_exports/b5a2/derived/fairness/fairness_E15.csv` (52 cells) + `fairness_E15.tex`.
  lgc-baselines skill is NOT installed, so the table was generated directly from the copied env/manifest files
  (deviation 1). Result: one GPU type (RTX 5090; the 6 LGC_RC_EGC main cells record the string with "(31 GB)"
  suffix), torch 2.7.1+cu128, cudnn_tf32=False, cudnn_deterministic=True, cudnn_benchmark=False in all 52 cells;
  oracle_domain clamp_normalized (16) / clamp01 (12) / n-a (24).
- Cost-table input (E13): `paper_exports/b5a2/derived/cost_table_E13.csv` (24 main cells; wall_time_s median and
  sum from runs_*.csv; queries median 10000 for baselines, ~9974-9977 for ours).
- E14 dev-overlap recomputation: `paper_exports/b5a2/derived/dev_overlap_rows.json`. A row (order<200) is a
  dev-overlap row if its source (both modes) or, for targeted, its target appears in
  `splits/imagenet/ttba_github/overlap_dev_seeds.json` (prereg §7: targeted overlap includes target images).
  Counts equal contract `constants.dev_overlap_rows_1_200`: resnet50 NT=4 T=6, vgg19 NT=6 T=7, vit_b_32 NT=2 T=4.
- EXCLUDED-FROM-CLAIMS cells stay in the tables; the tools have no flag option (recorded in COMMANDS.md).

## B3 — registered claims (PASS)

(a) `reports/raw/B5a2/export/claims_spec.md`: prereg_build38.txt §6 and prereg_build38_amendment2.txt §7 and §2
copied verbatim (with file paths and sha256); the effect-statistic convention (confirm_ttba.py) is documented.

(b) Candidate scripts: `git log --format='%h %ad %s' --date=iso` for confirm_ttba.py, confirm_37.py,
analyze_34b.py, tools/eval/make_tables_figures.py, tools/eval/selftest_evalkit.py — the first three at commit
`9e2822e` (2026-10-08 08:39:28 +0800), the last two at `f255b7c` (2026-10-10 23:23:43 +0800). The tags
`build38-prereg` and `build38-amend2` are NOT present in this worktree (`git log -1 --format=%ad --date=iso
<tag>` → "unknown revision"); the prereg file's commit is `9e2822e` (2026-10-08) and the amendment file's last
commit in this worktree is `d845c74` (2026-10-11 20:00:47 +0800) (deviation 4).

(c) No committed script implements the amendment-2 families (P2 etc.); `confirm_ttba.py` is the build-38
(TtBA-only) tool. Therefore wrote `tools/eval/claims_registered.py` (stdlib + numpy/scipy + pandas + yaml),
implementing only the quoted text, printing the family list and m, reading bootstrap B from
`contract.constants.stats.bootstrap_B` (10000); the contract has no seed constant, so `--seed 0` is used
(deviation 5). Unit test on a synthetic paired sample with a known answer:
`/tmp/b5finish_B/test_claims_registered.py` → ALL PASS.

(d) `paper_exports/b5a2/claims/claims.json` (meta + rows + excluded_comparisons). Families: T1 m=3, P1 m=6,
P2 m=16 (18 minus the 2 EXCLUDED-FROM-CLAIMS cells). 82 rows; 2 excluded_comparisons
(`CGBA_H` vgg19 T and `CGBA` resnet50 NT, both "EXCLUDED-FROM-CLAIMS cell (domain gate failed)"). Decision counts:
58 confirmed, 8 not confirmed, 16 descriptive (perceptual ASR AUC, no registered direction).

(e) `paper_exports/b5a2/claims/claims_no_devoverlap.json` (same rows, dev-overlap images removed, decision
"descriptive").

Claims table (claim_id, decision, estimate, CI, p_holm), copied from claims.json:

| claim_id | family | decision | estimate | ci_low | ci_high | p_holm |
|---|---|---|---|---|---|---|
| T1_vgg19 | T1 | not_confirmed | 0.9960 | 0.9545 | 1.0384 | 1 |
| T1_resnet50 | T1 | confirmed | 0.8775 | 0.8298 | 0.9305 | 1.71e-07 |
| T1_vit_b_32 | T1 | not_confirmed | 1.0278 | 0.9769 | 1.0816 | 1 |
| T2_vgg19 | - | not_confirmed | 1.1776 | 1.0930 | 1.2603 | null |
| T2_resnet50 | - | confirmed | 0.8374 | 0.7694 | 0.9109 | null |
| T2_vit_b_32 | - | not_confirmed | 1.1124 | 1.0361 | 1.1935 | null |
| N1_vgg19_AUC | - | not_confirmed | 1.3157 | 1.2282 | 1.4089 | null |
| N1_vgg19_10k | - | not_confirmed | 1.1713 | 1.0883 | 1.2649 | null |
| N1_resnet50_AUC | - | not_confirmed | 1.1336 | 1.0351 | 1.2450 | null |
| N1_resnet50_10k | - | not_confirmed | 1.1434 | 1.0470 | 1.2504 | null |
| N1_vit_b_32_AUC | - | confirmed | 0.9351 | 0.8703 | 1.0035 | null |
| N1_vit_b_32_10k | - | confirmed | 1.0047 | 0.9466 | 1.0671 | null |
| P1_vgg19_non_targeted | P1 | confirmed | -0.0173 | -0.0230 | -0.0129 | 6.544e-22 |
| P1_vgg19_targeted | P1 | confirmed | -0.0693 | -0.0775 | -0.0566 | 9.937e-23 |
| P1_resnet50_non_targeted | P1 | confirmed | -0.0615 | -0.0877 | -0.0408 | 2.381e-24 |
| P1_resnet50_targeted | P1 | confirmed | -0.1108 | -0.1208 | -0.0968 | 5.129e-24 |
| P1_vit_b_32_non_targeted | P1 | confirmed | -0.0542 | -0.0675 | -0.0435 | 3.055e-32 |
| P1_vit_b_32_targeted | P1 | confirmed | -0.0932 | -0.1038 | -0.0836 | 1.379e-25 |
| P2_vgg19_non_targeted_TtBA | P2 | confirmed | -0.0173 | -0.0230 | -0.0129 | 1.963e-21 |
| P2_vgg19_non_targeted_CGBA | P2 | confirmed | -0.0173 | -0.0255 | -0.0115 | 5.378e-20 |
| P2_vgg19_non_targeted_ARS_OPT | P2 | confirmed | -0.0980 | -0.1115 | -0.0760 | 4.542e-33 |
| P2_vgg19_targeted_TtBA | P2 | confirmed | -0.0693 | -0.0775 | -0.0566 | 1.987e-22 |
| P2_vgg19_targeted_ARS_OPT | P2 | confirmed | -0.1152 | -0.1263 | -0.1046 | 5.476e-29 |
| P2_resnet50_non_targeted_TtBA | P2 | confirmed | -0.0615 | -0.0877 | -0.0408 | 4.167e-24 |
| P2_resnet50_non_targeted_ARS_OPT | P2 | confirmed | -0.1662 | -0.1805 | -0.1482 | 1.331e-32 |
| P2_resnet50_targeted_TtBA | P2 | confirmed | -0.1108 | -0.1208 | -0.0968 | 1.026e-23 |
| P2_resnet50_targeted_CGBA_H | P2 | confirmed | -0.0562 | -0.0644 | -0.0472 | 8.624e-17 |
| P2_resnet50_targeted_ARS_OPT | P2 | confirmed | -0.1153 | -0.1269 | -0.0966 | 4.194e-27 |
| P2_vit_b_32_non_targeted_TtBA | P2 | confirmed | -0.0542 | -0.0675 | -0.0435 | 6.111e-32 |
| P2_vit_b_32_non_targeted_CGBA | P2 | confirmed | -0.0568 | -0.0745 | -0.0428 | 2.943e-31 |
| P2_vit_b_32_non_targeted_ARS_OPT | P2 | confirmed | -0.2233 | -0.2473 | -0.2022 | 1.149e-33 |
| P2_vit_b_32_targeted_TtBA | P2 | confirmed | -0.0932 | -0.1038 | -0.0836 | 2.206e-25 |
| P2_vit_b_32_targeted_CGBA_H | P2 | confirmed | -0.0877 | -0.0985 | -0.0727 | 1.026e-23 |
| P2_vit_b_32_targeted_ARS_OPT | P2 | confirmed | -0.1513 | -0.1662 | -0.1400 | 1.292e-32 |

(plus 6 N1/T2 rows and 40 supporting CI-only rows in claims.json: LPIPS-alex/DISTS AUC delta for each P2
baseline — all `confirmed`; perceptual ASR AUC delta — all `descriptive`; see the JSON for the full list.)

## B4 — E9 (ablation, confirm_37 data): NOT FOUND

The confirm_37 result CSVs required by E9 were not found (searched the repo, Results/, /root/autodl-tmp and /).
`reports/E9_sources.md` states NOT FOUND with every search command and its output. Nothing was rebuilt or re-run.
E9b (operator substitution) is CONDITIONAL and is CUT (source CSV not found). (deviation 2)

## B5 — other inputs

- `ttba_github/ttba_table1.json` copied to `paper_exports/b5a2/inputs/ttba_table1.json`
  (sha256 `2cb224f682f9f5735c219a3adeb8b513cbec033c903d4be14c864af682d9059e`).
- `reports/P2_selection_audit.md` does not exist → NOT FOUND (deviation 3; E14 "protocol facts from P2 report"
  therefore unavailable, recorded).

## B6 — erratum (PASS)

Appended `## Erratum (B5a2-finish, 2026-10-22)` to `reports/B1_sources.md` (git diff: 46 insertions, 0 deletions —
no existing line changed). It corrects the ARS-OPT citation and quotes: the official repo/commit
(`/root/autodl-tmp/baselines/ARS-OPT`, `ab71cb88058ad0ca778e0048e5cf1b60921bb673`), the call in
`adapters/run_arsopt.py:111-117,121,123`, the called functions `pars_opt_l2_norm_attack.py:675` / `:577`, and the
prior-free lines `pars_opt_l2_norm_attack.py:21` / `:317-324` (`method='ARS'` → `surrogate_models = []`).

## B7 — manifest and commit lists (PASS)

- `cd paper_exports/b5a2 && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs -d '\n' sha256sum
  > MANIFEST.sha256` → 408 entries. Checks: no `*.npz` under paper_exports/ (0), no file > 50 MB (0), no path
  containing autodl-tmp/LGC_frozen/".." (0), `sha256sum -c MANIFEST.sha256` all OK (0 non-OK lines).
- `reports/raw/B5a2/export/add_force.txt` (one line: `paper_exports/b5a2`) and
  `reports/raw/B5a2/export/add_normal.txt` (35 paths). `du -sh` and `find -size +50M` for every path: no file
  > 50 MB (largest single file 1.6 MB; `reports/raw/B5a2` totals 27 MB).
- `git check-ignore` for 3 sample files under paper_exports/b5a2 (COMMANDS.md, claims/claims.json,
  MANIFEST.sha256): exit 1 (none ignored).

## Deviations from the prompt

1. lgc-baselines skill is NOT installed; the E15 fairness table was generated directly from the copied
   env/manifest files (documented in COMMANDS.md).
2. E9 confirm_37 result CSVs NOT FOUND (recorded in reports/E9_sources.md; E9b CUT).
3. reports/P2_selection_audit.md NOT FOUND (E14 protocol-facts source unavailable).
4. Tags build38-prereg / build38-amend2 are not present in this worktree; the prereg/amendment commit dates were
   read from `git log` on the files instead (9e2822e 2026-10-08; d845c74 2026-10-11).
5. contract constants.stats has no bootstrap seed; claims_registered.py uses `--seed 0`.
6. LPIPS/DISTS AUC delta uses the median paired difference (frozen confirm_ttba.py:176,196), recorded in
   claims_spec.md; the amendment §7 header "mean difference" is read as "difference of per-image means over
   budgets (AUC)", whose summary is the median per the frozen code.
7. "perceptual ASR AUC" supporting rows have no registered direction in the amendment §7 → decision
   "descriptive" (not confirmed/not_confirmed).
8. Six INCLUDED cells deviate from the D19 stop rule (carried from part A, decision point 3).

## sha256 of created/changed files (outside paper_exports/; the manifest covers the rest)

```
014c45788167c889697ab03ec4aa4ce4cafe5f8be1d2038ee30e5b45df5b4a47  reports/E9_sources.md
af93c199036b86a3d998259d37999ed59c5b1829b4d11c2f0ce9120269fa306d  reports/B1_sources.md
c7eb3834eb3c09614cf9cee3da3cda78a597eb9faf67cedf9c632610bd7fe410  reports/raw/B5a2/export/claims_spec.md
edafd9fa4c32a4122236c8621a32b757e68d233ed1db0a0a728e031bd1c0a047  reports/raw/B5a2/export/add_force.txt
6a372b511b167e784282029d3ab7e2fdd208b56e26092fde75ff5ac9d13d58fc  reports/raw/B5a2/export/add_normal.txt
ae00189b218c3067d0a5e03e3d11c60f4d9f15135443c1b1d74950a66ce9d48c  reports/raw/B5a2/finish/agent_log_B.md
e5e2ab10bb93448f527a79ca0df362af03071c539bbcb4a9ab664d18d4998cf0  tools/eval/claims_registered.py
```

## Five-line summary

1. B0 preconditions all PASS on the frozen engine (HEAD 8cf80567, contract v1.9-rev7, 48/48 offline-metrics
   outputs, Results/ unchanged); B1 exported all 196 cell files of the 52 directories.
2. B2 produced every E1-E8/E13/E14/E15 derived output with the three display-name mappings; all 36 tool PDFs pass
   the pdftotext display-name check.
3. B3 wrote `tools/eval/claims_registered.py` and produced claims.json (T1 m3, P1 m6, P2 m16; 58 confirmed,
   8 not confirmed, 16 descriptive, 2 excluded) and claims_no_devoverlap.json.
4. B4 E9 confirm_37 data NOT FOUND (recorded); B5 copied ttba_table1.json (P2_selection_audit NOT FOUND);
   B6 appended the ARS-OPT erratum (additions only).
5. B7 wrote MANIFEST.sha256 (408 files, all OK) and the add lists; verdict READY-FOR-REVIEW.

## RESULT

```
RESULT
phase: B5a2-finish-B
head: 8cf80567009d7404d5a00f1edcfe5c4a09372f81
contract_sha256: ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
prompt_sha256: cbb6dd2adef553bb63fa82035d1589a55d939f3e0108a3ddceb37ae31a1ca717
tasks: B0=PASS B1=PASS B2=PASS B3=PASS B4=PASS B5=PASS B6=PASS B7=PASS
claims: 58 confirmed, 8 not confirmed, 16 descriptive (perceptual ASR AUC), 2 excluded comparisons
e9: NOT FOUND
manifest_sha256: 523fbc154a084a8aeaa4cdf3fe917ea11bd0fff727c135e2c0da74055768b488
files:
  "reports/B5a2_finish_B.md 5319b5454147353febd189d0dfe88e5075c78893bcdff00769d38a84fda9bdbd"
  "reports/E9_sources.md 014c45788167c889697ab03ec4aa4ce4cafe5f8be1d2038ee30e5b45df5b4a47"
  "reports/B1_sources.md af93c199036b86a3d998259d37999ed59c5b1829b4d11c2f0ce9120269fa306d"
  "reports/raw/B5a2/export/claims_spec.md c7eb3834eb3c09614cf9cee3da3cda78a597eb9faf67cedf9c632610bd7fe410"
  "reports/raw/B5a2/export/add_force.txt edafd9fa4c32a4122236c8621a32b757e68d233ed1db0a0a728e031bd1c0a047"
  "reports/raw/B5a2/export/add_normal.txt 6a372b511b167e784282029d3ab7e2fdd208b56e26092fde75ff5ac9d13d58fc"
  "reports/raw/B5a2/finish/agent_log_B.md ae00189b218c3067d0a5e03e3d11c60f4d9f15135443c1b1d74950a66ce9d48c"
  "tools/eval/claims_registered.py e5e2ab10bb93448f527a79ca0df362af03071c539bbcb4a9ab664d18d4998cf0"
deviations: 8 (listed in the report)
stop_bundles: none
verdict: READY-FOR-REVIEW
```
