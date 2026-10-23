# checker_review — B5a2-close (finish addendum L applied)

Independent review, fresh session (no author work in this session). Prompt:
`reports/raw/B5a2/close_review_prompt.txt` sha256
`99213b6e45fc9d5afe46eabb587a90222b824327abc77226f8f92e185ec73ba8`; addendum
`reports/raw/B5a2/finish/close_review_addendum.txt` sha256
`bfc9adb457746f08573de5cbd187a7729ae12e20120f868f488d6d7f716ccd3d`.
No `fix*_prompt.txt` under `reports/raw/B5a2/close/` → first review, whole phase reviewed.
Unattended per addendum §3: probes only under `/tmp/b5finish_close_review/`.

Reviewed in prompt order: `prompt.txt` (run spec), `close_prompt.txt`, `git status --short`
(only untracked `reports/*`, `paper_exports/`, stray `*.md` at root; `git diff HEAD` empty),
`parallel.log`, `gates/*` (incl. `summary.csv`), `close/*`, `reports/stop/B5a2*` (12 bundles),
then `reports/B5_close.md` and `reports/E17_sources.md`; addendum files
`reports/B5a2_stop_decisions.md` (H decision, addendum §2) and
`reports/raw/B5a2/finish/finish_A.txt` (A2 alias instruction).

## Criteria

| criterion | verdict | evidence (my own commands) |
|---|---|---|
| C0 no attack process; 52 job lines | PASS | `pgrep -af "run_b5_parallel\|parallel -j\|run_benchmark\|run_ttba\|run_cgba\|run_arsopt"` → only its own shell line; `awk 'NR>1' reports/raw/B5a2/parallel.log \| wc -l` → 52. Exitval != 0 exactly for seq 26 28 30 31 36 39 40 41 44 (awk on col 7) — matches report §STOP bundles line. |
| C1 completeness table 52 rows | PASS | csv-module recount of all 24 main cells (≥ 10 required; includes all 6 main TtBA and all 3 main CGBA_H cells): rows=200 images=200 status `{budget:200}` (LGC_RC_EGC `{ok:200}`) error_rows=0 nonfinite_sum=0; e17 4 cells: 30 rows / 30 images, status budget. Matches the report's C1 table (per-cell Exitval from parallel.log; partial/g5/domain_x/stop from `gates/summary.csv` identical). The author's logged probe `/tmp/b5finish_probe/build_c1.py` → `c1_table.txt` (54 lines) shows the same values. |
| C2 G5 rerun, all main TtBA/CGBA/CGBA_H cells | PASS | `tools/eval/g5_check.py` rerun per arch×mode with cell roots: resnet50/vgg19/vit_b_32 `--mode non_targeted` (shipped table): `VERDICT: PASS (all CIs intersect [0.5, 2] at every budget)` exit 0 ×3; `--mode targeted --table1 .../ttba_table1_alias.json` ×3: `VERDICT: PASS` exit 0. With the shipped table the three targeted runs crash `KeyError: 'CGBA_H'` exit 1 → confirms defect L28 (addendum §1: known defect, not a G5 FAIL). No PENDING/FAIL/FLAG verdict remains. |
| C2a alias table (addendum §1) | PASS | My script: alias differs from `ttba_github/ttba_table1.json` only by added `CGBA_H` keys, equal per arch to that arch's `CGBA-H`, for resnet50/vgg19/vit_b_32/efficientnet_b0/inception_v3; every shipped key/value unchanged. sha256 shipped `2cb224f682f9...9059e`, alias `5532b173f84d...e558` — both match the report and the RESULT block. |
| C3 domain CSVs / X recount | PASS | 40/40 non-TtBA domain CSVs present (12 TtBA cells SKIP); recount by column `class`: X>0 for exactly the 9 stopped cells (1,1,1,2,2,3 and CGBA_H 1,2,3; e.g. `main_CGBA_resnet50_NT` rows=1594 C=1368 Q9=223 X=3); LGC_RC_EGC nonzero X = 1,1,1,1,10,15,18,24,28,37; CGBA_D8only X = 0,0,1,1 — all match `summary.csv` and the report. |
| C3a H decision (addendum §2) | PASS | `reports/B5a2_stop_decisions.md` exists; rule text verbatim from `finish_A.txt` A1 incl. "Rule fixed by H on 2026-10-22 before any outcome statistic was computed." I re-applied the rule mechanically to every X row of the 9 cells (m_s1/m_s9/m_s6 from `gates/<cell>_domain.csv`): STORAGE/DEFECT/OTHER classification and per-cell decision identical to the file (6 INCLUDED, 3 EXCLUDED-FROM-CLAIMS). Duplicate-bundle finding (first E-G5, second E-OTHER) confirmed by reading the bundle pairs. |
| C4 human preflight | PASS | `grep -c FLAG reports/human_preflight_b5a2.md` = 54 incl. header line → 53 FLAG rows: 52 × "driver and hostname not recorded (--require_host)" + 1 "FLAG: more than one value across env files" (GPU names 46 × `NVIDIA GeForce RTX 5090`, 6 × `... (31 GB) \| tf32=False deterministic=True` for the LGC_RC_EGC roots). Matches the report's C4 explanation exactly. |
| C5 contract checker (partial) | PASS | Rerun with `--report /tmp/b5finish_close_review/contract_partial.md --status_out ...` (review rule) over the 48 main/E10/E11 roots: n_roots=48, I01–I08, I10–I12 PASS count=0, I09 WARN count=0, exit code 0. Matches report C5. |
| C6 E17 sources | PASS | `sha256sum reports/E17_sources.md` ×5 → `6a28efc9f60548cacb047a255db3a17600052a386656481ea498672b905b7dd7` every time (report value matches); 1212 entries, all 1212 path+sha256 pairs recomputed from disk: 0 missing, 0 mismatch; `find Results/appendix_cgba_d8only_*/non_targeted -type f \| wc -l` = 1212. |
| C7 lesson draft L28 | PASS | `docs/LESSONS.md` has no L28 (correct: report-only draft); draft text matches the observed defect (g5_check.py:34 `PUBKEY_CGBA = {"CGBA": "CGBA", "CGBA_H": "CGBA_H"}` vs shipped key `CGBA-H`). |
| no comparison between methods | PASS | `grep -niE 'asr\|median\|success rate\|figure' reports/B5_close.md` → no hits outside the G5 gate; no ASR table/figure/median across methods in the report or under `close/`. |
| RESULT block + sha256 values | PASS | Report ends with the RESULT block, verdict `READY-FOR-REVIEW`. My recomputations: HEAD `8cf80567009d7404d5a00f1edcfe5c4a09372f81`, `git describe --tags` `b5a2-go-2026-10-20`, contract `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`, `run_b5_parallel.sh` `50d8941c...40fb`, `close_prompt.txt` `e0f23a639706...d83`, `agent_log.md` `19c3008a...e521`, `contract_partial.md` `abd77e52...f294`, `contract_partial_status.md` `c00545ea...e3e4`, `human_preflight_b5a2.md` `c7bf0baa...4cdf`, alias `5532b173...`, E17 `6a28efc9...` — all identical to the report. All 12 grid PNG sha256 prefixes in the report (e75bb266, 1517f1a4, b71d6e43, 0021626f, 2280052d, 0f380eb8, af72f6ab, 7b532bb9, 2c943223, a8d2fa62, 37cae2fd, ebce5c84) match `sha256sum reports/human_preflight_b5a2_grid_*.png`. |
| agent_log vs report | PASS | `close/agent_log.md` rows C0–C6 map to the report sections; every C1 table value is shown by the logged command's output file `/tmp/b5finish_probe/c1_table.txt` (54 lines) and is reproducible from `parallel.log` + `gates/summary.csv` (I reproduced it). No report table value was found that no logged command and no file under `close/` shows (the 12 PNG sha256 lines are abbreviated in the report but recomputable and matching; no other untraceable value found). |

## Claims in reports/B5_close.md not supported by the evidence

None found beyond R1 below. All quoted verdicts, counts, sha256 values and decisions were reproduced by my own commands.

## Recall entries

| id | criterion | severity | evidence | scope | acceptance |
|---|---|---|---|---|---|
| R1 | lessons ledger (C7 context) | MINOR | `reports/B5_close.md:12-13` lists "Active lessons that apply (docs/LESSONS.md): … L15 … L22 …", but `docs/LESSONS.md` marks both rows status `encoded`, not `active` (`grep -E '^\| L(15|22) ' docs/LESSONS.md` → L15 `encoded`, L22 `encoded`); the active rows are L02-L07, L10-L12, L14, L19-L21, L23-L27 | `reports/B5_close.md` "Active lessons" line only | corrected lessons line in `reports/B5_close.md` (either drop L15/L22 or label the list "active + encoded rows that apply"), every listed id matching its `docs/LESSONS.md` status |

## Deviations from the prompt

- `check_contract.py` was run with `--report /tmp/b5finish_close_review/contract_partial.md --status_out /tmp/b5finish_close_review/contract_partial_status.md` (review-mode rule; the author's run used the `close/` paths required by its own prompt).
- Only the 24 main cells were recounted row-by-row for C1 (the prompt requires ≥ 10 with all TtBA/CGBA_H main cells among them); the 28 e10/e11/e17 cells were checked via `gates/summary.csv` counts, e17 runs CSVs (30/30) and the partial checker.

VERDICT: PASS-WITH-NOTES