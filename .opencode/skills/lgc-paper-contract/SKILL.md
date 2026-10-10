---
name: lgc-paper-contract
description: Binding paper-to-experiment contract for the LGC-EGC paper. Load it before writing or changing any runner, adapter, result writer, metric script, table/figure generator or manuscript Experiments text, and at every gate (B4 smoke, B5 progress, B6 integrity, P5 tables, R2/R5/R7 manuscript). It guarantees that every checkpoint, file (CSV/npz), metric and paper element promised in the plan is actually produced, before expensive runs start.
license: proprietary
compatibility: opencode
metadata:
  project: paper-one-revised
  version: "1.0"
  updated: "2026-10-10"
---

# Paper ↔ experiment contract (fail fast, never after 200 images)

## 1. Rules
1. **One source of truth.** `contract/paper_contract.yaml` (copied from the kit into `/root/autodl-tmp/LGC_frozen/contract/`, committed) defines:
   - budgets and checkpoints;
   - methods and their checkpoint convention;
   - required files, columns and npz keys;
   - invariants I01–I15;
   - every paper element E1–E16.

   Plans, skills and prompts that disagree with it are wrong. Report the conflict; do not "fix" the contract.
2. **Agents never edit the contract.** Only the user changes it (new `version` + `changelog` line). Stop and ask if a requirement cannot be met.
3. **No paper element without data, no data without a paper element.** A table/figure/number not in `paper_elements` must not be produced. A REQUIRED element whose inputs fail the checker blocks the paper. A CONDITIONAL element is either complete or marked CUT in `reports/contract_status.md` (its section is removed — never filled with legacy numbers).
4. **The checker decides, not the agent.** Every gate below runs `tools/eval/check_contract.py`. A FAIL stops the phase. Report the failing check ids and the first 10 offending rows. Do not continue, do not patch result files by hand, do not re-run selectively.
5. **Never weaken a check to make it pass.** Changing a threshold or severity is a contract change (rule 2).

## 2. Checker specification (`tools/eval/check_contract.py`, CPU only, stdlib + numpy + pandas + PyYAML)
CLI:
```
python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage {smoke,partial,final,paper,manuscript} \
    --roots <out dirs> [--column_map contract/column_map.yaml] [--n N] [--metrics_csv path] [--paper_dir paper/v3] \
    --report reports/contract_<stage>.md
```
- **`column_map.yaml`** (written in C1 by reading the writers' code, never guessed) maps each producer's real column names to the contract names, e.g. the EGC runner's names → `image, method, budget, found, state_query, pix_l2`. It is the only place where names may differ.
- **Stages:**
  - `smoke`: I01–I12 on the B4 rows 1–3 (`--n 3`); also checks disk ≥ 25 GB free.
  - `partial`: I01, I04–I08, I11 on completed images only. Missing images are not errors; count them.
  - `final`: I01–I12 with `--n 200`.
  - `paper`: `final` + I13 + I14 on the metrics CSV and P5 outputs.
  - `manuscript`: I14 + I15 on `paper/v3` and `generated/`.
- **Output:**
  - a Markdown table `check id | severity | PASS/FAIL/WARN | count | first offending rows (file:row)`;
  - exit code 1 on any FAIL;
  - `reports/contract_status.md` listing each E-element as COMPLETE / INCOMPLETE (missing inputs) / CUT.
- **Read-only:** it never writes into result folders.

**Acceptance tests** (`tools/eval/tests_contract/`, synthetic data, CPU, < 1 min; each must produce exactly the stated verdict):

| test | synthetic defect | expected |
|---|---|---|
| T1 | complete 3-image × 2-method cell | all PASS |
| T2 | one image lacks budget 500 | I01 FAIL naming that image |
| T3 | one npz missing for a found row | I05 FAIL |
| T4 | npz `l2` differs from `pix_l2` by 1% | I06 FAIL |
| T5 | `state_query` 10050 at budget 10000 | I04 FAIL |
| T6 | an image not in rows 1–3 of the split | I02 FAIL |
| T7 | tracker-best method with L2 increasing 3.0 → 3.2 | I08 FAIL |
| T8 | engine-state method with L2 increasing | I08 WARN only |
| T9 | `x0_mismatch.csv` present | I07 FAIL |
| T10 | env split sha256 differs | I10 FAIL |
| T11 | `ADV_STORE_BUDGETS=1000 10000` recorded | I11 FAIL |
| T12 | duplicate (image, method, budget) row | I01 FAIL |
| T13 | numbers.tex has `\NumX` = `??`, used in a .tex | I15 FAIL |
| T14 | E10 inputs absent and not marked CUT | I14 FAIL; marked CUT → PASS |

## 3. Gates (where the checker runs)

| phase | stage | blocks |
|---|---|---|
| C1 (new, CPU) | implement checker + tests T1–T14 + `column_map.yaml` | everything after B3 |
| B2 | the adapters must write the artefacts of `artefacts` exactly; `@checker` reviews against this skill | B3 |
| B4 smoke (rows 1–3, every method × victim × mode) | `smoke` | B5 (no 200-image run before the smoke passes for that method) |
| B5 | `partial` every 10 images per cell | that cell (stop it on FAIL) |
| B6 | `final` | P5 |
| P5 | `paper` (after `eval_metrics_offline.py` and `make_tables_figures.py`) | R5 |
| R2 | `numbers.tex` stubs are generated from E-elements (one macro family per element), not invented | R3 |
| R5 / R7 | `manuscript` | submission |

## 4. What each element needs (summary; the YAML is binding)
- E1 perceptual table @10k.
- E2 L2 mean (median) @2k/5k/10k + ASR (TtBA format, † block).
- E3 paired stats (AUC + 10k).
- E4 ASR_perc table / curve / threshold sweep.
- E5 curves at the 8 checkpoints.
- E6 Pareto @10k.
- E7 comparison grid @1k/10k from `adv_store`.
- E8 low-query @500/1k.
- E9 ablations (confirm_37 data, labelled).
- E9b–E12 CONDITIONAL.
- E13 cost.
- E14 selection + sensitivity.
- E15 fairness table.
- E16 threat-model table.

Together E1–E8 use every one of the 8 checkpoints (100–10k) and the npz store; a single missing checkpoint or npz breaks a REQUIRED element, which is why I01/I05/I11 are FAIL-level already at the 3-image smoke.
