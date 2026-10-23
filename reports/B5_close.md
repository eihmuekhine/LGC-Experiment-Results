# B5a2 close report (completeness and gate record of B5 attempt 2)

Scope: B5a2-close tasks C0-C7, executed from `reports/raw/B5a2/close_prompt.txt` (prompt sha256
`e0f23a639706c6b075eb1893bf038c5c832761dce7e6b260fdc3433cd0d8ed83`), with the three additions and the
G5 key-alias instruction from `reports/raw/B5a2/finish/finish_A.txt` part A2.

Environment: `git describe --tags` = `b5a2-go-2026-10-20`; `git rev-parse HEAD` =
`8cf80567009d7404d5a00f1edcfe5c4a09372f81`; contract sha256 =
`ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`; `run_b5_parallel.sh` sha256 =
`50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb`.

Active lessons that apply (docs/LESSONS.md): L02, L03, L04, L05, L06, L07, L10, L11, L12, L14, L15, L19,
L20, L21, L22, L23, L24, L25, L26, L27.

## C0 — no attack process; 52 job lines

`pgrep -af "run_b5_parallel|parallel -j|run_benchmark|run_ttba|run_cgba|run_arsopt"` returned only its own
shell line (no attack process). `reports/raw/B5a2/parallel.log` has 52 job lines (verified in A0:
`awk 'NR>1' reports/raw/B5a2/parallel.log | wc -l` = 52).

## C1 — completeness table (52 rows)

Every cell reached its n (200; E17 = 30). `error_rows = 0` and `nonfinite_sum = 0` for every cell. LGC_RC_EGC
rows use status `ok`; the baselines use `budget` (their "ran to budget" terminal state, cf. L10/D18).

| cell | Exitval | status | error | nonfinite | partial | g5 | domain_x | stop | bundles | n |
|---|---|---|---|---|---|---|---|---|---|---|
| main_TtBA_resnet50_NT | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| main_TtBA_vgg19_NT | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| main_TtBA_vgg19_T | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| e10_TtBA_densenet161_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| main_TtBA_resnet50_T | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| e10_TtBA_densenet161_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| e10_TtBA_resnet50_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| e10_ARS_OPT_densenet161_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e10_TtBA_resnet50_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| main_TtBA_vit_b_32_NT | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| e11_TtBA_Salman2020Do_R50_T | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| e11_TtBA_Salman2020Do_R50_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | — | 0 | — | 200 |
| e10_ARS_OPT_densenet161_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e10_CGBA_densenet161_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e10_CGBA_H_densenet161_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_TtBA_vit_b_32_T | 0 | budget:200 | 0 | 0 | PASS | PASS | — | 0 | — | 200 |
| main_ARS_OPT_vgg19_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_ARS_OPT_vgg19_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e10_LGC_RC_EGC_densenet161_places365_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_ARS_OPT_resnet50_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e10_LGC_RC_EGC_densenet161_places365_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 18 | 0 | — | 200 |
| e10_ARS_OPT_resnet50_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e11_ARS_OPT_Salman2020Do_R50_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_ARS_OPT_resnet50_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e11_ARS_OPT_Salman2020Do_R50_T | 1 | budget:200 | 0 | 0 | PASS | N/A | 1 | 1 | e11_ARS_OPT..._1835 | 200 |
| e10_ARS_OPT_resnet50_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_LGC_RC_EGC_vgg19_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 28 | 0 | — | 200 |
| main_ARS_OPT_vit_b_32_T | 1 | budget:200 | 0 | 0 | PASS | N/A | 2 | 1 | main_ARS_OPT..._1847 | 200 |
| main_CGBA_vgg19_NT | 1 | budget:200 | 0 | 0 | PASS | PASS | 2 | 1 | main_CGBA_vgg19..._1953 | 200 |
| main_CGBA_H_vgg19_T | 1 | budget:200 | 0 | 0 | PASS | FAIL(E-G5) | 3 | 1 | vgg19_T_1957 + _2 | 200 |
| main_ARS_OPT_vit_b_32_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_LGC_RC_EGC_resnet50_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 37 | 0 | — | 200 |
| main_LGC_RC_EGC_vgg19_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 1 | 0 | — | 200 |
| main_CGBA_H_resnet50_T | 1 | budget:200 | 0 | 0 | PASS | FAIL(E-G5) | 1 | 1 | resnet50_T_2104 + _2 | 200 |
| e10_LGC_RC_EGC_resnet50_places365_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 15 | 0 | — | 200 |
| e10_CGBA_H_resnet50_places365_T | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e11_LGC_RC_EGC_Salman2020Do_R50_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 10 | 0 | — | 200 |
| main_LGC_RC_EGC_vit_b_32_T | 0 | ok:200 | 0 | 0 | PASS | N/A | 24 | 0 | — | 200 |
| e11_CGBA_H_Salman2020Do_R50_T | 1 | budget:200 | 0 | 0 | PASS | N/A | 2 | 1 | e11_CGBA_H..._2209 | 200 |
| main_CGBA_resnet50_NT | 1 | budget:200 | 0 | 0 | PASS | PASS | 3 | 1 | main_CGBA_resnet50..._2215 | 200 |
| main_CGBA_vit_b_32_NT | 1 | budget:200 | 0 | 0 | PASS | PASS | 1 | 1 | main_CGBA_vit_b_32..._2216 | 200 |
| e10_CGBA_resnet50_places365_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| main_LGC_RC_EGC_resnet50_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 1 | 0 | — | 200 |
| main_CGBA_H_vit_b_32_T | 1 | budget:200 | 0 | 0 | PASS | FAIL(E-G5) | 1 | 1 | vit_b_32_T_2311 + _2 | 200 |
| e11_CGBA_Salman2020Do_R50_NT | 0 | budget:200 | 0 | 0 | PASS | N/A | 0 | 0 | — | 200 |
| e17_CGBA_D8only_resnet50_NT | 0 | budget:30 | 0 | 0 | SKIP | N/A | 1 | 0 | — | 30 |
| e17_CGBA_D8only_vit_b_32_NT | 0 | budget:30 | 0 | 0 | SKIP | N/A | 1 | 0 | — | 30 |
| e17_CGBA_D8only_vgg19_NT | 0 | budget:30 | 0 | 0 | SKIP | N/A | 0 | 0 | — | 30 |
| e11_LGC_RC_EGC_Salman2020Do_R50_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 1 | 0 | — | 200 |
| e17_CGBA_D8only_Salman2020Do_R50_NT | 0 | budget:30 | 0 | 0 | SKIP | N/A | 0 | 0 | — | 30 |
| main_LGC_RC_EGC_vit_b_32_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 1 | 0 | — | 200 |
| e10_LGC_RC_EGC_resnet50_places365_NT | 0 | ok:200 | 0 | 0 | PASS | N/A | 1 | 0 | — | 200 |

## C2 — G5 at n = 200 (main TtBA / CGBA / CGBA_H)

The run's `summary.csv` g5=FAIL / E-G5 rows for the three CGBA_H cells were a tool crash
(`KeyError 'CGBA_H'`), not a real gate failure. `tools/eval/g5_check.py:34` (`PUBKEY_CGBA`) looks up the
published key `"CGBA_H"`, but `ttba_github/ttba_table1.json` names it `"CGBA-H"`. Per finish_A.txt (iv),
before C2 I wrote `reports/raw/B5a2/close/g5/ttba_table1_alias.json`: a copy of the shipped table in which,
for every arch under `table1` that has `CGBA-H`, the key `CGBA_H` is added equal to that arch's `CGBA-H`
value; nothing else changes. Verification (python script) printed: only `CGBA_H` added for
resnet50/vgg19/vit_b_32/efficientnet_b0/inception_v3; all shipped keys/values unchanged; `CGBA_H == CGBA-H`
for every arch; sha256 shipped = `2cb224f682f9f5735c219a3adeb8b513cbec033c903d4be14c864af682d9059e`, alias =
`5532b173f84d61940988d317cc7002de449465538d2d223cfbd81b04c811e558`.

g5_check.py was run with the shipped table for the TtBA / CGBA (NT) cells and with the alias table for the
three CGBA_H cells. Every verdict quoted:

- resnet50 non_targeted (TtBA NT + CGBA NT, shipped table): `VERDICT: PASS (all CIs intersect [0.5, 2] at every budget)` (exit 0)
- vgg19 non_targeted (TtBA NT + CGBA NT, shipped table): `VERDICT: PASS` (exit 0)
- vit_b_32 non_targeted (TtBA NT + CGBA NT, shipped table): `VERDICT: PASS` (exit 0)
- resnet50 targeted (TtBA T + CGBA_H T, alias table): `VERDICT: PASS` (exit 0)
- vgg19 targeted (TtBA T + CGBA_H T, alias table): `VERDICT: PASS` (exit 0)
- vit_b_32 targeted (TtBA T + CGBA_H T, alias table): `VERDICT: PASS` (exit 0)

No cell remains PENDING; no G5 verdict is FAIL or FLAG. These C2 verdicts replace the three run-time
g5=FAIL / E-G5 rows.

## C3 — domain CSVs

Every non-TtBA cell already has a domain CSV under `reports/raw/B5a2/gates/` (40 of 40; TtBA cells are
`domain=SKIP`). No `domain_consistency.py` run was required. X counts (from `summary.csv` `domain_x`):
ARS_OPT / CGBA / CGBA_H X>0 for exactly the 9 stopped cells (see A1 / `reports/B5a2_stop_decisions.md`).
LGC_RC_EGC X is reported (not a stop rule): counts 1, 1, 1, 1, 10, 15, 18, 24, 28, 37. CGBA_D8only X: 0, 0, 1, 1.

## C4 — human preflight (`--require_host`)

`python tools/eval/human_preflight.py --roots <52 dirs> --out reports/human_preflight_b5a2 --require_host`
wrote `reports/human_preflight_b5a2.md` (FLAG lines = 53). Explanation of every FLAG line:

- 52 × "driver and hostname not recorded (--require_host)": by design the env files record GPU name, torch,
  backend flags and split sha256 but not driver version or hostname (the report's own note, line 121). The
  single host and single driver for the run are recorded in `reports/raw/B5a2/signoff_values.txt`
  (S6: driver `595.71.05`; S1/S6: 5× NVIDIA GeForce RTX 5090 on one host).
- 1 × "FLAG: more than one value across env files" (GPU names): the 6 LGC_RC_EGC roots record the device
  string `NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True` (extra fields embedded), while the
  46 baseline roots record `NVIDIA GeForce RTX 5090`. Same physical GPU.

Sections 2-7 raise no additional FLAG: x0 identity has 0 multi-hash (arch,image) keys; L2 recompute and
npz `l2` agree (rel err < 1e-3) on every sampled row; no flat/nonmono stall flags; no degeneracy flags.

## C5 — contract checker (partial)

`check_contract.py --stage partial` over the 48 main/E10/E11 roots: **exit code 0** (no FAIL). All of
I01-I12 PASS (count=0); I09 WARN count=0. Report: `reports/raw/B5a2/close/contract_partial.md`; status:
`reports/raw/B5a2/close/contract_partial_status.md`.

## C6 — E17 sources

`reports/E17_sources.md` records path + sha256 of all 1212 files under the four
`Results/appendix_cgba_d8only_<arch>/non_targeted` directories.

## C7 — lesson draft (report only; not written to docs/LESSONS.md)

Draft L28 (G5 key alias, decided 2026-10-22 after attempt 3 stopped at C2):

| id | date | area | severity | lesson | evidence | mitigation | status |
|---|---|---|---|---|---|---|---|
| L28 | 2026-10-22 | harness | minor | `tools/eval/g5_check.py:34` looks up the published key `"CGBA_H"`, but `ttba_github/ttba_table1.json` names it `"CGBA-H"`, so the tool raises `KeyError 'CGBA_H'` for every CGBA_H cell. | B5a2 run: g5=FAIL/E-G5 for the three CGBA_H cells (tool crash); C2 rerun with an alias table -> all PASS | never edit the tool; write an alias table (`CGBA_H = CGBA-H`) for the CGBA_H G5 check (finish_A.txt (iv)) | proposed |

## STOP bundles and the H decision

12 STOP bundles exist for the 9 stopped cells (Exitval != 0: seq 26, 28, 30, 31, 36, 39, 40, 41, 44). Three
cells have two bundles each (first = E-G5, second = E-OTHER domain X>0): `main_CGBA_H_resnet50_T`,
`main_CGBA_H_vgg19_T`, `main_CGBA_H_vit_b_32_T`. The H decision for every STOP bundle of the run is
`reports/B5a2_stop_decisions.md` (decision rule fixed by H on 2026-10-22 before any outcome statistic was
computed): 6 cells INCLUDED, 3 cells EXCLUDED-FROM-CLAIMS. Full per-X-row classification and per-cell
decision are in that file.

## Deviations from the prompt

1. Six INCLUDED cells deviate from the D19 stop rule (`X > 0` = STOP), per the H decision (decision point 3),
   listed here as required by finish_A.txt A1.

## Created files

| path | sha256 |
|---|---|
| reports/E17_sources.md | 6a28efc9f60548cacb047a255db3a17600052a386656481ea498672b905b7dd7 |
| reports/human_preflight_b5a2.md | c7bf0baa862afd7826c3a804d8cdc0874d8fddda81de098f6d74466b5c5e4cdf |
| reports/human_preflight_b5a2_grid_*.png (12) | see sha256sum below |
| reports/raw/B5a2/close/g5/ttba_table1_alias.json | 5532b173f84d61940988d317cc7002de449465538d2d223cfbd81b04c811e558 |
| reports/raw/B5a2/close/contract_partial.md | abd77e52d4b393a07cdee08b2b3c54065db7aad6c5bdac4e1b14897eef1af294 |
| reports/raw/B5a2/close/contract_partial_status.md | c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4 |
| reports/raw/B5a2/close/agent_log.md | 19c3008a5b8b940fa409599f5b54dea4680ee0c3c51e8ec6f3ef50b8154de521 |

Grid PNG sha256 (12): Salman2020Do_R50_non_targeted `e75bb266...`; Salman2020Do_R50_targeted `1517f1a4...`;
densenet161_places365_non_targeted `b71d6e43...`; densenet161_places365_targeted `0021626f...`;
resnet50_non_targeted `2280052d...`; resnet50_places365_non_targeted `0f380eb8...`;
resnet50_places365_targeted `af72f6ab...`; resnet50_targeted `7b532bb9...`; vgg19_non_targeted `2c943223...`;
vgg19_targeted `a8d2fa62...`; vit_b_32_non_targeted `37cae2fd...`; vit_b_32_targeted `ebce5c84...`.

## Five-line summary

1. All 52 B5a2 jobs ended; 52/52 cells reached their n (200; E17 30), with zero error rows and zero
   non-finite queries.
2. C1 completeness table is consistent with `summary.csv`; the 9 non-zero Exitval cells map to the 12 STOP
   bundles.
3. C2 G5 rerun (shipped table for TtBA/CGBA-NT, alias table for CGBA_H) gives PASS for all 6 arch/mode
   checks, replacing the three run-time g5=FAIL/E-G5 rows that were a `KeyError 'CGBA_H'` tool crash.
4. C3 needs no new domain run (all 40 non-TtBA domain CSVs present); C4 human_preflight FLAGs (53) are all
   explained (no driver/hostname in env files; one GPU name string difference); C5 partial checker exit 0.
5. H decision on the 9 stopped cells: 6 INCLUDED, 3 EXCLUDED-FROM-CLAIMS (`reports/B5a2_stop_decisions.md`).

## RESULT

```
RESULT
phase: B5a2-close
head: 8cf80567009d7404d5a00f1edcfe5c4a09372f81
contract_sha256: ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
prompt_sha256: e0f23a639706c6b075eb1893bf038c5c832761dce7e6b260fdc3433cd0d8ed83
tasks: C0=PASS C1=PASS C2=PASS C3=PASS C4=PASS C5=PASS C6=PASS C7=PASS
files:
  "reports/E17_sources.md 6a28efc9f60548cacb047a255db3a17600052a386656481ea498672b905b7dd7"
  "reports/human_preflight_b5a2.md c7bf0baa862afd7826c3a804d8cdc0874d8fddda81de098f6d74466b5c5e4cdf"
  "reports/human_preflight_b5a2_grid_*.png (12) e75bb266... ebce5c84... (see table)"
  "reports/raw/B5a2/close/g5/ttba_table1_alias.json 5532b173f84d61940988d317cc7002de449465538d2d223cfbd81b04c811e558"
  "reports/raw/B5a2/close/contract_partial.md abd77e52d4b393a07cdee08b2b3c54065db7aad6c5bdac4e1b14897eef1af294"
  "reports/raw/B5a2/close/contract_partial_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
  "reports/raw/B5a2/close/agent_log.md 19c3008a5b8b940fa409599f5b54dea4680ee0c3c51e8ec6f3ef50b8154de521"
deviations: 1 (six INCLUDED cells deviate from D19 stop rule, H decision)
stop_bundles: reports/stop/B5a2_e11_ARS_OPT_Salman2020Do_R50_T_20261020-1835.md
              reports/stop/B5a2_main_ARS_OPT_vit_b_32_T_20261020-1847.md
              reports/stop/B5a2_main_CGBA_vgg19_NT_20261020-1953.md
              reports/stop/B5a2_main_CGBA_H_vgg19_T_20261020-1957.md
              reports/stop/B5a2_main_CGBA_H_vgg19_T_20261020-1957_2.md
              reports/stop/B5a2_main_CGBA_H_resnet50_T_20261020-2104.md
              reports/stop/B5a2_main_CGBA_H_resnet50_T_20261020-2104_2.md
              reports/stop/B5a2_e11_CGBA_H_Salman2020Do_R50_T_20261020-2209.md
              reports/stop/B5a2_main_CGBA_resnet50_NT_20261020-2215.md
              reports/stop/B5a2_main_CGBA_vit_b_32_NT_20261020-2216.md
              reports/stop/B5a2_main_CGBA_H_vit_b_32_T_20261020-2311.md
              reports/stop/B5a2_main_CGBA_H_vit_b_32_T_20261020-2311_2.md
verdict: READY-FOR-REVIEW
```
