# B5a2-finish part A report (decisions, close, offline-metrics plan)

Scope: `reports/raw/B5a2/finish/finish_A.txt` (prompt sha256
`352d256d1600e383aff8324b599836883542611758ab4386ac9764c44c944395`). Unattended run.

Environment: HEAD `8cf80567009d7404d5a00f1edcfe5c4a09372f81`, branch `paper-runs`,
`git describe --tags` = `b5a2-go-2026-10-20`; contract v1.9-rev7 sha256
`ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`. One GPU:
NVIDIA GeForce RTX 5090.

Active lessons (docs/LESSONS.md): L02, L03, L04, L05, L06, L07, L10, L11, L12, L14, L15, L19, L20, L21,
L22, L23, L24, L25, L26, L27.

## A0 — preconditions (all PASS)

```
date                          -> Fri Oct 23 08:34:29 CST 2026
git describe --tags           -> b5a2-go-2026-10-20
git rev-parse HEAD            -> 8cf80567009d7404d5a00f1edcfe5c4a09372f81
git status --short | grep -v '^??'  -> (nothing; no tracked change)
sha256sum contract/paper_contract.yaml -> ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
sha256sum run_b5_parallel.sh  -> 50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb
nvidia-smi -L                 -> GPU 0: NVIDIA GeForce RTX 5090
awk 'NR>1' parallel.log | wc -l  -> 52
awk 'NR>1 && $7!=0 {print $1}' | sort -n  -> 26 28 30 31 36 39 40 41 44
ls reports/stop | grep '^B5a2_'  -> 12 bundle files
```

Expected values all match: HEAD 8cf80567..., no tracked change, contract
ce101e33..., 52 job lines, Exitval != 0 exactly for seq 26 28 30 31 36 39 40 41 44, RTX 5090 present.

## A1 — decisions on the 9 stopped cells

Full output: `reports/B5a2_stop_decisions.md` (rule verbatim, sign-convention quote, per-X-row table,
per-cell table, duplicate-bundle finding, deviation note). Summary:

- Margin/class definition: `adapters/diag/domain_consistency.py:12-15` (docstring) and `:73-79` (`margins`),
  `:95-103` (`classify`). Convention: m > 0 means the victim's decision is NOT adversarial.
- 12 STOP bundles for 9 cells. Three cells (main CGBA_H resnet50/vgg19/vit_b_32 T) have two bundles each
  (first = E-G5 `g5 FAIL`, second = E-OTHER `domain FAIL X>0`).
- 16 X rows across the 9 cells, classified STORAGE/DEFECT/OTHER by the H rule.

| cell | rows | C | Q9 | Q6 | X | STORAGE | OTHER | decision |
|---|---|---|---|---|---|---|---|---|
| e11_ARS_OPT_Salman2020Do_R50_T | 1600 | 1001 | 598 | 0 | 1 | 1 | 0 | INCLUDED |
| main_ARS_OPT_vit_b_32_T | 1600 | 1088 | 510 | 0 | 2 | 2 | 0 | INCLUDED |
| main_CGBA_H_vgg19_T | 1600 | 1168 | 429 | 0 | 3 | 2 | 1 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_vgg19_NT | 1599 | 1443 | 154 | 0 | 2 | 2 | 0 | INCLUDED |
| main_CGBA_H_resnet50_T | 1600 | 1085 | 514 | 0 | 1 | 1 | 0 | INCLUDED |
| e11_CGBA_H_Salman2020Do_R50_T | 1600 | 989 | 609 | 0 | 2 | 0 | 2 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_resnet50_NT | 1594 | 1368 | 223 | 0 | 3 | 1 | 2 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_vit_b_32_NT | 1574 | 1429 | 144 | 0 | 1 | 1 | 0 | INCLUDED |
| main_CGBA_H_vit_b_32_T | 1600 | 1282 | 316 | 1 | 1 | 1 | 0 | INCLUDED |

## A2 — phase 8 close

Executed `reports/raw/B5a2/close_prompt.txt` C0-C7 (full record: `reports/B5_close.md`; command log:
`reports/raw/B5a2/close/agent_log.md`). Results:

- C0: no attack process; 52 job lines. PASS.
- C1: 52-row completeness table (all cells reached n; error_rows = 0; nonfinite = 0). PASS.
- C2: G5 at n=200 — all 6 arch/mode checks PASS (TtBA + CGBA-NT on shipped table; CGBA_H on alias table).
  The run-time g5=FAIL/E-G5 rows were a `KeyError 'CGBA_H'` tool crash; the C2 verdicts replace them. PASS.
- C3: no new domain run needed (all 40 non-TtBA domain CSVs present). PASS.
- C4: human_preflight wrote `reports/human_preflight_b5a2.md`; 53 FLAG lines, each explained (env files do
  not record driver/hostname; one GPU-name string difference for the 6 LGC_RC_EGC roots). PASS.
- C5: `check_contract.py --stage partial` over 48 roots -> exit 0, I01-I12 all PASS. PASS.
- C6: `reports/E17_sources.md` (1212 files with sha256). PASS.
- C7: lesson draft L28 (G5 key alias) recorded in `reports/B5_close.md`. PASS.

Close verdict: **READY-FOR-REVIEW** (`reports/B5_close.md`).

## A3 — offline-metrics plan (no paper metric computed in this session)

(a) Inventory: `reports/raw/B5a2/export/inventory.md` (52 directories; counts/bytes of runs CSV,
per_checkpoint CSV, npz, env, run_manifest; adv_store npz 1600 for main/E10/E11 and ~240 for E17; no
non-npz file > 50 MB; no offline-metrics CSV exists yet).

(b) `tools/eval/eval_metrics_offline.py` (`--help` and argparse block at `:202-216`): inputs `--roots`,
`--out`, `--metrics` (default `l2 linf psnr ssim ms_ssim lpips_alex lpips_vgg dists fsim`), optional `--nr`
(brisque/clip_iqa), `--device` (default `cuda`), `--batch` (default 32), `--selftest`; one row per stored
npz; writes only to `--out` (no Results/ restriction) -> no STOP.

(c) Timing probe on smallest E17 directory (`appendix_cgba_d8only_vit_b_32`, no limit option exists):
`python tools/eval/eval_metrics_offline.py --roots Results/appendix_cgba_d8only_vit_b_32 --out
/tmp/b5finish_probe/metrics_probe.csv --device cuda:0 --batch 32` -> 237 stored states, 237 evaluated,
**10.0 s wall**, 237 output rows. `sha256sum -c /root/autodl-tmp/B5a2_manifest_20261021.sha256 | grep -vc
': OK$'` prints **0** (Results/ unchanged).

(d) `reports/raw/B5a2/export/metrics_commands.sh`: 48 command lines (main + E10 + E11, no E17), longest cell
first, each exactly `python tools/eval/eval_metrics_offline.py --roots <dir> --out
paper_exports/b5a2/metrics/<name>/metrics.csv --device cuda:0 --batch 32`; no `; & | < > \` $` in any line;
comment lines start with `#`.

Predicted total wall time (from c): ~8 s model load + ~0.0084 s/state -> 48 × 8 s + 76766 × 0.0084 s ≈
384 s + 645 s ≈ **~17 min** (round: ~18 min).

## Deviations

1. Six INCLUDED cells deviate from the D19 stop rule (X>0 = STOP), per the H decision (decision point 3),
   recorded in `reports/B5a2_stop_decisions.md` and listed in the close note.

## Five-line summary

1. A0 preconditions all PASS (HEAD, contract sha256, 52 job lines, the 9 non-zero-Exitval cells, RTX 5090).
2. A1 applied the H decision rule to the 16 X rows: 3 cells EXCLUDED-FROM-CLAIMS, 6 cells INCLUDED (a
   documented deviation from the D19 stop rule).
3. A2 completed the close C0-C7: G5 re-check all PASS (alias table fixes the CGBA_H KeyError), partial
   checker exit 0, human_preflight FLAGs all explained -> READY-FOR-REVIEW.
4. A3 produced the inventory, documented the offline-metrics tool, ran a 10 s timing probe (237 states,
   Results/ unchanged) and wrote the 48-line metrics_commands.sh.
5. Predicted offline-metrics wall time ~17-18 min; no metric was computed for the paper in this session.

## RESULT

```
RESULT
phase: B5a2-finish-A
head: 8cf80567009d7404d5a00f1edcfe5c4a09372f81
contract_sha256: ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
prompt_sha256: 352d256d1600e383aff8324b599836883542611758ab4386ac9764c44c944395
tasks: A0=PASS A1=PASS A2=PASS A3=PASS
decisions: e11_ARS_OPT_Salman2020Do_R50_T=INCLUDED main_ARS_OPT_vit_b_32_T=INCLUDED main_CGBA_H_vgg19_T=EXCLUDED-FROM-CLAIMS main_CGBA_vgg19_NT=INCLUDED main_CGBA_H_resnet50_T=INCLUDED e11_CGBA_H_Salman2020Do_R50_T=EXCLUDED-FROM-CLAIMS main_CGBA_resnet50_NT=EXCLUDED-FROM-CLAIMS main_CGBA_vit_b_32_NT=INCLUDED main_CGBA_H_vit_b_32_T=INCLUDED
close_verdict: verdict: READY-FOR-REVIEW
files:
  "reports/B5a2_stop_decisions.md c267faedcbd2c479a94e35d95c669b3b7fe2ac21af68cfb45e6b888154aecb03"
  "reports/B5_close.md 3c88e197f1ca5b7788a6025b627198e6b58b450f7b2f7c2ec8241d2523801b60"
  "reports/B5a2_finish_A.md e6f6592ad349e256d2d8a200462258e4b20e56366c4197fae90d1300d545decd"
  "reports/human_preflight_b5a2.md c7bf0baa862afd7826c3a804d8cdc0874d8fddda81de098f6d74466b5c5e4cdf"
  "reports/human_preflight_b5a2_grid_*.png (12) e75bb266... ebce5c84... (see B5_close.md table)"
  "reports/E17_sources.md 6a28efc9f60548cacb047a255db3a17600052a386656481ea498672b905b7dd7"
  "reports/raw/B5a2/close/g5/ttba_table1_alias.json 5532b173f84d61940988d317cc7002de449465538d2d223cfbd81b04c811e558"
  "reports/raw/B5a2/close/contract_partial.md abd77e52d4b393a07cdee08b2b3c54065db7aad6c5bdac4e1b14897eef1af294"
  "reports/raw/B5a2/close/contract_partial_status.md c00545eaea2ab81d0044da59f59abc85436d406b542ca0fd6d73c7b3d89fe3e4"
  "reports/raw/B5a2/close/agent_log.md 19c3008a5b8b940fa409599f5b54dea4680ee0c3c51e8ec6f3ef50b8154de521"
  "reports/raw/B5a2/finish/agent_log_A.md 13a26a673c12e8f2623af820b7d30b07e243eb3ec08d7ee7c2324b453a49ad8f"
  "reports/raw/B5a2/export/inventory.md f468ed19dee2bb8766972d10616e896cdadd84b904f71c6ea89da7cc1b6715aa"
  "reports/raw/B5a2/export/metrics_commands.sh 5b45d559758d9ff3eb121806ece992ec65b509a09b13f8b4c5ba46c7b3a33934"
deviations: 1 (six INCLUDED cells deviate from D19 stop rule, H decision point 3)
stop_bundles: reports/stop/B5a2_e11_ARS_OPT_Salman2020Do_R50_T_20261020-1835.md reports/stop/B5a2_main_ARS_OPT_vit_b_32_T_20261020-1847.md reports/stop/B5a2_main_CGBA_vgg19_NT_20261020-1953.md reports/stop/B5a2_main_CGBA_H_vgg19_T_20261020-1957.md reports/stop/B5a2_main_CGBA_H_vgg19_T_20261020-1957_2.md reports/stop/B5a2_main_CGBA_H_resnet50_T_20261020-2104.md reports/stop/B5a2_main_CGBA_H_resnet50_T_20261020-2104_2.md reports/stop/B5a2_e11_CGBA_H_Salman2020Do_R50_T_20261020-2209.md reports/stop/B5a2_main_CGBA_resnet50_NT_20261020-2215.md reports/stop/B5a2_main_CGBA_vit_b_32_NT_20261020-2216.md reports/stop/B5a2_main_CGBA_H_vit_b_32_T_20261020-2311.md reports/stop/B5a2_main_CGBA_H_vit_b_32_T_20261020-2311_2.md
verdict: READY-FOR-METRICS
```
