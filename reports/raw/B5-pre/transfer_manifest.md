# B5-pre transfer manifest (untracked files)

- Generated: 2026-10-19 05:33:21
- HEAD: `6d36c0e0053578ed29b67eef0f4d3be3a74a55c6`
- Remote:
  - `origin	git@github.com:eihmuekhine/LGC-Experiment-Results.git (fetch)`
  - `origin	git@github.com:eihmuekhine/LGC-Experiment-Results.git (push)`

## Method
- Scope: files under the repo root that are **not** tracked by git (`git ls-files`), i.e. the
  complement `find . -type f` minus tracked. Tracked files transfer via `git clone` and are not listed here.
- Grouping: first two path components (first component for single-segment paths).
- Size: `du -sb` apparent bytes (block-based `du -sh` may differ slightly).
- Reference check for NEEDED: name grep across `contract/paper_contract.yaml`, `adapters/*.py`, and manifest files
  (`*.py *.yaml *.json *.txt`, e.g. `reports/B4e2_smoke_roots.txt`, `reports/B5_postfix_roots.txt`,
  `reports/kat_runs_manifest.sha256`).

## Summary

| category | total size | groups |
|---|---|---|
| NEEDED | 821.1 MB | 28 |
| PROVENANCE | 27.9 MB | 1 |
| CACHE | 54.6 MB | 22 |
| STALE | 996.9 MB | 106 |
| UNCLEAR | 65.8 MB | 10 |

## Table

| path | size | category | why |
|---|---|---|---|
| `Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted` | 8.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted` | 8.0 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted` | 8.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted` | 8.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_resnet50` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted` | 8.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted` | 8.0 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_vgg19` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_LGC_RC_EGC_vit_b_32` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_Salman2020Do_R50` | 15.8 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_densenet161_places365` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_resnet50` | 16.0 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_resnet50_places365` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_vgg19` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4e2_smoke_ttba_vit_b_32` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_ARS_OPT_resnet50` | 15.7 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_ARS_OPT_vgg19` | 15.8 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_ARS_OPT_vit_b_32` | 15.9 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_Salman2020Do_R50` | 16.0 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_densenet161_places365` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_resnet50` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_resnet50_places365` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_vgg19` | 16.1 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `Results/b4h_t3_smoke_CGBA_vit_b_32` | 16.2 MB | NEEDED | smoke reference root (reports/B4e2_smoke_roots.txt / B5_postfix_roots.txt; D4.2 HOST-IDENTICAL) |
| `generated/smoke_canary` | 188.0 MB | NEEDED | KAT canary data + smoke reference outputs; adapters/_common.py:244 (KAT_DATA_DIR) + reports/kat_runs_manifest.sha256 |
| `models/imagenet` | 97.8 MB | NEEDED | RobustBench victim weight Salman2020Do_R50.pt; contract constants.victims + adapters/_common.py:95 |
| `models/places365` | 197.8 MB | NEEDED | Places365 victim weights resnet50/densenet161_places365.pth.tar; contract E10 sources + adapters/_common.py:82 |
| `splits/kat` | 1.2 KB | NEEDED | KAT synthetic splits; adapters/diag/kat_linear.py + adapters/_common.py |
| `reports/raw` | 27.9 MB | PROVENANCE | raw phase evidence (md/txt/csv/png/log, each <10 MB) under reports/ |
| `.git` | 75.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.ipynb_checkpoints/run_b5_parallel-checkpoint.sh` | 27.2 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.opencode/.gitignore` | 63.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.opencode/node_modules` | 52.8 MB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.opencode/package-lock.json` | 14.4 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.opencode/package.json` | 65.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.pytest_cache/.gitignore` | 37.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.pytest_cache/CACHEDIR.TAG` | 191.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.pytest_cache/README.md` | 302.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `.pytest_cache/v` | 58.0 B | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `__pycache__/advstore.cpython-312.pyc` | 9.4 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `__pycache__/run_benchmark_33b.cpython-312.pyc` | 132.0 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `adapters/__pycache__` | 150.8 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `adapters/diag` | 224.4 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `adapters/tests` | 59.6 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `contract/.ipynb_checkpoints` | 29.4 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `docs/.ipynb_checkpoints` | 21.8 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `lgc_bench/__pycache__` | 613.8 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `models/__pycache__` | 11.7 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `reports/.ipynb_checkpoints` | 12.2 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `tools/eval` | 504.8 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `ttba_github/__pycache__` | 23.1 KB | CACHE | cache / build artifact; reinstall fresh on target matching env_build38.txt |
| `Results/_bitid_off` | 10.1 KB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/_bitid_on` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_arsopt_resnet50` | 15.8 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_arsopt_vgg19` | 15.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_arsopt_vit_b_32` | 15.9 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_cgba_vgg19` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_cgba_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_egc_vgg19` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_egc_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_ttba_resnet50` | 15.9 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_ttba_vgg19` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4_smoke_ttba_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_H_Salman2020Do_R50` | 7.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_H_densenet161_places365` | 8.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_H_resnet50_places365` | 7.9 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_Salman2020Do_R50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_densenet161_places365` | 8.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_CGBA_resnet50_places365` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted` | 8.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted` | 8.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_densenet161_places365_non_targeted` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_densenet161_places365_targeted` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_resnet50_places365_non_targeted` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_LGC_RC_EGC_resnet50_places365_targeted` | 8.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_TtBA_Salman2020Do_R50` | 15.8 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_TtBA_densenet161_places365` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4b_smoke_TtBA_resnet50_places365` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_egc_generic_resnet50` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_profile_ttba` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_regress_arsopt` | 5.4 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_regress_cgba` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_regress_cgba_h` | 2.6 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4c_regress_ttba` | 5.4 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_conc2_egc_resnet50` | 13.5 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_conc2_ttba` | 13.3 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_conc_egc_resnet50` | 13.5 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_conc_ttba` | 13.3 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_solo_egc_resnet50` | 13.5 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4d_t3_solo_ttba` | 13.3 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_f1_cgba_resnet50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_g1a_cgba_resnet50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_arsopt_resnet50` | 15.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_arsopt_vgg19` | 15.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_arsopt_vit_b_32` | 15.9 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_cgba_Salman2020Do_R50` | 16.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_cgba_densenet161_places365` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_cgba_resnet50_places365` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_cgba_vgg19` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_smoke_cgba_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t1_A2_resnet50` | 17.4 KB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t1_A_resnet50` | 17.4 KB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t1_B_resnet50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t4_egc_resnet50` | 5.4 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t4_ttba_resnet50` | 5.3 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t5_egc_resnet50` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t5_ttba_resnet50` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e2_t6_cgba_resnet50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_arsopt_resnet50` | 15.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_arsopt_vgg19` | 15.8 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_arsopt_vit_b_32` | 15.9 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_Salman2020Do_R50` | 16.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_densenet161_places365` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_resnet50` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_resnet50_places365` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_vgg19` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_cgba_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_Salman2020Do_R50` | 15.8 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_densenet161_places365` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_resnet50` | 16.0 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_resnet50_places365` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_vgg19` | 16.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_smoke_ttba_vit_b_32` | 16.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_r50_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_r50_d12` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_r50places_6000` | 8.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_resnet50` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_resnet50_det` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_resnet50_repeat` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_robust_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_d8only_vgg19_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_fixed_r50_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_fixed_r50places_6000` | 8.2 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_fixed_robust_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_fixed_vgg19_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_t1_fixed_vit_6000` | 8.1 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b4e_test_arsopt_r50` | 2.7 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/b5_e10_TtBA_densenet161_places365_NT` | 3.2 KB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `Results/retired_b4b_egc_per_victim` | 48.5 MB | STALE | superseded pre-B5 smoke/regression/diagnostic result (not in reports/B4e2_smoke_roots.txt) |
| `contract/paper_contract.yaml.bak_20261011_225106` | 15.6 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.bak_20261016_024247` | 25.5 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.bak_rev3` | 26.2 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.bak_rev4_prep` | 26.2 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.bak_v17_20261013_100904` | 20.4 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.bak_v18_20261015_122547` | 23.5 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.v1.5.bak` | 18.1 KB | STALE | superseded contract backup |
| `contract/paper_contract.yaml.v1.6.bak` | 19.3 KB | STALE | superseded contract backup |
| `lgc_preflight_kit_v1.2.zip` | 27.8 KB | STALE | superseded kit zip |
| `logs_frame/b4d_t3_conc_ttba.log` | 17.8 KB | STALE | superseded frame-run log |
| `logs_frame/egc_resnet50_non_targeted.log` | 18.8 KB | STALE | superseded frame-run log |
| `logs_frame/egc_resnet50_targeted.log` | 6.6 KB | STALE | superseded frame-run log |
| `logs_frame/egc_vgg19_non_targeted.log` | 6.4 KB | STALE | superseded frame-run log |
| `logs_frame/egc_vgg19_targeted.log` | 6.5 KB | STALE | superseded frame-run log |
| `logs_frame/egc_vit_b_32_non_targeted.log` | 6.5 KB | STALE | superseded frame-run log |
| `logs_frame/egc_vit_b_32_targeted.log` | 6.5 KB | STALE | superseded frame-run log |
| `logs_frame/timing.txt` | 1.1 KB | STALE | superseded frame-run log |
| `reports/b5_attempt1` | 5.0 KB | STALE | abandoned B5 attempt 1 (reports/b5_attempt1/ABANDONED.md; contract D16) |
| `912.md` | 458.8 KB | UNCLEAR | root-level markdown note (scratch); not referenced by contract/adapters/manifest; not under reports/ |
| `940.md` | 425.0 KB | UNCLEAR | root-level markdown note (scratch); not referenced; not under reports/ |
| `Night1.md` | 408.9 KB | UNCLEAR | root-level markdown note (scratch); not referenced; not under reports/ |
| `Results/b3_ttba_adapter` | 15.9 MB | UNCLEAR | B3 fidelity smoke (superseded) but named in adapters/diag/diag_t4_ttba_completeness.py docstring |
| `Results/b3_ttba_official` | 30.3 KB | UNCLEAR | B3 fidelity smoke (superseded) but named in adapters/diag/diag_t4_ttba_completeness.py docstring |
| `Results/b4_smoke_cgba_resnet50` | 15.9 MB | UNCLEAR | superseded smoke but named in adapters/diag/diag_b4e_t1_compare.py docstring |
| `Results/b4_smoke_egc_resnet50` | 16.2 MB | UNCLEAR | superseded smoke but named in adapters/diag/diag_t1c_compare_egc.py docstring |
| `Results/b4e2_smoke_cgba_resnet50` | 16.1 MB | UNCLEAR | superseded by Results/b4h_t3_smoke_CGBA_resnet50 but named in adapters/diag/diag_b4f_t6_domain.py docstring |
| `opencode.json` | 6.3 KB | UNCLEAR | opencode agent config; not referenced; not experiment data |
| `places.md` | 408.9 KB | UNCLEAR | root-level markdown note (scratch); not referenced; not under reports/ |

## UNCLEAR (flag for H)

The following groups are not obviously NEEDED/PROVENANCE/CACHE/STALE. They are excluded from
`transfer_include.txt` and need a human decision:

- `912.md` (458.8 KB): root-level markdown note (scratch); not referenced by contract/adapters/manifest; not under reports/
- `940.md` (425.0 KB): root-level markdown note (scratch); not referenced; not under reports/
- `Night1.md` (408.9 KB): root-level markdown note (scratch); not referenced; not under reports/
- `Results/b3_ttba_adapter` (15.9 MB): B3 fidelity smoke (superseded) but named in adapters/diag/diag_t4_ttba_completeness.py docstring
- `Results/b3_ttba_official` (30.3 KB): B3 fidelity smoke (superseded) but named in adapters/diag/diag_t4_ttba_completeness.py docstring
- `Results/b4_smoke_cgba_resnet50` (15.9 MB): superseded smoke but named in adapters/diag/diag_b4e_t1_compare.py docstring
- `Results/b4_smoke_egc_resnet50` (16.2 MB): superseded smoke but named in adapters/diag/diag_t1c_compare_egc.py docstring
- `Results/b4e2_smoke_cgba_resnet50` (16.1 MB): superseded by Results/b4h_t3_smoke_CGBA_resnet50 but named in adapters/diag/diag_b4f_t6_domain.py docstring
- `opencode.json` (6.3 KB): opencode agent config; not referenced; not experiment data
- `places.md` (408.9 KB): root-level markdown note (scratch); not referenced; not under reports/

## Notes
- `models/places365` is stored inside the repo but `adapters/_common.py:82` reads
  `/root/autodl-tmp/models/places365`; the two weight files must land at the path the adapter expects.
- `models/imagenet/Linf/Salman2020Do_R50.pt` mirrors the RobustBench cache path
  (`~/.cache/robustbench/models`, `adapters/_common.py:92`); the weight file itself is L1 victim state.
- The 3 `b5host_*` roots in `reports/B4e2_smoke_roots.txt` are git-tracked (or absent) and are not in this
  untracked manifest.
- Nothing was transferred, deleted, or run on GPU in this phase (read-only).
