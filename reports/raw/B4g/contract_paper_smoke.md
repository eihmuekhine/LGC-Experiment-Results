# Contract check — stage `paper`

Exit code: **1**  (0 = no FAIL)

- roots=['Results/b4e2_smoke_LGC_RC_EGC_resnet50', 'Results/b4e2_smoke_LGC_RC_EGC_vgg19', 'Results/b4e2_smoke_LGC_RC_EGC_vit_b_32', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted', 'Results/b4e2_smoke_ttba_resnet50', 'Results/b4e2_smoke_ttba_vgg19', 'Results/b4e2_smoke_ttba_vit_b_32', 'Results/b4e2_smoke_ttba_resnet50_places365', 'Results/b4e2_smoke_ttba_densenet161_places365', 'Results/b4e2_smoke_ttba_Salman2020Do_R50', 'Results/b4e2_smoke_cgba_resnet50', 'Results/b4e2_smoke_cgba_vgg19', 'Results/b4e2_smoke_cgba_vit_b_32', 'Results/b4e2_smoke_cgba_resnet50_places365', 'Results/b4e2_smoke_cgba_densenet161_places365', 'Results/b4e2_smoke_cgba_Salman2020Do_R50', 'Results/b4e2_smoke_arsopt_resnet50', 'Results/b4e2_smoke_arsopt_vgg19', 'Results/b4e2_smoke_arsopt_vit_b_32'] stage=paper n=3 split_dirs=['splits/imagenet/ttba_github', 'splits/places365', 'splits/robust']
- per_checkpoint files=42 rows=1008; runs rows=126; npz=1008; env files=39

## Checks

| check id | severity | PASS/FAIL/WARN | count | first offending rows (file:row) |
|---|---|---|---|---|
| I01 | FAIL | PASS | 0 | — |
| I02 | FAIL | PASS | 0 | — |
| I03 | FAIL | PASS | 0 | — |
| I04 | FAIL | PASS | 0 | — |
| I05 | FAIL | PASS | 0 | — |
| I06 | FAIL | PASS | 0 | — |
| I07 | FAIL | PASS | 0 | — |
| I08 | FAIL | PASS | 0 | — |
| I09 | WARN | PASS | 0 | — |
| I10 | FAIL | PASS | 0 | — |
| I11 | FAIL | PASS | 0 | — |
| I12 | FAIL | PASS | 0 | — |
| I13 | FAIL | PASS | 0 | — |
| I14 | FAIL | FAIL | 1 | E12_low_query_breadth (CONDITIONAL): inputs absent and not marked CUT |

## Paper elements

| element | status |
|---|---|
| E10_places365 | COMPLETE |
| E11_robust_model | COMPLETE |
| E12_low_query_breadth | INCOMPLETE |
| E13_cost_table | COMPLETE |
| E14_selection_and_sensitivity | COMPLETE |
| E15_fairness_provenance_table | COMPLETE |
| E16_threat_model_table | COMPLETE |
| E17_cgba_official_appendix | COMPLETE |
| E1_main_perceptual_table | COMPLETE |
| E2_l2_table_ttba_format | COMPLETE |
| E3_paired_stats_table | COMPLETE |
| E4_asr_perc | COMPLETE |
| E5_curves | COMPLETE |
| E6_pareto | COMPLETE |
| E7_comparison_grid | COMPLETE |
| E8_low_query_table | COMPLETE |
| E9_ablation_table | COMPLETE |
| E9b_operator_substitution | COMPLETE |
