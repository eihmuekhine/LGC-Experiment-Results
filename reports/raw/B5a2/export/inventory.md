# B5a2 output-directory inventory (A3a)

52 output directories. Per directory: counts/bytes of runs CSV, per_checkpoint CSV, npz, env files, run_manifest, and other files; adv_store npz count (the files `eval_metrics_offline.py` reads); every non-npz file > 50 MB; and whether an offline-metrics CSV already exists.

| directory | runs_csv | per_ckpt_csv | npz (adv_store) | env | manifest | non-npz>50MB |
|---|---|---|---|---|---|---|
| Results/b5_e10_ARS_OPT_densenet161_places365_NT | 1 (19064 B) | 1 (203396 B) | 1800 (adv_store 1600) (444499461 B) | 1 | 1 | - |
| Results/b5_e10_ARS_OPT_densenet161_places365_T | 1 (18058 B) | 1 (199772 B) | 1800 (adv_store 1600) (423247931 B) | 1 | 1 | - |
| Results/b5_e10_ARS_OPT_resnet50_places365_NT | 1 (18462 B) | 1 (198449 B) | 1800 (adv_store 1600) (444777320 B) | 1 | 1 | - |
| Results/b5_e10_ARS_OPT_resnet50_places365_T | 1 (17459 B) | 1 (194875 B) | 1800 (adv_store 1600) (423001535 B) | 1 | 1 | - |
| Results/b5_e10_CGBA_H_densenet161_places365_T | 1 (18255 B) | 1 (198739 B) | 1800 (adv_store 1600) (434809499 B) | 1 | 1 | - |
| Results/b5_e10_CGBA_H_resnet50_places365_T | 1 (17656 B) | 1 (193764 B) | 1800 (adv_store 1600) (433948391 B) | 1 | 1 | - |
| Results/b5_e10_CGBA_densenet161_places365_NT | 1 (18674 B) | 1 (199624 B) | 1799 (adv_store 1599) (445414735 B) | 1 | 1 | - |
| Results/b5_e10_CGBA_resnet50_places365_NT | 1 (18071 B) | 1 (194547 B) | 1800 (adv_store 1600) (446892498 B) | 1 | 1 | - |
| Results/b5_e10_LGC_RC_EGC_densenet161_places365_NT | 1 (23483 B) | 1 (458993 B) | 1600 (adv_store 1600) (443639051 B) | 1 | 0 | - |
| Results/b5_e10_LGC_RC_EGC_densenet161_places365_T | 1 (22677 B) | 1 (450826 B) | 1600 (adv_store 1600) (436396045 B) | 1 | 0 | - |
| Results/b5_e10_LGC_RC_EGC_resnet50_places365_NT | 1 (22890 B) | 1 (453677 B) | 1600 (adv_store 1600) (444752600 B) | 1 | 0 | - |
| Results/b5_e10_LGC_RC_EGC_resnet50_places365_T | 1 (22086 B) | 1 (445723 B) | 1600 (adv_store 1600) (436075949 B) | 1 | 0 | - |
| Results/b5_e10_TtBA_densenet161_places365_NT | 1 (22900 B) | 1 (272396 B) | 1800 (adv_store 1600) (442656261 B) | 1 | 1 | - |
| Results/b5_e10_TtBA_densenet161_places365_T | 1 (22103 B) | 1 (267832 B) | 1800 (adv_store 1600) (434910722 B) | 1 | 1 | - |
| Results/b5_e10_TtBA_resnet50_places365_NT | 1 (22315 B) | 1 (267384 B) | 1800 (adv_store 1600) (443742964 B) | 1 | 1 | - |
| Results/b5_e10_TtBA_resnet50_places365_T | 1 (21510 B) | 1 (262956 B) | 1800 (adv_store 1600) (434610646 B) | 1 | 1 | - |
| Results/b5_e11_ARS_OPT_Salman2020Do_R50_NT | 1 (18254 B) | 1 (196638 B) | 1800 (adv_store 1600) (442952993 B) | 1 | 1 | - |
| Results/b5_e11_ARS_OPT_Salman2020Do_R50_T | 1 (17265 B) | 1 (193483 B) | 1800 (adv_store 1600) (416881817 B) | 1 | 1 | - |
| Results/b5_e11_CGBA_H_Salman2020Do_R50_T | 1 (17340 B) | 1 (192296 B) | 1800 (adv_store 1600) (431759659 B) | 1 | 1 | - |
| Results/b5_e11_CGBA_Salman2020Do_R50_NT | 1 (17866 B) | 1 (192168 B) | 1800 (adv_store 1600) (447648869 B) | 1 | 1 | - |
| Results/b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT | 1 (22689 B) | 1 (450702 B) | 1600 (adv_store 1600) (445489130 B) | 1 | 0 | - |
| Results/b5_e11_LGC_RC_EGC_Salman2020Do_R50_T | 1 (21886 B) | 1 (444215 B) | 1600 (adv_store 1600) (432268221 B) | 1 | 0 | - |
| Results/b5_e11_TtBA_Salman2020Do_R50_NT | 1 (22295 B) | 1 (264799 B) | 1800 (adv_store 1600) (442183924 B) | 1 | 1 | - |
| Results/b5_e11_TtBA_Salman2020Do_R50_T | 1 (21513 B) | 1 (261456 B) | 1800 (adv_store 1600) (432357246 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_resnet50_NT | 1 (16619 B) | 1 (183664 B) | 1800 (adv_store 1600) (442567440 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_resnet50_T | 1 (15656 B) | 1 (181297 B) | 1800 (adv_store 1600) (419465572 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_vgg19_NT | 1 (16026 B) | 1 (179845 B) | 1800 (adv_store 1600) (444745131 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_vgg19_T | 1 (15053 B) | 1 (176464 B) | 1800 (adv_store 1600) (421483207 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_vit_b_32_NT | 1 (16647 B) | 1 (183085 B) | 1800 (adv_store 1600) (440835239 B) | 1 | 1 | - |
| Results/b5_main_ARS_OPT_vit_b_32_T | 1 (15654 B) | 1 (181528 B) | 1800 (adv_store 1600) (421036048 B) | 1 | 1 | - |
| Results/b5_main_CGBA_H_resnet50_T | 1 (15672 B) | 1 (180115 B) | 1800 (adv_store 1600) (432604693 B) | 1 | 1 | - |
| Results/b5_main_CGBA_H_vgg19_T | 1 (15087 B) | 1 (175335 B) | 1800 (adv_store 1600) (431038706 B) | 1 | 1 | - |
| Results/b5_main_CGBA_H_vit_b_32_T | 1 (15724 B) | 1 (180467 B) | 1800 (adv_store 1600) (435744256 B) | 1 | 1 | - |
| Results/b5_main_CGBA_resnet50_NT | 1 (16257 B) | 1 (180061 B) | 1794 (adv_store 1594) (444828705 B) | 1 | 1 | - |
| Results/b5_main_CGBA_vgg19_NT | 1 (15640 B) | 1 (175954 B) | 1799 (adv_store 1599) (446539800 B) | 1 | 1 | - |
| Results/b5_main_CGBA_vit_b_32_NT | 1 (16258 B) | 1 (179417 B) | 1774 (adv_store 1574) (438040736 B) | 1 | 1 | - |
| Results/b5_main_LGC_RC_EGC_resnet50_NT | 1 (140387 B) | 1 (438547 B) | 1600 (adv_store 1600) (444387747 B) | 1 | 0 | - |
| Results/b5_main_LGC_RC_EGC_resnet50_T | 1 (130571 B) | 1 (431478 B) | 1600 (adv_store 1600) (435049280 B) | 1 | 0 | - |
| Results/b5_main_LGC_RC_EGC_vgg19_NT | 1 (139262 B) | 1 (435098 B) | 1600 (adv_store 1600) (445881787 B) | 1 | 0 | - |
| Results/b5_main_LGC_RC_EGC_vgg19_T | 1 (129611 B) | 1 (426784 B) | 1600 (adv_store 1600) (433887692 B) | 1 | 0 | - |
| Results/b5_main_LGC_RC_EGC_vit_b_32_NT | 1 (140233 B) | 1 (440162 B) | 1600 (adv_store 1600) (443954316 B) | 1 | 0 | - |
| Results/b5_main_LGC_RC_EGC_vit_b_32_T | 1 (137478 B) | 1 (432818 B) | 1600 (adv_store 1600) (436351530 B) | 1 | 0 | - |
| Results/b5_main_TtBA_resnet50_NT | 1 (19892 B) | 1 (246610 B) | 1800 (adv_store 1600) (443091290 B) | 1 | 1 | - |
| Results/b5_main_TtBA_resnet50_T | 1 (19082 B) | 1 (242748 B) | 1800 (adv_store 1600) (433580609 B) | 1 | 1 | - |
| Results/b5_main_TtBA_vgg19_NT | 1 (19281 B) | 1 (242347 B) | 1800 (adv_store 1600) (444364667 B) | 1 | 1 | - |
| Results/b5_main_TtBA_vgg19_T | 1 (18490 B) | 1 (238092 B) | 1800 (adv_store 1600) (431923193 B) | 1 | 1 | - |
| Results/b5_main_TtBA_vit_b_32_NT | 1 (19894 B) | 1 (246878 B) | 1800 (adv_store 1600) (442392212 B) | 1 | 1 | - |
| Results/b5_main_TtBA_vit_b_32_T | 1 (19091 B) | 1 (243234 B) | 1800 (adv_store 1600) (435399200 B) | 1 | 1 | - |
| Results/appendix_cgba_d8only_Salman2020Do_R50 | 1 (2976 B) | 1 (30597 B) | 270 (adv_store 240) (67184688 B) | 1 | 1 | - |
| Results/appendix_cgba_d8only_resnet50 | 1 (2739 B) | 1 (28814 B) | 269 (adv_store 239) (66638225 B) | 1 | 1 | - |
| Results/appendix_cgba_d8only_vgg19 | 1 (2646 B) | 1 (28193 B) | 270 (adv_store 240) (67095008 B) | 1 | 1 | - |
| Results/appendix_cgba_d8only_vit_b_32 | 1 (2738 B) | 1 (28699 B) | 267 (adv_store 237) (66224942 B) | 1 | 1 | - |

Notes:
- Baseline cells (TtBA/ARS_OPT/CGBA/CGBA_H/CGBA_D8only) also store per-image trace npz under `traces/`; the npz count above is total (adv_store + traces). `eval_metrics_offline.py` reads only `adv_store/`, so the metrics-relevant count is the `adv_store` column (1600 for main/E10/E11; ~240 for E17).
- LGC_RC_EGC cells store 1600 npz (adv_store only, no traces).
- No non-npz file exceeds 50 MB in any directory.
- No offline-metrics CSV exists under `Results/` or `paper_exports/` (paper_exports/ does not exist yet).
