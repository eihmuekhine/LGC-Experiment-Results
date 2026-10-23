# B5a2-finish part B — COMMANDS.md (B2 derived outputs)

All commands run from the repository root `/root/autodl-tmp/LGC_frozen` (CPU only).
Every value below was copied from the command output. Tool file sha256 values are listed next to each tool.

## Display-name mappings (spec v3 §7, finish_B.txt B2)

Exactly these three mappings were passed to `--rename` of `make_tables_figures.py` and `make_comparison_grid.py`:
  LGC_RC_EGC=LGC-EGC (ours)  CGBA_H=CGBA-H  ARS_OPT=ARS-OPT
`make_low_query_table.py` has the same `--rename` option; the same three mappings were passed to it.

## 1. make_tables_figures.py  (E1-E8 tables and figures, per (arch, mode))

tool sha256: 86c07e26099cdd66db55d298d49511cd511fac3b179386c57bfc5cc0cf42b0e3
Outputs per (arch, mode): summary_*.json, tab_l2_*.tex, tab_percep_*.tex, tab_paired_*.tex, tab_asrperc_*.tex, fig_curves_* and fig_pareto_* (.pdf/.png).
Statistics note: make_tables_figures.py p-values are two-sided Wilcoxon signed-rank (Holm within baselines) and are DESCRIPTIVE only; the registered claims (one-sided) are computed separately in B3 (claims.json).

Main cells (24 metrics CSVs; 6 (arch, mode)) — exit 0:
```
python tools/eval/make_tables_figures.py --metrics paper_exports/b5a2/metrics/b5_main_ARS_OPT_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vit_b_32_T/metrics.csv --ours LGC_RC_EGC --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --outdir paper_exports/b5a2/derived/make_tables_figures
```
E10 cells (16 metrics CSVs; 4 (arch, mode)) — exit 0:
```
python tools/eval/make_tables_figures.py --metrics paper_exports/b5a2/metrics/b5_e10_ARS_OPT_densenet161_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_ARS_OPT_densenet161_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_ARS_OPT_resnet50_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_ARS_OPT_resnet50_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_CGBA_H_densenet161_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_CGBA_H_resnet50_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_CGBA_densenet161_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_CGBA_resnet50_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_LGC_RC_EGC_densenet161_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_LGC_RC_EGC_densenet161_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_LGC_RC_EGC_resnet50_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_LGC_RC_EGC_resnet50_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_TtBA_densenet161_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_TtBA_densenet161_places365_T/metrics.csv paper_exports/b5a2/metrics/b5_e10_TtBA_resnet50_places365_NT/metrics.csv paper_exports/b5a2/metrics/b5_e10_TtBA_resnet50_places365_T/metrics.csv --ours LGC_RC_EGC --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --outdir paper_exports/b5a2/derived/make_tables_figures
```
E11 cells (8 metrics CSVs; 2 (arch, mode)) — exit 0:
```
python tools/eval/make_tables_figures.py --metrics paper_exports/b5a2/metrics/b5_e11_ARS_OPT_Salman2020Do_R50_NT/metrics.csv paper_exports/b5a2/metrics/b5_e11_ARS_OPT_Salman2020Do_R50_T/metrics.csv paper_exports/b5a2/metrics/b5_e11_CGBA_H_Salman2020Do_R50_T/metrics.csv paper_exports/b5a2/metrics/b5_e11_CGBA_Salman2020Do_R50_NT/metrics.csv paper_exports/b5a2/metrics/b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/metrics.csv paper_exports/b5a2/metrics/b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/metrics.csv paper_exports/b5a2/metrics/b5_e11_TtBA_Salman2020Do_R50_NT/metrics.csv paper_exports/b5a2/metrics/b5_e11_TtBA_Salman2020Do_R50_T/metrics.csv --ours LGC_RC_EGC --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --outdir paper_exports/b5a2/derived/make_tables_figures
```

## 2. make_low_query_table.py  (E8 low-query table, main cells)

tool sha256: 026c4bdd4cb2b7cfd8944d8a108b1e7ecdfa611b17ba9fe246cdfeef46ccae36
Outputs: tab_lowq_<arch>_<mode>.tex (6 files). Exit 0.
```
python tools/eval/make_low_query_table.py --roots Results/b5_main_ARS_OPT_resnet50_NT Results/b5_main_ARS_OPT_resnet50_T Results/b5_main_ARS_OPT_vgg19_NT Results/b5_main_ARS_OPT_vgg19_T Results/b5_main_ARS_OPT_vit_b_32_NT Results/b5_main_ARS_OPT_vit_b_32_T Results/b5_main_CGBA_H_resnet50_T Results/b5_main_CGBA_H_vgg19_T Results/b5_main_CGBA_H_vit_b_32_T Results/b5_main_CGBA_resnet50_NT Results/b5_main_CGBA_vgg19_NT Results/b5_main_CGBA_vit_b_32_NT Results/b5_main_LGC_RC_EGC_resnet50_NT Results/b5_main_LGC_RC_EGC_resnet50_T Results/b5_main_LGC_RC_EGC_vgg19_NT Results/b5_main_LGC_RC_EGC_vgg19_T Results/b5_main_LGC_RC_EGC_vit_b_32_NT Results/b5_main_LGC_RC_EGC_vit_b_32_T Results/b5_main_TtBA_resnet50_NT Results/b5_main_TtBA_resnet50_T Results/b5_main_TtBA_vgg19_NT Results/b5_main_TtBA_vgg19_T Results/b5_main_TtBA_vit_b_32_NT Results/b5_main_TtBA_vit_b_32_T --metrics paper_exports/b5a2/metrics/b5_main_ARS_OPT_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_ARS_OPT_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_H_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_CGBA_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_LGC_RC_EGC_vit_b_32_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_resnet50_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_resnet50_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vgg19_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vgg19_T/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vit_b_32_NT/metrics.csv paper_exports/b5a2/metrics/b5_main_TtBA_vit_b_32_T/metrics.csv --ours LGC_RC_EGC --order LGC_RC_EGC TtBA CGBA CGBA_H ARS_OPT --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --budgets 100 200 500 1000 --rmse 0.1 0.05 0.01 --outdir paper_exports/b5a2/derived/make_low_query_table
```

## 3. make_comparison_grid.py  (E7 grid, main cells, budgets 1000 and 10000, n=3, --select random --seed 0)

tool sha256: e287734598163807aea41c56f4b228c0f7785bab933603f6ab1cde246bd6ee1e
Input metrics: combined per-(arch, mode) CSVs under paper_exports/b5a2/derived/make_comparison_grid/metrics_<arch>_<mode>.csv (concatenation of the 4 method metrics CSVs, dedup on arch/mode/image/method/budget).
Each command writes <out>.pdf, <out>.png and <out>_caption.txt. All 12 exit 0.
```
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_resnet50_NT/non_targeted/adv_store Results/b5_main_TtBA_resnet50_NT/non_targeted/adv_store Results/b5_main_CGBA_resnet50_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_resnet50_NT/non_targeted/adv_store --arch resnet50 --budget 1000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_resnet50_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_resnet50_non_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_resnet50_NT/non_targeted/adv_store Results/b5_main_TtBA_resnet50_NT/non_targeted/adv_store Results/b5_main_CGBA_resnet50_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_resnet50_NT/non_targeted/adv_store --arch resnet50 --budget 10000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_resnet50_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_resnet50_non_targeted_Q10000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_resnet50_T/targeted/adv_store Results/b5_main_TtBA_resnet50_T/targeted/adv_store Results/b5_main_CGBA_H_resnet50_T/targeted/adv_store Results/b5_main_ARS_OPT_resnet50_T/targeted/adv_store --arch resnet50 --budget 1000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_resnet50_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_resnet50_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_resnet50_T/targeted/adv_store Results/b5_main_TtBA_resnet50_T/targeted/adv_store Results/b5_main_CGBA_H_resnet50_T/targeted/adv_store Results/b5_main_ARS_OPT_resnet50_T/targeted/adv_store --arch resnet50 --budget 10000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_resnet50_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_resnet50_targeted_Q10000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vgg19_NT/non_targeted/adv_store Results/b5_main_TtBA_vgg19_NT/non_targeted/adv_store Results/b5_main_CGBA_vgg19_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_vgg19_NT/non_targeted/adv_store --arch vgg19 --budget 1000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vgg19_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vgg19_non_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vgg19_NT/non_targeted/adv_store Results/b5_main_TtBA_vgg19_NT/non_targeted/adv_store Results/b5_main_CGBA_vgg19_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_vgg19_NT/non_targeted/adv_store --arch vgg19 --budget 10000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vgg19_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vgg19_non_targeted_Q10000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vgg19_T/targeted/adv_store Results/b5_main_TtBA_vgg19_T/targeted/adv_store Results/b5_main_CGBA_H_vgg19_T/targeted/adv_store Results/b5_main_ARS_OPT_vgg19_T/targeted/adv_store --arch vgg19 --budget 1000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vgg19_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vgg19_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vgg19_T/targeted/adv_store Results/b5_main_TtBA_vgg19_T/targeted/adv_store Results/b5_main_CGBA_H_vgg19_T/targeted/adv_store Results/b5_main_ARS_OPT_vgg19_T/targeted/adv_store --arch vgg19 --budget 10000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vgg19_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vgg19_targeted_Q10000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted/adv_store Results/b5_main_TtBA_vit_b_32_NT/non_targeted/adv_store Results/b5_main_CGBA_vit_b_32_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_vit_b_32_NT/non_targeted/adv_store --arch vit_b_32 --budget 1000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vit_b_32_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vit_b_32_non_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted/adv_store Results/b5_main_TtBA_vit_b_32_NT/non_targeted/adv_store Results/b5_main_CGBA_vit_b_32_NT/non_targeted/adv_store Results/b5_main_ARS_OPT_vit_b_32_NT/non_targeted/adv_store --arch vit_b_32 --budget 10000 --methods LGC_RC_EGC TtBA CGBA ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vit_b_32_non_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vit_b_32_non_targeted_Q10000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vit_b_32_T/targeted/adv_store Results/b5_main_TtBA_vit_b_32_T/targeted/adv_store Results/b5_main_CGBA_H_vit_b_32_T/targeted/adv_store Results/b5_main_ARS_OPT_vit_b_32_T/targeted/adv_store --arch vit_b_32 --budget 1000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vit_b_32_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vit_b_32_targeted_Q1000
python tools/eval/make_comparison_grid.py --roots Results/b5_main_LGC_RC_EGC_vit_b_32_T/targeted/adv_store Results/b5_main_TtBA_vit_b_32_T/targeted/adv_store Results/b5_main_CGBA_H_vit_b_32_T/targeted/adv_store Results/b5_main_ARS_OPT_vit_b_32_T/targeted/adv_store --arch vit_b_32 --budget 10000 --methods LGC_RC_EGC TtBA CGBA_H ARS_OPT --ours LGC_RC_EGC --rename 'LGC_RC_EGC=LGC-EGC (ours)' 'CGBA_H=CGBA-H' 'ARS_OPT=ARS-OPT' --metrics paper_exports/b5a2/derived/make_comparison_grid/metrics_vit_b_32_targeted.csv --select random --seed 0 --n 3 --out paper_exports/b5a2/derived/make_comparison_grid/fig_examples_vit_b_32_targeted_Q10000
```

## 4. pdftotext display-name check (B2 check)

For every PDF written by make_tables_figures.py or make_comparison_grid.py (36 PDFs):
`pdftotext <pdf> - | grep -c -E 'LGC_RC_EGC|ARS_OPT|CGBA_H'` prints 0 for every PDF (36/36).
For the .tex files written by make_tables_figures.py and make_low_query_table.py (54 tab_*.tex files),
`grep -c -E 'LGC_RC_EGC|ARS_OPT|CGBA_H'` prints 0 for every file.
The provenance table fairness_E15.tex (not a display-name table) prints 3 (it lists the internal method ids by design).

## 5. Fairness table (E15)

lgc-baselines skill (B7 fairness table) is NOT installed in this environment, so the table was generated
directly from the copied env files (env_*.json / env_frame.txt / config_single.json) and run_manifest.json.
Commands (probe scripts under /tmp/b5finish_B/):
```
python /tmp/b5finish_B/fairness.py      # -> paper_exports/b5a2/derived/fairness/fairness_E15.csv (52 cells)
python /tmp/b5finish_B/fairness_tex.py  # -> paper_exports/b5a2/derived/fairness/fairness_E15.tex
```
Result: one GPU type (RTX 5090; 6 LGC_RC_EGC main cells record the string with '(31 GB)' suffix), torch 2.7.1+cu128
in all 52 cells, cudnn_tf32=False, cudnn_deterministic=True, cudnn_benchmark=False in all cells;
oracle_domain clamp_normalized (CGBA/CGBA_H, 16) / clamp01 (ARS_OPT, 12) / n-a (ours + TtBA, 24).

## 6. Cost table input (E13)

```
python /tmp/b5finish_B/cost_table.py   # -> paper_exports/b5a2/derived/cost_table_E13.csv (24 main cells)
```
Per main cell: n_rows=200, wall_time_s median and sum from runs_*.csv (queries median 10000 for baselines, ~9974-9977 for ours).

## 7. E14 dev-overlap recomputation (contract constants.dev_overlap_rows_1_200)

```
python /tmp/b5finish_B/dev_overlap.py  # -> paper_exports/b5a2/derived/dev_overlap_rows.json
```
A row (order < 200) is a dev-overlap row if its source (both modes) or, for targeted, its target appears in
splits/imagenet/ttba_github/overlap_dev_seeds.json (prereg section 7: targeted overlap includes target images).
Counts match the contract constant: resnet50 NT=4 T=6, vgg19 NT=6 T=7, vit_b_32 NT=2 T=4.

## 8. EXCLUDED-FROM-CLAIMS cells

The three EXCLUDED-FROM-CLAIMS cells (main_CGBA_H_vgg19_T, e11_CGBA_H_Salman2020Do_R50_T, main_CGBA_resnet50_NT,
per reports/B5a2_stop_decisions.md) remain in the tables. make_tables_figures.py / make_low_query_table.py have no
flag option, so they are not flagged in the .tex/.pdf outputs; the flag is recorded in reports/B5a2_stop_decisions.md
and excluded from the B3 confirmatory claim families (see claims.json excluded_comparisons).

