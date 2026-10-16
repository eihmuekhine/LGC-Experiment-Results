# Contract check — stage `smoke`

Exit code: **0**  (0 = no FAIL)

- roots=['Results/b4b_smoke_LGC_RC_EGC_resnet50_places365', 'Results/b4b_smoke_LGC_RC_EGC_densenet161_places365', 'Results/b4b_smoke_LGC_RC_EGC_Salman2020Do_R50', 'Results/b4b_smoke_TtBA_resnet50_places365', 'Results/b4b_smoke_TtBA_densenet161_places365', 'Results/b4b_smoke_TtBA_Salman2020Do_R50', 'Results/b4b_smoke_CGBA_resnet50_places365', 'Results/b4b_smoke_CGBA_densenet161_places365', 'Results/b4b_smoke_CGBA_Salman2020Do_R50', 'Results/b4b_smoke_CGBA_H_resnet50_places365', 'Results/b4b_smoke_CGBA_H_densenet161_places365', 'Results/b4b_smoke_CGBA_H_Salman2020Do_R50'] stage=smoke n=3 split_dirs=['splits/imagenet/ttba_github', 'splits/places365', 'splits/robust']
- per_checkpoint files=18 rows=432; runs rows=54; npz=432; env files=15

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
| I09 | WARN | WARN | 6 | (Salman2020Do_R50,targeted): found=0 at budget 100: 0/9, (Salman2020Do_R50,targeted): found=0 at budget 200: 0/9, (densenet161_places365,targeted): found=0 at budget 100: 0/9, (densenet161_places365,targeted): found=0 at budget 200: 0/9, (resnet50_places365,targeted): found=0 at budget 100: 0/9, (resnet50_places365,targeted): found=0 at budget 200: 0/9 |
| I10 | FAIL | PASS | 0 | — |
| I11 | FAIL | PASS | 0 | — |
| I12 | FAIL | PASS | 0 | — |
| DISK | FAIL | PASS | 0 | 37.8 GB free |
