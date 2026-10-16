# Contract check — stage `smoke`

Exit code: **0**  (0 = no FAIL)

- roots=['Results/b4_smoke_egc_resnet50', 'Results/b4_smoke_egc_vgg19', 'Results/b4_smoke_egc_vit_b_32', 'Results/b4_smoke_ttba_resnet50', 'Results/b4_smoke_ttba_vgg19', 'Results/b4_smoke_ttba_vit_b_32', 'Results/b4_smoke_cgba_resnet50', 'Results/b4_smoke_cgba_vgg19', 'Results/b4_smoke_cgba_vit_b_32', 'Results/b4_smoke_arsopt_resnet50', 'Results/b4_smoke_arsopt_vgg19', 'Results/b4_smoke_arsopt_vit_b_32'] stage=smoke n=3
- per_checkpoint files=24 rows=576; runs rows=72; npz=576; env files=21

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
| I09 | WARN | WARN | 6 | (resnet50,targeted): found=0 at budget 100: 0/12, (resnet50,targeted): found=0 at budget 200: 0/12, (vgg19,targeted): found=0 at budget 100: 0/12, (vgg19,targeted): found=0 at budget 200: 0/12, (vit_b_32,targeted): found=0 at budget 100: 0/12, (vit_b_32,targeted): found=0 at budget 200: 0/12 |
| I10 | FAIL | PASS | 0 | — |
| I11 | FAIL | PASS | 0 | — |
| I12 | FAIL | PASS | 0 | — |
| DISK | FAIL | PASS | 0 | 38.9 GB free |
