# Human pre-flight report

roots: ['Results/b5host_ARS_OPT_Salman2020Do_R50']
**FLAG lines: 0** (0 is required before a GPU run; every FLAG needs a written explanation)


## 1. Environment (one GPU name, one torch, one driver, one host, identical backend flags, ADV_STORE_BUDGETS unset)

| root | env file | GPU | driver | host | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |
|---|---|---|---|---|---|---|---|---|
| b5host_ARS_OPT_Salman2020Do_R50 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.8.0+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 0f454a78c615 |
| b5host_ARS_OPT_Salman2020Do_R50 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.8.0+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 0f454a78c615 |

- GPU names: {'NVIDIA GeForce RTX 5090': 2}
- torch versions: {'2.8.0+cu128': 2}
- drivers: {}
- hosts: {}
- note: no driver or hostname recorded in any env file; host drift (L03) cannot be checked from these folders, use `nvidia-smi` and the B5 runner host log
- backend benchmark: {'False': 2}
- backend cudnn_tf32: {'False': 2}
- backend deterministic: {'True': 2}
- backend matmul_tf32: {'False': 2}
- env files read: 2

## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 3; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0

## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)

| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |
|---|---|---|---|---|---|---|---|
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | 3 | {'ok': 1, 'budget': 2} | 3 | 0 | 0 | 0 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |

Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash.

## 4. L2 recomputation from stored images (relative error must be < 1e-3)

| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |
|---|---|---|---|---|---|---|---|---|
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 17.43835 | 17.43839 | 17.43834 | 2.3e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 33.34469 | 33.34483 | 33.34468 | 4.1e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 6.00538 | 6.00538 | 6.00538 | 3.8e-08 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 12.66836 | 12.66837 | 12.66836 | 8.1e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 15.59431 | 15.59431 | 15.59431 | 2.2e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 1000 | 27.57043 | 27.57054 | 27.57043 | 3.7e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 1.38866 | 1.38866 | 1.38866 | 1.0e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 1.78428 | 1.78428 | 1.78428 | 5.3e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 33.63123 | 33.63129 | 33.63122 | 2.0e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 21.51129 | 21.51135 | 21.51128 | 2.6e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 135.47183 | 135.46885 | 135.47182 | 2.2e-05 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 200 | 99.45928 | 99.45923 | 99.45929 | 5.5e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 103.07894 | 103.07891 | 103.07894 | 3.4e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 100.89915 | 100.89913 | 100.89915 | 2.8e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 72.93933 | 72.93933 | 72.93933 | 6.6e-09 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 108.98226 | 108.98233 | 108.98225 | 6.4e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 99.45928 | 99.45923 | 99.45929 | 5.5e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 75.37897 | 75.37905 | 75.37897 | 1.1e-06 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 121.46954 | 121.46958 | 121.46954 | 2.8e-07 |
| b5host_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 69.44453 | 69.44454 | 69.44453 | 1.4e-07 |

- rows checked: 20

## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)

- **b5host_ARS_OPT_Salman2020Do_R50/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 33.3; 200: 1.00 / 32.2; 500: 1.00 / 32.2; 1000: 1.00 / 27; 2000: 1.00 / 21.5; 2500: 1.00 / 19.9; 5000: 1.00 / 15.6; 10000: 1.00 / 12.7
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5host_ARS_OPT_Salman2020Do_R50/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 119; 200: 1.00 / 119; 500: 1.00 / 114; 1000: 1.00 / 111; 2000: 1.00 / 109; 2500: 1.00 / 107; 5000: 1.00 / 101; 10000: 1.00 / 82.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 7. Degeneracy (silent collapse: identical methods, constant outputs, never found, duplicated inputs)

- no degenerate pattern found

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/raw/B5-host/t6_human_preflight_grid_Salman2020Do_R50_non_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5-host/t6_human_preflight_grid_Salman2020Do_R50_targeted.png: 3 images x 1 methods at budget 10000

