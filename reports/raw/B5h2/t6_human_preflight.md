# Human pre-flight report

roots: ['Results/b5h2_ARS_OPT_Salman2020Do_R50', 'Results/b5h2_ARS_OPT_resnet50_places365', 'Results/b5h2_ARS_OPT_densenet161_places365']
**FLAG lines: 0** (0 is required before a GPU run; every FLAG needs a written explanation)


## 1. Environment (one GPU name, one torch, one driver, one host, identical backend flags, ADV_STORE_BUDGETS unset)

| root | env file | GPU | driver | host | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |
|---|---|---|---|---|---|---|---|---|
| b5h2_ARS_OPT_Salman2020Do_R50 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 0f454a78c615 |
| b5h2_ARS_OPT_Salman2020Do_R50 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 0f454a78c615 |
| b5h2_ARS_OPT_resnet50_places365 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 0f454a78c615 |
| b5h2_ARS_OPT_resnet50_places365 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 0f454a78c615 |
| b5h2_ARS_OPT_densenet161_places365 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 0f454a78c615 |
| b5h2_ARS_OPT_densenet161_places365 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 0f454a78c615 |

- GPU names: {'NVIDIA GeForce RTX 5090': 6}
- torch versions: {'2.7.1+cu128': 6}
- drivers: {}
- hosts: {}
- note: no driver or hostname recorded in any env file; host drift (L03) cannot be checked from these folders, use `nvidia-smi` and the B5 runner host log
- backend benchmark: {'False': 6}
- backend cudnn_tf32: {'False': 6}
- backend deterministic: {'True': 6}
- backend matmul_tf32: {'False': 6}
- env files read: 6

## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 9; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0

## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)

| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |
|---|---|---|---|---|---|---|---|
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |

Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash.

## 4. L2 recomputation from stored images (relative error must be < 1e-3)

| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |
|---|---|---|---|---|---|---|---|---|
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 17.44399 | 17.44408 | 17.44398 | 5.0e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 33.34469 | 33.34483 | 33.34468 | 4.1e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 6.00538 | 6.00538 | 6.00538 | 3.8e-08 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 13.02896 | 13.02897 | 13.02896 | 6.6e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 15.59244 | 15.59248 | 15.59244 | 2.3e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 1000 | 27.71271 | 27.71281 | 27.71271 | 3.5e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 1.11348 | 1.11348 | 1.11348 | 1.2e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 1.56649 | 1.56649 | 1.56649 | 6.3e-08 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 33.63118 | 33.63123 | 33.63117 | 1.7e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 21.68390 | 21.68395 | 21.68389 | 2.2e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 135.47183 | 135.46885 | 135.47182 | 2.2e-05 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 200 | 99.45928 | 99.45923 | 99.45929 | 5.5e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 102.74019 | 102.74008 | 102.74019 | 1.1e-06 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 100.86748 | 100.86745 | 100.86748 | 2.4e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 72.94798 | 72.94794 | 72.94797 | 6.2e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 108.14600 | 108.14592 | 108.14599 | 7.2e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 99.45928 | 99.45923 | 99.45929 | 5.5e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 75.43311 | 75.43314 | 75.43311 | 5.0e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 121.46960 | 121.46964 | 121.46960 | 2.5e-07 |
| b5h2_ARS_OPT_Salman2020Do_R50/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 69.43133 | 69.43139 | 69.43133 | 8.8e-07 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 100 | 7.17223 | 7.17224 | 7.17223 | 1.1e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00023888 | 2000 | 21.15067 | 21.15072 | 21.15067 | 2.3e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 2500 | 24.65618 | 24.65625 | 24.65617 | 2.9e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 10000 | 0.58024 | 0.58024 | 0.58024 | 5.8e-07 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00023888 | 100 | 39.23969 | 39.23968 | 39.23969 | 2.2e-07 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 2000 | 27.24064 | 27.24072 | 27.24064 | 3.0e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 5000 | 18.11231 | 18.11237 | 18.11231 | 3.3e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 500 | 47.82698 | 47.82711 | 47.82697 | 2.8e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 500 | 6.54343 | 6.54342 | 6.54343 | 1.6e-06 |
| b5h2_ARS_OPT_resnet50_places365/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00023888 | 5000 | 12.14800 | 12.14802 | 12.14800 | 1.4e-06 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 2500 | 61.42619 | 61.42621 | 61.42618 | 3.7e-07 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 1000 | 68.49846 | 68.49860 | 68.49846 | 2.0e-06 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 100 | 74.84349 | 74.84262 | 74.84354 | 1.2e-05 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00023888 | 1000 | 66.00650 | 66.00651 | 66.00649 | 1.3e-07 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00023888 | 5000 | 40.11969 | 40.11973 | 40.11969 | 1.0e-06 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 10000 | 30.04997 | 30.04993 | 30.04997 | 1.4e-06 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 500 | 73.78000 | 73.77991 | 73.78000 | 1.2e-06 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00004317 | 5000 | 40.75629 | 40.75625 | 40.75628 | 9.5e-07 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 200 | 74.84349 | 74.84262 | 74.84354 | 1.2e-05 |
| b5h2_ARS_OPT_resnet50_places365/targeted | ARS_OPT | resnet50_places365 | Places365_val_00002172 | 10000 | 40.43505 | 40.43507 | 40.43505 | 4.2e-07 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 2500 | 18.58693 | 18.58695 | 18.58693 | 1.4e-06 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 200 | 42.06613 | 42.06621 | 42.06613 | 2.0e-06 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00004317 | 200 | 13.31908 | 13.31910 | 13.31908 | 1.3e-06 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 1000 | 27.30686 | 27.30692 | 27.30686 | 2.1e-06 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00004317 | 10000 | 1.21018 | 1.21018 | 1.21018 | 1.1e-06 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 2000 | 20.88894 | 20.88896 | 20.88894 | 8.4e-07 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 10000 | 5.06363 | 5.06362 | 5.06363 | 4.7e-07 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00004317 | 2500 | 3.27239 | 3.27239 | 3.27239 | 5.6e-07 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 500 | 42.06569 | 42.06572 | 42.06568 | 9.3e-07 |
| b5h2_ARS_OPT_densenet161_places365/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 5000 | 8.05703 | 8.05702 | 8.05703 | 1.1e-06 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00004317 | 10000 | 5.89766 | 5.89767 | 5.89766 | 2.3e-06 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 100 | 46.07434 | 46.07437 | 46.07435 | 6.4e-07 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 2500 | 38.60924 | 38.60924 | 38.60924 | 4.7e-08 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 5000 | 17.46691 | 17.46694 | 17.46691 | 1.7e-06 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00004317 | 500 | 37.35463 | 37.35463 | 37.35463 | 1.2e-07 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 10000 | 8.81135 | 8.81135 | 8.81135 | 4.4e-09 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00023888 | 2000 | 40.83001 | 40.83003 | 40.83001 | 5.0e-07 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 200 | 78.51077 | 78.51038 | 78.51078 | 5.0e-06 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 2000 | 33.18403 | 33.18398 | 33.18403 | 1.7e-06 |
| b5h2_ARS_OPT_densenet161_places365/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002172 | 100 | 78.51077 | 78.51038 | 78.51078 | 5.0e-06 |

- rows checked: 60

## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)

- **b5h2_ARS_OPT_Salman2020Do_R50/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 33.3; 200: 1.00 / 32.2; 500: 1.00 / 32.2; 1000: 1.00 / 27; 2000: 1.00 / 21.7; 2500: 1.00 / 20; 5000: 1.00 / 15.6; 10000: 1.00 / 13
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5h2_ARS_OPT_Salman2020Do_R50/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 119; 200: 1.00 / 119; 500: 1.00 / 114; 1000: 1.00 / 111; 2000: 1.00 / 108; 2500: 1.00 / 107; 5000: 1.00 / 101; 10000: 1.00 / 82.3
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5h2_ARS_OPT_resnet50_places365/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 39.2; 200: 1.00 / 37.9; 500: 1.00 / 37.9; 1000: 1.00 / 28.8; 2000: 1.00 / 21.2; 2500: 1.00 / 19.4; 5000: 1.00 / 12.1; 10000: 1.00 / 7.18
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5h2_ARS_OPT_resnet50_places365/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 76.9; 200: 1.00 / 76.9; 500: 1.00 / 74.9; 1000: 1.00 / 68.5; 2000: 1.00 / 58.4; 2500: 1.00 / 54.4; 5000: 1.00 / 40.8; 10000: 1.00 / 30
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5h2_ARS_OPT_densenet161_places365/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 40.3; 200: 1.00 / 38.4; 500: 1.00 / 38.4; 1000: 1.00 / 27.3; 2000: 1.00 / 17.1; 2500: 1.00 / 14; 5000: 1.00 / 8.06; 10000: 1.00 / 5.06
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5h2_ARS_OPT_densenet161_places365/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 46.1; 200: 1.00 / 46.1; 500: 1.00 / 45.6; 1000: 1.00 / 42.7; 2000: 1.00 / 33.2; 2500: 1.00 / 28.3; 5000: 1.00 / 17.5; 10000: 1.00 / 8.81
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 7. Degeneracy (silent collapse: identical methods, constant outputs, never found, duplicated inputs)

- no degenerate pattern found

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/raw/B5h2/t6_human_preflight_grid_Salman2020Do_R50_non_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5h2/t6_human_preflight_grid_Salman2020Do_R50_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5h2/t6_human_preflight_grid_densenet161_places365_non_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5h2/t6_human_preflight_grid_densenet161_places365_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5h2/t6_human_preflight_grid_resnet50_places365_non_targeted.png: 3 images x 1 methods at budget 10000
- reports/raw/B5h2/t6_human_preflight_grid_resnet50_places365_targeted.png: 3 images x 1 methods at budget 10000

