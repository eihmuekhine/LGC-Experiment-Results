# Human pre-flight report

roots: ['Results/b4e_smoke_arsopt_resnet50', 'Results/b4e_smoke_arsopt_vgg19', 'Results/b4e_smoke_arsopt_vit_b_32', 'Results/b4e_smoke_cgba_Salman2020Do_R50', 'Results/b4e_smoke_cgba_densenet161_places365', 'Results/b4e_smoke_cgba_resnet50', 'Results/b4e_smoke_cgba_resnet50_places365', 'Results/b4e_smoke_cgba_vgg19', 'Results/b4e_smoke_cgba_vit_b_32', 'Results/b4e_smoke_ttba_Salman2020Do_R50', 'Results/b4e_smoke_ttba_densenet161_places365', 'Results/b4e_smoke_ttba_resnet50', 'Results/b4e_smoke_ttba_resnet50_places365', 'Results/b4e_smoke_ttba_vgg19', 'Results/b4e_smoke_ttba_vit_b_32']
**FLAG lines: 6** (0 is required before a GPU run; every FLAG needs a written explanation)


## 1. Environment (one GPU name, one torch, identical backend flags, ADV_STORE_BUDGETS unset)

| root | env file | GPU | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |
|---|---|---|---|---|---|---|
| b4e_smoke_arsopt_resnet50 | non_targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, d06410d40db1 |
| b4e_smoke_arsopt_resnet50 | targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, d06410d40db1 |
| b4e_smoke_arsopt_vgg19 | non_targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, d06410d40db1 |
| b4e_smoke_arsopt_vgg19 | targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, d06410d40db1 |
| b4e_smoke_arsopt_vit_b_32 | non_targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, d06410d40db1 |
| b4e_smoke_arsopt_vit_b_32 | targeted/env_ARS_OPT.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, d06410d40db1 |
| b4e_smoke_cgba_Salman2020Do_R50 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 5f7fac692bb9 |
| b4e_smoke_cgba_Salman2020Do_R50 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 5f7fac692bb9 |
| b4e_smoke_cgba_densenet161_places365 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 5f7fac692bb9 |
| b4e_smoke_cgba_densenet161_places365 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 5f7fac692bb9 |
| b4e_smoke_cgba_resnet50 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 5f7fac692bb9 |
| b4e_smoke_cgba_resnet50 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 5f7fac692bb9 |
| b4e_smoke_cgba_resnet50_places365 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 5f7fac692bb9 |
| b4e_smoke_cgba_resnet50_places365 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 5f7fac692bb9 |
| b4e_smoke_cgba_vgg19 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 5f7fac692bb9 |
| b4e_smoke_cgba_vgg19 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 5f7fac692bb9 |
| b4e_smoke_cgba_vit_b_32 | non_targeted/env_CGBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 5f7fac692bb9 |
| b4e_smoke_cgba_vit_b_32 | targeted/env_CGBA_H.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 5f7fac692bb9 |
| b4e_smoke_ttba_Salman2020Do_R50 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 1b847d4c3acc, 1835a5bbe5e3, 9ac8bc1ce48d |
| b4e_smoke_ttba_Salman2020Do_R50 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 2985ebc400ea, 1835a5bbe5e3, 9ac8bc1ce48d |
| b4e_smoke_ttba_Salman2020Do_R50 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e_smoke_ttba_densenet161_places365 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 892528497715, b311cb0e8bea, 9ac8bc1ce48d |
| b4e_smoke_ttba_densenet161_places365 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 51935b752227, b311cb0e8bea, 9ac8bc1ce48d |
| b4e_smoke_ttba_densenet161_places365 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e_smoke_ttba_resnet50 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | e08a069e932e, 11ad3fa62ca7, 9ac8bc1ce48d |
| b4e_smoke_ttba_resnet50 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 550ec796e485, 11ad3fa62ca7, 9ac8bc1ce48d |
| b4e_smoke_ttba_resnet50 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e_smoke_ttba_resnet50_places365 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 351bc50f594f, 46529c86902b, 9ac8bc1ce48d |
| b4e_smoke_ttba_resnet50_places365 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 70bd3d5c5989, 46529c86902b, 9ac8bc1ce48d |
| b4e_smoke_ttba_resnet50_places365 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e_smoke_ttba_vgg19 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | 1e811637c264, dcbb9e9dad56, 9ac8bc1ce48d |
| b4e_smoke_ttba_vgg19 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | cf0a8efb783d, dcbb9e9dad56, 9ac8bc1ce48d |
| b4e_smoke_ttba_vgg19 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e_smoke_ttba_vit_b_32 | non_targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | a5f152ec215a, d86f8d99b3c3, 9ac8bc1ce48d |
| b4e_smoke_ttba_vit_b_32 | targeted/env_TtBA.json | NVIDIA RTX PRO 6000 Blackwell Server Edition | 2.7.1+cu128 | - | unset | c5eee382244c, d86f8d99b3c3, 9ac8bc1ce48d |
| b4e_smoke_ttba_vit_b_32 | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |

- GPU names: {'NVIDIA RTX PRO 6000 Blackwell Server Edition': 30}
- torch versions: {'2.7.1+cu128': 30}
- backend benchmark: {'False': 18}
- backend cudnn_tf32: {'False': 18}
- backend deterministic: {'True': 18}
- backend matmul_tf32: {'False': 18}
- env files read: 30

## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 18; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0

## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)

| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |
|---|---|---|---|---|---|---|---|
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |

Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash.

## 4. L2 recomputation from stored images (relative error must be < 1e-3)

| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |
|---|---|---|---|---|---|---|---|---|
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 5000 | 1.26449 | 1.26449 | 1.26449 | 2.3e-07 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 100 | 107.57796 | 107.57745 | 107.57799 | 4.7e-06 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 200 | 98.99802 | 98.99751 | 98.99796 | 5.1e-06 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 10000 | 6.86946 | 6.86945 | 6.86946 | 2.9e-07 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 5000 | 17.28113 | 17.28118 | 17.28113 | 2.8e-06 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 1000 | 51.86359 | 51.86369 | 51.86357 | 2.0e-06 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 38.25051 | 38.25051 | 38.25054 | 8.0e-07 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 2500 | 49.94231 | 49.94234 | 49.94233 | 4.5e-07 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 500 | 92.27279 | 92.27282 | 92.27276 | 3.7e-07 |
| b4e_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 2000 | 48.97884 | 48.97893 | 48.97884 | 1.7e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 100 | 52.34167 | 52.34120 | 52.34170 | 9.0e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 200 | 48.46287 | 48.46279 | 48.46287 | 1.6e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 18.51800 | 18.51800 | 18.51800 | 4.4e-07 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 5000 | 34.27119 | 34.27123 | 34.27119 | 1.1e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 5000 | 20.46529 | 20.46529 | 20.46529 | 9.3e-08 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 2000 | 72.11864 | 72.11866 | 72.11864 | 2.0e-07 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 100 | 48.46287 | 48.46279 | 48.46287 | 1.6e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 2000 | 32.06754 | 32.06753 | 32.06753 | 2.4e-07 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 1000 | 38.91516 | 38.91508 | 38.91517 | 2.1e-06 |
| b4e_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 10000 | 11.95406 | 11.95409 | 11.95406 | 2.0e-06 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 100 | 10.85875 | 10.85876 | 10.85874 | 8.0e-07 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 2000 | 1.22414 | 1.22414 | 1.22414 | 3.9e-07 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 2500 | 3.94977 | 3.94977 | 3.94977 | 1.0e-06 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 10000 | 0.31785 | 0.31785 | 0.31785 | 1.0e-06 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 100 | 11.45771 | 11.45776 | 11.45771 | 4.1e-06 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 2000 | 6.24157 | 6.24157 | 6.24157 | 1.2e-07 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 5000 | 1.36870 | 1.36870 | 1.36870 | 4.0e-07 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 500 | 30.39386 | 30.39397 | 30.39386 | 3.6e-06 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 500 | 9.10839 | 9.10840 | 9.10839 | 9.5e-07 |
| b4e_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 5000 | 0.31710 | 0.31710 | 0.31710 | 2.6e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 2500 | 31.51425 | 31.51428 | 31.51424 | 9.6e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 1000 | 103.56806 | 103.56800 | 103.56805 | 6.3e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 100 | 61.50500 | 61.50508 | 61.50501 | 1.2e-06 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 1000 | 46.30872 | 46.30871 | 46.30872 | 1.8e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 5000 | 22.94278 | 22.94275 | 22.94278 | 1.4e-06 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 10000 | 8.68472 | 8.68472 | 8.68472 | 2.4e-08 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 500 | 47.81596 | 47.81596 | 47.81596 | 1.6e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 5000 | 27.77750 | 27.77752 | 27.77750 | 5.4e-07 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 200 | 61.50500 | 61.50508 | 61.50501 | 1.2e-06 |
| b4e_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 10000 | 9.38412 | 9.38413 | 9.38412 | 8.6e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 22.37930 | 22.37931 | 22.37929 | 6.0e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 104.08455 | 104.08448 | 104.08455 | 6.5e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 200 | 95.80594 | 95.80586 | 95.80592 | 8.4e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 1000 | 42.65791 | 42.65790 | 42.65791 | 3.1e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 7.96115 | 7.96117 | 7.96115 | 1.4e-06 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 25.43427 | 25.43429 | 25.43427 | 7.1e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 3.07189 | 3.07189 | 3.07189 | 7.9e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 28.84567 | 28.84567 | 28.84567 | 1.4e-07 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 99.87387 | 99.87364 | 99.87386 | 2.3e-06 |
| b4e_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 4.95118 | 4.95118 | 4.95118 | 1.0e-06 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 7.51542 | 7.51543 | 7.51542 | 7.5e-07 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 39.69154 | 39.69153 | 39.69154 | 3.3e-07 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 18.46638 | 18.46637 | 18.46638 | 4.7e-07 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 14.61345 | 14.61342 | 14.61345 | 1.8e-06 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 53.74948 | 53.74945 | 53.74948 | 5.0e-07 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 8.14508 | 8.14508 | 8.14508 | 5.7e-09 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 20.96862 | 20.96860 | 20.96862 | 9.0e-07 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 48.87143 | 48.87180 | 48.87138 | 7.5e-06 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 27.64402 | 27.64399 | 27.64402 | 1.0e-06 |
| b4e_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 48.87143 | 48.87180 | 48.87138 | 7.5e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 0.85947 | 0.85947 | 0.85947 | 6.4e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 31.23740 | 31.23752 | 31.23740 | 3.6e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 19.57414 | 19.57415 | 19.57414 | 4.8e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 0.77451 | 0.77451 | 0.77451 | 8.0e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 13.18254 | 13.18255 | 13.18254 | 6.3e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 5.68772 | 5.68771 | 5.68772 | 9.5e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 5.68764 | 5.68765 | 5.68764 | 4.9e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2500 | 18.63033 | 18.63034 | 18.63033 | 7.3e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 9.34120 | 9.34121 | 9.34120 | 6.3e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 15.06092 | 15.06096 | 15.06092 | 2.1e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 50.44198 | 50.44191 | 50.44197 | 1.4e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 500 | 80.40415 | 80.40400 | 80.40415 | 1.9e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 117.52925 | 117.52929 | 117.52925 | 3.2e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 1000 | 102.91086 | 102.91086 | 102.91087 | 7.4e-08 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 200 | 116.31765 | 116.31762 | 116.31765 | 2.2e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 500 | 115.04238 | 115.04253 | 115.04239 | 1.3e-06 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 108.24114 | 108.24116 | 108.24113 | 2.0e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 72.05647 | 72.05650 | 72.05647 | 5.1e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 69.62467 | 69.62472 | 69.62466 | 6.3e-07 |
| b4e_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 72.13365 | 72.13360 | 72.13365 | 7.7e-07 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 2500 | 9.67960 | 9.67959 | 9.67960 | 1.5e-06 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 1000 | 1.60749 | 1.60749 | 1.60749 | 1.1e-06 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 100 | 4.60968 | 4.60968 | 4.60968 | 1.2e-06 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 100 | 36.59802 | 36.59806 | 36.59803 | 1.2e-06 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 10000 | 2.57356 | 2.57357 | 2.57356 | 1.6e-06 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 1000 | 13.18463 | 13.18463 | 13.18463 | 1.9e-07 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 200 | 38.07653 | 38.07652 | 38.07653 | 6.0e-08 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 500 | 2.35750 | 2.35750 | 2.35750 | 8.5e-07 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 2000 | 8.44232 | 8.44233 | 8.44232 | 7.9e-07 |
| b4e_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 2000 | 1.10273 | 1.10273 | 1.10273 | 5.0e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 500 | 41.91839 | 41.91843 | 41.91839 | 9.1e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 5000 | 15.48610 | 15.48610 | 15.48610 | 1.3e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 2500 | 18.78808 | 18.78809 | 18.78808 | 5.0e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 2500 | 9.83616 | 9.83616 | 9.83616 | 2.2e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 5000 | 12.94768 | 12.94767 | 12.94768 | 6.3e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 1000 | 45.18315 | 45.18309 | 45.18315 | 1.3e-06 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 10000 | 2.34471 | 2.34472 | 2.34471 | 6.0e-07 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 500 | 60.60648 | 60.60640 | 60.60648 | 1.5e-06 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 5000 | 4.69487 | 4.69487 | 4.69487 | 1.4e-08 |
| b4e_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 200 | 41.20610 | 41.20608 | 41.20610 | 6.2e-07 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 500 | 56.34610 | 56.34608 | 56.34610 | 3.0e-07 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 200 | 101.40413 | 101.40398 | 101.40414 | 1.4e-06 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 10000 | 6.33173 | 6.33173 | 6.33173 | 8.3e-07 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 5000 | 14.39271 | 14.39273 | 14.39271 | 1.5e-06 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 4.27450 | 4.27450 | 4.27450 | 3.3e-07 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 10000 | 2.60566 | 2.60566 | 2.60566 | 1.2e-06 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 100 | 107.64500 | 107.64523 | 107.64504 | 2.1e-06 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 10000 | 6.44977 | 6.44977 | 6.44977 | 5.2e-07 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 12.44827 | 12.44827 | 12.44827 | 6.3e-08 |
| b4e_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 200 | 64.66840 | 64.66825 | 64.66840 | 2.2e-06 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 10000 | 19.42974 | 19.42976 | 19.42974 | 6.1e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 2000 | 25.86901 | 25.86904 | 25.86901 | 1.2e-06 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 2000 | 21.42531 | 21.42531 | 21.42531 | 1.8e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 1000 | 72.11314 | 72.11318 | 72.11314 | 4.4e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 1000 | 32.04386 | 32.04386 | 32.04386 | 9.6e-08 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 5000 | 16.01384 | 16.01386 | 16.01384 | 1.3e-06 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 200 | 44.37506 | 44.37501 | 44.37506 | 9.6e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 2000 | 59.93637 | 59.93639 | 59.93637 | 2.8e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 500 | 81.18268 | 81.18275 | 81.18268 | 9.4e-07 |
| b4e_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 100 | 42.00632 | 42.00630 | 42.00632 | 4.7e-07 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 10000 | 4.51603 | 4.51603 | 4.51603 | 1.2e-06 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 5000 | 6.86662 | 6.86662 | 6.86662 | 1.5e-07 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 2500 | 10.25026 | 10.25025 | 10.25026 | 1.3e-06 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 2500 | 0.36097 | 0.36097 | 0.36097 | 3.0e-07 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 200 | 16.91656 | 16.91656 | 16.91656 | 3.2e-08 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 5000 | 0.26346 | 0.26346 | 0.26346 | 1.4e-06 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 500 | 0.92653 | 0.92653 | 0.92653 | 3.8e-07 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 5000 | 4.16895 | 4.16895 | 4.16895 | 1.4e-06 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 200 | 26.55421 | 26.55422 | 26.55421 | 3.7e-07 |
| b4e_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 100 | 18.90396 | 18.90395 | 18.90396 | 4.4e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 5000 | 24.64215 | 24.64213 | 24.64215 | 6.2e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 1000 | 58.67031 | 58.67032 | 58.67031 | 2.7e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 5000 | 38.01379 | 38.01380 | 38.01379 | 1.0e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 100 | 86.54550 | 86.54554 | 86.54550 | 4.6e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 200 | 73.99125 | 73.99127 | 73.99125 | 2.5e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 500 | 63.50596 | 63.50601 | 63.50596 | 7.7e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 10000 | 13.73230 | 13.73228 | 13.73230 | 1.1e-06 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 2500 | 46.90748 | 46.90746 | 46.90748 | 4.5e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 200 | 65.07099 | 65.07099 | 65.07100 | 1.2e-07 |
| b4e_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 2500 | 36.29361 | 36.29358 | 36.29361 | 9.6e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 0.56865 | 0.56865 | 0.56865 | 7.6e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 2000 | 0.30338 | 0.30338 | 0.30338 | 6.8e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.22681 | 1.22681 | 1.22681 | 6.3e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 10000 | 0.14714 | 0.14714 | 0.14714 | 5.9e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.31042 | 0.31042 | 0.31042 | 6.6e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 5000 | 0.18968 | 0.18968 | 0.18968 | 1.2e-06 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 1000 | 0.43214 | 0.43214 | 0.43214 | 7.9e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 30.04233 | 30.04239 | 30.04233 | 2.1e-06 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 2000 | 1.46538 | 1.46538 | 1.46538 | 5.2e-07 |
| b4e_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 200 | 8.09513 | 8.09513 | 8.09512 | 9.0e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 5000 | 6.41962 | 6.41962 | 6.41962 | 9.7e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 1000 | 31.11430 | 31.11435 | 31.11430 | 1.5e-06 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 2500 | 58.67213 | 58.67211 | 58.67212 | 2.7e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 100 | 51.30235 | 51.30236 | 51.30235 | 1.8e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 10000 | 5.75062 | 5.75062 | 5.75062 | 8.3e-08 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 2000 | 20.79606 | 20.79609 | 20.79606 | 1.5e-06 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 1000 | 31.28103 | 31.28104 | 31.28103 | 4.5e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 500 | 101.94424 | 101.94430 | 101.94424 | 6.2e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 2500 | 16.90603 | 16.90604 | 16.90603 | 1.3e-07 |
| b4e_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 10000 | 5.95353 | 5.95353 | 5.95353 | 3.7e-07 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 2.54690 | 2.54689 | 2.54690 | 1.3e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 2.96016 | 2.96016 | 2.96016 | 8.4e-07 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 0.19924 | 0.19924 | 0.19924 | 1.3e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 87.78716 | 87.78741 | 87.78715 | 2.9e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.57126 | 1.57126 | 1.57126 | 1.5e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 0.12353 | 0.12353 | 0.12353 | 1.3e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 18.52026 | 18.52024 | 18.52026 | 1.2e-06 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 37.84389 | 37.84388 | 37.84389 | 1.1e-07 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 9.29289 | 9.29289 | 9.29289 | 3.5e-07 |
| b4e_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 14.64742 | 14.64743 | 14.64742 | 2.8e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 23.20218 | 23.20217 | 23.20218 | 2.2e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 36.41232 | 36.41234 | 36.41232 | 6.8e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 38.91557 | 38.91554 | 38.91557 | 6.5e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 41.78167 | 41.78173 | 41.78167 | 1.5e-06 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 3.90134 | 3.90134 | 3.90134 | 1.4e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 3.66486 | 3.66486 | 3.66486 | 1.1e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 8.10122 | 8.10122 | 8.10122 | 8.5e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 28.07395 | 28.07393 | 28.07395 | 7.8e-07 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 40.62703 | 40.62707 | 40.62702 | 1.1e-06 |
| b4e_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 13.87274 | 13.87272 | 13.87274 | 1.4e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 18.66545 | 18.66550 | 18.66544 | 2.9e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 14.37983 | 14.37985 | 14.37983 | 1.8e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 11.28130 | 11.28131 | 11.28130 | 1.8e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 0.56849 | 0.56849 | 0.56849 | 2.2e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 14.19322 | 14.19327 | 14.19323 | 3.1e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 200 | 30.46525 | 30.46534 | 30.46525 | 3.0e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 26.46221 | 26.46224 | 26.46221 | 1.1e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 1000 | 19.96533 | 19.96537 | 19.96532 | 2.4e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 0.88766 | 0.88766 | 0.88766 | 5.1e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 1.27256 | 1.27256 | 1.27256 | 1.1e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 500 | 120.23787 | 120.23769 | 120.23787 | 1.5e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 95.53925 | 95.53930 | 95.53926 | 5.1e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 118.83317 | 118.83316 | 118.83317 | 1.1e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 58.83844 | 58.83841 | 58.83843 | 4.7e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 78.34359 | 78.34367 | 78.34358 | 1.0e-06 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 128.16905 | 128.16903 | 128.16905 | 1.9e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 75.46986 | 75.46984 | 75.46984 | 2.4e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 126.75220 | 126.75215 | 126.75220 | 4.2e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 107.91061 | 107.91055 | 107.91061 | 6.2e-07 |
| b4e_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 41.29353 | 41.29350 | 41.29353 | 7.7e-07 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 100 | 4.94234 | 4.94234 | 4.94234 | 1.2e-06 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 1000 | 17.93623 | 17.93624 | 17.93623 | 6.8e-07 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 1000 | 13.62899 | 13.62899 | 13.62899 | 3.7e-07 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 2500 | 7.32726 | 7.32728 | 7.32726 | 2.2e-06 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 100 | 36.07731 | 36.07732 | 36.07731 | 9.6e-08 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 10000 | 0.48276 | 0.48276 | 0.48276 | 5.7e-08 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2000 | 1.05277 | 1.05277 | 1.05277 | 2.5e-07 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2500 | 0.92687 | 0.92688 | 0.92687 | 3.0e-06 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 5000 | 3.80619 | 3.80618 | 3.80619 | 6.2e-07 |
| b4e_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 2000 | 9.42278 | 9.42280 | 9.42278 | 1.8e-06 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2500 | 13.50310 | 13.50311 | 13.50310 | 7.2e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 100 | 43.92746 | 43.92748 | 43.92746 | 4.2e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 2000 | 21.55757 | 21.55758 | 21.55757 | 1.8e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 500 | 65.54054 | 65.54047 | 65.54054 | 1.2e-06 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 10000 | 5.85498 | 5.85498 | 5.85498 | 5.8e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 1000 | 48.94812 | 48.94814 | 48.94812 | 5.1e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 10000 | 7.75049 | 7.75048 | 7.75049 | 1.2e-06 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 500 | 38.11999 | 38.11998 | 38.11999 | 4.2e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 5000 | 4.71696 | 4.71697 | 4.71696 | 7.5e-07 |
| b4e_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2000 | 18.09189 | 18.09191 | 18.09189 | 9.1e-07 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 500 | 55.06828 | 55.06836 | 55.06827 | 1.5e-06 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 500 | 59.87987 | 59.87992 | 59.87985 | 9.2e-07 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 200 | 59.12959 | 59.12949 | 59.12960 | 1.6e-06 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 2500 | 2.61189 | 2.61188 | 2.61189 | 9.0e-07 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 1000 | 23.41620 | 23.41624 | 23.41619 | 1.4e-06 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 1000 | 26.97807 | 26.97813 | 26.97806 | 2.5e-06 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 5000 | 4.46951 | 4.46951 | 4.46951 | 3.9e-07 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 0.72061 | 0.72061 | 0.72061 | 1.1e-06 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 7.19517 | 7.19516 | 7.19516 | 2.7e-07 |
| b4e_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 2500 | 11.43940 | 11.43940 | 11.43940 | 1.7e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 2000 | 68.26016 | 68.26019 | 68.26017 | 4.3e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 100 | 46.98930 | 46.98925 | 46.98930 | 1.0e-06 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 1000 | 27.67210 | 27.67213 | 27.67210 | 1.1e-06 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 100 | 100.54163 | 100.54157 | 100.54162 | 5.7e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 1000 | 29.87442 | 29.87441 | 29.87442 | 2.7e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 11.02277 | 11.02278 | 11.02277 | 6.8e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 100 | 45.21041 | 45.21035 | 45.21041 | 1.4e-06 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 5000 | 14.28099 | 14.28099 | 14.28099 | 1.8e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 2500 | 20.19067 | 20.19066 | 20.19067 | 4.7e-07 |
| b4e_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 44.95647 | 44.95647 | 44.95647 | 3.8e-08 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 5000 | 4.30478 | 4.30478 | 4.30478 | 9.2e-07 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 200 | 1.80850 | 1.80850 | 1.80850 | 1.7e-06 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 2500 | 6.11672 | 6.11671 | 6.11672 | 2.3e-07 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 500 | 17.99225 | 17.99228 | 17.99225 | 1.6e-06 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 500 | 0.95436 | 0.95435 | 0.95436 | 1.9e-06 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 200 | 17.95688 | 17.95685 | 17.95688 | 1.5e-06 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 100 | 19.97911 | 19.97913 | 19.97910 | 1.0e-06 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 2500 | 8.66786 | 8.66786 | 8.66786 | 1.2e-07 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 5000 | 0.27798 | 0.27798 | 0.27798 | 8.4e-07 |
| b4e_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 2000 | 0.44076 | 0.44076 | 0.44076 | 4.2e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 1000 | 54.84559 | 54.84564 | 54.84558 | 1.0e-06 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 10000 | 16.53258 | 16.53259 | 16.53258 | 6.4e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 2000 | 39.40837 | 39.40838 | 39.40837 | 9.7e-08 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 10000 | 29.20762 | 29.20765 | 29.20762 | 9.8e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 2000 | 47.16475 | 47.16470 | 47.16475 | 9.7e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 500 | 58.04853 | 58.04852 | 58.04853 | 2.6e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 2500 | 46.61505 | 46.61504 | 46.61505 | 3.1e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 100 | 76.22260 | 76.22257 | 76.22258 | 3.4e-07 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 100 | 69.88185 | 69.88175 | 69.88185 | 1.5e-06 |
| b4e_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 500 | 71.05180 | 71.05173 | 71.05179 | 9.0e-07 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 1000 | 0.45331 | 0.45331 | 0.45331 | 8.1e-07 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 500 | 12.43833 | 12.43836 | 12.43833 | 2.8e-06 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.47899 | 1.47899 | 1.47899 | 3.8e-07 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 10000 | 0.08960 | 0.08960 | 0.08960 | 1.7e-06 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 26.17952 | 26.17960 | 26.17952 | 3.0e-06 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 0.58420 | 0.58420 | 0.58420 | 3.5e-07 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 200 | 2.24611 | 2.24611 | 2.24611 | 1.5e-06 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 100 | 11.81661 | 11.81665 | 11.81662 | 2.8e-06 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 500 | 0.76848 | 0.76848 | 0.76848 | 6.8e-07 |
| b4e_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.20351 | 0.20351 | 0.20351 | 1.8e-06 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 2000 | 75.66352 | 75.66347 | 75.66351 | 6.2e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 2500 | 67.42777 | 67.42787 | 67.42776 | 1.4e-06 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 5000 | 9.79829 | 9.79829 | 9.79829 | 1.0e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 5.27679 | 5.27678 | 5.27679 | 9.2e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 200 | 45.89020 | 45.89026 | 45.89020 | 1.4e-06 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 2000 | 21.61076 | 21.61077 | 21.61076 | 4.0e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 200 | 46.21467 | 46.21466 | 46.21467 | 2.3e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 5000 | 5.26748 | 5.26748 | 5.26748 | 9.1e-08 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 100 | 127.60165 | 127.60174 | 127.60164 | 7.0e-07 |
| b4e_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 2500 | 15.20805 | 15.20805 | 15.20805 | 5.8e-07 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 23.34890 | 23.34895 | 23.34890 | 1.8e-06 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 11.72857 | 11.72857 | 11.72857 | 8.1e-08 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.62217 | 1.62217 | 1.62217 | 1.2e-06 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 1.86929 | 1.86929 | 1.86929 | 2.4e-07 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 30.04835 | 30.04834 | 30.04834 | 1.9e-07 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 59.35032 | 59.35031 | 59.35032 | 6.1e-08 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 12.70034 | 12.70035 | 12.70034 | 1.2e-06 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 200 | 36.16975 | 36.16977 | 36.16975 | 4.6e-07 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 0.27005 | 0.27005 | 0.27005 | 1.6e-06 |
| b4e_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 0.13464 | 0.13464 | 0.13464 | 7.9e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 2.83897 | 2.83897 | 2.83897 | 2.1e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 43.06328 | 43.06326 | 43.06328 | 3.2e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 17.85256 | 17.85257 | 17.85256 | 5.8e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 15.08250 | 15.08251 | 15.08250 | 7.5e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 17.83008 | 17.83007 | 17.83008 | 6.9e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 39.43153 | 39.43158 | 39.43153 | 1.2e-06 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 8.62490 | 8.62490 | 8.62490 | 2.0e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 100 | 72.30314 | 72.30314 | 72.30313 | 1.1e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 1000 | 24.53983 | 24.53983 | 24.53983 | 2.8e-07 |
| b4e_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 38.02777 | 38.02784 | 38.02777 | 1.7e-06 |

- rows checked: 300

## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)

- **b4e_smoke_arsopt_resnet50/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 99.1; 200: 1.00 / 99; 500: 1.00 / 96.1; 1000: 1.00 / 75; 2000: 1.00 / 49; 2500: 1.00 / 37.9; 5000: 1.00 / 17.3; 10000: 1.00 / 6.87
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_arsopt_resnet50/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 52.3; 200: 1.00 / 52.3; 500: 1.00 / 47; 1000: 1.00 / 38.9; 2000: 1.00 / 32.1; 2500: 1.00 / 28.7; 5000: 1.00 / 20.5; 10000: 1.00 / 12
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_arsopt_vgg19/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.5; 200: 1.00 / 11; 500: 1.00 / 11; 1000: 1.00 / 7.37; 2000: 1.00 / 1.22; 2500: 1.00 / 0.81; 5000: 1.00 / 0.424; 10000: 1.00 / 0.318
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_arsopt_vgg19/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 61.5; 200: 1.00 / 61.5; 500: 1.00 / 50.4; 1000: 1.00 / 46.3; 2000: 1.00 / 38.5; 2500: 1.00 / 36.6; 5000: 1.00 / 22.9; 10000: 1.00 / 9.38
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_arsopt_vit_b_32/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 101; 200: 1.00 / 95.8; 500: 1.00 / 92.6; 1000: 1.00 / 54; 2000: 1.00 / 25.4; 2500: 1.00 / 22.4; 5000: 1.00 / 11.5; 10000: 1.00 / 3.07
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_arsopt_vit_b_32/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 48.9; 200: 1.00 / 48.9; 500: 1.00 / 42.2; 1000: 1.00 / 32.9; 2000: 1.00 / 21.6; 2500: 1.00 / 18.5; 5000: 1.00 / 12.5; 10000: 1.00 / 8.01
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_Salman2020Do_R50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 31.2; 200: 1.00 / 29.7; 500: 1.00 / 22.9; 1000: 1.00 / 19; 2000: 1.00 / 14.5; 2500: 1.00 / 13.2; 5000: 1.00 / 9.34; 10000: 1.00 / 5.69
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_Salman2020Do_R50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 118; 200: 1.00 / 116; 500: 1.00 / 108; 1000: 1.00 / 100; 2000: 1.00 / 72.1; 2500: 1.00 / 69.6; 5000: 1.00 / 62; 10000: 1.00 / 48.4
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_densenet161_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 36.6; 200: 1.00 / 31.7; 500: 1.00 / 17.8; 1000: 1.00 / 13.2; 2000: 1.00 / 8.44; 2500: 1.00 / 7.04; 5000: 1.00 / 4.19; 10000: 1.00 / 2.4
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_densenet161_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 45.9; 200: 1.00 / 45.1; 500: 1.00 / 41.9; 1000: 1.00 / 33; 2000: 1.00 / 21.7; 2500: 1.00 / 18.8; 5000: 1.00 / 12.9; 10000: 1.00 / 6.37
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_resnet50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 112; 200: 1.00 / 81.5; 500: 1.00 / 56.3; 1000: 1.00 / 42.5; 2000: 1.00 / 28.2; 2500: 1.00 / 23.9; 5000: 1.00 / 12.4; 10000: 1.00 / 6.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_resnet50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 49; 200: 1.00 / 44.4; 500: 1.00 / 36; 1000: 1.00 / 32; 2000: 1.00 / 25.9; 2500: 1.00 / 23.3; 5000: 1.00 / 16; 10000: 1.00 / 8.95
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_resnet50_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 18.9; 200: 1.00 / 16.9; 500: 1.00 / 10.9; 1000: 1.00 / 8.68; 2000: 1.00 / 6.53; 2500: 1.00 / 5.82; 5000: 1.00 / 4.17; 10000: 1.00 / 3.01
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_resnet50_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 75.1; 200: 1.00 / 74; 500: 1.00 / 63.5; 1000: 1.00 / 57.3; 2000: 1.00 / 47.1; 2500: 1.00 / 42.8; 5000: 1.00 / 29.6; 10000: 1.00 / 16.7
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_vgg19/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.2; 200: 1.00 / 8.1; 500: 1.00 / 4.58; 1000: 1.00 / 1.96; 2000: 1.00 / 0.31; 2500: 1.00 / 0.264; 5000: 1.00 / 0.19; 10000: 1.00 / 0.147
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_vgg19/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 51.3; 200: 1.00 / 47.3; 500: 1.00 / 38.7; 1000: 1.00 / 31.3; 2000: 1.00 / 21.5; 2500: 1.00 / 18.6; 5000: 1.00 / 10; 10000: 1.00 / 5.75
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_vit_b_32/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 87.8; 200: 1.00 / 60.4; 500: 1.00 / 21.6; 1000: 1.00 / 10.9; 2000: 1.00 / 6.46; 2500: 1.00 / 4.63; 5000: 1.00 / 1.57; 10000: 1.00 / 1.19
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_cgba_vit_b_32/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 44.5; 200: 1.00 / 40.6; 500: 1.00 / 33; 1000: 1.00 / 26.2; 2000: 1.00 / 16.6; 2500: 1.00 / 13.9; 5000: 1.00 / 7.77; 10000: 1.00 / 3.66
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_Salman2020Do_R50/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 32.7; 200: 1.00 / 30.5; 500: 1.00 / 26.5; 1000: 1.00 / 20; 2000: 1.00 / 14.4; 2500: 1.00 / 12.8; 5000: 1.00 / 8.91; 10000: 1.00 / 6.61
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_Salman2020Do_R50/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 119; 200: 1.00 / 114; 500: 1.00 / 108; 1000: 1.00 / 102; 2000: 1.00 / 78.3; 2500: 1.00 / 75.5; 5000: 1.00 / 64.6; 10000: 1.00 / 44.8
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_densenet161_places365/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 29.3; 200: 1.00 / 27.2; 500: 1.00 / 19.6; 1000: 1.00 / 13.6; 2000: 1.00 / 8.78; 2500: 1.00 / 7.33; 5000: 1.00 / 3.81; 10000: 1.00 / 2.24
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_densenet161_places365/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 45.8; 200: 1.00 / 45.6; 500: 1.00 / 42.2; 1000: 1.00 / 30.7; 2000: 1.00 / 21.6; 2500: 1.00 / 19.7; 5000: 1.00 / 12.6; 10000: 1.00 / 5.85
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_resnet50/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 82.7; 200: 1.00 / 70.7; 500: 1.00 / 55.1; 1000: 1.00 / 23.7; 2000: 1.00 / 12.3; 2500: 1.00 / 9.57; 5000: 1.00 / 4.47; 10000: 1.00 / 2.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_resnet50/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 47; 200: 1.00 / 43.7; 500: 1.00 / 36.3; 1000: 1.00 / 29.9; 2000: 1.00 / 22.7; 2500: 1.00 / 20.2; 5000: 1.00 / 14.3; 10000: 1.00 / 8.59
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_resnet50_places365/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 20; 200: 1.00 / 18; 500: 1.00 / 12.7; 1000: 1.00 / 8.83; 2000: 1.00 / 6.73; 2500: 1.00 / 6.12; 5000: 1.00 / 4.3; 10000: 1.00 / 3.09
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_resnet50_places365/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 76.2; 200: 1.00 / 74.1; 500: 1.00 / 69.3; 1000: 1.00 / 54.8; 2000: 1.00 / 47.2; 2500: 1.00 / 45.7; 5000: 1.00 / 30.7; 10000: 1.00 / 16.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_vgg19/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.8; 200: 1.00 / 7; 500: 1.00 / 0.768; 1000: 1.00 / 0.453; 2000: 1.00 / 0.316; 2500: 1.00 / 0.271; 5000: 1.00 / 0.198; 10000: 1.00 / 0.151
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_vgg19/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 48.6; 200: 1.00 / 46.2; 500: 1.00 / 41.7; 1000: 1.00 / 34.7; 2000: 1.00 / 21.6; 2500: 1.00 / 18.8; 5000: 1.00 / 9.8; 10000: 1.00 / 5.28
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_vit_b_32/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 47.3; 200: 1.00 / 36.2; 500: 1.00 / 23.3; 1000: 1.00 / 11.7; 2000: 1.00 / 6.07; 2500: 1.00 / 4.85; 5000: 1.00 / 1.62; 10000: 1.00 / 0.787
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e_smoke_ttba_vit_b_32/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 43.1; 200: 1.00 / 39.4; 500: 1.00 / 33.6; 1000: 1.00 / 25.1; 2000: 1.00 / 17.8; 2500: 1.00 / 16.2; 5000: 1.00 / 8.62; 10000: 1.00 / 4.27
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/human_preflight_b4e_grid_Salman2020Do_R50_non_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_Salman2020Do_R50_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_densenet161_places365_non_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_densenet161_places365_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_resnet50_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e_grid_resnet50_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e_grid_resnet50_places365_non_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_resnet50_places365_targeted.png: 3 images x 2 methods at budget 10000
- reports/human_preflight_b4e_grid_vgg19_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e_grid_vgg19_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e_grid_vit_b_32_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e_grid_vit_b_32_targeted.png: 3 images x 3 methods at budget 10000

