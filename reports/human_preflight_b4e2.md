# Human pre-flight report

roots: ['Results/b4e2_smoke_LGC_RC_EGC_resnet50', 'Results/b4e2_smoke_LGC_RC_EGC_vgg19', 'Results/b4e2_smoke_LGC_RC_EGC_vit_b_32', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted', 'Results/b4e2_smoke_ttba_resnet50', 'Results/b4e2_smoke_ttba_vgg19', 'Results/b4e2_smoke_ttba_vit_b_32', 'Results/b4e2_smoke_ttba_resnet50_places365', 'Results/b4e2_smoke_ttba_densenet161_places365', 'Results/b4e2_smoke_ttba_Salman2020Do_R50', 'Results/b4e2_smoke_cgba_resnet50', 'Results/b4e2_smoke_cgba_vgg19', 'Results/b4e2_smoke_cgba_vit_b_32', 'Results/b4e2_smoke_cgba_resnet50_places365', 'Results/b4e2_smoke_cgba_densenet161_places365', 'Results/b4e2_smoke_cgba_Salman2020Do_R50', 'Results/b4e2_smoke_arsopt_resnet50', 'Results/b4e2_smoke_arsopt_vgg19', 'Results/b4e2_smoke_arsopt_vit_b_32']
**FLAG lines: 7** (0 is required before a GPU run; every FLAG needs a written explanation)


## 1. Environment (one GPU name, one torch, identical backend flags, ADV_STORE_BUDGETS unset)

| root | env file | GPU | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |
|---|---|---|---|---|---|---|
| b4e2_smoke_LGC_RC_EGC_resnet50 | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | - | unset | 835f1933d9e5, 9c62820e5faf, e08a069e932e |
| b4e2_smoke_LGC_RC_EGC_resnet50 | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_resnet50 | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_vgg19 | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | - | unset | 835f1933d9e5, 9c62820e5faf, 1e811637c264 |
| b4e2_smoke_LGC_RC_EGC_vgg19 | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_vgg19 | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_vit_b_32 | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | - | unset | 835f1933d9e5, 9c62820e5faf, a5f152ec215a |
| b4e2_smoke_LGC_RC_EGC_vit_b_32 | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_vit_b_32 | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 351bc50f594f, 46529c86902b |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 70bd3d5c5989, 46529c86902b |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 892528497715, b311cb0e8bea |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 51935b752227, b311cb0e8bea |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 1b847d4c3acc, 1835a5bbe5e3 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted | env_frame.txt | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | - | unset | 835f1933d9e5, 2985ebc400ea, 1835a5bbe5e3 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted | **FLAG** | | | not recorded in any env/config file: ['cudnn_tf32', 'deterministic'] | | |
| b4e2_smoke_ttba_resnet50 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 80cbcdcc19b2 |
| b4e2_smoke_ttba_resnet50 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 80cbcdcc19b2 |
| b4e2_smoke_ttba_vgg19 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 80cbcdcc19b2 |
| b4e2_smoke_ttba_vgg19 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 80cbcdcc19b2 |
| b4e2_smoke_ttba_vit_b_32 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 80cbcdcc19b2 |
| b4e2_smoke_ttba_vit_b_32 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 80cbcdcc19b2 |
| b4e2_smoke_ttba_resnet50_places365 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 80cbcdcc19b2 |
| b4e2_smoke_ttba_resnet50_places365 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 80cbcdcc19b2 |
| b4e2_smoke_ttba_densenet161_places365 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 80cbcdcc19b2 |
| b4e2_smoke_ttba_densenet161_places365 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 80cbcdcc19b2 |
| b4e2_smoke_ttba_Salman2020Do_R50 | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 80cbcdcc19b2 |
| b4e2_smoke_ttba_Salman2020Do_R50 | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 80cbcdcc19b2 |
| b4e2_smoke_cgba_resnet50 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 5f7fac692bb9 |
| b4e2_smoke_cgba_resnet50 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 5f7fac692bb9 |
| b4e2_smoke_cgba_vgg19 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 5f7fac692bb9 |
| b4e2_smoke_cgba_vgg19 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 5f7fac692bb9 |
| b4e2_smoke_cgba_vit_b_32 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 5f7fac692bb9 |
| b4e2_smoke_cgba_vit_b_32 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 5f7fac692bb9 |
| b4e2_smoke_cgba_resnet50_places365 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 5f7fac692bb9 |
| b4e2_smoke_cgba_resnet50_places365 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 5f7fac692bb9 |
| b4e2_smoke_cgba_densenet161_places365 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 5f7fac692bb9 |
| b4e2_smoke_cgba_densenet161_places365 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 5f7fac692bb9 |
| b4e2_smoke_cgba_Salman2020Do_R50 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 5f7fac692bb9 |
| b4e2_smoke_cgba_Salman2020Do_R50 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 5f7fac692bb9 |
| b4e2_smoke_arsopt_resnet50 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, d06410d40db1 |
| b4e2_smoke_arsopt_resnet50 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, d06410d40db1 |
| b4e2_smoke_arsopt_vgg19 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, d06410d40db1 |
| b4e2_smoke_arsopt_vgg19 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, d06410d40db1 |
| b4e2_smoke_arsopt_vit_b_32 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, d06410d40db1 |
| b4e2_smoke_arsopt_vit_b_32 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, d06410d40db1 |

- GPU names: {'NVIDIA GeForce RTX 5090': 39, 'NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True': 6}
  - **FLAG: more than one value across env files**
- torch versions: {'2.7.1+cu128': 42}
- backend benchmark: {'False': 30}
- backend cudnn_tf32: {'False': 36}
- backend deterministic: {'True': 36}
- backend matmul_tf32: {'False': 30}
- env files read: 45

## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 18; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0

## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)

| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |
|---|---|---|---|---|---|---|---|
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | 3 | {'ok': 3} | 0 | 3 | 0 | n/a |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |

Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash.

## 4. L2 recomputation from stored images (relative error must be < 1e-3)

| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |
|---|---|---|---|---|---|---|---|---|
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00015433 | 5000 | 0.54728 | 0.54728 | 0.54728 | 2.8e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 100 | 49.81208 | 49.81211 | 49.81208 | 5.6e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 200 | 44.64902 | 44.64907 | 44.64902 | 1.2e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 10000 | 14.40361 | 14.40363 | 14.40361 | 1.6e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 5000 | 18.15683 | 18.15684 | 18.15683 | 2.2e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00015433 | 1000 | 2.23173 | 2.23174 | 2.23173 | 2.1e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 5000 | 6.09390 | 6.09392 | 6.09390 | 2.5e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 2500 | 10.51145 | 10.51144 | 10.51144 | 7.5e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00015433 | 500 | 7.48980 | 7.48976 | 7.48980 | 5.2e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 2000 | 25.95687 | 25.95692 | 25.95687 | 1.7e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 100 | 47.76506 | 47.76505 | 47.76506 | 1.8e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 200 | 37.41671 | 37.41670 | 37.41671 | 4.5e-08 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 5000 | 7.64411 | 7.64410 | 7.64411 | 1.8e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00015433 | 5000 | 13.38205 | 13.38207 | 13.38205 | 1.9e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 5000 | 11.22987 | 11.22989 | 11.22987 | 1.1e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00015433 | 2000 | 27.26789 | 27.26795 | 27.26789 | 2.4e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 100 | 40.68262 | 40.68257 | 40.68262 | 1.1e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 2000 | 21.72725 | 21.72724 | 21.72725 | 4.4e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00025151 | 1000 | 20.77753 | 20.77747 | 20.77753 | 3.0e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00007450 | 10000 | 7.17223 | 7.17223 | 7.17223 | 8.3e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 100 | 4.20791 | 4.20790 | 4.20791 | 1.4e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.21800 | 0.21800 | 0.21800 | 9.2e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.46959 | 1.46959 | 1.46959 | 6.5e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 10000 | 0.15300 | 0.15300 | 0.15300 | 1.2e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00002148 | 100 | 3.76397 | 3.76397 | 3.76397 | 1.0e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 2000 | 1.90711 | 1.90711 | 1.90711 | 6.2e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 5000 | 0.87534 | 0.87533 | 0.87534 | 6.5e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 500 | 6.40746 | 6.40745 | 6.40746 | 2.1e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 500 | 1.25814 | 1.25814 | 1.25814 | 2.0e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00002148 | 5000 | 0.12282 | 0.12282 | 0.12282 | 1.2e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 2500 | 21.66546 | 21.66544 | 21.66546 | 1.3e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 1000 | 88.49270 | 88.49276 | 88.49269 | 7.2e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 100 | 48.67825 | 48.67821 | 48.67825 | 8.2e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00002148 | 1000 | 27.60992 | 27.60992 | 27.60992 | 1.7e-08 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00002148 | 5000 | 6.59220 | 6.59218 | 6.59220 | 3.0e-06 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 10000 | 7.43976 | 7.43976 | 7.43976 | 8.8e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 500 | 40.25975 | 40.25975 | 40.25974 | 1.9e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00039161 | 5000 | 11.84532 | 11.84532 | 11.84532 | 2.3e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 200 | 44.93258 | 44.93255 | 44.93258 | 6.1e-07 |
| b4e2_smoke_LGC_RC_EGC_vgg19/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037360 | 10000 | 7.75727 | 7.75727 | 7.75727 | 6.1e-08 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 0.23686 | 0.23685 | 0.23686 | 1.1e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 21.42295 | 21.42296 | 21.42295 | 3.8e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00030908 | 200 | 23.07058 | 23.07064 | 23.07058 | 2.4e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 1000 | 5.10793 | 5.10792 | 5.10793 | 1.7e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 1.37985 | 1.37985 | 1.37985 | 5.5e-08 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 0.26102 | 0.26102 | 0.26102 | 1.6e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 1.21600 | 1.21600 | 1.21600 | 2.3e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 7.24058 | 7.24062 | 7.24057 | 6.5e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 0.70409 | 0.70409 | 0.70409 | 1.6e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.65663 | 1.65663 | 1.65663 | 3.5e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 3.13754 | 3.13754 | 3.13754 | 5.7e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 44.20610 | 44.20614 | 44.20610 | 7.8e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 9.87942 | 9.87944 | 9.87942 | 1.2e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 5.01229 | 5.01229 | 5.01229 | 7.7e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 36.18024 | 36.18024 | 36.18024 | 7.2e-09 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 3.10151 | 3.10152 | 3.10151 | 1.1e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 11.56391 | 11.56393 | 11.56391 | 1.6e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 43.91204 | 43.91208 | 43.91204 | 9.4e-07 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 13.06782 | 13.06781 | 13.06782 | 1.1e-06 |
| b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 50.15878 | 50.15881 | 50.15877 | 6.2e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00004317 | 2000 | 0.55000 | 0.55000 | 0.55000 | 5.0e-08 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 100 | 32.86798 | 32.86805 | 32.86798 | 1.9e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 2000 | 7.28435 | 7.28436 | 7.28435 | 2.0e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00004317 | 2500 | 0.45201 | 0.45201 | 0.45201 | 1.3e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00002172 | 2500 | 6.57844 | 6.57842 | 6.57844 | 2.5e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 10000 | 2.65801 | 2.65801 | 2.65801 | 7.0e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00004317 | 100 | 5.49284 | 5.49285 | 5.49284 | 8.4e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 2500 | 6.19057 | 6.19059 | 6.19057 | 3.8e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00002172 | 5000 | 3.93113 | 3.93113 | 3.93113 | 1.6e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 5000 | 3.93777 | 3.93777 | 3.93777 | 9.3e-08 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00002172 | 10000 | 21.43463 | 21.43459 | 21.43463 | 1.9e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00002172 | 500 | 63.39032 | 63.39024 | 63.39032 | 1.3e-06 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 100 | 68.30566 | 68.30568 | 68.30566 | 2.5e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 1000 | 42.83981 | 42.83985 | 42.83982 | 7.8e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 200 | 61.91455 | 61.91456 | 61.91455 | 1.2e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00004317 | 500 | 59.98590 | 59.98586 | 59.98590 | 6.0e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 500 | 53.33322 | 53.33326 | 53.33321 | 8.6e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00004317 | 2000 | 40.94772 | 40.94770 | 40.94772 | 4.6e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00002172 | 2500 | 45.24490 | 45.24493 | 45.24491 | 6.2e-07 |
| b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023888 | 5000 | 16.51506 | 16.51508 | 16.51506 | 1.7e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00023888 | 2500 | 10.22669 | 10.22669 | 10.22669 | 4.8e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 1000 | 2.59677 | 2.59678 | 2.59677 | 1.6e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 100 | 12.27450 | 12.27450 | 12.27450 | 9.4e-08 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 100 | 20.22166 | 20.22165 | 20.22166 | 5.4e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 10000 | 1.78586 | 1.78586 | 1.78586 | 1.3e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 1000 | 7.37105 | 7.37103 | 7.37105 | 3.2e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00023888 | 200 | 32.32329 | 32.32332 | 32.32329 | 9.3e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 500 | 4.18741 | 4.18741 | 4.18741 | 1.8e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 2000 | 5.06447 | 5.06447 | 5.06447 | 9.4e-08 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 2000 | 1.29158 | 1.29158 | 1.29158 | 8.9e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00023888 | 500 | 44.06367 | 44.06366 | 44.06367 | 1.4e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 5000 | 8.85421 | 8.85419 | 8.85421 | 2.7e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00023888 | 2500 | 21.29878 | 21.29880 | 21.29878 | 6.8e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 2500 | 5.29162 | 5.29161 | 5.29162 | 2.1e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00023888 | 5000 | 14.32049 | 14.32050 | 14.32049 | 6.4e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 1000 | 27.09959 | 27.09961 | 27.09959 | 8.6e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 10000 | 1.65912 | 1.65912 | 1.65912 | 1.7e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00002172 | 500 | 38.89699 | 38.89700 | 38.89698 | 4.4e-07 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 5000 | 2.59571 | 2.59570 | 2.59571 | 3.5e-06 |
| b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004317 | 200 | 47.59063 | 47.59058 | 47.59063 | 9.5e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 3.59865 | 3.59866 | 3.59865 | 3.9e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 200 | 20.01635 | 20.01632 | 20.01635 | 1.5e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 1.54289 | 1.54288 | 1.54289 | 2.7e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 7.17181 | 7.17181 | 7.17181 | 1.8e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 0.44098 | 0.44098 | 0.44098 | 2.5e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 10000 | 0.37933 | 0.37933 | 0.37933 | 4.9e-09 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 20.33207 | 20.33208 | 20.33207 | 1.9e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 6.26689 | 6.26690 | 6.26689 | 1.4e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 1.74887 | 1.74887 | 1.74887 | 2.0e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 1.45891 | 1.45891 | 1.45891 | 3.3e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 20.73919 | 20.73931 | 20.73919 | 6.1e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 59.93702 | 59.93706 | 59.93702 | 5.5e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 21.57362 | 21.57362 | 21.57361 | 3.0e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 1000 | 107.98732 | 107.98729 | 107.98730 | 2.9e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 1000 | 68.41602 | 68.41606 | 68.41602 | 5.0e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 43.23030 | 43.23035 | 43.23029 | 1.0e-06 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 117.98757 | 117.98757 | 117.98757 | 3.9e-08 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 87.94010 | 87.94012 | 87.94009 | 2.3e-07 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 110.86343 | 110.86343 | 110.86343 | 2.1e-08 |
| b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 97.63165 | 97.63176 | 97.63165 | 1.1e-06 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 10000 | 2.33134 | 2.33133 | 2.33134 | 3.0e-06 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 5000 | 4.46951 | 4.46951 | 4.46951 | 3.9e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 2500 | 9.56814 | 9.56814 | 9.56814 | 3.6e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 2500 | 11.43940 | 11.43940 | 11.43940 | 1.7e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 200 | 70.71170 | 70.71171 | 70.71169 | 1.7e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 7.19517 | 7.19516 | 7.19516 | 2.7e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 500 | 35.87771 | 35.87770 | 35.87774 | 8.5e-07 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 0.72061 | 0.72061 | 0.72061 | 1.1e-06 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 200 | 78.70503 | 78.70503 | 78.70502 | 9.7e-08 |
| b4e2_smoke_ttba_resnet50/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 100 | 84.14943 | 84.14942 | 84.14941 | 2.7e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 5000 | 14.28099 | 14.28099 | 14.28099 | 1.8e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 1000 | 82.98430 | 82.98429 | 82.98430 | 1.1e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 11.02277 | 11.02278 | 11.02277 | 6.8e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 100 | 46.98930 | 46.98925 | 46.98930 | 1.0e-06 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00015433 | 200 | 99.88235 | 99.88227 | 99.88236 | 8.5e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 500 | 36.31205 | 36.31204 | 36.31206 | 3.1e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 10000 | 8.59054 | 8.59054 | 8.59054 | 6.6e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00025151 | 2500 | 18.15491 | 18.15492 | 18.15490 | 6.8e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 200 | 41.26606 | 41.26604 | 41.26606 | 5.2e-07 |
| b4e2_smoke_ttba_resnet50/targeted | TtBA | resnet50 | ILSVRC2012_val_00007450 | 2500 | 20.19067 | 20.19066 | 20.19067 | 4.7e-07 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 0.58237 | 0.58236 | 0.58237 | 1.8e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 2000 | 0.31640 | 0.31640 | 0.31640 | 1.4e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.47899 | 1.47899 | 1.47899 | 3.8e-07 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 10000 | 0.14932 | 0.14932 | 0.14932 | 1.8e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.20351 | 0.20351 | 0.20351 | 1.8e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 5000 | 0.19723 | 0.19723 | 0.19723 | 9.3e-07 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 1000 | 0.45662 | 0.45662 | 0.45662 | 3.4e-07 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 26.17952 | 26.17960 | 26.17952 | 3.0e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 2000 | 1.85356 | 1.85356 | 1.85356 | 1.2e-06 |
| b4e2_smoke_ttba_vgg19/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 200 | 7.00323 | 7.00324 | 7.00323 | 1.2e-06 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 5000 | 5.66718 | 5.66718 | 5.66718 | 2.2e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 1000 | 29.35008 | 29.35009 | 29.35008 | 3.3e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 2500 | 67.42776 | 67.42787 | 67.42776 | 1.5e-06 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 48.57384 | 48.57386 | 48.57384 | 4.6e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 5.27679 | 5.27678 | 5.27679 | 9.2e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 17.87574 | 17.87575 | 17.87574 | 8.5e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00037360 | 1000 | 34.69960 | 34.69959 | 34.69960 | 2.6e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 500 | 113.68205 | 113.68209 | 113.68206 | 3.0e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00002148 | 2500 | 15.20801 | 15.20802 | 15.20801 | 8.0e-07 |
| b4e2_smoke_ttba_vgg19/targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 10000 | 6.49463 | 6.49463 | 6.49463 | 3.3e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 2.73115 | 2.73115 | 2.73115 | 3.3e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 1.86929 | 1.86929 | 1.86929 | 2.4e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 0.27005 | 0.27005 | 0.27005 | 1.6e-06 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 23.19956 | 23.19957 | 23.19957 | 2.9e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.62217 | 1.62217 | 1.62217 | 1.2e-06 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 0.13464 | 0.13464 | 0.13464 | 7.9e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 11.72857 | 11.72857 | 11.72857 | 8.1e-08 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 23.34890 | 23.34895 | 23.34890 | 1.8e-06 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 6.07275 | 6.07275 | 6.07275 | 1.8e-07 |
| b4e2_smoke_ttba_vit_b_32/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 16.00712 | 16.00713 | 16.00711 | 9.3e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 21.76513 | 21.76511 | 21.76513 | 4.9e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 36.65372 | 36.65373 | 36.65371 | 4.1e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 38.02777 | 38.02784 | 38.02777 | 1.7e-06 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 37.41240 | 37.41238 | 37.41240 | 4.2e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 4.27339 | 4.27339 | 4.27339 | 1.7e-06 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 2.83897 | 2.83897 | 2.83897 | 2.1e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 10.38517 | 10.38518 | 10.38517 | 1.0e-06 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 32.12716 | 32.12713 | 32.12716 | 8.6e-07 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 39.43153 | 39.43158 | 39.43153 | 1.2e-06 |
| b4e2_smoke_ttba_vit_b_32/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00046402 | 2500 | 16.21590 | 16.21590 | 16.21589 | 1.2e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 2000 | 6.72972 | 6.72971 | 6.72972 | 6.6e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 2000 | 9.62627 | 9.62628 | 9.62627 | 4.7e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 10000 | 3.09488 | 3.09488 | 3.09488 | 7.7e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 5000 | 0.27798 | 0.27798 | 0.27798 | 8.4e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 5000 | 4.30478 | 4.30478 | 4.30478 | 9.2e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 200 | 17.95688 | 17.95685 | 17.95688 | 1.5e-06 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 500 | 12.70002 | 12.70001 | 12.70002 | 9.4e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 1000 | 13.25881 | 13.25882 | 13.25881 | 3.1e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 2000 | 0.44076 | 0.44076 | 0.44076 | 4.2e-07 |
| b4e2_smoke_ttba_resnet50_places365/non_targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 1000 | 0.66427 | 0.66427 | 0.66427 | 9.0e-08 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 500 | 71.05180 | 71.05173 | 71.05179 | 9.0e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 100 | 69.88185 | 69.88175 | 69.88185 | 1.5e-06 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 100 | 76.22260 | 76.22257 | 76.22258 | 3.4e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 10000 | 16.53258 | 16.53259 | 16.53258 | 6.4e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 2000 | 47.16475 | 47.16470 | 47.16475 | 9.7e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 100 | 82.91254 | 82.91255 | 82.91254 | 9.2e-08 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00002172 | 2500 | 35.51779 | 35.51778 | 35.51779 | 2.3e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 200 | 80.12180 | 80.12186 | 80.12180 | 6.5e-07 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00023888 | 500 | 69.26503 | 69.26503 | 69.26503 | 2.9e-08 |
| b4e2_smoke_ttba_resnet50_places365/targeted | TtBA | resnet50_places365 | Places365_val_00004317 | 5000 | 38.39606 | 38.39606 | 38.39606 | 2.0e-07 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 100 | 4.94234 | 4.94234 | 4.94234 | 1.2e-06 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 1000 | 17.93623 | 17.93624 | 17.93623 | 6.8e-07 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 1000 | 13.62899 | 13.62899 | 13.62899 | 3.7e-07 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 2500 | 7.32726 | 7.32728 | 7.32726 | 2.2e-06 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 100 | 36.07731 | 36.07732 | 36.07731 | 9.6e-08 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 10000 | 0.48276 | 0.48276 | 0.48276 | 5.7e-08 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2000 | 1.05277 | 1.05277 | 1.05277 | 2.5e-07 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2500 | 0.92687 | 0.92688 | 0.92687 | 3.0e-06 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 5000 | 3.80619 | 3.80618 | 3.80619 | 6.2e-07 |
| b4e2_smoke_ttba_densenet161_places365/non_targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 2000 | 9.42278 | 9.42280 | 9.42278 | 1.8e-06 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2500 | 13.50310 | 13.50311 | 13.50310 | 7.2e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 100 | 43.92746 | 43.92748 | 43.92746 | 4.2e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 2000 | 21.55757 | 21.55758 | 21.55757 | 1.8e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 500 | 65.54054 | 65.54047 | 65.54054 | 1.2e-06 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 10000 | 5.85498 | 5.85498 | 5.85498 | 5.8e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00002172 | 1000 | 48.94812 | 48.94814 | 48.94812 | 5.1e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00023888 | 10000 | 7.75049 | 7.75048 | 7.75049 | 1.2e-06 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 500 | 38.11999 | 38.11998 | 38.11999 | 4.2e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 5000 | 4.71696 | 4.71697 | 4.71696 | 7.5e-07 |
| b4e2_smoke_ttba_densenet161_places365/targeted | TtBA | densenet161_places365 | Places365_val_00004317 | 2000 | 18.09189 | 18.09191 | 18.09189 | 9.1e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 26.46221 | 26.46224 | 26.46221 | 1.1e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 500 | 27.14798 | 27.14805 | 27.14797 | 2.6e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 200 | 3.44851 | 3.44850 | 3.44851 | 1.6e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2500 | 17.61932 | 17.61940 | 17.61931 | 4.9e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 1000 | 22.55674 | 22.55677 | 22.55673 | 1.3e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 1000 | 19.96533 | 19.96537 | 19.96532 | 2.4e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 8.91006 | 8.91004 | 8.91006 | 1.9e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 14.19322 | 14.19327 | 14.19323 | 3.1e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 0.56849 | 0.56849 | 0.56849 | 2.2e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 0.79847 | 0.79847 | 0.79847 | 2.1e-06 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 2000 | 92.30684 | 92.30687 | 92.30683 | 2.9e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 128.16905 | 128.16903 | 128.16905 | 1.9e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 110.71509 | 110.71515 | 110.71507 | 5.6e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 118.83317 | 118.83316 | 118.83317 | 1.1e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 1000 | 81.99638 | 81.99640 | 81.99638 | 1.9e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 41.29353 | 41.29350 | 41.29353 | 7.7e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 95.53925 | 95.53930 | 95.53926 | 5.1e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 64.59241 | 64.59240 | 64.59241 | 2.4e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 75.46986 | 75.46984 | 75.46984 | 2.4e-07 |
| b4e2_smoke_ttba_Salman2020Do_R50/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 5000 | 76.78281 | 76.78293 | 76.78281 | 1.5e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 12.44827 | 12.44827 | 12.44827 | 6.3e-08 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 200 | 64.66840 | 64.66825 | 64.66840 | 2.2e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 2500 | 23.90650 | 23.90647 | 23.90649 | 1.2e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 500 | 67.23110 | 67.23119 | 67.23109 | 1.2e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 500 | 35.91553 | 35.91557 | 35.91553 | 1.2e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 200 | 81.46277 | 81.46281 | 81.46275 | 4.8e-07 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 100 | 107.64500 | 107.64523 | 107.64504 | 2.1e-06 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 2500 | 27.54531 | 27.54529 | 27.54531 | 8.5e-07 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 4.27450 | 4.27450 | 4.27450 | 3.3e-07 |
| b4e2_smoke_cgba_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 2000 | 8.73333 | 8.73335 | 8.73333 | 1.8e-06 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 1000 | 28.89596 | 28.89598 | 28.89596 | 8.7e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 10000 | 19.42974 | 19.42976 | 19.42974 | 6.1e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 2000 | 25.86901 | 25.86904 | 25.86901 | 1.2e-06 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 10000 | 6.57676 | 6.57676 | 6.57676 | 2.6e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 2000 | 21.42531 | 21.42531 | 21.42531 | 1.8e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 500 | 36.00297 | 36.00298 | 36.00297 | 2.7e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 2500 | 54.87243 | 54.87239 | 54.87243 | 7.5e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 100 | 98.43659 | 98.43657 | 98.43658 | 2.3e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 100 | 42.00632 | 42.00630 | 42.00632 | 4.7e-07 |
| b4e2_smoke_cgba_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 500 | 33.78445 | 33.78443 | 33.78445 | 8.0e-07 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 1000 | 0.43738 | 0.43738 | 0.43738 | 1.8e-07 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 500 | 7.36261 | 7.36261 | 7.36261 | 9.5e-09 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.30179 | 1.30180 | 1.30179 | 1.2e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 10000 | 0.09305 | 0.09305 | 0.09305 | 1.0e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 30.04232 | 30.04238 | 30.04231 | 2.1e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 0.57386 | 0.57386 | 0.57386 | 1.4e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 200 | 1.41453 | 1.41453 | 1.41453 | 1.5e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 100 | 11.18408 | 11.18409 | 11.18408 | 1.2e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 500 | 4.57859 | 4.57860 | 4.57859 | 1.5e-06 |
| b4e2_smoke_cgba_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.31042 | 0.31042 | 0.31042 | 6.6e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 2000 | 65.16906 | 65.16912 | 65.16906 | 9.7e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 2500 | 56.83735 | 56.83729 | 56.83735 | 1.1e-06 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 5000 | 8.56552 | 8.56551 | 8.56552 | 7.2e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 10000 | 4.58001 | 4.58001 | 4.58001 | 3.2e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 200 | 47.30540 | 47.30538 | 47.30540 | 3.4e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 2000 | 19.96654 | 19.96653 | 19.96654 | 8.3e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 200 | 45.40151 | 45.40147 | 45.40151 | 7.4e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 5000 | 6.79848 | 6.79847 | 6.79848 | 1.5e-06 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 100 | 123.05856 | 123.05864 | 123.05856 | 6.7e-07 |
| b4e2_smoke_cgba_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 2500 | 16.07424 | 16.07422 | 16.07423 | 9.7e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 37.84389 | 37.84388 | 37.84389 | 1.1e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 18.52026 | 18.52024 | 18.52026 | 1.2e-06 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.57126 | 1.57126 | 1.57126 | 1.5e-06 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 2.96016 | 2.96016 | 2.96016 | 8.4e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 21.55983 | 21.55982 | 21.55983 | 4.8e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 107.45505 | 107.45512 | 107.45504 | 6.8e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 6.46405 | 6.46405 | 6.46405 | 3.6e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 200 | 60.38816 | 60.38822 | 60.38816 | 1.1e-06 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 0.19908 | 0.19908 | 0.19908 | 1.5e-07 |
| b4e2_smoke_cgba_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 0.12353 | 0.12353 | 0.12353 | 3.2e-06 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 3.66486 | 3.66486 | 3.66486 | 1.1e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 44.47025 | 44.47022 | 44.47025 | 7.4e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 19.48162 | 19.48160 | 19.48162 | 8.6e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 15.68565 | 15.68565 | 15.68565 | 6.1e-08 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 16.22276 | 16.22275 | 16.22276 | 8.9e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 40.62703 | 40.62707 | 40.62702 | 1.1e-06 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 7.09683 | 7.09683 | 7.09683 | 1.1e-06 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 100 | 70.36873 | 70.36870 | 70.36872 | 4.4e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 1000 | 23.19506 | 23.19508 | 23.19507 | 5.4e-07 |
| b4e2_smoke_cgba_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 38.91557 | 38.91554 | 38.91557 | 6.5e-07 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 10000 | 4.17768 | 4.17768 | 4.17768 | 1.3e-06 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 500 | 11.31741 | 11.31740 | 11.31741 | 1.3e-06 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 5000 | 4.10811 | 4.10810 | 4.10811 | 1.6e-06 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 2000 | 0.41173 | 0.41173 | 0.41173 | 1.6e-07 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 1000 | 15.79433 | 15.79433 | 15.79433 | 1.5e-08 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 200 | 26.55421 | 26.55422 | 26.55421 | 3.7e-07 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 2500 | 0.35958 | 0.35958 | 0.35958 | 1.0e-07 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 2500 | 9.72056 | 9.72057 | 9.72056 | 1.3e-06 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 200 | 1.44717 | 1.44717 | 1.44717 | 5.1e-07 |
| b4e2_smoke_cgba_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 100 | 2.33605 | 2.33605 | 2.33605 | 3.4e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 500 | 63.50596 | 63.50601 | 63.50596 | 7.7e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 10000 | 17.05869 | 17.05867 | 17.05869 | 1.4e-06 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 2500 | 46.80900 | 46.80897 | 46.80899 | 6.0e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 1000 | 57.30161 | 57.30155 | 57.30161 | 1.0e-06 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 1000 | 51.12439 | 51.12440 | 51.12438 | 2.9e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 2500 | 36.06339 | 36.06339 | 36.06338 | 1.1e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 200 | 80.47553 | 80.47541 | 80.47553 | 1.5e-06 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 10000 | 26.39993 | 26.39995 | 26.39993 | 4.3e-07 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 100 | 75.12183 | 75.12192 | 75.12183 | 1.1e-06 |
| b4e2_smoke_cgba_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 100 | 86.54550 | 86.54554 | 86.54550 | 4.6e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 500 | 17.49729 | 17.49732 | 17.49729 | 1.7e-06 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 5000 | 4.09173 | 4.09173 | 4.09173 | 3.8e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 5000 | 0.64955 | 0.64955 | 0.64955 | 2.6e-06 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 2000 | 9.17729 | 9.17729 | 9.17729 | 6.2e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 100 | 41.75130 | 41.75134 | 41.75130 | 8.7e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 10000 | 0.48094 | 0.48094 | 0.48094 | 1.4e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 200 | 38.07653 | 38.07652 | 38.07653 | 6.0e-08 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 500 | 2.28896 | 2.28896 | 2.28896 | 2.0e-07 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 100 | 36.59802 | 36.59806 | 36.59803 | 1.2e-06 |
| b4e2_smoke_cgba_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 200 | 3.61657 | 3.61657 | 3.61657 | 7.1e-08 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 10000 | 6.16694 | 6.16693 | 6.16693 | 1.0e-06 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 500 | 31.17864 | 31.17866 | 31.17864 | 8.2e-07 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 5000 | 14.22347 | 14.22347 | 14.22347 | 9.7e-08 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 100 | 42.61666 | 42.61661 | 42.61666 | 1.1e-06 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 1000 | 33.72204 | 33.72203 | 33.72204 | 4.9e-07 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 5000 | 4.47320 | 4.47319 | 4.47320 | 1.1e-06 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 2500 | 27.06440 | 27.06446 | 27.06440 | 2.3e-06 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 500 | 41.91839 | 41.91843 | 41.91839 | 9.1e-07 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 200 | 74.42037 | 74.42047 | 74.42036 | 1.3e-06 |
| b4e2_smoke_cgba_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 100 | 77.49629 | 77.49623 | 77.49630 | 8.0e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 10000 | 0.42137 | 0.42137 | 0.42137 | 1.2e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 9.13976 | 9.13977 | 9.13975 | 1.1e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 25.69146 | 25.69147 | 25.69145 | 6.3e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 7.11094 | 7.11093 | 7.11094 | 2.0e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 31.22815 | 31.22816 | 31.22815 | 1.2e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 1.22468 | 1.22468 | 1.22468 | 1.2e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 500 | 1.93839 | 1.93839 | 1.93839 | 7.2e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 0.55379 | 0.55379 | 0.55379 | 2.1e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 0.85410 | 0.85410 | 0.85410 | 5.9e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 5.68764 | 5.68765 | 5.68764 | 5.7e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 107.46318 | 107.46323 | 107.46317 | 4.6e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 200 | 115.98516 | 115.98519 | 115.98517 | 2.5e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 104.24508 | 104.24508 | 104.24507 | 7.3e-08 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 56.76941 | 56.76940 | 56.76941 | 1.1e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 117.29975 | 117.29982 | 117.29974 | 5.9e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 94.28175 | 94.28185 | 94.28175 | 1.0e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 200 | 85.04713 | 85.04727 | 85.04713 | 1.7e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 500 | 79.70351 | 79.70366 | 79.70350 | 1.9e-06 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 69.88801 | 69.88798 | 69.88800 | 4.0e-07 |
| b4e2_smoke_cgba_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 51.39510 | 51.39515 | 51.39510 | 9.8e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 100 | 99.05638 | 99.05592 | 99.05632 | 4.6e-06 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 2000 | 10.04000 | 10.03999 | 10.03999 | 3.4e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 500 | 107.21599 | 107.21561 | 107.21601 | 3.5e-06 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 2000 | 55.99341 | 55.99345 | 55.99342 | 8.2e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 1000 | 87.16400 | 87.16395 | 87.16401 | 6.4e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 38.25051 | 38.25051 | 38.25054 | 8.0e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 200 | 107.21678 | 107.21654 | 107.21680 | 2.2e-06 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 10000 | 19.79452 | 19.79454 | 19.79450 | 1.2e-06 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 500 | 92.27279 | 92.27282 | 92.27276 | 3.7e-07 |
| b4e2_smoke_arsopt_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 100 | 107.57796 | 107.57745 | 107.57799 | 4.7e-06 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 10000 | 11.95406 | 11.95409 | 11.95406 | 2.0e-06 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 1000 | 38.01825 | 38.01827 | 38.01825 | 4.8e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 500 | 99.71770 | 99.71775 | 99.71770 | 4.2e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 500 | 47.01183 | 47.01179 | 47.01183 | 7.7e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 5000 | 20.46529 | 20.46529 | 20.46529 | 9.3e-08 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 18.51800 | 18.51800 | 18.51800 | 4.4e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 2500 | 61.57011 | 61.57010 | 61.57011 | 1.5e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 100 | 103.36298 | 103.36301 | 103.36298 | 2.2e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 2000 | 32.06754 | 32.06753 | 32.06753 | 2.4e-07 |
| b4e2_smoke_arsopt_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 2500 | 27.97534 | 27.97536 | 27.97534 | 7.8e-07 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 100 | 34.00189 | 34.00198 | 34.00188 | 2.5e-06 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 500 | 9.10839 | 9.10840 | 9.10839 | 9.1e-07 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 200 | 30.39413 | 30.39424 | 30.39413 | 3.3e-06 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 10000 | 0.22718 | 0.22718 | 0.22718 | 1.7e-06 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 5000 | 1.40678 | 1.40678 | 1.40678 | 7.9e-07 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 5000 | 0.42421 | 0.42422 | 0.42421 | 7.9e-07 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 100 | 10.85875 | 10.85876 | 10.85874 | 8.0e-07 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 200 | 10.96557 | 10.96561 | 10.96557 | 3.5e-06 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 1000 | 20.46704 | 20.46708 | 20.46704 | 2.0e-06 |
| b4e2_smoke_arsopt_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 2000 | 0.80637 | 0.80637 | 0.80637 | 7.9e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 5000 | 22.60891 | 22.60894 | 22.60891 | 1.5e-06 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 100 | 130.43442 | 130.43433 | 130.43439 | 7.0e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 2500 | 36.72248 | 36.72248 | 36.72248 | 2.1e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 200 | 52.55802 | 52.55828 | 52.55807 | 5.0e-06 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 1000 | 104.88872 | 104.88863 | 104.88872 | 8.1e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 100 | 52.55802 | 52.55828 | 52.55807 | 5.0e-06 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 2000 | 32.76114 | 32.76112 | 32.76114 | 8.1e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 2000 | 71.05802 | 71.05808 | 71.05802 | 8.8e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 1000 | 46.30874 | 46.30873 | 46.30873 | 2.7e-07 |
| b4e2_smoke_arsopt_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 100 | 61.50502 | 61.50509 | 61.50502 | 1.2e-06 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 104.08455 | 104.08448 | 104.08455 | 6.5e-07 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 92.60233 | 92.60227 | 92.60231 | 7.3e-07 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 99.87387 | 99.87364 | 99.87386 | 2.3e-06 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 4.95118 | 4.95118 | 4.95118 | 1.0e-06 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 500 | 77.61648 | 77.61656 | 77.61647 | 1.0e-06 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 34.83643 | 34.83644 | 34.83643 | 2.6e-07 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 25.43427 | 25.43429 | 25.43427 | 7.1e-07 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 104.10771 | 104.10762 | 104.10771 | 9.0e-07 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 83.56995 | 83.57006 | 83.56994 | 1.3e-06 |
| b4e2_smoke_arsopt_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 57.89168 | 57.89166 | 57.89168 | 4.0e-07 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 36.14140 | 36.14135 | 36.14140 | 1.5e-06 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 100 | 73.27971 | 73.28019 | 73.27973 | 6.6e-06 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 48.87143 | 48.87180 | 48.87138 | 7.5e-06 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 14.61345 | 14.61342 | 14.61345 | 1.8e-06 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 12.21362 | 12.21362 | 12.21362 | 2.6e-07 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 8.00754 | 8.00756 | 8.00754 | 1.6e-06 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 7.51542 | 7.51543 | 7.51542 | 7.5e-07 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 39.69154 | 39.69153 | 39.69154 | 3.3e-07 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 18.05935 | 18.05936 | 18.05935 | 4.0e-07 |
| b4e2_smoke_arsopt_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 1000 | 30.21271 | 30.21275 | 30.21270 | 1.4e-06 |

- rows checked: 420

## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)

- **b4e2_smoke_LGC_RC_EGC_resnet50/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 49.8; 200: 1.00 / 44.6; 500: 1.00 / 38.5; 1000: 1.00 / 28.5; 2000: 1.00 / 13.6; 2500: 1.00 / 10.5; 5000: 1.00 / 6.09; 10000: 1.00 / 3.8
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_resnet50/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 47.8; 200: 1.00 / 43.6; 500: 1.00 / 34.9; 1000: 1.00 / 29.5; 2000: 1.00 / 21.7; 2500: 1.00 / 18.1; 5000: 1.00 / 11.2; 10000: 1.00 / 7.17
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_vgg19/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 4.21; 200: 1.00 / 2.21; 500: 1.00 / 1.26; 1000: 1.00 / 0.791; 2000: 1.00 / 0.398; 2500: 1.00 / 0.312; 5000: 1.00 / 0.209; 10000: 1.00 / 0.153
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_vgg19/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 53; 200: 1.00 / 48.5; 500: 1.00 / 40.3; 1000: 1.00 / 33.5; 2000: 1.00 / 24.5; 2500: 1.00 / 21.7; 5000: 1.00 / 11.8; 10000: 1.00 / 7.44
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_vit_b_32/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 29.2; 200: 1.00 / 21.4; 500: 1.00 / 8.93; 1000: 1.00 / 5.11; 2000: 1.00 / 2.92; 2500: 1.00 / 2.47; 5000: 1.00 / 1.66; 10000: 1.00 / 1.22
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_vit_b_32/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 50.2; 200: 1.00 / 43.9; 500: 1.00 / 36.2; 1000: 1.00 / 25.9; 2000: 1.00 / 13.1; 2500: 1.00 / 9.88; 5000: 1.00 / 5.91; 10000: 1.00 / 3.14
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 27.1; 200: 1.00 / 21; 500: 1.00 / 16.7; 1000: 1.00 / 11.1; 2000: 1.00 / 7.28; 2500: 1.00 / 6.19; 5000: 1.00 / 3.93; 10000: 1.00 / 2.47
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 81.7; 200: 1.00 / 72.4; 500: 1.00 / 60; 1000: 1.00 / 51.5; 2000: 1.00 / 40.9; 2500: 1.00 / 39; 5000: 1.00 / 30.7; 10000: 1.00 / 15.1
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 20.2; 200: 1.00 / 15.6; 500: 1.00 / 11; 1000: 1.00 / 7.37; 2000: 1.00 / 5.06; 2500: 1.00 / 4.29; 5000: 1.00 / 2.72; 10000: 1.00 / 1.79
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 55.6; 200: 1.00 / 49.6; 500: 1.00 / 38.9; 1000: 1.00 / 27.1; 2000: 1.00 / 16.4; 2500: 1.00 / 14.2; 5000: 1.00 / 8.85; 10000: 1.00 / 6.07
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted/non_targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 20.3; 200: 1.00 / 8.88; 500: 1.00 / 3.6; 1000: 1.00 / 2.38; 2000: 1.00 / 2.06; 2500: 1.00 / 1.99; 5000: 1.00 / 1.75; 10000: 1.00 / 1.54
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted/targeted LGC_RC_EGC** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 118; 200: 1.00 / 117; 500: 1.00 / 95.5; 1000: 1.00 / 68.4; 2000: 1.00 / 59.9; 2500: 1.00 / 55.5; 5000: 1.00 / 24.6; 10000: 1.00 / 20.7
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_resnet50/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 82.7; 200: 1.00 / 70.7; 500: 1.00 / 55.1; 1000: 1.00 / 23.7; 2000: 1.00 / 12.3; 2500: 1.00 / 9.57; 5000: 1.00 / 4.47; 10000: 1.00 / 2.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_resnet50/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 47; 200: 1.00 / 43.7; 500: 1.00 / 36.3; 1000: 1.00 / 29.9; 2000: 1.00 / 22.7; 2500: 1.00 / 20.2; 5000: 1.00 / 14.3; 10000: 1.00 / 8.59
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_vgg19/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.8; 200: 1.00 / 7; 500: 1.00 / 0.768; 1000: 1.00 / 0.457; 2000: 1.00 / 0.316; 2500: 1.00 / 0.271; 5000: 1.00 / 0.197; 10000: 1.00 / 0.149
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_vgg19/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 48.6; 200: 1.00 / 46.2; 500: 1.00 / 41.7; 1000: 1.00 / 34.7; 2000: 1.00 / 21.6; 2500: 1.00 / 18.8; 5000: 1.00 / 9.8; 10000: 1.00 / 5.28
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_vit_b_32/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 47.3; 200: 1.00 / 36.2; 500: 1.00 / 23.3; 1000: 1.00 / 11.7; 2000: 1.00 / 6.07; 2500: 1.00 / 4.85; 5000: 1.00 / 1.62; 10000: 1.00 / 0.787
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_vit_b_32/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 43.1; 200: 1.00 / 39.4; 500: 1.00 / 33.6; 1000: 1.00 / 25.1; 2000: 1.00 / 17.8; 2500: 1.00 / 16.2; 5000: 1.00 / 8.62; 10000: 1.00 / 4.27
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_resnet50_places365/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 20; 200: 1.00 / 18; 500: 1.00 / 12.7; 1000: 1.00 / 8.83; 2000: 1.00 / 6.73; 2500: 1.00 / 6.12; 5000: 1.00 / 4.3; 10000: 1.00 / 3.09
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_resnet50_places365/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 76.2; 200: 1.00 / 74.1; 500: 1.00 / 69.3; 1000: 1.00 / 54.8; 2000: 1.00 / 47.2; 2500: 1.00 / 45.7; 5000: 1.00 / 30.7; 10000: 1.00 / 16.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_densenet161_places365/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 29.3; 200: 1.00 / 27.2; 500: 1.00 / 19.6; 1000: 1.00 / 13.6; 2000: 1.00 / 8.78; 2500: 1.00 / 7.33; 5000: 1.00 / 3.81; 10000: 1.00 / 2.24
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_densenet161_places365/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 45.8; 200: 1.00 / 45.6; 500: 1.00 / 42.2; 1000: 1.00 / 30.7; 2000: 1.00 / 21.6; 2500: 1.00 / 19.7; 5000: 1.00 / 12.6; 10000: 1.00 / 5.85
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_Salman2020Do_R50/non_targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 32.7; 200: 1.00 / 30.5; 500: 1.00 / 26.5; 1000: 1.00 / 20; 2000: 1.00 / 14.4; 2500: 1.00 / 12.8; 5000: 1.00 / 8.91; 10000: 1.00 / 6.61
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_ttba_Salman2020Do_R50/targeted TtBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 119; 200: 1.00 / 114; 500: 1.00 / 108; 1000: 1.00 / 102; 2000: 1.00 / 78.3; 2500: 1.00 / 75.5; 5000: 1.00 / 64.6; 10000: 1.00 / 44.8
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_resnet50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 112; 200: 1.00 / 81.5; 500: 1.00 / 56.3; 1000: 1.00 / 42.5; 2000: 1.00 / 28.2; 2500: 1.00 / 23.9; 5000: 1.00 / 12.4; 10000: 1.00 / 6.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_resnet50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 49; 200: 1.00 / 44.4; 500: 1.00 / 36; 1000: 1.00 / 32; 2000: 1.00 / 25.9; 2500: 1.00 / 23.3; 5000: 1.00 / 16; 10000: 1.00 / 9.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_vgg19/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.2; 200: 1.00 / 8.1; 500: 1.00 / 4.58; 1000: 1.00 / 1.96; 2000: 1.00 / 0.31; 2500: 1.00 / 0.264; 5000: 1.00 / 0.19; 10000: 1.00 / 0.147
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_vgg19/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 51.3; 200: 1.00 / 47.3; 500: 1.00 / 38.7; 1000: 1.00 / 31.1; 2000: 1.00 / 20; 2500: 1.00 / 16.8; 5000: 1.00 / 8.57; 10000: 1.00 / 4.58
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_vit_b_32/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 87.8; 200: 1.00 / 60.4; 500: 1.00 / 21.6; 1000: 1.00 / 10.9; 2000: 1.00 / 6.46; 2500: 1.00 / 4.63; 5000: 1.00 / 1.57; 10000: 1.00 / 1.19
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_vit_b_32/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 44.5; 200: 1.00 / 40.6; 500: 1.00 / 33; 1000: 1.00 / 26.2; 2000: 1.00 / 16.2; 2500: 1.00 / 13.3; 5000: 1.00 / 7.1; 10000: 1.00 / 3.66
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_resnet50_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 17.9; 200: 1.00 / 15.7; 500: 1.00 / 11.3; 1000: 1.00 / 8.88; 2000: 1.00 / 6.46; 2500: 1.00 / 5.83; 5000: 1.00 / 4.11; 10000: 1.00 / 2.93
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_resnet50_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 75.1; 200: 1.00 / 74; 500: 1.00 / 63.5; 1000: 1.00 / 57.3; 2000: 1.00 / 46.7; 2500: 1.00 / 43.4; 5000: 1.00 / 29.6; 10000: 1.00 / 17.1
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_densenet161_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 36.6; 200: 1.00 / 31.7; 500: 1.00 / 17.5; 1000: 1.00 / 13.2; 2000: 1.00 / 8.45; 2500: 1.00 / 7.23; 5000: 1.00 / 4.09; 10000: 1.00 / 2.46
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_densenet161_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 45.9; 200: 1.00 / 45.1; 500: 1.00 / 41.9; 1000: 1.00 / 33.7; 2000: 1.00 / 25.8; 2500: 1.00 / 22.8; 5000: 1.00 / 13.7; 10000: 1.00 / 6.17
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_Salman2020Do_R50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 31.2; 200: 1.00 / 29.7; 500: 1.00 / 22.9; 1000: 1.00 / 19; 2000: 1.00 / 14.4; 2500: 1.00 / 13.2; 5000: 1.00 / 9.37; 10000: 1.00 / 7.11
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_cgba_Salman2020Do_R50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 117; 200: 1.00 / 116; 500: 1.00 / 107; 1000: 1.00 / 102; 2000: 1.00 / 71.7; 2500: 1.00 / 69.9; 5000: 1.00 / 62.1; 10000: 1.00 / 49.3
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_resnet50/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 99.1; 200: 1.00 / 99; 500: 1.00 / 96.1; 1000: 1.00 / 75; 2000: 1.00 / 49; 2500: 1.00 / 37.9; 5000: 1.00 / 17.3; 10000: 1.00 / 6.87
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_resnet50/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 52.3; 200: 1.00 / 52.3; 500: 1.00 / 47; 1000: 1.00 / 38.9; 2000: 1.00 / 32.1; 2500: 1.00 / 28.7; 5000: 1.00 / 20.5; 10000: 1.00 / 12
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_vgg19/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.5; 200: 1.00 / 11; 500: 1.00 / 11; 1000: 1.00 / 7.37; 2000: 1.00 / 1.18; 2500: 1.00 / 0.801; 5000: 1.00 / 0.424; 10000: 1.00 / 0.319
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_vgg19/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 61.5; 200: 1.00 / 61.5; 500: 1.00 / 50.4; 1000: 1.00 / 46.3; 2000: 1.00 / 38.5; 2500: 1.00 / 36.7; 5000: 1.00 / 22.5; 10000: 1.00 / 8.95
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_vit_b_32/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 101; 200: 1.00 / 95.8; 500: 1.00 / 92.6; 1000: 1.00 / 54; 2000: 1.00 / 25.4; 2500: 1.00 / 22.4; 5000: 1.00 / 11.5; 10000: 1.00 / 3.07
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4e2_smoke_arsopt_vit_b_32/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 48.9; 200: 1.00 / 48.9; 500: 1.00 / 42.2; 1000: 1.00 / 32.9; 2000: 1.00 / 21.6; 2500: 1.00 / 18.5; 5000: 1.00 / 12.5; 10000: 1.00 / 8.01
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/human_preflight_b4e2_grid_Salman2020Do_R50_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_Salman2020Do_R50_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_densenet161_places365_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_densenet161_places365_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_resnet50_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4e2_grid_resnet50_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4e2_grid_resnet50_places365_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_resnet50_places365_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4e2_grid_vgg19_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4e2_grid_vgg19_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4e2_grid_vit_b_32_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4e2_grid_vit_b_32_targeted.png: 3 images x 4 methods at budget 10000

