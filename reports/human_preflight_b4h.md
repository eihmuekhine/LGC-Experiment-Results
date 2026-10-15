# Human pre-flight report

roots: ['Results/b4e2_smoke_LGC_RC_EGC_resnet50', 'Results/b4e2_smoke_LGC_RC_EGC_vgg19', 'Results/b4e2_smoke_LGC_RC_EGC_vit_b_32', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted', 'Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted', 'Results/b4e2_smoke_ttba_resnet50', 'Results/b4e2_smoke_ttba_vgg19', 'Results/b4e2_smoke_ttba_vit_b_32', 'Results/b4e2_smoke_ttba_resnet50_places365', 'Results/b4e2_smoke_ttba_densenet161_places365', 'Results/b4e2_smoke_ttba_Salman2020Do_R50', 'Results/b4h_t3_smoke_CGBA_resnet50', 'Results/b4h_t3_smoke_CGBA_vgg19', 'Results/b4h_t3_smoke_CGBA_vit_b_32', 'Results/b4h_t3_smoke_CGBA_resnet50_places365', 'Results/b4h_t3_smoke_CGBA_densenet161_places365', 'Results/b4h_t3_smoke_CGBA_Salman2020Do_R50', 'Results/b4h_t3_smoke_ARS_OPT_resnet50', 'Results/b4h_t3_smoke_ARS_OPT_vgg19', 'Results/b4h_t3_smoke_ARS_OPT_vit_b_32']
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
| b4h_t3_smoke_CGBA_resnet50 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 7596d79806ff |
| b4h_t3_smoke_CGBA_resnet50 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 7596d79806ff |
| b4h_t3_smoke_CGBA_vgg19 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 7596d79806ff |
| b4h_t3_smoke_CGBA_vgg19 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 7596d79806ff |
| b4h_t3_smoke_CGBA_vit_b_32 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 7596d79806ff |
| b4h_t3_smoke_CGBA_vit_b_32 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 7596d79806ff |
| b4h_t3_smoke_CGBA_resnet50_places365 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 7596d79806ff |
| b4h_t3_smoke_CGBA_resnet50_places365 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 7596d79806ff |
| b4h_t3_smoke_CGBA_densenet161_places365 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 7596d79806ff |
| b4h_t3_smoke_CGBA_densenet161_places365 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 7596d79806ff |
| b4h_t3_smoke_CGBA_Salman2020Do_R50 | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 7596d79806ff |
| b4h_t3_smoke_CGBA_Salman2020Do_R50 | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 7596d79806ff |
| b4h_t3_smoke_ARS_OPT_resnet50 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 0f454a78c615 |
| b4h_t3_smoke_ARS_OPT_resnet50 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 0f454a78c615 |
| b4h_t3_smoke_ARS_OPT_vgg19 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 0f454a78c615 |
| b4h_t3_smoke_ARS_OPT_vgg19 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 0f454a78c615 |
| b4h_t3_smoke_ARS_OPT_vit_b_32 | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 0f454a78c615 |
| b4h_t3_smoke_ARS_OPT_vit_b_32 | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 0f454a78c615 |

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
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | 3 | {'budget': 3} | 3 | 0 | 0 | 0 |

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
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 5000 | 0.92754 | 0.92754 | 0.92754 | 8.0e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 200 | 66.42878 | 66.42873 | 66.42877 | 7.6e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 2500 | 13.94320 | 13.94321 | 13.94320 | 9.4e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 500 | 72.93227 | 72.93220 | 72.93227 | 9.6e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 500 | 35.10347 | 35.10353 | 35.10347 | 1.6e-06 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 200 | 82.22847 | 82.22847 | 82.22845 | 1.9e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00015433 | 100 | 107.64500 | 107.64523 | 107.64504 | 2.1e-06 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00007450 | 2500 | 40.43368 | 40.43372 | 40.43367 | 9.8e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 5000 | 7.85360 | 7.85359 | 7.85360 | 5.4e-07 |
| b4h_t3_smoke_CGBA_resnet50/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00025151 | 2000 | 16.47438 | 16.47438 | 16.47437 | 5.0e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 1000 | 28.89596 | 28.89598 | 28.89596 | 8.7e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 10000 | 17.04081 | 17.04082 | 17.04081 | 9.1e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 2000 | 25.86901 | 25.86904 | 25.86901 | 1.2e-06 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 10000 | 6.65757 | 6.65756 | 6.65757 | 4.8e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 2000 | 21.42531 | 21.42531 | 21.42531 | 1.8e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 500 | 36.00297 | 36.00298 | 36.00297 | 2.7e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 2500 | 56.14594 | 56.14599 | 56.14594 | 9.1e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015433 | 100 | 98.43659 | 98.43657 | 98.43658 | 2.3e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00007450 | 100 | 42.00632 | 42.00630 | 42.00632 | 4.7e-07 |
| b4h_t3_smoke_CGBA_resnet50/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00025151 | 500 | 33.78445 | 33.78443 | 33.78445 | 8.0e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 1000 | 0.43266 | 0.43266 | 0.43266 | 5.5e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 500 | 8.77038 | 8.77038 | 8.77037 | 3.9e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 2500 | 1.38237 | 1.38237 | 1.38237 | 9.6e-08 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 10000 | 0.09134 | 0.09134 | 0.09134 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 100 | 30.05551 | 30.05558 | 30.05550 | 2.5e-06 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00037360 | 10000 | 0.58166 | 0.58166 | 0.58166 | 1.9e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00039161 | 200 | 1.39698 | 1.39698 | 1.39698 | 9.6e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 100 | 10.12687 | 10.12689 | 10.12687 | 1.7e-06 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 500 | 1.82243 | 1.82242 | 1.82242 | 1.3e-07 |
| b4h_t3_smoke_CGBA_vgg19/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00002148 | 2000 | 0.22065 | 0.22065 | 0.22065 | 3.1e-07 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 2000 | 61.42223 | 61.42227 | 61.42223 | 6.1e-07 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 2500 | 58.21443 | 58.21426 | 58.21442 | 2.9e-06 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 5000 | 10.39063 | 10.39064 | 10.39063 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 10000 | 5.62829 | 5.62828 | 5.62829 | 1.5e-06 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 200 | 48.14030 | 48.14029 | 48.14030 | 1.7e-07 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037360 | 2000 | 25.30400 | 25.30397 | 25.30400 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 200 | 45.40446 | 45.40443 | 45.40446 | 8.3e-07 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 5000 | 6.58104 | 6.58104 | 6.58104 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039161 | 100 | 123.05856 | 123.05864 | 123.05856 | 6.7e-07 |
| b4h_t3_smoke_CGBA_vgg19/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002148 | 2500 | 15.99140 | 15.99142 | 15.99140 | 9.8e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 37.57038 | 37.57037 | 37.57038 | 4.0e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 19.27079 | 19.27076 | 19.27079 | 1.5e-06 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 1.58255 | 1.58255 | 1.58255 | 4.7e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 3.57581 | 3.57581 | 3.57581 | 2.1e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 26.03342 | 26.03345 | 26.03342 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 107.45505 | 107.45512 | 107.45504 | 6.8e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 5.61879 | 5.61879 | 5.61879 | 5.1e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030908 | 200 | 60.48829 | 60.48835 | 60.48829 | 8.9e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 0.19837 | 0.19837 | 0.19837 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vit_b_32/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 0.12372 | 0.12372 | 0.12372 | 1.5e-06 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 10000 | 3.66486 | 3.66486 | 3.66486 | 1.1e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 44.47025 | 44.47022 | 44.47025 | 7.4e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 19.48162 | 19.48160 | 19.48162 | 8.6e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 2000 | 15.68565 | 15.68565 | 15.68565 | 6.1e-08 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 16.54394 | 16.54395 | 16.54394 | 3.8e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00024661 | 200 | 40.62703 | 40.62707 | 40.62702 | 1.1e-06 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 5000 | 7.75848 | 7.75847 | 7.75848 | 9.7e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00030908 | 100 | 70.36873 | 70.36870 | 70.36872 | 4.4e-07 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 1000 | 23.37966 | 23.37966 | 23.37966 | 7.8e-10 |
| b4h_t3_smoke_CGBA_vit_b_32/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 38.91557 | 38.91554 | 38.91557 | 6.5e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 10000 | 4.60063 | 4.60062 | 4.60063 | 1.1e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 500 | 10.49741 | 10.49738 | 10.49741 | 2.3e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00023888 | 5000 | 4.06174 | 4.06174 | 4.06174 | 4.0e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 2000 | 0.41276 | 0.41276 | 0.41276 | 1.5e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 1000 | 15.54973 | 15.54974 | 15.54973 | 6.5e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 200 | 26.89417 | 26.89416 | 26.89416 | 1.2e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 2500 | 0.36184 | 0.36184 | 0.36184 | 7.7e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00002172 | 2500 | 10.12568 | 10.12568 | 10.12568 | 9.4e-08 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 200 | 1.45768 | 1.45768 | 1.45768 | 2.2e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/non_targeted | CGBA | resnet50_places365 | Places365_val_00004317 | 100 | 2.33682 | 2.33682 | 2.33682 | 4.5e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 500 | 63.50596 | 63.50601 | 63.50596 | 7.7e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 10000 | 16.09535 | 16.09535 | 16.09535 | 4.3e-08 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 2500 | 46.80900 | 46.80897 | 46.80899 | 6.0e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 1000 | 57.30161 | 57.30155 | 57.30161 | 1.0e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 1000 | 51.08405 | 51.08404 | 51.08405 | 2.6e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00002172 | 2500 | 36.61983 | 36.61980 | 36.61984 | 8.3e-07 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 200 | 80.47553 | 80.47541 | 80.47553 | 1.5e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 10000 | 27.11556 | 27.11560 | 27.11556 | 1.5e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 100 | 75.12183 | 75.12192 | 75.12183 | 1.1e-06 |
| b4h_t3_smoke_CGBA_resnet50_places365/targeted | CGBA_H | resnet50_places365 | Places365_val_00004317 | 100 | 86.54550 | 86.54554 | 86.54550 | 4.6e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 500 | 17.38099 | 17.38097 | 17.38099 | 8.1e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 5000 | 3.97628 | 3.97628 | 3.97628 | 1.7e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 5000 | 0.64410 | 0.64410 | 0.64410 | 6.3e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 2000 | 13.45128 | 13.45128 | 13.45127 | 6.8e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 100 | 41.72100 | 41.72105 | 41.72101 | 1.1e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 10000 | 0.47921 | 0.47921 | 0.47921 | 2.4e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00023888 | 200 | 37.24584 | 37.24586 | 37.24584 | 4.6e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 500 | 2.28894 | 2.28894 | 2.28894 | 3.7e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00002172 | 100 | 36.70090 | 36.70099 | 36.70090 | 2.2e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/non_targeted | CGBA | densenet161_places365 | Places365_val_00004317 | 200 | 3.61675 | 3.61675 | 3.61675 | 6.6e-08 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 10000 | 5.97837 | 5.97837 | 5.97837 | 8.0e-08 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 500 | 31.17864 | 31.17866 | 31.17864 | 8.2e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 5000 | 14.16953 | 14.16952 | 14.16953 | 2.0e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 100 | 42.61666 | 42.61661 | 42.61666 | 1.1e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 1000 | 33.72204 | 33.72203 | 33.72204 | 4.9e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00004317 | 5000 | 4.36521 | 4.36520 | 4.36521 | 6.9e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 2500 | 27.06440 | 27.06446 | 27.06440 | 2.3e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00023888 | 500 | 41.91839 | 41.91843 | 41.91839 | 9.1e-07 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 200 | 74.42037 | 74.42047 | 74.42036 | 1.3e-06 |
| b4h_t3_smoke_CGBA_densenet161_places365/targeted | CGBA_H | densenet161_places365 | Places365_val_00002172 | 100 | 77.49629 | 77.49623 | 77.49630 | 8.0e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 10000 | 0.42484 | 0.42484 | 0.42484 | 7.0e-08 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 10000 | 2.79462 | 2.79461 | 2.79462 | 6.3e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 25.38339 | 25.38344 | 25.38339 | 2.0e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 7.06591 | 7.06590 | 7.06591 | 1.4e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 31.15835 | 31.15839 | 31.15834 | 1.3e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 1.22949 | 1.22949 | 1.22949 | 5.0e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 500 | 1.94227 | 1.94227 | 1.94227 | 1.5e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 5000 | 0.55677 | 0.55677 | 0.55677 | 1.3e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2000 | 0.85735 | 0.85736 | 0.85735 | 1.7e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 100 | 5.68656 | 5.68656 | 5.68656 | 1.7e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 500 | 107.79679 | 107.79676 | 107.79679 | 2.8e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 200 | 115.82278 | 115.82286 | 115.82278 | 7.2e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 1000 | 103.62211 | 103.62214 | 103.62211 | 3.5e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00024509 | 2500 | 60.86905 | 60.86908 | 60.86906 | 4.1e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00046453 | 100 | 117.29975 | 117.29982 | 117.29974 | 5.9e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 94.28175 | 94.28185 | 94.28175 | 1.0e-06 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 200 | 81.76640 | 81.76648 | 81.76640 | 9.4e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 500 | 78.79964 | 78.79964 | 78.79964 | 1.1e-07 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2500 | 69.16363 | 69.16363 | 69.16363 | 2.4e-08 |
| b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 10000 | 51.39684 | 51.39688 | 51.39683 | 8.1e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 100 | 107.00053 | 107.00009 | 107.00047 | 4.2e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 2000 | 11.77823 | 11.77823 | 11.77823 | 2.4e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 500 | 103.20998 | 103.20965 | 103.21000 | 3.1e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 2000 | 56.12565 | 56.12575 | 56.12566 | 1.9e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 1000 | 87.12535 | 87.12532 | 87.12536 | 3.1e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 31.59111 | 31.59111 | 31.59110 | 1.2e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 200 | 103.21046 | 103.21012 | 103.21049 | 3.4e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 10000 | 14.09719 | 14.09721 | 14.09721 | 1.3e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 500 | 92.32467 | 92.32463 | 92.32464 | 4.4e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 100 | 114.51996 | 114.51954 | 114.52000 | 3.6e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 10000 | 11.90057 | 11.90059 | 11.90057 | 1.6e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 1000 | 38.11651 | 38.11652 | 38.11651 | 2.0e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 500 | 99.71770 | 99.71775 | 99.71770 | 4.2e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 500 | 47.08271 | 47.08265 | 47.08271 | 1.2e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 5000 | 21.36633 | 21.36633 | 21.36633 | 1.3e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 5000 | 17.50982 | 17.50983 | 17.50982 | 3.5e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 2500 | 61.57463 | 61.57464 | 61.57463 | 1.1e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00015433 | 100 | 103.36298 | 103.36301 | 103.36298 | 2.2e-07 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00007450 | 2000 | 31.83581 | 31.83576 | 31.83581 | 1.7e-06 |
| b4h_t3_smoke_ARS_OPT_resnet50/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00025151 | 2500 | 25.39683 | 25.39680 | 25.39683 | 1.1e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 100 | 33.97517 | 33.97522 | 33.97517 | 1.5e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 500 | 8.83789 | 8.83790 | 8.83789 | 1.0e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 200 | 31.31774 | 31.31784 | 31.31774 | 3.1e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 10000 | 0.23178 | 0.23178 | 0.23178 | 2.8e-08 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 5000 | 1.19776 | 1.19776 | 1.19776 | 8.6e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 5000 | 0.41768 | 0.41768 | 0.41768 | 1.7e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 100 | 10.98992 | 10.98995 | 10.98992 | 2.9e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 200 | 11.62970 | 11.62975 | 11.62970 | 4.7e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 1000 | 15.45887 | 15.45889 | 15.45887 | 9.2e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 2000 | 0.80765 | 0.80765 | 0.80765 | 1.5e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 5000 | 33.05006 | 33.05009 | 33.05005 | 9.7e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 100 | 130.43442 | 130.43433 | 130.43439 | 7.0e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 2500 | 34.38728 | 34.38727 | 34.38728 | 4.5e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 200 | 52.55802 | 52.55828 | 52.55807 | 5.0e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 1000 | 105.81091 | 105.81081 | 105.81090 | 8.8e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 100 | 52.55802 | 52.55828 | 52.55807 | 5.0e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 2000 | 37.88830 | 37.88830 | 37.88830 | 1.3e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 2000 | 80.59042 | 80.59033 | 80.59042 | 1.1e-06 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002148 | 1000 | 46.32622 | 46.32621 | 46.32621 | 2.4e-07 |
| b4h_t3_smoke_ARS_OPT_vgg19/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00037360 | 100 | 61.50502 | 61.50509 | 61.50502 | 1.2e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 200 | 100.89242 | 100.89224 | 100.89242 | 1.7e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 500 | 95.72047 | 95.72023 | 95.72045 | 2.4e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 100.89178 | 100.89173 | 100.89177 | 4.3e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 5.09055 | 5.09054 | 5.09055 | 1.3e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 500 | 80.85155 | 80.85158 | 80.85155 | 3.0e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 2000 | 35.85210 | 35.85212 | 35.85210 | 5.6e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 26.54068 | 26.54069 | 26.54068 | 5.9e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 108.53977 | 108.53959 | 108.53977 | 1.6e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 93.56469 | 93.56473 | 93.56469 | 4.4e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 1000 | 58.99908 | 58.99916 | 58.99907 | 1.5e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 500 | 36.14088 | 36.14085 | 36.14088 | 7.7e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 100 | 73.27971 | 73.28019 | 73.27973 | 6.6e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 48.87143 | 48.87180 | 48.87138 | 7.5e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 5000 | 14.39863 | 14.39862 | 14.39863 | 9.9e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 5000 | 11.22581 | 11.22581 | 11.22581 | 2.1e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 10000 | 8.18393 | 8.18394 | 8.18393 | 1.4e-06 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 10000 | 6.48801 | 6.48801 | 6.48801 | 2.1e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 100 | 39.69154 | 39.69153 | 39.69154 | 3.3e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030908 | 2500 | 18.82843 | 18.82842 | 18.82843 | 9.0e-07 |
| b4h_t3_smoke_ARS_OPT_vit_b_32/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00046402 | 1000 | 30.19135 | 30.19137 | 30.19135 | 7.1e-07 |

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
- **b4h_t3_smoke_CGBA_resnet50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 112; 200: 1.00 / 82.2; 500: 1.00 / 57.3; 1000: 1.00 / 41.1; 2000: 1.00 / 20.9; 2500: 1.00 / 13.9; 5000: 1.00 / 7.85; 10000: 1.00 / 4.23
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_resnet50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 49; 200: 1.00 / 44.4; 500: 1.00 / 36; 1000: 1.00 / 32; 2000: 1.00 / 25.9; 2500: 1.00 / 23.3; 5000: 1.00 / 16; 10000: 1.00 / 9.33
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_vgg19/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 10.1; 200: 1.00 / 5; 500: 1.00 / 1.82; 1000: 1.00 / 0.481; 2000: 1.00 / 0.306; 2500: 1.00 / 0.264; 5000: 1.00 / 0.19; 10000: 1.00 / 0.147
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_vgg19/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 52.1; 200: 1.00 / 48.1; 500: 1.00 / 39; 1000: 1.00 / 33; 2000: 1.00 / 25.3; 2500: 1.00 / 20.3; 5000: 1.00 / 10.4; 10000: 1.00 / 5.63
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_vit_b_32/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 87.8; 200: 1.00 / 60.5; 500: 1.00 / 26; 1000: 1.00 / 11; 2000: 1.00 / 5.62; 2500: 1.00 / 3.25; 5000: 1.00 / 1.58; 10000: 1.00 / 1.2
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_vit_b_32/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 44.5; 200: 1.00 / 40.6; 500: 1.00 / 33; 1000: 1.00 / 26.2; 2000: 1.00 / 16.5; 2500: 1.00 / 13.6; 5000: 1.00 / 7.76; 10000: 1.00 / 3.66
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_resnet50_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 17.9; 200: 1.00 / 15.4; 500: 1.00 / 10.5; 1000: 1.00 / 8.43; 2000: 1.00 / 6.24; 2500: 1.00 / 5.67; 5000: 1.00 / 4.06; 10000: 1.00 / 2.88
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_resnet50_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 75.1; 200: 1.00 / 73.6; 500: 1.00 / 63.5; 1000: 1.00 / 55.3; 2000: 1.00 / 44.8; 2500: 1.00 / 40.9; 5000: 1.00 / 26.1; 10000: 1.00 / 16.1
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_densenet161_places365/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 36.7; 200: 1.00 / 27.2; 500: 1.00 / 17.4; 1000: 1.00 / 12.7; 2000: 1.00 / 8.18; 2500: 1.00 / 6.94; 5000: 1.00 / 3.98; 10000: 1.00 / 2.49
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_densenet161_places365/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 45.9; 200: 1.00 / 45.1; 500: 1.00 / 41.9; 1000: 1.00 / 33.7; 2000: 1.00 / 25.8; 2500: 1.00 / 22.8; 5000: 1.00 / 13.8; 10000: 1.00 / 5.98
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_Salman2020Do_R50/non_targeted CGBA** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 31.2; 200: 1.00 / 29.4; 500: 1.00 / 22.9; 1000: 1.00 / 18.9; 2000: 1.00 / 14.2; 2500: 1.00 / 12.9; 5000: 1.00 / 9.28; 10000: 1.00 / 2.79
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_CGBA_Salman2020Do_R50/targeted CGBA_H** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 117; 200: 1.00 / 116; 500: 1.00 / 108; 1000: 1.00 / 102; 2000: 1.00 / 70.9; 2500: 1.00 / 69.2; 5000: 1.00 / 61.8; 10000: 1.00 / 50.4
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_resnet50/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 107; 200: 1.00 / 103; 500: 1.00 / 103; 1000: 1.00 / 83.5; 2000: 1.00 / 56.1; 2500: 1.00 / 47.4; 5000: 1.00 / 31.6; 10000: 1.00 / 14.1
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_resnet50/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 52.3; 200: 1.00 / 52.3; 500: 1.00 / 47.1; 1000: 1.00 / 40.3; 2000: 1.00 / 31.8; 2500: 1.00 / 30.8; 5000: 1.00 / 21.4; 10000: 1.00 / 11.9
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_vgg19/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 11.6; 200: 1.00 / 11.6; 500: 1.00 / 11.4; 1000: 1.00 / 6.77; 2000: 1.00 / 1.03; 2500: 1.00 / 0.732; 5000: 1.00 / 0.418; 10000: 1.00 / 0.317
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_vgg19/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 61.5; 200: 1.00 / 61.5; 500: 1.00 / 50.4; 1000: 1.00 / 46.3; 2000: 1.00 / 37.9; 2500: 1.00 / 34.4; 5000: 1.00 / 21.9; 10000: 1.00 / 9.64
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_vit_b_32/non_targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 107; 200: 1.00 / 95.7; 500: 1.00 / 95.7; 1000: 1.00 / 55.4; 2000: 1.00 / 26.5; 2500: 1.00 / 20.8; 5000: 1.00 / 11.8; 10000: 1.00 / 3.16
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b4h_t3_smoke_ARS_OPT_vit_b_32/targeted ARS_OPT** (images 3): found rate / median pix_l2 per budget: 100: 1.00 / 48.9; 200: 1.00 / 48.9; 500: 1.00 / 42.2; 1000: 1.00 / 32.8; 2000: 1.00 / 21.9; 2500: 1.00 / 18.8; 5000: 1.00 / 12.6; 10000: 1.00 / 7.75
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 7. Degeneracy (silent collapse: identical methods, constant outputs, never found, duplicated inputs)

- no degenerate pattern found

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/human_preflight_b4h_grid_Salman2020Do_R50_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_Salman2020Do_R50_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_densenet161_places365_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_densenet161_places365_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_resnet50_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4h_grid_resnet50_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4h_grid_resnet50_places365_non_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_resnet50_places365_targeted.png: 3 images x 3 methods at budget 10000
- reports/human_preflight_b4h_grid_vgg19_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4h_grid_vgg19_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4h_grid_vit_b_32_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b4h_grid_vit_b_32_targeted.png: 3 images x 4 methods at budget 10000

