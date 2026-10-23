# Human pre-flight report

roots: ['Results/b5_e10_ARS_OPT_densenet161_places365_NT', 'Results/b5_e10_ARS_OPT_densenet161_places365_T', 'Results/b5_e10_ARS_OPT_resnet50_places365_NT', 'Results/b5_e10_ARS_OPT_resnet50_places365_T', 'Results/b5_e10_CGBA_H_densenet161_places365_T', 'Results/b5_e10_CGBA_H_resnet50_places365_T', 'Results/b5_e10_CGBA_densenet161_places365_NT', 'Results/b5_e10_CGBA_resnet50_places365_NT', 'Results/b5_e10_LGC_RC_EGC_densenet161_places365_NT', 'Results/b5_e10_LGC_RC_EGC_densenet161_places365_T', 'Results/b5_e10_LGC_RC_EGC_resnet50_places365_NT', 'Results/b5_e10_LGC_RC_EGC_resnet50_places365_T', 'Results/b5_e10_TtBA_densenet161_places365_NT', 'Results/b5_e10_TtBA_densenet161_places365_T', 'Results/b5_e10_TtBA_resnet50_places365_NT', 'Results/b5_e10_TtBA_resnet50_places365_T', 'Results/b5_e11_ARS_OPT_Salman2020Do_R50_NT', 'Results/b5_e11_ARS_OPT_Salman2020Do_R50_T', 'Results/b5_e11_CGBA_H_Salman2020Do_R50_T', 'Results/b5_e11_CGBA_Salman2020Do_R50_NT', 'Results/b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT', 'Results/b5_e11_LGC_RC_EGC_Salman2020Do_R50_T', 'Results/b5_e11_TtBA_Salman2020Do_R50_NT', 'Results/b5_e11_TtBA_Salman2020Do_R50_T', 'Results/b5_main_ARS_OPT_resnet50_NT', 'Results/b5_main_ARS_OPT_resnet50_T', 'Results/b5_main_ARS_OPT_vgg19_NT', 'Results/b5_main_ARS_OPT_vgg19_T', 'Results/b5_main_ARS_OPT_vit_b_32_NT', 'Results/b5_main_ARS_OPT_vit_b_32_T', 'Results/b5_main_CGBA_H_resnet50_T', 'Results/b5_main_CGBA_H_vgg19_T', 'Results/b5_main_CGBA_H_vit_b_32_T', 'Results/b5_main_CGBA_resnet50_NT', 'Results/b5_main_CGBA_vgg19_NT', 'Results/b5_main_CGBA_vit_b_32_NT', 'Results/b5_main_LGC_RC_EGC_resnet50_NT', 'Results/b5_main_LGC_RC_EGC_resnet50_T', 'Results/b5_main_LGC_RC_EGC_vgg19_NT', 'Results/b5_main_LGC_RC_EGC_vgg19_T', 'Results/b5_main_LGC_RC_EGC_vit_b_32_NT', 'Results/b5_main_LGC_RC_EGC_vit_b_32_T', 'Results/b5_main_TtBA_resnet50_NT', 'Results/b5_main_TtBA_resnet50_T', 'Results/b5_main_TtBA_vgg19_NT', 'Results/b5_main_TtBA_vgg19_T', 'Results/b5_main_TtBA_vit_b_32_NT', 'Results/b5_main_TtBA_vit_b_32_T', 'Results/appendix_cgba_d8only_Salman2020Do_R50', 'Results/appendix_cgba_d8only_resnet50', 'Results/appendix_cgba_d8only_vgg19', 'Results/appendix_cgba_d8only_vit_b_32']
**FLAG lines: 53** (0 is required before a GPU run; every FLAG needs a written explanation)


## 1. Environment (one GPU name, one torch, one driver, one host, identical backend flags, ADV_STORE_BUDGETS unset)

| root | env file | GPU | driver | host | torch | backend flags | ADV_STORE_BUDGETS | split sha256 (first 12) |
|---|---|---|---|---|---|---|---|---|
| b5_e10_ARS_OPT_densenet161_places365_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 0f454a78c615 |
| b5_e10_ARS_OPT_densenet161_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_ARS_OPT_densenet161_places365_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 0f454a78c615 |
| b5_e10_ARS_OPT_densenet161_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_ARS_OPT_resnet50_places365_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 0f454a78c615 |
| b5_e10_ARS_OPT_resnet50_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_ARS_OPT_resnet50_places365_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 0f454a78c615 |
| b5_e10_ARS_OPT_resnet50_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_CGBA_H_densenet161_places365_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, 7596d79806ff |
| b5_e10_CGBA_H_densenet161_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_CGBA_H_resnet50_places365_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, 7596d79806ff |
| b5_e10_CGBA_H_resnet50_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_CGBA_densenet161_places365_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, 7596d79806ff |
| b5_e10_CGBA_densenet161_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_CGBA_resnet50_places365_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, 7596d79806ff |
| b5_e10_CGBA_resnet50_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 892528497715, b311cb0e8bea |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_LGC_RC_EGC_densenet161_places365_T | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 51935b752227, b311cb0e8bea |
| b5_e10_LGC_RC_EGC_densenet161_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 351bc50f594f, 46529c86902b |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_LGC_RC_EGC_resnet50_places365_T | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 70bd3d5c5989, 46529c86902b |
| b5_e10_LGC_RC_EGC_resnet50_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_TtBA_densenet161_places365_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 892528497715, b311cb0e8bea, c05876acee65 |
| b5_e10_TtBA_densenet161_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_TtBA_densenet161_places365_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 51935b752227, b311cb0e8bea, c05876acee65 |
| b5_e10_TtBA_densenet161_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_TtBA_resnet50_places365_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 351bc50f594f, 46529c86902b, c05876acee65 |
| b5_e10_TtBA_resnet50_places365_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e10_TtBA_resnet50_places365_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 70bd3d5c5989, 46529c86902b, c05876acee65 |
| b5_e10_TtBA_resnet50_places365_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 0f454a78c615 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_ARS_OPT_Salman2020Do_R50_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 0f454a78c615 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_CGBA_H_Salman2020Do_R50_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, 7596d79806ff |
| b5_e11_CGBA_H_Salman2020Do_R50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_CGBA_Salman2020Do_R50_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 7596d79806ff |
| b5_e11_CGBA_Salman2020Do_R50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 1b847d4c3acc, 1835a5bbe5e3 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T | env_frame.txt | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 835f1933d9e5, 2985ebc400ea, 1835a5bbe5e3 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_TtBA_Salman2020Do_R50_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, c05876acee65 |
| b5_e11_TtBA_Salman2020Do_R50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_e11_TtBA_Salman2020Do_R50_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 2985ebc400ea, 1835a5bbe5e3, c05876acee65 |
| b5_e11_TtBA_Salman2020Do_R50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_resnet50_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 0f454a78c615 |
| b5_main_ARS_OPT_resnet50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_resnet50_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 0f454a78c615 |
| b5_main_ARS_OPT_resnet50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_vgg19_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 0f454a78c615 |
| b5_main_ARS_OPT_vgg19_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_vgg19_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 0f454a78c615 |
| b5_main_ARS_OPT_vgg19_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_vit_b_32_NT | non_targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 0f454a78c615 |
| b5_main_ARS_OPT_vit_b_32_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_ARS_OPT_vit_b_32_T | targeted/env_ARS_OPT.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 0f454a78c615 |
| b5_main_ARS_OPT_vit_b_32_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_H_resnet50_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, 7596d79806ff |
| b5_main_CGBA_H_resnet50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_H_vgg19_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, 7596d79806ff |
| b5_main_CGBA_H_vgg19_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_H_vit_b_32_T | targeted/env_CGBA_H.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, 7596d79806ff |
| b5_main_CGBA_H_vit_b_32_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_resnet50_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 7596d79806ff |
| b5_main_CGBA_resnet50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_vgg19_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 7596d79806ff |
| b5_main_CGBA_vgg19_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_CGBA_vit_b_32_NT | non_targeted/env_CGBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 7596d79806ff |
| b5_main_CGBA_vit_b_32_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_resnet50_NT | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_resnet50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_resnet50_T | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_resnet50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_vgg19_NT | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_vgg19_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_vgg19_T | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_vgg19_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_vit_b_32_NT | non_targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_vit_b_32_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_LGC_RC_EGC_vit_b_32_T | targeted/config_single.json | NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True | ? | ? | 2.7.1+cu128 | cudnn_tf32=False; deterministic=True | MISSING |  |
| b5_main_LGC_RC_EGC_vit_b_32_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_resnet50_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, c05876acee65 |
| b5_main_TtBA_resnet50_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_resnet50_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 550ec796e485, 11ad3fa62ca7, c05876acee65 |
| b5_main_TtBA_resnet50_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_vgg19_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, c05876acee65 |
| b5_main_TtBA_vgg19_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_vgg19_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | cf0a8efb783d, dcbb9e9dad56, c05876acee65 |
| b5_main_TtBA_vgg19_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_vit_b_32_NT | non_targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, c05876acee65 |
| b5_main_TtBA_vit_b_32_NT | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| b5_main_TtBA_vit_b_32_T | targeted/env_TtBA.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | c5eee382244c, d86f8d99b3c3, c05876acee65 |
| b5_main_TtBA_vit_b_32_T | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| appendix_cgba_d8only_Salman2020Do_R50 | non_targeted/env_CGBA_D8only.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1b847d4c3acc, 1835a5bbe5e3, 7596d79806ff |
| appendix_cgba_d8only_Salman2020Do_R50 | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| appendix_cgba_d8only_resnet50 | non_targeted/env_CGBA_D8only.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | e08a069e932e, 11ad3fa62ca7, 7596d79806ff |
| appendix_cgba_d8only_resnet50 | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| appendix_cgba_d8only_vgg19 | non_targeted/env_CGBA_D8only.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | 1e811637c264, dcbb9e9dad56, 7596d79806ff |
| appendix_cgba_d8only_vgg19 | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |
| appendix_cgba_d8only_vit_b_32 | non_targeted/env_CGBA_D8only.json | NVIDIA GeForce RTX 5090 | ? | ? | 2.7.1+cu128 | benchmark=False; cudnn_tf32=False; deterministic=True; matmul_tf32=False | unset | a5f152ec215a, d86f8d99b3c3, 7596d79806ff |
| appendix_cgba_d8only_vit_b_32 | **FLAG** | | driver and hostname not recorded (--require_host) | | | | | |

- GPU names: {'NVIDIA GeForce RTX 5090': 46, 'NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True': 6}
  - **FLAG: more than one value across env files** (drivers/hosts: expected only if roots from earlier hosts are included on purpose; B5 roots must show one value)
- torch versions: {'2.7.1+cu128': 52}
- drivers: {}
- hosts: {}
- note: no driver or hostname recorded in any env file; host drift (L03) cannot be checked from these folders, use `nvidia-smi` and the B5 runner host log
- backend benchmark: {'False': 46}
- backend cudnn_tf32: {'False': 52}
- backend deterministic: {'True': 52}
- backend matmul_tf32: {'False': 46}
- env files read: 52

## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 1200; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0

## 3. Query budget (baselines must spend exactly the budget; nothing may exceed it)

| root/mode | method | images | status counts | queries == budget | < budget | > budget | non-finite queries |
|---|---|---|---|---|---|---|---|
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | 200 | {'ok': 200} | 0 | 200 | 0 | n/a |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | 200 | {'budget': 200} | 200 | 0 | 0 | 0 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | 30 | {'budget': 30} | 30 | 0 | 0 | 0 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | 30 | {'budget': 30} | 30 | 0 | 0 | 0 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | 30 | {'budget': 30} | 30 | 0 | 0 | 0 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | 30 | {'budget': 30} | 30 | 0 | 0 | 0 |

Read '< budget' rows: fine for our engine (it stops below the budget); for a baseline it means an early stop or a crash.

## 4. L2 recomputation from stored images (relative error must be < 1e-3)

| root/mode | method | arch | image | budget | pix_l2 (csv) | ||delta|| (npz) | l2 (npz) | rel err |
|---|---|---|---|---|---|---|---|---|
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00008590 | 5000 | 1.74720 | 1.74720 | 1.74720 | 6.3e-07 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00027966 | 10000 | 9.54380 | 9.54382 | 9.54380 | 2.8e-06 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00009555 | 2500 | 5.13661 | 5.13660 | 5.13661 | 1.1e-06 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00023129 | 500 | 62.11625 | 62.11636 | 62.11625 | 1.7e-06 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00017891 | 2500 | 15.69440 | 15.69439 | 15.69440 | 6.5e-07 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00004416 | 5000 | 3.36642 | 3.36642 | 3.36642 | 2.3e-07 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00009555 | 500 | 32.20010 | 32.20015 | 32.20009 | 1.5e-06 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00010800 | 10000 | 5.90898 | 5.90898 | 5.90898 | 1.1e-06 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00021959 | 2500 | 4.88619 | 4.88619 | 4.88619 | 4.4e-07 |
| b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted | ARS_OPT | densenet161_places365 | Places365_val_00033455 | 2000 | 19.88494 | 19.88492 | 19.88494 | 8.7e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00003940 | 5000 | 49.23373 | 49.23372 | 49.23373 | 1.8e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00023553 | 1000 | 50.53022 | 50.53035 | 50.53022 | 2.6e-06 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00028576 | 5000 | 85.67103 | 85.67113 | 85.67103 | 1.2e-06 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00025048 | 5000 | 41.00586 | 41.00588 | 41.00586 | 5.1e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00009555 | 2500 | 56.25023 | 56.25022 | 56.25023 | 2.4e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00004531 | 10000 | 28.41075 | 28.41076 | 28.41075 | 4.9e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00002067 | 200 | 101.45917 | 101.45828 | 101.45919 | 8.8e-06 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00023163 | 100 | 151.02933 | 151.02930 | 151.02933 | 1.8e-07 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00030836 | 200 | 104.94835 | 104.94835 | 104.94836 | 7.3e-08 |
| b5_e10_ARS_OPT_densenet161_places365_T/targeted | ARS_OPT | densenet161_places365 | Places365_val_00024738 | 1000 | 75.54380 | 75.54376 | 75.54380 | 5.8e-07 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00018510 | 100 | 5.18565 | 5.18564 | 5.18565 | 1.5e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00034898 | 200 | 48.31274 | 48.31279 | 48.31274 | 1.0e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00021969 | 100 | 51.18034 | 51.18039 | 51.18034 | 9.6e-07 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00011698 | 100 | 44.25642 | 44.25637 | 44.25641 | 1.1e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00018407 | 2000 | 10.99057 | 10.99059 | 10.99057 | 1.5e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00002450 | 5000 | 4.42897 | 4.42897 | 4.42897 | 1.1e-07 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00008732 | 100 | 15.40225 | 15.40233 | 15.40229 | 4.7e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00016707 | 2500 | 15.26340 | 15.26348 | 15.26341 | 5.0e-06 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00006719 | 100 | 21.82424 | 21.82425 | 21.82424 | 5.5e-07 |
| b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted | ARS_OPT | resnet50_places365 | Places365_val_00015637 | 5000 | 6.23162 | 6.23161 | 6.23162 | 6.3e-07 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00032314 | 5000 | 36.32687 | 36.32691 | 36.32687 | 1.1e-06 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00031890 | 5000 | 69.54736 | 69.54735 | 69.54736 | 1.3e-07 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00021969 | 2000 | 105.02628 | 105.02626 | 105.02628 | 2.6e-07 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00032340 | 5000 | 107.39147 | 107.39147 | 107.39148 | 7.1e-08 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00026932 | 5000 | 6.98187 | 6.98186 | 6.98187 | 1.2e-06 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00030327 | 5000 | 23.03750 | 23.03754 | 23.03750 | 1.3e-06 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00034898 | 100 | 127.15216 | 127.15205 | 127.15216 | 8.7e-07 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00032287 | 500 | 152.80579 | 152.80610 | 152.80580 | 2.0e-06 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00019178 | 100 | 136.93985 | 136.93967 | 136.93985 | 1.3e-06 |
| b5_e10_ARS_OPT_resnet50_places365_T/targeted | ARS_OPT | resnet50_places365 | Places365_val_00001986 | 2500 | 47.22057 | 47.22059 | 47.22057 | 4.0e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00027705 | 5000 | 53.75996 | 53.75992 | 53.75996 | 7.4e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00014492 | 100 | 89.88557 | 89.88549 | 89.88557 | 9.1e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00010243 | 500 | 18.93812 | 18.93813 | 18.93812 | 3.1e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00010904 | 100 | 67.52319 | 67.52325 | 67.52319 | 8.2e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00009555 | 100 | 82.27859 | 82.27864 | 82.27859 | 6.1e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00017435 | 10000 | 6.97486 | 6.97487 | 6.97486 | 1.3e-06 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00023916 | 200 | 81.25220 | 81.25217 | 81.25221 | 4.4e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00004531 | 500 | 111.67892 | 111.67875 | 111.67892 | 1.5e-06 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00010033 | 5000 | 29.42031 | 29.42032 | 29.42031 | 2.7e-07 |
| b5_e10_CGBA_H_densenet161_places365_T/targeted | CGBA_H | densenet161_places365 | Places365_val_00032686 | 1000 | 64.73062 | 64.73069 | 64.73062 | 1.1e-06 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00030327 | 500 | 23.42627 | 23.42622 | 23.42627 | 2.2e-06 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00013527 | 200 | 127.76550 | 127.76544 | 127.76549 | 4.2e-07 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00032287 | 10000 | 15.24286 | 15.24284 | 15.24286 | 1.6e-06 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00023888 | 2500 | 40.93206 | 40.93207 | 40.93206 | 3.6e-07 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00012969 | 2000 | 32.17135 | 32.17134 | 32.17135 | 3.0e-07 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00002067 | 1000 | 31.79364 | 31.79364 | 31.79364 | 6.0e-08 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00028200 | 100 | 149.84077 | 149.84078 | 149.84077 | 6.9e-08 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00004313 | 10000 | 6.76884 | 6.76883 | 6.76884 | 1.3e-06 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00021721 | 500 | 47.72070 | 47.72063 | 47.72070 | 1.6e-06 |
| b5_e10_CGBA_H_resnet50_places365_T/targeted | CGBA_H | resnet50_places365 | Places365_val_00019178 | 500 | 107.87017 | 107.87010 | 107.87016 | 6.6e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00034458 | 100 | 41.15958 | 41.15966 | 41.15958 | 1.9e-06 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00010033 | 100 | 55.93513 | 55.93520 | 55.93513 | 1.2e-06 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00031655 | 1000 | 6.56608 | 6.56609 | 6.56608 | 6.2e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00029503 | 100 | 42.45816 | 42.45810 | 42.45815 | 1.3e-06 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00016648 | 1000 | 0.67977 | 0.67977 | 0.67977 | 2.0e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00020099 | 1000 | 13.15757 | 13.15757 | 13.15757 | 2.3e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00018407 | 10000 | 0.16164 | 0.16164 | 0.16164 | 2.1e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00010455 | 1000 | 2.37572 | 2.37572 | 2.37572 | 9.8e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00020636 | 2000 | 8.47235 | 8.47235 | 8.47235 | 8.0e-07 |
| b5_e10_CGBA_densenet161_places365_NT/non_targeted | CGBA | densenet161_places365 | Places365_val_00025524 | 100 | 29.73000 | 29.73005 | 29.73000 | 1.7e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00013466 | 100 | 7.61133 | 7.61134 | 7.61132 | 2.1e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00009555 | 200 | 8.27096 | 8.27096 | 8.27096 | 7.2e-08 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00013068 | 500 | 6.16836 | 6.16837 | 6.16836 | 1.6e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00008167 | 2500 | 9.30260 | 9.30259 | 9.30260 | 1.1e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00027798 | 2500 | 0.59324 | 0.59324 | 0.59324 | 1.2e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00003356 | 500 | 12.06726 | 12.06728 | 12.06726 | 1.6e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00011698 | 10000 | 5.87687 | 5.87686 | 5.87687 | 8.0e-07 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00029616 | 200 | 35.96418 | 35.96423 | 35.96418 | 1.3e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00021969 | 10000 | 2.79462 | 2.79461 | 2.79462 | 1.4e-06 |
| b5_e10_CGBA_resnet50_places365_NT/non_targeted | CGBA | resnet50_places365 | Places365_val_00036169 | 5000 | 0.74382 | 0.74382 | 0.74382 | 5.0e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00012799 | 2500 | 0.25247 | 0.25247 | 0.25247 | 7.4e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00033949 | 500 | 16.49366 | 16.49369 | 16.49365 | 2.1e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00001404 | 1000 | 6.97101 | 6.97101 | 6.97101 | 2.7e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00032534 | 10000 | 1.37817 | 1.37818 | 1.37817 | 3.6e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00021721 | 5000 | 2.97254 | 2.97254 | 2.97253 | 2.3e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00033949 | 1000 | 10.20892 | 10.20891 | 10.20892 | 4.3e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00034458 | 1000 | 2.64329 | 2.64328 | 2.64329 | 3.7e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00028076 | 5000 | 1.92860 | 1.92860 | 1.92860 | 1.4e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00027798 | 1000 | 0.38363 | 0.38363 | 0.38363 | 2.8e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00012969 | 100 | 43.23592 | 43.23607 | 43.23592 | 3.5e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00033086 | 200 | 99.01426 | 99.01421 | 99.01426 | 4.8e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00013527 | 10000 | 11.37572 | 11.37576 | 11.37572 | 2.8e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004531 | 2500 | 17.44888 | 17.44893 | 17.44888 | 2.7e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00027463 | 100 | 102.55681 | 102.55680 | 102.55680 | 1.3e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00020186 | 1000 | 100.66653 | 100.66649 | 100.66653 | 3.6e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00020240 | 10000 | 10.55172 | 10.55171 | 10.55172 | 1.4e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00030327 | 2500 | 22.64108 | 22.64106 | 22.64108 | 5.6e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00032534 | 10000 | 11.21850 | 11.21852 | 11.21850 | 1.4e-06 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00004313 | 1000 | 37.32267 | 37.32267 | 37.32267 | 1.7e-07 |
| b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted | LGC_RC_EGC | densenet161_places365 | Places365_val_00001856 | 100 | 69.98560 | 69.98552 | 69.98561 | 1.1e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023553 | 10000 | 0.56741 | 0.56741 | 0.56741 | 8.5e-08 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00017764 | 2500 | 0.02503 | 0.02503 | 0.02503 | 1.1e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00009088 | 2500 | 0.06599 | 0.06599 | 0.06599 | 8.5e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00010726 | 200 | 27.70659 | 27.70657 | 27.70659 | 6.2e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00015637 | 500 | 5.77211 | 5.77211 | 5.77211 | 1.1e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023553 | 200 | 13.09911 | 13.09910 | 13.09911 | 8.3e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023163 | 200 | 0.33432 | 0.33432 | 0.33432 | 8.2e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00011900 | 200 | 15.85644 | 15.85643 | 15.85644 | 7.5e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023556 | 500 | 0.14813 | 0.14813 | 0.14813 | 8.5e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00018663 | 100 | 21.00942 | 21.00956 | 21.00942 | 7.0e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00021050 | 2000 | 11.48008 | 11.48005 | 11.48008 | 2.9e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00020636 | 100 | 73.78805 | 73.78805 | 73.78805 | 3.0e-08 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00032314 | 200 | 83.05448 | 83.05457 | 83.05447 | 1.1e-06 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00010726 | 500 | 91.21043 | 91.21039 | 91.21042 | 3.5e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00023915 | 5000 | 65.61067 | 65.61064 | 65.61066 | 4.6e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00011698 | 10000 | 26.35615 | 26.35617 | 26.35615 | 5.8e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00015024 | 1000 | 51.28096 | 51.28095 | 51.28097 | 3.5e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00003743 | 2500 | 29.46078 | 29.46075 | 29.46077 | 9.9e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00014330 | 200 | 68.35490 | 68.35492 | 68.35490 | 1.7e-07 |
| b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted | LGC_RC_EGC | resnet50_places365 | Places365_val_00021738 | 10000 | 21.26403 | 21.26404 | 21.26402 | 5.3e-07 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00003837 | 10000 | 1.31412 | 1.31412 | 1.31412 | 9.1e-08 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00036169 | 2500 | 0.18788 | 0.18788 | 0.18788 | 3.5e-07 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00006284 | 2500 | 18.28390 | 18.28388 | 18.28390 | 9.2e-07 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00013466 | 500 | 5.69760 | 5.69760 | 5.69760 | 9.2e-07 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00036169 | 10000 | 0.10118 | 0.10118 | 0.10118 | 1.3e-06 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00023578 | 5000 | 1.03542 | 1.03542 | 1.03542 | 1.9e-06 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00022993 | 200 | 4.26044 | 4.26043 | 4.26044 | 1.2e-06 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00033883 | 200 | 36.80049 | 36.80051 | 36.80048 | 5.4e-07 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00032534 | 2500 | 2.51590 | 2.51590 | 2.51590 | 1.5e-06 |
| b5_e10_TtBA_densenet161_places365_NT/non_targeted | TtBA | densenet161_places365 | Places365_val_00010726 | 1000 | 6.31539 | 6.31540 | 6.31539 | 1.2e-06 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00013122 | 500 | 50.44777 | 50.44776 | 50.44777 | 2.9e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00028576 | 1000 | 78.08476 | 78.08474 | 78.08476 | 2.4e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00023129 | 1000 | 43.19179 | 43.19175 | 43.19179 | 1.0e-06 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00021678 | 500 | 42.76287 | 42.76285 | 42.76287 | 6.6e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00031198 | 5000 | 2.63685 | 2.63686 | 2.63685 | 1.1e-06 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00030225 | 1000 | 46.81286 | 46.81292 | 46.81286 | 1.3e-06 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00033997 | 5000 | 39.87791 | 39.87792 | 39.87791 | 1.0e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00003940 | 500 | 76.71576 | 76.71571 | 76.71576 | 6.9e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00031747 | 2000 | 28.07642 | 28.07643 | 28.07641 | 4.2e-07 |
| b5_e10_TtBA_densenet161_places365_T/targeted | TtBA | densenet161_places365 | Places365_val_00013246 | 5000 | 77.85907 | 77.85899 | 77.85907 | 1.0e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00021721 | 1000 | 9.58088 | 9.58089 | 9.58088 | 6.7e-07 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00014850 | 10000 | 6.17335 | 6.17336 | 6.17335 | 1.4e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00034321 | 2500 | 2.51504 | 2.51503 | 2.51504 | 2.7e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00032686 | 5000 | 2.08099 | 2.08100 | 2.08099 | 1.2e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00033029 | 5000 | 4.38784 | 4.38784 | 4.38784 | 2.3e-07 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00004416 | 5000 | 2.18299 | 2.18299 | 2.18299 | 1.7e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00032287 | 2000 | 1.70836 | 1.70835 | 1.70836 | 1.2e-06 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00032076 | 2000 | 11.78605 | 11.78604 | 11.78605 | 3.5e-07 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00035791 | 10000 | 5.25384 | 5.25384 | 5.25384 | 3.3e-07 |
| b5_e10_TtBA_resnet50_places365_NT/non_targeted | TtBA | resnet50_places365 | Places365_val_00002067 | 5000 | 1.81956 | 1.81956 | 1.81956 | 1.3e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00005041 | 2000 | 78.06341 | 78.06348 | 78.06342 | 9.2e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00000905 | 100 | 69.63622 | 69.63623 | 69.63622 | 1.7e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00006719 | 2500 | 33.34981 | 33.34980 | 33.34981 | 3.7e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00033768 | 2000 | 35.03148 | 35.03148 | 35.03148 | 1.1e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00018407 | 5000 | 38.04987 | 38.04988 | 38.04987 | 3.4e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00026173 | 10000 | 11.31295 | 11.31294 | 11.31295 | 6.9e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00030836 | 10000 | 14.32350 | 14.32351 | 14.32350 | 6.2e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00008590 | 2000 | 11.69332 | 11.69332 | 11.69332 | 3.0e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00009555 | 500 | 75.29131 | 75.29128 | 75.29131 | 4.2e-07 |
| b5_e10_TtBA_resnet50_places365_T/targeted | TtBA | resnet50_places365 | Places365_val_00031890 | 1000 | 71.10834 | 71.10841 | 71.10834 | 1.0e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00006296 | 10000 | 1.55617 | 1.55618 | 1.55617 | 9.5e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00009286 | 1000 | 57.54803 | 57.54815 | 57.54803 | 2.0e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00010593 | 2000 | 32.58898 | 32.58900 | 32.58899 | 4.9e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00011895 | 500 | 51.82937 | 51.82944 | 51.82936 | 1.3e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00022673 | 10000 | 30.50563 | 30.50573 | 30.50563 | 3.2e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00002400 | 2500 | 25.96141 | 25.96160 | 25.96140 | 7.2e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00030649 | 2500 | 2.23223 | 2.23223 | 2.23223 | 1.7e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00036578 | 200 | 41.62264 | 41.62266 | 41.62263 | 4.9e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00000836 | 5000 | 20.43840 | 20.43848 | 20.43839 | 4.0e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00030649 | 10000 | 0.97086 | 0.97086 | 0.97086 | 5.0e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00033607 | 100 | 115.82991 | 115.82972 | 115.82992 | 1.6e-06 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00009286 | 500 | 86.60364 | 86.60367 | 86.60365 | 3.7e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00046102 | 200 | 67.44016 | 67.44012 | 67.44017 | 6.6e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00045410 | 10000 | 19.73136 | 19.73137 | 19.73135 | 5.5e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00008676 | 1000 | 66.39607 | 66.39612 | 66.39606 | 7.1e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00001034 | 2500 | 80.10003 | 80.10000 | 80.10003 | 3.0e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00014642 | 500 | 132.34578 | 132.34590 | 132.34578 | 9.1e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00003257 | 5000 | 83.91887 | 83.91892 | 83.91887 | 6.2e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00038428 | 500 | 65.15401 | 65.15404 | 65.15401 | 3.5e-07 |
| b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted | ARS_OPT | Salman2020Do_R50 | ILSVRC2012_val_00021369 | 1000 | 42.98009 | 42.98004 | 42.98009 | 1.2e-06 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00043392 | 100 | 118.27248 | 118.27256 | 118.27247 | 7.0e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00037863 | 100 | 147.58176 | 147.58162 | 147.58176 | 9.4e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00020177 | 5000 | 47.24971 | 47.24970 | 47.24971 | 1.5e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00005654 | 2500 | 53.71473 | 53.71472 | 53.71473 | 2.4e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00034588 | 10000 | 22.82962 | 22.82956 | 22.82962 | 2.5e-06 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00031524 | 100 | 76.22331 | 76.22332 | 76.22331 | 1.3e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00038928 | 5000 | 59.30112 | 59.30114 | 59.30112 | 2.7e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00043063 | 200 | 58.23481 | 58.23482 | 58.23481 | 1.9e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00028245 | 500 | 150.54221 | 150.54211 | 150.54219 | 6.6e-07 |
| b5_e11_CGBA_H_Salman2020Do_R50_T/targeted | CGBA_H | Salman2020Do_R50 | ILSVRC2012_val_00004298 | 100 | 154.67262 | 154.67258 | 154.67261 | 2.8e-07 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00001034 | 200 | 22.39770 | 22.39779 | 22.39770 | 4.2e-06 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00022353 | 100 | 43.71482 | 43.71480 | 43.71482 | 4.6e-07 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00033607 | 2000 | 4.18990 | 4.18990 | 4.18990 | 4.7e-07 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00035080 | 5000 | 18.36208 | 18.36207 | 18.36208 | 6.6e-07 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00007047 | 100 | 20.03999 | 20.04006 | 20.03998 | 3.8e-06 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00017418 | 10000 | 29.31430 | 29.31420 | 29.31430 | 3.3e-06 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046284 | 2500 | 15.20952 | 15.20955 | 15.20952 | 1.8e-06 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00022792 | 2500 | 10.42448 | 10.42448 | 10.42447 | 7.1e-07 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00017418 | 500 | 50.36024 | 50.36036 | 50.36024 | 2.5e-06 |
| b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted | CGBA | Salman2020Do_R50 | ILSVRC2012_val_00046137 | 500 | 22.83436 | 22.83437 | 22.83436 | 9.8e-08 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046137 | 100 | 31.86180 | 31.86184 | 31.86179 | 1.4e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00019109 | 1000 | 0.80704 | 0.80703 | 0.80704 | 1.0e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00045746 | 100 | 43.37428 | 43.37426 | 43.37428 | 4.7e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00038928 | 1000 | 33.51831 | 33.51837 | 33.51831 | 1.7e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00033049 | 5000 | 0.65252 | 0.65252 | 0.65252 | 3.0e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00032482 | 2500 | 19.18725 | 19.18733 | 19.18725 | 3.7e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00049021 | 2500 | 13.42605 | 13.42604 | 13.42604 | 2.3e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00025020 | 10000 | 5.39611 | 5.39609 | 5.39611 | 4.2e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00035080 | 10000 | 14.46839 | 14.46841 | 14.46839 | 1.3e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00022792 | 1000 | 8.88738 | 8.88736 | 8.88738 | 2.2e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046701 | 2500 | 61.09743 | 61.09737 | 61.09742 | 1.0e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00044980 | 2500 | 50.53307 | 50.53307 | 50.53307 | 7.5e-08 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00046137 | 2500 | 20.99202 | 20.99201 | 20.99202 | 1.9e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00030085 | 1000 | 53.13019 | 53.13015 | 53.13019 | 7.6e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00048242 | 10000 | 19.67011 | 19.67011 | 19.67011 | 1.3e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00014642 | 100 | 134.62737 | 134.62753 | 134.62737 | 1.2e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00020759 | 5000 | 15.35540 | 15.35543 | 15.35540 | 1.4e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00032482 | 2500 | 58.34868 | 58.34869 | 58.34868 | 2.4e-07 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00020177 | 2500 | 49.11105 | 49.11117 | 49.11106 | 2.4e-06 |
| b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted | LGC_RC_EGC | Salman2020Do_R50 | ILSVRC2012_val_00020608 | 2000 | 44.46117 | 44.46124 | 44.46117 | 1.8e-06 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00044546 | 5000 | 0.76976 | 0.76976 | 0.76976 | 7.7e-08 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00040330 | 1000 | 6.07186 | 6.07187 | 6.07186 | 3.2e-07 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00003815 | 500 | 16.46031 | 16.46030 | 16.46031 | 6.9e-07 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00008234 | 2000 | 9.71797 | 9.71796 | 9.71797 | 6.4e-07 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00000550 | 100 | 36.65862 | 36.65879 | 36.65862 | 4.6e-06 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00032447 | 500 | 15.17326 | 15.17326 | 15.17326 | 3.0e-07 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00030085 | 1000 | 20.94218 | 20.94225 | 20.94217 | 3.2e-06 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00045305 | 5000 | 6.25908 | 6.25909 | 6.25908 | 1.7e-06 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00025589 | 5000 | 1.69714 | 1.69714 | 1.69714 | 4.1e-07 |
| b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00008234 | 1000 | 14.74762 | 14.74765 | 14.74762 | 1.8e-06 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00003257 | 2500 | 85.19270 | 85.19268 | 85.19270 | 2.1e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00028245 | 100 | 151.43475 | 151.43481 | 151.43475 | 3.9e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00017841 | 1000 | 95.31672 | 95.31666 | 95.31672 | 5.8e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00029074 | 2000 | 81.47371 | 81.47359 | 81.47371 | 1.5e-06 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00024477 | 500 | 78.29713 | 78.29704 | 78.29713 | 1.2e-06 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00044800 | 10000 | 26.01289 | 26.01284 | 26.01288 | 1.7e-06 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 100 | 95.53925 | 95.53930 | 95.53926 | 5.1e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00026903 | 2000 | 24.27906 | 24.27908 | 24.27906 | 9.6e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00019109 | 500 | 71.82130 | 71.82134 | 71.82129 | 5.9e-07 |
| b5_e11_TtBA_Salman2020Do_R50_T/targeted | TtBA | Salman2020Do_R50 | ILSVRC2012_val_00045305 | 500 | 95.11086 | 95.11092 | 95.11084 | 6.9e-07 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00011140 | 500 | 15.73677 | 15.73679 | 15.73677 | 1.1e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00011335 | 2000 | 5.16108 | 5.16109 | 5.16108 | 2.3e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00026746 | 5000 | 0.21071 | 0.21071 | 0.21071 | 5.2e-07 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00006837 | 1000 | 7.73148 | 7.73150 | 7.73148 | 2.7e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00002910 | 10000 | 2.78035 | 2.78035 | 2.78035 | 1.3e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00031358 | 1000 | 28.13489 | 28.13489 | 28.13489 | 1.9e-07 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00037041 | 1000 | 41.37395 | 41.37401 | 41.37395 | 1.2e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00042948 | 1000 | 90.12111 | 90.12117 | 90.12109 | 6.5e-07 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00044766 | 200 | 33.54066 | 33.54073 | 33.54066 | 1.9e-06 |
| b5_main_ARS_OPT_resnet50_NT/non_targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00004632 | 2000 | 2.47135 | 2.47135 | 2.47135 | 1.9e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00031358 | 100 | 70.14653 | 70.14583 | 70.14652 | 1.0e-05 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00048345 | 5000 | 16.67486 | 16.67486 | 16.67486 | 1.1e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00036175 | 1000 | 81.88887 | 81.88892 | 81.88886 | 5.9e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00044766 | 1000 | 58.40955 | 58.40956 | 58.40955 | 1.1e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00037863 | 2500 | 84.42195 | 84.42208 | 84.42197 | 1.5e-06 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00020497 | 2500 | 93.16006 | 93.16011 | 93.16006 | 6.0e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00024661 | 2500 | 55.23183 | 55.23184 | 55.23183 | 9.8e-08 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00014825 | 100 | 74.99771 | 74.99856 | 74.99773 | 1.1e-05 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00033659 | 200 | 94.40774 | 94.40776 | 94.40774 | 2.3e-07 |
| b5_main_ARS_OPT_resnet50_T/targeted | ARS_OPT | resnet50 | ILSVRC2012_val_00003731 | 200 | 86.69326 | 86.69321 | 86.69327 | 5.4e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00012244 | 2000 | 1.66425 | 1.66425 | 1.66425 | 6.3e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00022480 | 5000 | 7.99357 | 7.99357 | 7.99357 | 2.7e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00045770 | 1000 | 17.69576 | 17.69579 | 17.69576 | 1.8e-06 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00019526 | 5000 | 12.69483 | 12.69486 | 12.69483 | 2.0e-06 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00015726 | 200 | 6.41654 | 6.41653 | 6.41653 | 7.4e-08 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00035476 | 500 | 12.97170 | 12.97171 | 12.97170 | 8.5e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00049827 | 1000 | 24.81663 | 24.81672 | 24.81662 | 3.7e-06 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00005000 | 2000 | 1.21965 | 1.21965 | 1.21965 | 4.8e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00023734 | 200 | 83.59489 | 83.59488 | 83.59496 | 7.3e-07 |
| b5_main_ARS_OPT_vgg19_NT/non_targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00019317 | 100 | 45.02207 | 45.02216 | 45.02206 | 2.0e-06 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00040385 | 2500 | 85.05378 | 85.05375 | 85.05378 | 3.3e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00032950 | 1000 | 115.08253 | 115.08259 | 115.08253 | 5.1e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00047355 | 500 | 99.98392 | 99.98390 | 99.98392 | 2.4e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039615 | 100 | 103.19953 | 103.19951 | 103.19953 | 1.8e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00031025 | 2500 | 45.89954 | 45.89957 | 45.89955 | 7.3e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00017507 | 2500 | 43.29887 | 43.29887 | 43.29888 | 1.3e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00045770 | 10000 | 27.54507 | 27.54506 | 27.54507 | 2.6e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00006108 | 500 | 58.36877 | 58.36873 | 58.36877 | 7.4e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00039161 | 10000 | 9.63699 | 9.63699 | 9.63699 | 3.7e-07 |
| b5_main_ARS_OPT_vgg19_T/targeted | ARS_OPT | vgg19 | ILSVRC2012_val_00002011 | 2500 | 102.17919 | 102.17918 | 102.17921 | 1.5e-07 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00038330 | 5000 | 4.38868 | 4.38868 | 4.38868 | 1.7e-07 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00036855 | 500 | 33.01708 | 33.01713 | 33.01708 | 1.6e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00023166 | 1000 | 23.42904 | 23.42901 | 23.42904 | 1.4e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00000497 | 100 | 110.17626 | 110.17641 | 110.17630 | 1.4e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00020066 | 100 | 4.12509 | 4.12508 | 4.12509 | 2.0e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00016228 | 100 | 27.46794 | 27.46802 | 27.46794 | 3.1e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00026001 | 200 | 74.38898 | 74.38937 | 74.38898 | 5.2e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00003516 | 200 | 88.11095 | 88.11112 | 88.11095 | 1.9e-06 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00007605 | 5000 | 1.73938 | 1.73938 | 1.73938 | 3.2e-07 |
| b5_main_ARS_OPT_vit_b_32_NT/non_targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00014060 | 2000 | 29.32129 | 29.32136 | 29.32129 | 2.5e-06 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00038351 | 2000 | 62.15398 | 62.15388 | 62.15398 | 1.6e-06 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024809 | 200 | 89.68498 | 89.68492 | 89.68497 | 6.6e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00023166 | 2500 | 19.62431 | 19.62432 | 19.62431 | 3.4e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00014060 | 2000 | 37.79236 | 37.79239 | 37.79236 | 7.8e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00030717 | 200 | 74.44781 | 74.44787 | 74.44782 | 7.0e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00002996 | 200 | 71.22679 | 71.22684 | 71.22680 | 6.7e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00045746 | 500 | 75.17525 | 75.17531 | 75.17525 | 7.7e-07 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00036659 | 2000 | 38.98069 | 38.98074 | 38.98069 | 1.2e-06 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00024661 | 100 | 48.87143 | 48.87180 | 48.87138 | 7.5e-06 |
| b5_main_ARS_OPT_vit_b_32_T/targeted | ARS_OPT | vit_b_32 | ILSVRC2012_val_00015835 | 2500 | 34.04860 | 34.04859 | 34.04860 | 3.7e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00004916 | 10000 | 10.18331 | 10.18332 | 10.18331 | 1.2e-06 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00023755 | 500 | 35.49158 | 35.49159 | 35.49158 | 4.7e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00026468 | 10000 | 15.85161 | 15.85161 | 15.85161 | 9.2e-08 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00033162 | 2500 | 46.25173 | 46.25174 | 46.25172 | 3.1e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00009089 | 2000 | 77.62731 | 77.62738 | 77.62732 | 8.3e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00048936 | 100 | 49.56556 | 49.56556 | 49.56556 | 1.0e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00024650 | 1000 | 82.66521 | 82.66513 | 82.66521 | 1.0e-06 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00015816 | 10000 | 9.19479 | 9.19480 | 9.19479 | 1.8e-06 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00029626 | 2000 | 41.26419 | 41.26417 | 41.26419 | 3.0e-07 |
| b5_main_CGBA_H_resnet50_T/targeted | CGBA_H | resnet50 | ILSVRC2012_val_00008547 | 2000 | 74.81953 | 74.81952 | 74.81953 | 1.0e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00002309 | 200 | 29.79748 | 29.79747 | 29.79748 | 4.4e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00037009 | 10000 | 31.24689 | 31.24685 | 31.24689 | 1.1e-06 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00033704 | 200 | 48.50016 | 48.50022 | 48.50016 | 1.2e-06 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00011306 | 2500 | 42.77649 | 42.77653 | 42.77649 | 8.9e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00039281 | 200 | 35.87086 | 35.87087 | 35.87086 | 1.2e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00009655 | 500 | 78.38628 | 78.38622 | 78.38628 | 8.2e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00042330 | 10000 | 24.70548 | 24.70548 | 24.70548 | 3.6e-08 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00010290 | 500 | 77.15150 | 77.15146 | 77.15150 | 5.3e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00014723 | 500 | 58.41530 | 58.41535 | 58.41529 | 8.7e-07 |
| b5_main_CGBA_H_vgg19_T/targeted | CGBA_H | vgg19 | ILSVRC2012_val_00003866 | 200 | 64.06961 | 64.06961 | 64.06961 | 4.1e-08 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00036605 | 100 | 31.62158 | 31.62157 | 31.62159 | 4.1e-07 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00004871 | 1000 | 21.45613 | 21.45613 | 21.45613 | 8.9e-08 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00006336 | 5000 | 7.17128 | 7.17128 | 7.17128 | 6.8e-07 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00000497 | 200 | 85.23572 | 85.23560 | 85.23572 | 1.4e-06 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00032793 | 10000 | 2.74924 | 2.74925 | 2.74924 | 2.1e-06 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00020919 | 200 | 187.56511 | 187.56526 | 187.56511 | 7.8e-07 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00038169 | 10000 | 9.84136 | 9.84137 | 9.84136 | 1.4e-06 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00023111 | 100 | 94.08612 | 94.08605 | 94.08611 | 7.0e-07 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00019218 | 200 | 53.70242 | 53.70235 | 53.70242 | 1.4e-06 |
| b5_main_CGBA_H_vit_b_32_T/targeted | CGBA_H | vit_b_32 | ILSVRC2012_val_00027876 | 10000 | 0.52908 | 0.52908 | 0.52908 | 5.6e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00027097 | 1000 | 14.14448 | 14.14450 | 14.14448 | 1.8e-06 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00039775 | 10000 | 0.29803 | 0.29803 | 0.29803 | 6.2e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00009475 | 5000 | 0.10174 | 0.10174 | 0.10174 | 3.9e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00028257 | 2000 | 5.25000 | 5.25000 | 5.25000 | 1.1e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00033418 | 100 | 90.17674 | 90.17718 | 90.17681 | 4.9e-06 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00033948 | 10000 | 0.30180 | 0.30180 | 0.30180 | 2.8e-06 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00024661 | 100 | 45.57399 | 45.57399 | 45.57399 | 8.5e-08 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00034822 | 5000 | 0.71827 | 0.71827 | 0.71828 | 1.8e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00045721 | 1000 | 15.11748 | 15.11748 | 15.11748 | 1.7e-07 |
| b5_main_CGBA_resnet50_NT/non_targeted | CGBA | resnet50 | ILSVRC2012_val_00033418 | 1000 | 15.86331 | 15.86332 | 15.86331 | 1.2e-06 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00028797 | 1000 | 7.75094 | 7.75093 | 7.75094 | 4.1e-07 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00034371 | 1000 | 4.26392 | 4.26392 | 4.26392 | 1.9e-06 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00009409 | 2000 | 2.04490 | 2.04490 | 2.04490 | 1.2e-07 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00011306 | 2500 | 1.92535 | 1.92535 | 1.92535 | 2.5e-08 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00033331 | 2500 | 0.27292 | 0.27292 | 0.27292 | 7.1e-07 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00012507 | 5000 | 0.38488 | 0.38488 | 0.38488 | 1.5e-06 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00015695 | 200 | 31.86124 | 31.86140 | 31.86123 | 5.0e-06 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00003642 | 5000 | 1.03945 | 1.03945 | 1.03945 | 9.6e-07 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00015695 | 2000 | 1.91286 | 1.91286 | 1.91286 | 1.9e-07 |
| b5_main_CGBA_vgg19_NT/non_targeted | CGBA | vgg19 | ILSVRC2012_val_00045944 | 10000 | 0.79111 | 0.79111 | 0.79111 | 1.5e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00001034 | 500 | 6.98334 | 6.98335 | 6.98334 | 1.7e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00027253 | 100 | 46.96030 | 46.96025 | 46.96030 | 9.8e-07 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00043472 | 5000 | 0.60425 | 0.60425 | 0.60425 | 5.6e-07 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00046636 | 2000 | 2.94918 | 2.94919 | 2.94918 | 4.8e-07 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00035592 | 2500 | 0.17400 | 0.17400 | 0.17400 | 1.3e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00038086 | 2000 | 15.32564 | 15.32562 | 15.32564 | 1.3e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00030717 | 2000 | 8.37272 | 8.37272 | 8.37272 | 7.2e-08 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00011129 | 500 | 3.68951 | 3.68952 | 3.68951 | 1.9e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00019827 | 1000 | 19.00222 | 19.00227 | 19.00222 | 2.9e-06 |
| b5_main_CGBA_vit_b_32_NT/non_targeted | CGBA | vit_b_32 | ILSVRC2012_val_00021056 | 1000 | 3.45038 | 3.45039 | 3.45038 | 6.6e-07 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00036467 | 200 | 10.17189 | 10.17191 | 10.17189 | 1.8e-06 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00017240 | 500 | 2.91425 | 2.91425 | 2.91425 | 8.0e-07 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00044358 | 1000 | 7.67275 | 7.67274 | 7.67275 | 1.7e-06 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00031968 | 1000 | 33.19497 | 33.19499 | 33.19498 | 5.1e-07 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00006837 | 100 | 39.52811 | 39.52805 | 39.52811 | 1.7e-06 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00024661 | 5000 | 3.32671 | 3.32671 | 3.32672 | 9.2e-07 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00023789 | 5000 | 0.29242 | 0.29242 | 0.29242 | 1.3e-06 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00043592 | 5000 | 7.02396 | 7.02394 | 7.02396 | 2.6e-06 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00005793 | 500 | 7.72344 | 7.72345 | 7.72344 | 7.3e-07 |
| b5_main_LGC_RC_EGC_resnet50_NT/non_targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00035834 | 2000 | 3.02496 | 3.02496 | 3.02496 | 1.8e-06 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00039001 | 500 | 26.21426 | 26.21431 | 26.21426 | 2.2e-06 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00028484 | 2000 | 69.98640 | 69.98629 | 69.98639 | 1.5e-06 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00012414 | 2000 | 66.68026 | 66.68022 | 66.68026 | 5.7e-07 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00016978 | 5000 | 14.33699 | 14.33698 | 14.33699 | 7.0e-07 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00008547 | 2000 | 54.89537 | 54.89530 | 54.89536 | 1.2e-06 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00028226 | 10000 | 67.67204 | 67.67208 | 67.67204 | 6.8e-07 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00033659 | 200 | 86.37635 | 86.37634 | 86.37636 | 1.5e-07 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00017651 | 200 | 86.50755 | 86.50758 | 86.50755 | 3.7e-07 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00048868 | 1000 | 91.30579 | 91.30599 | 91.30577 | 2.2e-06 |
| b5_main_LGC_RC_EGC_resnet50_T/targeted | LGC_RC_EGC | resnet50 | ILSVRC2012_val_00009724 | 200 | 42.92183 | 42.92179 | 42.92183 | 9.8e-07 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00019526 | 10000 | 0.81659 | 0.81659 | 0.81659 | 2.0e-06 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00000398 | 1000 | 0.54536 | 0.54536 | 0.54536 | 1.2e-06 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00008307 | 2000 | 0.60820 | 0.60820 | 0.60820 | 9.4e-07 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00003642 | 2000 | 3.08780 | 3.08780 | 3.08780 | 1.9e-06 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00011230 | 1000 | 4.33142 | 4.33141 | 4.33142 | 2.6e-06 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00034845 | 2000 | 15.04965 | 15.04966 | 15.04965 | 7.7e-07 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00037009 | 5000 | 10.66211 | 10.66208 | 10.66211 | 3.0e-06 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00016828 | 2500 | 1.06560 | 1.06560 | 1.06560 | 4.8e-07 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00021552 | 10000 | 0.32711 | 0.32711 | 0.32711 | 8.0e-07 |
| b5_main_LGC_RC_EGC_vgg19_NT/non_targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00033704 | 2500 | 2.62778 | 2.62778 | 2.62778 | 5.0e-08 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00026707 | 5000 | 9.89332 | 9.89335 | 9.89332 | 2.7e-06 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00003866 | 100 | 70.61398 | 70.61403 | 70.61398 | 8.3e-07 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00034371 | 5000 | 22.14439 | 22.14436 | 22.14439 | 1.3e-06 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00023585 | 100 | 56.31542 | 56.31547 | 56.31542 | 9.7e-07 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00005593 | 200 | 127.86388 | 127.86376 | 127.86388 | 9.4e-07 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00031432 | 500 | 99.46506 | 99.46521 | 99.46505 | 1.5e-06 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00001219 | 200 | 44.45876 | 44.45866 | 44.45876 | 2.2e-06 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00048951 | 1000 | 79.88677 | 79.88674 | 79.88677 | 3.6e-07 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00028007 | 10000 | 5.93721 | 5.93723 | 5.93721 | 2.9e-06 |
| b5_main_LGC_RC_EGC_vgg19_T/targeted | LGC_RC_EGC | vgg19 | ILSVRC2012_val_00040031 | 100 | 168.72710 | 168.72726 | 168.72711 | 9.7e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00016336 | 1000 | 5.22613 | 5.22613 | 5.22613 | 9.1e-08 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00042141 | 1000 | 3.04692 | 3.04693 | 3.04692 | 2.8e-06 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00037343 | 5000 | 1.67134 | 1.67135 | 1.67134 | 5.2e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00035592 | 5000 | 0.13396 | 0.13396 | 0.13396 | 4.9e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00028690 | 200 | 33.99994 | 33.99991 | 33.99994 | 8.7e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00044435 | 2500 | 3.22784 | 3.22783 | 3.22784 | 2.0e-06 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00007544 | 2500 | 0.18526 | 0.18527 | 0.18527 | 9.7e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00006623 | 10000 | 1.35436 | 1.35436 | 1.35436 | 6.6e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00046636 | 1000 | 4.55759 | 4.55759 | 4.55759 | 4.1e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00044833 | 10000 | 1.02147 | 1.02147 | 1.02147 | 4.4e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00047207 | 100 | 79.72443 | 79.72450 | 79.72443 | 7.9e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00044905 | 10000 | 4.65301 | 4.65300 | 4.65301 | 1.8e-06 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00023166 | 1000 | 16.83082 | 16.83080 | 16.83082 | 1.1e-06 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00015844 | 200 | 38.12678 | 38.12679 | 38.12678 | 1.2e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00013588 | 2500 | 22.16436 | 22.16435 | 22.16435 | 3.8e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00008986 | 2500 | 12.84310 | 12.84309 | 12.84310 | 6.9e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00035592 | 1000 | 31.07916 | 31.07916 | 31.07916 | 1.7e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00015816 | 500 | 74.87200 | 74.87190 | 74.87200 | 1.4e-06 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00036605 | 10000 | 1.76788 | 1.76788 | 1.76788 | 5.2e-07 |
| b5_main_LGC_RC_EGC_vit_b_32_T/targeted | LGC_RC_EGC | vit_b_32 | ILSVRC2012_val_00036855 | 5000 | 2.17302 | 2.17302 | 2.17302 | 1.6e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00045960 | 2500 | 3.01702 | 3.01702 | 3.01702 | 4.4e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00022641 | 2500 | 8.98391 | 8.98391 | 8.98391 | 4.1e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00040453 | 100 | 8.40453 | 8.40453 | 8.40453 | 4.6e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00047013 | 500 | 14.77984 | 14.77986 | 14.77984 | 1.1e-06 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00028484 | 2500 | 3.07629 | 3.07629 | 3.07629 | 6.9e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00045918 | 100 | 29.56459 | 29.56459 | 29.56459 | 1.4e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00045139 | 5000 | 3.55359 | 3.55359 | 3.55359 | 4.1e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00028861 | 10000 | 0.09397 | 0.09397 | 0.09397 | 8.8e-07 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00016336 | 2500 | 0.25443 | 0.25443 | 0.25443 | 1.1e-06 |
| b5_main_TtBA_resnet50_NT/non_targeted | TtBA | resnet50 | ILSVRC2012_val_00026449 | 500 | 0.70605 | 0.70604 | 0.70605 | 1.0e-06 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00012019 | 500 | 113.29893 | 113.29886 | 113.29893 | 5.6e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00049323 | 2000 | 30.32622 | 30.32622 | 30.32622 | 2.1e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00008603 | 500 | 24.82596 | 24.82596 | 24.82596 | 5.7e-08 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00044921 | 500 | 35.86545 | 35.86546 | 35.86544 | 3.2e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00028257 | 2500 | 46.35539 | 46.35541 | 46.35539 | 3.1e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00028833 | 2000 | 54.73979 | 54.73975 | 54.73979 | 6.1e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00007920 | 100 | 125.33907 | 125.33913 | 125.33907 | 4.3e-07 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00004355 | 2000 | 67.52489 | 67.52497 | 67.52490 | 1.1e-06 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00008603 | 10000 | 7.55587 | 7.55588 | 7.55587 | 1.5e-06 |
| b5_main_TtBA_resnet50_T/targeted | TtBA | resnet50 | ILSVRC2012_val_00032758 | 100 | 59.92545 | 59.92552 | 59.92545 | 1.1e-06 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00048315 | 100 | 10.87026 | 10.87026 | 10.87026 | 7.1e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00003060 | 1000 | 0.34666 | 0.34666 | 0.34666 | 6.9e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00039161 | 100 | 4.70395 | 4.70395 | 4.70395 | 6.7e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00010466 | 200 | 9.93448 | 9.93448 | 9.93448 | 4.2e-08 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00012640 | 5000 | 2.50672 | 2.50671 | 2.50672 | 8.2e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00027221 | 5000 | 3.55033 | 3.55034 | 3.55033 | 1.5e-06 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00016828 | 2500 | 1.86230 | 1.86230 | 1.86230 | 3.4e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00001161 | 500 | 54.79186 | 54.79191 | 54.79185 | 1.0e-06 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00001576 | 200 | 45.00780 | 45.00778 | 45.00780 | 4.8e-07 |
| b5_main_TtBA_vgg19_NT/non_targeted | TtBA | vgg19 | ILSVRC2012_val_00021610 | 5000 | 0.36360 | 0.36360 | 0.36360 | 1.2e-06 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00036604 | 2500 | 47.53680 | 47.53676 | 47.53680 | 8.1e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00004011 | 100 | 115.26787 | 115.26785 | 115.26786 | 1.4e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00035574 | 2500 | 74.86348 | 74.86345 | 74.86348 | 4.6e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00012640 | 2000 | 39.65767 | 39.65772 | 39.65766 | 1.3e-06 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00016389 | 500 | 71.66010 | 71.66014 | 71.66010 | 5.7e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00040385 | 10000 | 17.74897 | 17.74897 | 17.74897 | 1.7e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00021070 | 10000 | 8.44340 | 8.44341 | 8.44340 | 7.0e-07 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00006108 | 10000 | 2.97629 | 2.97629 | 2.97629 | 2.3e-08 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00041536 | 500 | 87.72425 | 87.72434 | 87.72425 | 1.0e-06 |
| b5_main_TtBA_vgg19_T/targeted | TtBA | vgg19 | ILSVRC2012_val_00046345 | 2000 | 28.45835 | 28.45836 | 28.45835 | 2.1e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00025104 | 200 | 8.98121 | 8.98121 | 8.98121 | 3.8e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00044630 | 2500 | 0.87863 | 0.87863 | 0.87863 | 6.8e-08 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00025492 | 2000 | 2.40739 | 2.40739 | 2.40739 | 3.6e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00042392 | 1000 | 3.00176 | 3.00176 | 3.00176 | 4.4e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00007208 | 10000 | 1.43030 | 1.43030 | 1.43030 | 1.0e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00037987 | 1000 | 2.11911 | 2.11911 | 2.11911 | 1.9e-06 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00023118 | 500 | 12.94364 | 12.94364 | 12.94364 | 3.1e-07 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00025056 | 10000 | 2.26936 | 2.26936 | 2.26936 | 1.0e-06 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00027876 | 1000 | 0.71397 | 0.71397 | 0.71397 | 1.2e-06 |
| b5_main_TtBA_vit_b_32_NT/non_targeted | TtBA | vit_b_32 | ILSVRC2012_val_00001583 | 100 | 31.20345 | 31.20341 | 31.20345 | 1.4e-06 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00044435 | 100 | 70.53155 | 70.53154 | 70.53156 | 1.1e-07 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00037908 | 2000 | 10.98526 | 10.98527 | 10.98526 | 1.6e-07 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00014392 | 200 | 85.30973 | 85.30973 | 85.30973 | 2.9e-08 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00016228 | 2500 | 10.48508 | 10.48510 | 10.48508 | 1.8e-06 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00033435 | 1000 | 46.87855 | 46.87856 | 46.87855 | 2.2e-07 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00001468 | 1000 | 58.71441 | 58.71434 | 58.71441 | 1.3e-06 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00024906 | 1000 | 29.88871 | 29.88872 | 29.88871 | 4.3e-07 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00040228 | 2000 | 39.57251 | 39.57244 | 39.57251 | 1.9e-06 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00034111 | 500 | 51.93935 | 51.93934 | 51.93935 | 6.8e-08 |
| b5_main_TtBA_vit_b_32_T/targeted | TtBA | vit_b_32 | ILSVRC2012_val_00023435 | 200 | 45.15514 | 45.15518 | 45.15514 | 7.7e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00003815 | 500 | 16.39260 | 16.39261 | 16.39260 | 3.7e-08 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00008570 | 1000 | 53.14880 | 53.14883 | 53.14879 | 6.9e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00008234 | 100 | 32.04947 | 32.04953 | 32.04947 | 2.0e-06 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00037273 | 10000 | 0.97911 | 0.97911 | 0.97911 | 8.5e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 5000 | 9.27896 | 9.27895 | 9.27896 | 8.9e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00030496 | 2500 | 32.19803 | 32.19802 | 32.19803 | 2.8e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00046790 | 5000 | 27.90713 | 27.90717 | 27.90713 | 1.4e-06 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00008570 | 2000 | 43.07471 | 43.07475 | 43.07472 | 8.9e-07 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00010608 | 2000 | 14.16043 | 14.16041 | 14.16043 | 1.6e-06 |
| appendix_cgba_d8only_Salman2020Do_R50/non_targeted | CGBA_D8only | Salman2020Do_R50 | ILSVRC2012_val_00004469 | 500 | 53.76603 | 53.76603 | 53.76602 | 7.1e-08 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00008547 | 1000 | 1.49034 | 1.49034 | 1.49034 | 6.3e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00000599 | 5000 | 0.73932 | 0.73932 | 0.73932 | 4.5e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00028226 | 5000 | 0.10691 | 0.10691 | 0.10691 | 3.5e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00039775 | 5000 | 0.49888 | 0.49888 | 0.49888 | 1.1e-06 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00037041 | 500 | 3.79938 | 3.79938 | 3.79938 | 4.8e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00040389 | 2500 | 0.74522 | 0.74522 | 0.74522 | 2.0e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00004632 | 500 | 4.46925 | 4.46925 | 4.46925 | 5.6e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00025334 | 100 | 9.81505 | 9.81507 | 9.81505 | 2.0e-06 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00038491 | 500 | 48.60375 | 48.60379 | 48.60374 | 7.7e-07 |
| appendix_cgba_d8only_resnet50/non_targeted | CGBA_D8only | resnet50 | ILSVRC2012_val_00025028 | 2500 | 18.05000 | 18.05001 | 18.05000 | 1.9e-07 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00046951 | 1000 | 1.88851 | 1.88851 | 1.88851 | 9.3e-07 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00037360 | 200 | 17.95649 | 17.95652 | 17.95648 | 2.0e-06 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00022794 | 2000 | 3.28914 | 3.28915 | 3.28914 | 1.7e-06 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00001389 | 2500 | 0.45334 | 0.45334 | 0.45334 | 7.4e-07 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00039161 | 500 | 0.71682 | 0.71682 | 0.71682 | 2.5e-07 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00036362 | 5000 | 2.31252 | 2.31252 | 2.31252 | 1.3e-06 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00028797 | 200 | 14.68291 | 14.68292 | 14.68291 | 1.0e-06 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00036362 | 10000 | 1.43109 | 1.43110 | 1.43109 | 1.4e-06 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00002981 | 200 | 5.70710 | 5.70711 | 5.70710 | 4.8e-07 |
| appendix_cgba_d8only_vgg19/non_targeted | CGBA_D8only | vgg19 | ILSVRC2012_val_00011230 | 500 | 14.53705 | 14.53711 | 14.53705 | 3.5e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00024906 | 5000 | 6.21188 | 6.21188 | 6.21188 | 1.1e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00023118 | 1000 | 6.52662 | 6.52662 | 6.52662 | 1.0e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00038119 | 100 | 19.07617 | 19.07619 | 19.07617 | 1.0e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00027253 | 10000 | 0.56901 | 0.56901 | 0.56901 | 1.9e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00029074 | 500 | 11.82977 | 11.82978 | 11.82977 | 1.2e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00013588 | 2000 | 1.68816 | 1.68816 | 1.68816 | 3.1e-07 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00046481 | 2500 | 2.20433 | 2.20432 | 2.20433 | 2.1e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00046402 | 2000 | 5.61879 | 5.61879 | 5.61879 | 5.1e-07 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00037952 | 2500 | 1.01015 | 1.01015 | 1.01015 | 1.7e-06 |
| appendix_cgba_d8only_vit_b_32/non_targeted | CGBA_D8only | vit_b_32 | ILSVRC2012_val_00026874 | 200 | 7.01477 | 7.01478 | 7.01477 | 5.3e-07 |

- rows checked: 520

## 5. Curves and stalls (descriptive; a human reads the shape, not the winner)

- **b5_e10_ARS_OPT_densenet161_places365_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 30.3; 200: 1.00 / 28.8; 500: 1.00 / 26.7; 1000: 1.00 / 17.4; 2000: 1.00 / 10.8; 2500: 1.00 / 8.65; 5000: 1.00 / 4.79; 10000: 1.00 / 2.68
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_ARS_OPT_densenet161_places365_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 90.1; 200: 1.00 / 90.1; 500: 1.00 / 80.6; 1000: 1.00 / 69; 2000: 1.00 / 57.4; 2500: 1.00 / 52.1; 5000: 1.00 / 37.9; 10000: 1.00 / 22
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_ARS_OPT_resnet50_places365_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 31; 200: 1.00 / 29.5; 500: 1.00 / 27.5; 1000: 1.00 / 18.2; 2000: 1.00 / 10.5; 2500: 1.00 / 8.74; 5000: 1.00 / 4.88; 10000: 1.00 / 3.15
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_ARS_OPT_resnet50_places365_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 93; 200: 1.00 / 93; 500: 1.00 / 84.9; 1000: 1.00 / 72.3; 2000: 1.00 / 61.6; 2500: 1.00 / 57.1; 5000: 1.00 / 42.6; 10000: 1.00 / 27.2
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_CGBA_H_densenet161_places365_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 84; 200: 1.00 / 78.4; 500: 1.00 / 65.1; 1000: 1.00 / 55.5; 2000: 1.00 / 41.7; 2500: 1.00 / 35.8; 5000: 1.00 / 20.5; 10000: 1.00 / 9.83
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_CGBA_H_resnet50_places365_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 85.4; 200: 1.00 / 80.7; 500: 1.00 / 69.5; 1000: 1.00 / 60.5; 2000: 1.00 / 48.3; 2500: 1.00 / 43.4; 5000: 1.00 / 26.6; 10000: 1.00 / 13.6
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_CGBA_densenet161_places365_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 0.99 / 19.2; 200: 1.00 / 16; 500: 1.00 / 8.87; 1000: 1.00 / 5.51; 2000: 1.00 / 3.09; 2500: 1.00 / 2.73; 5000: 1.00 / 1.68; 10000: 1.00 / 1.11
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_CGBA_resnet50_places365_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 19.8; 200: 1.00 / 17.4; 500: 1.00 / 9.38; 1000: 1.00 / 6.27; 2000: 1.00 / 3.32; 2500: 1.00 / 2.76; 5000: 1.00 / 1.58; 10000: 1.00 / 1.02
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_LGC_RC_EGC_densenet161_places365_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 26.7; 200: 1.00 / 18.2; 500: 1.00 / 11.4; 1000: 1.00 / 6.14; 2000: 1.00 / 3.63; 2500: 1.00 / 3.09; 5000: 1.00 / 1.88; 10000: 1.00 / 1.13
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_LGC_RC_EGC_densenet161_places365_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 85.1; 200: 1.00 / 73.9; 500: 1.00 / 63.9; 1000: 1.00 / 50.2; 2000: 1.00 / 35.3; 2500: 1.00 / 29.8; 5000: 1.00 / 15.8; 10000: 1.00 / 9.07
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_LGC_RC_EGC_resnet50_places365_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 25.9; 200: 1.00 / 18.3; 500: 1.00 / 11.7; 1000: 1.00 / 6.2; 2000: 1.00 / 3.67; 2500: 1.00 / 3.09; 5000: 1.00 / 1.72; 10000: 1.00 / 1.16
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_LGC_RC_EGC_resnet50_places365_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 87.3; 200: 1.00 / 77.6; 500: 1.00 / 67; 1000: 1.00 / 53.8; 2000: 1.00 / 41.2; 2500: 1.00 / 34.5; 5000: 1.00 / 20.3; 10000: 1.00 / 11.2
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_TtBA_densenet161_places365_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 19.3; 200: 1.00 / 15.1; 500: 1.00 / 9.75; 1000: 1.00 / 5.54; 2000: 1.00 / 3.3; 2500: 1.00 / 2.79; 5000: 1.00 / 1.72; 10000: 1.00 / 1.02
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_TtBA_densenet161_places365_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 82.4; 200: 1.00 / 79.5; 500: 1.00 / 68.6; 1000: 1.00 / 53.6; 2000: 1.00 / 41.7; 2500: 1.00 / 36.5; 5000: 1.00 / 19.4; 10000: 1.00 / 9.39
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_TtBA_resnet50_places365_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 19.9; 200: 1.00 / 15.8; 500: 1.00 / 9.89; 1000: 1.00 / 5.69; 2000: 1.00 / 3.5; 2500: 1.00 / 3.02; 5000: 1.00 / 1.72; 10000: 1.00 / 1.1
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e10_TtBA_resnet50_places365_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 85.4; 200: 1.00 / 82.2; 500: 1.00 / 75.6; 1000: 1.00 / 65.2; 2000: 1.00 / 51.6; 2500: 1.00 / 46.1; 5000: 1.00 / 28; 10000: 1.00 / 12.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_ARS_OPT_Salman2020Do_R50_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 33.5; 200: 1.00 / 33.2; 500: 1.00 / 32.8; 1000: 1.00 / 26.2; 2000: 1.00 / 20.2; 2500: 1.00 / 18.2; 5000: 1.00 / 13.4; 10000: 1.00 / 10.2
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_ARS_OPT_Salman2020Do_R50_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 90.3; 200: 1.00 / 90.3; 500: 1.00 / 86.7; 1000: 1.00 / 83.3; 2000: 1.00 / 79.1; 2500: 1.00 / 77.4; 5000: 1.00 / 70.5; 10000: 1.00 / 60.2
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_CGBA_H_Salman2020Do_R50_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 87.9; 200: 1.00 / 86.6; 500: 1.00 / 80.3; 1000: 1.00 / 76.2; 2000: 1.00 / 68.2; 2500: 1.00 / 65.5; 5000: 1.00 / 52.6; 10000: 1.00 / 36.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_CGBA_Salman2020Do_R50_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 31.5; 200: 1.00 / 29.7; 500: 1.00 / 23; 1000: 1.00 / 18.9; 2000: 1.00 / 13.6; 2500: 1.00 / 12.5; 5000: 1.00 / 8.97; 10000: 1.00 / 6.32
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_LGC_RC_EGC_Salman2020Do_R50_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 30.8; 200: 1.00 / 22.1; 500: 1.00 / 14.9; 1000: 1.00 / 10.1; 2000: 1.00 / 8.09; 2500: 1.00 / 7.47; 5000: 1.00 / 6.27; 10000: 1.00 / 5.35
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_LGC_RC_EGC_Salman2020Do_R50_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 89.8; 200: 1.00 / 83.3; 500: 1.00 / 77.2; 1000: 1.00 / 66; 2000: 1.00 / 52; 2500: 1.00 / 45.9; 5000: 1.00 / 31.8; 10000: 1.00 / 22.3
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_TtBA_Salman2020Do_R50_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 29; 200: 1.00 / 26.6; 500: 1.00 / 22.3; 1000: 1.00 / 17.6; 2000: 1.00 / 13.4; 2500: 1.00 / 12.1; 5000: 1.00 / 8.9; 10000: 1.00 / 6.29
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_e11_TtBA_Salman2020Do_R50_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 87.4; 200: 1.00 / 86.5; 500: 1.00 / 82.8; 1000: 1.00 / 76.7; 2000: 1.00 / 70.3; 2500: 1.00 / 67.3; 5000: 1.00 / 53.9; 10000: 1.00 / 35.9
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_resnet50_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 49.5; 200: 1.00 / 45.5; 500: 1.00 / 44; 1000: 1.00 / 27.6; 2000: 1.00 / 14.8; 2500: 1.00 / 12.2; 5000: 1.00 / 5.01; 10000: 1.00 / 2.37
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_resnet50_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 79; 200: 1.00 / 79; 500: 1.00 / 73.2; 1000: 1.00 / 67.1; 2000: 1.00 / 55.3; 2500: 1.00 / 51; 5000: 1.00 / 34.6; 10000: 1.00 / 18.7
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_vgg19_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 30.3; 200: 1.00 / 27.6; 500: 1.00 / 25.3; 1000: 1.00 / 13.6; 2000: 1.00 / 6.94; 2500: 1.00 / 5.23; 5000: 1.00 / 2.57; 10000: 1.00 / 1.3
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_vgg19_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 92.5; 200: 1.00 / 92.5; 500: 1.00 / 83.5; 1000: 1.00 / 72; 2000: 1.00 / 55.1; 2500: 1.00 / 48.7; 5000: 1.00 / 29.5; 10000: 1.00 / 15.3
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_vit_b_32_NT/non_targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 79.8; 200: 1.00 / 68.4; 500: 1.00 / 64.9; 1000: 1.00 / 29.5; 2000: 1.00 / 14.4; 2500: 1.00 / 11.5; 5000: 1.00 / 5.87; 10000: 1.00 / 3.24
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_ARS_OPT_vit_b_32_T/targeted ARS_OPT** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 79.2; 200: 1.00 / 79.2; 500: 1.00 / 68.3; 1000: 1.00 / 53.8; 2000: 1.00 / 37.3; 2500: 1.00 / 32.5; 5000: 1.00 / 19.1; 10000: 1.00 / 9.87
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_H_resnet50_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 75.1; 200: 1.00 / 72; 500: 1.00 / 64.1; 1000: 1.00 / 55.7; 2000: 1.00 / 46.1; 2500: 1.00 / 42.1; 5000: 1.00 / 27.9; 10000: 1.00 / 14.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_H_vgg19_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 87.4; 200: 1.00 / 82; 500: 1.00 / 67.7; 1000: 1.00 / 55.8; 2000: 1.00 / 40.4; 2500: 1.00 / 35.1; 5000: 1.00 / 19.6; 10000: 1.00 / 8.11
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_H_vit_b_32_T/targeted CGBA_H** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 70.7; 200: 1.00 / 65.3; 500: 1.00 / 49.1; 1000: 1.00 / 36.7; 2000: 1.00 / 22.2; 2500: 1.00 / 17.6; 5000: 1.00 / 8.23; 10000: 1.00 / 4.22
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_resnet50_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 0.97 / 38.6; 200: 1.00 / 32; 500: 1.00 / 19.2; 1000: 1.00 / 12; 2000: 1.00 / 5.87; 2500: 1.00 / 4.33; 5000: 1.00 / 1.93; 10000: 1.00 / 1.06
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_vgg19_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 0.99 / 20.1; 200: 1.00 / 15.2; 500: 1.00 / 7.74; 1000: 1.00 / 3.96; 2000: 1.00 / 2.23; 2500: 1.00 / 1.87; 5000: 1.00 / 1.12; 10000: 1.00 / 0.771
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_CGBA_vit_b_32_NT/non_targeted CGBA** (images 200): found rate / median pix_l2 per budget: 100: 0.88 / 70.7; 200: 0.99 / 25.1; 500: 1.00 / 11.1; 1000: 1.00 / 5.94; 2000: 1.00 / 3.52; 2500: 1.00 / 2.87; 5000: 1.00 / 1.66; 10000: 1.00 / 1.16
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_resnet50_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 35.5; 200: 1.00 / 26.6; 500: 1.00 / 17.4; 1000: 1.00 / 9.68; 2000: 1.00 / 5.87; 2500: 1.00 / 4.72; 5000: 1.00 / 2.72; 10000: 1.00 / 1.26
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_resnet50_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 80.9; 200: 1.00 / 74.4; 500: 1.00 / 66; 1000: 1.00 / 53.2; 2000: 1.00 / 41.7; 2500: 1.00 / 35.9; 5000: 1.00 / 19.2; 10000: 1.00 / 9.68
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_vgg19_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 23.9; 200: 1.00 / 15.6; 500: 1.00 / 9.87; 1000: 1.00 / 5.64; 2000: 1.00 / 3.41; 2500: 1.00 / 2.61; 5000: 1.00 / 1.35; 10000: 1.00 / 0.822
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_vgg19_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 89.7; 200: 1.00 / 83.8; 500: 1.00 / 72.7; 1000: 1.00 / 58.1; 2000: 1.00 / 42.2; 2500: 1.00 / 35.3; 5000: 1.00 / 18; 10000: 1.00 / 9.65
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_vit_b_32_NT/non_targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 25; 200: 1.00 / 14.9; 500: 1.00 / 8.62; 1000: 1.00 / 5.19; 2000: 1.00 / 3.23; 2500: 1.00 / 2.67; 5000: 1.00 / 1.7; 10000: 1.00 / 1.13
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_LGC_RC_EGC_vit_b_32_T/targeted LGC_RC_EGC** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 75; 200: 1.00 / 66; 500: 1.00 / 54.2; 1000: 1.00 / 36.9; 2000: 1.00 / 20.1; 2500: 1.00 / 16.7; 5000: 1.00 / 8.68; 10000: 1.00 / 4.9
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_resnet50_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 36.7; 200: 1.00 / 29.8; 500: 1.00 / 17.5; 1000: 1.00 / 9.95; 2000: 1.00 / 5.84; 2500: 1.00 / 4.73; 5000: 1.00 / 2.12; 10000: 1.00 / 1.16
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_resnet50_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 75.3; 200: 1.00 / 73.6; 500: 1.00 / 66.6; 1000: 1.00 / 59.8; 2000: 1.00 / 48.2; 2500: 1.00 / 43; 5000: 1.00 / 28.8; 10000: 1.00 / 13.5
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_vgg19_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 21.4; 200: 1.00 / 14.7; 500: 1.00 / 7.8; 1000: 1.00 / 3.99; 2000: 1.00 / 2.26; 2500: 1.00 / 1.88; 5000: 1.00 / 1.2; 10000: 1.00 / 0.814
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_vgg19_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 86; 200: 1.00 / 83; 500: 1.00 / 72.1; 1000: 1.00 / 57.5; 2000: 1.00 / 41.8; 2500: 1.00 / 36.7; 5000: 1.00 / 20.4; 10000: 1.00 / 8.46
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_vit_b_32_NT/non_targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 37.7; 200: 1.00 / 25.7; 500: 1.00 / 11.7; 1000: 1.00 / 5.71; 2000: 1.00 / 3.46; 2500: 1.00 / 2.95; 5000: 1.00 / 1.74; 10000: 1.00 / 1.17
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **b5_main_TtBA_vit_b_32_T/targeted TtBA** (images 200): found rate / median pix_l2 per budget: 100: 1.00 / 72.1; 200: 1.00 / 65.9; 500: 1.00 / 54.4; 1000: 1.00 / 37.2; 2000: 1.00 / 20.8; 2500: 1.00 / 16.9; 5000: 1.00 / 7.98; 10000: 1.00 / 4.07
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **appendix_cgba_d8only_Salman2020Do_R50/non_targeted CGBA_D8only** (images 30): found rate / median pix_l2 per budget: 100: 1.00 / 34.4; 200: 1.00 / 33; 500: 1.00 / 26.5; 1000: 1.00 / 22.3; 2000: 1.00 / 17.3; 2500: 1.00 / 15; 5000: 1.00 / 9.42; 10000: 1.00 / 6.7
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **appendix_cgba_d8only_resnet50/non_targeted CGBA_D8only** (images 30): found rate / median pix_l2 per budget: 100: 0.97 / 44.8; 200: 1.00 / 32.4; 500: 1.00 / 17.5; 1000: 1.00 / 8.09; 2000: 1.00 / 3.91; 2500: 1.00 / 2.47; 5000: 1.00 / 0.963; 10000: 1.00 / 0.62
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **appendix_cgba_d8only_vgg19/non_targeted CGBA_D8only** (images 30): found rate / median pix_l2 per budget: 100: 1.00 / 26.7; 200: 1.00 / 16.3; 500: 1.00 / 7.99; 1000: 1.00 / 3.95; 2000: 1.00 / 1.97; 2500: 1.00 / 1.69; 5000: 1.00 / 0.987; 10000: 1.00 / 0.68
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0
- **appendix_cgba_d8only_vit_b_32/non_targeted CGBA_D8only** (images 30): found rate / median pix_l2 per budget: 100: 0.90 / 67.5; 200: 1.00 / 25.6; 500: 1.00 / 12.3; 1000: 1.00 / 5.89; 2000: 1.00 / 3.38; 2500: 1.00 / 2.78; 5000: 1.00 / 1.61; 10000: 1.00 / 1.13
  - flat from 2000 to 10000 (possible stall): 0; images with an L2 increase between budgets: 0

An L2 increase is allowed only for our engine-state convention (I08 WARN); for a tracker-best baseline it is a bug. Flat 2000 -> 10000 for a baseline is what the CGBA NaN stall looked like.

## 7. Degeneracy (silent collapse: identical methods, constant outputs, never found, duplicated inputs)

- Salman2020Do_R50/non_targeted: CGBA and CGBA_D8only identical on 30/30 images  (declared identical by design)
- resnet50/non_targeted: CGBA and CGBA_D8only identical on 29/30 images  (declared identical by design)
- vgg19/non_targeted: CGBA and CGBA_D8only identical on 30/30 images  (declared identical by design)
- vit_b_32/non_targeted: CGBA and CGBA_D8only identical on 27/30 images  (declared identical by design)
- no degenerate pattern found

## 6. Visual grid (open every PNG: same clean image per row, plausible perturbations, no blank or NaN tiles)

- reports/human_preflight_b5a2_grid_Salman2020Do_R50_non_targeted.png: 3 images x 5 methods at budget 10000
- reports/human_preflight_b5a2_grid_Salman2020Do_R50_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_densenet161_places365_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_densenet161_places365_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_resnet50_non_targeted.png: 3 images x 5 methods at budget 10000
- reports/human_preflight_b5a2_grid_resnet50_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_resnet50_places365_non_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_resnet50_places365_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_vgg19_non_targeted.png: 3 images x 5 methods at budget 10000
- reports/human_preflight_b5a2_grid_vgg19_targeted.png: 3 images x 4 methods at budget 10000
- reports/human_preflight_b5a2_grid_vit_b_32_non_targeted.png: 3 images x 5 methods at budget 10000
- reports/human_preflight_b5a2_grid_vit_b_32_targeted.png: 3 images x 4 methods at budget 10000

