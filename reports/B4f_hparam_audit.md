# B4f T4 — hyper-parameter audit (L2/L3)

Official defaults are read from the clones (ast-parsed `__init__` signatures + `PARS_OPT.json`), not from memory or READMEs. Adapter values are read from the adapter `PARAMS`/`BASE_PARAMS` dicts and the explicit constructor calls with file:line.

## Official defaults extracted from the clones (evidence)

- TtBA `Attacker.__init__` (`TtBA.py:18-20`): `{'attack_method': 'TtBA', 'dim_reduc_factor': 4, 'iteration': 100, 'initial_query': 30, 'tol': 0.0001, 'sigma': 0.0003}`
- CGBA `Proposed_attack.__init__` (`proposed_attack.py:13-16`): `{'dim_reduc_factor': 4, 'attack_method': 'CGBA_H', 'iteration': 93, 'initial_query': 30, 'tol': 0.0001, 'sigma': 0.0002, 'verbose_control': 'Yes'}`; `grad_estimator_batch_size = 40` (`proposed_attack.py:32`, class attribute)
- ARS-OPT `ParsOptL2Norm.__init__` (`pars_opt_l2_norm_attack.py:16-18`): `{'batch_size': 1, 'k': 100, 'alpha': 0.2, 'beta': 0.001, 'iterations': 1000, 'maximum_queries': 10000, 'method': 'ARS', 'clip_grad_max_norm': 1.0, 'best_initial_target_sample': False, 'resized': False, 'dim_reduce_factor': 4, 'line_search': False}`
- ARS-OPT `PARS_OPT.json` ImageNet.l2: untargeted `{'epsilon': 10.0, 'clip_grad_max_norm': 1.0, 'tol': 0.0001, 'lr': 0.4}`, targeted `{'epsilon': 10.0, 'clip_grad_max_norm': 1.0, 'tol': 0.0001, 'lr': 0.2}`

## TtBA (both modes; ImageNet and E10/E11)

| name | adapter value | adapter file:line | official default | official file:line | layer | DIFFERS? | L3 reason |
|---|---|---|---|---|---|---|---|
| budget | 10000 (contract) | run_ttba_v2.py:162,349 | 10000 | main.py:22 | L1 | no |  |
| epsilon | 2.5 | run_ttba_v2.py:242 | 0.3 | main.py:21 | L2 | yes | ImageNet L2 success threshold (contract asr_l2 2.5); early=0 so it only sets the recorded success_eps, never the trajectory |
| early | 0 | run_ttba_v2.py:162,349 | 0 | main.py:23 | L1 | no |  |
| remember | 0 | run_ttba_v2.py:162,349 | 0 | main.py:24 | L1 | no |  |
| targeted | mode | run_ttba_v2.py:162,349 | 0/1 | main.py:19 | L1 | no |  |
| norm | 'TtBA' | run_ttba_v2.py:162,349 | main.py:70 branch | main.py:70 | L1 | no |  |
| dim_reduc_factor | 4 (relied) | run_ttba_v2.py:55 | 4 | TtBA.py:19 | L2 | no |  |
| iteration | 100 (relied) | run_ttba_v2.py:59 | 100 | TtBA.py:20 | L2 | no |  |
| initial_query | 30 (relied) | run_ttba_v2.py:58 | 30 | TtBA.py:20 | L2 | no |  |
| tol | 1e-4 (relied) | run_ttba_v2.py:56 | 1e-4 | TtBA.py:20 | L2 | no |  |
| sigma | 3e-4 (relied) | run_ttba_v2.py:57 | 3e-4 | TtBA.py:20 | L2 | no |  |
| attack_method | 'TtBA' (relied) | TtBA.py:18 | 'TtBA' | TtBA.py:18 | L2 | no |  |

## CGBA (non_targeted) / CGBA_H (targeted) / CGBA_D8only (non_targeted, E17 appendix)

| name | adapter value | adapter file:line | official default | official file:line | layer | DIFFERS? | L3 reason |
|---|---|---|---|---|---|---|---|
| dim_reduc_factor | 4 | run_cgba.py:50 | 4 | proposed_attack.py:14 | L2 | no |  |
| iteration | 93 | run_cgba.py:51 | 93 | Non_targeted_attack.py:30 / Targeted_attack.py:33 | L2 | no |  |
| initial_query | 30 | run_cgba.py:52 | 30 | proposed_attack.py:15 | L2 | no |  |
| tol | 1e-4 | run_cgba.py:53 | 1e-4 | proposed_attack.py:15 | L2 | no |  |
| sigma | 2e-4 | run_cgba.py:54 | 2e-4 | proposed_attack.py:15 | L2 | no |  |
| verbose_control | 'No' | run_cgba.py:55 | 'Yes' | proposed_attack.py:16 | L3 | yes | print-only logging control; algorithm unchanged (allowed logging change) |
| attack_method | 'CGBA' (NT) / 'CGBA_H' (T) | run_cgba.py:255 | 'CGBA_H' (class default) | proposed_attack.py:14 | L2 | no | CGBA_H is the class default; the driver selects CGBA/CGBA_H per run (Non_targeted_attack.py:32) |
| mean / std | ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225] | run_cgba.py:180 (C.MEAN/C.STD) | same | Non_targeted_attack.py:35-36 / Targeted_attack.py:36-37 | L2 | no |  |
| lb / ub | normalize(0)/normalize(1) | run_cgba.py:178-179 | valid_bounds(delta=255) -> normalize(0)/normalize(1) | Non_targeted_attack.py:80-81,91-92 / utils.py:13-34 | L2 | no |  |
| grad_estimator_batch_size | 40 (relied) | proposed_attack.py:32 | 40 | proposed_attack.py:32 | L2 | no |  |
| D8 count fix (boundary-walk query count) | go_to_boundary_* return true count | run_cgba.py:128-136 | official returns num_calls==40 after 39 calls (off-by-one) | proposed_attack.py:144,164,252 | L4 | yes | contract D8: faithful model + boundary-walk query-count fix (trajectory bit-identical, count corrected) |
| D11 acos clip (fixed variant) | clamp(arg,-1,1) at SemiCircular_boundary_search | run_cgba.py:150-153 | unclamped torch.acos(arg) | proposed_attack.py:182 | L4 | yes | contract D11_outcome (verdict b): NaN freeze; verbatim copy, only value-changing edit is the clamp |
| model object | raw net (normalised tensor in) | run_cgba.py:82-84 (_faithful_model) | net = resnet50(pretrained=True) (no normalisation layer) | Non_targeted_attack.py:42 / Targeted_attack.py:43 | L3 | yes | faithful model: normalised tensor goes straight into the un-normalised net (B3b T2) |

## ARS-OPT (non_targeted / targeted; prior-free method='ARS')

| name | adapter value | adapter file:line | official default | official file:line | layer | DIFFERS? | L3 reason |
|---|---|---|---|---|---|---|---|
| method | 'ARS' (prior-free) | run_arsopt.py:112 | 'ARS' (class default) | pars_opt_l2_norm_attack.py:16 | L2 | no |  |
| epsilon | 10.0 | run_arsopt.py:36 | 10.0 | PARS_OPT.json ImageNet.l2 | L2 | no |  |
| clip_grad_max_norm | 1.0 | run_arsopt.py:37 | 1.0 | PARS_OPT.json ImageNet.l2 | L2 | no |  |
| tol | 1e-4 | run_arsopt.py:38 | 1e-4 | PARS_OPT.json ImageNet.l2 | L2 | no |  |
| k (est_grad_samples) | 200 | run_arsopt.py:39 | 200 | attack.py:62 / pars_opt_l2_norm_attack.py:17 | L2 | no |  |
| beta | 0.001 | run_arsopt.py:40 | 0.001 | pars_opt_l2_norm_attack.py:17 | L2 | no |  |
| iterations | 1000 | run_arsopt.py:41 | 1000 | pars_opt_l2_norm_attack.py:17 | L2 | no |  |
| maximum_queries | 10000 (contract) | run_arsopt.py:42,112 | 10000 (class) but driver forces 20000 for ImageNet targeted | pars_opt_l2_norm_attack.py:18 / attack.py:127-128 | L3 | yes | contract budget 10000 for every method (F4); the official driver's 20000 override is bypassed by calling targeted_attack directly |
| dim_reduce_factor | 4 | run_arsopt.py:43 | 4 | attack.py:98 / pars_opt_l2_norm_attack.py:18 | L2 | no |  |
| lr (alpha) | 0.4 (NT) / 0.2 (T) | run_arsopt.py:50,132 | 0.4 (NT) / 0.2 (T) | PARS_OPT.json ImageNet.l2 | L2 | no |  |
| grad_binary_search_tol | None | run_arsopt.py:114 | None | pars_opt_l2_norm_attack.py:18 | L2 | no |  |
| best_initial_target_sample | False | run_arsopt.py:114 | False | pars_opt_l2_norm_attack.py:18 | L2 | no |  |
| resized | False | run_arsopt.py:114 | False | pars_opt_l2_norm_attack.py:18 | L2 | no |  |
| line_search | False | run_arsopt.py:115 | False | pars_opt_l2_norm_attack.py:19 | L2 | no |  |
| batch_size | 1 | run_arsopt.py:111 | 1 (driver default) | attack.py:65 | L2 | no |  |
| model object | HardLabelTracker (decision-only) | run_arsopt.py:98-102 | StandardModel (returns logits) | models/standard_model.py | L3 | yes | hard-label wrapper; self.model(x).max(1)[1] reads only the top-1 decision |
| data input | our x0 + CSV target image | run_arsopt.py:116-121 | ImageNet dataset / select_random_image_of_target_class | attack.py:893 | L3 | yes | shared split (F1/F7); bypasses the repo's own dataset selection |
| import shims (glog/bidict/dataset.npz_dataset) | no-op stubs | run_arsopt.py:63-94 | real deps (dataset.npz_dataset missing from clone) | — | L3 | yes | never executed by the ARS loop; exist only so the official module imports |

## Layer summary

- L2 (method internals): all at official ImageNet defaults (DIFFERS = no), except the two L4 corrections and the L3 interface/logger changes listed above.
- L3 (interface): model object, data feeding, query-limit (maximum_queries=10000), verbose_control='No'.
- L4 (corrections, contract): CGBA D8 count fix + D11 acos clip.

