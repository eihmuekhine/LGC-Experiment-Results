# B5a2 stop decisions (A1)

Rule fixed by H on 2026-10-22 before any outcome statistic was computed.

## Decision rule (verbatim)

For each X row print m_s9/m_s1 and m_s6/m_s9 and classify it:
    STORAGE: 0 < m_s1 < 1e-2 and m_s1 < m_s9 < m_s6
    DEFECT:  m_s1 >= 0.1 and m_s9 < m_s1
    OTHER:   anything else
  Cell decision:
    INCLUDED              if every X row of the cell is STORAGE;
    EXCLUDED-FROM-CLAIMS  otherwise.

## Sign-convention quote

`adapters/diag/domain_consistency.py:12-15` (docstring):

> * adversarial: NT ``argmax != y``; T ``argmax == t``;
> * margin m: NT ``l_y - max_{j!=y} l_j``; T ``max_{j!=t} l_j - l_t``;
> * class: ``C`` if adversarial at s=1; ``Q9`` if not at s=1 but adversarial at s=1+2^-9; ``Q6`` if the first
>   adversarial point is s=1+2^-6; ``X`` otherwise.

Implementation: `adapters/diag/domain_consistency.py:73-79` (`margins`) and `:95-103` (`classify`). Convention: m > 0
means the victim's decision on that image is NOT adversarial.

## Per-X-row classification

| cell | image | budget | source | target | m_s1 | m_s9 | m_s6 | m_s9/m_s1 | m_s6/m_s9 | class |
|---|---|---|---|---|---|---|---|---|---|---|
| e11_ARS_OPT_Salman2020Do_R50_T | ILSVRC2012_val_00048242 | 10000 | 8 | 980 | 0.0004200935363769531 | 0.00063323974609375 | 0.005953788757324219 | 1.50738 | 9.40211 | STORAGE |
| main_ARS_OPT_vit_b_32_T | ILSVRC2012_val_00038119 | 10000 | 126 | 121 | 8.96453857421875e-05 | 0.00013303756713867188 | 0.01583242416381836 | 1.48404 | 119.007 | STORAGE |
| main_ARS_OPT_vit_b_32_T | ILSVRC2012_val_00011074 | 10000 | 322 | 555 | 1.52587890625e-05 | 0.0005350112915039062 | 0.0664224624633789 | 35.0625 | 124.152 | STORAGE |
| main_CGBA_H_vgg19_T | ILSVRC2012_val_00026013 | 2500 | 765 | 150 | 0.0011262893676757812 | 0.013422489166259766 | 0.17752408981323242 | 11.9174 | 13.2259 | STORAGE |
| main_CGBA_H_vgg19_T | ILSVRC2012_val_00026013 | 5000 | 765 | 150 | 9.870529174804688e-05 | 0.022234439849853516 | 0.1855020523071289 | 225.261 | 8.34301 | STORAGE |
| main_CGBA_H_vgg19_T | ILSVRC2012_val_00026013 | 10000 | 765 | 150 | 0.0029206275939941406 | 0.0028061866760253906 | 0.03597068786621094 | 0.960816 | 12.8184 | OTHER |
| main_CGBA_vgg19_NT | ILSVRC2012_val_00032950 | 100 | 420 | | 0.0004482269287109375 | 0.00201416015625 | 0.01775074005126953 | 4.49362 | 8.81297 | STORAGE |
| main_CGBA_vgg19_NT | ILSVRC2012_val_00029574 | 100 | 875 | | 0.0001659393310546875 | 0.0004949569702148438 | 0.006702423095703125 | 2.98276 | 13.5414 | STORAGE |
| main_CGBA_H_resnet50_T | ILSVRC2012_val_00004916 | 1000 | 529 | 371 | 0.0005707740783691406 | 0.0033278465270996094 | 0.03032827377319336 | 5.83041 | 9.11348 | STORAGE |
| e11_CGBA_H_Salman2020Do_R50_T | ILSVRC2012_val_00012970 | 10000 | 856 | 176 | 0.0011811256408691406 | 9.250640869140625e-05 | 0.0001544952392578125 | 0.0783205 | 1.6701 | OTHER |
| e11_CGBA_H_Salman2020Do_R50_T | ILSVRC2012_val_00048936 | 2500 | 460 | 238 | 0.0003895759582519531 | 0.0003733634948730469 | 0.004421710968017578 | 0.958384 | 11.8429 | OTHER |
| main_CGBA_resnet50_NT | ILSVRC2012_val_00018952 | 500 | 107 | | 0.0006923675537109375 | 2.956390380859375e-05 | 0.03822946548461914 | 0.0426997 | 1293.11 | OTHER |
| main_CGBA_resnet50_NT | ILSVRC2012_val_00028484 | 200 | 826 | | 0.0006680488586425781 | 0.0024871826171875 | 0.01952838897705078 | 3.72305 | 7.85161 | STORAGE |
| main_CGBA_resnet50_NT | ILSVRC2012_val_00016786 | 100 | 902 | | 0.00014352798461914062 | 1.239776611328125e-05 | 0.0006842613220214844 | 0.0863787 | 55.1923 | OTHER |
| main_CGBA_vit_b_32_NT | ILSVRC2012_val_00047355 | 200 | 560 | | 0.00017213821411132812 | 0.00017452239990234375 | 0.0006771087646484375 | 1.01385 | 3.87978 | STORAGE |
| main_CGBA_H_vit_b_32_T | ILSVRC2012_val_00005153 | 10000 | 377 | 995 | 0.00014066696166992188 | 0.011196613311767578 | 0.08883857727050781 | 79.5966 | 7.93442 | STORAGE |

## Per-cell decision

| cell | method | rows | C | Q9 | Q6 | X | STORAGE | DEFECT | OTHER | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| e11_ARS_OPT_Salman2020Do_R50_T | ARS_OPT | 1600 | 1001 | 598 | 0 | 1 | 1 | 0 | 0 | INCLUDED |
| main_ARS_OPT_vit_b_32_T | ARS_OPT | 1600 | 1088 | 510 | 0 | 2 | 2 | 0 | 0 | INCLUDED |
| main_CGBA_H_vgg19_T | CGBA_H | 1600 | 1168 | 429 | 0 | 3 | 2 | 0 | 1 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_vgg19_NT | CGBA | 1599 | 1443 | 154 | 0 | 2 | 2 | 0 | 0 | INCLUDED |
| main_CGBA_H_resnet50_T | CGBA_H | 1600 | 1085 | 514 | 0 | 1 | 1 | 0 | 0 | INCLUDED |
| e11_CGBA_H_Salman2020Do_R50_T | CGBA_H | 1600 | 989 | 609 | 0 | 2 | 0 | 0 | 2 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_resnet50_NT | CGBA | 1594 | 1368 | 223 | 0 | 3 | 1 | 0 | 2 | EXCLUDED-FROM-CLAIMS |
| main_CGBA_vit_b_32_NT | CGBA | 1574 | 1429 | 144 | 0 | 1 | 1 | 0 | 0 | INCLUDED |
| main_CGBA_H_vit_b_32_T | CGBA_H | 1600 | 1282 | 316 | 1 | 1 | 1 | 0 | 0 | INCLUDED |

## Duplicate-bundle finding

Three CGBA_H T cells have two STOP bundles each; in each pair the first bundle records an E-G5 stop
(`g5 FAIL: G5 FAIL (CI outside [0.5,2] at a budget)`) and the second records an E-OTHER domain stop
(`domain FAIL: X > 0 (stop rule for CGBA_H)`). `diff` of the pairs changes only lines 3-5 (code / signature / evidence):

- `main_CGBA_H_resnet50_T`: `_20261020-2104.md` = E-G5; `_20261020-2104_2.md` = E-OTHER (X=1).
- `main_CGBA_H_vgg19_T`: `_20261020-1957.md` = E-G5; `_20261020-1957_2.md` = E-OTHER (X=3).
- `main_CGBA_H_vit_b_32_T`: `_20261020-2311.md` = E-G5; `_20261020-2311_2.md` = E-OTHER (X=1).

## Note (decision point 3)

The six INCLUDED cells are a deviation from the D19 stop rule (`X > 0` = STOP), under the H decision rule fixed
2026-10-22 (decision point 3). This deviation is to be listed in the close note.
