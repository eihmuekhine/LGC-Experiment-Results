# B5 T0b STOP — E-OTHER (X > 0)
Date: 2026-10-18T11:28:57+08:00
HEAD: a3107a7b55698f3a8f969750bc956f483e36c349
contract sha256: 8c3327f0096a386debd3551a23e36a0c804abc39ac07f3e63de5d9c8293971a4
prompt.txt sha256: 2eb8698e3ac874196b518386c310702add8be63509689ceae2300ec89b905afb
Exit code: 3

## Trigger
X > 0 for LGC_RC_EGC on 2 cells (post-fix rule: any X>0 is E-OTHER STOP).

## X>0 cells
- LGC_RC_EGC vgg19/targeted: X=2/24
- LGC_RC_EGC Salman2020Do_R50/targeted: X=2/24

## Root provenance
X>0 ONLY on b4e2_smoke_LGC_RC_EGC_* roots (pre-D19-fix data).
b4h_t3_smoke_* (post-D19-fix) and b5host_* roots: X=0.

## All other cells
46/48 cells: X=0

## X rows (per image, class=X; class is column 8, CR stripped)
```
LGC_RC_EGC,vgg19,targeted,ILSVRC2012_val_00002148,100,398,133,X,0.0014324188232421875,0.0055141448974609375,0.03186607360839844,0,0,0,1
LGC_RC_EGC,vgg19,targeted,ILSVRC2012_val_00002148,200,398,133,X,0.00074005126953125,0.0048084259033203125,0.04148244857788086,0,0,0,0
LGC_RC_EGC,Salman2020Do_R50,targeted,ILSVRC2012_val_00046453,1000,581,669,X,0.0001697540283203125,0.02070331573486328,0.16577816009521484,0,0,0,1
LGC_RC_EGC,Salman2020Do_R50,targeted,ILSVRC2012_val_00046453,2000,581,669,X,0.00012636184692382812,0.02385711669921875,0.19058752059936523,0,0,0,1
```

## Analysis
- Both X>0 cells are TARGETED mode, LGC_RC_EGC method.
- All m_s1, m_s9, m_s6 are POSITIVE (target not argmax even at s=1+2^-6).
- adv_uint8=1 for 3/4 rows: uint8-quantized image IS adversarial,
  but float16->float32 stored-delta evaluation is NOT.
  Consistent with float16 storage precision at the decision boundary.
- X>0 ONLY on b4e2_smoke_LGC_RC_EGC_* roots (pre-D19-fix data).
- Post-fix roots (b4h_t3_smoke_*, b5host_*): X=0 for all methods.

## Hypotheses
H1: Pre-D19-fix stored deltas lose enough precision in float16->float32
    that the decision flips for boundary images (supported by adv_uint8=1).
H2: Pre-fix per_checkpoint 'found=1' was recorded under a different
    precision path than the current D12 evaluation.

## Recommended next step
Run T0b smoke on post-D19-fix roots only (b4h_t3_smoke_* + b5host_*).
If X=0 there, pre-fix X>0 is an artifact of pre-fix data, and B5
(which generates fresh post-fix data) should be safe to launch.
Requires H decision.

## Post-fix smoke result (follow-up)
- Roots: b4h_t3_smoke_* + b5host_* only (12 roots)
- X count: 0
- EXIT: 0
- Message: "[+] no X>0 for any method (ARS_OPT and CGBA/CGBA_H both clamp correctly)"

## ⚠️ Limitation (important)
Post-fix roots contain ONLY CGBA/CGBA_H and ARS_OPT.
LGC_RC_EGC has NO post-D19-fix data in the current roots.
Therefore this smoke does NOT directly test the method that showed X>0.
The X=0 result supports H1 (pre-fix artifact) only indirectly.
A definitive test requires fresh LGC_RC_EGC data generated under the
current post-D19-fix code path — i.e., the B5 run itself.
