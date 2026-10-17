# Amendment-2 deviations (B5 pre-run)

## 1. E10 (Places365) CUT
Reason: ImageNet-only scope. Cost after D19 widening made Places365 prohibitively expensive. E11 (RobustBench R50) provides the generalization evidence.

## 2. E12 (Low-query breadth) CUT
Reason: E12 was redefined as "low-query extension on robust R50 (budget 1000, n=200)". Since E11 records all 8 checkpoints (including budget 1000) per D1, a separate E12 run is redundant.

## 3. Study GPU
RTX 5090 (driver 595.71.05) confirmed. RTX PRO 6000 migration deferred (D16 rev1).

## 4. Oracle domain (D19)
ARS_OPT uses clamp01. CGBA / CGBA_H use clamp_normalized. Both are implemented and gated by domain_consistency.py.

## 5. E11 ARS_OPT addition
ARS_OPT added to E11 roster (same victim, modes and n). Contract v1.9-rev4 reflects this.
