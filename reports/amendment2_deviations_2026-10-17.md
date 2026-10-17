# Amendment-2 deviations (B5 pre-run)

## 1. E10 (Places365) - RESTORED (Reviewer-defensible reason)
Reason: After reassessing the paper's core claims, we found that the generalization evidence cannot rely solely on the robust model (E11). The TIFS rejection cited a lack of cross-dataset generalization. While E10 was initially cut due to cost after D19 widening, the revised budget (using the 8x RTX 5090 efficiently) allows us to restore E10. This ensures the paper has the necessary cross-dataset evidence for TMM/PR.

## 2. E12 (Low-query breadth) CUT
Reason: E12 was redefined as "low-query extension on robust R50". Since E11 records all 8 checkpoints per D1, a separate E12 run is redundant.

## 3. Study GPU
RTX 5090 (driver 595.71.05) confirmed. RTX PRO 6000 migration deferred (D16 rev1).

## 4. Oracle domain (D19)
ARS_OPT uses clamp01. CGBA / CGBA_H use clamp_normalized. Both are implemented and gated by domain_consistency.py.

## 5. E11 ARS_OPT addition
ARS_OPT added to E11 roster (same victim, modes and n). Contract v1.9-rev5 reflects this.
