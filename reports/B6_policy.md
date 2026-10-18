# B6/P5 Stage Policy

## Decision
Option 1: B6 and P5 will run with B5 (this host) roots only.
Old-host roots are excluded from the root list for B6/P5 gates.

## Rationale
1. B5 already established and documented the HOST-DIFFERENT finding
   (torch/cuDNN build divergence). Re-triggering I10 at B6/P5 would be
   redundant and would guarantee a gate FAIL.
2. Final validation (B6) and paper stage (P5) require a clean,
   homogeneous root set for interpretable results.
3. Excluding old-host roots does not hide the heterogeneity — it is
   reported separately from B5.
4. Avoids a ~18h run disruption from a predictable I10 FAIL.

## Scope
- Applies to: check_contract.py --stage final (B6), --stage paper (P5)
- Root list: B5 (this host) roots only
- Old-host roots: excluded (documented in B5_host_probe.md)
