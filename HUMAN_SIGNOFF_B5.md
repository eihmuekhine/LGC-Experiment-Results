# Human Signoff — B5

## Decision Point 2
- Date: 2026-10-18
- Authorized by: H
- Scope: B5 confirmatory run launch

## Verdict accepted
HOST-DIFFERENT — torch/cuDNN build divergence causes tiny floating-point
divergences in pix_l2; within-host reproducibility confirmed.

## Deviations acknowledged
1. Contract rev6 edit (commit 173e73a) — H authorized
2. prompt.txt update (rev5->rev6, OMP=8, sha256=2eb8698e3ac874196b518386c310702add8be63509689ceae2300ec89b905afb) — H authorized
3. nproc/cpu.max investigation (real=208, OMP=8 chosen) — H authorized
4. Results/ junk-dir cleanup (.ipynb_checkpoints) — H authorized
5. run_b5_parallel.sh edit (commit 1170dc1) — H authorized
6. run_b5_parallel.sh OMP=1->8 revert (commit 9cc75c7) — H authorized
7. B6/P5 stage policy (Option 1: B5-host roots only) — see reports/B6_policy.md
8. Premature tag attempt b5-go-2026-10-18 before signoff (deleted locally, never pushed)

## B6/P5 policy
Option 1 — B6 and P5 will run with B5 (this host) roots only.
See reports/B6_policy.md.

## Go/No-Go
- [x] Go — tag b5-go-2026-10-18
