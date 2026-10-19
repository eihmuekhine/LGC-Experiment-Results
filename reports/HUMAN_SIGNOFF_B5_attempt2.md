# Human sign-off — B5 attempt 2 (decision point 2)

Fill every value yourself from the file or command named; never from an AI summary. An empty row means NO-GO.

Values below are taken from reports/raw/B5a2/signoff_values.txt (sha256 2d773e211bb2db199bb5264ae7a91a24ba55e3339e4f0f79305fe52f678e6474).

| # | item | evidence (file / command) | value | ok |
|---|---|---|---|---|
| S1 | contract v1.9-rev7 committed | `sha256sum contract/paper_contract.yaml` | ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c | ok |
| S2 | attempt-1 note committed | `git log --oneline -1 -- reports/b5_attempt1/ABANDONED.md` | bb7fa0d contract v1.9-rev7 (D13/D14/D16/D19 notes, H edit); B5 attempt-1 note; LESSONS L24-L25 | ok |
| S3 | launcher v2 reviewed after fix round 1 | reports/raw/B5-fix/fix1_fix1b_review.md:122 REVIEW_VERDICT; `bash tools/eval/tests_b5/sim/run_sim.sh \| tail -1` | sha256 run_b5_parallel.sh: 50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb; REVIEW_VERDICT: PASS-WITH-NOTES; sim: 21 PASS, 0 FAIL, ALL PASS | ok |
| S4 | host probe | reports/B5h2_host_probe.md verdict; reports/raw/B5h2/checker_review.md verdict | B5h2 (1 GPU): HOST-IDENTICAL, reviewer VERDICT: PASS-WITH-NOTES (checker_review.md:60). B5h2n5 (5 GPU, commit 60cc82b): HOST-IDENTICAL, no independent reviewer | ok |
| S5 | software stack on this host | probe T0 / python import | torch 2.7.1+cu128 / tv 0.22.1+cu128 / cuDNN 90701 | ok |
| S6 | GPUs | `nvidia-smi -L`; driver | NGPU = 5; name: NVIDIA GeForce RTX 5090; driver: 595.71.05 | ok |
| S7 | disk | `df -h /root/autodl-tmp` | free: 47G (50G total, 8% used) | ok |
| S8 | fresh output directories, no probe STOP files under the B5 name | `ls -d Results/b5_* Results/appendix_cgba_d8only_* reports/stop/B5a2_*` | prints nothing (count 0) | ok |
| S9 | dry run inside tmux | `NGPU=5 bash run_b5_parallel.sh --dry_run` | guard PASS ([env] = expected); 52 cells; slots 1..5 of 5 (gpu_slot 0..4 < NGPU=5); sum expected h = 99.5467; run inside tmux ($TMUX set) | ok |
| S10 | prompts archived | `sha256sum reports/raw/B5-fix/prompt.txt reports/raw/B5h2/prompt.txt reports/raw/B5a2/prompt.txt` | B5-fix 8520cf03a8657bc81ab18c03f84ca6d807307c53607efd3bdda9fbda1dd61be8; B5h2 0be547fb96096f968d817ca6af47c7f9d3c42520c6edc3a4f1cd9a7c6def2395; B5a2 8d7671fe3bd17298c61862a6109849cdcca15c63b8b9768ba7119c60247cf018 | ok |
| S11 | worktree clean | `git status --short \| grep -v '^??'` | prints nothing | ok |
| S12 | open items | see list | 1) driver 595.71.05 differs from reference-root host 580.105.08; B5h2n5 T2/T3/T5 bitwise IDENTICAL, so no effect. 2) B5h2n5 had no independent reviewer. 3) sitecustomize.py shim outside the repo, sha256 d0161cad705ad8b7cbdd82650fda548452065f099c04e4759617126142178458, H-approved. 4) .ipynb_checkpoints dirs moved to /root/autodl-tmp/_junk_ipynb. 5) stale 2026-10-18 no-GPU launcher logs moved to /root/autodl-tmp/_stale_B5a2_20261018. 6) 7 ARS_OPT cells (e10/e11) have ASSUMED s/img, so ETA is uncertain. 7) G5 stays PENDING until 30 common images (never PASS by default). 8) kit is at /root/autodl-tmp/b5a2_kit_v2_1 (SHA256SUMS 14/14 OK). Expected 99.55 GPU-h, about 20-22 h wall on 5 GPUs. | ok |

Decision: GO — H, 2026-10-20 22:35

After committing this file: tag `b5a2-go-<YYYY-MM-DD>` on that commit, push the tag, never move it.