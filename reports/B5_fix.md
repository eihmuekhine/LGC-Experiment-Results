# B5-fix — rebuild the B5 launcher for attempt 2 (one host, N GPUs)

## 1. Scope and prompt hash

- Phase: B5-fix. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, contract `v1.9-rev7`.
- Prompt sha256: `8520cf03a8657bc81ab18c03f84ca6d807307c53607efd3bdda9fbda1dd61be8` (reports/raw/B5-fix/prompt.txt).
- HEAD at review time: `bb7fa0de72134c2a5558c8af54a56d92c035ff23` (later replaced by the review-follow-up commit).
- Contract version line: `version: "1.9-rev7"` (contract/paper_contract.yaml:9).

Reference sha256 (printed at start):
- contract/paper_contract.yaml `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`
- run_b5_parallel.sh (c20cc349, committed) `c20cc34990e9ed2cf5fb1c8a18a79517d67617ba98884c13425461e3c21d17d2`
- reports/raw/B5/prompt.txt `1fb343b4d0c9803655e35762867882a8dcd61020eee2e7aa9ae41a1bc381ea94`
- reports/raw/B5-fix/prompt.txt `8520cf03a8657bc81ab18c03f84ca6d807307c53607efd3bdda9fbda1dd61be8`

## 2. Environment block

- GPU (measured during the B5-fix session, GPU on): `NVIDIA GeForce RTX 5090 D` (1 GPU), driver `595.71.05`, 32607 MiB.
  - The instance was later re-created/cloned (new container id `autodl-container-60914aaf20-99a77476`); the GPU is currently off; B5 attempt 2 re-verifies the environment at T0.
  - The GPU name string reports "RTX 5090 D" while env_build38.txt:2 records "NVIDIA GeForce RTX 5090"; driver 595.71.05 matches. No GPU name check was requested by this prompt (env guard = torch/tv/cuDNN only); recorded here for provenance.
- torch `2.7.1+cu128`, torchvision `0.22.1+cu128` (match env_build38.txt:3); cuDNN `90701` (requirement from prompt T1c, enforced at run_b5_parallel.sh:47-49; env_build38.txt records no cuDNN value).
- cgroup at B5-fix time `cpu.max = 2500000 100000` -> Q = 25.0; `nproc` (env -u OMP) = 208. Current (GPU off) cpu.max = `50000 100000` -> Q = 0.5; the launcher recomputes Q at launch.
- HEAD `bb7fa0de72134c2a5558c8af54a56d92c035ff23` at B5-fix time; contract sha256 above.

## 3. Active LESSONS ids (docs/LESSONS.md) that apply

active: L02, L03, L04, L05, L06, L07, L10, L11, L12, L14, L19, L20, L21, L23, L24, L25.
encoded (invariants kept as the reason for the code): L01, L08, L09, L13, L15, L16, L17, L18, L22.

## 4. F1–F10 verdicts (against c20cc349 = `git show bb7fa0d:run_b5_parallel.sh`; that was HEAD at review time)

| id | verdict | evidence |
|---|---|---|
| F1 | CONFIRMED | c20cc349:343 `g5_check.py --roots … --n 30`; g5_check.py:156-158 require `--arch --mode --ttba_root` (no `--roots`/`--n`) -> argparse exit 2. |
| F2 | CONFIRMED | c20cc349:344-348 `if [ $? -ne 0 ]; then … exit 1; fi` after the G5 call; the (always) nonzero exit skips the domain block at :351-364. |
| F3 | CONFIRMED | c20cc349:353 condition includes `*"LGC_RC_EGC"*`; D19 rev7 makes LGC_RC_EGC X report-only. |
| F4 | CONFIRMED | ttba_github/ttba_table1.json `table1` keys = {efficientnet_b0, inception_v3, resnet50, vgg19, vit_b_32}; E10/E11 victims absent; D13 rev7. |
| F5 | CONFIRMED | c20cc349:333-349 run partial + G5 once after the cell, not every 10 images / after the first 30. |
| F6 | CONFIRMED | c20cc349:79 `grep -q ",error,"`; TtBA runs header contains literal `,error,` (Results/b4_smoke_ttba_resnet50/non_targeted/runs_TtBA.csv:1 `…,wall_time_s,error,success_eps,…`). |
| F7 | CONFIRMED | c20cc349:92 `OUT="Results/b5_${CELL}"`; E17 writes `Results/b5_e17_CGBA_D8only_*` (c20cc349:312-327); contract E17 inputs `Results/appendix_cgba_d8only_<arch>/non_targeted` (contract/paper_contract.yaml:347). |
| F8 | CONFIRMED | c20cc349:331 `esac > $LOG_DIR/${CELL}.log 2>&1` with no `rc=$?` capture / §7 STOP bundle (only `set -e` at :3). |
| F9 | CONFIRMED | c20cc349:81,337,346,361 single-line `echo "STOP: …" > reports/stop/…` (not the seven §7 fields). |
| F10 | CONFIRMED | c20cc349:68 `parallel -j 5`, :70 `% 5`. |

## 5. Decisions in force (rev7 quotes)

- D13 rev7: G5 applies to the main-study (ImageNet: resnet50, vgg19, vit_b_32) TtBA / CGBA / CGBA_H cells only; no published reference for the E10/E11 victims (n/a); E17 outside G5; a CGBA/CGBA_H cell is checked against the TtBA cell of the same victim and mode on their common images.
- D14 rev7: "Solo" means no other attack process on the same GPU; the host-level concurrency slowdown is reported next to E13.
- D19 rev7: LGC_RC_EGC domain_consistency X is reported, not a stop rule; ARS_OPT / CGBA / CGBA_H X > 0 stops; CGBA_D8only (E17) is outside the checker roots, its X reported.

## 6. Changes and diff summary

- `run_b5_parallel.sh` rewritten (473 insertions / 351 deletions): NGPU required + `nvidia-smi -L` match, `parallel -j "$NGPU"`, GPU slot `( {%} - 1 ) % NGPU`; OMP/MKL threads from cgroup; torch/tv/cuDNN guard (exit 3); tmux refusal; LPT queue (E17 last) + fresh-directory rule + `--resume` (csv error-row check); attack exit-status checked -> §7 STOP bundle; gates delegated to `b5_gate_cell.py`; `--dry_run`; joblog summary. The 52 attack command lines are byte-identical to c20cc349 after variable expansion **except** the 4 E17 `--out_dir` values (H4). Full diff: reports/raw/B5-fix/commands_diff.txt.
- E17 output directory verified: `adapters/_common.py:688` `run_dir = os.path.join(args.out_dir, args.mode)` -> `run_cgba.py` writes `<out_dir>/non_targeted/runs_CGBA_D8only.csv`, matching contract E17 `Results/appendix_cgba_d8only_<arch>/non_targeted`.
- New `tools/eval/b5_gate_cell.py`: partial / G5 / domain gates, no short-circuit, one summary line per cell, §7 STOP bundles.
- New `tools/eval/b5_monitor.py`: CPU-only (CUDA_VISIBLE_DEVICES=""), read-only on RESULTS_ROOT; status/error/nonfinite/wall-time; partial at each +10 completed; early G5 (first 30 images) under monitor/g5early; PENDING re-evaluation; E-G5/E-OTHER/E-NAN STOP bundles.

## 7. Test results (CPU; evidence under reports/raw/B5-fix/)

| # | test | result |
|---|---|---|
| 1 | `bash -n run_b5_parallel.sh` | PASS |
| 2 | `--dry_run` NGPU=1: 52 unique cmds, 52 unique outdirs, E17 H4 paths, slots < NGPU | PASS |
| 3 | `--dry_run` NGPU=8 (same) | PASS |
| 4 | `--dry_run` NGPU=10 (same) | PASS |
| 5 | fresh-directory refusal (exit 2) | PASS |
| 6 | tmux refusal (TMUX unset, exit 2) | PASS |
| 7 | NGPU mismatch refusal (NGPU=8 vs 1, exit 2) | PASS |
| 8 | env guard refuses wrong torch/tv ENV_FILE (exit 3) | PASS |
| 9 | env guard passes the real env_build38.txt (reaches fresh-dir exit 2) | PASS |
| 10 | b5_gate_cell G5: CGBA+TtBA smoke -> "tool check only" -> PENDING (never PASS) | PASS |
| 11 | b5_gate_cell G5: E10 TtBA places365 -> "G5 n/a (no published reference)" | PASS |
| 12 | b5_gate_cell G5: missing TtBA root -> PENDING | PASS |
| 13 | domain counter on reports/raw/B5/t0b_smoke.csv = 4 X rows of 1152 | PASS |
| 14 | one b5_monitor.py pass over B4e2 smoke roots (status table + cell row) | PASS |
| 15 | 52-command byte-identity diff (only 4 E17 cells differ, `--out_dir` only) | PASS |
| 16 | `python tools/eval/selftest_evalkit.py` | ALL PASS 20/20 (no OOM; L19 not triggered on this instance) |

Total: 16/16 PASS (bash 9/9 + python 9/9, plus selftest 20/20).

Review follow-up (R's checker_review.md, sha256 `e3a6db81590b02eb435a9ec2817c6ae7f85b653c5261988f0f1f22b8101e2332`): C6 fixed — run_b5_parallel.sh now exports `GPU_NAME DRIVER TORCH CUDNN TF32_MM TF32_CUDNN CUDNN_DET CUDNN_BENCH CONTRACT_SHA` (parallel workers keep STOP field 4) and exits 1 on a nonzero joblog (Exitval or Signal); b5_gate_cell.py adds driver + contract sha256 to field 4 and rows-completed to field 5, with every external probe guarded (works on a no-GPU instance). Re-verified: bash -n OK, export probe OK, dry-run rc=0 52 cmds, test_gates_monitor.py ALL PASS 9/9.

## 8. Ordered queue (LPT, E17 last), hours and source

Full table in reports/raw/B5-fix/queue.txt (54 lines). Expected hours = n × s/img / 3600 from reports/B4e_preflight.md T5; ARS_OPT on E10/E11 = resnet50 value (densenet161 ×1.6), marked ASSUMED. Longest cell: e10_TtBA_densenet161_places365_NT (6.5722 h). E17 cells (n=30) last.

Predicted wall-clock (makespan ≈ max(longest cell, total/NGPU)):
- NGPU=1: ≈ 99.55 h
- NGPU=8: ≈ 12.44 h
- NGPU=10: ≈ 9.95 h

(total expected GPU-hours over the 52-cell queue = 99.55 h; longest single cell = 6.57 h.)

## 9. Deviations from the prompt

- The GPU was on during the B5-fix session (`RTX 5090 D`, driver 595.71.05); the prompt described the instance as a "no-GPU instance" (L19). The instance was re-created/cloned after the session (new container id `autodl-container-60914aaf20-99a77476`); the current GPU/CPU state is verified at B5 launch (T0). No GPU process was started by the B5-fix session (all tests CPU-only). selftest_evalkit.py completed without OOM, so the L19 OOM fallback was not needed.
- `b5_monitor.py` adds `--cell`/`--root`/`--ttba_root` test-only overrides (default behaviour unchanged: full 52-cell list, same RESULTS_ROOT/LOG_DIR defaults); used by T3 test 14.
- `b5_gate_cell.py` `--print_errors` returns exit 1 when error rows exist (used by the `--resume` check to refuse a resume with error rows, E-OTHER).

## 10. sha256 of changed / new files

Post-review (current) values:
- run_b5_parallel.sh `6c6117a99cbe34d1c76948ea5cae05999ecd425c5a9ee9efac790f57fac093ac`
- tools/eval/b5_gate_cell.py `1e879338dd2a0f825133cc3f5ce6dbb1e4f28ed0bb9466e53347385b7285e885`
- tools/eval/b5_monitor.py `f506daa6f90186855af8d99c7190333010f485bc85cfc7d8525c6b9cde0d87c1` (unchanged by review)
- tools/eval/tests_b5/test_gates_monitor.py `f895d27c3dfc3e699976296aa82c875e556a0a971e153273effb1b381515e04b` (unchanged)
- tools/eval/tests_b5/run_tests.sh `b3e38e0e1e6e15d0dcc90bcdc2400cbfb4542cfa56e3bcdfaa0e596fd3c92bbe` (unchanged)
- reports/raw/B5-fix/checker_review.md `e3a6db81590b02eb435a9ec2817c6ae7f85b653c5261988f0f1f22b8101e2332` (R review)

Historical B5-fix session values (superseded by the review follow-up; kept for provenance):
- run_b5_parallel.sh `197cd9eb17298aca5c36e7c141c1a70c6cd0f71b48c655c45434e138f65a3bae`
- tools/eval/b5_gate_cell.py `0c2e62a3ff3cc1c4820909f00c57b95599afc217ce33620223e1939670180bb8`

## 11. Five-line summary

1. Rebuilt the B5 launcher (`run_b5_parallel.sh`) for attempt 2, fixing all ten confirmed defects F1–F10 (wrong G5 CLI, skipped domain check, LGC_RC_EGC stop rule, missing E10/E11 G5 references, once-only gates, header-matching error grep, E17 outdir, unchecked attack exit, one-line stops, hard-coded `-j 5`/`% 5`).
2. The 52 attack command lines stay byte-identical to c20cc349 after variable expansion except the 4 E17 `--out_dir` values (H4 `Results/appendix_cgba_d8only_<arch>`), verified by an automated diff.
3. Gates are delegated to a new `tools/eval/b5_gate_cell.py` (partial/G5/domain, no short-circuit, §7 STOP bundles) and a new CPU-only, read-only `tools/eval/b5_monitor.py` (per-10 partial, early G5 at 30 images, PENDING re-evaluation).
4. All 16 acceptance tests PASS (dry-run ×3, four refusals, env guard, three G5 scenarios, domain counter, monitor pass, byte-identity diff) and `selftest_evalkit.py` = ALL PASS 20/20.
5. Post-review (R's checker_review.md): C6 fixed in the launcher (export + exit 1 on joblog failures) and in `b5_gate_cell` (driver/contract/rows + guarded probes). Predicted wall-clock ≈ 99.55 h (1 GPU), 12.44 h (8), 9.95 h (10). No GPU process was started by the B5-fix session. STOP.