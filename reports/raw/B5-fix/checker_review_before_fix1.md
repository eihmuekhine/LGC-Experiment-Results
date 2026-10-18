# B5-fix checker review (independent)

Independent review session: no author work in this session; read order per `reports/raw/B5-fix/review_prompt.txt`
(sha256 `af5d1aa348efe03f52eb84b055d20c1989726de7b839af0a154c5363f2d4f895`). Author prompt
`reports/raw/B5-fix/prompt.txt` sha256 `8520cf03a8657bc81ab18c03f84ca6d807307c53607efd3bdda9fbda1dd61be8` (verified).
HEAD `bb7fa0de72134c2a5558c8af54a56d92c035ff23`, contract sha256
`ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c` (both verified).

Active LESSONS applying to this review: L04 (author summaries unreliable → raw outputs read first), L05 (new
reviewer session), L06 (archived prompts + sha256), L11 (check_contract only with --report/--status_out outside
reports/), L14 (spec checked against contract), L19 (no-GPU instance), L22/L23 (domain-gate context), L24/L25.

Review environment (this instance, at review time): NO GPU visible — `/usr/bin/nvidia-smi` is a 0-byte stub
(mtime 2024-01-08), `/dev/nvidia*` absent, `torch.cuda.device_count()=0`; `/sys/fs/cgroup/cpu.max` =
`50000 100000` (Q=0.5); torch 2.7.1+cu128 / tv 0.22.1+cu128 / cuDNN 90701. All commands CPU-only
(`CUDA_VISIBLE_DEVICES=""`); 0 GPU minutes; nothing written under `Results/`; no git checkout/reset/add/commit.

## Criteria

| id | verdict | evidence (rerun by reviewer unless noted) |
|---|---|---|
| C1 F1–F10 verdicts | PASS | All ten CONFIRMED at the cited lines of `git show HEAD:run_b5_parallel.sh` (sha256 `c20cc349…` verified): F1 :343 `g5_check.py --roots … --n 30` vs g5_check.py:156-159 (`--arch --mode --ttba_root` required, no `--roots`/`--n`); F2 :344-348 `if [ $? -ne 0 ] … exit 1` before the domain block :351-364; F3 :353 condition includes `*"LGC_RC_EGC"*`; F4 ttba_table1.json keys = {efficientnet_b0, inception_v3, resnet50, vgg19, vit_b_32} (E10/E11 victims absent); F5 :334/:343 gates once after the cell; F6 :79 `grep -q ",error,"` matches the TtBA header (Results/b4_smoke_ttba_resnet50/non_targeted/runs_TtBA.csv:1 contains `,wall_time_s,error,success_eps`; `grep -c ",error,"` = 1); F7 :92 `OUT="Results/b5_${CELL}"` vs contract E17 (paper_contract.yaml:347); F8 :331 `esac > …log 2>&1`, no rc capture (set -e does not propagate into the parallel template); F9 :81/:337/:346/:361 one-line stop files; F10 :68 `-j 5`, :70 `% 5`. |
| C2 byte-identity | PASS | Reviewer `--dry_run` (NGPU=1/8/10) reproduces 52 unique commands, 52 unique outdirs, slots < NGPU; my dry log is byte-identical to the author's `dry_run_ngpu1.log` except the `[threads]` line. `test_gates_monitor.py` [4]: old=52 new=52, exactly the 4 E17 cells differ, `--out_dir` only → `Results/appendix_cgba_d8only_<arch>` = contract E17 (paper_contract.yaml:347) = where run_cgba.py writes (`<out_dir>/<mode>/runs_CGBA_D8only.csv`; adapters/_common.py:688, adapters/run_cgba.py:211,228,261). |
| C3 G5 gate | PASS | b5_gate_cell.py:258-264 builds exactly g5_check.py's argparse CLI (:156-159); pairing = main TtBA cell of same arch+mode (:152-157); scope = main TtBA/CGBA/CGBA_H only (:239-240); rc0 without "VERDICT: PASS" → PENDING (:276-279); missing TtBA root → PENDING (:253-262); rc1 → FAIL/E-G5, rc2 → FLAG/E-OTHER, both stopping (:270-275, :341-350). Rerun: smoke CGBA+TtBA → "tool check only" → PENDING; E10 places365 → "G5 n/a (no published reference)"; missing root → PENDING. |
| C4 domain | PASS | count_x reads the column named `class` (b5_gate_cell.py:161-170); stop only for ARS_OPT/CGBA/CGBA_H (:293-299), LGC_RC_EGC and CGBA_D8only recorded only (D19 rev7). Rerun on reports/raw/B5/t0b_smoke.csv: X=4 of 1152 data rows, `class` column present. |
| C5 guards | PASS | NGPU required int ≥1 (run_b5_parallel.sh:31-34) and `nvidia-smi -L` match unless --dry_run (:116-121); GPU slot `( {%}-1 ) % NGPU` < NGPU for NGPU=1/8/10 (rerun); OMP = max(1, min(8, floor(Q/NGPU))) from cpu.max, "max" → `env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc` (:123-140; rerun prints Q=0.5 → 1; author's log Q=25.0 → 8); env guard reads ENV_FILE, requires torch/tv match + cuDNN 90701, exit 3 (rerun: wrong file → 3, real file passes); tmux refusal exit 2 (rerun); fresh-directory refusal exit 2 (rerun); --resume runs only that cell, error rows via csv module (b5_gate_cell.py:173-185); LPT queue with E17 last and every s/img equal to reports/B4e_preflight.md T5 (all 30 numbers checked; ARS_OPT E10/E11 ASSUMED, densenet ×1.6: 52.16=32.6×1.6, 47.2=29.5×1.6). |
| C6 exit status + §7 bundles | FAIL | Attack exit status IS checked (run_b5_parallel.sh:437-447: rc≠0 → STOP bundle, return 1, no gates) and both writers emit seven numbered fields. BUT the launcher's bundle field 4 is empty in real runs: `TORCH`/`CUDNN` (:38,:40) and `GPU_NAME`/`DRIVER` (:121-125) are not exported (:476 exports only NGPU RESULTS_ROOT LOG_DIR OMP MKL), and parallel workers do not inherit unexported vars — reviewer probe: `GPU_NAME=UNSET TORCH=UNSET DRIVER=UNSET`. So an attack-failure bundle (the main E-INFRA path) records `GPU=? driver=? torch= cuDNN=`. Also skill §7 field 4 wants "GPU name, driver, torch, backend flags, HEAD, contract sha256": neither writer includes the contract sha256; b5_gate_cell.py:189-202 (_env_block) also omits the driver; the launcher's bundle omits backend flags. Field 5 wants "rows already completed": both writers record only the cell name (run_b5_parallel.sh:143; b5_gate_cell.py:349). |
| C7 monitor | PASS | CPU-only for itself and every subprocess (b5_monitor.py:47, 50-53); read-only on RESULTS_ROOT — all writes under LOG_DIR (:143-147,:205,:318) or reports/stop; the only shutil.rmtree is under LOG_DIR (:154); no kill/signal anywhere; early-G5 copy = first 30 distinct images in runs_*.csv row order (:88-97, :134-147); PENDING cells re-evaluated from gates/summary.csv + state (:209-216, :276-303). Rerun: one pass over Results/b4e2_smoke_ttba_resnet50 → one status table, cell row present, rc=0, no STOP bundle (3 budget rows, 0 nonfinite). |
| C8 tests reproduce | PASS | All 16 rows of reports/B5_fix.md §7 reproduced: bash -n; dry NGPU=1/8/10 (52/52/52 cmds, unique outdirs, E17 H4 paths, slots<NGPU); fresh refusal (2); tmux refusal (2); NGPU mismatch (2, now "count=0" — see C9 env note); env guard wrong (3); env guard real (reaches fresh-dir 2, no guard failure); 3 G5 scenarios; domain counter 4/1152; monitor pass; byte-identity diff; `selftest_evalkit.py` ALL PASS 20/20. Reviewer outputs under /tmp/opencode/b5rev/ (author's evidence not overwritten). |
| C9 unsupported claims | 5 found | See next section. |

## Claims the evidence does not support (file:line, one-line fix)

1. reports/B5_fix.md:18-20 — "GPU: NVIDIA GeForce RTX 5090 D (nvidia-smi; 1 GPU) … 32607 MiB" is not
   reproducible on this instance at review time (`/usr/bin/nvidia-smi` is a 0-byte stub mtime 2024-01-08, no
   `/dev/nvidia*`, `torch.cuda.device_count()=0`); fix: re-record the environment (the instance appears
   reconfigured after 21:12) and note it in the report.
2. reports/B5_fix.md:22 — "cpu.max = 2500000 100000 -> Q = 25.0" is stale: current cpu.max is
   `50000 100000` → Q=0.5 (reviewer dry-run prints `Q=0.5 … =1`); fix: re-record Q at launch time, not from
   the B5-fix session.
3. reports/B5_fix.md:21 — "cuDNN 90701 (all match env_build38.txt:3)": env_build38.txt has no cuDNN entry
   (grep -i cudnn → nothing); the 90701 requirement comes from prompt T1c and is hard-coded in
   run_b5_parallel.sh:47-49; fix: cite the prompt/launcher line, not env_build38.txt:3.
4. reports/B5_fix.md:27 — L09 and L22 are listed as active; docs/LESSONS.md marks both "encoded" (L09 row
   :19, L22 row :32), and active rows L05/L07/L20 are not listed; fix: move L09/L22 to the encoded list and
   state why L05/L07/L20 do not apply.
5. reports/B5_fix.md:93 — "The instance has a GPU" is stale at review time (no GPU visible, see claim 1);
   in the current state the launcher's NGPU gate can never pass (`nvidia-smi -L | wc -l` = 0); fix: re-verify
   the instance before B5 attempt 2 and update the deviation note.

## Other problems found (file:line, one-line fix)

6. run_b5_parallel.sh:476 — add `GPU_NAME DRIVER TORCH CUDNN` to the export line so worker STOP bundles keep
   field 4 (root cause of the C6 FAIL).
7. tools/eval/b5_gate_cell.py:189-202 — _env_block lacks driver version and contract sha256 (skill §7 field
   4); fix: append both (contract sha256 is already computed elsewhere as `ce101e33…`).
8. run_b5_parallel.sh:142 / tools/eval/b5_gate_cell.py:349 — §7 field 5 lacks "rows already completed"; fix:
   count rows in the cell's runs_*.csv at bundle time and append to the scope line.
9. run_b5_parallel.sh:10 — header claims launcher exit 1 on attack/gate failure, but the script always exits 0
   after the summary (last command is the awk at :489); fix: `exit 1` if the joblog has nonzero Exitval, or
   reword the comment to "per-cell exit recorded in the joblog".
10. tools/eval/b5_monitor.py:343-350 — a persistent condition (error row, nonfinite, G5 FAIL) writes a new
    timestamped STOP bundle on every pass (≈144/day at the 10-min default); fix: dedup via the existing
    state.json (record fired bundle keys).

## Deviations from the review prompt

- Reviewer shell is not in tmux; the fresh-directory and NGPU-mismatch reruns set `TMUX=1` to pass the tmux
  gate (which precedes them); the tmux refusal itself was rerun with TMUX unset. The author ran inside tmux.
- The first combined guard-test command hit the tool's 120 s timeout (Q=0.5 cgroup makes each torch import
  ~30-60 s); rerun individually with a larger timeout; nothing left running.
- `bash tools/eval/tests_b5/run_tests.sh` was not rerun as-is (it overwrites the author's evidence under
  reports/raw/B5-fix/); each of its 9 checks was rerun individually with outputs under /tmp/opencode/b5rev/.
- check_contract.py was never run directly by the reviewer; the monitor test invoked it with
  `--report`/`--status_out` under a /tmp log dir (L11 respected).

## Verdict

C1–C5, C7, C8 PASS; C6 FAIL (STOP-bundle field 4 empty in parallel workers — one export line to fix — plus
missing contract sha256/driver/rows-completed per skill §7); 5 report claims unsupported or stale at review
time (all environment/ledger citations, not protocol violations). The 52 attack commands are byte-identical
to c20cc349 except the 4 E17 `--out_dir` values per H4; gates, monitor and refusals behave per contract
v1.9-rev7 (D13/D14/D19) in every rerun. Fix findings 6–9 and re-record the environment (findings 1, 2, 5)
before B5 attempt 2 launches; no GPU process was started by this review. STOP.
