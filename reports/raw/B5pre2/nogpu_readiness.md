# B5-pre2 — no-GPU readiness (CPU only)

Prompt: `reports/raw/B5pre2/nogpu_readiness_prompt.txt`
Prompt sha256: `77982adcd89095481cc6ce4d61dfb6af2e3abd5b9c73eb73722336aac375547e`
Date: 2026-10-19 (UTC+8 box time), container has no GPU attached. No `nvidia-smi`, no `Results/` access, no attack code.

## Head check (before Step 0)
```
$ git log --oneline -1
56ba062 B5-fix: launcher v2 fix round 1+1b (R1-R7 + test_gates_monitor.py pin), reviewed PASS-WITH-NOTES
```
PASS (expected 56ba062 or descendant).

## Step 0 — facts
```
$ sha256sum contract/paper_contract.yaml
ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c  contract/paper_contract.yaml

$ opencode --version
1.18.35

$ ls .opencode/skills
lgc-paper-contract  lgc-preflight

$ df -h /root/autodl-tmp
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0         50G  3.5G   47G   7% /root/autodl-tmp

$ CUDA_VISIBLE_DEVICES="" python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__)"
2.8.0+cu128 0.23.0+cu128

$ git status --short
?? reports/raw/B5-pre/
?? reports/raw/B5pre2/
```
PASS.

## Step 1 — CPU simulation harness
First attempt: `CANNOT RUN: GNU parallel not found` (exit 2). GNU parallel was not installed.
With user approval: `apt-get install -y parallel` -> `Setting up parallel (20210822+ds-2) ...`;
`parallel --version | head -1` -> `GNU parallel 20210822`.

Re-run:
```
$ CUDA_VISIBLE_DEVICES="" bash tools/eval/tests_b5/sim/run_sim.sh
[sim] workdir /tmp/b5sim.j9C8kC (launcher sha256 50d8941c6201196d...)
PASS S0 dry-run: the 52 attack commands and their order equal e8f20b9
PASS S1a launcher exit 1 when a job fails (got 1)
PASS S1b joblog has 52 jobs (got 52)
PASS S1c one attack process per GPU at any time, 4 GPUs used (max_concurrent gpus_used = 1 4)
PASS S1d a Python exception is E-OTHER, not E-INFRA
PASS S1e "CUDA out of memory" is E-INFRA
PASS S1f a killed attack (SIGKILL, rc 137) is E-INFRA
PASS S1g STOP field 4 carries GPU name, torch and the contract sha256
PASS S1h STOP field 3 carries the attack log lines
PASS S1i the joblog summary names the failed cells
PASS S2 a second launch into existing output directories is refused with exit 2 (got 2)
PASS S3a resume of a failed cell exits 0 (got 0)
PASS S3b resume leaves the main joblog parallel.log unchanged
PASS S3c resume writes its own joblog parallel_resume_main_ARS_OPT_vgg19_T_<time>.log
PASS S4 resume is refused with exit 2 while another attack process runs (got 2)
PASS S5 resume of a cell with an error row is refused with exit 1 (got 1)
PASS S6a a failing gate exits 1 and writes a STOP bundle (got 1)
PASS S6b gate STOP field 3 carries the last attack log lines
PASS S6c a gate tool that writes no output is E-OTHER, not E-INFRA
PASS S6d two stopping gates of one cell are both kept (partial and domain), no bundle overwritten
PASS S7 two monitor passes over one persistent error row write one STOP bundle (got 1)

[sim] 21 PASS, 0 FAIL (outputs kept in /tmp/b5sim.j9C8kC)
ALL PASS
```
PASS 21/21.

## Step 2 — OpenCode guard
`.opencode/opencode.json` did not exist. Guard found at `/root/autodl-tmp/b5a2_kit_v2_1/tools/opencode_guard.py`.
```
$ CUDA_VISIBLE_DEVICES="" python /root/autodl-tmp/b5a2_kit_v2_1/tools/opencode_guard.py --write
config: /root/.config/opencode/opencode.jsonc (exists)
  global: deny rules appended to bash
backup: /root/.config/opencode/opencode.jsonc.bak_20261019-151612
written: /root/.config/opencode/opencode.jsonc
```
Note (separate, not executed here): the appended deny block is present in
`/root/.config/opencode/opencode.jsonc` (33 deny entries; verified with grep: `nvidia-smi`, `git push*`,
`rm -rf *`, `*git commit*`, `*git tag*`, `*git add *`, `*git rm *`, `*git mv *`, `*git reset*`, `*git checkout*`,
`*git switch*`, `*git restore*`, `*git stash*`, `*git rebase*`, `*git merge*`, `*git pull*`, `*git cherry-pick*`,
`*git revert*`, `*git clean*`, `*git config*`, `*git remote add*`, `*git remote set-url*`, `*git remote remove*`,
`*git branch -d*`, `*git branch -D*`, `*git branch -m*`, `*git branch -f*`, `*git update-ref*`,
`*git filter-branch*`, `kill *`, `pkill *`, `killall *`, plus `"edit": "ask"`).
PASS.

## Step 3 — contract checker
First attempt: `ModuleNotFoundError: No module named 'pandas'` — pandas absent from the base conda env
(python 3.12.3, /root/miniconda3/bin/python). With user approval: checked for pinned package lists
(`find . -maxdepth 2 -iname 'requirements*.txt' -o -iname 'environment*.yml'` -> none), listed third-party
imports of the four eval files (numpy, pandas, yaml — all present except pandas), installed exactly the missing
module: `pip install pandas` -> `Successfully installed pandas-3.0.6` (`pandas 3.0.6`).

Re-run:
```
$ CUDA_VISIBLE_DEVICES="" python tools/eval/tests_contract/test_contract.py | tail -3
   PASS  T18 --status_out honored

ALL PASS  (21/21)
```
PASS.

## Step 4 — dry-run identity (N in 1 5 8 10)
```
$ for N in 1 5 8 10; do NGPU=$N bash run_b5_parallel.sh --dry_run | grep '^python ' | sha256sum; done
NGPU=1:  9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
NGPU=5:  9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
NGPU=8:  9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
NGPU=10: 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6   (ran alone; took >120 s on CPU)
```
All four equal `9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6`. PASS (MATCH all N).

## Step 5 — real 5-GPU queue (informational, prints only)
`CUDA_VISIBLE_DEVICES="" NGPU=5 python tools/eval/b5_gate_cell.py --print_queue` printed the full ordered
52-cell queue (cells 1..52, columns: #, cell, scope, method, arch, mode, n, s/img, source, expected h, outdir),
from `e10_TtBA_densenet161_places365_NT` (6.5722 h) down to `e17_CGBA_D8only_Salman2020Do_R50_NT` (0.1817 h).
See `reports/raw/B5pre2/nogpu_readiness_queue.txt` for the verbatim table. The tool emits no aggregate
"predicted-hours" summary line; per-cell expected hours are in the table's `expected h` column.
PASS (informational).

## Step 6 — pre-stage B5h2 prompt for N=5
```
$ N=5; sed "s/<N>/$N/g" /root/autodl-tmp/b5a2_kit_v2_1/prompts/B5h2_prompt.txt > reports/raw/B5pre2/B5h2_prompt_staged.txt
$ grep -c '<N>' reports/raw/B5pre2/B5h2_prompt_staged.txt
0
$ sha256sum reports/raw/B5pre2/B5h2_prompt_staged.txt
a7ef1b1ab23debffcf995e72026e11bfe29216cbb71fe93f569c1c1fc853c1f0  reports/raw/B5pre2/B5h2_prompt_staged.txt
```
PASS (0 remaining `<N>`; opencode was NOT run on the staged prompt).

## Step 7 — version-pin re-verification (host, run outside this session)
Verbatim results as reported (pasted, not re-run):
```
$ pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
Requirement already satisfied: torch==2.7.1+cu128 ... (2.7.1+cu128)
Requirement already satisfied: torchvision==0.22.1+cu128 ... (0.22.1+cu128)

$ CUDA_VISIBLE_DEVICES="" python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__)"
2.7.1+cu128 0.22.1+cu128

$ grep -n "__version__\|env_build38" run_b5_parallel.sh
31:ENV_FILE="${ENV_FILE:-env_build38.txt}"
45:TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "NONE")
46:TV=$(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo "NONE")

$ bash tools/eval/tests_b5/sim/run_sim.sh -> 21 PASS, 0 FAIL, ALL PASS
$ python tools/eval/tests_contract/test_contract.py | tail -3 -> ALL PASS (21/21)
$ for N in 1 5 8 10: NGPU=$N dry-run sha256 -> all four 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
```
`env_build38.txt` expects `torch 2.7.1+cu128 tv 0.22.1+cu128` -> env MATCHES the build38 pin (the Step 0
reading of 2.8.0+cu128/0.23.0+cu128 is superseded by the corrected pin). sim, contract and dry-run identity
re-verified PASS. PASS.

### Step 7 companion — launcher env guard (run_b5_parallel.sh:44-57; reported, not edited)
Probe `TORCH`/`TV`/`CUDNN` at lines 45-47; `ENV_FILE` existence (exit 3) at line 48; `EXP_TORCH`/`EXP_TV`
parsed from `ENV_FILE` at lines 49-50; line 51 always logs. The match is a REAL refusal, not log-only:
the compare/refuse block (lines 52-57) runs when `DRY_RUN -eq 0` and exits 3 on
`TORCH != EXP_TORCH || TV != EXP_TV || CUDNN != 90701`. Caveats: under `--dry_run` it is log-only; the
cuDNN expectation `90701` is hardcoded rather than read from `ENV_FILE`.

## Files written this run (sha256)
- `reports/raw/B5pre2/nogpu_readiness.md` -> <this file>
- `reports/raw/B5pre2/agent_log.md` -> see log (sha256 recorded in log)
- `reports/raw/B5pre2/B5h2_prompt_staged.txt` -> `a7ef1b1ab23debffcf995e72026e11bfe29216cbb71fe93f569c1c1fc853c1f0`
- `reports/raw/B5pre2/nogpu_readiness_queue.txt` -> queue dump from Step 5

## Deviations
1. Step 1 first run could not start: GNU parallel missing -> installed with user approval (`apt-get install -y
   parallel`, GNU parallel 20210822), then re-ran. Final table: 21/21 PASS.
2. Step 3 first run failed: pandas missing -> installed with user approval (`pip install pandas` -> 3.0.6),
   then re-ran. Final: ALL PASS (21/21).
3. Step 4 NGPU=10 dry-run exceeded the 120 s command timeout on CPU when batched with N=1/5/8; re-run alone
   with a longer timeout -> hash matches.
4. Step 5 output has no aggregate predicted-hours line (tool prints per-cell expected hours only).

## RESULT
phase: B5-pre2 no-GPU readiness
head: 56ba062
sim_harness: PASS 21/21
contract_check: ALL PASS 21/21
dry_run_identity: MATCH all N
env_pin: "matched (2.7.1+cu128/0.22.1+cu128), re-verified sim/contract/dry-run all PASS"
queue_ngpu5: see report (per-cell expected h; no aggregate line emitted by the tool)
guard: applied
b5h2_staged: a7ef1b1ab23debffcf995e72026e11bfe29216cbb71fe93f569c1c1fc853c1f0
deviations: 4 (listed above; none change results)
stop_bundles: none
verdict: READY-FOR-GPU
STOP.

### Step 7b/7c addendum (host, H run)
Interpreter check: python and torch.__file__ both under /root/miniconda3 (normal).
pip show Location matches torch.__file__ parent -> same environment.
Guard refusal test: ENV_FILE=/tmp/env_wrong.txt NGPU=1 bash run_b5_parallel.sh -> environment guard FAILED; EXIT_CODE=3.
Dry-run test: ENV_FILE=/tmp/env_wrong.txt NGPU=1 bash run_b5_parallel.sh --dry_run -> EXIT_CODE=0 (log-only, guard does not refuse).
ENV_FILE missing test: bash run_b5_parallel.sh --dry_run -> "[!] ENV_FILE not found" (line 48 existence guard works).
