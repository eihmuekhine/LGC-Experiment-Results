# B5-pre2 no-GPU readiness — agent log

Prompt: reports/raw/B5pre2/nogpu_readiness_prompt.txt
Prompt sha256: 77982adcd89095481cc6ce4d61dfb6af2e3abd5b9c73eb73722336aac375547e

## Initial head check
- `git log --oneline -1` -> 56ba062 B5-fix: launcher v2 fix round 1+1b (R1-R7 + test_gates_monitor.py pin), reviewed PASS-WITH-NOTES

## Step 0 — facts
- `sha256sum contract/paper_contract.yaml` -> ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
- `opencode --version` -> 1.18.35
- `ls .opencode/skills` -> lgc-paper-contract, lgc-preflight
- `df -h /root/autodl-tmp` -> /dev/md0 50G, 3.5G used, 47G avail, 7%
- `CUDA_VISIBLE_DEVICES="" python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__)"` -> 2.8.0+cu128 0.23.0+cu128
- `git status --short` -> ?? reports/raw/B5-pre/, ?? reports/raw/B5pre2/

## Step 1 — CPU simulation harness
- `CUDA_VISIBLE_DEVICES="" bash tools/eval/tests_b5/sim/run_sim.sh` -> `CANNOT RUN: GNU parallel not found` (exit 2)
- `command -v parallel` -> empty; not found under /root or /usr/bin; `apt-get` available.
- BLOCKED: cannot produce the pass/fail table without GNU parallel. Awaiting user approval to install (rule 10). Not continuing to Step 3/4 until resolved.

## Unblock (user-approved)
- `apt-get install -y parallel` -> Setting up parallel (20210822+ds-2) ... OK
- `parallel --version | head -1` -> GNU parallel 20210822

## Step 1 — CPU simulation harness (re-run)
- `CUDA_VISIBLE_DEVICES="" bash tools/eval/tests_b5/sim/run_sim.sh` -> 21 PASS, 0 FAIL, ALL PASS (outputs in /tmp/b5sim.j9C8kC)
  - S0 dry-run PASS; S1a-S1i PASS; S2 PASS; S3a-S3c PASS; S4 PASS; S5 PASS; S6a-S6d PASS; S7 PASS

## Step 2 — OpenCode guard
- `.opencode/opencode.json` absent; guard located at /root/autodl-tmp/b5a2_kit_v2_1/tools/opencode_guard.py
- `CUDA_VISIBLE_DEVICES="" python .../opencode_guard.py --write` -> wrote global config /root/.config/opencode/opencode.jsonc
  (backup: /root/.config/opencode/opencode.jsonc.bak_20261019-151612), 33 deny entries incl. nvidia-smi, git push
- NOTE (separate, not executed here — appended deny block text):
```
"git push*": "deny", "rm -rf *": "deny", "nvidia-smi": "deny", "*git push*": "deny",
"*git commit*": "deny", "*git tag*": "deny", "*git add *": "deny", "*git rm *": "deny",
"*git mv *": "deny", "*git reset*": "deny", "*git checkout*": "deny", "*git switch*": "deny",
"*git restore*": "deny", "*git stash*": "deny", "*git rebase*": "deny", "*git merge*": "deny",
"*git pull*": "deny", "*git cherry-pick*": "deny", "*git revert*": "deny", "*git clean*": "deny",
"*git config*": "deny", "*git remote add*": "deny", "*git remote set-url*": "deny",
"*git remote remove*": "deny", "*git branch -d*": "deny", "*git branch -D*": "deny",
"*git branch -m*": "deny", "*git branch -f*": "deny", "*git update-ref*": "deny",
"*git filter-branch*": "deny", "kill *": "deny", "pkill *": "deny", "killall *": "deny"
```

## Step 3 — contract checker
- `CUDA_VISIBLE_DEVICES="" python tools/eval/tests_contract/test_contract.py | tail -3` -> FAILED to run:
  `ModuleNotFoundError: No module named 'pandas'` (python = /root/miniconda3/bin/python 3.12.3, base env)
- pandas not installed anywhere on the box; BLOCKED. Awaiting user approval to install (rule 10). Not continuing to Step 4 until resolved.

## Unblock 2 (user-approved)
- `find . -maxdepth 2 \( -iname 'requirements*.txt' -o -iname 'environment*.yml' \)` -> none found
- Third-party imports of the four eval files: numpy, pandas, yaml (all others stdlib). Already present: numpy 2.3.2, PyYAML 6.0.2. Missing: pandas.
- `pip install pandas` -> Successfully installed pandas-3.0.6

## Step 3 — contract checker (re-run)
- `CUDA_VISIBLE_DEVICES="" python tools/eval/tests_contract/test_contract.py | tail -3` -> ALL PASS (21/21)

## Step 4 — dry-run identity
- `for N in 1 5 8 10: NGPU=$N bash run_b5_parallel.sh --dry_run | grep '^python ' | sha256sum`
  - NGPU=1  -> 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
  - NGPU=5  -> 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
  - NGPU=8  -> 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6
  - NGPU=10 -> 9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6 (re-run alone; first batch attempt exceeded 120 s command timeout on CPU)
- MATCH all N.

## Step 5 — real 5-GPU queue (informational)
- `CUDA_VISIBLE_DEVICES="" NGPU=5 python tools/eval/b5_gate_cell.py --print_queue` -> 52-cell ordered queue printed;
  verbatim copy saved to reports/raw/B5pre2/nogpu_readiness_queue.txt (54 lines). Tool emits no aggregate
  predicted-hours line; per-cell hours in `expected h` column.

## Step 6 — pre-stage B5h2 prompt for N=5
- `N=5; sed "s/<N>/$N/g" /root/autodl-tmp/b5a2_kit_v2_1/prompts/B5h2_prompt.txt > reports/raw/B5pre2/B5h2_prompt_staged.txt`
- `grep -c '<N>' reports/raw/B5pre2/B5h2_prompt_staged.txt` -> 0
- `sha256sum reports/raw/B5pre2/B5h2_prompt_staged.txt` -> a7ef1b1ab23debffcf995e72026e11bfe29216cbb71fe93f569c1c1fc853c1f0
- opencode was NOT run on the staged prompt.

## Step 7 — version-pin re-verification (host, run outside this session)
Verbatim results as reported (pasted, not re-run here):
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
- `env_build38.txt` expects `torch 2.7.1+cu128 tv 0.22.1+cu128` -> env now MATCHES the build38 pin. (Step 0 recorded 2.8.0+cu128/0.23.0+cu128; Step 7 supersedes that with the corrected pin.)
- sim, contract and dry-run identity all re-verified PASS at the pinned versions.

### Launcher env-guard finding (run_b5_parallel.sh:44-57) — reported, not edited
Read of lines 40-70: `TORCH`/`TV`/`CUDNN` are probed at lines 45-47; `ENV_FILE` existence is checked (exit 3)
at line 48; expected pins `EXP_TORCH`/`EXP_TV` are parsed from `ENV_FILE` at lines 49-50; line 51 always logs
`[env] torch=... (expected ...)`.
The comparison is REAL, not log-only: lines 52-57 run only when `DRY_RUN -eq 0` and, on
`TORCH != EXP_TORCH || TV != EXP_TV || CUDNN != 90701`, print `environment guard FAILED` and `exit 3`.
Caveats: (a) in `--dry_run` mode the whole compare/refuse block is skipped, so there it is log-only;
(b) the cuDNN expectation is hardcoded `90701`, not read from `ENV_FILE`.
Conclusion: on a real launch it is refused on mismatch (exit 3); under `--dry_run` it only logs.

## Files written this run
- reports/raw/B5pre2/agent_log.md (this file)
- reports/raw/B5pre2/nogpu_readiness.md
- reports/raw/B5pre2/nogpu_readiness_queue.txt
- reports/raw/B5pre2/B5h2_prompt_staged.txt -> sha256 a7ef1b1ab23debffcf995e72026e11bfe29216cbb71fe93f569c1c1fc853c1f0



## Step 7b — interpreter / torch-file disambiguation (host, H run)
```
$ which python && python -c "import sys; print(sys.executable)"
/root/miniconda3/bin/python
/root/miniconda3/bin/python

$ CUDA_VISIBLE_DEVICES="" python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__, torch.__file__)"
2.7.1+cu128 0.22.1+cu128 /root/miniconda3/lib/python3.12/site-packages/torch/__init__.py
```
Interpretation: normal — python executable and torch.__file__ both under /root/miniconda3; live torch/torchvision match env_build38.txt pin. No interpreter mismatch. Step 0 reading of 2.8.0+cu128/0.23.0+cu128 is not explained by a PATH/interpreter change; env was changed between Step 0 and Step 7 by an unlogged install.

### pip show (Step 7b)
```
Name: torch
Version: 2.7.1+cu128
Location: /root/miniconda3/lib/python3.12/site-packages
Name: torchvision
Version: 0.22.1+cu128
Location: /root/miniconda3/lib/python3.12/site-packages
```

## Step 7c — guard refusal-path test (H-approved)
ENV_FILE=/tmp/env_wrong.txt NGPU=1 bash run_b5_parallel.sh -> [env] line + "environment guard FAILED"; EXIT_CODE=3
ENV_FILE=/tmp/env_wrong.txt NGPU=1 bash run_b5_parallel.sh --dry_run -> EXIT_CODE=0 (log-only, as documented)
Conclusion: guard refuses on real launch (exit 3); dry-run only logs. Matches the code read.

## Note — coder model change
Session 1140NN ran on Coder model "DeepSeek V4.1 Flash Free"; earlier workflow used "DeepSeek V4 Pro".
Recorded as a pipeline difference; no effect on results (all evidence is host-terminal output).

## Note — evidence provenance for Step 7
The Step 7 sim/contract/dry-run results were run by H on the host terminal and pasted verbatim into this log; they were NOT re-run inside the 1140NN coder session. Evidence rests on the host terminal output.
ENV_FILE=/tmp/env_wrong.txt (deleted) NGPU=1 bash run_b5_parallel.sh --dry_run -> [!] ENV_FILE not found (line 48 existence guard works).
