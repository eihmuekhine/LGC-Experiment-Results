# Human checks and decisions: B5 host probe -> B5 -> paper (v1.0, 2026-10-15)

Who does what: H = you (the only one who decides, edits the contract, commits, tags); A = coder agent (DeepSeek V4 Pro);
R = `reviewer` agent in a NEW OpenCode session (GLM 5.3); L = Claude chat.
`D` rows are decisions (only H). `C` rows are checks H runs or opens personally; never tick a check from an AI summary.
Every command runs from the worktree root `/root/autodl-tmp/LGC_frozen` unless stated. Never paste YAML into bash; edit
YAML in an editor (nano, vim or the JupyterLab editor of AutoDL).

The six human decision points: (1) contract changes -> stage 3; (2) B5 GO / NO-GO -> stage 7; (3) every STOP except
E-INFRA, including G5 FAIL -> stages 4 and 9; (4) result-panel verdicts -> stage 10; (5) claim wording and limitations
-> stage 10; (6) final quality gate and submission -> stage 10.

New read-only tools (kit v1.4, copy to `tools/eval/`):
`compare_roots.py` (bitwise A-vs-B check of two result roots), `find_in_contract.py` (search the contract YAML without
knowing its layout), `human_preflight.py --require_host` (driver and hostname per root; FLAG on more than one value).

Install (once, H): the delta zip holds only new or changed files; it never touches `docs/LESSONS.md`, the skill or
the contract.
```bash
cd /root/autodl-tmp/LGC_frozen
sha256sum tools/eval/human_preflight.py      # expect 66cd6a9c...be27 (kit v1.3); if different, stop and tell L
unzip -o /path/to/lgc_preflight_kit_v1.4_delta.zip -d .
python tools/eval/compare_roots.py --help && python tools/eval/find_in_contract.py --help
git add tools/eval/compare_roots.py tools/eval/find_in_contract.py tools/eval/human_preflight.py \
        docs/HUMAN_CHECKS_B5.md docs/PREFLIGHT_RUNBOOK.md prompts/B5-host_prompt.txt
git commit -m "preflight kit v1.4 delta (human checks B5, compare_roots, find_in_contract, host columns)"
```

## Stage 0. Decisions already taken (record them; nothing to run)

| # | decision | status | where it is recorded |
|---|---|---|---|
| D0.1 | E11 adds ARS_OPT (same victim, modes and n as the other E11 methods) | YES, H 2026-10-15 | contract v1.9-rev4 (stage 3) |
| D0.2 | E10 CUT and E12 CUT: one true sentence each, in your words | open | amendment-2 notes (stage 6) |
| D0.3 | Hardware: one container, 8 x RTX 5090, one process per GPU | recommended | B5 prompt + sign-off 6.3 |

## Stage 1. Instance (H, before anything runs)

| # | check | command | PASS |
|---|---|---|---|
| C1.1 | GPUs | `nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv` | 8 rows; one name `NVIDIA GeForce RTX 5090`; one driver |
| D1.1 | driver differs from 595.71.05 (D16 rev1) | (decision) | either rent again, or note it as D16 rev2 in the contract; the host probe decides comparability either way |
| C1.2 | CPU, RAM, disk | `hostname; nproc; free -g; df -h / /root/autodl-tmp` | nproc >= 64 (8 processes x OMP 8); write the numbers down |
| C1.3 | code state | `git rev-parse HEAD; sha256sum contract/paper_contract.yaml` | same HEAD and contract sha256 as in your notes |
| C1.4 | thread setting | `echo $OMP_NUM_THREADS` | the login shell says 0; every agent command must `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` |

## Stage 2. Repository hygiene (H)

| # | check | command | PASS |
|---|---|---|---|
| C2.1 | untracked files | apply `.gitignore`, then `git status --short` | only files you intend to commit |
| C2.2 | secrets | block C2.2 below | first command prints nothing; second prints 0 |
| C2.3 | evidence binaries as a manifest | block C2.3 below | commit the manifest and the text reports, not the npz files |

```bash
# C2.2
git ls-files | grep -Ei "kaggle|\.env|key"
git log --all -p | grep -c "[REDACTED]
# C2.3
find generated/smoke_canary/kat_runs -type f -name '*.npz' -print0 | sort -z | xargs -0 sha256sum > reports/kat_runs_manifest.sha256
wc -l reports/kat_runs_manifest.sha256
```

## Stage 3. Contract v1.9-rev4 (decision point 1; only H edits)

1. Backup: `cp contract/paper_contract.yaml contract/paper_contract.yaml.bak_rev3`.
2. Open the file in an editor. In `paper_elements -> E11_robust_model -> inputs -> methods` add `ARS_OPT` at the end
   of the list. In the v1.8 layout that line reads
   `inputs: {victims: [RobustBench Salman2020Do_R50], modes: [non_targeted], n: 100, methods: [LGC_RC_EGC, TtBA, CGBA], ...}`;
   after the edit: `... methods: [LGC_RC_EGC, TtBA, CGBA, ARS_OPT], ...`. Change nothing else in that line.
3. Set `version:` to your rev4 string and add one changelog line, e.g.
   `"1.9-rev4 <date> user: E11 methods + ARS_OPT (newest baseline on the robust victim; same modes and n as the other E11 methods; oracle domain clamp01 per D19; ARS_OPT not in G5)"`.

| # | check | command | PASS |
|---|---|---|---|
| C3.1 | edit is what you meant | `python tools/eval/find_in_contract.py E11` | the methods list ends with `ARS_OPT`; `status` of E11 is `REQUIRED`; prints `YAML OK` |
| C3.2 | checker still works | `python tools/eval/tests_contract/test_contract.py` and `python tools/eval/tests_g5/test_g5.py` | all PASS |
| C3.3 | commit (H) | `git add contract/paper_contract.yaml && git commit -m "contract v1.9-rev4: E11 + ARS_OPT (user)"` | `git log --oneline -1` shows it |
| C3.4 | dump for L | `python tools/eval/find_in_contract.py "" > reports/contract_dump_rev4.txt` | paste the file to L |

## Stage 4. B5-host probe (A runs T0-T6; H checks; ~20-30 GPU-min)

Save the prompt with an editor as `reports/raw/B5-host/prompt.txt`; `sha256sum` it; start A in tmux.

| # | check | command / file | PASS |
|---|---|---|---|
| C4.1 | right agent | first line of the A session | `Coder · DeepSeek V4 Pro` |
| C4.2 | T1 determinism on this host | `python tools/eval/compare_roots.py --a Results/b5host_A --b Results/b5host_A2` | `VERDICT: IDENTICAL`, exit 0 |
| C4.3 | T1 observation-only checkpoints | `python tools/eval/compare_roots.py --a Results/b5host_A --b Results/b5host_B --budgets 500 1000 2000 2500 5000 10000 --no_npz` | IDENTICAL |
| C4.4 | T2/T3 cross-host | the verdict table in `reports/B5_host_probe.md`; re-run one line yourself: `compare_roots.py --a <reference EGC resnet50 NT root> --b Results/b5host_B` | your result equals the report's |
| C4.5 | T5 every GPU equals GPU 0 | block C4.5 below | 7 x IDENTICAL per method |
| C4.6 | T5 concurrency cost | s/img table (8 processes at once vs solo) | informational; if > 1.3 x solo, check nproc / `top` before B5 |
| C4.7 | T6 ARS_OPT on Salman | report T6: queries = 10000 per image, domain_consistency X = 0, oob_queries listed | all three hold |
| C4.8 | your own pre-flight | `python tools/eval/human_preflight.py --roots Results/b5host_* --out reports/human_preflight_b5host` | `FLAG lines: 0`; one driver, one host; open the grids |

```bash
# C4.5 (directory names as written in reports/B5_host_probe.md; adjust if A used other names)
for m in EGC TtBA; do for i in 1 2 3 4 5 6 7; do
  echo -n "$m gpu$i: "; python tools/eval/compare_roots.py --a Results/b5host_gpu0_$m --b Results/b5host_gpu${i}_$m | tail -1
done; done
```

| # | decision | rule |
|---|---|---|
| D4.1 | T1 FAIL (A != A2 or A != B) | NO-GO; STOP bundle to L (decision point 3) |
| D4.2 | HOST-IDENTICAL | smoke references transfer to this host |
| D4.3 | HOST-DIFFERENT | all B5 cells run on this one host; the smoke set stays pre-B5 code evidence; L writes it into the amendment-2 notes |

## Stage 5. Independent review (R, NEW session)

| # | check | how | PASS |
|---|---|---|---|
| C5.1 | right reviewer | first line of the session | `Reviewer · GLM 5.3` (not Coder, not the A session) |
| C5.2 | review content | `reports/raw/B5-host/checker_review.md` | every criterion PASS with a file:line or table quote |
| C5.3 | reviewer edited nothing else | `git status --short` | only the review file is new or changed |
| D5.1 | R and L disagree on an item | (decision) | you decide each item; write it in the sign-off "Open items" |

## Stage 6. Paste to L; L drafts; H reads

Paste: `reports/contract_dump_rev4.txt`, `reports/B5_host_probe.md`, the R review, the FLAG summary of
`reports/human_preflight_b5host.md`, the output of C1.1/C1.2, and your two sentences for D0.2.
L returns the amendment-2 notes, the evidence column of `reports/HUMAN_SIGNOFF_B5.md`, and the B5 prompt.

| # | check | how | PASS |
|---|---|---|---|
| C6.1 | amendment-2 notes are true | read every line; every reason is in your own words | you would defend each line to a reviewer |
| C6.2 | notes committed before any n = 200 run | `git log --oneline -1 -- reports/amendment2_deviations_*.md` | the commit is older than the tag in stage 7 |

## Stage 7. Sign-off and tag (decision point 2)

| # | check / decision | how | PASS |
|---|---|---|---|
| C7.1 | fill `reports/HUMAN_SIGNOFF_B5.md` yourself (S0-S9) | open each evidence file or run each command | every box has evidence; an empty box = NO-GO |
| D7.1 | GO / NO-GO | your signature in the file | GO only if C7.1 holds |
| C7.2 | tag | `git tag -a b5-go-<date> -m "B5 GO, contract v1.9-rev4 <sha256>"`; `git show b5-go-<date> --stat` | the tag points at the commit that holds the sign-off |

## Stage 8. Launch (H, first 15 minutes)

| # | check | command | PASS |
|---|---|---|---|
| C8.1 | B5 prompt saved and hashed | `sha256sum reports/raw/B5/prompt.txt` | hash written in your notes |
| C8.2 | runs inside tmux | `tmux new -s b5`; detach `Ctrl-b d`; re-attach `tmux attach -t b5` | the Mac can sleep; the run does not depend on your SSH session |
| C8.3 | one process per GPU | `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` | every GPU has memory used > 0; no GPU idle while jobs wait |
| C8.4 | each process sees one GPU, OMP 8 | block C8.4 below | one distinct GPU index per process, `OMP_NUM_THREADS=8` |
| C8.5 | E13 timing rows ran solo | the queue log lines for rows 1-10 | no other attack process during those rows |

```bash
# C8.4 (inside an AutoDL container nvidia-smi often cannot list PIDs, so read each process's environment instead)
for p in $(pgrep -f "python.*(adapters|run_)"); do
  echo "$p $(tr '\0' '\n' < /proc/$p/environ | grep -E '^(CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS)=' | tr '\n' ' ')"
done
```

## Stage 9. Monitoring during B5 (each check-in; ~8 h on 8 GPUs)

| # | check | command | PASS / action |
|---|---|---|---|
| C9.1 | progress | the queue status file named in the B5 prompt | moving; no cell stuck for > 2 x its expected time |
| C9.2 | error rows | `grep -l ",error," Results/<B5 roots>/*/runs_*.csv` | empty; any file -> STOP bundle; never resume that cell (decision point 3) |
| C9.3 | same host all the time | `python tools/eval/human_preflight.py --roots <B5 roots> --out reports/human_preflight_b5_<time> --require_host` | one driver, one host, `FLAG lines: 0`; open 2-3 new grids |
| C9.4 | partial checker (every 10 images) | the partial checker report named in the B5 prompt | exit 0 |
| C9.5 | G5 at n = 30 (TtBA, CGBA, CGBA_H) | the G5 report | CI intersects [0.5, 2]; FAIL -> STOP E-G5; no tuning; H decides with L (decision point 3) |
| C9.6 | oracle domain (D19 cells: ARS_OPT, CGBA, CGBA_H, CGBA_D8only) | `adapters/diag/domain_consistency.py` output per cell | X = 0 |
| C9.7 | disk | `df -h /root/autodl-tmp` | > 20 % free |
| C9.8 | do not | stop or restart the instance; `git checkout`; edit code; start a second process on a busy GPU | — |

## Stage 10. After B5 (decision points 4-6)

| # | check / decision | how | PASS |
|---|---|---|---|
| C10.1 | final checker | `python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage paper ...; echo $?` | exit 0 |
| C10.2 | final pre-flight | `human_preflight.py --roots <all B5 roots> --require_host` | FLAG 0 or each FLAG explained in writing |
| C10.3 | G5 at n = 200 | G5 report | as C9.5 |
| C10.4 | R review of B5 close (new session) | `reports/raw/B5/checker_review.md` | every criterion PASS |
| D10.1 | result panel (Skeptic, Methodologist, L): which claims the paper makes | panel files | you decide claim by claim (decision point 4) |
| D10.2 | claim wording and limitations | manuscript diff | your words (decision point 5) |
| C10.5 | every number in the manuscript traces to a generated file; every citation exists | `scan_numbers`, `verify_citations` | 0 untraced numbers; 0 unverified citations |
| D10.3 | final quality gate and submission | all of the above | decision point 6 |
