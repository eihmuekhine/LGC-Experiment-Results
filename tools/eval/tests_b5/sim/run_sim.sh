#!/bin/bash
# CPU-only end-to-end simulation of the B5 launcher v2 (run_b5_parallel.sh), its STOP classification, its resume path
# and the monitor's STOP bundles.  No GPU and no attack: tests_b5/sim/stub/{python,nvidia-smi} fake 4 GPUs, the
# torch / torchvision / cuDNN version queries and every attack process (each "attack" sleeps and writes one row).
# Works on a copy of the launcher, b5_gate_cell.py, b5_monitor.py, the contract and env_build38.txt in a temporary
# directory; never touches Results/ or reports/ of the worktree.  Needs GNU parallel and python3 (no torch needed).
#
#   bash tools/eval/tests_b5/sim/run_sim.sh            # from the worktree root; prints PASS/FAIL per check
#
# Exit 0 = ALL PASS, 1 = at least one FAIL, 2 = cannot run (missing tool).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
REAL_PY=$(command -v python3 || command -v python)
command -v parallel >/dev/null 2>&1 || { echo "CANNOT RUN: GNU parallel not found"; exit 2; }
[ -n "$REAL_PY" ] || { echo "CANNOT RUN: python3 not found"; exit 2; }

# sha256 of the 52 attack command lines of `NGPU=10 bash run_b5_parallel.sh --dry_run` (lines starting "python ",
# queue order, RESULTS_ROOT=Results) at e8f20b9 -- the fix round must not change any attack command or the order.
DRY_SHA_E8F20B9=9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6

W=$(mktemp -d /tmp/b5sim.XXXXXX)
mkdir -p "$W/repo/tools/eval" "$W/repo/contract"
cp "$REPO/run_b5_parallel.sh" "$REPO/env_build38.txt" "$W/repo/"
cp "$REPO/tools/eval/b5_gate_cell.py" "$REPO/tools/eval/b5_monitor.py" "$W/repo/tools/eval/"
cp "$REPO/contract/paper_contract.yaml" "$W/repo/contract/"
export SIM_REAL_PY="$REAL_PY" SIM_NGPU=4 SIM_SLEEP=0.3 SIM_TRACE="$W/trace.txt"
export PATH="$HERE/stub:$PATH"
unset CUDA_VISIBLE_DEVICES
cd "$W/repo" || exit 2
echo "[sim] workdir $W (launcher sha256 $(sha256sum run_b5_parallel.sh | cut -c1-16)...)"

NP=0; NF=0
check() {  # check <name> <bash condition>
  if eval "$2"; then echo "PASS $1"; NP=$((NP + 1)); else echo "FAIL $1"; NF=$((NF + 1)); fi
}
bundle() { ls reports/stop/B5a2_"$1"_*.md 2>/dev/null | head -1; }

# ---------------------------------------------------------------- S0 dry run: attack commands unchanged
DRY=$(NGPU=10 bash run_b5_parallel.sh --dry_run 2>/dev/null | grep '^python ' | sha256sum | cut -d' ' -f1)
check "S0 dry-run: the 52 attack commands and their order equal e8f20b9" '[ "$DRY" = "$DRY_SHA_E8F20B9" ]'

# ---------------------------------------------------------------- S1 main run: 4 GPUs, three failing cells
export SIM_FAIL_SPEC="b5_main_ARS_OPT_vgg19_T:value,b5_main_CGBA_vgg19_NT:oom,b5_main_TtBA_vit_b_32_T:kill"
TMUX=sim NGPU=4 RESULTS_ROOT="$W/res" LOG_DIR="$W/log" bash run_b5_parallel.sh > "$W/launch.out" 2>&1
RC=$?
unset SIM_FAIL_SPEC
check "S1a launcher exit 1 when a job fails (got $RC)" '[ "$RC" -eq 1 ]'
JOBS=$(($(wc -l < "$W/log/parallel.log") - 1))
check "S1b joblog has 52 jobs (got $JOBS)" '[ "$JOBS" -eq 52 ]'
CONC=$("$REAL_PY" - "$W/trace.txt" <<'PY'
import collections, sys
ev = []
for line in open(sys.argv[1]):
    p = line.split()
    ev.append((float(p[1]), 0 if p[0] == "end" else 1, p[2].split("=")[1]))
ev.sort()
act, mx, used = collections.Counter(), collections.Counter(), set()
for _, start, g in ev:
    if start:
        act[g] += 1; used.add(g); mx[g] = max(mx[g], act[g])
    else:
        act[g] -= 1
print(f"{max(mx.values())} {len(used)}")
PY
)
check "S1c one attack process per GPU at any time, 4 GPUs used (max_concurrent gpus_used = $CONC)" '[ "$CONC" = "1 4" ]'
B_VAL=$(bundle main_ARS_OPT_vgg19_T); B_OOM=$(bundle main_CGBA_vgg19_NT); B_KILL=$(bundle main_TtBA_vit_b_32_T)
check "S1d a Python exception is E-OTHER, not E-INFRA" '[ -n "$B_VAL" ] && grep -q "code: E-OTHER" "$B_VAL"'
check "S1e \"CUDA out of memory\" is E-INFRA" '[ -n "$B_OOM" ] && grep -q "code: E-INFRA" "$B_OOM"'
check "S1f a killed attack (SIGKILL, rc 137) is E-INFRA" '[ -n "$B_KILL" ] && grep -q "code: E-INFRA" "$B_KILL"'
check "S1g STOP field 4 carries GPU name, torch and the contract sha256" \
  '[ -n "$B_VAL" ] && grep -q "GeForce RTX 5090" "$B_VAL" && grep -q "2.7.1+cu128" "$B_VAL" && grep -Eq "[0-9a-f]{64}" "$B_VAL"'
check "S1h STOP field 3 carries the attack log lines" '[ -n "$B_VAL" ] && grep -q "simulated adapter failure" "$B_VAL"'
check "S1i the joblog summary names the failed cells" \
  'sed -n "/^\[summary\]/,\$p" "$W/launch.out" | grep -q "main_ARS_OPT_vgg19_T"'

# ---------------------------------------------------------------- S2 fresh-directory rule
TMUX=sim NGPU=4 RESULTS_ROOT="$W/res" LOG_DIR="$W/log_again" bash run_b5_parallel.sh > "$W/fresh.out" 2>&1
RC=$?
check "S2 a second launch into existing output directories is refused with exit 2 (got $RC)" '[ "$RC" -eq 2 ]'

# ---------------------------------------------------------------- S3 resume keeps the main joblog
MAIN_SHA=$(sha256sum "$W/log/parallel.log" | cut -d' ' -f1)
TMUX=sim NGPU=4 RESULTS_ROOT="$W/res" LOG_DIR="$W/log" bash run_b5_parallel.sh --resume main_ARS_OPT_vgg19_T \
  > "$W/resume.out" 2>&1
RC=$?
check "S3a resume of a failed cell exits 0 (got $RC)" '[ "$RC" -eq 0 ]'
check "S3b resume leaves the main joblog parallel.log unchanged" \
  '[ "$(sha256sum "$W/log/parallel.log" | cut -d" " -f1)" = "$MAIN_SHA" ]'
check "S3c resume writes its own joblog parallel_resume_main_ARS_OPT_vgg19_T_<time>.log" \
  'ls "$W"/log/parallel_resume_main_ARS_OPT_vgg19_T_*.log >/dev/null 2>&1'

# ---------------------------------------------------------------- S4 resume refused while an attack process runs
bash -c 'exec -a "python adapters/run_cgba.py --sim_busy" sleep 60' &
BUSY=$!
sleep 1
TMUX=sim NGPU=4 RESULTS_ROOT="$W/res" LOG_DIR="$W/log" bash run_b5_parallel.sh --resume main_CGBA_vgg19_NT \
  > "$W/resume_busy.out" 2>&1
RC=$?
kill "$BUSY" 2>/dev/null; wait "$BUSY" 2>/dev/null
check "S4 resume is refused with exit 2 while another attack process runs (got $RC)" '[ "$RC" -eq 2 ]'

# ---------------------------------------------------------------- S5 resume refused on error rows
mkdir -p "$W/res/b5_main_TtBA_vgg19_T/targeted"
printf 'image,status,error\nimg9,error,boom\n' > "$W/res/b5_main_TtBA_vgg19_T/targeted/runs_TtBA.csv"
TMUX=sim NGPU=4 RESULTS_ROOT="$W/res" LOG_DIR="$W/log" bash run_b5_parallel.sh --resume main_TtBA_vgg19_T \
  > "$W/resume_err.out" 2>&1
RC=$?
check "S5 resume of a cell with an error row is refused with exit 1 (got $RC)" '[ "$RC" -eq 1 ]'

# ---------------------------------------------------------------- S6 gate STOP bundle (b5_gate_cell.py, real code)
mkdir -p "$W/glog"; echo "SIM-ATTACK-LOG-LINE 42" > "$W/glog/main_ARS_OPT_resnet50_NT.log"
mkdir -p "$W/res/b5_main_ARS_OPT_resnet50_NT/non_targeted"
CUDA_VISIBLE_DEVICES="" "$REAL_PY" tools/eval/b5_gate_cell.py main_ARS_OPT_resnet50_NT --results_root "$W/res" \
  --log_dir "$W/glog" > "$W/gate.out" 2>&1
RC=$?
GB=$(ls reports/stop/B5a2_main_ARS_OPT_resnet50_NT_*.md 2>/dev/null)
check "S6a a failing gate exits 1 and writes a STOP bundle (got $RC)" '[ "$RC" -eq 1 ] && [ -n "$GB" ]'
check "S6b gate STOP field 3 carries the last attack log lines" '[ -n "$GB" ] && grep -q "SIM-ATTACK-LOG-LINE 42" $GB'
check "S6c a gate tool that writes no output is E-OTHER, not E-INFRA" '[ -n "$GB" ] && ! grep -q "code: E-INFRA" $GB'
check "S6d two stopping gates of one cell are both kept (partial and domain), no bundle overwritten" \
  '[ -n "$GB" ] && grep -q "partial" $GB && grep -q "domain" $GB'

# ---------------------------------------------------------------- S7 monitor: one STOP bundle per condition
rm -f reports/stop/B5a2_main_ARS_OPT_resnet50_NT_*.md
printf 'image,status,error,nonfinite_queries,wall_time_s\nimg1,error,boom,0,1.0\n' \
  > "$W/res/b5_main_ARS_OPT_resnet50_NT/non_targeted/runs_ARS_OPT.csv"
"$REAL_PY" tools/eval/b5_monitor.py --once --results_root "$W/res" --log_dir "$W/mlog" > "$W/mon1.out" 2>&1
sleep $((61 - 10#$(date +%S)))       # next minute: a repeated bundle would get a new file name
"$REAL_PY" tools/eval/b5_monitor.py --once --results_root "$W/res" --log_dir "$W/mlog" > "$W/mon2.out" 2>&1
NB=$(ls reports/stop/B5a2_main_ARS_OPT_resnet50_NT_*.md 2>/dev/null | wc -l)
check "S7 two monitor passes over one persistent error row write one STOP bundle (got $NB)" '[ "$NB" -eq 1 ]'

echo
echo "[sim] $NP PASS, $NF FAIL (outputs kept in $W)"
[ "$NF" -eq 0 ] && { echo "ALL PASS"; exit 0; }
exit 1
