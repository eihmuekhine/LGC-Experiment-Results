#!/bin/bash
# E9 ablation launcher (prereg_build38_amendment3.txt).  One host, NGPU GPUs, PER_GPU attack processes per GPU.
# L, 2026-10-23.  Built from run_b5_parallel.sh (B5 attempt 2): same environment guard, tmux rule, fresh-directory
# rule, GNU parallel slots, STOP bundles.  The 18 attack command lines are the B5 LGC_RC_EGC command lines with
# only --methods and --out_dir changed.  Run from the repository root /root/autodl-tmp/LGC_frozen.
#
#   NGPU=5 bash run_e9_parallel.sh --dry_run      print the 18 commands and the queue; start nothing
#   NGPU=5 bash run_e9_parallel.sh                inside tmux: run the queue
#   NGPU=5 bash run_e9_parallel.sh --resume CELL  resume one interrupted cell (finished images are skipped)
#
# Exit: 0 all cells ran and passed the gate; 1 a cell failed (STOP bundle under reports/stop/); 2 usage / NGPU /
#       tmux / fresh-directory refusal; 3 environment-guard or provenance refusal.
set -euo pipefail

DRY_RUN=0
RESUME_CELL=""
usage() { echo "usage: NGPU=<n> [PER_GPU=1] $0 [--dry_run] [--resume <cell>]"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry_run) DRY_RUN=1 ;;
    --resume) [ $# -ge 2 ] || { echo "[!] --resume needs a cell name" >&2; exit 2; }; RESUME_CELL="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[!] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

RESULTS_ROOT="${RESULTS_ROOT:-Results}"
LOG_DIR="${LOG_DIR:-reports/raw/E9}"
ENV_FILE="${ENV_FILE:-env_build38.txt}"
PER_GPU="${PER_GPU:-1}"
ENGINE_SHA_PREFIX="835f1933d9e5"
RUNNER_SHA="9c62820e5fafbac7e8c3f2cf3a688e3e6e7c53f3c5fc7f709ef5f34d17861873"   # amendment 2, item 8

if [ -z "${NGPU:-}" ]; then echo "[!] NGPU is required (integer >= 1)" >&2; exit 2; fi
case "$NGPU" in ''|*[!0-9]*) echo "[!] NGPU must be an integer >= 1" >&2; exit 2;; esac
case "$PER_GPU" in ''|*[!0-9]*) echo "[!] PER_GPU must be an integer >= 1" >&2; exit 2;; esac
if [ "$NGPU" -lt 1 ] || [ "$PER_GPU" -lt 1 ]; then echo "[!] NGPU and PER_GPU must be >= 1" >&2; exit 2; fi
JOBS=$(( NGPU * PER_GPU ))

# ---------------------------------------------------------------------------------------------- provenance guard
[ -f prereg_build38_amendment3.txt ] || { echo "[!] prereg_build38_amendment3.txt not found (wrong directory?)" >&2; exit 3; }
if [ "$DRY_RUN" -eq 0 ]; then
  TAGGED=$(git tag --points-at HEAD 2>/dev/null | grep -c '^build38-amend3$' || true)
  if [ "$TAGGED" -ne 1 ] && ! git merge-base --is-ancestor build38-amend3 HEAD 2>/dev/null; then
    echo "[!] tag build38-amend3 is not HEAD or an ancestor of HEAD: register the amendment first (README step 2)" >&2
    exit 3
  fi
  if ! git diff --quiet HEAD -- prereg_build38_amendment3.txt tools/eval/claims_e9.py tools/eval/e9_gate.py \
       run_e9_parallel.sh lgc_bench/engine.py run_benchmark_33b.py; then
    echo "[!] registered files differ from HEAD (git diff): refusing" >&2
    exit 3
  fi
fi
ENGINE_SHA=$(sha256sum lgc_bench/engine.py | awk '{print $1}')
case "$ENGINE_SHA" in "$ENGINE_SHA_PREFIX"*) ;; *) echo "[!] engine sha256 $ENGINE_SHA is not the frozen engine" >&2; exit 3;; esac
GOT_RUNNER=$(sha256sum run_benchmark_33b.py | awk '{print $1}')
[ "$GOT_RUNNER" = "$RUNNER_SHA" ] || { echo "[!] run_benchmark_33b.py sha256 $GOT_RUNNER != $RUNNER_SHA" >&2; exit 3; }
[ -z "${ADV_STORE_BUDGETS:-}" ] || { echo "[!] ADV_STORE_BUDGETS is set: unset it" >&2; exit 3; }
if ! python tools/eval/e9_gate.py --check_inputs; then
  if [ "$DRY_RUN" -eq 0 ]; then echo "[!] inputs missing (lines [inputs] FAIL above): refusing" >&2; exit 3; fi
  echo "[dry_run] inputs incomplete: a real launch would refuse"
fi

# ---------------------------------------------------------------------------------------------- env guard
TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "NONE")
TV=$(python -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo "NONE")
CUDNN=$(python -c 'import torch; print(torch.backends.cudnn.version())' 2>/dev/null || echo "NONE")
[ -f "$ENV_FILE" ] || { echo "[!] ENV_FILE not found: $ENV_FILE" >&2; exit 3; }
EXP_TORCH=$(grep -oE 'torch [0-9][^ ]*' "$ENV_FILE" | head -1 | awk '{print $2}')
EXP_TV=$(grep -oE 'tv [0-9][^ ]*' "$ENV_FILE" | head -1 | awk '{print $2}')
echo "[env] torch=$TORCH (expected $EXP_TORCH)  torchvision=$TV (expected $EXP_TV)  cuDNN=$CUDNN (expected 90701)"
if [ "$DRY_RUN" -eq 0 ]; then
  if [ "$TORCH" != "$EXP_TORCH" ] || [ "$TV" != "$EXP_TV" ] || [ "$CUDNN" != "90701" ]; then
    echo "[!] environment guard FAILED: torch/torchvision/cuDNN do not match $ENV_FILE" >&2
    exit 3
  fi
  if [ -z "${TMUX:-}" ]; then echo "[!] not inside tmux; refusing to start" >&2; exit 2; fi
  NGPU_ACTUAL=$( (nvidia-smi -L 2>/dev/null || true) | wc -l)
  if [ "$NGPU" -ne "$NGPU_ACTUAL" ]; then echo "[!] NGPU=$NGPU != nvidia-smi -L count=$NGPU_ACTUAL" >&2; exit 2; fi
fi

# ---------------------------------------------------------------------------------------------- queue
ORDER_FILE=$(mktemp)
CELLS_FILE=$(mktemp)
trap 'rm -f "$ORDER_FILE" "$CELLS_FILE"' EXIT
python tools/eval/e9_gate.py --print_order --results_root "$RESULTS_ROOT" > "$ORDER_FILE"
if [ -n "$RESUME_CELL" ]; then
  RESUME_OUTDIR=$(awk -F'\t' -v c="$RESUME_CELL" '$1==c{print $2}' "$ORDER_FILE")
  [ -n "$RESUME_OUTDIR" ] || { echo "[!] unknown cell: $RESUME_CELL" >&2; exit 2; }
  if ps -eo args= 2>/dev/null | grep -Eq '^python[[:space:]]+run_benchmark_33b\.py'; then
    echo "[!] an attack process is already running; refusing --resume" >&2; exit 2
  fi
  if ! python tools/eval/e9_gate.py --print_errors --root "$RESUME_OUTDIR"; then
    echo "[!] error rows under $RESUME_OUTDIR; refusing resume (needs investigation)" >&2; exit 1
  fi
  printf '%s\t%s\n' "$RESUME_CELL" "$RESUME_OUTDIR" > "$CELLS_FILE"
else
  if [ "$DRY_RUN" -eq 0 ]; then
    while IFS=$'\t' read -r C O; do
      [ -n "$C" ] || continue
      if [ -e "$O" ]; then echo "[!] $O exists (cell $C); use --resume $C" >&2; exit 2; fi
    done < "$ORDER_FILE"
  fi
  cp "$ORDER_FILE" "$CELLS_FILE"
fi

# ---------------------------------------------------------------------------------------------- threads
read -r THREADS Q < <(python - "$JOBS" <<'PY'
import os, subprocess, sys
jobs = int(sys.argv[1])
cg = "/sys/fs/cgroup/cpu.max"
q = None
if os.path.exists(cg):
    parts = open(cg).read().split()
    if parts and parts[0] != "max":
        q = int(parts[0]) / int(parts[1])
if q is None:
    q = float(subprocess.run(["env", "-u", "OMP_NUM_THREADS", "-u", "OMP_THREAD_LIMIT", "nproc"],
                             capture_output=True, text=True).stdout.strip())
print(max(1, min(8, int(q // jobs))), q)
PY
)
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "[threads] Q=$Q JOBS=$JOBS (NGPU=$NGPU x PER_GPU=$PER_GPU) -> OMP_NUM_THREADS=$THREADS"

# ---------------------------------------------------------------------------------------------- attack
run_attack() {
  local CELL="$1" OUT="$2" METHOD ARCH M MODE
  METHOD=$(printf '%s' "$CELL" | sed -E 's/^e9_(.*)_(vgg19|resnet50|vit_b_32)_(NT|T)$/\1/')
  ARCH=$(printf '%s' "$CELL" | sed -E 's/^e9_.*_(vgg19|resnet50|vit_b_32)_(NT|T)$/\1/')
  M=$(printf '%s' "$CELL" | sed -E 's/^e9_.*_(NT|T)$/\1/')
  case "$METHOD" in LGC_RC|LGC_RC_EGCI|LGC_RC_HT1000) ;; *) echo "[!] bad method in $CELL" >&2; return 2;; esac
  case "$M" in NT) MODE=non_targeted;; T) MODE=targeted;; *) echo "[!] bad mode in $CELL" >&2; return 2;; esac
  python run_benchmark_33b.py --dataset imagenet --mode "$MODE" --archs "$ARCH" \
    --methods "$METHOD" --data_dir /root/autodl-tmp/ttba_mini_flat --labels_file ttba_mini_val_labels.txt \
    --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
    --victim_ckpt_dir /root/autodl-tmp/TtBA/model --num_samples 200 --max_queries 10000 \
    --checkpoints 100 200 500 1000 2000 2500 5000 10000 --seed 42 --exclude_seeds 992 2027 3141 2718 \
    --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv
}

stop_bundle() {
  local CELL="$1" RC="$2" LOG="$3" P
  mkdir -p reports/stop
  P="reports/stop/E9_${CELL}_$(date +%Y%m%d-%H%M%S).md"
  {
    echo "# STOP $CELL"
    echo "1. exit status: $RC"
    echo "2. last 50 log lines of $LOG:"
    echo '```'; tail -50 "$LOG" 2>/dev/null; echo '```'
    echo "3. env: GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) torch=$TORCH cuDNN=$CUDNN HEAD=$(git rev-parse HEAD 2>/dev/null)"
    echo "4. gate output (if any):"
    echo '```'; cat "$LOG_DIR/${CELL}_gate.log" 2>/dev/null; echo '```'
    echo "5. no fix applied; send this file to L"
  } > "$P"
  echo "[stop] wrote $P"
}

run_cell() {
  local LINE="$1" SLOT="$2" CELL OUT rc
  IFS=$'\t' read -r CELL OUT <<< "$LINE"
  export CUDA_VISIBLE_DEVICES=$(( (SLOT - 1) % NGPU ))
  echo "=== $CELL -> GPU $CUDA_VISIBLE_DEVICES (slot $SLOT) $(date +%H:%M:%S) ==="
  set +e
  run_attack "$CELL" "$OUT" > "$LOG_DIR/${CELL}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then stop_bundle "$CELL" "$rc" "$LOG_DIR/${CELL}.log"; return 1; fi
  python tools/eval/e9_gate.py "$CELL" --results_root "$RESULTS_ROOT" > "$LOG_DIR/${CELL}_gate.log" 2>&1
  rc=$?
  cat "$LOG_DIR/${CELL}_gate.log"
  if [ "$rc" -ne 0 ]; then stop_bundle "$CELL" "gate $rc" "$LOG_DIR/${CELL}.log"; return 1; fi
  return 0
}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry_run] NGPU=$NGPU PER_GPU=$PER_GPU RESULTS_ROOT=$RESULTS_ROOT LOG_DIR=$LOG_DIR (nothing started)"
  python() { printf 'python'; local x; for x in "$@"; do printf ' %s' "$x"; done; printf '\n'; }
  I=0
  while IFS=$'\t' read -r C O; do
    [ -n "$C" ] || continue
    I=$(( I + 1 ))
    echo "# $I $C -> $O"
    run_attack "$C" "$O"
  done < "$CELLS_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR" reports/stop
if [ -n "$RESUME_CELL" ]; then
  JOBLOG="$LOG_DIR/parallel_resume_${RESUME_CELL}_$(date +%Y%m%d-%H%M%S).log"
else
  JOBLOG="$LOG_DIR/parallel.log"
fi
echo "[launch] $(date) NGPU=$NGPU PER_GPU=$PER_GPU GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) HEAD=$(git rev-parse --short HEAD)"
echo "[launch] queue:"; cat "$CELLS_FILE"
export NGPU RESULTS_ROOT LOG_DIR TORCH CUDNN
export -f run_cell run_attack stop_bundle
set +e
parallel -j "$JOBS" --joblog "$JOBLOG" --eta 'run_cell {} {%}' < "$CELLS_FILE"
PRC=$?
set -e
NFAIL=$(awk 'NR>1 && ($7 != "0" || $8 != "0"){c++} END{print c+0}' "$JOBLOG")
echo "[summary] $(date) jobs failed or signalled: $NFAIL (parallel rc=$PRC); joblog $JOBLOG"
if [ "$NFAIL" -ne 0 ] || [ "$PRC" -ne 0 ]; then echo "[summary] FAILED: see reports/stop/"; exit 1; fi
echo "[summary] OK: all cells ran and passed the gate. Next: bash e9_finish.sh"
