#!/bin/bash
# B5 launcher v2 (attempt 2).  One host, N GPUs (NGPU >= 1), one attack process per GPU (D14).
#
# Contract v1.9-rev7.  Frozen engine.  This rebuilds the B5 attempt-1 launcher (c20cc349) to fix F1-F10.
#
# The 52 attack command lines are byte-identical to c20cc349 after variable expansion, EXCEPT the E17
# output directory (H4: Results/appendix_cgba_d8only_<arch>).  Gates are delegated to
# tools/eval/b5_gate_cell.py (partial / G5 / domain); error rows are read with the csv module, never grep.
#
# Exit codes: 0 ok; 1 at least one job exited nonzero or was signalled (attack failure or stopping gate;
#             per-cell record in the joblog, STOP bundle under reports/stop/); 2 usage / NGPU / tmux /
#             fresh-directory refusal; 3 environment-guard refusal.
set -euo pipefail

# -------------------------------------------------------------------------------------------------- 1. args
DRY_RUN=0
RESUME_CELL=""
usage() { echo "usage: $0 [--dry_run] [--resume <cell>]"; }
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
LOG_DIR="${LOG_DIR:-reports/raw/B5a2}"
ENV_FILE="${ENV_FILE:-env_build38.txt}"

# NGPU is a required environment variable (integer >= 1) -- T1a.
if [ -z "${NGPU:-}" ]; then echo "[!] NGPU is required (integer >= 1)" >&2; exit 2; fi
case "$NGPU" in ''|*[!0-9]*) echo "[!] NGPU must be an integer >= 1" >&2; exit 2;; esac
if [ "$NGPU" -lt 1 ]; then echo "[!] NGPU must be >= 1" >&2; exit 2; fi

attack_running() {
  # True while an attack process runs: the START of its command line is "python" followed by one of the
  # five attack entry points.  Anchored so that editors, grep or log viewers never block a resume.
  ps -eo args= 2>/dev/null | grep -Eq '^python[[:space:]]+(run_benchmark_33b\.py|adapters/run_ttba_v2\.py|adapters/run_cgba\.py|adapters/run_arsopt\.py|ttba_github/run_frame_b1_generic\.py)'
}

# -------------------------------------------------------------------------------------------------- 2. env guard (T1c)
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
fi

# -------------------------------------------------------------------------------------------------- 3. tmux (T1d)
if [ "$DRY_RUN" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  echo "[!] not inside tmux; refusing to start (use --dry_run to bypass)" >&2
  exit 2
fi

# -------------------------------------------------------------------------------------------------- 4. queue + fresh-directory (T1e, T1f)
ORDER_FILE=$(mktemp)
CELLS_FILE=$(mktemp)
trap 'rm -f "$ORDER_FILE" "$CELLS_FILE"' EXIT
python tools/eval/b5_gate_cell.py --print_order --results_root "$RESULTS_ROOT" > "$ORDER_FILE"

if [ -n "$RESUME_CELL" ]; then
  RESUME_OUTDIR=$(awk -F'\t' -v c="$RESUME_CELL" '$1==c{print $2}' "$ORDER_FILE")
  [ -n "$RESUME_OUTDIR" ] || { echo "[!] unknown cell: $RESUME_CELL" >&2; exit 2; }
  printf '%s\t%s\n' "$RESUME_CELL" "$RESUME_OUTDIR" > "$CELLS_FILE"
  echo "[resume] running only $RESUME_CELL -> $RESUME_OUTDIR"
  # a resume always gets GPU 0: refuse while another attack process runs (match the START of its command line).
  if attack_running; then
    echo "[!] an attack process is already running; refusing --resume (GPU 0 contention)" >&2
    exit 2
  fi
  # error-row check reads the status/error columns with the csv module, never grep ',error,'.
  if python tools/eval/b5_gate_cell.py --print_errors --root "$RESUME_OUTDIR"; then
    echo "[resume] no error rows; proceeding"
  else
    echo "[!] error rows found under $RESUME_OUTDIR; refusing resume (E-OTHER, needs investigation)" >&2
    exit 1
  fi
else
  if [ "$DRY_RUN" -eq 0 ]; then
    while IFS=$'\t' read -r C O; do
      [ -n "$C" ] || continue
      if [ -e "$O" ]; then
        echo "[!] output directory already exists: $O (cell $C); refusing. Use --resume $C or a fresh RESULTS_ROOT." >&2
        exit 2
      fi
    done < "$ORDER_FILE"
  fi
  cp "$ORDER_FILE" "$CELLS_FILE"
fi

# -------------------------------------------------------------------------------------------------- 5. NGPU match (T1a)
if [ "$DRY_RUN" -eq 0 ]; then
  NGPU_ACTUAL=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$NGPU" -ne "$NGPU_ACTUAL" ]; then
    echo "[!] NGPU=$NGPU != nvidia-smi -L count=$NGPU_ACTUAL; refusing" >&2
    exit 2
  fi
fi

# -------------------------------------------------------------------------------------------------- 6. threads (T1b)
CGROUP_FILE=/sys/fs/cgroup/cpu.max
read -r THREADS Q < <(python - "$NGPU" "$CGROUP_FILE" <<'PY'
import os, subprocess, sys
ngpu = int(sys.argv[1])
cg = sys.argv[2]
if os.path.exists(cg):
    with open(cg) as f:
        parts = f.read().split()
    if parts and parts[0] == "max":
        q = float(int(subprocess.run(["env", "-u", "OMP_NUM_THREADS", "-u", "OMP_THREAD_LIMIT", "nproc"],
                                     capture_output=True, text=True).stdout.strip()))
    else:
        q = int(parts[0]) / int(parts[1])
else:
    q = float(int(subprocess.run(["env", "-u", "OMP_NUM_THREADS", "-u", "OMP_THREAD_LIMIT", "nproc"],
                                 capture_output=True, text=True).stdout.strip()))
t = max(1, min(8, int(q // ngpu)))
print(t, q)
PY
)
OMP_NUM_THREADS=$THREADS
MKL_NUM_THREADS=$THREADS
echo "[threads] Q=$Q NGPU=$NGPU -> OMP_NUM_THREADS=MKL_NUM_THREADS=$THREADS"

# -------------------------------------------------------------------------------------------------- helpers
GPU_NAME=""
DRIVER=""
probe_gpu() {
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
}

stop_bundle() {
  # fields 1-7 per skill §7.  $5 (optional) = cell output dir, for "rows already completed".
  local CELL="$1" CODE="$2" SIG="$3" LOG="$4" OUT="${5:-}"
  local TS; TS=$(date +%Y%m%d-%H%M)
  mkdir -p reports/stop
  local P="reports/stop/B5a2_${CELL}_${TS}.md"
  local N=2
  while [ -e "$P" ]; do
    P="reports/stop/B5a2_${CELL}_${TS}_${N}.md"
    N=$((N + 1))
  done
  local NROWS=0
  if [ -n "$OUT" ] && [ -d "$OUT" ]; then
    NROWS=$(find "$OUT" -type f -name 'runs*.csv' -print0 2>/dev/null \
            | xargs -0 -r awk 'FNR>1 {c++} END {print c+0}' 2>/dev/null) || NROWS=0
    NROWS=${NROWS:-0}
  fi
  {
    echo "# STOP $CELL"
    echo
    echo "1. code: $CODE"
    echo "2. signature: $SIG"
    echo "3. evidence: (last 50 log lines of $LOG)"
    echo '```'
    tail -50 "$LOG" 2>/dev/null
    echo '```'
    echo "4. env: GPU=${GPU_NAME:-?} driver=${DRIVER:-?} torch=${TORCH:-?} cuDNN=${CUDNN:-?} flags(matmul_tf32,cudnn_tf32,deterministic,benchmark)=${TF32_MM:-?},${TF32_CUDNN:-?},${CUDNN_DET:-?},${CUDNN_BENCH:-?} HEAD=$(git rev-parse HEAD 2>/dev/null || echo '?') contract=${CONTRACT_SHA:-?}"
    echo "5. scope: cell $CELL; ${NROWS} rows completed"
    echo "6. candidates: not diagnosed (no fix applied)"
    echo "7. proposed lesson: none"
  } > "$P"
  echo "[stop] wrote $P"
}

# run_attack CELL OUT -- execute the (byte-identical) attack command; returns the attack exit status.
run_attack() {
  local CELL="$1" OUT="$2"
  local CKPTS="100 200 500 1000 2000 2500 5000 10000"
  local FLAT=/root/autodl-tmp/ttba_mini_flat
  local TTBA_REPO=/root/autodl-tmp/TtBA
  local CGBA_REPO=/root/autodl-tmp/baselines/CGBA
  local ARS_REPO=/root/autodl-tmp/baselines/ARS-OPT

  case "$CELL" in
    # ---- Main (ImageNet): resnet50 / vgg19 / vit_b_32 ----
    main_LGC_RC_EGC_resnet50_NT)
      python run_benchmark_33b.py --dataset imagenet --mode non_targeted --archs resnet50 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_LGC_RC_EGC_resnet50_T)
      python run_benchmark_33b.py --dataset imagenet --mode targeted --archs resnet50 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_LGC_RC_EGC_vgg19_NT)
      python run_benchmark_33b.py --dataset imagenet --mode non_targeted --archs vgg19 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_LGC_RC_EGC_vgg19_T)
      python run_benchmark_33b.py --dataset imagenet --mode targeted --archs vgg19 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_LGC_RC_EGC_vit_b_32_NT)
      python run_benchmark_33b.py --dataset imagenet --mode non_targeted --archs vit_b_32 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_LGC_RC_EGC_vit_b_32_T)
      python run_benchmark_33b.py --dataset imagenet --mode targeted --archs vit_b_32 \
        --methods LGC_RC_EGC --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TTBA_REPO/model" --num_samples 200 --max_queries 10000 \
        --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv ;;
    main_TtBA_resnet50_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-resnet50 --targeted 0 \
        --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_TtBA_resnet50_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-resnet50 --targeted 1 \
        --split_csv splits/imagenet/ttba_github/resnet50_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_TtBA_vgg19_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-vgg --targeted 0 \
        --split_csv splits/imagenet/ttba_github/vgg19_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_TtBA_vgg19_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-vgg --targeted 1 \
        --split_csv splits/imagenet/ttba_github/vgg19_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_TtBA_vit_b_32_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-vit --targeted 0 \
        --split_csv splits/imagenet/ttba_github/vit_b_32_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_TtBA_vit_b_32_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --dataset imagenet-vit --targeted 1 \
        --split_csv splits/imagenet/ttba_github/vit_b_32_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    main_CGBA_resnet50_NT)
      python adapters/run_cgba.py --arch resnet50 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_CGBA_vgg19_NT)
      python adapters/run_cgba.py --arch vgg19 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vgg19_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_CGBA_vit_b_32_NT)
      python adapters/run_cgba.py --arch vit_b_32 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vit_b_32_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_CGBA_H_resnet50_T)
      python adapters/run_cgba.py --arch resnet50 --mode targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_CGBA_H_vgg19_T)
      python adapters/run_cgba.py --arch vgg19 --mode targeted \
        --split_csv splits/imagenet/ttba_github/vgg19_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_CGBA_H_vit_b_32_T)
      python adapters/run_cgba.py --arch vit_b_32 --mode targeted \
        --split_csv splits/imagenet/ttba_github/vit_b_32_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    main_ARS_OPT_resnet50_NT)
      python adapters/run_arsopt.py --arch resnet50 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    main_ARS_OPT_resnet50_T)
      python adapters/run_arsopt.py --arch resnet50 --mode targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    main_ARS_OPT_vgg19_NT)
      python adapters/run_arsopt.py --arch vgg19 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vgg19_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    main_ARS_OPT_vgg19_T)
      python adapters/run_arsopt.py --arch vgg19 --mode targeted \
        --split_csv splits/imagenet/ttba_github/vgg19_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    main_ARS_OPT_vit_b_32_NT)
      python adapters/run_arsopt.py --arch vit_b_32 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vit_b_32_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    main_ARS_OPT_vit_b_32_T)
      python adapters/run_arsopt.py --arch vit_b_32 --mode targeted \
        --split_csv splits/imagenet/ttba_github/vit_b_32_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;

    # ---- E10 (Places365): resnet50_places365 / densenet161_places365 ----
    e10_LGC_RC_EGC_resnet50_places365_NT)
      python ttba_github/run_frame_b1_generic.py --arch resnet50_places365 --mode non_targeted \
        --split_csv splits/places365/resnet50_places365_non_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e10_LGC_RC_EGC_resnet50_places365_T)
      python ttba_github/run_frame_b1_generic.py --arch resnet50_places365 --mode targeted \
        --split_csv splits/places365/resnet50_places365_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e10_LGC_RC_EGC_densenet161_places365_NT)
      python ttba_github/run_frame_b1_generic.py --arch densenet161_places365 --mode non_targeted \
        --split_csv splits/places365/densenet161_places365_non_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e10_LGC_RC_EGC_densenet161_places365_T)
      python ttba_github/run_frame_b1_generic.py --arch densenet161_places365 --mode targeted \
        --split_csv splits/places365/densenet161_places365_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e10_TtBA_resnet50_places365_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch resnet50_places365 --targeted 0 \
        --split_csv splits/places365/resnet50_places365_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e10_TtBA_resnet50_places365_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch resnet50_places365 --targeted 1 \
        --split_csv splits/places365/resnet50_places365_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e10_TtBA_densenet161_places365_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch densenet161_places365 --targeted 0 \
        --split_csv splits/places365/densenet161_places365_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e10_TtBA_densenet161_places365_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch densenet161_places365 --targeted 1 \
        --split_csv splits/places365/densenet161_places365_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e10_CGBA_resnet50_places365_NT)
      python adapters/run_cgba.py --arch resnet50_places365 --mode non_targeted \
        --split_csv splits/places365/resnet50_places365_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e10_CGBA_densenet161_places365_NT)
      python adapters/run_cgba.py --arch densenet161_places365 --mode non_targeted \
        --split_csv splits/places365/densenet161_places365_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e10_CGBA_H_resnet50_places365_T)
      python adapters/run_cgba.py --arch resnet50_places365 --mode targeted \
        --split_csv splits/places365/resnet50_places365_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e10_CGBA_H_densenet161_places365_T)
      python adapters/run_cgba.py --arch densenet161_places365 --mode targeted \
        --split_csv splits/places365/densenet161_places365_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e10_ARS_OPT_resnet50_places365_NT)
      python adapters/run_arsopt.py --arch resnet50_places365 --mode non_targeted \
        --split_csv splits/places365/resnet50_places365_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    e10_ARS_OPT_resnet50_places365_T)
      python adapters/run_arsopt.py --arch resnet50_places365 --mode targeted \
        --split_csv splits/places365/resnet50_places365_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    e10_ARS_OPT_densenet161_places365_NT)
      python adapters/run_arsopt.py --arch densenet161_places365 --mode non_targeted \
        --split_csv splits/places365/densenet161_places365_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    e10_ARS_OPT_densenet161_places365_T)
      python adapters/run_arsopt.py --arch densenet161_places365 --mode targeted \
        --split_csv splits/places365/densenet161_places365_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;

    # ---- E11 (RobustBench): Salman2020Do_R50 ----
    e11_LGC_RC_EGC_Salman2020Do_R50_NT)
      python ttba_github/run_frame_b1_generic.py --arch Salman2020Do_R50 --mode non_targeted \
        --split_csv splits/robust/Salman2020Do_R50_non_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e11_LGC_RC_EGC_Salman2020Do_R50_T)
      python ttba_github/run_frame_b1_generic.py --arch Salman2020Do_R50 --mode targeted \
        --split_csv splits/robust/Salman2020Do_R50_targeted.csv \
        --out_dir "$OUT" --num_samples 200 --save_adv ;;
    e11_TtBA_Salman2020Do_R50_NT)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch Salman2020Do_R50 --targeted 0 \
        --split_csv splits/robust/Salman2020Do_R50_non_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e11_TtBA_Salman2020Do_R50_T)
      python adapters/run_ttba_v2.py --ttba_repo "$TTBA_REPO" --arch Salman2020Do_R50 --targeted 1 \
        --split_csv splits/robust/Salman2020Do_R50_targeted.csv \
        --out_dir "$OUT" --limit 200 ;;
    e11_CGBA_Salman2020Do_R50_NT)
      python adapters/run_cgba.py --arch Salman2020Do_R50 --mode non_targeted \
        --split_csv splits/robust/Salman2020Do_R50_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e11_CGBA_H_Salman2020Do_R50_T)
      python adapters/run_cgba.py --arch Salman2020Do_R50 --mode targeted \
        --split_csv splits/robust/Salman2020Do_R50_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 200 --cgba_variant fixed ;;
    e11_ARS_OPT_Salman2020Do_R50_NT)
      python adapters/run_arsopt.py --arch Salman2020Do_R50 --mode non_targeted \
        --split_csv splits/robust/Salman2020Do_R50_non_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;
    e11_ARS_OPT_Salman2020Do_R50_T)
      python adapters/run_arsopt.py --arch Salman2020Do_R50 --mode targeted \
        --split_csv splits/robust/Salman2020Do_R50_targeted.csv \
        --repo "$ARS_REPO" --out_dir "$OUT" --limit 200 ;;

    # ---- E17 (CGBA D8-only appendix, rows 1-30, non_targeted; H4 appendix dir) ----
    e17_CGBA_D8only_resnet50_NT)
      python adapters/run_cgba.py --arch resnet50 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 30 --cgba_variant d8only ;;
    e17_CGBA_D8only_vgg19_NT)
      python adapters/run_cgba.py --arch vgg19 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vgg19_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 30 --cgba_variant d8only ;;
    e17_CGBA_D8only_vit_b_32_NT)
      python adapters/run_cgba.py --arch vit_b_32 --mode non_targeted \
        --split_csv splits/imagenet/ttba_github/vit_b_32_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 30 --cgba_variant d8only ;;
    e17_CGBA_D8only_Salman2020Do_R50_NT)
      python adapters/run_cgba.py --arch Salman2020Do_R50 --mode non_targeted \
        --split_csv splits/robust/Salman2020Do_R50_non_targeted.csv \
        --repo "$CGBA_REPO" --out_dir "$OUT" --limit 30 --cgba_variant d8only ;;
    *)
      echo "[!] unknown cell $CELL" >&2
      return 2 ;;
  esac
}

describe_gates() {
  local CELL="$1"
  local PART G5 DOM
  case "$CELL" in e17_*) PART="skip (E17)";; *) PART="check_contract --stage partial";; esac
  case "$CELL" in
    main_TtBA_*|main_CGBA_*|main_CGBA_H_*) G5="g5_check (main study)";;
    e10_*|e11_*)
      case "$CELL" in *TtBA*|*CGBA*) G5="G5 n/a (no published reference)";; *) G5="n/a";; esac ;;
    *) G5="n/a";;
  esac
  case "$CELL" in
    *TtBA*) DOM="n/a";;
    *LGC_RC_EGC*|*CGBA_D8only*) DOM="domain_consistency (X recorded, no stop)";;
    *ARS_OPT*|*CGBA*) DOM="domain_consistency (X>0 stops)";;
    *) DOM="n/a";;
  esac
  echo "partial=$PART; g5=$G5; domain=$DOM"
}

run_cell() {
  local LINE="$1" SLOT="$2"
  local CELL OUT
  IFS=$'\t' read -r CELL OUT <<< "$LINE"
  local GPU_ID=$(( (SLOT - 1) % NGPU ))
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  export OMP_NUM_THREADS=$OMP_NUM_THREADS
  export MKL_NUM_THREADS=$MKL_NUM_THREADS
  echo "=== $CELL -> GPU $GPU_ID (slot $SLOT) ==="
  set +e
  run_attack "$CELL" "$OUT" > "$LOG_DIR/${CELL}.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    local CODE=E-OTHER
    if [ "$rc" -ge 128 ] || grep -qE 'CUDA out of memory|CUDA error|cudaError|out of memory|No space left on device|Killed' "$LOG_DIR/${CELL}.log"; then
      CODE=E-INFRA
    fi
    stop_bundle "$CELL" "$CODE" "attack exit status $rc (see last 50 log lines)" "$LOG_DIR/${CELL}.log" "$OUT"
    echo "[cell $CELL] attack FAILED rc=$rc -> STOP ($CODE)"
    return 1
  fi
  python tools/eval/b5_gate_cell.py "$CELL" --results_root "$RESULTS_ROOT" --log_dir "$LOG_DIR" \
    > "$LOG_DIR/${CELL}_gates.log" 2>&1
  local grc=$?
  echo "[cell $CELL] gates exit $grc"
  return $grc
}

# -------------------------------------------------------------------------------------------------- dry run (T1j)
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry_run] NGPU=$NGPU RESULTS_ROOT=$RESULTS_ROOT LOG_DIR=$LOG_DIR (no GPU process started)"
  # shim: print the command instead of running it
  python() { printf 'python'; local a; for a in "$@"; do printf ' %s' "$a"; done; printf '\n'; }
  IDX=0
  while IFS=$'\t' read -r CELL OUT; do
    [ -n "$CELL" ] || continue
    IDX=$(( IDX + 1 ))
    SLOT=$(( (IDX - 1) % NGPU + 1 ))
    GPU=$(( (SLOT - 1) % NGPU ))
    echo "# cell: $CELL"
    echo "# gpu_slot: $GPU   (formula GPU_ID=\$(( ({%} - 1) % NGPU )); job slot $SLOT of $NGPU)"
    echo "# outdir: $OUT"
    echo "# gates: $(describe_gates "$CELL")"
    run_attack "$CELL" "$OUT"
  done < "$CELLS_FILE"
  echo
  echo "[dry_run] queue (LPT, E17 last) with expected hours:"
  command python tools/eval/b5_gate_cell.py --print_queue --results_root "$RESULTS_ROOT"
  exit 0
fi

# -------------------------------------------------------------------------------------------------- launch (T1a/T1k)
if [ -n "$RESUME_CELL" ]; then
  JOBLOG="$LOG_DIR/parallel_resume_${RESUME_CELL}_$(date +%Y%m%d-%H%M%S).log"
else
  JOBLOG="$LOG_DIR/parallel.log"
fi
mkdir -p "$LOG_DIR" reports/stop
probe_gpu

# D12 backend flags as applied by adapters/_common.py at import (skill §7 field 4), and the contract hash.
FLAGS="? ? ? ?"
if FLAGS_OUT=$(PYTHONPATH="$PWD/adapters:$PWD" python - <<'PY' 2>/dev/null
import torch
import _common  # configure_precision() runs at import of adapters/_common.py (D12)
print(int(torch.backends.cuda.matmul.allow_tf32), int(torch.backends.cudnn.allow_tf32),
      int(torch.backends.cudnn.deterministic), int(torch.backends.cudnn.benchmark))
PY
); then
  FLAGS=$(printf '%s\n' "$FLAGS_OUT" | tail -n1)
fi
read -r TF32_MM TF32_CUDNN CUDNN_DET CUDNN_BENCH <<< "$FLAGS"
CONTRACT_SHA=$(sha256sum contract/paper_contract.yaml 2>/dev/null | awk '{print $1}') || true
CONTRACT_SHA=${CONTRACT_SHA:-?}

echo "[launch] NGPU=$NGPU RESULTS_ROOT=$RESULTS_ROOT LOG_DIR=$LOG_DIR threads=$THREADS tmux=${TMUX:-<none>}"
echo "[launch] env: GPU=${GPU_NAME:-?} driver=${DRIVER:-?} torch=${TORCH:-?} cuDNN=${CUDNN:-?} flags=${FLAGS} contract=${CONTRACT_SHA}"
echo "[launch] ordered queue (LPT, E17 last) with expected hours:"
python tools/eval/b5_gate_cell.py --print_queue --results_root "$RESULTS_ROOT"
echo "[launch] ordered queue (cell -> outdir):"
cat "$CELLS_FILE"

export NGPU RESULTS_ROOT LOG_DIR OMP_NUM_THREADS MKL_NUM_THREADS \
       GPU_NAME DRIVER TORCH CUDNN TF32_MM TF32_CUDNN CUDNN_DET CUDNN_BENCH CONTRACT_SHA
export -f run_cell run_attack stop_bundle describe_gates

set +e
cat "$CELLS_FILE" | parallel -j "$NGPU" --joblog "$JOBLOG" --eta \
  'run_cell {} {%}'
PARALLEL_RC=$?
set -e

# -------------------------------------------------------------------------------------------------- summary (T1k)
echo
echo "[summary] from $JOBLOG:"
NFAIL=0
if [ -f "$JOBLOG" ]; then
  awk 'NR==1{print; next} {n++; if($7 != "0" || $8 != "0") fail++} END{printf "jobs=%d  nonzero_or_signalled=%d\n", n, fail+0}' "$JOBLOG"
  echo "per-job exit column (Seq Exitval Signal cell):"
  awk -F'\t' 'NR>1 {
    cell = $9
    sub(/^run_cell ./, "", cell)
    print $1, $7, $8, cell
  }' "$JOBLOG"
  NFAIL=$(awk 'NR>1 && ($7 != "0" || $8 != "0"){c++} END{print c+0}' "$JOBLOG")
  NFAIL=${NFAIL:-0}
fi
if [ "$NFAIL" -ne 0 ] || [ "$PARALLEL_RC" -ne 0 ]; then
  echo "[summary] FAILED: jobs with nonzero exit/signal = $NFAIL (parallel rc=$PARALLEL_RC); see the joblog and reports/stop/"
  exit 1
fi
echo "[summary] OK: all jobs exited 0"