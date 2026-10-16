#!/bin/bash
# Frame B1 (Build 38, Amendment 2): official TtBA protocol, first N=200 rows of the dumped split per (victim, mode).
# v2: EGC runs with --save_adv (adversarial image at EVERY checkpoint -> <out>/<mode>/adv_store/, see advstore.py), so
#     any image-quality metric can be added later with eval_metrics_offline.py WITHOUT re-running an attack.
#     Fresh output folder OUT_PREFIX_<arch> (default Results/b1_<arch>) so that every image has a stored state.
#     Official TtBA runs only with RUN_TTBA=1 AFTER ttba_run_official.py has the same --save_adv patch (TTBA_EXTRA).
# LGC_RC_EGC (frozen) and official TtBA (exact main.py RNG) run on the SAME rows, SEQUENTIALLY (one process at a time).
# Resume-safe: both runners skip images that are already complete in the SAME output folder.
#   bash ttba_github/run_frame_b1.sh "resnet50 vgg19 vit_b_32"
# Optional breadth (TtBA Table 1 victims):  MODES="0" bash ttba_github/run_frame_b1.sh "inception_v3 efficientnet_b0"
# Smoke timing first:  N=3 bash ttba_github/run_frame_b1.sh "resnet50"   (then rerun with N=200; resume continues)
set -u
LGC=${LGC:-/root/autodl-tmp/LGC_frozen}   # kit v5: clean worktree of build38-prereg + paper-runs branch
TT=${TT:-/root/autodl-tmp/TtBA}
FLAT=${FLAT:-/root/autodl-tmp/ttba_mini_flat}
N=${N:-200}
OUT_PREFIX=${OUT_PREFIX:-Results/b1}
RUN_TTBA=${RUN_TTBA:-0}               # 1 after the wrapper patch
TTBA_EXTRA=${TTBA_EXTRA:-""}          # e.g. "--save_adv" once the wrapper supports it
MODES=${MODES:-"0 1"}                       # 0 = non_targeted, 1 = targeted
THREADS=${THREADS:-8}                       # caps CPU threads per process (Build 38 slowed ~13x under contention)
export LPIPS_NET=vgg OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
ARCHS=${1:-"resnet50 vgg19 vit_b_32"}
declare -A DS=([resnet50]=imagenet-resnet50 [vgg19]=imagenet-vgg [vit_b_32]=imagenet-vit
               [inception_v3]=imagenet-inceptionv3 [efficientnet_b0]=imagenet-efficient)
cd "$LGC" || exit 1
# kit v5 guards: frozen engine, --save_adv patch, advstore at the repo root, frozen split hashes
sha256sum lgc_bench/engine.py | grep -q '^835f1933d9e5' || { echo '[!] engine sha is not 835f1933d9e5'; exit 1; }
grep -q -- '--save_adv' run_benchmark_33b.py || { echo '[!] run apply_advstore_patch.py first'; exit 1; }
[ -f advstore.py ] || { echo '[!] advstore.py missing at the repo root (cp tools/eval/advstore.py .)'; exit 1; }
[ -f contract/paper_contract.yaml ] || { echo '[!] contract/paper_contract.yaml missing (run C0 first)'; exit 1; }
# checkpoints come from the contract only (single source of truth); CKPTS may not be overridden by hand
CKPTS=$(python3 -c "import yaml;print(*yaml.safe_load(open('contract/paper_contract.yaml'))['constants']['checkpoints'])") || exit 1
echo "[*] checkpoints (contract): $CKPTS"
[ -z "${ADV_STORE_BUDGETS:-}" ] || { echo '[!] ADV_STORE_BUDGETS is set: unset it (contract I11)'; exit 1; }
mkdir -p logs_frame
for A in $ARCHS; do
  OUT=${OUT_PREFIX}_$A
  mkdir -p "$OUT"
  { date; python -c "import torch;print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0))";
    nvidia-smi --query-gpu=name,driver_version --format=csv; git rev-parse HEAD; sha256sum lgc_bench/engine.py run_benchmark_33b.py;
    sha256sum splits/imagenet/ttba_github/${A}_non_targeted.csv splits/imagenet/ttba_github/${A}_targeted.csv contract/paper_contract.yaml;
    echo "checkpoints $CKPTS"; echo "ADV_STORE_DTYPE ${ADV_STORE_DTYPE:-float16} ADV_STORE_BUDGETS unset"; } >> "$OUT/env_frame.txt" 2>&1
  for T in $MODES; do
    if [ "$T" = 1 ]; then M=targeted; else M=non_targeted; fi
    SPLIT=$LGC/splits/imagenet/ttba_github/${A}_${M}.csv
    if [ ! -f "$SPLIT" ]; then echo "[!] missing $SPLIT (run ttba_dump_split.py first)"; exit 1; fi
    t0=$(date +%s)
    # Same attack arguments as Build 38; --save_adv is evaluation-only (no query, no RNG).
    python run_benchmark_33b.py --dataset imagenet --mode $M --archs "$A" --methods LGC_RC_EGC \
      --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
      --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
      --victim_ckpt_dir "$TT/model" --num_samples "$N" --max_queries 10000 \
      --checkpoints $CKPTS --seed 42 --exclude_seeds 992 2027 3141 2718 \
      --out_dir "$OUT" --num_shards 1 --shard_index 0 --save_adv >> "logs_frame/egc_${A}_${M}.log" 2>&1
    t1=$(date +%s)
    if [ "$RUN_TTBA" = 1 ]; then
      python ttba_github/ttba_run_official.py --ttba_repo "$TT" --dataset "${DS[$A]}" --targeted $T \
        --split_csv "$SPLIT" --out_dir "$LGC/$OUT" --lpips_net "$LPIPS_NET" --limit "$N" $TTBA_EXTRA \
        >> "logs_frame/ttba_${A}_${M}.log" 2>&1
    fi
    t2=$(date +%s)
    echo "[*] $A $M  EGC $((t1-t0)) s  TtBA $((t2-t1)) s  $(date)" | tee -a logs_frame/timing.txt
  done
done
echo DONE | tee -a logs_frame/timing.txt
