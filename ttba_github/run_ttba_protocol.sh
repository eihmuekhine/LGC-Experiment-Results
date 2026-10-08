#!/bin/bash
# Build 38: official TtBA protocol, one victim after the other.
#   bash ttba_github/run_ttba_protocol.sh "resnet50 vgg19 vit_b_32"
# For each victim it starts the official TtBA code and our frozen method LGC_RC_EGC on the SAME dumped list (both
# modes, in parallel), waits, and moves on.  After the last victim it runs the pre-registered confirm_ttba.py.
# Environment variables (defaults in brackets):
#   LGC [/root/autodl-tmp/Latent-Geometric-Chords]  TT [/root/autodl-tmp/TtBA]  FLAT [/root/autodl-tmp/ttba_mini_flat]
#   N [1000] images per (victim, mode)   EGC_SHARDS [2] processes per mode for ours
#   TTBA_SHARDS [1]: 1 = exact main.py RNG stream (one process per mode); >1 = per-image seeds, sharded
#   LPIPS_NET [alex]: MUST equal the net used by lgc_bench/metrics.py (grep -n "LPIPS(" lgc_bench/metrics.py)
set -u
LGC=${LGC:-/root/autodl-tmp/Latent-Geometric-Chords}
TT=${TT:-/root/autodl-tmp/TtBA}
FLAT=${FLAT:-/root/autodl-tmp/ttba_mini_flat}
N=${N:-1000}
EGC_SHARDS=${EGC_SHARDS:-2}
TTBA_SHARDS=${TTBA_SHARDS:-1}
LPIPS_NET=${LPIPS_NET:-alex}
ARCHS=${1:-"resnet50 vgg19 vit_b_32"}
declare -A DS=([resnet50]=imagenet-resnet50 [vgg19]=imagenet-vgg [vit_b_32]=imagenet-vit
               [inception_v3]=imagenet-inceptionv3 [efficientnet_b0]=imagenet-efficient)
cd "$LGC" || exit 1
mkdir -p logs_ttbagh
for A in $ARCHS; do
  OUT=Results/ttbagh_$A
  mkdir -p "$OUT"
  python -c "import torch;print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(0))" > "$OUT/env.txt"
  nvidia-smi --query-gpu=name,driver_version --format=csv >> "$OUT/env.txt"
  for T in 0 1; do
    if [ "$T" = 1 ]; then M=targeted; else M=non_targeted; fi
    SPLIT=$LGC/splits/imagenet/ttba_github/${A}_${M}.csv
    if [ ! -f "$SPLIT" ]; then echo "[!] missing $SPLIT (run ttba_dump_split.py first)"; exit 1; fi
    if [ "$TTBA_SHARDS" = 1 ]; then
      python ttba_github/ttba_run_official.py --ttba_repo "$TT" --dataset "${DS[$A]}" --targeted $T \
        --split_csv "$SPLIT" --out_dir "$LGC/$OUT" --lpips_net "$LPIPS_NET" --limit "$N" \
        > "logs_ttbagh/ttba_${A}_${M}.log" 2>&1 &
    else
      for S in $(seq 0 $((TTBA_SHARDS - 1))); do
        python ttba_github/ttba_run_official.py --ttba_repo "$TT" --dataset "${DS[$A]}" --targeted $T \
          --split_csv "$SPLIT" --out_dir "$LGC/$OUT" --lpips_net "$LPIPS_NET" --limit "$N" \
          --seed_mode per_image --num_shards "$TTBA_SHARDS" --shard_index $S \
          > "logs_ttbagh/ttba_${A}_${M}_$S.log" 2>&1 &
      done
    fi
    for S in $(seq 0 $((EGC_SHARDS - 1))); do
      python run_benchmark_33b.py --dataset imagenet --mode $M --archs "$A" --methods LGC_RC_EGC \
        --data_dir "$FLAT" --labels_file ttba_mini_val_labels.txt \
        --split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv" --image_loader ttba_github \
        --victim_ckpt_dir "$TT/model" --num_samples "$N" --max_queries 10000 \
        --checkpoints 500 1000 2000 2500 5000 10000 --seed 42 --exclude_seeds 992 2027 3141 2718 \
        --out_dir "$OUT" --num_shards "$EGC_SHARDS" --shard_index $S \
        > "logs_ttbagh/egc_${A}_${M}_$S.log" 2>&1 &
    done
  done
  wait
  echo "[*] $A finished $(date)"
done
python confirm_ttba.py --archs $ARCHS --res_pattern "Results/ttbagh_{arch}" \
  --split_pattern "splits/imagenet/ttba_github/{arch}_{mode}.csv" --our_lpips_net "$LPIPS_NET" \
  > confirm_ttba_log.txt 2>&1
echo DONE >> confirm_ttba_log.txt
