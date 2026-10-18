#!/bin/bash
# Parallel B5 Run for 5 GPUs
set -e

LOG_DIR="reports/raw/B5"
mkdir -p $LOG_DIR
mkdir -p reports/stop

# Cell စာရင်း (Main 24, E10 16, E11 8, E17 4)
CELLS=(
  "main_LGC_RC_EGC_resnet50_NT"
  "main_LGC_RC_EGC_resnet50_T"
  "main_TtBA_resnet50_NT"
  "main_TtBA_resnet50_T"
  "main_CGBA_resnet50_NT"
  "main_CGBA_H_resnet50_T"
  "main_ARS_OPT_resnet50_NT"
  "main_ARS_OPT_resnet50_T"
  "main_LGC_RC_EGC_vgg19_NT"
  "main_LGC_RC_EGC_vgg19_T"
  "main_TtBA_vgg19_NT"
  "main_TtBA_vgg19_T"
  "main_CGBA_vgg19_NT"
  "main_CGBA_H_vgg19_T"
  "main_ARS_OPT_vgg19_NT"
  "main_ARS_OPT_vgg19_T"
  "main_LGC_RC_EGC_vit_b_32_NT"
  "main_LGC_RC_EGC_vit_b_32_T"
  "main_TtBA_vit_b_32_NT"
  "main_TtBA_vit_b_32_T"
  "main_CGBA_vit_b_32_NT"
  "main_CGBA_H_vit_b_32_T"
  "main_ARS_OPT_vit_b_32_NT"
  "main_ARS_OPT_vit_b_32_T"
  "e10_LGC_RC_EGC_resnet50_places365_NT"
  "e10_LGC_RC_EGC_resnet50_places365_T"
  "e10_TtBA_resnet50_places365_NT"
  "e10_TtBA_resnet50_places365_T"
  "e10_CGBA_resnet50_places365_NT"
  "e10_CGBA_H_resnet50_places365_T"
  "e10_ARS_OPT_resnet50_places365_NT"
  "e10_ARS_OPT_resnet50_places365_T"
  "e10_LGC_RC_EGC_densenet161_places365_NT"
  "e10_LGC_RC_EGC_densenet161_places365_T"
  "e10_TtBA_densenet161_places365_NT"
  "e10_TtBA_densenet161_places365_T"
  "e10_CGBA_densenet161_places365_NT"
  "e10_CGBA_H_densenet161_places365_T"
  "e10_ARS_OPT_densenet161_places365_NT"
  "e10_ARS_OPT_densenet161_places365_T"
  "e11_LGC_RC_EGC_Salman2020Do_R50_NT"
  "e11_LGC_RC_EGC_Salman2020Do_R50_T"
  "e11_TtBA_Salman2020Do_R50_NT"
  "e11_TtBA_Salman2020Do_R50_T"
  "e11_CGBA_Salman2020Do_R50_NT"
  "e11_CGBA_H_Salman2020Do_R50_T"
  "e11_ARS_OPT_Salman2020Do_R50_NT"
  "e11_ARS_OPT_Salman2020Do_R50_T"
  "e17_CGBA_D8only_resnet50_NT"
  "e17_CGBA_D8only_vgg19_NT"
  "e17_CGBA_D8only_vit_b_32_NT"
  "e17_CGBA_D8only_Salman2020Do_R50_NT"
)

printf "%s
" "${CELLS[@]}" > /tmp/b5_cells.txt

cat /tmp/b5_cells.txt | parallel -j 5 --joblog $LOG_DIR/parallel.log --eta '
    CELL={}
    GPU_ID=$(( ({%} - 1) % 5 ))
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8

    echo "🚀 Starting $CELL on GPU $GPU_ID"

    # ၁။ D6 Compliance
    if ls Results/b5_${CELL}/*/runs_*.csv >/dev/null 2>&1; then
        if grep -q ",error," Results/b5_${CELL}/*/runs_*.csv; then
            echo "❌ ERROR ROW in $CELL. STOPPING."
            echo "STOP: Error row in $CELL" > reports/stop/B5_${CELL}_error.md
            exit 1
        fi
    fi

    # ၂။ Experiment Run (Contract v1.9-rev5 -> per-cell command)
    CKPTS="100 200 500 1000 2000 2500 5000 10000"
    FLAT=/root/autodl-tmp/ttba_mini_flat
    TTBA_REPO=/root/autodl-tmp/TtBA
    CGBA_REPO=/root/autodl-tmp/baselines/CGBA
    ARS_REPO=/root/autodl-tmp/baselines/ARS-OPT
    OUT="Results/b5_${CELL}"

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

      # ---- E17 (CGBA D8-only appendix, rows 1-30, non_targeted) ----
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
        echo "❌ UNKNOWN CELL $CELL. STOPPING."
        exit 1 ;;
    esac > $LOG_DIR/${CELL}.log 2>&1

    # ၃။ Partial Checker
    python tools/eval/check_contract.py --stage partial --roots Results/b5_${CELL} > $LOG_DIR/${CELL}_checker.log 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ CHECKER FAILED in $CELL. STOPPING."
        echo "STOP: Checker failed" > reports/stop/B5_${CELL}_checker.md
        exit 1
    fi

    # ၄။ G5 Check
    if [[ "$CELL" == *"TtBA"* || "$CELL" == *"CGBA"* ]]; then
        python tools/eval/g5_check.py --roots Results/b5_${CELL} --n 30 > $LOG_DIR/${CELL}_g5.log 2>&1
        if [ $? -ne 0 ]; then
            echo "❌ G5 FAILED in $CELL. STOPPING."
            echo "STOP: G5 failed" > reports/stop/B5_${CELL}_g5.md
            exit 1
        fi
    fi

    # ၅။ Domain Consistency (ARS_OPT / CGBA / CGBA_H / LGC_RC_EGC)
    # X>0 = STOP (post-D19-fix rule; E-OTHER).
    if [[ "$CELL" == *"ARS_OPT"* || "$CELL" == *"CGBA"* || "$CELL" == *"LGC_RC_EGC"* ]]; then
        python adapters/diag/domain_consistency.py \
            --roots "Results/b5_${CELL}" \
            --out "$LOG_DIR/${CELL}_domain.csv" > "$LOG_DIR/${CELL}_domain.log" 2>&1
        DOMAIN_EXIT=$?
        X_COUNT=$(tr -d '\r' < "$LOG_DIR/${CELL}_domain.csv" 2>/dev/null | awk -F',' 'NR>1 && $8=="X"' | wc -l)
        if [ "$DOMAIN_EXIT" -ne 0 ] || [ "$X_COUNT" -gt 0 ]; then
            echo "❌ DOMAIN FAILED in $CELL (X=$X_COUNT, exit=$DOMAIN_EXIT). STOPPING."
            echo "STOP: domain_consistency X=$X_COUNT (exit=$DOMAIN_EXIT)" > reports/stop/B5_${CELL}_domain.md
            exit 1
        fi
    fi
    echo "✅ $CELL PASSED."
'
echo "🎉 ALL CELLS COMPLETED."
