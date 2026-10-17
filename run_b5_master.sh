#!/bin/bash
# Orchestration Script for B5 Main Run
set -e 

LOG_DIR="reports/raw/B5"
mkdir -p $LOG_DIR
mkdir -p reports/stop

# Cell စာရင်း (လိုအပ်သလို ဖြည့်ပါ)
CELLS=(
  "main_LGC_RC_EGC_resnet50_NT" 
  "main_TtBA_resnet50_NT"
)

for CELL in "${CELLS[@]}"; do
    echo "🚀 Starting Cell: $CELL"
    
    # ၁။ D6 Compliance: Error Row စစ်ဆေးခြင်း
    if ls Results/b5_${CELL}/*/runs_*.csv >/dev/null 2>&1; then
        if grep -q ",error," Results/b5_${CELL}/*/runs_*.csv; then
            echo "❌ ERROR ROW DETECTED in $CELL. STOPPING."
            echo "STOP: Error row in $CELL" > reports/stop/B5_${CELL}_error.md
            exit 1
        fi
    fi

    # ၂။ Experiment ကို run ခြင်း (Agent က ဒီနေရာမှာ command အစစ်ကို ထည့်ပေးပါမယ်)
    python run_experiment.py --cell $CELL > $LOG_DIR/${CELL}.log 2>&1
    
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

    # ၅။ Domain Consistency
    if [[ "$CELL" == *"ARS_OPT"* || "$CELL" == *"CGBA"* ]]; then
        python adapters/diag/domain_consistency.py --roots Results/b5_${CELL} > $LOG_DIR/${CELL}_domain.log 2>&1
        if [ $? -ne 0 ]; then
            echo "❌ DOMAIN X>0 in $CELL. STOPPING."
            echo "STOP: Domain X>0" > reports/stop/B5_${CELL}_domain.md
            exit 1
        fi
    fi
    echo "✅ $CELL PASSED."
done
echo "🎉 ALL CELLS COMPLETED."
