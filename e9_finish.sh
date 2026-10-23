#!/bin/bash
# E9 finish (prereg_build38_amendment3.txt): gate, offline metrics, export, registered claims, manifest.  L, 2026-10-23.
# Run from /root/autodl-tmp/LGC_frozen after run_e9_parallel.sh ended with "[summary] OK".  One GPU is enough
# (offline metrics, about 1-2 min per cell); everything else is CPU.  Idempotent: metrics are resume-safe and the
# export is rebuilt from Results/ each time.  It never runs git; the commit, tag and push are H's (README step 5).
#
#   bash e9_finish.sh            all steps
#   DEVICE=cuda:1 bash e9_finish.sh
set -euo pipefail

DEVICE="${DEVICE:-cuda:0}"
RESULTS_ROOT="${RESULTS_ROOT:-Results}"
EXP=paper_exports/e9
LOG=reports/raw/E9/finish_$(date +%Y%m%d-%H%M%S).log
mkdir -p reports/raw/E9 "$EXP/cells" "$EXP/metrics" "$EXP/claims"
exec > >(tee -a "$LOG") 2>&1
echo "[finish] $(date) HEAD=$(git rev-parse --short HEAD) device=$DEVICE"

echo "== 1. gate (all 18 cells)"
python tools/eval/e9_gate.py --all --results_root "$RESULTS_ROOT"

CELLS=$(python tools/eval/e9_gate.py --print_order --results_root "$RESULTS_ROOT" | cut -f1)

echo "== 2. offline metrics (one implementation for every method; resume-safe)"
for C in $CELLS; do
  OUT="$EXP/metrics/$C/metrics.csv"
  mkdir -p "$EXP/metrics/$C"
  python tools/eval/eval_metrics_offline.py --roots "$RESULTS_ROOT/$C" --out "$OUT" --device "$DEVICE" --batch 32
  N=$(python -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))" "$OUT")
  echo "   $C: $N metric rows"
done

echo "== 3. export the cells (per_checkpoint, runs, config; no images)"
for C in $CELLS; do
  case "$C" in *_NT) M=non_targeted;; *) M=targeted;; esac
  mkdir -p "$EXP/cells/$C/$M"
  cp "$RESULTS_ROOT/$C/$M/"per_checkpoint*.csv "$RESULTS_ROOT/$C/$M/"runs*.csv "$RESULTS_ROOT/$C/$M/config_single.json" \
     "$EXP/cells/$C/$M/"
done

echo "== 4. registered claims (amendment 3, section 6) and the dev-overlap recomputation"
python tools/eval/claims_e9.py
python tools/eval/claims_e9.py --no-devoverlap

echo "== 5. COMMANDS.md and MANIFEST.sha256"
{
  echo "# E9 export (prereg_build38_amendment3.txt), $(date +%Y-%m-%d)"
  echo
  echo "HEAD at finish: $(git rev-parse HEAD)"
  echo "Attack command: run_e9_parallel.sh run_attack() (B5 LGC_RC_EGC command line; only --methods and --out_dir differ)."
  echo "LGC_RC_EGC rows: paper_exports/b5a2 (tag b5a2-done-2026-10-23), not re-run."
  echo
  echo "Tool sha256:"
  sha256sum prereg_build38_amendment3.txt run_e9_parallel.sh e9_finish.sh tools/eval/e9_gate.py \
            tools/eval/claims_e9.py tools/eval/claims_registered.py tools/eval/eval_metrics_offline.py \
            lgc_bench/engine.py run_benchmark_33b.py | sed 's/^/    /'
  echo
  echo "Commands (per cell): python tools/eval/eval_metrics_offline.py --roots $RESULTS_ROOT/<cell> --out $EXP/metrics/<cell>/metrics.csv --device $DEVICE --batch 32"
  echo "Claims: python tools/eval/claims_e9.py ; python tools/eval/claims_e9.py --no-devoverlap"
} > "$EXP/COMMANDS.md"
( cd "$EXP" && find . -type f ! -name MANIFEST.sha256 | LC_ALL=C sort | xargs sha256sum > MANIFEST.sha256 )
echo "   $(wc -l < "$EXP/MANIFEST.sha256") files in $EXP/MANIFEST.sha256"

echo "== done. Send L: the claims lines printed above and this log ($LOG)."
echo "   Then commit (README step 5): git add $EXP reports/raw/E9 && git commit -m 'E9 ablation (amendment 3): export and claims'"
