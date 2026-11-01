#!/usr/bin/env bash
# Face results for the paper (L, 2026-11-01): domain classes, rule-based figure cells, export into the evidence repo.
# Run on the container after "MAIN: PASS" and "METRICS: PASS". Reads the face store; writes only
#   $OUT (default /root/autodl-tmp/face/paper_face)  and  $LGC/paper_exports/face (new files only; no git command).
# usage: bash face_export.sh [budget]      (budget of the figure, default 1000)
set -u
B=${1:-1000}
FACE=${FACE_ROOT:-/root/autodl-tmp/face}
KIT=${FACE_KIT:-$FACE/kit}
LGC=${LGC_FROZEN:-/root/autodl-tmp/LGC_frozen}
OUT=${OUT:-$FACE/paper_face}
HERE=$(cd "$(dirname "$0")" && pwd)
DEVICE=${DEVICE:-cuda:0}
stop() { echo "STOP: $*"; exit 1; }
[ -f "$FACE/metrics/face_summary.md" ] && [ -f "$FACE/metrics/face_b5_summary.md" ] || stop "metrics/face_summary.md or face_b5_summary.md missing"
[ -f "$FACE/store_main/store_index.csv" ] || stop "store_main/store_index.csv missing"
[ "$(git -C "$LGC" rev-parse --abbrev-ref HEAD)" = "paper-runs" ] || stop "$LGC is not on paper-runs (never switch it)"
[ -e "$LGC/paper_exports/face" ] && stop "$LGC/paper_exports/face exists already (move it aside first)"
mkdir -p "$OUT"
python "$HERE/face_domain.py" --face_root "$FACE" --kit "$KIT" --out "$OUT/domain" --device "$DEVICE" | tee "$OUT/domain.log" \
  || stop "face_domain.py failed"
python "$HERE/face_figure.py" --face_root "$FACE" --kit "$KIT" --domain "$OUT/domain" --budget "$B" --out "$OUT/figure" \
  | tee "$OUT/figure.log" || stop "face_figure.py failed"
E="$LGC/paper_exports/face"
mkdir -p "$E/figure" "$E/domain" "$E/metrics" "$E/kit"
cp "$FACE/metrics/face_summary.md" "$FACE/metrics/face_b5_summary.md" "$E/metrics/"
cp "$OUT/domain/domain_summary.json" "$OUT/domain/domain_summary.md" "$OUT/domain/domain_rows.csv" "$E/domain/"
cp "$OUT/figure/"*.png "$OUT/figure/selection.json" "$E/figure/"
cp "$HERE/face_domain.py" "$HERE/face_figure.py" "$HERE/face_export.sh" "$E/kit/"
(cd "$E" && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256)
echo "== exported to $E ($(wc -l < "$E/MANIFEST.sha256") files)"
cat "$OUT/domain/domain_summary.md"
echo "== next (H): cd $LGC && git add paper_exports/face && git commit -m 'Face export for the paper' && git tag face-done-\$(date +%F) && git push origin paper-runs --tags"
