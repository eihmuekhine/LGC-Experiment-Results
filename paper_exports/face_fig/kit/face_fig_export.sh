#!/usr/bin/env bash
# face_fig_export.sh - package the author-chosen figure cells as their own evidence export (L, 2026-11-03).
#
# The face export of 2026-11-01 is frozen and must not change, so the cells of the author-chosen figure go into a
# separate, additive export: $LGC/paper_exports/face_fig. No GPU and no attack: face_figure.py reads the stored
# states and the frozen metric tables, and the domain classes come from the already imported domain_rows.csv.
#
#   usage: bash kits/face_paper/face_fig_export.sh [budget]        (default 2000)
#   env:   PICK   (default "identity=425__img1.png gender=male__img244.png")
#          PAPER  (default /root/autodl-tmp/paper), FACE_ROOT, LGC_FROZEN, OUT
set -u
B=${1:-2000}
PICK=${PICK:-"identity=425__img1.png gender=male__img244.png"}
P=${PAPER:-/root/autodl-tmp/paper}
FACE=${FACE_ROOT:-/root/autodl-tmp/face}
KIT=${FACE_KIT:-$FACE/kit}
LGC=${LGC_FROZEN:-/root/autodl-tmp/LGC_frozen}
OUT=${OUT:-$FACE/paper_face_fig}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
export PYTHONUNBUFFERED=1
stop() { echo "STOP: $*"; exit 1; }

DOMAIN=$P/evidence/face/domain
[ -f "$DOMAIN/domain_rows.csv" ] || stop "no $DOMAIN/domain_rows.csv (import the face export first)"
[ "$(git -C "$LGC" rev-parse --abbrev-ref HEAD)" = "paper-runs" ] || stop "$LGC is not on paper-runs (never switch it)"
E="$LGC/paper_exports/face_fig"
[ -e "$E" ] && stop "$E exists already: move it aside before re-exporting"

PICKARGS=""
for item in $PICK; do PICKARGS="$PICKARGS --pick $item"; done

echo "== 1/2 figure cells at $B queries, pairs: $PICK"
rm -rf "$OUT/figure"
# shellcheck disable=SC2086
"$PY" "$HERE/face_figure.py" --face_root "$FACE" --kit "$KIT" --domain "$DOMAIN" --budget "$B" \
  $PICKARGS --out "$OUT/figure" | tee "$OUT/figure.log" || stop "face_figure.py failed"
[ -s "$OUT/figure/cell_metrics.csv" ] || stop "cell_metrics.csv was not written"

echo "== 2/2 export to $E"
mkdir -p "$E"
cp -r "$OUT/figure" "$E/figure"
mkdir -p "$E/kit"
cp "$HERE/face_figure.py" "$HERE/face_fig_export.sh" "$E/kit/"
printf 'budget=%s\npicks=%s\nselection=authors\ndomain_rows=evidence/face/domain/domain_rows.csv\n' \
  "$B" "$PICK" > "$E/EXPORT.txt"
( cd "$E" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256 ) || stop "manifest failed"
( cd "$E" && sha256sum -c --quiet MANIFEST.sha256 ) || stop "manifest does not verify"
echo "wrote $(wc -l < "$E/MANIFEST.sha256") files to $E"
echo
echo "Next, in $LGC (never switch its branch):"
echo "    git -C $LGC add paper_exports/face_fig"
echo "    git -C $LGC commit -m 'Face figure cells at $B queries (author-chosen pairs)'"
echo "    git -C $LGC tag face-fig-$(date +%Y-%m-%d)"
echo "    git -C $LGC push origin paper-runs --tags"
echo "Then send L the tag name and the first lines of $E/figure/cell_metrics.csv."
