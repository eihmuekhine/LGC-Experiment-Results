#!/bin/bash
# B5-fix launcher tests (CPU only).  Run from the worktree root:
#     CUDA_VISIBLE_DEVICES="" bash tools/eval/tests_b5/run_tests.sh
# Writes evidence under reports/raw/B5-fix/ and prints a PASS/FAIL summary.
set -u
cd "$(dirname "$0")/../../.." || exit 1
REPO="$PWD"
OUT="$REPO/reports/raw/B5-fix"
mkdir -p "$OUT"
PASS=0; FAIL=0
ok()   { echo "   PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "   FAIL  $1"; FAIL=$((FAIL+1)); }

echo "=== B5-fix launcher tests ==="

# --- 1. syntax ---
echo "[1] bash -n run_b5_parallel.sh"
if bash -n run_b5_parallel.sh 2>"$OUT/bash_n.err"; then ok "bash -n"; else bad "bash -n"; fi

# --- 2. dry_run with NGPU = 1, 8, 10 ---
echo "[2] --dry_run NGPU=1/8/10: 52 unique commands, unique outdirs, E17 paths, GPU slot < NGPU"
for N in 1 8 10; do
  NGPU=$N ./run_b5_parallel.sh --dry_run > "$OUT/dry_run_ngpu$N.log" 2>&1
  python3 - "$OUT/dry_run_ngpu$N.log" "$N" <<'PY'
import sys, re
log = sys.argv[1]; N = int(sys.argv[2])
txt = open(log).read()
cmds = [l for l in txt.splitlines() if l.startswith('python ')]
slots = [int(m) for m in re.findall(r'^# gpu_slot: (\d+)', txt, flags=re.M)]
outdirs = re.findall(r'--out_dir (\S+)', txt)
e17 = [l for l in cmds if 'd8only' in l]
e17_ok = all(re.search(r'--out_dir Results/appendix_cgba_d8only_\w+ ', l) for l in e17)
uniq_cmds = len(set(cmds)) == 52 and len(cmds) == 52
uniq_out = len(set(outdirs)) == 52 and len(outdirs) == 52
slots_ok = all(0 <= s < N for s in slots) and len(slots) == 52
e17_ok = e17_ok and len(e17) == 4
ok = uniq_cmds and uniq_out and e17_ok and slots_ok
print(f"NGPU={N} cmds={len(cmds)} uniq_cmds={uniq_cmds} uniq_out={uniq_out} e17_ok={e17_ok} slots_ok={slots_ok}")
sys.exit(0 if ok else 1)
PY
  if [ $? -eq 0 ]; then ok "dry_run NGPU=$N"; else bad "dry_run NGPU=$N"; fi
done

# --- 3. fresh-directory refusal ---
echo "[3] fresh-directory refusal (RESULTS_ROOT with one existing output dir)"
FRESH=$(mktemp -d /tmp/b5fresh.XXXXXX)
mkdir -p "$FRESH/b5_main_TtBA_resnet50_NT"
NGPU=1 RESULTS_ROOT="$FRESH" ./run_b5_parallel.sh > "$OUT/fresh_refusal.log" 2>&1
rc=$?
if [ "$rc" -eq 2 ] && grep -q "already exists" "$OUT/fresh_refusal.log"; then ok "fresh-directory refusal (exit 2)"; else bad "fresh-directory refusal (exit $rc)"; fi

# --- 4. tmux refusal ---
echo "[4] tmux refusal (TMUX unset)"
TMUX_SAVE="${TMUX:-}"
unset TMUX
NGPU=1 RESULTS_ROOT="$(mktemp -d /tmp/b5tmux.XXXXXX)" ./run_b5_parallel.sh > "$OUT/tmux_refusal.log" 2>&1
rc=$?
[ -n "$TMUX_SAVE" ] && export TMUX="$TMUX_SAVE"
if [ "$rc" -eq 2 ] && grep -q "not inside tmux" "$OUT/tmux_refusal.log"; then ok "tmux refusal (exit 2)"; else bad "tmux refusal (exit $rc)"; fi

# --- 5. NGPU mismatch refusal ---
echo "[5] NGPU mismatch refusal (NGPU=8 vs nvidia-smi)"
NGPU=8 RESULTS_ROOT="$(mktemp -d /tmp/b5ngpu.XXXXXX)" ./run_b5_parallel.sh > "$OUT/ngpu_refusal.log" 2>&1
rc=$?
if [ "$rc" -eq 2 ] && grep -q "NGPU=" "$OUT/ngpu_refusal.log"; then ok "NGPU mismatch refusal (exit 2)"; else bad "NGPU mismatch refusal (exit $rc)"; fi

# --- 6. environment guard ---
echo "[6] environment guard"
WRONG=$(mktemp /tmp/b5env_wrong.XXXXXX)
printf 'torch 2.8.0+cu128 tv 0.23.0+cu128\n' > "$WRONG"
NGPU=1 ENV_FILE="$WRONG" ./run_b5_parallel.sh > "$OUT/envguard_wrong.log" 2>&1
rc=$?
if [ "$rc" -eq 3 ] && grep -q "environment guard FAILED" "$OUT/envguard_wrong.log"; then ok "env guard refuses wrong torch/tv (exit 3)"; else bad "env guard wrong torch/tv (exit $rc)"; fi

# real file passes the guard: non-dry-run with a fresh dir -> the guard must NOT be the exit code 3;
# it must reach the fresh-directory refusal (exit 2) after the guard passed.
REAL=$(mktemp -d /tmp/b5env_real.XXXXXX)
mkdir -p "$REAL/b5_main_TtBA_resnet50_NT"
NGPU=1 ENV_FILE=env_build38.txt RESULTS_ROOT="$REAL" ./run_b5_parallel.sh > "$OUT/envguard_real.log" 2>&1
rc=$?
if [ "$rc" -eq 2 ] && ! grep -q "environment guard FAILED" "$OUT/envguard_real.log"; then ok "env guard passes the real env_build38.txt (reaches fresh-dir exit 2)"; else bad "env guard real file (exit $rc)"; fi

echo
echo "SUMMARY: PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
