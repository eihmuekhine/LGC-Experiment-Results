# Independent review — B5-fix fix round 1 (R1–R7) + fix round 1b (test_gates_monitor.py pin)

Reviewer session: fresh (z-ai/glm-5.3, agent `reviewer`). **This session had not seen any of the
author's work before this review** — the first file read was the review prompt itself
(`reports/raw/B5-fix/fix1_fix1b_review_prompt.txt`, sha256
`a730cc37b93591339326fbc5e787c9da7cce81a3da47bedecd79b271cf01c05f`, printed before anything else).
Reading order followed the prompt: fix1/fix1b specs → full `git diff e8f20b9` → raw evidence →
only then the author's §12/§13.

## Environment block

- HEAD: `6d36c0e0053578ed29b67eef0f4d3be3a74a55c6` (branch `paper-runs`; matches prompt line 3)
- `git diff --stat e8f20b9..HEAD` → `docs/LESSONS.md | 4 +++-` only — confirmed: HEAD's only committed
  change over e8f20b9 is docs/LESSONS.md (L26–L27).
- contract sha256: `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c` (v1.9-rev7)
- review prompt sha256: `a730cc37b93591339326fbc5e787c9da7cce81a3da47bedecd79b271cf01c05f`
- fix1 prompt sha256: `53dbcf1f4aa2204d548f556c08bf430db5853dfd46d055bf8258ad098eaebc9a` (matches §12)
- fix1b prompt sha256: `3910f0a1349a33aacd04a6e378d9d4d6e5cadec0245052bc549fed9325b1c3c5` (matches §13)
- GPU: none started (CPU only). Working tree: 5 tracked modifications
  (`run_b5_parallel.sh`, `tools/eval/b5_gate_cell.py`, `tools/eval/b5_monitor.py`,
  `tools/eval/tests_b5/test_gates_monitor.py`, `reports/B5_fix.md`) + untracked
  `tools/eval/tests_b5/sim/`, `reports/raw/B5-fix/*`, `reports/stop/*`, root session-log copies,
  `reports/raw/B5-pre/`.
- Active LESSONS rows applying to this phase: L04, L05, L06, L10, L14, L24, L25, L26, L27
  (L14/L27 are directly instantiated by this review: the fix1b prompt's stale 8/8 count).

## My own re-run evidence (all commands re-executed, none trusted from the report)

| command | my result | expected |
|---|---|---|
| `bash -n run_b5_parallel.sh` | OK | OK |
| `bash tools/eval/tests_b5/sim/run_sim.sh` | `21 PASS, 0 FAIL` / `ALL PASS` (S0–S7 all PASS) | 21 PASS, 0 FAIL |
| `for N in 1 8 10; do NGPU=$N bash run_b5_parallel.sh --dry_run 2>/dev/null \| grep '^python ' \| sha256sum; done` | `9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6` ×3 | same ×3 |
| `CUDA_VISIBLE_DEVICES="" bash tools/eval/tests_b5/run_tests.sh` | `SUMMARY: PASS=9 FAIL=0` | PASS=9 FAIL=0 |
| `CUDA_VISIBLE_DEVICES="" python tools/eval/tests_b5/test_gates_monitor.py` | `ALL PASS (9/9)`, `old=52 new=52` | ALL PASS |
| `sha256sum` of all changed files | launcher `50d8941c…`, gate `e90a71bb…`, monitor `43e1f1c7…`, test `fcac1def…`, agent_log `76f5bad8…`, sim files == kit `SHA256SUMS` | match §12/§13 |
| `git show bb7fa0d:run_b5_parallel.sh \| sha256sum` | `c20cc34990e9ed2cf5fb1c8a18a79517d67617ba98884c13425461e3c21d17d2` | c20cc349… (pre-v2) |

## PASS/FAIL table

| entry | verdict | evidence (my own) |
|---|---|---|
| R1 resume joblog + attack-running refusal | PASS | sim S3a/S3b/S3c/S4 PASS; diff: `JOBLOG` block :497–501, `--joblog "$JOBLOG"` :532, summary reads `$JOBLOG`; `attack_running()` :38–42 anchored `^python[[:space:]]+(5 entry points)`, resume guard exit 2 :77–80 |
| R2 STOP code rule | PASS | sim S1d/S1e/S1f/S6c PASS; diff :459 `[ "$rc" -ge 128 ] \|\| grep -qE 'CUDA out of memory\|CUDA error\|cudaError\|out of memory\|No space left on device\|Killed'` (case-sensitive, no `-i`); `b5_gate_cell.py:334` missing-domain-CSV → `E-OTHER` |
| R3 STOP bundles never overwrite | PASS | sim S6d PASS; `_2,_3…` loops at launcher :149–153 and `b5_gate_cell.py` `write_stop_bundle` :246–249 |
| R4 gate STOP field 3 = signature + last 50 log lines | PASS | sim S6b PASS; `run_gates` :388–397 builds `evidence = sig + last 50 lines of <log_dir>/<cell>.log`; `write_stop_bundle` writes it as field 3 |
| R5 monitor ≤1 bundle per (cell, code) | PASS | sim S7 PASS (got 1); fired keys in `<log_dir>/monitor/state.json` (`stops` list), `_write_status` untouched → status files still list every open condition |
| R6 joblog summary cell names + mktemp | PASS | sim S1i PASS, S0/S2 PASS; summary awk :536–551 parses `run_cell '<cell>…'` Command field; `ORDER_FILE`/`CELLS_FILE` = `mktemp` + `trap … EXIT` :66–68 |
| R7 report §7/§8 only | PASS | `git diff e8f20b9 -- reports/B5_fix.md` = exactly 3 hunks: §7 total paragraph (16 rows, 9 bash + 9 python itemised), §8 lower-bound labels + LPT range, appended §12/§13. No other earlier-section edit. |
| fix1b pin `test_gates_monitor.py` HEAD→bb7fa0d | PASS | my run `ALL PASS (9/9)`; single-line diff at :111; `git cat-file -e bb7fa0d^{commit}` OK; `git show bb7fa0d:run_b5_parallel.sh \| head -1` → `#!/bin/bash`; sha256 of that file = `c20cc349…` |

## Judgment call (a): the bb7fa0d pin is correct and masks nothing — CONFIRMED

1. `git show bb7fa0d:run_b5_parallel.sh | sha256sum` = `c20cc34990e9ed2c…` — i.e. **c20cc349 is the
   sha256 prefix of exactly the pre-v2 launcher held at bb7fa0d**, the file the fix1b prompt says the
   line ranges were written for. Lines 10–63 of that file hold `CELLS=( … )`; lines 95–327 hold the
   case patterns (checked directly).
2. With the pin, `_gen_old_commands()` reconstructs `old=52` and the test asserts the current
   launcher's 52 commands are byte-identical except exactly the 4 E17 `--out_dir` cells (my run:
   PASS on all three identity checks). The E17 allowance is pre-existing test logic (lines 156–173,
   unchanged context in the diff), not something this fix round introduced.
3. Independent, stronger cross-check: my `--dry_run` sha256 = `9b24fb04…` for NGPU=1/8/10 (×3) and
   the sim harness's frozen `DRY_SHA_E8F20B9=9b24fb04…` (run_sim.sh:20) — S0 PASS in my own sim run.
   So current commands == e8f20b9 (v2) == bb7fa0d (pre-v2) modulo the H4-sanctioned E17 `--out_dir`.
   The pin restores the test's intended reference; it hides no real difference.
4. 9/9 vs the prompt's stale 8/8: verified from the code, not from the author's reasoning.
   `test_command_identity` records "52 cells both old and new" always; on `set(old) != set(new)` it
   records "same cell names" FAIL and **returns** (test_gates_monitor.py:150–152), skipping the two
   E17 checks → pre-fix (old=0) 8 checks total, "6/8" (matches the 0631 STOP bundle's recorded output);
   post-fix (sets equal) 3 identity checks → 9 total. The only edit in the file is :111 (diff), so
   this is the early-return short-circuit, **not** a hidden behavior change.

## Judgment call (b): R1–R7 stayed in scope — CONFIRMED (line-by-line diff, not the author's summary)

`git diff e8f20b9 -- . ':!reports'` = 253 lines, 5 files, read hunk by hunk:
- `run_b5_parallel.sh` (10 hunks): all map to R1 (`attack_running`, resume guard, `JOBLOG` selection,
  summary source), R2 (:459 STOP rule), R3 (:149–153 loop), R6 (mktemp :66–68, summary awk).
  **`run_attack` is not in the diff** → attack commands unchanged; queue order comes from
  `b5_gate_cell.py --print_order/--print_queue`, untouched.
- `tools/eval/b5_gate_cell.py` (3 hunks): R3 (:246–249), R2 (:334 code label only — the domain-gate
  verdict/threshold logic is unchanged context), R4 (:388–397 evidence). No gate rule or threshold
  changed.
- `tools/eval/b5_monitor.py` (3 hunks): R5 only (`load_state` default, `_write_stops` fired-key skip,
  `save_state`); `_write_status` untouched.
- `tools/eval/tests_b5/test_gates_monitor.py` (1 line): the fix1b pin.
- `docs/LESSONS.md`: the committed HEAD change, allowed by the fix1 prompt.
- `reports/B5_fix.md`: R7-scoped §7/§8 edits + appended §12/§13 only (3 hunks).
- Sim harness/stubs byte-identical to `/root/autodl-tmp/b5a2_kit_v2_1/SHA256SUMS` (verified).
No attack command, queue order, gate rule or threshold change found anywhere in the diff.

## Claims in §12–§13 that my evidence does not support (or cannot support)

1. §13 deviation 3 is stale vs the worktree: it lists only `649N.md` as the extra untracked root file,
   but `717N.md` (mtime 2026-10-19 07:17, the resume session's own saved log) is also present and
   unacknowledged. No code impact.
2. The fix1b edit was found **pre-applied** (§13 deviation 1; agent_log "edit pre-applied"; 717N.md
   transcript confirms the resume session discovered it already present). The session/event that
   applied it left no `agent_log.md` row (X5 gap). If H applied it manually this is no violation, but
   §13 does not say who did.
3. §12/§13 sha256 literals for `reports/B5_fix.md` (`975e8c70…`, `1f9a5f9b…`) are historical /
   self-referential and unverifiable (current file sha256 `dc80d1ca…`); §13 flags the self-reference
   itself, so this is a noted limitation, not a false claim.
4. Minor line-range imprecisions in §12 (e.g. the launcher `stop_bundle` anti-overwrite loop is
   :149–153, not ":149–152"; `_write_stops` call is :308, `save_state` :309). Immaterial.
5. §12's pre-fix `test_gates_monitor.py -> FAILED (6/8)` is the author's recorded output; I could not
   re-run the pre-fix state without editing the file (forbidden), but the 0631 STOP bundle records it
   and the code's early-return confirms the 8-check count. Consistent with all evidence.

## Deviations from the review prompt

None. Every listed command was re-run; `tools/eval/check_contract.py` not run (optional per prompt).
Side effect of my mandated re-runs: `run_tests.sh` rewrote its own evidence files under
`reports/raw/B5-fix/` (`bash_n.err`, `dry_run_ngpu*.log`, mtime 07:23) — intended per the fix1 prompt.
No file edited except this review file; no git checkout/reset/commit/add/stash/tag.

## Verdict

All R1–R7 acceptance criteria and the fix1b pin pass in my own re-runs; both judgment calls confirmed
with independent evidence. The findings above are reporting/provenance notes, none of which undermine
an acceptance criterion.

REVIEW_VERDICT: PASS-WITH-NOTES