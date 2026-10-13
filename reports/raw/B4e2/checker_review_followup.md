# Checker review — phase B4e2 follow-up (review mode, skill lgc-preflight §6)

Independent review session: this session contains **no author work** (fresh reviewer session; the author's
code/report were only read, never edited). Worktree `/root/autodl-tmp/LGC_frozen`, base `aee5027`, HEAD
`a67ee7e` (parent verified: `git rev-parse a67ee7e^` = `aee5027f33c2c210722fffa9097170e1adada845`).
Spec `reports/raw/B4e2/prompt_followup.txt` present, sha256
`4dd0b63642e6b5af1c4459e274a9f5e99fd54d1b987808c6acb3d869039a1f44` (recomputed, matches report §1).
Findings source `reports/raw/B4e2/checker_review.md` sha256
`822a7bbedbbc02fed33428a74caf479ddee44689b7e0eeb5056d88a3bc6f923d` (recomputed, matches report §7).
Contract v1.8 sha256 `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` and engine sha256
`835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` both recomputed — unchanged (frozen
engine and contract untouched). Commit `a67ee7e` touches exactly the 9 allowed files (`adapters/run_cgba.py`,
`adapters/diag/diag_b4e2_t6_resume.py`, `reports/*`); no forbidden file in `git diff aee5027..HEAD`.
Pre-existing worktree dirt (`.opencode/skills/lgc-preflight/SKILL.md`, `docs/PREFLIGHT_RUNBOOK.md`,
`reports/HUMAN_SIGNOFF_B5_TEMPLATE.md`) is NOT part of the commit: its diff vs `aee5027` equals its diff vs
HEAD, and mtimes are 2026-10-13 21:22–21:23, i.e. before the B4e2 phase began — so the report's allowed-list
compliance claim stands.

## Verdicts per F-item (my own evidence; cheap commands rerun)

### F1 — resume-safe `clip_events_182` merge — **PASS** (one residual caveat, see below)

- Code matches the prompt's design exactly: `_existing_stems` snapshots the stems in
  `runs_<method>.csv` before `C.run` (`adapters/run_cgba.py:191-202`, called at `:255`), and
  `_write_clip_events(..., pre_existing)` assigns `_CLIP_EVENTS` entries only to rows whose stem is not in
  the snapshot, in order (`:222-226`). Path convention equals `C.run`'s (`adapters/_common.py:538-541`
  `<out_dir>/<mode>/runs_<method>.csv`); `--mode` is `required=True` in `C.run` (`_common.py:502`) so the peek
  default cannot diverge. Counter semantics (`run_cgba.py:150-153`, `:188`) and all attack code are untouched
  by the diff; `diag_b4e_t1_clip_source.py` untouched (not in the diff).
- **F1(a) rerun comparison (my own script, both dirs):** `per_checkpoint_CGBA.csv` 24/24 rows, **0 field
  diffs across ALL columns** (not just the 5 numeric ones); npz `delta` **24/24 bit-equal** (plus 3 `x0.npy`
  bit-equal = 27/27 files); `clip_events_182` = `['0','0','0']` in
  `Results/b4e2_f1_cgba_resnet50/non_targeted/runs_CGBA.csv` for the three images. F1 run env records the
  study GPU `NVIDIA GeForce RTX 5090`, D12 flags `matmul_tf32=False, cudnn_tf32=False, deterministic=True,
  benchmark=False` (set at `_common.py:52-55`), repo `/root/autodl-tmp/baselines/CGBA` commit `73d3e9c9`
  (matches `git -C` of the clone), split sha `e08a069e…`, variant `fixed` — all consistent with the reported
  command. Wall times 25.35/24.48/24.15 s match `F1.log` (25.4/24.5/24.2) and the report's "≈74 s".
- **F1(b) sandbox test rerun by me:** `python3 adapters/diag/diag_b4e2_t6_resume.py` → 5/5 PASS
  (A untouched/empty, B=7, C=13, snapshot={A}, column name preserved), exit 0. The test implements the
  prompt's F1(b) scenario faithfully (pre-existing ok row without the column, two new rows, counts in
  processing order).
- **Residual caveat (unreported, demonstrated by me):** the merge is keyed on stem only, and
  `adapters/_common.py:425` `done_images` still treats only `status in ("ok","budget")` as done — the very
  divergence the report documents as deviation §6. On an **error-row resume** the errored image is
  re-attacked and a SECOND row with the same stem is appended; the merge then skips that new row (stem is in
  `pre_existing`) and shifts its count onto the next new image. My CPU sandbox (pre-existing error row X;
  new rows X,Y; counts [5,9]) yields X-new = `''` and Y = `5` — misaligned. Fix (one line): snapshot only
  pre-existing rows with `status in ("ok","budget")` as "existing" (or key on `(stem, occurrence)`).
  This does not fail the prompt's acceptance criteria (the prompt mandates the snapshot design and the
  done-image case is fixed), but it is a B5 risk exactly in the resume-after-error scenario deviation §6
  says CGBA will hit.

### F2 — `reports/B4e2_close.md` edits — **PASS** (all five verified against the checker findings)

- (i) The unsupported "Printed counts: …" sentence is replaced by the evidence path
  `Results/b4e2_t6_cgba_resnet50/non_targeted/runs_CGBA.csv` column `clip_events_182` (values 0,0,0) — I read
  that CSV myself: `0,0,0` for `ILSVRC2012_val_00007450/_00015433/_00025151`. Checker finding 1 resolved.
- (ii) Deviation §6 is present (close report §4 item 6) and code-accurate on both sides:
  `adapters/_common.py:421-425` still filters `status in ("ok","budget")` (read by me), and
  `ttba_github/ttba_common.py:83` `done_images` has no status filter (read by me). Checker finding 2 resolved
  (as a documented user decision, contract v1.9).
- (iii) `reports/raw/B4e2/T3_gate.md` is committed and **byte-identical** to `/tmp/b4e2_gate.md`
  (both sha256 `031fbe9f1dd15d48…`; `diff` empty). `reports/human_preflight_b4e2.md` is committed (656 lines,
  in the a67ee7e diff); FLAG count verified by me: 6 `**FLAG**` table rows (E10/E11 env_frame gaps) + 1
  "FLAG: more than one value across env files" (GPU name) = **7**, matching "FLAG lines: 7" at its line 4.
  Checker finding 4 resolved.
- (iv) The §5 log table lists 16 logs; I recomputed **all 16 sha256s — every one matches** (T1_A `ea4b1fa7…`
  … T8 `f142c3dd…`). The HEAD note is present in the header and the §2 environment row
  ("base `e9ff7da` … committed at the new HEAD `aee5027` → follow-up commit — not `e9ff7da`"). Checker
  finding 5 resolved. (The close report's own §5 entry `run_cgba.py = 99f5c11f…` was correct at `aee5027` —
  verified via `git show aee5027:adapters/run_cgba.py | sha256sum`; the current file is `d3c83758…` after F1,
  which the follow-up report §7 records correctly.)
- (v) The CGBA_D8only constant-0 note is present in the T6 section and is code-true: only
  `variant == "fixed"` overrides `SemiCircular_boundary_search` (`run_cgba.py:138`), the counter is
  initialised at `:119` and never incremented on the d8only path; the E17 NaN-share is deferred to B4g/B5.

### F3 — re-verify — **PASS** (every command rerun by me)

| command (rerun by me, OMP_NUM_THREADS=8) | my result |
|---|---|
| `python3 adapters/diag/diag_b4e_t1_clip_source.py` | PASS (clamp = only value-changing edit; d8only inherits official), exit 0 |
| `python3 adapters/diag/diag_b4e_t5_done_images.py` | PASS 5/5, exit 0 |
| `python3 adapters/diag/diag_b4e2_t6_resume.py` | PASS 5/5, exit 0 |
| `python3 tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21), exit 0 |
| `python3 tools/eval/selftest_evalkit.py` | ALL PASS (20/20), exit 0 |
| `check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --roots $(cat reports/B4e2_smoke_roots.txt) --report /tmp/checker_followup_gate.md` | **exit 0**, I01–I12 + DISK all PASS |

- My gate report `/tmp/checker_followup_gate.md` is **byte-identical** to the committed
  `reports/raw/B4e2/T3_gate.md`, to the author's `/tmp/b4e2_followup_gate.md` and to the original
  `/tmp/b4e2_gate.md` (all sha256 `031fbe9f1dd15d48…`) — the author's F3 rerun is genuine and reproducible.
- Tracked `reports/contract_status.md` untouched (status written next to my /tmp report;
  `git status` clean for it).

## Claims in `reports/B4e2_followup.md` that the evidence does not support

1. §8 point 1 — "so a resumed CGBA run no longer misaligns the column" is **over-qualified**: it is true for
   resumes over done (`ok`/`budget`) images, but false for a resume that re-attacks a previously **errored**
   image (the `_common.py:425` policy deviation §6 itself documents): my sandbox shows the re-attacked
   image's new row gets `''` and its count lands on the next new row. Fix: qualify the sentence ("resumes
   over done images") or apply the one-line snapshot fix above; either way record the residual case.
2. (Format, skill lgc-preflight §5 item 4) The report has **no "Deviations from the prompt" section** (not
   even "none"). No substantive deviation was found by me, so the section would be empty — but it is
   required. Fix: add the (empty) section.

No other unsupported claim found: every sha256 in §7 and in the close report's 16-log table, the F1(a)
identity claims (24/24 rows, 0 diffs, 24 npz bit-equal, clip 0,0,0), the F1(b) 5/5, all six F3 results, the
environment block (RTX 5090 / driver 580.105.08 / (12,0) / 170 SMs / torch 2.7.1+cu128 / D12 flags via the
run env file), the "≈74 s" GPU time, the allowed-list compliance, and "B5 was not started" (no B5 result
dirs or reports exist) all reproduce.

## Deviations from my review protocol

None. No files edited except this review file; no git checkout/reset/commit/add; `check_contract.py` run only
with `--report /tmp/checker_followup_gate.md`; all verification commands CPU-only and run with
`OMP_NUM_THREADS=8`; the F1 edge-case demo ran as an inline heredoc against a temp directory (no repo file
created or modified). Note (not a deviation): an untracked `reports/raw/B4e2/.ipynb_checkpoints/` directory
(Jupyter dirt) exists in the worktree; harmless, but it should not be committed.

## Summary (5 lines)

1. F1 PASS: the stem-keyed merge matches the prompt's design; my own comparison gives 24/24 identical
   per_checkpoint rows, 24 bit-equal npz deltas, clip_events_182 = 0,0,0, and the sandbox test passes 5/5.
2. Residual F1 caveat: after an error-row resume (the `_common.py` policy of deviation §6) the merge still
   misaligns — demonstrated; one-line fix available; the report's §8 "no longer misaligns" claim needs
   qualifying.
3. F2 PASS: all five close-report edits verified against the findings — evidence path, deviation §6
   (code-accurate on both sides), committed T3_gate.md (byte-identical) and human_preflight (FLAG=7), all 16
   log sha256s correct, HEAD note and CGBA_D8only constant-0 note present.
4. F3 PASS: all three diags, tests_contract (21/21), selftest (20/20) and the smoke gate (exit 0, gate report
   byte-identical to the committed one) rerun by me; tracked contract_status.md untouched.
5. Only two report defects: the over-qualified §8 resume claim and the missing (empty) Deviations section;
   frozen engine, contract v1.8 and prereg untouched; B5 not started.
