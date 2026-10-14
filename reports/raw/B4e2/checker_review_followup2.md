# Checker review — phase B4e2 follow-up-2 (review mode, skill lgc-preflight §6)

Independent review session: this session contains **no author work** (fresh reviewer session; the author's
code/report were only read, never edited). Worktree `/root/autodl-tmp/LGC_frozen`, base `a67ee7e`, HEAD
`e2b7a2e` (two commits: `4a7cd28` = G1+G2+G3, `e2b7a2e` = post-edit sha256 fix of the `B4e2_followup.md` row in
the follow-up-2 §8 table; both verified by `git log --format="%H %s" a67ee7e..HEAD`).

- Spec `reports/raw/B4e2/prompt_followup2.txt` present, sha256
  `1d7525b95ae0197160b7cb1c35a4a1c8b1a0e8dc8e95e690379e3380e4a06271` (recomputed, matches report §1 and §8).
- Findings source `reports/raw/B4e2/checker_review_followup.md` sha256
  `403d9d219155de48047c1222b9a374be24a4f68481f529782c39cb609b510172` (recomputed, matches report §8).
- Contract v1.8 sha256 `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` (`version: "1.8"` read
  at `contract/paper_contract.yaml:9`) and engine sha256
  `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` both recomputed — unchanged (frozen engine
  and contract untouched).
- Allowed-list compliance: `git diff a67ee7e..HEAD --name-only` touches exactly 7 files
  (`adapters/run_cgba.py`, `adapters/diag/diag_b4e2_t6_resume.py`, `reports/B4e2_followup.md`,
  `reports/B4e2_followup2.md`, `reports/raw/B4e2/{prompt_followup2.txt,checker_review_followup.md,G1a.log}`) —
  all inside the prompt's allowed set; no forbidden file in the diff. Pre-existing worktree dirt
  (`.opencode/skills/lgc-preflight/SKILL.md`, `docs/PREFLIGHT_RUNBOOK.md`, `reports/HUMAN_SIGNOFF_B5_TEMPLATE.md`)
  is NOT part of the commits: per file, `sha256(git diff a67ee7e -- f) == sha256(git diff HEAD -- f)` — unchanged
  during the phase.

## Verdicts per G-item (my own evidence; cheap commands rerun)

### G1 — error-row-resume fix — **PASS**

- Code matches the required behaviour: `_existing_stems` (stem set) is replaced by `_pre_existing_count`
  (row count, `adapters/run_cgba.py:191-206`, called at `:260`), and `_write_clip_events(..., pre_existing)`
  assigns `_CLIP_EVENTS` entries to rows at index `>= pre_existing` in order (`:228-231`, called at `:264`).
  Column name `clip_events_182` (`:226`, `:231`), counter semantics (`:119`, `:152`, `:188` — untouched by the
  diff) and the done-image skip (`adapters/_common.py:562-563` + `:425`) are unchanged.
- Core assumption verified by reading `C.run`: exactly one runs-CSV row is appended per `run_official` call —
  error row at `adapters/_common.py:584`, ok/budget row at `:598` — in processing order, and `run_official`
  appends exactly one `_CLIP_EVENTS` entry per call in a `finally` block (`run_cgba.py:186-188`), so row position
  corresponds 1:1 to entry position. `append_csv` (`_common.py:410-418`) writes header and row together, so a
  header-only CSV (empty-rows edge in `_write_clip_events`) cannot occur through `C.run`.
- **G1(b) rerun by me:** `OMP_NUM_THREADS=8 python3 adapters/diag/diag_b4e2_t6_resume.py` → scenario (b) 5/5
  PASS (old error X stays empty, **X-new = 5, Y = 9**, snapshot count 1, column name preserved), exit 0 — the
  reviewer's demo from `checker_review_followup.md` finding 1 now passes. The scenario is a genuine regression
  test: on the old stem-keyed code it yields X-new = `''` and Y = `5`.
- **G1(c) rerun by me:** same run → scenario (c) 5/5 PASS (**A untouched/empty, B = 7, C = 13**), exit 0.
- **My own extra sandbox (beyond the author's diag):** pre-existing A(ok, clip=3), X1(error), X2(error),
  B(budget, clip=0); this run appends X1(ok), X2(error), C(budget); counts [5,9,11] → A keeps `3`, B keeps `0`,
  old error rows keep `''`, X1-new = 5, X2-new = 9, C = 11, `pre_existing = 4`. PASS — pre-existing values are
  preserved and multiple/re-errored rows stay aligned.
- **G1(a) artifacts verified by me (CPU comparison of the author's throwaway dir
  `Results/b4e2_g1a_cgba_resnet50/non_targeted` vs `Results/b4e2_smoke_cgba_resnet50/non_targeted`):**
  `per_checkpoint_CGBA.csv` **24/24 rows, 0 field diffs across ALL columns**; npz `delta` **24/24 bit-equal**
  (3 images × 8 checkpoints) plus 3/3 `x0.npy` bit-equal (27/27 files); `clip_events_182` = `['0','0','0']` in
  `runs_CGBA.csv` (3 rows, all `budget`). Wall times 26.5/24.66/24.02 s (report's 26.5/24.7/24.0 = the log's
  `.1f` prints; sum 75.18 s ≈ "≈75 s"). `G1a.log` L2@10000 values (6.449769496917725 / 6.3317341804504395 /
  2.6056599617004395) match the per_checkpoint CSV exactly.
- G1(a) provenance: `env_CGBA.json` records `gpu = NVIDIA GeForce RTX 5090`, D12 flags
  `precision = {cuda_matmul_allow_tf32: False, cudnn_allow_tf32: False, cudnn_deterministic: True,
  cudnn_benchmark: False}`, `adapter_sha256 = c2779755…` (the POST-G1 adapter — the re-run used the fixed code),
  `repo_commit = 73d3e9c94423b1503053258494a54ad4d2892efa` (matches `git -C /root/autodl-tmp/baselines/CGBA
  log --oneline -1`; clone clean except `__pycache__`), `split_csv_sha256 = e08a069e…`, checkpoints
  100…10000, `max_queries = 10000`, `cgba_variant = fixed`. The smoke baseline was NOT regenerated: its files
  have mtime 2026-10-14 00:36 (before the follow-up-2 work; G1a dir 03:43–03:44) and its env records the
  pre-F1 adapter sha `5f7fac69…`, so the bit-identity claim spans three adapter versions
  (smoke `5f7fac69` → F1 `d3c83758` → G1 `c2779755`).
- The author's option-1-insufficiency claim (report §3) is correct: with a stem snapshot filtered to
  `ok`/`budget`, the old error row's stem is NOT in the snapshot, so the old merge loop would assign it the
  first count (old error X = 5, X-new = 9, Y = `''`) — option 1 alone fails acceptance criterion (b).
- The "would need a forbidden `_common.py` edit" justification is accurate: `run_official(tracker, x0, y_src,
  x_tgt, y_tgt, params)` (`_common.py:575`) carries no stem and `HardLabelTracker`/`make_tracker`
  (`run_cgba.py:82-85`) store none, so the prompt's option 2 as literally worded ((stem, count) pairs collected
  in `run_official`) is implementable only by editing `adapters/_common.py`, which is outside the allowed list.
- **Limitation (GPU off in this review session:** `torch.cuda.is_available() = False`, `nvidia-smi` empty**):**
  I could not reproduce the G1(a) GPU re-run itself; I verified the produced artifacts, the recorded environment
  and the log↔CSV consistency instead. Nothing in the evidence contradicts the author's run.

### G2 — `reports/B4e2_followup.md` corrections — **PASS**

- (i) The summary's first point (now §9.1 after the renumbering) reads exactly
  "so a resume over done (ok/budget) images no longer misaligns the column; an error-row resume is fixed in G1
  above." (`reports/B4e2_followup.md:156`) — the prompt's required scoped wording, verbatim.
- (ii) The required **"Deviations from the prompt"** section is present (§7, `:125`), listing the over-qualified
  claim with the G1 evidence (extended diag scenario (b) + bit-identical F1(a) re-run).
- (iii) `checker_review_followup.md` finding 1 + G1 evidence are cited in the §3 G1 blockquote, in §7 and in
  §9.1. Section numbering (§7 Deviations, §8 sha256, §9 Summary) matches the report's §4 description.

### G3 — re-verify — **PASS** (every command rerun by me)

| command (rerun by me, OMP_NUM_THREADS=8) | my result |
|---|---|
| `python3 adapters/diag/diag_b4e_t1_clip_source.py` | PASS (clamp = only value-changing edit; d8only inherits official), exit 0 |
| `python3 adapters/diag/diag_b4e_t5_done_images.py` | PASS 5/5, exit 0 (tests `ttba_common.done_images`, any-status-done — distinct from the adapter path) |
| `python3 adapters/diag/diag_b4e2_t6_resume.py` | PASS 5/5 (c) + 5/5 (b), exit 0 |
| `python3 tools/eval/tests_contract/test_contract.py` | ALL PASS (21/21), exit 0 |
| `python3 tools/eval/selftest_evalkit.py` | ALL PASS (20/20), exit 0 |
| `check_contract.py --stage smoke --split_dir splits/imagenet/ttba_github splits/places365 splits/robust --roots $(cat reports/B4e2_smoke_roots.txt) --report /tmp/checker_followup2_gate.md` | **exit 0**, I01–I12 + DISK all PASS |

- My gate report `/tmp/checker_followup2_gate.md` is **byte-identical** to the author's
  `/tmp/b4e2_followup2_gate.md` (`diff` empty) — the author's gate rerun is genuine and reproducible. It differs
  from the committed `reports/raw/B4e2/T3_gate.md` only in the time-varying DISK row (36.8 vs 36.9 GB free) —
  expected, since T3_gate.md was produced at an earlier HEAD.
- Tracked `reports/contract_status.md` untouched (`git status` clean for it).

## Claims in `reports/B4e2_followup2.md` that the evidence does not support

**None found.** Every substantive claim reproduces: the prompt/contract/engine sha256s (§1, §8), the allowed-list
compliance (§1), the G1 code description (§3), the G1(a) identity table (24/24 rows, 0 field diffs, 24/24 npz
bit-equal, 3/3 x0, clip 0,0,0, wall times, §3), the G1(b)/(c) diag results (§3), the G2 edits (§4), all six G3
results (§5), the findings table (§6), and the §8 sha256 table (all 6 listed files recomputed and matching,
including the corrected `B4e2_followup.md` sha `2f0da00d…` from commit `e2b7a2e`). "B5 was not started" holds (no
B5 result dirs or reports; only the pre-existing `HUMAN_SIGNOFF_B5_TEMPLATE.md`).

Items I could not independently re-verify in this session (GPU off), all consistent with the recorded evidence
and the previous checker's verification on this same instance — not unsupported, listed for completeness:
1. §2 driver `580.105.08` and capability `(12, 0)` / 170 SMs (env file records the GPU name and torch
   `2.7.1+cu128`, which I did verify).
2. §2 `OMP/MKL/OPENBLAS_NUM_THREADS=8` for the G1(a) run (recorded command only; no artifact contradicts it).

Minor remark (not a defect): §7 classifies the row-position implementation as a "design note (not a
deviation)". Defensible — the prompt's option 1 demonstrably fails acceptance criterion (b) (verified by
analysis above), option 2's literal wording needs a forbidden `_common.py` edit, and the substitution is
disclosed in both §3 and §7 — but a stricter reading would list it under Deviations. The required behaviour
("a pre-existing ERROR row's stem is NOT treated as 'existing'", column name / counter semantics / done-image
behaviour unchanged) is met.

## Deviations from my review protocol

1. The GPU is OFF in this review session, so I did not rerun the G1(a) CGBA attack myself; I verified the
   author's artifacts (bit-comparison, env file, log↔CSV cross-check) instead. All other verifications are my
   own command reruns.
2. `check_contract.py` was run only with `--report /tmp/checker_followup2_gate.md` (plus the tests_contract
   sandbox, which writes only under its own tempdir). No files edited except this review file; no git
   checkout/reset/commit/add; all commands CPU-only with `OMP_NUM_THREADS=8`; my extra sandbox test ran as an
   inline heredoc against a temp directory.

## Summary (5 lines)

1. G1 PASS: the row-position merge (`_pre_existing_count`) is correct by code reading (1 row ↔ 1 `_CLIP_EVENTS`
   entry per attacked image, in order) and by tests — the author's diag passes 5/5 (c) + 5/5 (b) (X-new=5,
   Y=9), and my own mixed scenario (done rows with existing values, two error rows, one re-errored) also passes.
2. G1(a) PASS on artifacts: 24/24 identical per_checkpoint rows (0 field diffs, all columns), 24/24 bit-equal
   npz deltas, clip_events_182 = 0,0,0; the smoke baseline was not regenerated (older mtime, pre-F1 adapter
   sha), so the identity claim is genuine; the GPU rerun itself was not reproducible in this session (GPU off).
3. G2 PASS: §9.1 carries the exact scoped claim, §7 Deviations is present, finding 1 + G1 evidence are cited
   in §3/§7/§9.1.
4. G3 PASS: all six commands rerun by me with matching results (21/21, 20/20, gate exit 0; my gate report
   byte-identical to the author's); contract_status.md clean.
5. No unsupported claim found; frozen engine, contract v1.8 and prereg untouched; allowed list respected;
   B5 not started.
