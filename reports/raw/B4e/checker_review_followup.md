# Checker review — phase B4e follow-up (review mode, lgc-preflight §6)

- **Reviewer:** independent checker (@checker), read-only. No files edited except this review.
- **Base:** `616b41a` → **HEAD:** `85fb714` (two follow-up commits: `7122f3b` "record D12 precision block in
  TtBA env files", `85fb714` "deviations from prompt (D16/D17 waiver), HEAD/vit corrections, TtBA resume bug"),
  branch `paper-runs`, worktree `/root/autodl-tmp/LGC_frozen`.
- **Scope of this review:** the follow-up prompt's three items — F1 (precision block + `human_preflight`
  FLAG = 0), F2 (deviations section + HEAD/vit corrections), F3 (TtBA resume bug) — as re-verified independently.
- **Spec:** `reports/raw/B4e/prompt.txt` still does **not** exist. Original B4e spec taken from the transcript
  `/root/autodl-tmp/Latent-Geometric-Chords/B4e.md` (lines 11–71); the follow-up spec (F1/F2/F3) exists only in
  the session transcript, not as a file (see "remaining issues" #5).
- **Checker environment:** no GPU (`nvidia-smi` → "No devices were found"; `torch.cuda.is_available()` False),
  so no attack re-run was possible; everything else re-run on CPU. All my outputs went to `/tmp/opencode/`.
- Contract v1.8 sha256 re-verified: `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`.
  Engine sha `835f1933d9e57107…` (frozen, not in the diff). Diff `616b41a..HEAD` = 3 files:
  `adapters/run_ttba_v2.py` (+12), `reports/B4e_preflight.md` (+101/−9), `reports/raw/B4e/checker_review.md`
  (new, 211 lines). No `lgc_bench/*`, no contract, no prereg, no `ttba_github/*` touched.

## (a) `human_preflight` FLAG lines = 0 — **FAIL as written (FLAG = 6); deviation properly recorded**

- Independent re-run: `python tools/eval/human_preflight.py --roots Results/b4e_smoke_* --out
  /tmp/opencode/hp_followup_check` → `FLAG lines = 6`, exactly the six `b4e_smoke_ttba_*` roots, each
  "not recorded in any env/config file: ['cudnn_tf32', 'deterministic']". Identical to the author's recorded
  result. **The criterion "confirm FLAG lines = 0" is NOT met.**
- Root cause verified: 12 `env_TtBA.json` files exist (6 roots × 2 modes); **0 contain `precision`**. All 18
  CGBA/ARS env files contain it (and those roots are not flagged).
- F1 code verified: both env dicts in `adapters/run_ttba_v2.py` carry the 4-key `precision` block with keys
  identical to `_common.build_env` (AST comparison, 2 sites); `import torch` precedes both constructions;
  `sha256sum` = `80cbcdcc19b2d428b3c56890e7a430b93a77def8caba86f99544e4b0e83bb8ee` (matches the report);
  `py_compile` OK; `selftest_evalkit` **20/20**; `tests_contract` **18/18** (all re-run by me).
- The author's prediction ("any fresh TtBA run with this adapter will emit the block and clear the FLAG") is
  **verified without a GPU**: `canon_flags()` on a simulated fixed `env_TtBA.json` returns
  `{matmul_tf32: False, cudnn_tf32: False, deterministic: True, benchmark: False}` → no missing keys → no FLAG;
  and the 18 CGBA/ARS env files carrying the same block are indeed unflagged.
- No re-run was performed (no GPU); no throwaway dir exists; the six old roots are untouched (env files still
  stale). Correctly so: a CPU re-run would record `gpu: "cpu"` (the code falls back) and create a *new* FLAG.
- Note: re-running **one** TtBA cell into a throwaway dir could not reach FLAG = 0 anyway — the other five old
  roots would still FLAG. FLAG = 0 requires re-running all 6 roots **and** superseding the old dirs. The author's
  plan states exactly this ("the re-run is the whole TtBA smoke … then supersede the old dirs"). Sound.

## (b) F2 items — **PASS (all five verified accurate)**

- **(a) `run_ttba_v2.py` outside the allowed list:** `git diff --name-only 729cf19..616b41a` = 13 files; the
  B4e prompt's allowed list (`adapters/_common.py`, `adapters/run_cgba.py`, `tools/eval/g5_check.py` + tests,
  `adapters/diag/*`) covers all code files except `run_ttba_v2.py`. The 616b41a diff to that file = the
  `nonfinite_queries` column + routing the tracker through `C.HardLabelTracker` (the guarded tracker, 2 sites) —
  observation-only (guard design verified by the previous checker). *Minor wording nit:* "one extra column"
  understates the diff (the tracker rebind is part of it).
- **(b) densenet161 NT / CGBA_H acceptance cells not run:** verified — `Results/b4e_t1_fixed_*` =
  {r50, r50places, robust, vgg19, vit}`_6000` only; no densenet dir; **no `targeted/` mode dir in any
  `b4e_t1_*`**. Supporting fact re-verified: `diag_b4e_t1_compare.py --new-root Results/b4e_t1_d8only_r50_d12
  --old-root Results/b4e_smoke_cgba_resnet50 --cmp identical` → **3/3 IDENTICAL** (8 npz deltas bitwise equal
  each); `nonfinite_queries = 0`; pix_l2@10k = 6.449769 / 6.331734 / 2.605660 (report: 6.4498/6.3317/2.6057).
- **(c) trace-prefix:** re-verified from the stored traces — `00007450`: fixed-6000 vs old-5090 first diverge at
  trace index 11, **q=118, l2 `112.37966918945312` vs `112.37967681884766` (7.629e-06)**; d8only-6000 diverges
  from old-5090 at the same index with the same value; fixed-6000 vs d8only-6000: **no divergence** → GPU-caused,
  not clip-caused; q=118 ≪ first-NaN 774. Every number in the report matches to all printed digits.
- **(d) combined gate:** re-run by me → **exit 1**, I10 FAIL "multiple GPU names across cells: ['geforce rtx
  5090', 'rtx pro 6000 blackwell server edition']"; baselines-only roots → **exit 0**. Matches the report.
- **(e) raw logs:** `reports/raw/B4e/` contains only `checker_review.md` (no B4e run logs). Accurate — the text
  itself names the checker review as the only raw artifact.

## Corrections — **PASS**

- **HEAD `729cf19` → `616b41a`:** verified — `git show 616b41a --stat` includes `reports/B4e_preflight.md`
  (+232 lines), i.e. the report is committed in `616b41a`, as now stated.
- **vit `≈1e-7` → per-image numbers:** re-verified from the CSVs (fixed-6000 vs old-5090, vit_b_32 NT):
  `00024661` first divergence @500, |Δpix_l2| = **4.768e-07**; `00046402` first divergence @5000,
  |Δpix_l2| = **1.529e-04**; 2/3 images diverge. Both corrected values in the report are accurate.

## (c) resume-bug note — **PASS**

- `ttba_github/ttba_common.py:83-87` re-read: `done_images` returns only rows with `status == "ok"`. ✓
- Census re-run: 30 `runs_*.csv` under `Results/b4e_smoke_*` (15 roots × 2 modes), 90 rows total,
  status counts = `{budget: 90}` — "every baseline run ends `status == "budget"`" is exact. ✓
- The proposed one-line fix
  (`return {r["image"] for r in csv.DictReader(f) if r.get("status") in ("ok", "budget")}`) validated on a
  sandbox CSV in `/tmp`: current function returns only the `ok` image; the fixed line also skips the
  `budget` image; `error` rows would still be retried (matching the report's "also consider `"error"`" note). ✓
- `ttba_common.py` is untouched in `616b41a..HEAD` (0 files) — the fix was correctly **not** applied (outside the
  allowed list; L4 decision deferred to the user / contract v1.9). ✓

## Remaining unsupported claims / imprecisions (none material)

1. **Dangling pointer:** the corrected T1(d)2 text says "(no d8only control exists for vit_b_32; see Deviations)",
   but the Deviations section contains no vit-d8only item. The *fact* is true (no `b4e_t1_d8only_vit*` run
   exists — verified), so the previous checker's unsupported-claim #2 is disclosed, just not in the section the
   pointer names.
2. **Terminology:** "the whole TtBA smoke (6 cells, 3 images each)" — in the report's own convention (30 cells =
   15 roots × 2 modes) TtBA is 12 cells / 6 roots; "6 cells" here means roots. The adjacent sentence gives the
   correct counts (6 roots, 12 env files).
3. Deviation (a)'s "one extra column" understates the 616b41a diff (also the `C.HardLabelTracker` rebind at 2
   sites — the T1(c) guard mechanism, observation-only).
4. The verification-table row "precision-block keys == `_common.build_env` (unit check, CPU)" cites no command,
   unlike the other rows. I re-ran an equivalent AST check: identical keys at both sites. Passes.
5. The follow-up prompt itself is not archived under `reports/raw/B4e/` (same gap as `prompt.txt` before it);
   the Deviations section cites the session id instead of a file + hash.
6. Commit `85fb714` includes `reports/raw/B4e/checker_review.md` beyond "code + report" — traceability-positive
   (the report references it); noted for completeness, not a violation.

## Deviations from the review prompt (checker's own)

1. `reports/raw/B4e/prompt.txt` does not exist; the original spec was taken from `B4e.md` (transcript) and the
   follow-up spec from the session (not archived).
2. No GPU at review time: no attack re-run possible; all verification was CPU (CSV/npz/trace parsing,
   `check_contract.py`, `human_preflight.py`, unit tests, sandbox CSV test).
3. My `check_contract.py` re-runs overwrote `reports/contract_status.md` as a tool side effect (it writes that
   file regardless of `--report`); restored with `git checkout --` — worktree verified clean vs HEAD afterwards.
   `human_preflight` output went to `/tmp/opencode/`, not the author's `reports/human_preflight_b4e`.
4. This review file is the only file written.

## Verdict

- **(a) FAIL as written:** `human_preflight` FLAG = 6 ≠ 0 (re-run independently). The cause is fully verified
  (12 stale `env_TtBA.json` files; no GPU to re-run), the deviation is recorded with a sound deferred plan, and
  the fix itself is verified correct and sufficient (canon_flags simulation + 18/18 CGBA/ARS env files with the
  block are unflagged). FLAG = 0 still requires the 6-root TtBA smoke re-run (throwaway dirs + supersede) once
  the GPU returns — before B5 relies on the preflight report.
- **(b) PASS:** every F2 item (a)–(e) is accurate; both corrections (HEAD, vit) verified to the printed digits.
- **(c) PASS:** the resume-bug note is accurate (code, 90/90 `budget` census, validated 1-line fix, correctly
  not applied).
- No material unsupported claim remains; six minor imprecisions are listed above (#1–#6).
- Still open for contract v1.9 / the user: the D16/D17 waiver text, the E17 wording under D12, the resume-bug L4
  decision, and the D16 GPU-migration record (combined I10 gate).
