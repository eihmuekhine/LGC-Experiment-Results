# Checker review — phase B4e2 (review mode, skill lgc-preflight §6)

Independent review session: this session contains **no author work** (fresh reviewer session; the author's
code/report were only read, never edited). Worktree `/root/autodl-tmp/LGC_frozen`, base `e9ff7da`, HEAD
`aee5027`. Spec `reports/raw/B4e2/prompt.txt` present (sha256 `ce4f18615a9cd3f72dd16aa7144e739b1b97c50ef1c02de4b4572fc3cb5820a5`,
matches report §1). Contract v1.8 sha256 `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` (matches report §1).
Engine sha256 `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` unchanged (frozen engine untouched).
Diff `e9ff7da..HEAD` touches only allowed files (adapters/run_cgba.py, ttba_github/ttba_common.py,
tools/eval/check_contract.py, tools/eval/tests_contract/test_contract.py, adapters/diag/*, reports/*); no
forbidden file (lgc_bench/*, run_benchmark_33b.py, run_frame_b1.sh, official clones, contract, prereg) edited.

## Verdicts per T-item (my own evidence, commands rerun unless stated)

### T4 row-split invariance — **PASS** (criterion: run rows 2–3 alone, compare found/state_query/pix_l2/npz delta vs T2, report IDENTICAL or first difference, quote seeding file:line, fix nothing)

- Throwaway runs exist: `Results/b4e2_t4_egc_resnet50`, `Results/b4e2_t4_ttba_resnet50`; throwaway split
  `/tmp/opencode/b4e2_t4/resnet50_non_targeted.csv` = header + rows 2–3 of the frozen split (order 1,2; verified
  against `splits/imagenet/ttba_github/resnet50_non_targeted.csv`).
- **NOT IDENTICAL, both runners — report's first differences reproduce exactly:**
  - EGC `ILSVRC2012_val_00015433` @100: T2 `per_checkpoint.csv` pix_l2 `32.86622619628906` vs T4
    `27.180103302001953`; found=1 both; npz `delta` arrays differ (Q100), x0.npy identical.
  - TtBA `ILSVRC2012_val_00015433` @100: T2 state_query `100` / pix_l2 `84.14942932128906` vs T4 `94` /
    `72.87100982666016`; npz Q100 `delta` differs.
- Seeding quotes verified: `run_benchmark_33b.py:1325` `seed_img = derive_seed(args.seed, img_idx)`, `:667`
  `pairs = pairs[: args.num_samples]`, `:1312` `indexed = list(enumerate(pairs))[args.shard_index :: args.num_shards]`,
  `:1315` enumerate; `lgc_bench/env.py:106-111` `derive_seed`. Recomputed `derive_seed(42,0/1/2) =
  42012471/42020390/42028309` — matches the `seed` column of both per_checkpoint CSVs (T2 rows 2–3: 42020390/42028309;
  T4 rows: 42012471/42020390). TtBA: `adapters/run_ttba_v2.py:258-259` (exact mode refuses shards), `:356-358`
  (`torch.manual_seed(42 + int(r["order"]))` per_image alternative), docstring `:9-11` (exact RNG stream).
- Nothing was "fixed" (diff contains no runner change). Consequence for B5 (no row splitting) is correctly carried
  into T8.

### T5 done_images resume fix — **PASS** (criteria: any row = done; sandbox unit test; fresh-run invariance row 1 == T2)

- Fix verified: `ttba_github/ttba_common.py:87` now `{r["image"] for r in csv.DictReader(f)}` (status filter
  removed); function at `:83` as the prompt names it. Matches contract D6 ("resumed only for images without a
  completed row", contract `paper_contract.yaml:55-58`).
- Sandbox test rerun by me: `python3 adapters/diag/diag_b4e_t5_done_images.py` → 5/5 PASS, VERDICT PASS.
- Fresh-run invariance rerun by me: EGC row 1 (`Results/b4e2_t5_egc_resnet50`) vs T2 row 1: 8/8 budgets, 0 diffs in
  found/state_query/pix_l2 **and** all 8 npz `delta` bit-equal; TtBA row 1 (`Results/b4e2_t5_ttba_resnet50`) vs T2:
  8/8 budgets, 0 diffs in all 16 CSV fields (incl. `pix_l2_iter`) and all 8 npz `delta` bit-equal.

### T6 CGBA clip_events_182 — **PASS with caveats** (criteria: counter written to runs_*.csv for CGBA/CGBA_H; diag asserts clamp is the only value-changing edit; re-run numerics IDENTICAL to T2; print the counts)

- Counter verified in code (`adapters/run_cgba.py:150-153` hoists `arg`, counts `arg < -1.0 or arg > 1.0`,
  `theta = torch.acos(torch.clamp(arg, -1.0, 1.0))`) and in output: `Results/b4e2_t6_cgba_resnet50/non_targeted/runs_CGBA.csv`
  has column `clip_events_182` = 0,0,0 for the three images. Column is written for CGBA, CGBA_H and CGBA_D8only
  (method id selected at `run_cgba.py:222-226`).
- Diag rerun by me: `python3 adapters/diag/diag_b4e_t1_clip_source.py` → PASS (arg verbatim, theta clamp form,
  counter lines the only additions, remaining lines byte-identical, d8only inherits official).
- Numerics rerun by me: 24/24 `per_checkpoint_CGBA.csv` rows equal T2; all 24 npz `delta` bit-equal; T6.log L2
  values equal T2_cgba_main.log values.
- **Caveat (unsupported claim, see below): the counts are NOT printed in `T6.log`** — they exist only in the runs
  CSV. The prompt said "print the counts".
- **Caveat (B5 risk, unreported): `_write_clip_events` merges by row index** (`run_cgba.py:204`
  `r["clip_events_182"] = _CLIP_EVENTS[i] ...`); on a resumed run, done images are skipped without appending to
  `_CLIP_EVENTS`, so counts land on the wrong rows. Fresh runs (T6) are correct; any interrupted+resumed B5
  CGBA/CGBA_H cell would record a silently misaligned column.

### T7 check_contract --status_out — **PASS** (criteria: status next to --report unless --status_out; tests_contract all PASS incl. one new test; selftest_evalkit 20/20)

- Code verified: `tools/eval/check_contract.py:786` `status_path = status_out or (os.path.splitext(report_path)[0]
  + "_status.md")`; `--status_out` added in `main()` (`:809-810`); default `--report` keeps status out of the
  tracked `reports/contract_status.md`.
- Rerun by me: `check_contract.py --stage smoke --n 3 --split_dir splits/imagenet/ttba_github splits/places365
  splits/robust --roots $(cat reports/B4e2_smoke_roots.txt) --report /tmp/checker_review_b4e2_gate2.md` → exit 0,
  I01–I12 + DISK all PASS (reproduces the author's T3 gate, `/tmp/b4e2_gate.md`, exactly); status written to
  `/tmp/checker_review_b4e2_gate2_status.md`; tracked `reports/contract_status.md` unchanged (git-clean, no diff
  `e9ff7da..HEAD`).
- Rerun by me: `python3 tools/eval/tests_contract/test_contract.py` → ALL PASS (21/21), incl. new T18
  (default status next to --report; tracked file not rewritten; --status_out honored).
- Rerun by me: `python3 tools/eval/selftest_evalkit.py` → ALL PASS (20/20). AGENTS.md rule 8 satisfied.

### T8 cost table — **PASS** (criteria: s/img from T2; GPU-h = 200 × s/img / 3600; totals main/E10-E11/E17; makespan N=4/8/12, longest first, 1 proc/GPU, row splitting only if T4 IDENTICAL)

- All 42 T2 s/img values recomputed by me from the runs CSVs (mean wall_time_s, n=3 each): every value matches
  `reports/raw/B4e2/T8.log` to 1 decimal (e.g. TtBA densenet161_places365 NT 105.75→105.8; CGBA resnet50 NT
  22.75→22.8; EGC vgg19 T 29.61→29.6).
- Totals recomputed from unrounded s/img: main 24 cells 45.27 GPU-h, E10/E11 18 cells 40.63, E17 6 cells 1.29,
  **total 87.18** — all four match the report exactly. (Summing the log's 1-decimal cells gives 87.19; the report
  used unrounded values — consistent.)
- Makespan recomputed (48 cells, longest-first greedy, 1 proc/GPU): N=4 21.86 h, N=8 11.29 h, N=12 7.40 h — all
  match the report exactly. No row splitting, correctly justified by T4 NOT IDENTICAL.
- E17 s/img proxy (CGBA NT timings for CGBA_D8only, 30 imgs/cell) is documented as deviation §5 and reproduces
  (0.19+0.30+0.19+0.19+0.22+0.20 = 1.29).
- Two largest cells claim verified: TtBA/densenet161_places365 5.88 + 5.83 GPU-h.

## Claims in the author's report that the evidence does not support

1. `reports/B4e2_close.md:118` — "Printed counts: `clip_events_182 = 0` for all three images": the counts are
   **not printed** anywhere in `reports/raw/B4e2/T6.log`; they exist only in
   `Results/b4e2_t6_cgba_resnet50/non_targeted/runs_CGBA.csv` (values 0/0/0 themselves verified). The prompt
   required "print the counts" and the report does not list this as a deviation.
   Fix: reword to "recorded in runs_CGBA.csv" or append the counts to T6.log, and add it to Deviations.

2. `reports/B4e2_close.md:200-201` (summary point 2) — "fixed `done_images` so a row with any status
   (ok/budget/error) counts as done" reads as a uniform resume policy, but only the TtBA path changed:
   `adapters/_common.py:421-425` (used by CGBA/CGBA_H/ARS_OPT/CGBA_D8only via `C.run`) still filters
   `status in ("ok", "budget")`, so those adapters re-run errored images on resume. Out of the author's allowed
   scope for B4e2 (prompt allows only `ttba_github/ttba_common.py` for T5), but the divergence is unreported.
   Fix: add one sentence to the report stating the `_common.py` divergence and ask the user which policy binds
   for B5 baseline cells.

3. Unreported B5 risk in the T6 deliverable: `adapters/run_cgba.py:204` merges `_CLIP_EVENTS` positionally, so
   after any resume the `clip_events_182` column is misaligned (done images occupy the first rows but contribute
   no entries). Also `CGBA_D8only` always records 0 (its `SemiCircular_boundary_search` is not overridden), so
   the column is constant-0/meaningless for E17 cells. Fix: key the merge on the image stem (collect
   `(stem, count)` in `run_official`) and either count via a d8only-specific hook or drop the column for E17.

4. T3 raw evidence is not under `reports/raw/B4e2/`: the gate report lives only at `/tmp/b4e2_gate.md` and
   `reports/human_preflight_b4e2.md` (FLAG lines: 7 — verified: 6 E10/E11 env_frame.txt gaps + 1 GPU-name value
   mismatch, exactly as deviation §1 describes; sections 2–5 clean, stall/nonfinite/error 0) is untracked.
   Fix: copy the gate report to `reports/raw/B4e2/T3_gate.md` and commit `human_preflight_b4e2.md` (or its
   section 1 + FLAG count).

5. Minor, report §5: no sha256 for the raw logs (skill lgc-preflight §5 asks for sha256 of every created/changed
   file; logs are listed with a path note only), and the environment block's HEAD (`e9ff7da`) is the pre-commit
   HEAD — the report itself is committed in `aee5027`. Fix: add the log sha256s and a "committed as aee5027" note.

## Claims verified as supported (no action)

- Prompt hash, contract v1.8 sha256, engine sha, all six §5 file sha256s (recomputed: ttba_common.py
  `53d518b3…`, run_cgba.py `99f5c11f…`, diag_t1 `0a8695b3…`, diag_t5 `219d1b53…`, check_contract.py
  `ce2cf71c…`, test_contract.py `6ab5deaa…`).
- Environment block: RTX 5090, driver 580.105.08, capability (12,0), 170 SMs, torch 2.7.1+cu128 (rerun
  `nvidia-smi` + `torch.cuda`).
- T1 (context): A == A2 bitwise (18 rows) and A == B on the 6 registered budgets (18/18 common rows equal;
  B has the 8 contract budgets) — rerun by me from `Results/b4e2_t1_{A,A2,B}_resnet50`.
- T2 (context): all 42 cells re-run on the study GPU (correct branch: 0 env files under `Results/b4e_smoke_*`
  record the RTX 5090 — they record "NVIDIA RTX PRO 6000 Blackwell Server Edition"); 126 runs rows, no errors in
  any T2 log; all 30 baseline env files + EGC config_single.json record the RTX 5090; smoke roots file lists
  exactly the 24 final roots.
- T3 (context): checker exit 0 with I01–I12 + DISK PASS reproduced by me; FLAG = 7 characterization matches
  `reports/human_preflight_b4e2.md` exactly. Note: the prompt's T3 acceptance was FLAG = 0, so T3 strictly missed
  its criterion; the report discloses this as deviation §1 and defers to B4f T7 — honest, no hidden failure.
- GPU-minutes §5: T4 EGC 46.3 s + T4 TtBA 105.5 s + T5 EGC 26.1 s + T5 TtBA 52.9 s + T6 CGBA 70.3 s — all match
  the logs.
- T4 table values, seed values, file:line quotes, "found is 1 in both", EGC-sharding-preserves-img_idx claim
  (`enumerate` before slicing at `run_benchmark_33b.py:1312`), TtBA per_image alternative — all verified.

## Deviations from my review protocol

None. No files edited except this review file; no git checkout/reset/commit/add; the checker was run only with
`--report /tmp/*.md`; all GPU-free (CPU/read-only) verification commands were run with `OMP_NUM_THREADS=8`.

## Summary (5 lines)

1. T4 PASS: row splitting is NOT invariant for either runner; every reported first difference, seed value and
   file:line quote reproduces exactly.
2. T5 PASS: the fix, the sandbox test and the bit-identical fresh runs (CSV + npz) all reproduce; but the same
   policy was NOT applied to `adapters/_common.py` (CGBA/ARS_OPT still re-run error rows) and the report does not
   say so.
3. T6 PASS with caveats: counter + diag + bit-identical re-run reproduce, but the counts were never "printed"
   (CSV only, contrary to the prompt) and the index-based CSV merge is misaligned after any resume (B5 risk).
4. T7 PASS: status placement, 21/21 tests, 20/20 selftest and untouched tracked status file all reproduce.
5. T8 PASS: every s/img, per-cell GPU-h, total 87.18 GPU-h and all three makespans reproduce exactly; no
   unsupported cost claims found.
