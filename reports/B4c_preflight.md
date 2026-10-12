# B4c — Pre-B5 preflight (gate G4, GPU ON)

Status: **COMPLETE (2026-10-13)**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`. Contract **v1.6**
(`bb3f54f070c6…`, sha recorded in amendment 2 deviation D10). Engine sha `835f1933d9e5` (frozen, unchanged). GPU:
RTX 5090 (32 GB), torch 2.7.1+cu128. Budget 10000, smoke rows 1–3 (`n = 3`).

Scope: verify the generic EGC runner is bit-identical to the prereg runner, that the E10/E11 adapter changes did not
regress ImageNet, that the checker's stricter I09/I10/I11 hold, and that the E10/E11 EGC smoke cells are re-run into
per-(victim, mode) out dirs so the smoke gate exits 0. **No B5 work started.**

---

## T1 — Generic EGC runner == prereg runner (ImageNet)

### (a) `--arch` / `--data_dir` added

`ttba_github/run_frame_b1_generic.py` now accepts `--arch {vgg19,resnet50,vit_b_32,resnet50_places365,
densenet161_places365,Salman2020Do_R50}` and `--data_dir` (default `/root/autodl-tmp/ttba_mini_flat`), routed through
`C.load_victim_any` / `C.load_x0_any(..., data_dir=...)`. No attack logic, seed or hyper-parameter changed.

### (b) EngineConfig diff vs `run_benchmark_33b.py` built from `run_frame_b1.sh` argv

`adapters/diag/diag_t1b_cfg_diff.py` builds both `EngineConfig`s and prints every differing
`dataclasses.asdict` field (allowed: `log_geometry`, `save_trace`, `verbose`).

**Verdict: IDENTICAL** — no `EngineConfig` field differs.

### (c) resnet50 NT+T, rows 1–3, `--save_adv` → `Results/b4c_egc_generic_resnet50`

Compared with `Results/b4_smoke_egc_resnet50` per (mode, image, budget): `found`, `state_query`, `pix_l2` exactly
equal; npz `delta` bitwise equal.

**Verdict: IDENTICAL** — 48 rows match; 48/48 npz deltas bitwise equal.

---

## T2 — ImageNet adapter regression (resnet50 row 1, current adapters)

Each cell re-run with the CURRENT adapters (`--limit 1`) into `Results/b4c_regress_<method>`, compared with the same
image (`ILSVRC2012_val_00007450`) in `Results/b4_smoke_{ttba,cgba,arsopt}_resnet50` (found/state_query/pix_l2 at all
8 checkpoints; npz delta bitwise).

| method | mode | verdict |
|---|---|---|
| TtBA | non_targeted | IDENTICAL |
| TtBA | targeted | IDENTICAL |
| CGBA | non_targeted | IDENTICAL |
| CGBA_H | targeted | IDENTICAL |
| ARS_OPT | non_targeted | IDENTICAL |
| ARS_OPT | targeted | IDENTICAL |

`adapter_sha256` B4 (run_manifest.json) vs now:

| adapter | B4 `adapter_sha256` | now | changed? |
|---|---|---|---|
| `run_ttba_v2.py` | `c769d9773412…` | `d97292408ff1…` | yes (E10/E11 `--arch` + `_main_generic`) |
| `run_cgba.py` | `1763e9503b32…` | `4c3fdf714ea7…` | yes (`_faithful_model` -> `C.raw_net`) |
| `run_arsopt.py` | `d06410d40db1…` | `d06410d40db1…` | no |

**Verdict per method: IDENTICAL** — the E10/E11 adapter changes did not change ImageNet resnet50 results.

---

## T3 — CGBA magnitude (G5 preview; descriptive, no tuning)

### (a) mean/median pix_l2 @2000/5000/10000 (n=3) vs `ttba_github/ttba_table1.json`

`ttba_github/ttba_table1.json` is **MISSING (UNVERIFIED)** — no published ratios can be computed, so no cell is
flagged (the ratio/flag step has no denominator). Our baseline magnitudes:

| method | arch | mode | @2000 | @5000 | @10000 |
|---|---|---|---|---|---|
| CGBA | resnet50 | NT | mean 71.45 / med 74.57 | 71.45 / 74.57 | 71.45 / 74.57 |
| CGBA | vgg19 | NT | 2.30 / 2.21 | 1.06 / 1.30 | 0.70 / 0.99 |
| CGBA | vit_b_32 | NT | 6.10 / 6.46 | 1.58 / 1.57 | 0.91 / 1.19 |
| CGBA_H | resnet50 | T | 58.64 / 42.95 | 54.17 / 38.32 | 49.92 / 35.05 |
| CGBA_H | vgg19 | T | 63.63 / 43.07 | 51.57 / 29.57 | 38.43 / 22.02 |
| CGBA_H | vit_b_32 | T | 18.37 / 16.22 | 7.28 / 7.10 | 3.61 / 3.66 |
| TtBA | resnet50 | NT | 36.56 / 40.25 | 21.09 / 22.96 | 10.95 / 11.83 |
| TtBA | vgg19 | NT | 0.75 / 0.35 | 0.40 / 0.21 | 0.29 / 0.16 |
| TtBA | vit_b_32 | NT | 7.17 / 6.07 | 1.25 / 1.62 | 0.71 / 0.79 |
| TtBA | resnet50 | T | 47.11 / 31.76 | 37.67 / 24.61 | 27.66 / 16.42 |
| TtBA | vgg19 | T | 40.68 / 25.97 | 21.72 / 12.30 | 7.11 / 6.04 |
| TtBA | vit_b_32 | T | 18.23 / 17.83 | 8.08 / 8.62 | 3.95 / 4.27 |

Note (descriptive): CGBA resnet50 NT finishes ~761 queries (boundary-walk cap) and carries forward, hence the flat
2000/5000/10000 values; CGBA on vgg19/vit_b_32 NT is far smaller. This is the G5 preview, not a tuning decision.

### (b) L2 mapping check (CGBA resnet50 NT row 1)

Final returned iterate `x_n` (normalised space, clipped to `[lb,ub]` as `Attack()` returns, `proposed_attack.py:264`):

| quantity | value |
|---|---|
| d_n = ||x_n − normalize(x0)||₂ | 376.387695 |
| d_p = ||to_pixel(x_n) − x0||₂ | 85.077782 |
| d_n / d_p | **4.424042** (required [4.37, 4.46]) |
| tracker pix_l2@10000 | 85.077705 (required ≤ d_p = 85.077782) |

**Verdict: OK** — the normalised↔pixel L2 mapping is correct.

### (c) argument table (conditional)

Not triggered: (a) flagged no CGBA/CGBA_H cell (no `ttba_table1.json` to ratio against). For the record, every
argument `run_cgba.py` passes to `Proposed_attack` (values `run_cgba.py` PARAMS: `dim_reduc_factor=4` :44, `iteration=93`
:45, `initial_query=30` :46, `tol=1e-4` :47, `sigma=2e-4` :48, `attack_method='CGBA'/'CGBA_H'` :139, `mean/std`
`C.MEAN/C.STD` :125) equals the official drivers' ImageNet values (`Non_targeted_attack.py:30-36` /
`Targeted_attack.py:33-37`; `proposed_attack.py:15` class defaults). The only difference is `verbose_control='No'`
(print-only logging) vs the official default `'Yes'`. **Nothing changed.**

---

## T4 — Generic runner provenance

- **Guards** (added to `run_frame_b1_generic.py`, before any model load): refuse to start unless
  `sha256(lgc_bench/engine.py)` starts with `835f1933d9e5`, `sha256(<ae_checkpoint>)` starts with `1d004c634f1b`, and
  `ADV_STORE_BUDGETS` is unset.
- **env_frame.txt**: now writes the real `ADV_STORE_BUDGETS` value
  (`ADV_STORE_DTYPE <dtype> ADV_STORE_BUDGETS <value>`, value defaults to `unset`), the same token format as
  `run_frame_b1.sh:43`; the old line hard-coded `unset` and used a space (never matched the I11 regex).

Split sha256 in each existing env file:

| env file | split sha256(s) contained |
|---|---|
| `b4_smoke_egc_resnet50/env_frame.txt` | `e08a069e…` (resnet50 NT) **and** `550ec796…` (resnet50 T) |
| `b4_smoke_egc_vgg19/env_frame.txt` | `1e811637…` (vgg19 NT) **and** `cf0a8efb…` (vgg19 T) |
| `b4_smoke_egc_vit_b_32/env_frame.txt` | `a5f152ec…` (vit_b_32 NT) **and** `c5eee382…` (vit_b_32 T) |
| `b4b_smoke_LGC_RC_EGC_resnet50_places365/env_frame.txt` | `70bd3d5c…` (resnet50_places365 **T only** — broken) |
| `b4b_smoke_LGC_RC_EGC_densenet161_places365/env_frame.txt` | `51935b75…` (densenet161_places365 **T only** — broken) |
| `b4b_smoke_LGC_RC_EGC_Salman2020Do_R50/env_frame.txt` | `2985ebc4…` (Salman2020Do_R50 **T only** — broken) |

`run_frame_b1.sh` names out dirs **per arch** (`OUT=${OUT_PREFIX}_$A`, e.g. `Results/b1_resnet50`), with both modes
in one dir and `env_frame.txt` at the root holding both modes' split shas. For B5, ours uses **one out_dir per
(victim, mode)**, so NT and T never share `env_frame.txt` (the b4b per-victim dirs above were the bug that this
fixes).

---

## T5 — Checker (stricter only) + re-run

Changes in `tools/eval/check_contract.py`:
- **I10 per cell (no pooling)**: the env file(s) in a cell's root or root/`<mode>` must contain that cell's
  `split_sha256`; a cell whose `<arch>_<mode>` is missing from `constants.split_sha256` is FAIL ("unregistered split").
- **I11**: an env file of ours (`env_frame.txt`) without an `ADV_STORE_BUDGETS` token is FAIL (regex widened to
  `ADV_STORE_BUDGETS[=\s]+(\S+)` to match both the shell and runner token format).
- **I09**: emit a WARN line only when the `found = 0` count is > 0.

New tests (in `tools/eval/tests_contract/test_contract.py`): T15 unregistered split → I10 FAIL; T16 env holds the
other mode's split → I10 FAIL; T17 our env without ADV_STORE_BUDGETS → I11 FAIL.

| suite | result |
|---|---|
| `python tools/eval/tests_contract/test_contract.py` | **ALL PASS (18/18)** |
| `python tools/eval/selftest_evalkit.py` | **ALL PASS (20/20)** |

Re-ran the 6 E10/E11 EGC smoke cells with the fixed runner into per-mode out dirs
`Results/b4b_smoke_LGC_RC_EGC_<victim>_<mode>` (rows 1–3, `--save_adv`). The three superseded per-victim dirs were
moved to `Results/retired_b4b_egc_per_victim/` (preserved, excluded from the `b4b_smoke_*` glob).

Combined smoke gate (all B4 + B4b roots): **exit 0, I01–I12 + DISK all PASS** (I09 now PASS, count 0). Report:
`reports/contract_smoke_b4c.md`.

---

## T6 — CGBA raw-net equivalence on E10/E11

For rows 1–3 of each new victim, `max|victim(x0) − raw_net((x0−mu)/sd)|` and argmax equality, with
`mu=[0.485,0.456,0.406]`, `sd=[0.229,0.224,0.225]` exactly as `run_cgba.py` passes them (`C.MEAN/C.STD`,
`_common.py:42-43`; `run_cgba.py:125`).

| victim | max|victim − raw_net| (3 rows) | argmax equal |
|---|---|---|---|
| resnet50_places365 | 0.000e+00 | True |
| densenet161_places365 | 0.000e+00 | True |
| Salman2020Do_R50 | 0.000e+00 | True |

RobustBench `normalize` constants: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

---

## T7 — Disk

Per-cell usage (n=3) scaled ×200, 42 cells total (24 main + 18 E10/E11):

- smoke total (n=3): **0.328 GiB**
- **scaled total (n=200): 21.895 GiB** (≈ 0.52 GiB/cell; matches contract `~0.48 GB/cell` + `x0.npy`).
- `df`: total 50.0 GiB, used 12.3 GiB, **free 37.7 GiB** ≥ 25 GiB (and ≥ 21.9 GiB needed).

`adapters/diag/diag_t7_disk.py` prints the per-cell table.

---

## T8 — Commit (paper-runs)

Committed: `adapters/`, `ttba_github/run_frame_b1_generic.py`, `ttba_github/ttba_dump_split_places365.py`,
`ttba_github/ttba_dump_split_robust.py`, `splits/places365/`, `splits/robust/`, `tools/eval/check_contract.py`,
`tools/eval/tests_contract/`, `contract/column_map.yaml`, `reports/*.md`. NOT added: `Results/`, `cache`,
`checkpoints/`, `models/`. HEAD: `16ead02a7e732e22497a10b964f326f6aca2e16d`.

## T9 — cProfile (optional; TtBA adapter, resnet50 NT row 1)

Top 15 by cumulative time (total 72.6 s profiled, 63.6 s in `attack`):

| # | function | cumtime (s) |
|---|---|---|
| 1 | `TtBA.py:189 attack` | 63.64 |
| 2 | `TtBA.py:97 normal_vector_approximation` | 54.84 |
| 3 | `hardlabel_tracker.py:161 predict_label` | 52.29 |
| 4 | `hardlabel_tracker.py:123 labels` | 52.25 |
| 5 | `torch/nn/modules/module.py:1747 _wrapped_call_impl` | 50.46 |
| 6 | `torch/nn/modules/module.py:1755 _call_impl` | 50.44 |
| 7 | `_common.py:241 ClampedVictim.forward` | 50.10 |
| 8 | `general_torch_model.py:17 forward` | 49.98 |
| 9 | `torch/nn/parallel/data_parallel.py:171 forward` | 49.01 |
| 10 | `torchvision/models/resnet.py:284 forward` | 32.67 |
| 11 | `torchvision/models/resnet.py:266 _forward_impl` | 32.66 |
| 12 | `torch/nn/modules/container.py:238 forward` | 30.25 |
| 13 | `torchvision/models/resnet.py:143 forward` | 29.67 |
| 14 | `torch/nn/modules/module.py:2600 _named_members` | 12.17 |
| 15 | `torch/nn/modules/conv.py:553 forward` | 11.50 |

(Observed: the `python -m cProfile` runner makes `__file__` relative, which breaks the adapter's post-`os.chdir`
`abspath(__file__)`; using an absolute script path is required. This is a cProfile quirk, not an adapter bug — the
adapter runs unchanged under the normal `python` launcher.)

---

## sha256 of every changed/new file

| file | sha256 |
|---|---|
| `ttba_github/run_frame_b1_generic.py` | `ec027f6e82a11f64b2b64efecc1f3033e87a48b4e2804b90f4052dd454c6072e` |
| `tools/eval/check_contract.py` | `d708667d1e57c206d0aa5a00fa4a72a1a132da24e9c05994b1932e3f4aeb0a72` |
| `tools/eval/tests_contract/test_contract.py` | `fd955c7fe2a14193f1961427ed09630dc27aa61192a8ee35dbba4a472ce7c5e6` |
| `adapters/diag/diag_t1b_cfg_diff.py` | `91030e042c89c25d3c873a93069486afbd9a6c1e2a795c8fc63fda01cea06f66` |
| `adapters/diag/diag_t1c_compare_egc.py` | `73ebf2e2cab813c3becb3ce4f287788531acc669817adb2fa03dc1a65fbb8caa` |
| `adapters/diag/diag_t2_regress.py` | `e1181ed90d54d1ecdbb7183c42045f5af89d370572ade271a36747777efd43af` |
| `adapters/diag/diag_t3a_magnitude.py` | `dfc768940ed95527edbc7db001cf9ef22775f9e3ce9e81bb86f18fc6ee0bee3a` |
| `adapters/diag/diag_t3b_cgba_l2map.py` | `337e3b01fdbe8dce7a9de64bce597f15eabce806d758ccd49b53426986bd11cf` |
| `adapters/diag/diag_t6_rawnet.py` | `ec3b9af9bf1c8d650dd93e95f989a1fc71d5bb05402fbcba67a0aad6ac7add75` |
| `adapters/diag/diag_t7_disk.py` | `17d30e4cdbf85d4907fa94378a5ce618f361888d93604b1c82d2e719ca5a46e7` |

Frozen engine unchanged: `sha256sum lgc_bench/engine.py` = `835f1933d9e5…` (verified, not edited).

## GPU minutes

~20 GPU-minutes (T1(c) ~2.4 min, T2 ~3.5 min, T3(b) ~1.3 min incl. failed attempts, T5 EGC re-run ~8.2 min, T6 ~0.5
min, T9 ~1.2 min, plus per-invocation model load).

---

**STOP.** No B5 work started.
