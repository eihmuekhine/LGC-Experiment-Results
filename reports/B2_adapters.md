# B2 — Baseline adapters + G1 tests

Status: COMPLETE (2026-10-11). Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD `f255b7cbb949161290373343b3430e64b42fe7bd`.

## 1. What was done

Wrote the five baseline adapters (`adapters/run_<method>.py`) that run each official attack unchanged
through the decision-only `HardLabelTracker`, store the adversarial images at every contract checkpoint, and
write the contract artefacts (`per_checkpoint*.csv`, `runs*.csv`, `env_*.json`, `run_manifest.json`, adv_store,
traces). Shared infrastructure was consolidated in `adapters/_common.py` so fairness rules F1–F12 hold by
construction (contract checkpoints/budget read from `contract/paper_contract.yaml` at runtime, shared TtBA victim
loader, shared TtBA image loader, one seeding rule, manifest-matched resume).

Resumed from an interrupted B2: four defects were found by running the adapter smoke test and fixed:

| # | Defect | Fix | File |
|---|---|---|---|
| 1 | `UnnormalizeVictim` registered CPU mean/std buffers, so CGBA's normalised cuda inputs hit a cuda/cpu mismatch | materialise mean/std on `x.device` (cached per device/dtype) | `_common.py` |
| 2 | ARS-OPT shim `_Npz` had no `__init__`, so `NpzDataset(datasetname)` failed | add `__init__(self, *a, **k)` to the stub (the real `dataset/npz_dataset.py` is missing from the clone) | `run_arsopt.py` |
| 3 | Sign-OPT's loop formats the label with `{:d}` and compares it to an int, but the tracker's `predict_label` returns a `(B,)` tensor | wrap the tracker in `_SignOPTModel` whose `predict_label` returns a 0-dim scalar (matching official `PytorchModel.predict_label`, `models.py:45`) | `run_signopt.py` |
| 4 | TtBA targeted start mixed a CPU `x0` with a cuda `tar_x` in `HardLabelTracker.set_initial` | pass `x0.cuda()` to the tracker (the TtBA loader returns CPU tensors) | `run_ttba_v2.py` |

Additionally `run_signopt.py` now locates the attackbox clone from `sys.path` (`_common.run` inserts `--repo` at
front) instead of a `_REPO` global, so `--help` and the required `--repo` argument are handled by the shared parser.

The adapter smoke test (`adapters/tests/test_g1_adapters.py`) was restructured to run each `(method, mode)` in its
own subprocess: the four official repositories share top-level module names (`utils`, `config`, `dataset`, …) that
collide in one interpreter, and CGBA/Sign-OPT hard-code `.cuda()`, so the dummy victim/image live on cuda.

## 2. Adapter → method mapping (matches `contract/paper_contract.yaml` methods)

| Adapter | Contract method id(s) | Official entry (from B1) | Mode |
|---|---|---|---|
| `run_ttba_v2.py` | `TtBA` | `TtBA.Attacker` (repo `/root/autodl-tmp/TtBA`), plus `pix_l2_iter` official iterate | non_targeted, targeted |
| `run_cgba.py` | `CGBA` (NT) / `CGBA_H` (T) | `Proposed_attack` `go_to_boundary_CGBA` / `go_to_boundary_CGBA_H` (`proposed_attack.py:151/:131`) | NT / T |
| `run_hsja.py` | `HSJA` | `hsja.hsja` (`hsja.py:4`) | non_targeted, targeted |
| `run_arsopt.py` | `ARS_OPT` | `ParsOptL2Norm`, prior-free `method='ARS'` (`pars_opt_l2_norm_attack.py:15`) | non_targeted, targeted |
| `run_signopt.py` | `Sign_OPT` | `OPT_attack_sign_SGD` (`attack/Sign_OPT.py:42`) | **non_targeted only** |

## 3. Allowed changes (documented in each `env_<method>.json`)

- **Model object**: the `HardLabelTracker`; CGBA/HJSA/Sign-OPT/TtBA use `ClampedVictim` (reproduces the official
  wrappers' [0,1] clamp), CGBA uses `UnnormalizeVictim` (its attack runs in ImageNet-normalised space). Only the
  top-1 decision leaves the tracker.
- **Data input**: our image `x0` and, for targeted methods, the CSV target image (CGBA `tar_img`, HSJA
  `target_image`, ARS-OPT `target_class_image`). TtBA runs its own exact RNG stream on the shared split rows.
- **Budget/binding**: the tracker's 10 000 budget binds for every method. HSJA `num_iterations=64` (its
  `max_num_evals` is per-iteration, `hsja.py:43-46`); Sign-OPT `query_limit=10000`; ARS-OPT `maximum_queries=10000`
  (the official driver forces 20000 for ImageNet targeted, overridden). No success-threshold early stop is active
  in any of the four loops (B1); TtBA `early=0`.
- **ARS-OPT import shims only** (adapter-side, never executed by the ARS loop): no-op `glog`, minimal `bidict`,
  and a `dataset.npz_dataset` stub (`NpzDataset/NpzExtraDataset/NpzClipDataset`) because that file is absent from
  the clone.
- Nothing else (step sizes, probe counts, schedules, search rules, initialisation logic) is changed; each
  hyper-parameter carries a `source: file:line` in `HYPER_SOURCES`.

## 4. Files (sha256)

| File | sha256 |
|---|---|
| `adapters/_common.py` | `a248956fe3551939f4db0d46b7124dd867bd1a36b547269410c89dc682531099` |
| `adapters/run_ttba_v2.py` | `c769d9773412403e416b970869ebb60e523c5ed643a3dce4e600223bfdb9e685` |
| `adapters/run_cgba.py` | `f7fb1e3128fd7537b1217d5be14f872ce43fc183da43d101edb0c14f62367b73` |
| `adapters/run_hsja.py` | `18073d87a7349d2548c6c47c865792221e49ab9b52f6f303d6c94da09a8e3978` |
| `adapters/run_arsopt.py` | `d06410d40db15a83415df23754de10147da9dedc4f1de018b0a9f59fb58f8549` |
| `adapters/run_signopt.py` | `002a4c08e9615845a6a43abfcc7df2777ee2ab4b4214ab813fb42bae22b489d1` |
| `adapters/tests/test_g1_helpers.py` | `2865f2a06f84ddb307d7d8d7022ed5b60d40f8e4104da278350210c307f71f16` |
| `adapters/tests/test_g1_tracker.py` | `692b8db949056ba8e72ff11cafbc1b126083498f300266a862062e24eb0126eb` |
| `adapters/tests/test_g1_adapters.py` | `7597ce11998014282b350a00f7bd24b11b92a9cd3324cbb613063bba5699b7a9` |
| `tools/eval/hardlabel_tracker.py` (unchanged) | `d5ca5098cffbb7d77639d1e62308ede2ad583983f053d0339deecb640e02da89` |
| `tools/eval/advstore.py` (unchanged) | `6c27feb91ab1a0931ef205fc6d8f99dffaf3cbb52df2e734fc811e0c3d8c80e3` |
| `contract/paper_contract.yaml` (unchanged) | `a8c28e4e033541743d38ad7812fb9c2565a610be075a405edf9ca331e440d2aa` |

Official repo commits (from B1, unchanged): TtBA `ea6d1649fccda0eb8392c9171e43cc9e0974acf9`,
CGBA `73d3e9c94423b1503053258494a54ad4d2892efa`, HSJA `daecd5c7055d5214b39c34a7a28a98acd3557fbc`,
ARS-OPT `ab71cb88058ad0ca778e0048e5cf1b60921bb673`, Sign-OPT `65a82f8ea6beedc1b4339aa05b08443d5c489b8a`.

## 5. Test results (G1 gate)

```
python adapters/tests/test_g1_helpers.py   -> ALL PASS (9/9)
python adapters/tests/test_g1_tracker.py   -> ALL PASS (22/22)
python adapters/tests/test_g1_adapters.py  -> ALL PASS (8/8)
python tools/eval/selftest_evalkit.py      -> ALL PASS (20/20)
```

- `test_g1_tracker.py` covers the G1 criteria with a dummy linear model: one-hot output only, every query counted,
  budget inside a batch → `BudgetExhausted`, checkpoint = best at that query, carry-forward without back-fill,
  targeted start at query 0, `finalize()` writes contract-shaped rows + adv_store npz.
- `test_g1_helpers.py` asserts the contract checkpoints `[100,200,500,1000,2000,2500,5000,10000]` and budget 10000
  override the evalkit defaults, and the env/manifest round-trip + resume refusal.
- `test_g1_adapters.py` wires each adapter's `make_tracker` + `run_official` against a dummy cuda victim (tiny
  image/budget); every case reaches `BudgetExhausted` with a positive query count, and Sign-OPT correctly refuses
  targeted mode.

## 6. Open items / UNVERIFIED

1. **Contract vs B1 conflict (needs user decision).** `contract/paper_contract.yaml` lists `Sign_OPT.modes:
   [non_targeted, targeted]`, but the official `OPT_attack_sign_SGD` has no targeted mode
   (`attack/Sign_OPT.py:370` raises `NotImplementedError`). The adapter refuses `--mode targeted`. Per the contract
   skill rule 2, this is reported, not "fixed" by an agent: either the contract is amended (drop targeted for
   Sign_OPT) or Sign-OPT is noted as untargeted-only in the threat-model table (E16). No n = 200 result is opened
   before this is resolved.
2. **Query-count offsets to cross-check in B3.** The tracker counts every victim evaluation, which is the fairness
   rule (F5); the official counters differ slightly: TtBA's `Attacker.queries` misses the `__init__` `x0_label`
   and the final `out_label` evaluations (so `tracker.queries ≈ atk.queries + 1` under budget exhaustion), HSJA has
   no total counter, and CGBA/ARS-OPT/Sign-OPT keep internal counters. B3 records both per image and reports the
   difference.
3. **HSJA license**: none in the repo (B1) — not redistributed; pin commit + clone script only.
4. **ARS-OPT shims**: `glog`, `bidict`, `dataset.npz_dataset` are stubbed adapter-side because the clone ships
   without `dataset/npz_dataset.py`; the stub is only instantiated for `len()` (the attack is driven directly via
   `untargeted_attack`/`targeted_attack`, never through the dataset loader). Verified in the smoke test; the real
   fidelity check is G3.
5. Sign-OPT's ImageNet hyper-parameters are taken from the code defaults (the repo has only CIFAR/MNIST examples);
   the `query_limit` is the contract budget. No ImageNet default exists in code (B1) → UNVERIFIED against the paper.

## 7. Next phase (B3 — hard-label audit + fidelity, GPU minutes)

G2 (1 image/method: attack never sees logits, tracker count == official count, no query after 10 000) and G3
(3 images/method on ResNet-50 V2 both modes: adapter vs official L2 ratio within [0.90, 1.10] and same query count;
TtBA vs Table 1 magnitude). Estimated < 1 GPU-hour total. No long runs before B3 passes.

---
**Plain-English summary (5 lines).**
1. Wrote the five baseline adapters that run the official TtBA, CGBA/CGBA-H, HSJA, ARS-OPT and Sign-OPT code
   unchanged, wrapped so each attack sees only the top-1 decision and every query is counted against the 10 000 budget.
2. All methods use the same images, target pairs, victim weights and preprocessing; adversarial images are stored at
   the eight contract checkpoints so metrics are computed later without re-running anything.
3. Fixed four defects found by the smoke test (a device-mismatch bug in the CGBA victim wrapper, a missing
   constructor in the ARS-OPT stub, the Sign-OPT label-return shape, and the TtBA targeted start).
4. All four test suites pass: helpers 9/9, tracker bookkeeping 22/22, per-adapter smoke 8/8, evalkit self-test 20/20.
5. One decision is needed before the main runs: the paper contract lists Sign-OPT as having a targeted mode, but the
   official code is untargeted-only — the contract should be amended or Sign-OPT documented as untargeted-only.
