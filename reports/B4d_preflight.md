# B4d — Pre-B5 preflight (gate G4/G5 tooling, GPU ON)

Status: **COMPLETE (2026-10-13)**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, HEAD
`45514403b98932199784dfab58b284f6e9006578` ("paper contract v1.7: D11 CGBA stall decision tree"). Contract **v1.7**
(sha256 `05923ff049397f1912ee5677c0d77eb0b85351cf25823fae0dd2505e02e6753f`, matches). Engine sha `835f1933d9e5`
(frozen, unchanged). GPU: RTX 5090 (32 GB), torch 2.7.1+cu128, driver 595.71.05. Budget 10000, contract checkpoints
100/200/500/1000/2000/2500/5000/10000. **No B5 work started.**

Scope: (T1) precision parity, (T2) TtBA DataParallel diagnosis, (T3) concurrency neutrality + throughput, (T4) CGBA
stall diagnosis and the D11 verdict, (T5) register `ttba_table1.json`, (T6) the B5 G5 gate tool. No attack logic, seed
or hyper-parameter was changed; all new files are diagnostic under `adapters/diag/`.

---

## T1 — Precision parity (diagnostic only, nothing changed)

Values (torch 2.7.1+cu128 defaults, fresh process): `matmul.allow_tf32=False`, `cudnn.allow_tf32=True`,
`cudnn.deterministic=False`, `cudnn.benchmark=False`, `float32_matmul_precision=highest`.

The ONLY place any runner sets backend precision is `lgc_bench/env.py:configure_backend` (`lgc_bench/env.py:46-52`),
which sets `matmul.allow_tf32=allow_tf32`, `cudnn.allow_tf32=allow_tf32`, `cudnn.deterministic=deterministic`,
`cudnn.benchmark=False`. No runner sets `float32_matmul_precision`.

| runner | matmul.allow_tf32 | cudnn.allow_tf32 | cudnn.deterministic | cudnn.benchmark | fp32_matmul_precision | where set |
|---|---|---|---|---|---|---|
| `run_benchmark_33b.py` (EGC) | **False** | **False** | **True** | **False** | highest | `configure_backend(...)` @ `run_benchmark_33b.py:1157` (flags `--allow_tf32`/`--nondeterministic` @ :460-461, both default False; `run_frame_b1.sh` passes neither) |
| `ttba_github/run_frame_b1_generic.py` (EGC E10/E11) | **False** | **False** | **True** | **False** | highest | `configure_backend(device)` @ `run_frame_b1_generic.py:95` |
| `adapters/run_cgba.py` (CGBA/CGBA_H) | False (default) | **True** (default) | **False** (default) | **False** | highest | none in adapter (torch defaults) |
| `adapters/run_ttba_v2.py` (TtBA) | False (default) | **True** (default) | **True** | **False** | highest | `cudnn.deterministic=True` @ official `TtBA/dataset.py:220` (load_dataset_model) |
| `adapters/run_arsopt.py` (ARS_OPT) | False (default) | **True** (default) | **False** (default) | **False** | highest | none in adapter (torch defaults) |

**TF32 is DISABLED only in `lgc_bench/env.py:configure_backend`**, i.e. only by the two EGC runners. The TtBA paper
environment (torch 2.3.0, RTX 4090) ran with `cudnn.allow_tf32=True` (the PyTorch 2.3 default); our **TtBA adapter
does NOT disable TF32**, so it matches the paper default. The EGC engine's TF32-off + deterministic setting is the
frozen engine's documented configuration (`lgc_bench/env.py:5-8`: exact FP32, identical decisions to CPU runs; the
comment notes TF32 gives almost no speed-up for these models). This is **required for the frozen engine's
bit-identical reproducibility** (B0/B4c prove the engine is bit-stable) and is a property of the frozen engine, not a
fairness gap. `cudnn.benchmark=False` everywhere (T3 re-confirms). **Nothing changed.**

---

## T2 — TtBA DataParallel diagnosis (observation only)

Grep of the official clone `/root/autodl-tmp/TtBA` (@ `ea6d1649`):
- `dataset.py:220` `torch.backends.cudnn.deterministic = True`
- `dataset.py:290-293` (imagenet-resnet50) `model = models.resnet50().cuda(); model.load_state_dict(...);
  model = torch.nn.DataParallel(model, device_ids=[0])`  → **DataParallel is in the chain**
- `dataset.py:337` `return test_loader, torch_model.cuda()`
- `TtBA.py:22` `.cuda()`, `general_torch_model.py:31/33` `.cuda()` (mean/std tensors); no `.to(device)` usage.

`adapters/diag/diag_ttba_profile.py` on resnet50 NT row 1 (exact main.py path, unchanged):

| quantity | value |
|---|---|
| model chain at `predict_label` | `ClampedVictim → GeneralTorchModel → nn.DataParallel → torchvision ResNet` |
| DataParallel in call chain | **True** (created at `TtBA/dataset.py:293`) |
| per-query victim-forward wall time | median(200) = **2.589 ms**, median(all 10000) = **2.603 ms** |
| control: `DataParallel.forward` vs raw `resnet50.forward` (200 sync'd calls) | 2.902 ms vs 2.446 ms → **DataParallel overhead = 0.456 ms/call (15.7%)** |
| attack wall time (10 000 queries) | 36.2 s |

**Verdict:** the ~46 s/img (B4) vs historical ~19 s/img gap is **NOT explained by precision parity** (the TtBA
adapter matches the paper's `cudnn.allow_tf32=True` / `cudnn.deterministic=True`) and is **only marginally
DataParallel** (15.7% of the forward, 0.46 ms/call). The dominant cost is the per-query victim forward itself
(2.45 ms on RTX 5090/torch 2.7.1) plus the hard-label tracker's per-query bookkeeping (L2/argmax/core.process,
≈1 ms/query): 10 000 × ~3.6 ms ≈ 36 s + finalize (SSIM/advstore). The "19 s/img" dev figure is an unverified estimate
that does not account for the tracker wrapper. **Change nothing.**

---

## T3 — Concurrency neutrality + throughput (GPU)

Fresh solo references (single process) and a 2-process concurrent pair (TtBA resnet50 NT + LGC_RC_EGC resnet50 NT,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8`), same image = row 1.

**Identity (solo vs concurrent, row 1, all 8 checkpoints):** `adapters/diag/diag_b4d_t3_compare.py`
| method | verdict |
|---|---|
| TtBA | **IDENTICAL** (8/8 npz `delta` bitwise equal) |
| LGC_RC_EGC | **IDENTICAL** (8/8 npz `delta` bitwise equal) |

**Throughput (≥5 images each):**

| | TtBA (5 img) | EGC (5 img) | total wall |
|---|---|---|---|
| solo (sum) | 189.7 s | 106.8 s | 296.5 s |
| 2-process concurrent | 284.2 s (per-process sum) | 128.3 s | **290 s** wall |

combined images/s = 10/290 = 0.0345 vs solo sum 10/296.5 = 0.0337 → **throughput ratio ≈ 1.02× (well below 1.6×)**.

The B4 §3 1.72× figure was for a **balanced TtBA+TtBA** pair; the imbalanced **TtBA+EGC** pair is GPU-bound on both
sides and gives ~no speed-up (TtBA image 1 also showed a 122 s startup-contention anomaly; images 2–5 settle at
~38–46 s). **Verdict: FAIL the 1.6× threshold — B5 must run solo (one attack process per GPU).**

`cudnn.benchmark` == **False** in every runner (T1 table; EGC sets it False in `configure_backend`, the adapters leave
it at the torch default False). No kernel-switch risk under contention. **Change nothing.**

---

## T4 — CGBA stall diagnosis (D11)

### (a) Stall scan (zero GPU) — `adapters/diag/diag_b4d_t4a_stall.py`

`pix_l2@2000 == pix_l2@10000` ⇔ stalled. **7 unique stalled images, all CGBA non-targeted**:
- resnet50 NT **3/3** stalled (`state_query@10k` = 761 / 892 / 1144);
- Salman2020Do_R50 NT **3/3** stalled (155 / 672 / 450);
- resnet50_places365 NT **1/3** stalled (1344).

CGBA_H (targeted) **never** stalls; CGBA NT on vgg19 / vit_b_32 / densenet161 **never** stalls.

### (b) Boundary-walk trace — `adapters/diag/cgba_trace.py` (resnet50 NT rows 1–3)

Geometry is Alg.-1/Eq.-7-**consistent**: for every non-degenerate walk `r_i = cos(psi_i)` exactly
(`first_geometry_deviation(|r_i−cosψi|>1e-4) = None` for all 3 images). The first candidate (i=1, ψ=45°) is already
non-adversarial in every healthy walk (`break_i=1`), so the walk does the semi-circular search each time.

**The walk degenerates to NaN:** in each image the semi-circular search's `theta = acos(dot/(‖·‖·‖·‖))` receives an
argument of `1.0000001` (float rounding of `dot=1.0`) and `torch.acos` returns **NaN**
(`proposed_attack.py:182`). The NaN propagates `d = cos(theta)*norm_dis` (:183) → `x_mid` (:184) → returned boundary
point. From the next iteration `x_b` is NaN: every subsequent walk caps at 39 candidates (argmax of NaN logits returns
a fixed class, counted "adversarial") and the best-so-far is frozen. Capped walks with no best decrease: 45 / 43 / 41
for rows 1 / 2 / 3.

The m/zeta/p_near_boundary lines **next to Eq. (7)** are correct — units are **radians**, exponent **2^i**, sign of
cos(theta) **negative**:
```
proposed_attack.py:157  m = sin(theta)*cos((pi/2)*(1 - 1/2^num_calls)) / sin((pi/2)*(1 - 1/2^num_calls)) - cos(theta)
proposed_attack.py:158  zeta = (eta + m*v) / norm(eta + m*v)
proposed_attack.py:159  p_near_boundary = x_s + zeta*norm(x_b-x_s)*dot(v, zeta)
proposed_attack.py:160  p_near_boundary = clip_image_values(p_near_boundary, lb, ub)
```
`psi_i = (pi/2)*(1 - 1/2^i)` rad = `90 − 90/2^i` deg — matches the paper. `r_i = ||x_q−x_s||/||x_b−x_s|| = cos(psi_i)`.

### (c) Unpatched official code — `adapters/diag/diag_b4d_t4c_unpatched.py`

`proposed_attack.py:252` `assert self.all_queries == q_num` **raises** on the first capped walk (the D8 off-by-one:
the cap branch returns `num_calls==40` after only 39 `is_adversarial` calls):

| image | crash iter | q_num | all_queries | tracker best at crash |
|---|---|---|---|---|
| 00007450 | 8 | 994 | 993 | 85.0777 |
| 00015433 | 10 | 1309 | 1308 | 74.5685 |
| 00025151 | 12 | 1489 | 1488 | 54.7046 |

(Matches D8 / B3b T3; our `PatchedProposedAttack` corrects only the count, trajectory unchanged.)

### (d) `grep -rni cgba /root/autodl-tmp/TtBA` → **none** (no CGBA implementation or driver behind TtBA Table 1).

### (e) machanic cross-check (never reported as CGBA) — `adapters/diag/diag_b4d_t4e_machanic.py`

`machanic/hard_label_attacks` `CGBA/attack_gpu.py` @ `ab71cb88`, same victim/images/tracker/flags, resnet50 NT 1–3:

| image | machanic 2k/5k/10k | stalled | official (Farhamdur) 2k/5k/10k | stalled |
|---|---|---|---|---|
| 00007450 | 82.88 / 77.45 / 68.38 | **no** | 85.08 / 85.08 / 85.08 | yes |
| 00015433 | 77.02 / 75.20 / 68.17 | **no** | 74.57 / 74.57 / 74.57 | yes |
| 00025151 | 45.39 / 41.07 / 29.86 | **no** | 54.71 / 54.71 / 54.71 | yes |

The machanic reimplementation **does not stall**. Its `semi_circular_boundary_search` uses
`torch.acos(torch.clamp(…, -1.0, 1.0))` (`attack_gpu.py:263-264`), with the unclamped version commented out
(`:261-262`) — i.e. the reimplementation explicitly guards the same acos that crashes the official code.

### (f) D11 verdict — **(b): official code deviates from Alg. 1**

- Not (a): our adapter faithfully reproduces the official code (B3b T2/T3 trajectory bit-identical; the faithful
  model path was re-verified in B4c).
- Not (c): the geometry is Alg.-1-consistent, but the improvement halts because of a **numerical defect**, not
  because the geometry genuinely cannot improve (the machanic cross-check keeps improving to 30–68).
- **(b)** at `proposed_attack.py:182`: `theta = torch.acos(dot/(norm·norm))` **without clamping** to `[-1,1]`;
  float rounding yields `1.0000001 > 1`, `torch.acos` → NaN → NaN boundary point → frozen best-so-far.

**Minimal paper-faithful fix (apply NOTHING here):** clamp the acos argument
`torch.clamp(dot/(norm·norm), -1.0, 1.0)` at `proposed_attack.py:182` (exactly what the machanic reimplementation
does at `attack_gpu.py:263-264`), and optionally guard `mid_dir = mid_dir/norm` against zero norm. Disclosed as a
paper-faithful fix; official-code numbers kept in the appendix; **m = 3**.

---

## T5 — `ttba_table1.json` register (paper-faithful, no content change)

- File present and already tracked in the worktree (`git ls-files` → `ttba_github/ttba_table1.json`), committed in
  **`c823018`** ("B4c: commit ttba_table1.json (published TtBA Table 1, for E2 dagger block)"). Content unchanged
  since (`git status` clean), so no new commit is needed (an empty commit is forbidden).
- sha256 `2cb224f682f9f5735c219a3adeb8b513cbec033c903d4be14c864af682d9059e` (identical to the main-repo source).
- top-level keys `["format", "source", "table1"]`; `"table1"` victim keys
  `["efficientnet_b0", "inception_v3", "resnet50", "vgg19", "vit_b_32"]`; `"format"` =
  `[mean, median] L2 at the budget; 1000 images per model; their image list and environment (torch 2.3.0, RTX 4090) — NOT paired with our runs`.
- 72 G5 cells (CGBA-NT, CGBA-H-T, TtBA-NT, TtBA-T × resnet50/vgg19/vit_b_32 × 2k/5k/10k × mean/median) verified
  against the source: **max |Δ| = 0.0** over all 300 value-cells (bit-identical file). No cell differs.
- `adapters/diag/diag_t3a_magnitude.py` reads **`/root/autodl-tmp/LGC_frozen/ttba_github/ttba_table1.json`**
  (the worktree copy, not the main repo).
- JSON content **not modified**.

---

## T6 — G5 gate tool (new) — `adapters/diag/g5_table1.py`

```
python adapters/diag/g5_table1.py --out_dir <cell> --arch {resnet50|vgg19|vit_b_32} --mode {non_targeted|targeted}
```
Reads `<cell>/<mode>/per_checkpoint_*.csv` (method inferred from the file name), computes mean/median `pix_l2` at
2000/5000/10000 (rows with `found==1`), compares with `ttba_github/ttba_table1.json` (method ids
CGBA / CGBA-H / TtBA), prints `ratio(mean)` and `ratio(median)` per budget, and **PASS iff every ratio ∈ [0.5, 2]**
else FLAG. Exit 0 = PASS, 1 = FLAG, 2 = usage/read error. Writes nothing.

Smoke: CGBA resnet50 NT → **FLAG** (exit 1, ratio 3.85–29.84); TtBA resnet50 NT → FLAG (2.97–8.19); CGBA vgg19 NT →
PASS (exit 0); TtBA vgg19 T → PASS (exit 0).

---

## Files created/changed (sha256)

| file | sha256 |
|---|---|
| `adapters/diag/diag_b4d_t1_precision.py` | `f56f00b5c9892f805a7f538efdcd42c69669531405420024cd57cdb75da65259` |
| `adapters/diag/diag_ttba_profile.py` | `1c9c810b759205613cb1d7e01f02bc21a770d53f5bdb9d90853c85c59ee56c56` |
| `adapters/diag/diag_b4d_t3_compare.py` | `29c1150af5e53a280787b93f458b4682f454bb57f091a9133fa0890c56047cc9` |
| `adapters/diag/diag_b4d_t4a_stall.py` | `8dfd7bfd8e5f095e6abdcb0c9c64b2b687cb750689d3594b79d2532e25b2c385` |
| `adapters/diag/cgba_trace.py` | `12eb65f6a0cb11b1592ab64406cc3e2cfed492bc2c2c2b5ccef486dd0a875699` |
| `adapters/diag/diag_b4d_t4c_unpatched.py` | `fb91ace0e584c398c787d1e892ae0c854e57e204f8cb8742d7193fd6b5e74c54` |
| `adapters/diag/diag_b4d_t4e_machanic.py` | `447085cc668a8d9f1552e02898e9c28110c5bc29058a7c086e34ffb17edb686d` |
| `adapters/diag/g5_table1.py` | `df5050c1cd29a448542a4d1499b63f5f0c034505940395d870cbfe43607e86e4` |
| `ttba_github/ttba_table1.json` (unchanged, already committed c823018) | `2cb224f682f9f5735c219a3adeb8b513cbec033c903d4be14c864af682d9059e` |

No file under `lgc_bench/`, `run_benchmark_33b.py`, `ttba_github/run_frame_b1.sh`, the official clones, or
`contract/paper_contract.yaml` was modified. Frozen engine unchanged (`sha256sum lgc_bench/engine.py` = `835f1933d9e5…`).

## GPU minutes

≈ 25 GPU-minutes total: T2 ~1 min; T4(b) ~3 min, T4(c) ~1 min, T4(e) ~1.5 min; T3 ~18 min (solo TtBA 5 + solo EGC 5
+ three concurrent-pair launches, incl. two aborted/redone attempts). T1/T4(a)/(d)/T5/T6 are CPU-only.

---

## Summary

1. TF32 is disabled only by the frozen EGC engine (`lgc_bench/env.py:configure_backend`); the TtBA adapter matches the
   paper's `cudnn.allow_tf32=True` default — the 46 s vs 19 s TtBA gap is neither precision nor (mostly) DataParallel
   (15.7%), it is per-query forward + tracker bookkeeping.
2. 2-process (TtBA+EGC) is **bit-identical** to solo, but gives only **≈1.02×** throughput (< 1.6×) → B5 runs solo.
3. CGBA NT stall is **D11(b)**: unclamped `torch.acos` at `proposed_attack.py:182` → NaN boundary point; the machanic
   reimplementation clamps it and does not stall. Minimal fix = `torch.clamp(…,-1,1)`; nothing applied here.
4. `ttba_table1.json` is already committed (c823018), content bit-identical to source; `g5_table1.py` is ready for the
   B5 G5 gate (PASS ⇔ both ratios ∈ [0.5, 2], exit 0/1).

**STOP.** No B5 work started.
