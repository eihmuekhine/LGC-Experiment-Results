# B4f — Audit (T1–T7)

Status: **COMPLETE**. Worktree `/root/autodl-tmp/LGC_frozen`, branch `paper-runs`, base HEAD `69a763c`
(`B4e2 follow-up-2`). This phase is **read-mostly / CPU**, with GPU forward passes only in T3 and T6 (as
authorised). No B5 work started. Frozen engine `835f1933d9e5…` untouched.

## 1. Scope and prompt hash

- Prompt saved by the human at `reports/raw/B4f/prompt.txt`; sha256 (printed here):
  `66c94fbdc213483b235c16744e883719d1489b694fb5b43285d4aac15cdf9387`
- Contract `contract/paper_contract.yaml` **v1.8** sha256 (printed here):
  `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645`
- Study GPU in `reports/B4e2_close.md`: `NVIDIA GeForce RTX 5090`. It is **free** this session
  (`nvidia-smi` shows 0 MiB used), so the RTX 6000 fallback authorisation was **not needed**; T3/T6 ran on the
  RTX 5090 under D12 flags.

## 2. Environment block

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU) |
| driver / CUDA | 580.76.05 / 13.0 (nvidia-smi) |
| capability / SM count | (12, 0) / 170 |
| torch / cuda / cudnn | 2.7.1+cu128 / 12.8 / 90701 |
| backend flags (D12) | `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `float32_matmul_precision=highest` |
| worktree HEAD (base) | `69a763c0222c673d83f1139c721cd741fe2b34be` |
| engine sha | `835f1933d9e571070e952aedafc5a50b9948717168115ea60b5433277006c28a` |
| OMP/MKL/OPENBLAS threads | 8 (every GPU command) |

---

## T1 — Loader audit (L1)

### The single function that turns a file into x0

| dataset | function (file:line) | decoder | resize | crop | ToTensor | dtype |
|---|---|---|---|---|---|---|
| ImageNet (ttba_github) | `lgc_bench/baseline_protocol.py:325-333` `ttba_github_tensor` | `PIL.Image.open(f)` + `.convert("RGB")` | `T.Resize(256)` | `T.CenterCrop(224)` | `T.ToTensor()` | float32, [0,1] |
| Places365 | `adapters/_common.py:362-372` `load_image_places365` | `PIL.Image.open(fh)` + `.convert("RGB")` | `T.Resize(256)` | `T.CenterCrop(224)` | `T.ToTensor()` | float32, [0,1] |
| RobustBench pool | `adapters/_common.py:388-397` `load_x0_any` → `load_image` → `ttba_github_tensor` (`baseline_protocol.py:325`) | same as ImageNet | `T.Resize(256)` | `T.CenterCrop(224)` | `T.ToTensor()` | float32, [0,1] |

For the EGC ImageNet runner, the actual x0 is the same `ttba_github_tensor` via `run_benchmark_33b.py:696-718`
`load_x` (`ref = load_image(...)` is only a shape/dtype reference and is discarded). For the TtBA ImageNet path,
x0 comes from the official repo's own `ImageFolder` loader (`TtBA/dataset.py:173-182`, `Resize(256)→CenterCrop(224)→ToTensor`),
identical to `ttba_github_tensor` (confirmed by the x0-identity check, T3: 0 mismatches).

### Grep hits in every executed adapter / wrapper / runner / official-clone path

Search command: `grep -rn "Resize|resize|interpolate|CenterCrop|transforms|ToTensor|PIL.Image.open|read_image"` over
`adapters/`, `ttba_github/`, `tools/eval/`, `lgc_bench/`, `run_benchmark_33b.py`, and the executed official-clone paths
`TtBA/`, `baselines/CGBA/`, `baselines/ARS-OPT/PARS_OPT/`.

| hit | file:line | executed for | before / after x0 |
|---|---|---|---|
| `T.Resize(256)+CenterCrop(224)+ToTensor` (Compose) | `adapters/_common.py:367` | Places365 loader (all methods) | **before** (produces x0) |
| `PIL.Image.open(fh)` | `adapters/_common.py:370` | Places365 loader | **before** |
| `T.Resize(256)+CenterCrop(224)+ToTensor` (Compose) | `lgc_bench/baseline_protocol.py:330` | ImageNet/RobustBench loader (`ttba_github_tensor`) | **before** |
| `PIL.Image.open(f)` | `lgc_bench/baseline_protocol.py:332` | ImageNet/RobustBench loader | **before** |
| `T.Resize((224,224))+CenterCrop+ToTensor` (`baseline_tensor`) | `lgc_bench/baseline_protocol.py:272` | **NOT executed** (`--image_loader baseline` unused) | — |
| `_TF = {resize, center_crop}` Compose | `lgc_bench/data.py:21-22` | data-module loader `ref` only (shape/dtype check, discarded) | before (reference, not the x0) |
| `Image.open(path).convert("RGB")` | `lgc_bench/data.py:49` | data-module `ref` loader (discarded) | before |
| `transforms.Resize(256)/CenterCrop(224)/ToTensor` | `TtBA/dataset.py:176-178` | TtBA ImageNet `ImageFolder` loader (official x0) | **before** |
| `transforms.Resize(256)/CenterCrop(224)/ToTensor` | `TtBA/dataset.py:174-179` (CIFAR/MNIST branches :156-171 not executed) | TtBA ImageNet loader | **before** |
| `transforms.Resize((224,224))` / `CenterCrop(224)` / `Normalize` | `baselines/CGBA/Non_targeted_attack.py:78,84-88` + `Targeted_attack.py:86,94-98` | **NOT executed** — the adapter bypasses the driver; it imports only `proposed_attack.py` (which has no PIL/transforms) | — |
| `Image.open` + `img.resize((224,224))` | `TtBA/DataTools.py:10-14,20-21` | **NOT executed** — `save_and_resize_image`/`concatenate_images` are visualisation-only; only `cosine_similarity` (`DataTools.py:59`, used at `TtBA.py:185`) runs | — |
| `F.interpolate(...)` | `baselines/ARS-OPT/PARS_OPT/pars_opt_l2_norm_attack.py:303-304, 332, 338, 350, 709, 844` | **NOT executed** in prior-free `method='ARS'` — `:303-304` only via `get_prior_grad` (PARS-only, `:317-324`), `:332/338/350` gated by `self.resized=False`, `:709` gated by `best_initial_target_sample=False`, `:844` only in `attack_all_images` (never called) | — |

**Verdict: PASS.** No resize / crop / interpolate is applied *after* x0 is produced for any method. Every hit is
either the x0-producing loader itself or a code path that is not executed (CGBA driver preprocessing, TtBA
visualisation helpers, ARS-OPT PARS/resized branches).

---

## T2 — Normalisation constants (6 victims)

| victim | wrapper mean/std (file:line) | source mean/std | equal? |
|---|---|---|---|
| vgg19 (ImageNet) | `lgc_bench/baseline_protocol.py:56-57` (MEAN/STD, used by `_Norm.forward` :364-365) | torchvision `VGG19_Weights.IMAGENET1K_V1.transforms()` = (0.485,0.456,0.406)/(0.229,0.224,0.225) | **equal** |
| resnet50 (ImageNet) | `baseline_protocol.py:56-57` | `ResNet50_Weights.IMAGENET1K_V2.transforms()` = (0.485,0.456,0.406)/(0.229,0.224,0.225) | **equal** |
| vit_b_32 (ImageNet) | `baseline_protocol.py:56-57` | `ViT_B_32_Weights.IMAGENET1K_V1.transforms()` = (0.485,0.456,0.406)/(0.229,0.224,0.225) | **equal** |
| resnet50_places365 | `adapters/_common.py:60-61` (MEAN/STD, `NormalizedVictim.forward` :245-246) | official Places365 demo `run_placesCNN_unified.py` `returnTF()` `trn.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])` (also `reports/E10_sources.md`) | **equal** |
| densenet161_places365 | `_common.py:60-61` | same official Places365 `Normalize(...)` | **equal** |
| Salman2020Do_R50 (RobustBench) | robustbench `normalize_model` `ImageNormalizer` (`model_zoo/imagenet.py:16-17,33`; `utils_architectures.py:14-23`) | robustbench `imagenet.py:16-17` `mu=(0.485,0.456,0.406)`, `sigma=(0.229,0.224,0.225)` | **equal** |

The TtBA official path uses the same ImageNet constants at `TtBA/dataset.py:278-279,295-296,313-314` (`im_mean=[0.485,0.456,0.406]`, `im_std=[0.229,0.224,0.225]`). Verified at runtime (`torchvision … .transforms()` printed identical constants; robustbench `imagenet.py:16-17` read from the installed package). **Verdict: PASS.**

---

## T3 — x0 identity and pairs

Command: `python tools/eval/human_preflight.py --roots $(cat reports/B4e2_smoke_roots.txt) --out reports/raw/B4f/human_preflight`
→ `reports/raw/B4f/human_preflight.md` (FLAG lines = 7, unchanged from B4e2 — see T7 / Deviations).

**Section 2 (pasted):**

```
## 2. x0 identity (every method must attack the same clean tensor)

- (arch, image) keys: 18; with more than one x0 hash: **0**
- keys seen in only one root (cannot be cross-checked): 0
```

**GPU re-verification of the 12 frozen split CSVs (rows 1–200) under D12 flags** (`adapters/diag/diag_b4f_t3_pairs.py`,
log `reports/raw/B4f/T3_pairs.log`):

| file | n | wrong source | wrong target | same-label | dups |
|---|---|---|---|---|---|
| splits/imagenet/ttba_github/vgg19_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/imagenet/ttba_github/vgg19_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/imagenet/ttba_github/resnet50_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/imagenet/ttba_github/resnet50_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/imagenet/ttba_github/vit_b_32_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/imagenet/ttba_github/vit_b_32_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/places365/resnet50_places365_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/places365/resnet50_places365_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/places365/densenet161_places365_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/places365/densenet161_places365_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/robust/Salman2020Do_R50_non_targeted.csv | 200 | 0 | 0 | 0 | 0 |
| splits/robust/Salman2020Do_R50_targeted.csv | 200 | 0 | 0 | 0 | 0 |

Files with any violation: **0/12**. **Verdict: PASS.**

---

## T4 — Hyper-parameter audit (L2/L3)

`adapters/diag/hparam_audit.py` reads the official defaults from the clones (ast-parsed `__init__` signatures +
`PARS_OPT.json`) and the adapter values from the adapter `PARAMS`/`BASE_PARAMS` dicts + constructor calls, and writes
`reports/B4f_hparam_audit.md` (full tables; reproduced below in condensed form).

Extracted official defaults (evidence): TtBA `Attacker.__init__` = `{dim_reduc_factor=4, iteration=100, initial_query=30, tol=1e-4, sigma=3e-4}` (`TtBA.py:18-20`); CGBA `Proposed_attack.__init__` = `{dim_reduc_factor=4, attack_method='CGBA_H', iteration=93, initial_query=30, tol=1e-4, sigma=2e-4, verbose_control='Yes'}` (`proposed_attack.py:13-16`) + `grad_estimator_batch_size=40` (`:32`); ARS-OPT `ParsOptL2Norm.__init__` = `{k=100, alpha=0.2, beta=0.001, iterations=1000, maximum_queries=10000, method='ARS', clip_grad_max_norm=1.0, dim_reduce_factor=4, resized=False, line_search=False, ...}` (`pars_opt_l2_norm_attack.py:16-18`) + `PARS_OPT.json` ImageNet.l2 `{epsilon=10.0, tol=1e-4, lr 0.4(NT)/0.2(T)}`.

Key findings (full table in `reports/B4f_hparam_audit.md`):

- **TtBA**: all L2 internals at official defaults (relied, not overridden). `budget=10000`, `early=0` = official
  main.py defaults. **DIFFERS**: `epsilon=2.5` (adapter `run_ttba_v2.py:242`) vs main.py default `0.3` (`main.py:21`) —
  the ImageNet L2 success threshold (contract `asr_l2` 2.5/12); because `early=0`, epsilon only sets the recorded
  `success_eps` flag, never the trajectory.
- **CGBA / CGBA_H / CGBA_D8only**: all L2 internals at official ImageNet defaults (`iteration=93`, `dim_reduc_factor=4`,
  `tol=1e-4`, `sigma=2e-4`, `initial_query=30`, `grad_estimator_batch_size=40`, ImageNet mean/std, `lb/ub=normalize(0)/normalize(1)`).
  L3: `verbose_control='No'` (print-only); faithful model (normalised tensor → un-normalised net). L4 (contract):
  D8 count fix + D11 acos clip.
- **ARS-OPT**: all L2 internals at official defaults (`epsilon=10.0`, `k=200` from `attack.py:62 --est-grad-samples`,
  `lr 0.4/0.2`, `beta=0.001`, `iterations=1000`, `clip_grad_max_norm=1.0`, `tol=1e-4`, `dim_reduce_factor=4`,
  `method='ARS'`, `resized/line_search/best_initial_target_sample=False`). L3: `maximum_queries=10000` (contract F4;
  official driver forces 20000 for ImageNet targeted at `attack.py:127-128`), model = tracker, data = shared split,
  import shims.

**Verdict: PASS** (no L2 parameter differs from the official default without an L4 contract decision; all L3 changes
are interface-only and each has a documented reason).

---

## T5 — Oracle and query audit

### (a) Every place the official code evaluates the victim, and tracker-only proof

All three baselines receive the **tracker** as their `model`/`self.model`; every victim evaluation below therefore
reaches the victim *only* through `HardLabelTracker` (which returns only the top-1 decision — one-hot via `__call__`
`hardlabel_tracker.py:151-157`, labels via `predict_label` `:161-163` — and counts each image element-wise).

| method | victim-eval sites (file:line) | proof the victim is reachable only through the tracker |
|---|---|---|
| TtBA | `TtBA.py:23` (ctor `x0_label`), `:65`, `:78` (tangent), `:126` (normal), `:164` (TtBA via `is_adversarial`), `:245` (attack end), `:260` (`is_adversarial`), `:296` (`bin_search_fast`), `:308` (`TtBA_simp`) — all `self.model.predict_label(...)` | adapter builds `C.HardLabelTracker(C.ClampedVictim(torch_model), …)` and passes it as `model` to `SnapAttacker(args, tracker, x0, …)` (`run_ttba_v2.py:178,370` / `:184,376`); `self.model` is the only model reference inside `TtBA.py` |
| CGBA / CGBA_H | `proposed_attack.py:19` (ctor `src_lbl`), `:22` (ctor `tar_lbl`), `:44` (`is_adversarial`), `:106` (batched `normal_vector_approximation_batch`) — all `self.model.forward(...)`/`self.is_adversarial(...)` | adapter builds `PatchedProposedAttack(tracker, src, mean, std, lb, ub, …)` (`run_cgba.py:180-184`); `self.model = tracker` |
| ARS-OPT | `pars_opt_l2_norm_attack.py:92,100,111,120` (local binsearch), `:137,145,156,165` (local targeted), `:181,194` / `:207,220` (fine binsearch), `:254,258,261,264,270,273` (g-function), `:372,376` (v_t estimator), `:596,667` (untargeted), `:683,711,802` (targeted) — all `self.model(x).max(1)[1]` | adapter builds `ParsOptL2Norm(tracker, [], "ImageNet", …)` (`run_arsopt.py:109-115`); `self.model = tracker` |

### (b) Termination / early-stop / iteration-cap conditions

| method | condition (file:line) | neutralisation (L3) or why it cannot fire before the budget |
|---|---|---|
| TtBA | `for i in range(self.Max_iter)` (`TtBA.py:204`, `Max_iter=iteration=100`); `if self.queries >= budget: break` (`:237-238`); `if early==1 and success==1: break` (`:242-243`); `find_init_adv_x` unbounded `while` (`:275`); `bin_search_fast` `while l2 diff>tol` (`:293`); `TtBA`/`TtBA_simp` `while abs(R-L)>1e-3` (`:162`,`:306`) | `early=0` (adapter) ⇒ the success early-stop can never fire; the tracker raises `BudgetExhausted` (`hardlabel_tracker.py:131,147`) which is the binding 10 000 budget; `Max_iter=100` with ~`Init_query*sqrt(i+1)` per iteration reaches 10 000 first |
| CGBA / CGBA_H | `for i in range(self.iteration)` (`:233`, `iteration=93`); `bin_search` `num_calls>=100` break (`:83`); `go_to_boundary_*` `num_calls>=40` return (`:144`,`:164`); `SemiCircular_boundary_search` `num_calls>100` break (`:194`); `find_random_adversarial` unbounded `while` (`:61`) | no success-threshold stop exists (B1:47); 93 iterations × `int(30*sqrt(i+1))` (+ boundary searches) ≫ 10 000, so the tracker budget binds before `iteration`; the inner caps are the official boundary-walk geometry, not success stops |
| ARS-OPT | `for i in range(num_directions=100)` (`:593`) init scan; `for i in range(self.iterations=1000)` (`:618`,`:750`) with `if query>=maximum_queries: break` (`:619`,`:751`); binsearch `count>=200` caps (`:198`,`:224`) | `maximum_queries=10000` (adapter, contract F4) binds long before `iterations=1000` (≈200 queries/iteration ⇒ ≈50 iterations); no success-threshold stop exists |

### (c) Evaluation-only victim calls by the adapter, and invisibility

| method | eval-only call (file:line) | proof the attack cannot see it |
|---|---|---|
| TtBA | `pred = torch_model.predict_label(x0.cuda())` on the **raw** model (`run_ttba_v2.py:171` generic / `:363` ImageNet) | computed before the tracker is built; result is written only to the `x0_pred_ok` column of `runs_TtBA.csv` *after* `atk.attack()` returns (`:221`,`:413`) — never passed to `SnapAttacker` |
| CGBA / CGBA_H / CGBA_D8only | none in the adapter loop; the shared `tracker.finalize(..., eval_model=victim)` (`_common.py:591-592`) computes the uint8 validity (`adv_uint8`, `hardlabel_tracker.py:198-200`) | runs after `run_official` returns/raises; calls the raw victim directly (not `self.labels`), so it is uncounted and the finished attack object never receives it |
| ARS-OPT | none in the adapter loop; same shared `finalize(eval_model=victim)` | same as above |

### (d) Query accounting per cell (B4e2 smoke; budget 10000) — log `reports/raw/B4f/T5_queries.log`

| method | n/cell | ==10000 | <10000 (status) | >10000 |
|---|---|---|---|---|
| TtBA (6 ImageNet + 6 E10/E11 cells) | 3 each | 3/3 | — | 0 |
| CGBA / CGBA_H (6 + 6 cells) | 3 each | 3/3 | — | 0 |
| ARS_OPT (6 cells) | 3 each | 3/3 | — | 0 |
| LGC_RC_EGC (ours; 9 + 9 cells) | 3 each | 0 | 3/3 `ok` (9973–9976) | 0 |

**Verdict:** every baseline spends exactly the budget (tracker raises `BudgetExhausted`); nothing exceeds it;
our engine (engine-state convention) stops just below budget with `status=ok`. **PASS.**

### (e) Official counter vs tracker count (TtBA; `queries`=tracker, `total_queries`=official `atk.queries`)

Tracker is always exactly 10 000; official counter is 9 779–9 999, i.e. the **documented offset** `tracker − official`
ranges **1 … 221** across the 36 TtBA smoke images (full per-image table in `reports/raw/B4f/T5_queries.log`). The
tracker is the binding counter (F5); the official counter is strictly ≤ the tracker because the attack's own
`self.queries` stops at its internal `>= budget` check while the tracker still counts the final partial batch.
CGBA/ARS-OPT keep their internal counters only inside the algorithm; the recorded `queries` column is the tracker.

---

## T6 — Domain and clipping

Log `reports/raw/B4f/T6_domain.log`.

| method | attack space | bounds (file:line) | clamp (file:line) | to_pixel |
|---|---|---|---|---|
| TtBA | pixel [0,1] | `[0,1]` via `GeneralTorchModel.predict_label` clamp (`general_torch_model.py:53`) | `general_torch_model.py:53` (`torch.clamp(images,0,1)`); adapter `ClampedVictim` (`_common.py:249-260`) | `to_pixel_clamp` (`_common.py:284-286`) |
| CGBA / CGBA_H / CGBA_D8only | ImageNet-normalised | `lb=normalize(0)`, `ub=normalize(1)` (`run_cgba.py:178-179`; drivers `valid_bounds(delta=255)`) | `clip_image_values` (`utils.py:6-10`) at `proposed_attack.py:65,140,160,264` | `to_pixel_unnormalize` (`_common.py:289-293`) |
| ARS-OPT | pixel [0,1] | `images + gg*xg` (epsilon=10.0 gate only, not a hard clamp) | no in-loop clamp; tracker `to_pixel_clamp` clamps for the decision | `to_pixel_clamp` (`_common.py:284-286`) |
| LGC_RC_EGC | latent (decoder → pixel [0,1]) | ambient residual space, decoder clamps to [0,1] | `space.to_image` (frozen engine) | identity (already pixel) |

**d_n/d_p for the normalised-space method (CGBA), 3 stored points** (must lie in [1/sd_max, 1/sd_min] = [4.367, 4.444]):

| point | d_p | d_n | d_n/d_p | in [4.367,4.444]? |
|---|---|---|---|---|
| ILSVRC2012_val_00007450/CGBA_Q10000 | 6.449778 | 28.535267 | 4.424225 | yes |
| ILSVRC2012_val_00015433/CGBA_Q10000 | 6.331733 | 28.012817 | 4.424194 | yes |
| ILSVRC2012_val_00025151/CGBA_Q10000 | 2.605664 | 11.527633 | 4.424066 | yes |

**max |victim(x0) − raw_net((x0−mu)/sd)| for all 6 victims** (must be 0):

| victim | raw net | max abs diff |
|---|---|---|
| vgg19 | torchvision VGG | 0 |
| resnet50 | torchvision ResNet | 0 |
| vit_b_32 | torchvision VisionTransformer | 0 |
| resnet50_places365 | torchvision ResNet | 0 |
| densenet161_places365 | torchvision DenseNet | 0 |
| Salman2020Do_R50 | torchvision ResNet | 0 |

**Verdict: PASS** (all 3 ratios inside [4.367, 4.444]; all 6 victims bit-identical to `raw_net((x0−mu)/sd)`).

> **Erratum (B4h, contract v1.9 D19 — added 2026-10-16).** The ARS-OPT and CGBA/CGBA_H rows of the table above are
> superseded. This T6 audit described the pre-fix state, where the *decision* and the *stored/scored image* could be
> two different tensors: ARS-OPT queried unclamped `[0,1]` pixels (tracker clamped for the decision via
> `to_pixel_clamp` but the attack loop itself never clipped), and CGBA/CGBA_H `bin_search` (`proposed_attack.py:77-85`)
> and `SemiCircular_boundary_search` (`:179-191`) probed in ImageNet-normalised space *without* clipping (the
> `clip_image_values` at `proposed_attack.py:65,140,160,264` does NOT cover those two boundary searches). The D19
> oracle-domain fix (B4h-T3) wraps the tracker victim in `ClampedVictim` (ARS-OPT, `oracle_domain = "clamp01"`) and
> `ClampedVictimNormalized` (CGBA/CGBA_H, `oracle_domain = "clamp_normalized"`) so every query is clamped to the exact
> `[0,1]` pixel image that is stored and scored. The domain-consistency gate (`adapters/diag/domain_consistency.py`)
> verifies X = 0 on all affected cells (`reports/B4h_t3_fix_and_gate.md`). The raw-net identity and d_n/d_p results
> above are unchanged by D19.

---

## T7 — Env recording

### Code changes (allowed edits, env-recording only)

1. `adapters/_common.py` — new `backend_flags()` helper (returns the five flags read from torch at run time under the
   literal `backend.*` keys, using `torch.get_float32_matmul_precision()` because `torch.backends.float32_matmul_precision`
   does not exist in torch 2.7) and `build_env()` now emits them (plus `float32_matmul_precision` in the `precision`
   dict). This covers CGBA / CGBA_H / CGBA_D8only and ARS_OPT env files (all go through `C.run → build_env`).
2. `ttba_github/run_frame_b1_generic.py` — `env_frame.txt` now writes the five `backend.*` lines. This covers the six
   EGC E10/E11 generic-runner roots (the B4e2 "FLAG: not recorded … cudnn_tf32, deterministic" gap).

Verified: `python -m py_compile adapters/_common.py ttba_github/run_frame_b1_generic.py` → OK; `C.backend_flags()` →
`{'backend.cuda_matmul_allow_tf32': False, 'backend.cudnn_allow_tf32': False, 'backend.cudnn_deterministic': True,
'backend.cudnn_benchmark': False, 'backend.float32_matmul_precision': 'highest'}`.

### `run_frame_b1.sh` (frozen) — where the precision flags are recorded instead

The frozen EGC runner (`run_frame_b1.sh` → `run_benchmark_33b.py`) does **not** put backend flags in `env_frame.txt`;
it records them inside **`<out>/<mode>/config_single.json`**, in the `device` field, which is produced by
`describe_device` (`lgc_bench/env.py:114-123`, the `tf32=… deterministic=…` token at `:122`) and written at
`run_benchmark_33b.py:1265-1267`. Example (`Results/b4_smoke_egc_resnet50/non_targeted/config_single.json`):

```
"device": "device=cuda | torch=2.7.1+cu128 | gpu=NVIDIA GeForce RTX 5090 (31 GB) | tf32=False deterministic=True",
```

### Re-run human_preflight section 1

`reports/raw/B4f/human_preflight.md` §1 (re-run this phase) still reports **FLAG = 7**, unchanged from B4e2: the 6
EGC E10/E11 generic-runner roots (their `env_frame.txt` were written by the pre-fix code) plus the 1 GPU-name
mismatch from the frozen `config_single.json` device string (the `(31 GB)` suffix, not the GPU itself). These are
**historical** env files; they are not regenerated because that would require re-running attacks (rule 6: never
re-run for a metric). The code fixes above guarantee that every B5 run records all five flags. Remaining gap: the
**TtBA** env files are written inline by `run_ttba_v2.py` (not via `build_env`), which is *outside* the B4f
allowed-edit list, so `env_TtBA.json` records the four D12 flags under `precision` but not `float32_matmul_precision`
(see Deviations).

---

## 4. Deviations from the prompt

1. **TtBA env does not record `float32_matmul_precision`.** `run_ttba_v2.py` builds its own `env` dict inline (its
   `precision` block has `cuda_matmul_allow_tf32`, `cudnn_allow_tf32`, `cudnn_deterministic`, `cudnn_benchmark` only)
   and is **not** in the B4f allowed-edit list (`adapters/diag/*`, `adapters/_common.py`, `run_frame_b1_generic.py`,
   `reports/*`). `float32_matmul_precision` stays at its torch default `"highest"` (D12 does not change it), so this is
   a recording-completeness gap, not a data-validity issue; it is flagged for a follow-up edit to `run_ttba_v2.py`.
2. **GPU-name mismatch FLAG (T7 §1).** The frozen `config_single.json` device string `NVIDIA GeForce RTX 5090 (31 GB)`
   is parsed by `human_preflight` as a second GPU name. It is the study GPU plus a memory suffix, not a different GPU
   (documented in B4e2; not a code fix per the frozen-runner constraint).
3. **The six EGC E10/E11 `env_frame.txt` still FLAG in the re-run §1** because those files predate this phase's fix;
   they are historical records and are not rewritten (rewriting would require re-running attacks, which is forbidden).
   The generic runner now records the flags for every future run.
4. **`float32_matmul_precision` read via `torch.get_float32_matmul_precision()`**, not the `torch.backends` attribute
   named in skill §4 (which does not exist in torch 2.7.1); the value recorded is identical (`"highest"`).
5. **Places365 normalisation source** is quoted from the live official `run_placesCNN_unified.py` (fetched this
   session) plus `reports/E10_sources.md`; the demo's *loader* uses `Resize((224,224))` (aspect-distorting) whereas our
   loader uses `Resize(256)→CenterCrop(224)` (U4 fix) — only the normalisation constants are shared.
6. **`hparam_audit.py` ast note:** ARS-OPT `ParsOptL2Norm.__init__` class default is `k=100`, but the official ImageNet
   driver overrides it to `200` via `attack.py:62 --est-grad-samples`; the audit uses `200` (the executed value).
7. No RTX 6000 used (RTX 5090 was free); the fallback authorisation was not exercised.

## 5. sha256 of created/changed files; GPU minutes

| file | sha256 |
|---|---|
| `adapters/_common.py` (edited, T7) | `f15122bd70bb79f0ef3608c7fc25059c33a8bb2acbfef8cb58a2761a85ebd277` |
| `ttba_github/run_frame_b1_generic.py` (edited, T7) | `c8bbd153d67354c9bb10562e68f8c757eb6c472a0cb92fec38f25310ebd2737f` |
| `adapters/diag/diag_b4f_t3_pairs.py` (new) | `2b72e4b4c3ff7344c76d84c40a1a6463c6ed48984cc3167c2b54e78235b916d8` |
| `adapters/diag/diag_b4f_t6_domain.py` (new) | `6cd3ac7f1a759185dbe3ba81cde1904a2b958aea2cbebc6fb34ee2da5d5d74a7` |
| `adapters/diag/diag_b4f_t5_queries.py` (new) | `b5bff2f8c7c74d1d29acc82735eab0ab9f2a4fc7e7c3ddc023e32ac6b8aa9501` |
| `adapters/diag/hparam_audit.py` (new) | `99238186738e76fb84074dd080c0a4b828962ba50d73339df1641423a26c3046` |
| `reports/B4f_hparam_audit.md` (generated) | `86a64ab90688b662b07dc2d9efe17cf81d19950314c9ac4c2d117bafb0a992a3` |
| `reports/raw/B4f/human_preflight.md` (+12 grid PNGs) | `b6c7a0b5a1c0c7525f90732d130868c8ac2e4e8220d1cfefa897a4f93da19f3c` |
| `reports/raw/B4f/T3_pairs.log` | `9279831ea37c9f684166c73085112e4208a58b761e502f00ecaa057e516c2dc0` |
| `reports/raw/B4f/T5_queries.log` | `37fe55ea99786d111573f0ca905c452b8a07b649bc9ce88e3a4cc190fd30af31` |
| `reports/raw/B4f/T6_domain.log` | `453696c862097a0351512df1cf6a51c9df8385cf2e6fb68dd1d811e9feb5dbd5` |
| `reports/raw/B4f/prompt.txt` | `66c94fbdc213483b235c16744e883719d1489b694fb5b43285d4aac15cdf9387` |

GPU minutes: ≈ 3 min of attack/forward time (T3 pairs 12×200 sources + 1200 targets; T6 d_n/d_p + 6 victim forward
passes). Frozen engine untouched (`835f1933d9e5…`).

## 6. Summary (plain English)

1. Loaders: the only resize/crop/ToTensor are in the x0-producing loaders (one per dataset); no method resizes or
   crops *after* x0, and every `interpolate`/driver-preprocessing hit is in a non-executed branch (ARS-OPT PARS/resized
   mode, CGBA driver, TtBA visualisation helpers). PASS.
2. All six victims normalise with ImageNet mean/std, matching `weights.transforms()` / the Places365 and RobustBench
   sources exactly. PASS.
3. x0 is identical across methods (0 hash mismatches); all 12 split CSVs (rows 1–200) re-verified on the GPU: 0 wrong
   source, 0 wrong target, 0 same-label, 0 duplicate sources. PASS.
4. Hyper-parameters: every baseline is at its official ImageNet defaults; the only differences are the documented L3
   interface changes and the two contract L4 corrections (CGBA D8/D11), plus TtBA `epsilon=2.5` (success flag only).
5. Every victim evaluation reaches the model only through the decision-only tracker; budgets bind at 10 000 (baselines
   spend exactly 10 000, ours stops below); the only evaluation-only calls are invisible to the attack.
6. Domain/clipping and the normalised↔pixel distance ratio (d_n/d_p ∈ [4.367,4.444]) and
   `victim(x0)==raw_net((x0−mu)/sd)` all pass; the two env-recording gaps (TtBA `float32_matmul_precision`, and the
   pre-existing EGC generic/frozen env files) are documented as deviations and fixed in code for all future runs.
