# B4h-T1a — triage addendum to the T1 STOP bundle (read-only, no fix)

Sub-report of the existing STOP bundle `reports/stop/B4h_T1_20261015-0739.md` (E-OTHER). No `reports/stop/`
file was written here; nothing was edited and nothing was committed.

## 1. Scope and prompt hash

- Prompt: `reports/raw/B4h/prompt_T1a.txt`, sha256 `780a9e65c4529a36b312b0afef6430bd00cdf5643bf664833a515220f016c81a`.
- Parent prompt: `reports/raw/B4h/prompt.txt`, sha256 `733cae11e8edc469cb055202e3d7a8ce422afd1f70ab6be32e0edd4b1dfcfa08`.
- Contract `contract/paper_contract.yaml` sha256 `95d10f8a15f3477971e3aa3d86901b2086c88df8174d4741bed73bd699152645` (v1.8, matches the STOP bundle).
- Worktree `/root/autodl-tmp/LGC_frozen`, HEAD `3f9bcb444b8a8dbd1d47ac51200cbba4b0a7b85f` (matches the STOP bundle; not changed).

## 2. Environment block

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 5090` (STUDY_GPU) |
| driver | 595.71.05 |
| torch / cuda | 2.7.1+cu128 / 12.8 |
| capability / SM count | (12, 0) / 170 |
| backend flags (D12) | matmul_tf32=False cudnn_tf32=False deterministic=True benchmark=False float32_matmul_precision=highest |
| OMP/MKL threads | 8 (exported for every GPU command) |
| worktree HEAD | `3f9bcb444b8a8dbd1d47ac51200cbba4b0a7b85f` |
| CGBA official clone | `/root/autodl-tmp/baselines/CGBA` @ `73d3e9c94423b1503053258494a54ad4d2892efa` (env_CGBA.json:5-6) |

## 3. A1 — Budget-stratified persistence of the flagged (non-ARS_OPT) X rows

Rows with `class == X` at ANY checkpoint, excluding ARS_OPT (ARS_OPT X is the expected pre-fix state; it is out of
scope of the STOP). There are exactly 6 such rows over 4 (method, cell, image) triples. Class at every contract
checkpoint (100/200/500/1000/2000/2500/5000/10000), read from `reports/raw/B4h/t1_rows.csv`:

| method | cell (arch/mode) | image | @100 | @200 | @500 | @1000 | @2000 | @2500 | @5000 | @10000 |
|---|---|---|---|---|---|---|---|---|---|---|
| LGC_RC_EGC | vgg19/targeted | ILSVRC2012_val_00002148 | X | X | C | C | C | C | C | C |
| LGC_RC_EGC | Salman2020Do_R50/targeted | ILSVRC2012_val_00046453 | Q9 | Q9 | C | X | X | Q9 | C | Q9 |
| CGBA | resnet50/non_targeted | ILSVRC2012_val_00007450 | C | C | Q6 | X | Q6 | Q9 | Q6 | Q9 |
| CGBA | vit_b_32/non_targeted | ILSVRC2012_val_00046402 | C | X | Q9 | Q9 | Q9 | C | C | C |

Per flagged row, resolution at the final checkpoint (budget 10000):

| flagged row | class@10000 | resolved to C? |
|---|---|---|
| LGC_RC_EGC vgg19/targeted 00002148 @100 (X) | C | **resolved** (C from 500 onward) |
| LGC_RC_EGC vgg19/targeted 00002148 @200 (X) | C | **resolved** |
| LGC_RC_EGC Salman2020Do_R50/targeted 00046453 @1000 (X) | Q9 | **NOT resolved** (remains Q9; non-monotonic C/X/Q9) |
| LGC_RC_EGC Salman2020Do_R50/targeted 00046453 @2000 (X) | Q9 | **NOT resolved** |
| CGBA resnet50/non_targeted 00007450 @1000 (X) | Q9 | **NOT resolved** (remains Q9; non-monotonic C/Q6/X/Q9) |
| CGBA vit_b_32/non_targeted 00046402 @200 (X) | C | **resolved** (C from 2500 onward) |

So 3 of the 6 flagged rows resolve to C at the final checkpoint; 3 do not (2 LGC_RC_EGC Salman + 1 CGBA resnet50
remain Q9 at 10000). The X is not a one-off early-budget artefact in every case.

## 4. A2 — Full-population class mix at low budgets (CGBA and LGC_RC_EGC)

Population = ALL `found == 1` rows in `reports/raw/B4h/t1_rows.csv` (the diagnostic only writes found=1 rows), i.e.
the whole low-query population, not the 6 flagged examples. `C`/`Q9|Q6`/`X` per budget, with Wilson 95% CI on the X
rate.

**CGBA** (6 cells × 3 images = n = 18 per budget):

| budget | n | C (frac) | Q9/Q6 (frac) | X (frac) | X 95% CI |
|---|---|---|---|---|---|
| 100 | 18 | 12 (66.7%) | 6 (33.3%) | 0 (0.0%) | [0.000, 0.176] |
| 200 | 18 | 12 (66.7%) | 5 (27.8%) | 1 (5.6%) | [0.010, 0.258] |
| 500 | 18 | 7 (38.9%) | 11 (61.1%) | 0 (0.0%) | [0.000, 0.176] |
| 1000 | 18 | 5 (27.8%) | 12 (66.7%) | 1 (5.6%) | [0.010, 0.258] |

**LGC_RC_EGC** (12 cells × 3 images = n = 36 per budget):

| budget | n | C (frac) | Q9/Q6 (frac) | X (frac) | X 95% CI |
|---|---|---|---|---|---|
| 100 | 36 | 15 (41.7%) | 20 (55.6%) | 1 (2.8%) | [0.005, 0.142] |
| 200 | 36 | 22 (61.1%) | 13 (36.1%) | 1 (2.8%) | [0.005, 0.142] |
| 500 | 36 | 25 (69.4%) | 11 (30.6%) | 0 (0.0%) | [0.000, 0.096] |
| 1000 | 36 | 24 (66.7%) | 11 (30.6%) | 1 (2.8%) | [0.005, 0.142] |

Interpretation for the E8 tool (T7, `make_low_query_table.py`, contract E8_low_query_table = "found rate, ASR at
RMSE 0.1/0.05/0.01, median pix_l2 and median LPIPS-vgg at 100/200/500/1000"): the 6 flagged rows are a strict subset
of the population. The TRUE X rate is 0–5.6% for CGBA and 0–2.8% for LGC_RC_EGC at low budgets (0 at 100/500 for
CGBA, 0 at 500 for LGC_RC_EGC). Any ASR/found-rate statistic the E8 tool computes on the stored images would inherit
a downward bias of at most this order at the affected checkpoints, plus the Q9/Q6 fraction (float16-storage effects,
not defects per the protocol) which would also score as "not adversarial at s=1" if the tool only tests the stored
reconstruction. These rates are point estimates on n=18 / n=36; CIs are wide.

## 5. A3 — CGBA oracle-domain evidence (quoting only)

The CGBA cell actually executed is `adapters/run_cgba.py` with `--cgba_variant fixed` (env_CGBA.json:67) on the
official clone `/root/autodl-tmp/baselines/CGBA` @ `73d3e9c…`; the tracker is `tools/eval/hardlabel_tracker.py`.

**(a) where the attack/tracker decides "adversarial"**

Official attack — `proposed_attack.py:43-53`:
```python
def is_adversarial(self, image):
    predict_label = torch.argmax(self.model.forward(Variable(image, requires_grad=True)).data).item()
    self.all_queries += 1
    if self.tar_img == None:
        is_adv = predict_label != self.src_lbl
    else:
        is_adv = predict_label == self.tar_lbl
```
(batch estimator also calls `self.model.forward` and argmax at `proposed_attack.py:106`.)

Tracker (the `self.model` above IS the `HardLabelTracker`) — `hardlabel_tracker.py:133-139`:
```python
logits = self.model(xb)
self._ncls = int(logits.shape[1])
lab = logits.argmax(dim=1)
pix = self.to_pixel(xb).float()
l2 = (pix - self.x0.to(pix.device)).flatten(1).norm(dim=1)
adv = (lab == self.y_tgt) if self.y_tgt is not None else (lab != self.y_src)
```
The tracker's `self.model` for CGBA is the RAW (un-normalised) net: `run_cgba.py:79-83`
```python
def _faithful_model(victim):
    return C.raw_net(victim)
```
with `make_tracker(..., to_pixel=C.to_pixel_unnormalize, ...)` at `run_cgba.py:82-85`; `C.raw_net` returns `.m`/`.net`
(`adapters/_common.py:173-189`), which for the TtBA victims is `_Norm.m` (`baseline_protocol.py` builds `_Norm(net)`
with `self.m = m`).

**(b) tensor space of that call's input**

ImageNet-normalised. `run_cgba.py:176` `src = C.normalize(x0)` and `:178-179` `lb = C.normalize(zeros)`,
`ub = C.normalize(ones)`; the official driver feeds the normalised tensor into an un-normalised net
(`Non_targeted_attack.py:42` `net = torch_models.resnet50(pretrained=True)` with no normalisation layer, input
normalised at `:84-88`). So `xb` in the tracker is the normalised tensor, and the argmax in (a) runs on the raw net at
that normalised point.

**(c) clamp / to_pixel / unnormalise — before or after the decision**

- Attack-side clip (BEFORE the decision, in normalised space, to `lb/ub = normalize(0)/normalize(1)`):
  `proposed_attack.py:65` (`find_random_adversarial`), `:140` (`go_to_boundary_CGBA_H`), `:160`
  (`go_to_boundary_CGBA`, `p_near_boundary`), `:264` (final `x_adv`). `clip_image_values` is a `torch.max`/`torch.min`
  clamp to `[lb, ub]` (`utils.py:7-10`).
- **Not** clipped: `bin_search` (`proposed_attack.py:77-85`, `mid = (cln + adv) / 2.0`, no clip) and
  `SemiCircular_boundary_search` (`proposed_attack.py:179-191`, `x_mid = x_0 + mid_dir*d`, no clip; the patched
  `fixed` copy at `run_cgba.py:147-166` also does **not** clip `x_mid`).
- Tracker-side unnormalise+clamp (AFTER the decision; the decision uses the un-clamped `xb`):
  `adapters/_common.py:389-393`
  ```python
  def to_pixel_unnormalize(x):
      return (x * sd + mu).clamp(0.0, 1.0)
  ```
  called at `hardlabel_tracker.py:137` `pix = self.to_pixel(xb).float()`. This clamp is applied only to the value
  that is stored/measured (`pix`), never to the input that produced `lab`/`adv`.

**(d) where the stored best/pix (used for pix_l2 and the npz) is written**

`hardlabel_tracker.py:137` `pix = self.to_pixel(xb).float()`, `:138` (L2), `:143` `self.best = pix[v:v+1].detach().clone()`
(on an "improve" event), `:145` `self.snap[v] = self.best` (checkpoint), `:196` `pix_l2`; then `:204`
`store_adv(store_root, arch, image, method, b, x, self.x0, ...)`, and `advstore.py:92` `delta = xa - x0n`,
`:97` `dtype = np.float16` (default), `:101` `delta=delta.astype(dtype)`.

**Conclusion (evidence only, no fix).** CGBA's recorded `found`/`best` is decided on the RAW net at the UNCLAMPED
normalised point `xb` (`hardlabel_tracker.py:136,139`), but the stored/scored image is `clamp(xb*sd+mu, 0, 1)`
(`hardlabel_tracker.py:137` + `_common.py:393`). These are the same image **only when** `xb*sd+mu ∈ [0,1]`
elementwise. The attack clips SOME queries in-loop (`proposed_attack.py:65/140/160/264`), but **not** `bin_search`
(`:77-85`) and **not** `SemiCircular_boundary_search` (`:179-191`, patched `:147-166`); the boundary-refinement
queries from those two functions are exactly where the queried point can leave the pixel box, so the stored (clamped)
projection disagrees with the queried point and re-scores clean. The measured X rows are consistent with this: the two
CGBA X rows have `m_s1 = +0.748` and `+0.324` (NT margin `l_y - max_{j≠y} l_j > 0` = confidently CLEAN), i.e. the
stored image is a clean clamped projection, not a boundary point.

**Comparison with the B4g KAT distinction.** The KAT (`reports/B4g_kat_canary.md`) ran the SAME adapter and attack —
`adapters/run_cgba.py` with the default `--cgba_variant fixed`, `make_tracker` with `to_pixel=to_pixel_unnormalize`
(`run_cgba.py:82-85`), and the same patched `SemiCircular_boundary_search` (no clip of `x_mid`) — but on the
`kat_linear` victim. The KAT's "CGBA clips in-loop" characterisation (its deviation #2) is therefore only partially
true: the clip calls exist, but `bin_search` and `SemiCircular_boundary_search` do not clip, so the same
"decide on unclamped normalised point, store clamped pixel" mechanism as ARS_OPT is present in CGBA too. The KAT did
**not** exercise the code path that produced the X rows in two ways: (i) it ran on the linear `kat_linear` victim, not
the real deep-net victims (resnet50 / vit_b_32), and (ii) its check (a) (`kat_linear.py:370-377`) is a
margin-Lipschitz bound `m <= lip * U * ||delta||` evaluated on the clamped victim, not the strict argmax decision
probe that `domain_consistency.py:81-91` performs; on the linear victim no CGBA stored image was clean (35/35 pass),
so the X mechanism did not manifest there. The B4g KAT's characterisation of CGBA as fully "clipped in-loop" is
therefore incomplete and would not have predicted the real-victim X rows.

## 6. A4 — LGC_RC_EGC evidence (quoting only, no edit)

**Where `found` is computed.** The per-checkpoint `found` flag is set by the RUNNER, not by `lgc_bench`. The engine
(`lgc_bench/engine.py`) records float32 checkpoint states at `engine.py:5439`, `:5483`, `:5505`
(`self.checkpoint_states[cp] = (state.detach().clone(), state_q)`), returned via `AttackResult` (`:5510`). The runner
`ttba_github/run_frame_b1_generic.py:174-194` then converts the state to an image and writes the row:
```python
if cp in res.checkpoint_states:
    z, sq = res.checkpoint_states[cp]
    img = sp.to_image(z)                                  # float32, clamped to [0,1] (spaces.py:126 / :157)
    m = image_metrics(img, x0, oracle.label_uncounted, oracle.is_adv_label, ...)   # adv_float / adv_uint8
    ...
    row.update({"found": 1, "state_query": sq, "lat_l2": lat})
    ...
    store_adv(..., img, x0, state_query=sq, adv_flag=m.get("adv_float", -1), ...)
else:
    row.update({"found": 0, ...})
```
`found = 1` is set whenever a checkpoint state exists — it is NOT itself an adversariality test. The float32
adversarial decision is `adv_float` (`lgc_bench/metrics.py:108` `out["adv_float"] = int(is_adv_label(label_fn(img)))`,
with `label_fn = HardLabelOracle.label_uncounted`, `lgc_bench/spaces.py:237-239`, which returns only the top-1
argmax). The oracle is decision-only: `_labels` (`spaces.py:196-212`) computes logits but returns only
`logits.argmax(1)` (`:210`) — **no logit margin is ever recorded in the engine or runner.**

**Where adv_store serialises to float16.** `advstore.py:92` `delta = xa - x0n`, `:97`
`dtype = np.float32 if os.environ.get("ADV_STORE_DTYPE", "float16") == "float32" else np.float16`, `:101`
`delta=delta.astype(dtype)` (so `float16` by default).

**The two margins.** The float32 in-engine LOGIT margin is **not obtainable read-only**: the engine is decision-only
(only the argmax boolean `adv_float` is recorded), and the npz stores the delta as float16 (the float32 `img` is not
persisted), so the float32 margin cannot be reconstructed from stored artefacts. The only internally recorded float32
evidence that settles the question is `adv_float`:

| image (targeted) | checkpoint (flagged) | adv_float (float32, internal) | pix_l2 (float32) | reconstructed margin m(s=1) from npz (float16) | float16 error bound 2^-11·pix_l2 |
|---|---|---|---|---|---|
| vgg19 00002148 (398→133) | 100 | **1** (adversarial) | 52.957 | +0.001432 | 0.02586 |
| vgg19 00002148 (398→133) | 200 | **1** | 48.456 | +0.000731 | 0.02366 |
| Salman2020Do_R50 00046453 (581→669) | 1000 | **1** | 107.987 | +0.000173 | 0.05273 |
| Salman2020Do_R50 00046453 (581→669) | 2000 | **1** | 87.940 | +0.000125 | 0.04294 |

`adv_float = 1` at every checkpoint for both images (also at 10000). So the float32 engine state genuinely WAS
adversarial at the flagged checkpoints: the `found = 1` flag was NOT wrong/marginal before serialisation. The X arises
at float16 serialisation: the stored delta is the float16 rounding of `img - x0`, whose reconstruction lands just
across a very near decision boundary (reconstructed margin m(s=1) is positive, i.e. clean, and the float16 error bound
2^-11·pix_l2 is 15–420× the reconstructed margin, so the boundary point sits well inside float16 rounding range of the
boundary). This is candidate (b) of the STOP bundle, not an oracle-domain defect like CGBA's (a).

## 7. Deviations from the prompt

None. (One clarification, not a deviation: the prompt's A4 says "Where in lgc_bench … the `found` flag is computed";
the `found` flag is actually written by the runner `ttba_github/run_frame_b1_generic.py:188`, while `lgc_bench/engine.py`
only records the float32 `checkpoint_states`. Both are quoted above.)

## 8. sha256 of created file; GPU minutes

| file | sha256 |
|---|---|
| `reports/raw/B4h/t1a_triage.md` | (this file) |

GPU minutes: ~0 (this phase is read-only; no GPU command was run beyond one short `torch` device-info call, <1 s).

## 9. Summary (plain English)

1. Of the 6 non-ARS_OPT X rows (4 images), 3 resolve to C by budget 10000 and 3 do not (2 LGC_RC_EGC Salman + 1 CGBA
   resnet50 stay Q9 at the end), so the X is not always a transient early-budget artefact.
2. The true low-budget X rate is 0–5.6% for CGBA and 0–2.8% for LGC_RC_EGC (n=18 / n=36 per checkpoint); the 6
   flagged rows are a small sample, and any E8 ASR computed before a fix inherits at most this order of bias.
3. CGBA decides "adversarial" on the raw net at the UNCLAMPED normalised point but stores the CLAMPED pixel projection
   `clamp(x*sd+mu, 0, 1)`; they coincide only in-range, and the boundary-search queries are not clipped, so the stored
   image can be clean (the two CGBA X rows are confidently clean, m=+0.32…+0.75). The B4g KAT ran the same
   adapter/attack but on the linear victim with a margin-based check, so it did not reveal this.
4. LGC_RC_EGC's float32 state is genuinely adversarial at every checkpoint (`adv_float=1`); the X is a float16
   serialisation effect near a very close decision boundary (reconstructed margin +0.0001…+0.0014 vs a 2^-11 rounding
   bound of 0.024…0.053), not a wrong `found` flag and not an oracle-domain defect.
