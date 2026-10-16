# B1 — Baseline sources (read from code, not just READMEs)

All clones are read-only; no file inside a clone was edited. CPU only, no GPU used.
GitHub was unreachable directly (connection timeout); clones were fetched via the `ghfast.top`
proxy (`git clone https://ghfast.top/https://github.com/<org>/<repo>`). Commit hashes below are the
actual `git rev-parse HEAD` of each clone (shallow).

| Method | Clone path | `git rev-parse HEAD` | License |
|---|---|---|---|
| TtBA (existing) | `/root/autodl-tmp/TtBA` | `ea6d1649fccda0eb8392c9171e43cc9e0974acf9` | (see repo LICENSE) |
| CGBA / CGBA-H | `/root/autodl-tmp/baselines/CGBA` | `73d3e9c94423b1503053258494a54ad4d2892efa` | MIT (`LICENSE`) |
| HSJA | `/root/autodl-tmp/baselines/HSJA` | `daecd5c7055d5214b39c34a7a28a98acd3557fbc` | **none** (no LICENSE file) |
| ARS-OPT | `/root/autodl-tmp/baselines/ARS-OPT` | `ab71cb88058ad0ca778e0048e5cf1b60921bb673` | LGPL-3.0 (`LICENSE`) |
| Sign-OPT | `/root/autodl-tmp/baselines/Sign-OPT` | `65a82f8ea6beedc1b4339aa05b08443d5c489b8a` | MIT (`LICENSE`) |

---

## CGBA / CGBA-H  (`Farhamdur/CGBA`)

- **Entry class**: `Proposed_attack` — `proposed_attack.py:12`.
  Constructor `__init__(model, src_img, mean, std, lb, ub, tar_img=None, dim_reduc_factor=4,
  attack_method='CGBA_H', iteration=93, initial_query=30, tol=0.0001, sigma=0.0002,
  verbose_control='Yes')` (`proposed_attack.py:13-16`). Main entry method `Attack()` (`proposed_attack.py:214`).
- **CGBA-H location**: `go_to_boundary_CGBA_H` — `proposed_attack.py:131`; selected by
  `attack_method='CGBA_H'` (default) at `proposed_attack.py:249-250`. The original CGBA boundary
  walk is `go_to_boundary_CGBA` — `proposed_attack.py:151`. (This is the *official* copy; the legacy
  modified team copies `CGBA-main/Very Very Important/proposed_attack.py` and
  `IEEE_CGBA_Image_proposed_attack.py` are NOT used.)
- **Model call / return**: `self.model.forward(Variable(image, requires_grad=True)).data` then
  `torch.argmax(...)` (`proposed_attack.py:44`, `:106`). The model is a `torch.nn.Module` returning
  **logits**; the attack only ever reads `argmax` → decision-only in practice. `is_adversarial`
  (`proposed_attack.py:43-53`) is the single scalar decision oracle.
- **Input range / normalisation**: operates on the **ImageNet-normalised** tensor
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`, `Non_targeted_attack.py:35-36` /
  `Targeted_attack.py:36-37`), NCHW. `lb`/`ub` are the normalised bounds from
  `valid_bounds(img, delta=255)` (`utils.py:13`; `Non_targeted_attack.py:80-81`).
  `inv_tf` (`proposed_attack.py:269-278`) un-normalises to [0,1] **only** to report norms; the attack
  itself works in normalised space.
- **Targeted interface**: pass `tar_img` (normalised target image); `is_adversarial` checks
  `predict_label == tar_lbl` (`proposed_attack.py:49`); when `tar_img` is set the random search is
  skipped and `x_random = tar_img` (`proposed_attack.py:221-222`). Target label is read from the
  target image (`proposed_attack.py:21-22`).
- **Internal query counter**: `self.all_queries`, incremented per call in `is_adversarial`
  (`proposed_attack.py:45`) and by `+q_max` for the batched gradient estimate in
  `normal_vector_approximation_batch` (`proposed_attack.py:109`); a cumulative list `n_query` is
  returned. Consistency asserted via `assert self.all_queries == q_num` (`proposed_attack.py:252`).
- **Early-stop / success thresholds**: none in the main `Attack()` loop (runs all `iteration` steps).
  Inner early returns: `go_to_boundary_CGBA_H`/`CGBA` return after `num_calls>=40`
  (`proposed_attack.py:144`, `:164`); `bin_search` stops at `tol` or `num_calls>=100`
  (`proposed_attack.py:83`); `find_random_adversarial` loops until adversarial (`proposed_attack.py:61`).
- **Iteration / query limits**: `iteration=93` (`Non_targeted_attack.py:30`), `initial_query N0=30`,
  per-iteration `q_opt = int(N0*sqrt(i+1))` (`proposed_attack.py:234`), gradient-estimator batch
  size `40` (`proposed_attack.py:32`).
- **Default ImageNet hyper-parameters** (from the official drivers, read-only):
  `iteration=93`, `dim_reduc_factor=4`, `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`,
  `delta=255` (`Non_targeted_attack.py:30-36`, `Targeted_attack.py:31-37`); class defaults
  `tol=1e-4`, `sigma=2e-4`, `initial_query=30` (`proposed_attack.py:15`).
  ⚠ **Legacy preprocessing in driver** `Resize((224,224))` then `CenterCrop(224)`
  (`Non_targeted_attack.py:78,85`) — must be replaced by the shared TtBA loader per F3/F4 (adapter work, B2).
- **License**: MIT.

## HSJA  (`Jianbo-Lab/HSJA`)

- **Entry function**: `hsja(model, sample, clip_max=1, clip_min=0, constraint='l2',
  num_iterations=40, gamma=1.0, target_label=None, target_image=None,
  stepsize_search='geometric_progression', max_num_evals=1e4, init_num_evals=100, verbose=True)`
  — `hsja.py:4`. Driver `main.py` (`main.py:83-94`).
- **Model call / return**: `model.predict(x)` must return **probability scores**; the code uses
  `np.argmax(model.predict(sample))` (`hsja.py:55`) and `decision_function` uses
  `np.argmax(prob, axis=1)` (`hsja.py:146-150`). Only the argmax is used → decision-only.
- **Input range**: `[0,1]` (`clip_min=0`, `clip_max=1`, `hsja.py:6-7`); normalisation lives in the
  model wrapper (`build_model.py`).
- **Targeted interface**: `target_label` (int) + `target_image` (array, same shape as sample)
  (`hsja.py:11-12`); `decision_function` flips comparison (`hsja.py:147-150`); `initialize` uses
  `target_image` as the start (`hsja.py:302-303`).
- **Internal query counter**: **none** for the whole run. `max_num_evals` is a *per-iteration*
  gradient-estimate cap, not a total (`hsja.py:43-46`, docstring; and `README.md:7`). Per-iteration
  `num_evals = min(init_num_evals*sqrt(j+1), max_num_evals)` (`hsja.py:95-96`). Adapter must keep its
  own total counter (F5).
- **Early-stop / success thresholds**: `initialize` loops until a misclassified noise is found and
  asserts `num_evals < 1e4` (`hsja.py:277-285`). No success-threshold early stop in the main loop —
  it always runs all `num_iterations` (`hsja.py:88`).
- **Iteration / query limits**: `num_iterations=40` default (`hsja.py:9`); `main.py` default is `64`
  (`main.py:124`). `max_num_evals=1e4`, `init_num_evals=100` (`hsja.py:14-15`).
- **Default ImageNet hyper-parameters**: **repo is CIFAR-10 only** — no ImageNet defaults in code.
  Defaults present: `constraint='l2'`, `gamma=1.0`, `stepsize_search='geometric_progression'`,
  `max_num_evals=1e4`, `init_num_evals=100`. ImageNet setting must come from the paper (cited) with
  `max_num_evals`/`num_iterations` set so the 10 000 total budget binds (skill Section 3).
- **License**: **none** (no LICENSE file) → do not redistribute; pin commit + clone script.

## ARS-OPT  (`machanic/hard_label_attacks`, paper "Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors", arXiv 2512.21241)

- **Entry class**: `ParsOptL2Norm` — `PARS_OPT/pars_opt_l2_norm_attack.py:15`; driven by
  `PARS_OPT/attack.py` (method chosen by `--method {ARS,PARS}`, `attack.py:77`).
- **Prior-free (no surrogate) mode — YES, possible**: `--method ARS` runs with **no surrogate**.
  `self.surrogate_models = surrogate_models if method == 'PARS' else []` (`pars_opt_l2_norm_attack.py:21`);
  `get_prior_grad` returns `[]` unless `method=='PARS'` (`pars_opt_l2_norm_attack.py:317-324`);
  `get_orthogonal_basis` uses the random basis only when not PARS (`pars_opt_l2_norm_attack.py:326-327`).
  The logits-based `cw_loss`/`get_g_grad` (`pars_opt_l2_norm_attack.py:229-315`) are used **only** for
  the PARS surrogate prior, never in ARS mode.
- **Model call / return**: `self.model(x).max(1)[1].item()` (e.g. `pars_opt_l2_norm_attack.py:92`,
  `:596`, `:667`, `:802`). `StandardModel` returns logits; only the argmax is used → decision-only.
  The gradient is estimated by hard-label sign queries / binary search, not logits.
- **Input range / normalisation**: image tensors in `[0,1]`; `StandardModel` normalises internally
  (docstring `models/standard_model.py:34` "always accept standard image: in [0,1] range"; mean/std
  applied in `forward`).
- **Targeted interface**: `targeted_attack(image_index, images, target_labels, target_class_image)`
  (`pars_opt_l2_norm_attack.py:675`); target label chosen by `--target_type` (`increment` default,
  `attack.py:73`); starting image `target_class_image = select_random_image_of_target_class(...)`
  (`attack.py:893`). For fair setup the adapter supplies the CSV target image as `target_class_image`.
- **Internal query counter**: per-image `query` (torch scalar) counts init directions + binary-search
  + sign-estimator queries; stored in `self.query_all` / `success_query_all` (`pars_opt_l2_norm_attack.py:43-49`, `:916-920`).
  Loop breaks on `query >= self.maximum_queries` (`pars_opt_l2_norm_attack.py:619`, `:751`).
- **Early-stop / success thresholds**: `success_stop_queries` records when distortion ≤ `epsilon`
  (`count_stop_query_and_distortion`, `pars_opt_l2_norm_attack.py:816-823`) but there is **no**
  success-threshold early stop in the main loop (runs to `iterations` or `maximum_queries`).
- **Iteration / query limits**: `iterations=1000`, `maximum_queries=10000` (`pars_opt_l2_norm_attack.py:17-18`);
  **ImageNet targeted forced to 20000** (`attack.py:127-128`); `q = k = est_grad_samples = 200`
  (`attack.py:62`, `pars_opt_l2_norm_attack.py:33`); `alpha(lr)=0.2`, `beta=0.001`.
- **Default ImageNet hyper-parameters** (`configures/PARS_OPT.json`): l2 untargeted
  `epsilon=10.0, clip_grad_max_norm=1.0, tol=1e-4, lr=0.2`; l2 targeted `epsilon=10.0,
  clip_grad_max_norm=1.0, tol=1e-4, lr=0.2`.
- **License**: LGPL-3.0.

## Sign-OPT  (`cmhcbb/attackbox`)

- **Entry class file**: `OPT_attack_sign_SGD` — `attack/Sign_OPT.py:42` (this is the Sign-OPT
  (ICLR 2020) class; sibling files `attack/Sign_OPT_lf.py`, `attack/Sign_OPT_v2.py` are the l∞ /
  variants). Invoked via `test_attack.py --attack Sign_OPT`.
- **Model call / return**: `model.predict_label(x)` returns the **top-1 label** (`models.py:30-45`);
  every oracle call is a `predict_label(...)` (e.g. `Sign_OPT.py:78`, `:221`, `:317`). Decision-only.
- **Input range / normalisation**: image in `[0,1]`; `PytorchModel` clamps to `bounds` and the
  underlying model normalises internally (`models.py:13-19`; `allmodels.py:454,508` mean/std;
  ImageNet loader `Resize(256)+CenterCrop(224)` at `allmodels.py:511`).
- **Targeted interface**: **not implemented** — `__call__(..., targeted=True)` raises
  `NotImplementedError` (`Sign_OPT.py:370`); `attack_targeted` is absent. Sign-OPT here is
  untargeted-only.
- **Internal query counter**: `query_count` in `attack_untargeted` (`Sign_OPT.py:62`, `:167`);
  `PytorchModel` also keeps `num_queries` (`models.py:11`, `:18`, `:27`, `:39`). `query_limit` default
  is `20000` in `attack_untargeted` (`Sign_OPT.py:52`) but `4000` in `__call__` (`Sign_OPT.py:368`);
  the shell script uses `--query 40000` (`exp_scripts/signopt-l2.sh`).
- **Early-stop / success thresholds**: breaks when `query_count > query_limit` (`Sign_OPT.py:171`)
  and when `beta < 1e-8` (`Sign_OPT.py:160-161`). The success-distortion early stop is commented out
  (`Sign_OPT.py:178-180`).
- **Iteration / query limits**: `MAX_ITER=1000` (`Sign_OPT.py:11`), `iterations=1000` (`Sign_OPT.py:52`),
  `k=200` gradient directions (`Sign_OPT.py:43`), `num_directions=100` init (`Sign_OPT.py:70`),
  `alpha=0.2`, `beta=0.001`, `start_learning_rate=1.0` (`Sign_OPT.py:10`).
- **Default ImageNet hyper-parameters**: repo examples are CIFAR-10/MNIST (`exp_scripts/signopt-l2.sh`:
  `epsilon=0.5`, `query=40000`). **No ImageNet defaults in code** — take from the paper.
  README example labelled "Sign-OPT on ImageNet" is a typo: it runs `--attack OPT_attack
  --dataset CIFAR10` (`README.md:37-39`).
- **License**: MIT.

---

## Notes for B2 (adapter contract, not done here)

- All four methods use only `argmax`/`predict_label` → the `HardLabelTracker` decision-only wrapper
  is a faithful drop-in. HSJA wants `predict` returning probabilities; the wrapper must return an
  argmax-able array (one-hot). CGBA wants a `torch.nn.Module` with `forward` → return a one-hot/logit
  tensor whose argmax is the label. Sign-OPT wants `predict_label` → return label. ARS-OPT wants a
  `StandardModel`-like `__call__` → return a one-hot tensor.
- Query accounting must be imposed by the tracker (F5): HSJA has no total counter; CGBA/ARS-OPT/Sign-OPT
  have internal counters to cross-check.
- Targeted start: CGBA `tar_img`, HSJA `target_image`, ARS-OPT `target_class_image`; Sign-OPT has no
  targeted mode (appendix, untargeted only — report to the threat-model table).
- Early-stop/success thresholds to disable or carry forward per skill Section 4 / U12: none of the four
  main loops stop on success, so no change needed; only the total query budget must bind at 10 000.
