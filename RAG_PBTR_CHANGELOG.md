# RAG-PBTR code revision

## New attack methods

- `RAG_PBTR`: RAG-aware pullback-metric boundary trust-region optimizer, using the same normal-estimation schedule as `LGC` for a clean optimizer-only comparison.
- `RAG_PBTR_LQ`: the same optimizer plus a low-query normal estimator using a small orthogonal Gaussian probe set and a historical normal prior.

## Core local model

For the RAG chord `G(z)-G(z_s)`, the code uses

`f(z) = 0.5 * ||G(z)-G(z_s)||_2^2`

and the pullback metric

`M(z) = J_G(z)^T J_G(z) + lambda I`.

`M` is never materialized. Matrix-vector products are computed with JVP/VJP and two small conjugate-gradient solves are used to obtain `M^{-1}g` and `M^{-1}eta`.

The estimated boundary-tangent natural descent is

`p = -M^{-1}g + M^{-1}eta * (eta^T M^{-1}g)/(eta^T M^{-1}eta)`.

It is scaled to an adaptive trust region in the pullback metric. Only decoder derivatives are used; victim-model access remains hard-label only.

## Query-saving design

- Decoder objective checks and pullback products cost no victim queries.
- Backtracking candidates are rejected with zero victim queries if they do not improve the RAG chord objective.
- Only a promising trial is queried.
- Boundary restoration uses a capped binary refinement (`pbtr_refine_steps`, default 8) rather than an exact long binary search.
- `RAG_PBTR_LQ` additionally uses a fixed small normal-query budget (`pbtr_normal_queries`, default 12) plus previous-normal reuse.

## Important correctness changes

1. `all_queries` is now the authoritative attack query counter in `Attack()`.
2. The old `optimize_target_latent()` used victim logits and `loss.backward()`, which violated the strict decision-based threat model. In this revision it only performs a top-1 check and returns `None` if the pseudo-target is not already in the target class.
3. Orthogonal Gaussian probes are rescaled by `sqrt(D)` so their norm matches ordinary `N(0,I)` probes instead of accidentally reducing the effective sampling radius.

## Suggested first comparison

Use identical samples, initialization, seed, autoencoder, victim model, and checkpoints for:

`LGC`, `RPGO`, `LRCA`, `NGQS`, `RAG_PBTR`, `RAG_PBTR_LQ`.

For the paper-level ablation, compare `LGC` vs `RAG_PBTR` first. This keeps the normal-estimation budget the same and isolates the optimizer. Then compare `RAG_PBTR` vs `RAG_PBTR_LQ` to measure the benefit of low-query normal estimation.

Recommended initial PBTR hyperparameters:

- `pbtr_damping=1e-3`
- `pbtr_cg_iters=6`
- `pbtr_trust_scale=0.12`
- `pbtr_trust_min=0.02`
- `pbtr_trust_max=0.50`
- `pbtr_expand=1.25`
- `pbtr_shrink=0.50`
- `pbtr_refine_steps=8`
- `pbtr_normal_queries=12`
- `pbtr_history_beta=0.50`

These are starting values, not claimed optimal settings.
