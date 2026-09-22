# LGC / LGC-H / LGC-PRTR / LGC-FD comparison

This bundle extends the previous LGC/LGC-H/LGC-PRTR framework with the standalone **FD** method from the attached `lgc_attack(1).py`. The four comparison methods are:

- `LGC` — original latent semicircular search.
- `LGC_H` — original accelerated angular/binary search.
- `LGC_PRTR` — pullback Gauss-Newton / Riemannian trust-region replacement.
- `LGC_FD` — the attached four-fold directional (FD) idea integrated as the fourth comparison method.

The attached file also contains experimental `QS` and `Hybrid` branches. They are **not** included here, because the requested comparison contains exactly four methods; the standalone block explicitly labeled `FD` is treated as the requested third/original-new method.

## FD integration note

The attached FD comments define

`u(phi) = cos(phi)e1 + sin(phi)e2`

and

`g(phi) = 1 + beta cos(fold phi)`.

However, the attached `FD(...)` helper computes `e1,e2` but applies its perturbation only along `eta_unit`, so `phi` changes only the amplitude and does not rotate the direction. In this comparison bundle, `LGC_FD` implements the **stated FD formulation** so that `phi` genuinely explores the 2-D sector. Query counting is also centralized so every victim top-1 label query is charged to the common budget.

Default FD parameters:

- `fd_beta = 0.15`
- `fd_fold = 4`
- `fd_num_directions = 4`
- `fd_max_bisection_steps = 40`

## Hard-label threat model

All four attack optimizers use only victim top-1 labels during attack search. `LGC_PRTR` differentiates only through the attacker-owned decoder/RAG map. The targeted initializer in this framework also avoids victim logits/gradients: it uses decoder-side target reconstruction plus periodic victim label checks.

## Smoke test

```bash
python run_places365_attack.py \
  --mode non_targeted \
  --target_archs resnet50 \
  --attack_methods LGC LGC_H LGC_PRTR LGC_FD \
  --ae_checkpoint /path/to/vgg16_autoencoder.pth \
  --base_dir ./data/Places365 \
  --num_samples 5 \
  --reserve_pairs 5 \
  --max_queries 1000 \
  --query_checkpoints 500 1000 \
  --prtr_max_iterations 80
```

Targeted:

```bash
python run_places365_attack.py \
  --mode targeted \
  --target_archs resnet50 \
  --attack_methods LGC LGC_H LGC_PRTR LGC_FD \
  --ae_checkpoint /path/to/vgg16_autoencoder.pth \
  --base_dir ./data/Places365 \
  --num_samples 5 \
  --reserve_pairs 10 \
  --max_queries 1000 \
  --query_checkpoints 500 1000 \
  --prtr_max_iterations 80
```

## Full 200-image comparison

```bash
python run_places365_attack.py \
  --mode non_targeted \
  --target_archs resnet50 densenet161 \
  --attack_methods LGC LGC_H LGC_PRTR LGC_FD \
  --ae_checkpoint /path/to/vgg16_autoencoder.pth \
  --base_dir ./data/Places365 \
  --num_samples 200 \
  --reserve_pairs 30 \
  --max_queries 10000 \
  --query_checkpoints 500 1000 2500 5000 10000
```

## Output

Each method is saved under:

`Results/places365_<mode>/<arch>/<method>/metrics.npz`

The main arrays are fixed-shape numeric arrays `[num_samples, num_checkpoints]`: `state_queries`, `l2_lat`, `linf_lat`, `l2_pix`, `linf_pix`, `ssim`, `lpips`, `asr`, and `adv_label`. A cross-method `comparison_summary.csv` is also written.

## Validation performed

- Python syntax compilation passed for the attack, runner, and model builder.
- CPU toy smoke test passed for all four methods within a fixed query budget.
- Dedicated FD and PRTR toy tests passed.

For the first scientific comparison, keep dataset split, source/target pairs, autoencoder checkpoint, RAG mapping, victim model, seed, query budgets/checkpoints, and metrics identical. Change only `attack_method`.
