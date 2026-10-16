# B0 — Environment + frozen worktree + advstore patch

Status: COMPLETE (2026-10-10).

## Pre-flight (verified)
- Worktree `/root/autodl-tmp/LGC_frozen` exists, branch `paper-runs`, HEAD `9e2822e8bb5c1aee6b04d39fea7f813ed0086454` (tag `build38-prereg`).
- Engine sha `835f1933d9e5` verified (frozen, never edited).
- Evalkit unzipped: `tools/eval/` complete. Selftest ALL PASS 20/20.
- GPU: RTX 5090 (32 GB), driver 595.71.05, `nvidia-smi` reports CUDA 13.2, no processes, Util 0%.
- Victim weight hashes match `TtBA_provenance.txt`.
- `reports/` and `docs/provenance/` folders created.

## Step 1 — advstore patch (dry run then applied)
- Patch tool: `/root/autodl-tmp/LGC_revision/frame_kit/apply_advstore_patch.py` (byte-identical copy at
  `/root/autodl-tmp/Latent-Geometric-Chords/apply_advstore_patch.py`).
- Insertions only (no line edited/removed): 1 argparse line (`--save_adv`, line 455) + 5 hook lines (line 1437).
- `run_benchmark_33b.py` sha256: before `b8bd8b7264e0` → after `9c62820e5faf` (verified).
- `advstore.py` copied to `/root/autodl-tmp/LGC_frozen/` (5284 bytes).
- `checkpoints/` and `cache/` symlinked from `/root/autodl-tmp/Latent-Geometric-Chords/`.
- `splits/` present; all 6 CSVs match prereg lines 56–61.
- AE checkpoint sha `1d004c634f1b` verified.
- Hook runs only when `--save_adv` is set; with `--save_adv` OFF the runner behaviour is unchanged.

## Step 2 — environment provenance
Written to `docs/provenance/env_paper_runs.txt`:
- date 2026-10-10 (UTC 2026-10-10T14:04Z); container id `2ba7efw2s1-30de4454`; GPU `NVIDIA GeForce RTX 5090`;
  driver `595.71.05`; CUDA: nvidia-smi max `13.2`, torch built `12.8` (+cu128), nvcc toolkit `12.1.105`.
- torch/torchvision/cudnn: `2.7.1+cu128 0.22.1+cu128 12.8 90701`.
- pip freeze (selected): torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 2.5.2, scipy 1.18.1, pandas 3.0.5,
  matplotlib 3.9.0, piq 0.8.0, lpips 0.1.4, torchmetrics 1.9.0, PyYAML 6.0.1.
- git: `rev-parse HEAD` = `9e2822e8bb5c1aee6b04d39fea7f813ed0086454`, `describe --tags` = `build38-prereg`.
- Note recorded: `OMP_NUM_THREADS=0` / `MKL_NUM_THREADS=0` present in the container env (invalid; libgomp warns,
  falls back to single thread). Shared by all runs.

## Step 3 — bit-identity test (--save_adv OFF vs ON)
ResNet-50 row 1 (non_targeted), `LGC_RC_EGC`, `--num_samples 1 --max_queries 10000 --seed 42`
`--exclude_seeds 992 2027 3141 2718`.

- Command A (no `--save_adv`, checkpoints 500 1000 2000 2500 5000 10000) → `Results/_bitid_off`.
- Command B (with `--save_adv`, checkpoints 100 200 500 1000 2000 2500 5000 10000) → `Results/_bitid_on`.

Comparison of the 6 registered budgets on `pix_l2`, `state_query`, `adv_float`:

| budget | state_query (off/on) | pix_l2 (off/on) | adv_float (off/on) | result |
|---|---|---|---|---|
| 500   | 447 / 447 | 39.89765548706055 | 1 / 1 | BIT-IDENTICAL |
| 1000  | 978 / 978 | 33.71754455566406 | 1 / 1 | BIT-IDENTICAL |
| 2000  | 1901 / 1901 | 25.956871032714844 | 1 / 1 | BIT-IDENTICAL |
| 2500  | 2350 / 2350 | 23.74616813659668 | 1 / 1 | BIT-IDENTICAL |
| 5000  | 4993 / 4993 | 18.15683364868164 | 1 / 1 | BIT-IDENTICAL |
| 10000 | 9975 / 9975 | 14.403605461120605 | 1 / 1 | BIT-IDENTICAL |

PASS: all 6 budgets BIT-IDENTICAL. `--save_adv` introduces no change to attack trajectory or query accounting.

Advstore output (Command B) verified present: `Results/_bitid_on/non_targeted/adv_store/resnet50/ILSVRC2012_val_00007450/`
with `x0.npy` + `LGC_RC_EGC_Q{100,200,500,1000,2000,2500,5000,10000}.npz`.

## Conclusion
B0 complete. Environment recorded, frozen worktree + advstore patch verified, bit-identity PASS.
Next phase: B1 (sources).
