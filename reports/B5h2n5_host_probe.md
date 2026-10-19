# B5h2n5 — host re-qualification after the instance changed from 1 GPU to 5 GPUs

Status: **COMPLETE**. Verdict: **HOST-IDENTICAL** — T2, T3 (all three methods, domain X = 0) and T5 (all 5 GPUs
bitwise IDENTICAL to GPU 0, both methods) pass; T5b n/a (launcher threads = 8). B5 not started; no `Results/b5_*`
or `Results/appendix_cgba_d8only_*` directory created; no files edited outside `reports/*` plus the new
`Results/b5h2n5_*` data roots.

Lessons ledger rows read at phase start; the minimum set (L03 L04 L06 L21 L23 L24 L26 L27) is active and applies.
None contradicted; L26/L24 dictate the X1–X6 execution rules used throughout; L03 motivates T0's device check;
L22/L23 motivate the domain_consistency gate used in T3; L09/L14/D14 motivate the one-process-per-GPU rule used in T5.

## 1. Scope and prompt

- Prompt (executed byte-for-byte): `reports/raw/B5h2n5/prompt.txt`
- Prompt sha256: `937a2ae10d1aac0828320de0f6b10932a8783c246988dd4309aba1e5f9aab4e2`
- Contract `contract/paper_contract.yaml` **v1.9-rev7** sha256: `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c`
- `git rev-parse HEAD`: `576c99d6cc1c183d7842e65173f4556fea1e7c6e` (branch paper-runs)
- `run_b5_parallel.sh` sha256: `50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb`
- Task texts and rules X1–X6 from the staged N=5 prompt `reports/raw/B5pre2/B5h2_prompt_staged.txt` apply; T0, T2,
  T3, T5, T5b executed (T1/T4/T6/T7a/T8 not repeated, done on this data disk).
- Every GPU command ran with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` and was tee'd to `reports/raw/B5h2n5/<step>.log`.
- `lgc-baselines` skill is NOT installed in `.opencode/skills/` (only lgc-paper-contract, lgc-preflight).

## 2. Environment block (T0)

| field | value | verdict |
|---|---|---|
| hostname | `autodl-container-851548b4b6-1e67012b` | — |
| GPU count / name | 5 x `NVIDIA GeForce RTX 5090` (direct `nvidia-smi -L`, rc 0; matches H's `nvidia_smi_L.txt`/`nvidia_smi_query.txt`; `torch.cuda.get_device_name(i)` identical for i = 0..4) | count 5 == NGPU; every name == D16 study GPU → OK |
| GPU details | capability (12, 0), 170 SM, 32110 MiB per GPU | OK |
| driver | `595.71.05` (direct query + H's saved files + `env_build38.txt`) | **finding only** (per the prompt): differs from the 1-GPU probe and all reference roots (580.105.08); equals env_build38.txt → no STOP |
| torch / torchvision / cuDNN | `2.7.1+cu128` / `0.22.1+cu128` / `90701` (CUDA 12.8) | exactly as required → OK |
| backend flags (D12, as `adapters/_common.py` sets them) | matmul_tf32=False, cudnn_tf32=False, deterministic=True, benchmark=False, float32_matmul_precision=highest | OK |
| CPU quota | `/sys/fs/cgroup/cpu.max` = `8000000 100000` → Q = **80** | Q = 80 ≥ 8 x NGPU = 40 → OK (80 expected ✓) |
| `env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc` | `128` | ≥ 40 → OK |
| RAM | `free -g`: 754 GB total, 709 GB available | OK |
| disk | `df -h /root/autodl-tmp`: 50 G total, 3.9 G used, 47 G free | ≥ 30 GB free → OK |
| TMUX | `TMUX=/tmp/tmux-0/default,1888,0` (non-empty) | finding: OpenCode was started inside a tmux session; no tmux started by the agent |
| engine sha | `835f1933d9e5…` (frozen) | OK |

Dry run `NGPU=5 bash run_b5_parallel.sh --dry_run` (tee `reports/raw/B5h2n5/T0_dryrun.log`, exit 0). Guard lines:

```
[env] torch=2.7.1+cu128 (expected 2.7.1+cu128)  torchvision=0.22.1+cu128 (expected 0.22.1+cu128)  cuDNN=90701 (expected 90701)
[threads] Q=80.0 NGPU=5 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8
[dry_run] NGPU=5 RESULTS_ROOT=Results LOG_DIR=reports/raw/B5a2 (no GPU process started)
```

First 3 queued cells: `e10_TtBA_densenet161_places365_NT` (gpu_slot 0), `e10_TtBA_densenet161_places365_T`
(gpu_slot 1), `main_TtBA_vgg19_T` (gpu_slot 2).

## 3. T2 — cross-host identity, LGC_RC_EGC resnet50 NT (GPU 0)

T1 (B0' determinism) is not repeated; T2 re-ran the T1-B command on GPU 0 of the new host (deviation 2) —
`CUDA_VISIBLE_DEVICES=0 MODES=0 N=3 OUT_PREFIX=Results/b5h2n5_B bash ttba_github/run_frame_b1.sh resnet50`
(8 contract checkpoints, `--save_adv`, OMP=MKL=8, tee `reports/raw/B5h2n5/T2_B.log`, exit 0, EGC 76 s, 3/3 ok) —
and compared `Results/b5h2n5_B_resnet50` with `Results/b4e2_smoke_LGC_RC_EGC_resnet50`
(`tools/eval/compare_roots.py`, tee `reports/raw/B5h2n5/T2_compare_roots.log`). Last 3 lines:

```
checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
npz files: shared 24, only in A 24, only in B 0; differences: 0
VERDICT: IDENTICAL
```

24/24 shared non_targeted checkpoints identical, 24/24 npz identical; the "only in A" rows are the reference root's
targeted-mode rows (B ran non_targeted only). **Verdict: IDENTICAL — the new host (5 GPUs, driver 595.71.05) is
bitwise poolable with the reference hosts for LGC_RC_EGC (pre-registered stack torch 2.7.1+cu128 / cuDNN 90701).**

## 4. T3 — TtBA / CGBA (fixed) / ARS_OPT on resnet50 NT rows 1–3 (GPU 0)

Runs into `Results/b5h2n5_{TtBA,CGBA,ARS_OPT}_resnet50` with the current adapters (D19 domains,
`--cgba_variant fixed`, TtBA default `--seed_mode exact`), each `CUDA_VISIBLE_DEVICES=0`, OMP=MKL=8, exit 0, 3/3
rows (tee `reports/raw/B5h2n5/T3_*.log`). compare_roots vs the reference roots — last 3 lines each
(tee `reports/raw/B5h2n5/T3_compare_roots.log`):

```
TtBA    vs b4e2_smoke_ttba_resnet50:     checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
                                         npz files: shared 24, only in A 24, only in B 0; differences: 0
                                         VERDICT: IDENTICAL
CGBA    vs b4h_t3_smoke_CGBA_resnet50:   checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
                                         npz files: shared 24, only in A 24, only in B 0; differences: 0
                                         VERDICT: IDENTICAL
ARS_OPT vs b4h_t3_smoke_ARS_OPT_resnet50: checkpoint differences: 0; max |pix_l2 A - B| = 0.000e+00
                                         npz files: shared 24, only in A 24, only in B 0; differences: 0
                                         VERDICT: IDENTICAL
```

oob_queries / nonfinite_queries (new runs, of budget 10000; reference values do not exist — the counter postdates
the reference roots; tee `reports/raw/B5h2n5/T3_oob.txt`):

| method | 07450 / 15433 / 25151 oob_queries | nonfinite | status |
|---|---|---|---|
| TtBA | 8435 / 8546 / 8457 | 0 | budget x3 |
| CGBA | 9800 / 9369 / 9828 | 0 | budget x3 |
| ARS_OPT | 10000 / 10000 / 10000 | 0 | budget x3 |

(identical to the 1-GPU probe's T3 values.) domain_consistency on the new CGBA and ARS_OPT roots
(tee `reports/raw/B5h2n5/T3_domain_consistency.log`, csv `t3_domain_consistency.csv`, 48 rows):

```
| CGBA | resnet50/non_targeted | 24 | 19 | 5 | 0 | 0 | …
| ARS_OPT | resnet50/non_targeted | 24 | 15 | 9 | 0 | 0 | …
[+] no X>0 for any method
```

**X = 0 by the column named `class` for both methods. PASS.**

## 5. T5 — per-GPU concurrency identity (5 GPUs, each IDENTICAL to GPU 0)

D14 (one attack process per GPU at any time) and X2 (foreground, `wait`) → two sequential waves, each with exactly
one process per GPU, all 5 GPUs concurrent (deviation 4):

- Wave 1: `CUDA_VISIBLE_DEVICES=$i python run_benchmark_33b.py … --num_samples 1 … --out_dir Results/b5h2n5_gpu${i}_LGC_RC_EGC --save_adv` for i = 0..4, launched together, `wait` — 5/5 exit 0, row 1, status ok, q = 9975.
- Wave 2: `CUDA_VISIBLE_DEVICES=$i python adapters/run_ttba_v2.py … --out_dir Results/b5h2n5_gpu${i}_TtBA --limit 1` for i = 0..4, launched together, `wait` — 5/5 exit 0, row 1, status budget, q = 9999, L2@10k = 2.3313405513763428 on every GPU.

compare_roots GPU i vs GPU 0 (tee `reports/raw/B5h2n5/T5_compare_roots.log`) — last 3 lines of each of the 8
comparisons:

```
EGC  gpu1 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
EGC  gpu2 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
EGC  gpu3 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
EGC  gpu4 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
TtBA gpu1 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
TtBA gpu2 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
TtBA gpu3 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
TtBA gpu4 vs gpu0: checkpoint differences: 0 … npz files: shared 8, only in A 0, only in B 0; differences: 0 → VERDICT: IDENTICAL
```

Wall time under 5-way concurrency vs solo (T2/T3 on GPU 0): EGC 24.8–25.3 s vs 25.3 s/img solo; TtBA 45.9–46.9 s
vs 46.8 s/img solo → **no concurrency slowdown** (Q = 80 ≫ 5 × 8 threads).

**T5: PASS — all 5 GPUs bitwise IDENTICAL to GPU 0 for both methods.**

## 6. T5b

The T0 dry-run `[threads]` line is `Q=80.0 NGPU=5 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8`, i.e. the launcher's own
value is 8, not below 8. **T5b n/a (launcher threads = 8).**

## 7. Deviations from the prompt

1. Direct `nvidia-smi` was NOT denied this time (the OpenCode guard allowed it); GPU count, names and driver from my
   direct queries agree byte-for-byte with H's saved `reports/raw/B5h2n5/nvidia_smi_L.txt` /
   `nvidia_smi_query.txt` and with `torch.cuda.get_device_name(i)`.
2. T2 interpretation: T1 is not repeated, so T2 itself re-ran the T1-B command on GPU 0
   (`Results/b5h2n5_B_resnet50`; `run_frame_b1.sh` appends `_<arch>`, the script is not editable — same convention
   as the 1-GPU probe) and compared it with `Results/b4e2_smoke_LGC_RC_EGC_resnet50` (the staged prompt's
   `Results/b5h2_B` under the `b5h2_* → b5h2n5_*` substitution).
3. Driver is `595.71.05`, not the 1-GPU probe/reference-roots' `580.105.08` (equals `env_build38.txt`) — recorded
   as a finding only per the prompt; every GPU name matches D16 → no STOP.
4. T5 ran as two sequential waves (one method at a time), each with exactly one attack process per GPU — the old
   prompt's single 10-way launch would put two attack processes on one GPU, which the staged B5h2 prompt's D14
   clause ("at most one attack process per GPU at any time") forbids.
5. `TMUX` is non-empty (OpenCode inside a tmux session) — reported as a finding; no tmux was started by the agent.

## 8. sha256 of created/changed files (all under `reports/raw/B5h2n5/` unless stated)

| file | sha256 |
|---|---|
| `reports/raw/B5h2n5/prompt.txt` | `937a2ae10d1aac0828320de0f6b10932a8783c246988dd4309aba1e5f9aab4e2` |
| `reports/raw/B5h2n5/agent_log.md` | `9504ec58a80d2d2b6c98fbe4146d86e7ead4dce27b8ea041531f06416e17a8e4` |
| `reports/raw/B5h2n5/T0_host.log` | `f8df5b3237a451f9cc5079b9b867aad6fe90e475a419af18356cf64773df5807` |
| `reports/raw/B5h2n5/T0_sys.log` | `ef0a9385b5c74cadaca755901e6cec4f91ff2894fc8b54a0bc84dec7436c5f02` |
| `reports/raw/B5h2n5/T0_dryrun.log` | `f2e3ef47c47d8c642c331b60670d207a30e7b86ae97fe27da829514c38a409f2` |
| `reports/raw/B5h2n5/T2_B.log` | `a01150f918d987664057dab0d1cbfe7c966d23ee11b8dfec1c8fb9338019ce4d` |
| `reports/raw/B5h2n5/T2_compare_roots.log` | `b4d4cec5ceac62d442529075b2b7736d0010815f8c3a199f99ba9588576f8b23` |
| `reports/raw/B5h2n5/T3_TtBA.log` | `f22fa1b235f92f093e091eadfb1360446e145bcd9f915cc71508c08d88f994f2` |
| `reports/raw/B5h2n5/T3_CGBA.log` | `76c456af67099d79d60a1062e74d9e3ed56d17f3230b72620707025f6ab3e0cf` |
| `reports/raw/B5h2n5/T3_ARS_OPT.log` | `70acc3bece35c6f78107db2b409c73171f28861a29870fe95fd3f473bf83339e` |
| `reports/raw/B5h2n5/T3_compare_roots.log` | `677b99afcac83e700de8dc6e16b810220e2f163c54cc1c460db1fb4debf6bc84` |
| `reports/raw/B5h2n5/T3_oob.txt` | `0f8bd3f2dbccec51712a4a6edff0d1fb865fb76975efbe15f59bc06afb9bd48e` |
| `reports/raw/B5h2n5/T3_domain_consistency.log` | `6f0ad3f199ea05fdcbe056839cf0a7cc57ee1af3e4fe2d6269ff941896b787e0` |
| `reports/raw/B5h2n5/t3_domain_consistency.csv` | `83139e3ab40f819e57bd95a375d56d8b42ae9196479b350cea448020437f5e9d` |
| `reports/raw/B5h2n5/T5_EGC_wave.log` | `2cb413b8d4609e8f1824c7d11b02a29d881ddf92ebd4798a0925e793843d90b7` |
| `reports/raw/B5h2n5/T5_TtBA_wave.log` | `6b2e08148f27a11f55dd18fafb81916adb333a97982491abc031d780dca2b193` |
| `reports/raw/B5h2n5/T5_EGC_gpu0.log` | `488149ac96a23e98e497f54f1c0a1f74236c0643cd38a75695b432afd23fc895` |
| `reports/raw/B5h2n5/T5_EGC_gpu1.log` | `c7e91aad6389c24347198057728f6c23c3b856902d1ed0cf7e2df44c69882018` |
| `reports/raw/B5h2n5/T5_EGC_gpu2.log` | `3f378e716fbc297630982e8bd6ce106c900fbb8fb38bee686e8f15784b55cffe` |
| `reports/raw/B5h2n5/T5_EGC_gpu3.log` | `fb577bf6127d2a4e306437dbf5813c4906d68d683f481d54ebb885f746bf248a` |
| `reports/raw/B5h2n5/T5_EGC_gpu4.log` | `15d1d8f62877e6d75124cd5df926a6818cc845265592ed438fd51ec238128fdb` |
| `reports/raw/B5h2n5/T5_TtBA_gpu0.log` | `a02b82007495ba84f33f5cca5e7cd808d9a79b61eb101b890516b99d8904697a` |
| `reports/raw/B5h2n5/T5_TtBA_gpu1.log` | `cbb5967619df7c17a2de367c0c43af135d10fd6551db62df6e2616aa4d48e0a0` |
| `reports/raw/B5h2n5/T5_TtBA_gpu2.log` | `057853f1061ee4c819d713d864be66d462a1a230e1525b84d8661e639c31e738` |
| `reports/raw/B5h2n5/T5_TtBA_gpu3.log` | `44097b7bff6f28bd27778af2e50cc2ac780a6c8d262b4853dbed131ddd4cb03f` |
| `reports/raw/B5h2n5/T5_TtBA_gpu4.log` | `38fc0eb962964c62683e4d57bf76388bc924bf48e2fea12418e249fc782716df` |
| `reports/raw/B5h2n5/T5_compare_roots.log` | `245d147c3470a588ed8d2c44eecacd6fe04032d0f17bacfc1a65290ce08f3571` |
| `reports/raw/B5h2n5/nvidia_smi_L.txt` (H, unchanged) | `0d51ed301e37e2a6c2847b79608513e13fc166b370b578a6d797a57a99261133` |
| `reports/raw/B5h2n5/nvidia_smi_query.txt` (H, unchanged) | `cec7af99304b4ee97fde7cf4ea7460a64eb12623120824220763cf37b636c279` |

Result dirs created (data, not committed): `Results/b5h2n5_B_resnet50`, `Results/b5h2n5_TtBA_resnet50`,
`Results/b5h2n5_CGBA_resnet50`, `Results/b5h2n5_ARS_OPT_resnet50`, `Results/b5h2n5_gpu{i}_LGC_RC_EGC` (i = 0..4),
`Results/b5h2n5_gpu{i}_TtBA` (i = 0..4).

GPU minutes: T2 ≈ 1.3 (76 s), T3 ≈ 2.3 + 0.9 + 1.4 ≈ 4.6 + domain_consistency ≈ 0.7, T5 ≈ 2.1 (5 x 25 s EGC) +
3.9 (5 x 46.5 s TtBA) → **≈ 12.6 GPU-minutes total**.

## 9. Summary (plain English)

1. The instance now exposes 5 x RTX 5090 (driver 595.71.05 — recorded as a finding; equals env_build38.txt), the
   pre-registered stack (torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / cuDNN 90701), the correct D12 FP32 flags,
   CPU quota Q = 80 (nproc 128), 709 GB available RAM, 47 GB free disk; OpenCode itself runs inside a tmux session.
2. T2 is IDENTICAL: the new host's GPU 0 reproduces the LGC_RC_EGC resnet50 NT reference bitwise (24/24 checkpoints,
   24/24 npz) — the host may be pooled with the reference hosts.
3. T3 is IDENTICAL for all three methods (TtBA, CGBA fixed, ARS_OPT: 0 checkpoint differences, 24/24 npz each);
   oob_queries match the 1-GPU probe; domain_consistency X = 0 for CGBA and ARS_OPT.
4. T5 passes: all 5 GPUs, running concurrently with one attack process per GPU, are bitwise IDENTICAL to GPU 0 for
   both LGC_RC_EGC and TtBA (8/8 npz each, 0 differences), with no concurrency slowdown at OMP = 8 x 5.
5. T5b n/a (launcher threads = 8) → **verdict HOST-IDENTICAL**, B5 not started.

## RESULT

```
RESULT
phase: B5h2n5
head: 576c99d6cc1c183d7842e65173f4556fea1e7c6e
contract_sha256: ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c
prompt_sha256: 937a2ae10d1aac0828320de0f6b10932a8783c246988dd4309aba1e5f9aab4e2
tasks: T0=PASS T1=n/a T2=PASS T3=PASS T4=n/a T5=PASS T5b=n/a T6=n/a T7a=n/a T8=n/a
files: "reports/raw/B5h2n5/prompt.txt 937a2ae10d1aac0828320de0f6b10932a8783c246988dd4309aba1e5f9aab4e2"
       "reports/raw/B5h2n5/agent_log.md 9504ec58a80d2d2b6c98fbe4146d86e7ead4dce27b8ea041531f06416e17a8e4"
       "reports/raw/B5h2n5/T0_host.log f8df5b3237a451f9cc5079b9b867aad6fe90e475a419af18356cf64773df5807"
       "reports/raw/B5h2n5/T0_sys.log ef0a9385b5c74cadaca755901e6cec4f91ff2894fc8b54a0bc84dec7436c5f02"
       "reports/raw/B5h2n5/T0_dryrun.log f2e3ef47c47d8c642c331b60670d207a30e7b86ae97fe27da829514c38a409f2"
       "reports/raw/B5h2n5/T2_B.log a01150f918d987664057dab0d1cbfe7c966d23ee11b8dfec1c8fb9338019ce4d"
       "reports/raw/B5h2n5/T2_compare_roots.log b4d4cec5ceac62d442529075b2b7736d0010815f8c3a199f99ba9588576f8b23"
       "reports/raw/B5h2n5/T3_TtBA.log f22fa1b235f92f093e091eadfb1360446e145bcd9f915cc71508c08d88f994f2"
       "reports/raw/B5h2n5/T3_CGBA.log 76c456af67099d79d60a1062e74d9e3ed56d17f3230b72620707025f6ab3e0cf"
       "reports/raw/B5h2n5/T3_ARS_OPT.log 70acc3bece35c6f78107db2b409c73171f28861a29870fe95fd3f473bf83339e"
       "reports/raw/B5h2n5/T3_compare_roots.log 677b99afcac83e700de8dc6e16b810220e2f163c54cc1c460db1fb4debf6bc84"
       "reports/raw/B5h2n5/T3_oob.txt 0f8bd3f2dbccec51712a4a6edff0d1fb865fb76975efbe15f59bc06afb9bd48e"
       "reports/raw/B5h2n5/T3_domain_consistency.log 6f0ad3f199ea05fdcbe056839cf0a7cc57ee1af3e4fe2d6269ff941896b787e0"
       "reports/raw/B5h2n5/t3_domain_consistency.csv 83139e3ab40f819e57bd95a375d56d8b42ae9196479b350cea448020437f5e9d"
       "reports/raw/B5h2n5/T5_EGC_wave.log 2cb413b8d4609e8f1824c7d11b02a29d881ddf92ebd4798a0925e793843d90b7"
       "reports/raw/B5h2n5/T5_TtBA_wave.log 6b2e08148f27a11f55dd18fafb81916adb333a97982491abc031d780dca2b193"
       "reports/raw/B5h2n5/T5_EGC_gpu0.log 488149ac96a23e98e497f54f1c0a1f74236c0643cd38a75695b432afd23fc895"
       "reports/raw/B5h2n5/T5_EGC_gpu1.log c7e91aad6389c24347198057728f6c23c3b856902d1ed0cf7e2df44c69882018"
       "reports/raw/B5h2n5/T5_EGC_gpu2.log 3f378e716fbc297630982e8bd6ce106c900fbb8fb38bee686e8f15784b55cffe"
       "reports/raw/B5h2n5/T5_EGC_gpu3.log fb577bf6127d2a4e306437dbf5813c4906d68d683f481d54ebb885f746bf248a"
       "reports/raw/B5h2n5/T5_EGC_gpu4.log 15d1d8f62877e6d75124cd5df926a6818cc845265592ed438fd51ec238128fdb"
       "reports/raw/B5h2n5/T5_TtBA_gpu0.log a02b82007495ba84f33f5cca5e7cd808d9a79b61eb101b890516b99d8904697a"
       "reports/raw/B5h2n5/T5_TtBA_gpu1.log cbb5967619df7c17a2de367c0c43af135d10fd6551db62df6e2616aa4d48e0a0"
       "reports/raw/B5h2n5/T5_TtBA_gpu2.log 057853f1061ee4c819d713d864be66d462a1a230e1525b84d8661e639c31e738"
       "reports/raw/B5h2n5/T5_TtBA_gpu3.log 44097b7bff6f28bd27778af2e50cc2ac780a6c8d262b4853dbed131ddd4cb03f"
       "reports/raw/B5h2n5/T5_TtBA_gpu4.log 38fc0eb962964c62683e4d57bf76388bc924bf48e2fea12418e249fc782716df"
       "reports/raw/B5h2n5/T5_compare_roots.log 245d147c3470a588ed8d2c44eecacd6fe04032d0f17bacfc1a65290ce08f3571"
       "reports/B5h2n5_host_probe.md <given in the closing message>"
deviation_note: driver 595.71.05 recorded as a finding only; T2 re-ran the T1-B command on GPU 0 (T1 not repeated); T5 two D14-compliant waves
deviations: 5 (listed in the report)
stop_bundles: none
verdict: READY-FOR-REVIEW
```

Verdict line: **HOST-IDENTICAL** — T2, T3 (TtBA / CGBA / ARS_OPT, domain X = 0) and T5 (5 GPUs concurrent, each
bitwise IDENTICAL to GPU 0, both methods) all pass on the pre-registered stack torch 2.7.1+cu128 / cuDNN 90701;
T5b n/a (launcher threads = 8). B5 not started; no `Results/b5_*` or `Results/appendix_cgba_d8only_*` directory
created. STOP.
