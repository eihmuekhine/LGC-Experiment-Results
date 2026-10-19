# agent_log.md — B5h2n5 (host re-qualification, NGPU=5)

Head at phase start: `576c99d6cc1c183d7842e65173f4556fea1e7c6e` (paper-runs).
Prompt: `reports/raw/B5h2n5/prompt.txt` sha256 `937a2ae10d1aac0828320de0f6b10932a8783c246988dd4309aba1e5f9aab4e2`.
Contract: `contract/paper_contract.yaml` sha256 `ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c` (v1.9-rev7).
`run_b5_parallel.sh` sha256 `50d8941c6201196d4f8a4700d5bc13d305552b78fc1140ed6fa2b65d92a640fb`.

| time | task | command | exit | output file / first line |
|---|---|---|---|---|
| 04:55 | T0 | sha256sum prompt/contract/launcher; git rev-parse HEAD; hostname; cat env_build38.txt | 0 | 937a2ae1… ce101e33… 50d8941c… 576c99d6… autodl-container-851548b4b6-1e67012b |
| 04:55 | T0 | nvidia-smi -L; torch device names/versions; D12 flags via `import _common` | 0 | T0_host.log — 5 x NVIDIA GeForce RTX 5090; torch 2.7.1+cu128 tv 0.22.1+cu128 cudnn 90701; flags F/F/T/F/highest |
| 04:55 | T0 | driver query; cpu.max; nproc; free -g; df -h; echo TMUX | 0 | T0_sys.log — driver 595.71.05; cpu.max 8000000 100000 (Q=80); nproc 128; 709G avail; 47G free; TMUX=/tmp/tmux-0/default,1888,0 |
| 04:55 | T0 | `NGPU=5 bash run_b5_parallel.sh --dry_run` | 0 | T0_dryrun.log — [threads] Q=80.0 NGPU=5 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8; first cells e10_TtBA_densenet161_places365_NT/T, main_TtBA_vgg19_T |
| 04:56 | T2 | `CUDA_VISIBLE_DEVICES=0 MODES=0 N=3 OUT_PREFIX=Results/b5h2n5_B bash ttba_github/run_frame_b1.sh resnet50` (OMP=MKL=8) | 0 | T2_B.log — EGC 76 s, 3/3 ok -> Results/b5h2n5_B_resnet50 |
| 04:57 | T2 | compare_roots `--a Results/b4e2_smoke_LGC_RC_EGC_resnet50 --b Results/b5h2n5_B_resnet50` | 0 | T2_compare_roots.log — VERDICT: IDENTICAL (0 diffs, 24/24 npz) |
| 04:57 | T3 | `CUDA_VISIBLE_DEVICES=0 python adapters/run_ttba_v2.py --ttba_repo /root/autodl-tmp/TtBA --dataset imagenet-resnet50 --targeted 0 --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv --out_dir Results/b5h2n5_TtBA_resnet50 --limit 3` | 0 | T3_TtBA.log — 3/3 budget (46.8/47.4/46.6 s) |
| 04:59 | T3 | `CUDA_VISIBLE_DEVICES=0 python adapters/run_cgba.py --arch resnet50 --mode non_targeted --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv --repo /root/autodl-tmp/baselines/CGBA --out_dir Results/b5h2n5_CGBA_resnet50 --limit 3 --cgba_variant fixed` | 0 | T3_CGBA.log — 3/3 budget (19.5/18.2/18.7 s) |
| 05:01 | T3 | `CUDA_VISIBLE_DEVICES=0 python adapters/run_arsopt.py --arch resnet50 --mode non_targeted --split_csv splits/imagenet/ttba_github/resnet50_non_targeted.csv --repo /root/autodl-tmp/baselines/ARS-OPT --out_dir Results/b5h2n5_ARS_OPT_resnet50 --limit 3` | 0 | T3_ARS_OPT.log — 3/3 budget (28.2/27.8/28.2 s) |
| 05:02 | T3 | compare_roots vs b4e2_smoke_ttba_resnet50 / b4h_t3_smoke_CGBA_resnet50 / b4h_t3_smoke_ARS_OPT_resnet50 | 0/0/0 | T3_compare_roots.log — all three VERDICT: IDENTICAL (0 diffs, 24/24 npz) |
| 05:02 | T3 | oob_queries/nonfinite from runs_*.csv | 0 | T3_oob.txt — TtBA 8435/8546/8457; CGBA 9800/9369/9828; ARS_OPT 10000x3; nonfinite 0 |
| 05:03 | T3 | `python adapters/diag/domain_consistency.py --roots Results/b5h2n5_CGBA_resnet50 Results/b5h2n5_ARS_OPT_resnet50 --out reports/raw/B5h2n5/t3_domain_consistency.csv` | 0 | T3_domain_consistency.log — X=0 both methods (48 rows) |
| 05:03 | T5 | wave1 EGC row 1: `CUDA_VISIBLE_DEVICES=$i python run_benchmark_33b.py … --num_samples 1 … --out_dir Results/b5h2n5_gpu${i}_LGC_RC_EGC --save_adv` x5 concurrent + wait | 0 | T5_EGC_gpu0-4.log — 5/5 ok, q=9975, 25.2/24.84/25.27/24.96/25.03 s |
| 05:04 | T5 | wave2 TtBA row 1: `CUDA_VISIBLE_DEVICES=$i python adapters/run_ttba_v2.py … --out_dir Results/b5h2n5_gpu${i}_TtBA --limit 1` x5 concurrent + wait | 0 | T5_TtBA_gpu0-4.log — 5/5 budget, q=9999, L2@10k 2.3313405513763428, 46.7/46.9/46.8/45.9/46.2 s |
| 05:05 | T5 | compare_roots gpu$i vs gpu0, i=1..4, both methods | 0 | T5_compare_roots.log — all 8 VERDICT: IDENTICAL (0 diffs, 8/8 npz) |
| 05:05 | T5b | read [threads] of T0 dry run: `Q=80.0 NGPU=5 -> OMP_NUM_THREADS=MKL_NUM_THREADS=8` | n/a | 8 is not < 8 → T5b n/a (launcher threads = 8) |
