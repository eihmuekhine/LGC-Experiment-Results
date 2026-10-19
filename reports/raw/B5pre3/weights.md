# B5-pre3 Step 3 — weights & template literal (Beijing, CPU)

## 3a. Victim weights and the LGC autoencoder

| victim | loaded at (file:line) | file or URL | exists locally? |
|---|---|---|---|
| resnet50 | adapters/_common.py:146-153 `load_victim` -> TTBA_GH_WEIGHTS map (lgc_bench/baseline_protocol.py:319); engine build38: run_benchmark_33b.py:677-683 (`--victim_ckpt_dir`) | `/root/autodl-tmp/TtBA/model/resnet50-11ad3fa6.pth` (IMAGENET1K_V2; hub copy `~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth` also present) | YES |
| vgg19 | same as above | `/root/autodl-tmp/TtBA/model/vgg19-dcbb9e9d.pth` (hub cache has NO vgg19 copy; not needed — loaded from file) | YES |
| vit_b_32 | same as above | `/root/autodl-tmp/TtBA/model/vit_b_32-d86f8d99.pth` (hub cache has no vit_b_32 copy; not needed — loaded from file) | YES |
| resnet50_places365 | adapters/_common.py:82,87-93,194-213 (`build_places365_victim`); engine lgc_bench/models.py:22-25,50-84 (URL fallback) | `/root/autodl-tmp/models/places365/resnet50_places365.pth.tar` (sha256 pinned: 46529c86…) + hub copy present | YES |
| densenet161_places365 | same as above | `/root/autodl-tmp/models/places365/densenet161_places365.pth.tar` (sha256 pinned: b311cb0e…) + hub copy present | YES |
| Salman2020Do_R50 | adapters/_common.py:93-97,219-234 (`build_robust_victim` -> robustbench.utils.load_model, model_dir=`~/.cache/robustbench/models`) | `~/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt` (sha256 pinned: 1835a5bb…) | YES |
| LGC autoencoder (vgg16) | run_benchmark_33b.py:417 (`--ae_checkpoint` default `checkpoints/vgg16_imagenet.pth`), lgc_bench/models.py:115-150 (`load_autoencoder`, `torch.load` — no auto-download); ttba_github/run_frame_b1_generic.py:72,84-86 (asserts sha256 prefix `1d004c634f1b`) | `checkpoints/vgg16_imagenet.pth` (repo root; repo has NO `checkpoints/` dir) | **MISSING (GAP)** — `find /` (depth 6) found no `vgg16_imagenet*.pth` anywhere on this host; the loader does not download, so the B5 engine/ttba-frame cells would abort at AE load |

Checks done (extant files): `ls ~/.cache/torch/hub/checkpoints/`, `ls /root/autodl-tmp/models/places365/`,
`ls ~/.cache/robustbench/models/imagenet/Linf/`, `ls /root/autodl-tmp/TtBA/model/`, `ls checkpoints/` (missing),
`find / -maxdepth 6 …` for the AE file.

Hub-cache note: contains resnet50-0676ba61 (V1), resnet50-11ad3fa6 (V2), vgg16-397923af (lpips), vit_b_16-c867db91,
densenet161-8d451a50, both places365 .pth.tar, `dists_weights.pt` (piq), alexnet, resnet18, densenet121.
No download should occur for any of the 6 victims at attack time (all loaded from local files).

## 3b. The `{arch}_{mode}.csv` template literal

- Template source: `run_b5_parallel.sh` engine cells pass
  `--split_csv "splits/imagenet/ttba_github/{arch}_{mode}.csv"` (lines 191–226).
- Expansion: `run_benchmark_33b.py:631` does `args.split_csv.format(arch=arch, mode=args.mode)`, with
  arch ∈ {resnet50, vgg19, vit_b_32} and mode ∈ {non_targeted, targeted} -> 6 files.
- Adapters do NOT expand the template: `run_b5_parallel.sh` passes the already-expanded literal path to every
  TtBA/CGBA/ARS-OPT cell (lines 232–320: `splits/imagenet/ttba_github/<arch>_<mode>.csv`). The same template
  string also appears (diagnostics only) in `adapters/fidelity.py:264,317`.
- All 6 expansions exist under `splits/imagenet/ttba_github/`, each with 1000 image rows (rows 1–200 available):
  resnet50_non_targeted.csv, resnet50_targeted.csv, vgg19_non_targeted.csv, vgg19_targeted.csv,
  vit_b_32_non_targeted.csv, vit_b_32_targeted.csv (plus matching .json and overlap_dev_seeds.json).
  VERDICT: **explained, all 6 present**.