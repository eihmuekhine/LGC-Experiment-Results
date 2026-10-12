# C2b — Contract artefacts: Places365 (E10) + RobustBench (E11) support (contract v1.5 D9)

Status: **DONE**. All four deliverables produced and verified on CPU/GPU; both test suites green.

## 1. Code files (new / changed) — sha256

| File | Change | SHA256 |
|---|---|---|
| `adapters/_common.py` | added `build_places365_victim`, `load_image_places365`, `load_places365_labels`, `build_robust_victim`, `robust_victim_info`, `NormalizedVictim`, E10/E11 constants | `7869457f0cc119c4165c64a0c466e5e83c3e217eec1b9c5ed6607f22ba3f16e9` |
| `ttba_github/ttba_dump_split_places365.py` | new split dumper (E10) | `dc55ed6c5185461c1e52fb0b4adeceb39216f0ef43074fd77f3287092cd1e005` |
| `ttba_github/ttba_dump_split_robust.py` | new split dumper (E11) | `3eb302b2bec928125bba57ed0bb444600651aa6dcd136b21df14ec2d32403080` |
| `tools/eval/check_contract.py` | `--split_dir` now `nargs="+"` (multi-path, backward compatible) | `56a8040cb3701ff6b68698e63487a38d1bcfa45b408653e11ca691413ad0f128` |
| `tools/eval/tests_contract/test_contract.py` | T14a/T14b added (E10/E11 REQUIRED cannot be cut); 15 checks | `3fc03c693d2578f2eca0840344f84431035ee7b801ac7c7d19fa8ad10e5e79b0` |

Victim weight sha256 (asserted at load time in `_common.py`):
- `resnet50_places365.pth.tar` — `46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5`
- `densenet161_places365.pth.tar` — `b311cb0e8bea8ecf34dee58da317c6217894db94721005c54cadcb7f2f2f9598`
- `Salman2020Do_R50.pt` — `1835a5bbe5e31ef35915889dc949b96528bed513205bb9ec1cac80c32c84ce46`

Note: `build_places365_victim` strips the `module.` prefix and, for `densenet161`, remaps the
official 2018 checkpoint's old dense-layer names (`norm.1`/`norm.2`/`conv.1`/`conv.2`) to the current
torchvision names (`norm1`/`norm2`/`conv1`/`conv2`); missing `num_batches_tracked` is handled by `_BatchNorm`.

## 2. New split CSVs — sha256

| File | SHA256 |
|---|---|
| `splits/places365/resnet50_places365_non_targeted.csv` | `351bc50f594f7ddcf7746cf7769ffc3ea5329271b150d762099feed4bf60012f` |
| `splits/places365/resnet50_places365_targeted.csv` | `70bd3d5c5989df6829b39a310a59203d6a5ee760f7481a5b213a2df49107eaaa` |
| `splits/places365/densenet161_places365_non_targeted.csv` | `89252849771532358f422f0046f47b91c391429c77731993881946780166a547` |
| `splits/places365/densenet161_places365_targeted.csv` | `51935b7522274cebd361b5b91e4649d18bb8da5af739010670e733b2efc4b790` |
| `splits/robust/Salman2020Do_R50_non_targeted.csv` | `1b847d4c3acc858a992a202bbdc6120f525922a1a8a42a0be2c6575a480778a7` |
| `splits/robust/Salman2020Do_R50_targeted.csv` | `2985ebc400ea2f292077f6c84dc809ef6591629bbd81e606b0c429a8c062d00a` |

Each CSV has 200 rows (rows 1–200 = the first 200 correctly-classified sources under seed 42; targeted
pairs are 200 distinct sources, each with the same-row target). `unique_sources = 200` for every split.
Each has a matching `*.json` meta file (sha256 available in the splits directory).

## 3. Accuracy gate (E10, both Places365 victims) — PASS

| Victim | tested | correct | natural accuracy | reference | gate (+/-0.05) |
|---|---|---|---|---|---|
| `resnet50_places365` | 1000 | 548 | 0.5480 | 0.550 | OK |
| `densenet161_places365` | 1000 | 570 | 0.5700 | 0.565 | OK |

Both within +/-5 pts of the reference; neither STOP was triggered.

## 4. RobustBench (E11) victim inspection

`build_robust_victim(device)` = `robustbench.utils.load_model('Salman2020Do_R50', 'imagenet', 'Linf')`.

- **type**: `torch.nn.modules.container.Sequential`
- **submodules**: `["normalize", "model"]`
- **internal normalisation**: **yes** (ImageNet `normalize` layer is inside the wrapper) → input `[0,1]`, no
  external normalisation needed; the victim returns logits directly.
- **forward probe**: `rand(1,3,224,224)` in `[0,1]` → logits shape `[1,1000]`, sample argmax **741**
  (deterministic across repeated random `[0,1]` inputs, a known behaviour of this adversarially trained model).
- **victim_weight_file** = `Salman2020Do_R50.pt`; **victim_weight_sha256** =
  `1835a5bbe5e31ef35915889dc949b96528bed513205bb9ec1cac80c32c84ce46`.

## 5. `--split_dir` CLI (multi-path)

`tools/eval/check_contract.py --split_dir` now accepts one or more directories (first dir containing
`<arch>_<mode>.csv` wins); a single path still works unchanged.

```bash
# single path (backward compatible)
python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage final \
    --roots <out_dirs> --split_dir splits/imagenet/ttba_github --report reports/contract_final.md

# multiple paths (E10/E11)
python tools/eval/check_contract.py --contract contract/paper_contract.yaml --stage final \
    --roots <out_dirs> \
    --split_dir splits/imagenet/ttba_github splits/places365 splits/robust \
    --report reports/contract_final.md
```

## 6. Tests

| Suite | Result |
|---|---|
| `python tools/eval/tests_contract/test_contract.py` | **ALL PASS (15/15)** |
| `python tools/eval/selftest_evalkit.py` | **ALL PASS (20/20)** |
