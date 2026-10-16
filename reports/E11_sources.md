# E11 — RobustBench Sources (2026-10-12)

## Model
- `Salman2020Do_R50` — RobustBench, ImageNet, Linf, eps=4/255
- Clean acc: 64.02% | Robust acc: 34.96%
- Loaded via `robustbench.utils.load_model('Salman2020Do_R50', 'imagenet', 'Linf')`
- Cache: `~/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt`
- SHA256: `1835a5bbe5e31ef35915889dc949b96528bed513205bb9ec1cac80c32c84ce46`
- Source: RobustBench official (Google Drive ID `1TmT5oGa1UvVjM3d-XeSj_XmKqBNRUg8r`);
  downloaded on Mac, transferred to AutoDL via rsync (Google Drive is unreachable from AutoDL/China).

## Protocol
- n = 200 per mode (both non_targeted and targeted)
- Loader: Resize(256) → CenterCrop(224) → ToTensor; [0,1] input, ImageNet normalisation inside the wrapper
- Same 8 checkpoints, budget 10,000, same artefacts as the main study
- Reduced roster {LGC_RC_EGC, TtBA, CGBA(NT)/CGBA_H(T)} per contract v1.5 D9
