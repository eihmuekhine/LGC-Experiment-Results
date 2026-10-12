# E10 — Places365 Sources (2026-10-12)

## Models (official MIT Places365)
| File | SHA256 |
|---|---|
| `resnet50_places365.pth.tar` (93 MB) | `46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5` |
| `densenet161_places365.pth.tar` (106 MB) | `b311cb0e8bea8ecf34dee58da317c6217894db94721005c54cadcb7f2f2f9598` |
| `categories_places365.txt` (6.7 KB) | (same as in filelist tar) |

## Data
- **val_256.tar**: `http://data.csail.mit.edu/places/places365/val_256.tar` (501 MB, via aria2c)
- **Extracted**: `/root/autodl-tmp/places365_data/val_256/` — **36,500 images**, flat (`Places365_val_XXXXXXXX.jpg`)
- **Label file**: `filelist_places365-standard.tar` (`http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar`, 64 MB)
- **`places365_val.txt`**: 1.1 MB, **36,500 entries**, format `Places365_val_XXXXXXXX.jpg <class_index>`
- **Class indices are 0..364** (365 classes, 100 images/class); ordering is **randomized** (NOT contiguous)

## Preprocessing
- Loader: Resize(256) → CenterCrop(224) → ToTensor → ImageNet normalisation
  (mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`) — same as the official Places365 demo
- Victim input: [0,1] NCHW, 224x224; normalisation inside the wrapper

## Protocol
- n = 200 per (victim, mode); both `non_targeted` and `targeted`
- Selection: seeded random permutation of correctly classified images (seed = 42 + row_index)
- 8 contract checkpoints; budget 10,000; same artefacts as the main study
