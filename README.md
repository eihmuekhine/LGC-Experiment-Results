# Latent Geometric Chords for Query-Efficient Decision-Based Adversarial Attacks

Official code release for **Latent Geometric Chords (LGC)** and its efficient variant **LGC-H**, a decision-based hard-label black-box adversarial attack that performs curvature-aware search in a compact latent space while using **Residual-based Adversarial Generation (RAG)** to reduce autoencoder reconstruction artifacts.

## Overview

LGC combines three components:

1. **Latent-space optimization**
   \[
   z_s = E(x_0).
   \]
2. **Curvature-aware geometric boundary search** using hard-label queries.
3. **Residual-based Adversarial Generation (RAG)**
   \[
   x_{adv}=\operatorname{clip}_{[0,1]}\left[x_0+G(z_{adv})-G(z_s)\right].
   \]

The implementation supports both **non-targeted** and **targeted** attacks.

---

## Threat Model

LGC operates in a **decision-based black-box** setting.

- **Non-targeted:** success when the predicted class differs from the original class.
- **Targeted:** success only when the predicted class equals the target class.
- Attack optimization uses only hard-label model outputs.

---

## LGC and LGC-H

### LGC

LGC estimates the local decision-boundary normal and performs a semicircular geometric search in the 2-D plane induced by the current source-to-boundary direction and the estimated normal. A binary search along the semicircular path refines the next boundary point.

### LGC-H

LGC-H uses an aggressive angular-halving strategy to locate a valid improving direction more quickly while retaining the same latent-space and RAG formulation.

---

## Autoencoder Backbones

### VGG16 Autoencoder

Default latent representation used by LGC:

- Input: `3 x 224 x 224`
- Latent representation: `512 x 7 x 7`
- Decoder output: `3 x 224 x 224`
- Final decoder activation: sigmoid
- VGG16 configuration: `[2, 2, 3, 3, 3]`

### ResNet50 Autoencoder

Used for the autoencoder-backbone comparison:

- Input: `3 x 224 x 224`
- Latent representation: `2048 x 7 x 7`
- Decoder output: `3 x 224 x 224`
- Final decoder activation: sigmoid

---

## Recommended Repository Structure

```text
Latent-Geometric-Chords/
├── README.md
├── requirements.txt
├── environment.yml
│
├── models/
│   ├── builder.py
│   ├── vgg.py
│   └── resnet.py
│
├── attacks/
│   └── lgc.py
│
├── scripts/
│   ├── run_places365_targeted.py
│   ├── run_places365_nontargeted.py
│   └── evaluate_places365.py
│
├── training/
│   ├── train.py
│   ├── eval.py
│   ├── dataloader.py
│   ├── train.sh
│   └── reconstruct.py
│
├── metadata/
│   ├── categories_places365.txt
│   └── adv_test_labels.txt
│
├── splits/
│   ├── places365_resnet50_nontargeted_200.txt
│   ├── places365_resnet50_targeted_200.txt
│   ├── places365_densenet161_nontargeted_200.txt
│   └── places365_densenet161_targeted_200.txt
│
└── checkpoints/
    └── README.md
```

---

## Environment

The manuscript reports experiments using:

```text
Python 3.12
PyTorch 2.8.0
NVIDIA GPU
```

Additional packages used by the reviewed code include:

```text
torchvision
numpy
Pillow
lpips
torchmetrics
matplotlib
loguru
```

For exact reproducibility, provide a pinned `requirements.txt` or `environment.yml` generated from the environment used for the reported experiments.

---

## Places365 Target Models

Supported target classifiers:

```text
ResNet50
DenseNet161
```

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Images are converted to RGB and resized to `224 x 224` in the current Places365 attack scripts.

---

## Data Preparation

Expected Places365 directory:

```text
Places365/
├── images/
├── adv_test_labels.txt
└── categories_places365.txt
```

`adv_test_labels.txt` maps each experiment image filename to its ground-truth Places365 class index.

`categories_places365.txt` contains the 365 Places365 classes indexed from `0` to `364`.

For exact paper reproduction, use fixed experiment lists rather than sampling new examples at runtime.

The manuscript reports:

- **200 correctly classified images** for each Places365 non-targeted experiment.
- **200 correctly classified source-target pairs** for each Places365 targeted experiment.

Any 30-image or 30-pair files used during development should be treated only as smoke-test/debug subsets.

---

## Autoencoder Checkpoint

LGC requires a trained autoencoder checkpoint, for example:

```text
checkpoints/vgg16_imagenet_autoencoder.pth
```

The public release should provide either the exact checkpoint used for the paper or a download link together with its SHA256 checksum.

Document:

```text
checkpoint filename
autoencoder architecture
training dataset
training epoch
input preprocessing
checkpoint SHA256
```

---

## Autoencoder Training

The reviewed training launcher uses:

```text
epochs          = 100
batch size      = 64
learning rate   = 0.05
momentum        = 0.9
weight decay    = 1e-4
workers         = 16
```

Example:

```bash
bash train.sh vgg16 ImageNet
```

The training pipeline requires the corresponding `train.py`, `dataloader.py`, and evaluation code. These files must be included if users are expected to reproduce the autoencoder checkpoint from scratch.

For exact reproduction, keep the original loss function, optimizer, learning-rate schedule, preprocessing, random seed, and checkpoint-selection rule unchanged.

---

## Reconstruction Check

Before running LGC, verify the trained autoencoder:

```bash
python reconstruct.py \
    --arch vgg16 \
    --resume checkpoints/vgg16_imagenet_autoencoder.pth \
    --val_list list/ImageNet_list.txt
```

This provides a visual comparison between the input and reconstructed image and is useful because RAG depends directly on `G(z) - G(z_s)`.

---

## Targeted Initialization

For source image \(x_0\) and target image \(x_{target}\), the targeted implementation constructs a pseudo-target.

First compute:

\[
R=x_0-G(z_s).
\]

Then:

\[
z_{target}=E\left(\operatorname{clip}_{[0,1]}(x_{target}-R)\right).
\]

Before starting the targeted boundary search, the pseudo-target should be verified to predict the intended target class.

---

## Running Places365 Non-Targeted LGC

The reviewed code contains the non-targeted LGC/LGC-H attack implementation. The public repository should also include a complete non-targeted Places365 experiment runner.

Recommended interface:

```bash
python run_places365_attack.py \
    --mode non_targeted \
    --attack_method LGC_H \
    --target_archs resnet50 densenet161 \
    --ae_checkpoint checkpoints/vgg16_imagenet.pth \
    --base_dir ./data/Places365 \
    --num_samples 5
```
---

## Running Places365 Targeted LGC

Using the current development runner:

```bash
python run_places365_attack.py \
    --mode targeted \
    --attack_method LGC \
    --target_archs resnet50 densenet161 \
    --ae_checkpoint checkpoints/vgg16_imagenet.pth \
    --base_dir ./data/Places365 \
    --num_samples 5
```

The runner supports:

```text
LGC
LGC_H
```

For a cleaned public release, expose the target classifier, split, attack, seed, and query budget as command-line arguments.

---

## Main Attack Hyperparameters

The reviewed Places365 implementation uses:

```text
seed              = 992
initial_query      = 30
sigma              = 0.0002
tol                = 1e-4
max_queries        = 10000
gradient batch     = 40
```

The number of normal-estimation samples grows with iteration:

\[
q_i=\left\lfloor N_0\sqrt{i+1}\right\rfloor.
\]

For exact reproduction, keep these synchronized with the final manuscript and released experiment configuration.

---

## Evaluation Metrics

Primary pixel-space metrics:

```text
L2
Linf
SSIM
LPIPS
ASR
```

SSIM is computed on images in `[0,1]`.

LPIPS inputs are converted from `[0,1]` to `[-1,1]`.

Targeted ASR:

```text
1 if predicted_label == target_label
```

Non-targeted ASR:

```text
1 if predicted_label != original_label
```

---

## Query Checkpoints

For exact paper-table reproduction, use fixed query budgets such as:

```text
1000
2500
5000
10000
```

A checkpoint should store the last valid adversarial state whose source query count does not exceed that budget.

Recommended fixed-shape output:

```text
l2_pixel      : [N, num_checkpoints]
linf_pixel    : [N, num_checkpoints]
ssim          : [N, num_checkpoints]
lpips         : [N, num_checkpoints]
asr           : [N, num_checkpoints]
source_query  : [N, num_checkpoints]
```

---

## Output

The current targeted runner saves NumPy `.npz` files containing fields including:

```text
image_names
source_names
target_names
original_labels
target_labels
adversarial_labels
all_queries
all_adv_labels
l2_latent
linf_latent
l2_pixel
linf_pixel
ssim
lpips
asr
```

Visual outputs include clean images, adversarial images, and normalized perturbation visualizations at selected query milestones.

---

## Reproducibility Checklist

- [ ] Exact VGG16 autoencoder checkpoint is provided.
- [ ] Autoencoder checkpoint SHA256 is documented.
- [ ] `train.py`, `dataloader.py`, and `eval.py` are included if training from scratch is supported.
- [ ] Exact autoencoder training preprocessing is documented.
- [ ] Exact 200-image non-targeted splits are provided.
- [ ] Exact 200-pair targeted splits are provided.
- [ ] ResNet50 and DenseNet161 experiment configurations are included.
- [ ] Query checkpoints match the manuscript exactly.
- [ ] No query budget is exceeded when a checkpoint result is recorded.
- [ ] Random seed behavior is documented.
- [ ] Python and dependency versions are pinned.
- [ ] Target labels are explicitly validated in targeted attacks.
- [ ] A script regenerates the reported tables from saved results.

---

## Development-File Cleanup

The reviewed development code contains duplicate or legacy filenames such as:

```text
builer.py
vgg(1).py
utils(2).py
Targeted_Places365.py
Targeted_proposed_attack.py
```

For a public release, keep one canonical implementation and move obsolete versions outside the main execution path.

Recommended canonical names:

```text
builder.py
vgg.py
resnet.py
lgc.py
run_places365_targeted.py
run_places365_nontargeted.py
```

---

## Citation

```bibtex
@article{khine2026latent,
  title   = {Latent Geometric Chords for Query-Efficient Decision-Based Adversarial Attacks},
  author  = {Khine, Ei Hmue and Li, Yao and Sun, Jiebao and Shi, Shengzhu and Guo, Zhichang and Wu, Boying},
  journal = {arXiv preprint arXiv:2605.31219},
  year    = {2026}
}
```

---

## Acknowledgment

LGC builds on geometric decision-based black-box attack research and investigates how curvature-aware boundary search can be transferred from pixel space to compact semantic representations while reducing autoencoder reconstruction artifacts through RAG.
