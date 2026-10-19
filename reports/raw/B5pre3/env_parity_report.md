# B5-pre3 — Step 6 environment parity report (Beijing vs Harbin, CPU)

Raw logs: `reports/raw/B5pre3/agent_log.md` (Steps 1-5), `pip_check_before.txt`,
`pip_check_after.txt`, `pip_freeze_beijing.txt`, `pip_freeze_beijing_after_step4.txt`,
`pip_freeze_harbin.txt`, `parity.csv`, `reachability.csv`, `weights.md`, `ignored_harbin.txt`.

Head at start of this session: `fbeaacc` (B5-pre2 ... READY-FOR-GPU). Worktree: /root/autodl-tmp/LGC_frozen.

## Step 4 — installs performed (H rule-10 approval, exact list + rule-2 cascades)

Every command used `pip install --no-deps`; after each: `import` check and `pip check`.
All earlier-list packages were already satisfied (installed by the prior session, log truncated mid-
torchmetrics); each approved command was re-run (no-op) so the order is auditable in agent_log.md.

### Approved list
| package | target (Harbin) | result |
|---|---|---|
| numpy | 2.5.2 | already 2.5.2 (H override of rule (d)) |
| pillow | 10.3.0 | already 10.3.0 (H override of rule (d)) |
| huggingface_hub | 1.29.0 | already 1.29.0 |
| safetensors | 0.8.0 | already 0.8.0 |
| scipy | 1.18.1 | already 1.18.1 |
| timm | 1.0.28 | already 1.0.28 |
| transformers | 5.16.1 | already 5.16.1 |
| torchmetrics | 1.9.0 | already 1.9.0 |
| lpips | 0.1.4 | already 0.1.4 |
| piq | 0.8.0 | already 0.8.0 |
| robustbench | git@78fcc9e48a07a861268f295a777b975f25155964 | INSTALLED this session (robustbench-1.1) |

### Rule-2 cascades (pre-approved; each revealed by an import error; MISSING_ON_BEIJING with Harbin version)
- robustbench -> `geotorch==0.3.0`, `torchdiffeq==0.2.5`, `gdown==6.4.2`, `autoattack` @ git commit
  a39220048b3c9f2cca9a4d3a54604793c68eca7e (H's example direct-url) — all installed, all import OK.
- Earlier session cascades (logged in agent_log.md): `click==8.5.0`, `hf-xet==1.6.0`, `regex==2026.7.19`,
  `tokenizers==0.23.1`, `typer==0.27.1`, `annotated-doc==0.0.5`, `rich==15.0.0`, `shellingham==1.5.4`,
  `markdown-it-py==4.2.0`, `mdurl==0.1.2`, `lightning-utilities==0.15.3`.

### Known pip-check warning (no change made)
`robustbench 1.1 has requirement gdown==5.1.0, but you have gdown 6.4.2.` — gdown 6.4.2 is the Harbin
pin (parity.csv); installing 5.1.0 would deviate from Harbin, and rule 2 only permits MISSING_ON_BEIJING
packages. robustbench imports fine. The same warning is expected on Harbin.

## Reachable-but-missing modules: decision table (reachability.csv ∩ Harbin freeze)

| module | top/lazy | Harbin | decision |
|---|---|---|---|
| huggingface_hub | top | 1.29.0 | installed 1.29.0 |
| lpips | lazy | 0.1.4 | installed 0.1.4 |
| piq | lazy | 0.8.0 | installed 0.8.0 |
| robustbench | lazy | git@78fcc9e | installed git@78fcc9e |
| safetensors | top | 0.8.0 | installed 0.8.0 |
| scipy | top | 1.18.1 | installed 1.18.1 |
| timm | top | 1.0.28 | installed 1.0.28 |
| torchmetrics | lazy | 1.9.0 | installed 1.9.0 |
| transformers | top | 5.16.1 | installed 5.16.1 |
| apex | top | absent from Harbin | not installed (H point 4) |
| bidict | top | absent from Harbin | not installed (H point 4) |
| glog | top | absent from Harbin | not installed (H point 4) |
| inplace_abn | top | absent from Harbin | not installed (H point 4) |

Reachable but VERSION_DIFF and NOT missing on Beijing (rule (b) excludes them; not in H's list — recorded, unchanged):
matplotlib 3.10.5 (Harbin 3.9.0), pandas 3.0.6 (Harbin 3.0.5), typing-extensions 4.14.1 (Harbin 4.12.1),
pyyaml 6.0.2 (Harbin 6.0.1).

## H point 3c — autoencoder checkpoint
`checkpoints/vgg16_imagenet.pth` EXISTS (H copied from Harbin).
sha256 = `1d004c634f1bdc72fff9b12ab8fe8545cc2a2558512e5d51aa25181890f7f790` — starts with the asserted
prefix `1d004c634f1b` (ttba_github/run_frame_b1_generic.py:84-86). GAP from weights.md closed.

## H point 3d — ignored_harbin.txt
Of 44 `!!` paths: 26 PRESENT, 18 MISSING (git-ignored on Harbin, absent here). Missing list:
`.ipynb_checkpoints/run_b5_parallel-checkpoint.sh`, `.pytest_cache/`, `Night1.md`, `cache`,
`contract/.ipynb_checkpoints/`, `contract/paper_contract.yaml.bak_20261011_225106`,
`contract/paper_contract.yaml.bak_20261016_024247`, `contract/paper_contract.yaml.bak_rev3`,
`contract/paper_contract.yaml.bak_rev4_prep`, `contract/paper_contract.yaml.bak_v17_20261013_100904`,
`contract/paper_contract.yaml.bak_v18_20261015_122547`, `contract/paper_contract.yaml.v1.5.bak`,
`contract/paper_contract.yaml.v1.6.bak`, `lgc_preflight_kit_v1.2.zip`, `logs_frame/`, `opencode.json`,
`places.md`, `tools/eval/.ipynb_checkpoints/`. Nothing copied.

## H point 3e — from_pretrained on the torchvision-victim path
- All 4 `from_pretrained` hits are in /root/autodl-tmp/TtBA/dataset.py:247,248 (`cifar10-vit`,
  model_dir=`model/cifar10-vit`) and 257,258 (`cifar100-vit`, model_dir=`model/Vit_cifar100`).
- The torchvision victims load as datasetName `imagenet-resnet50` (dataset.py:289-296), `imagenet-vgg`
  (271-279), `imagenet-vit` (306-314) — all `torch.load("model/*.pth")`, no from_pretrained.
- adapters/run_ttba_v2.py and every file it reaches (14 files, reach_records.json) contain no
  from_pretrained / AutoModel / AutoImage / huggingface calls.
- VERDICT: no from_pretrained call is on the code path for resnet50 / vgg19 / vit_b_32; nothing would be
  downloaded (the two calls that exist take local directories, not remote model ids).

## H point 5e — disk
`overlay 30G, 13G used, 18G avail, 43% /`; `/dev/md0 50G, 3.8G used, 47G avail, 8% /root/autodl-tmp`
(down from 79% full before installs).

## Step 5 — post-install verification
- 5a: `2.7.1+cu128 0.22.1+cu128 90701` — exact match required.
- 5b: pip_check_after.txt diff vs before: only the robustbench/gdown line above.
- 5c: sim harness `[sim] 21 PASS, 0 FAIL ... ALL PASS`; contract `ALL PASS (21/21)`;
  dry-run identity NGPU=1 and NGPU=5 both `9b24fb045d3f1481fea39fb14105ba00d38c035a797115e6640af77378b669b6` → MATCH.
- 5d: lpips.LPIPS(net='vgg') on two seeded 1x3x64x64 tensors → 0.316558…; piq.DISTS() → 0.184203…; both ok,
  no downloads (vgg16-397923af.pth and dists_weights.pt present in ~/.cache/torch/hub/checkpoints).

## Deviations from the prompt
1. numpy 2.3.2→2.5.2 and pillow 11.3.0→10.3.0: H overrode prompt rule (d) ("never upgrade/downgrade") for
   these two packages only (H point 1). Both now equal Harbin.
2. huggingface_hub used 1.29.0 (H correction; parity.csv Harbin value), not 1.0.29.
3. Most approved installs were already performed by the previous session before this approval message
   (agent_log.md truncated mid-torchmetrics); this session re-ran each approved command (no-ops) and
   completed robustbench + its cascades.
4. pip check now reports one warning (robustbench/gdown pin) that was absent before; no change made —
   both versions are the Harbin pins.
5. Reachable VERSION_DIFF modules (matplotlib, pandas, typing-extensions, pyyaml) were left unchanged:
   rule (b) requires the module to be MISSING on Beijing; they are installed. Not in H's approval list.

## sha256 of files written this session
- reports/raw/B5pre3/agent_log.md (appended): 37475e33533e132b1e183200f59dbb1a88d9bfa1097d54ac68046c8ac0847358
- reports/raw/B5pre3/pip_check_after.txt: 2f29f197a86a0c21d0582086be8d7c7a1be29c810f414913565fbd954ebd1630
- reports/raw/B5pre3/pip_freeze_beijing_after_step4.txt: a115914ca01ce38dadaeeeb3d9a211e34833ab1753eb0a8687857fa709302fbd
- reports/raw/B5pre3/env_parity_report.md: sha256 printed in the session reply (cannot embed its own hash).

RESULT
phase: B5-pre3 env parity (CPU)
head: fbeaacc
parity_diff: 87 packages differ (post-Step-4); torch/torchvision unchanged yes; numpy changed 2.3.2 -> 2.5.2 (H override, = Harbin), pillow changed 11.3.0 -> 10.3.0 (H override, = Harbin); numpy/pillow after Step 4: 2.5.2 / 10.3.0
reachable_missing_before: huggingface_hub, lpips, piq, robustbench, safetensors, scipy, timm, torchmetrics, transformers (apex, bidict, glog, inplace_abn not in Harbin)
installed: numpy==2.5.2, pillow==10.3.0, huggingface_hub==1.29.0, safetensors==0.8.0, scipy==1.18.1, timm==1.0.28, transformers==5.16.1, torchmetrics==1.9.0, lpips==0.1.4, piq==0.8.0, robustbench@78fcc9e48a07a861268f295a777b975f25155964, geotorch==0.3.0, torchdiffeq==0.2.5, gdown==6.4.2, autoattack@a39220048b3c9f2cca9a4d3a54604793c68eca7e, click==8.5.0, hf-xet==1.6.0, regex==2026.7.19, tokenizers==0.23.1, typer==0.27.1, annotated-doc==0.0.5, rich==15.0.0, shellingham==1.5.4, markdown-it-py==4.2.0, mdurl==0.1.2, lightning-utilities==0.15.3
not_installed_with_reason: apex/bidict/glog/inplace_abn (absent from Harbin freeze); matplotlib 3.10.5 vs 3.9.0, pandas 3.0.6 vs 3.0.5, typing-extensions 4.14.1 vs 4.12.1, pyyaml 6.0.2 vs 6.0.1 (installed on Beijing, rule (b) excludes; not in H list)
victim_weights: all located
template_literal: explained
torch_after: 2.7.1+cu128 0.22.1+cu128 90701
sim: PASS 21/21
contract: PASS 21/21
dry_run_identity: MATCH N=1,5
lpips_dists_smoke: ok
verdict: READY-FOR-GPU
ae_checkpoint: present sha 1d004c634f1bdc72fff9b12ab8fe8545cc2a2558512e5d51aa25181890f7f790
ignored_missing: 18 (.ipynb_checkpoints/run_b5_parallel-checkpoint.sh, .pytest_cache/, Night1.md, cache, contract/.ipynb_checkpoints/, contract/paper_contract.yaml.bak_20261011_225106, contract/paper_contract.yaml.bak_20261016_024247, contract/paper_contract.yaml.bak_rev3, contract/paper_contract.yaml.bak_rev4_prep, contract/paper_contract.yaml.bak_v17_20261013_100904, contract/paper_contract.yaml.bak_v18_20261015_122547, contract/paper_contract.yaml.v1.5.bak, contract/paper_contract.yaml.v1.6.bak, lgc_preflight_kit_v1.2.zip, logs_frame/, opencode.json, places.md, tools/eval/.ipynb_checkpoints/)
STOP. Do not fix anything beyond Step 4. Do not commit.
