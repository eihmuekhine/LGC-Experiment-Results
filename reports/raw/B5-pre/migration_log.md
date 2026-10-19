# B5-pre migration log (agent)

Prompt: `reports/raw/B5-pre/migration_prompt.txt`
Prompt sha256: `1059a536fecde5300651aedb257a1978ef36779aa46b2b058fcf79171ab8a30f`
Active LESSONS read at start: L01, L02, L03, L06, L22, L23, L24, L25, L26 (see docs/LESSONS.md).

## Step 0 — investigate (raw outputs)

- `grep -c 'models/places365\|ttba_mini_flat\|TtBA/data' reports/raw/B5-pre/transfer_include.txt` -> `1`
  (the single match is line 27 `models/places365`; `ttba_mini_flat` and `TtBA/data` are NOT in the manifest).
- `du -sh /root/autodl-tmp/TtBA/data/imagenet/archive/imagenet-mini/val` -> `489M`
- `diff <(cd /root/autodl-tmp/models/places365 && sha256sum *.pth.tar) <(cd /root/autodl-tmp/LGC_frozen/models/places365 && sha256sum *.pth.tar)` -> empty (diff-exit=0; identical content)
- `ls -la /root/autodl-tmp/LGC_frozen/models/imagenet/` -> `Linf/` containing `Salman2020Do_R50.pt`
- `find ~/.cache/torch/hub -type f | xargs du -ch | tail -5` -> `1.7G total` (alexnet, densenet121/161, resnet18/50, resnet50_places365, densenet161_places365, vgg16, vit_b_16, dists_weights.pt)
- `find /root/autodl-tmp -iname '*robust*bench*' -o -iname '*salman*'` -> `models/robustbench.sha256` + `models/imagenet/Linf/Salman2020Do_R50.pt` (repo) + empty `/root/autodl-tmp/.cache/robustbench`; the actual RobustBench weight is at `/root/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt` (sha256 `1835a5bbe5e3…`, identical to the repo copy)
- `find /root/autodl-tmp/ttba_mini_flat/images -type l ! -path '*.ipynb_checkpoints*' | wc -l` -> `3923` (matches; no E-SPEC)

### Additional investigation (Step 0 / Step 1 evidence)

Reading `run_b5_parallel.sh` (already git-tracked, will be on Beijing via clone) and `adapters/_common.py` shows
hard-coded external paths that the B5 launcher and victim loaders require beyond the git clone + transfer_include.txt:

| path | needed for | evidence |
|---|---|---|
| `/root/autodl-tmp/TtBA` | TtBA baseline (`--ttba_repo`) + main-ImageNet victim weights (`--victim_ckpt_dir "$TTBA_REPO/model"`) + x0 images (`data/imagenet/archive/imagenet-mini/val`) | `run_b5_parallel.sh:182,192`; `adapters/run_ttba_v2.py:279-337` (`import TtBA`, `import dataset`, `load_dataset_model`) |
| `/root/autodl-tmp/ttba_mini_flat` | EGC data_dir + labels | `run_b5_parallel.sh:181,190`; `adapters/_common.py:534` |
| `/root/autodl-tmp/models/places365` | Places365 victim weights | `adapters/_common.py:82,204` (`PLACES365_MODEL_DIR`) |
| `/root/autodl-tmp/places365_data` | Places365 x0 images + labels | `adapters/_common.py:83-85,508-531` (`val_256`, `places365_val.txt`) |
| `/root/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt` | RobustBench victim weight | `adapters/_common.py:94,229` (`ROBUST_MODEL_DIR`) |
| `/root/autodl-tmp/baselines/CGBA` | CGBA baseline (`--repo "$CGBA_REPO"`) | `run_b5_parallel.sh:183,257` |
| `/root/autodl-tmp/baselines/ARS-OPT` | ARS-OPT baseline (`--repo "$ARS_REPO"`) | `run_b5_parallel.sh:184,279` |

Notes:
- The prompt's context (b) said "prefer /root/autodl-tmp/LGC_frozen/models/places365/ (the one the repo code actually
  ships with)"; the code actually loads Places365 weights from `/root/autodl-tmp/models/places365`
  (`_common.py:82`), so the external copy is the real dependency. The repo copy (`models/places365`) is already in
  transfer_include.txt line 27.
- The torch hub cache (`~/.cache/torch/hub`, 1.7G) is NOT used by the B5 flow: main-ImageNet victims load via
  `build_ttba_github_victim` (torchvision constructor WITHOUT weights + `load_state_dict` from `TtBA/model`),
  not `tvm.get_model(weights=…)`. So it is omitted from the manifest.
- `HSJA` / `Sign-OPT` appear in adapter code but are not referenced by `run_b5_parallel.sh`; omitted.

## Step 1 — externals_manifest.txt

Wrote `reports/raw/B5-pre/externals_manifest.txt` (one absolute path per line, deduplicated):

```
/root/autodl-tmp/TtBA
/root/autodl-tmp/ttba_mini_flat
/root/autodl-tmp/models/places365
/root/autodl-tmp/places365_data
/root/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt
/root/autodl-tmp/baselines/CGBA
/root/autodl-tmp/baselines/ARS-OPT
```

## Step 2 — git clone on Beijing

- Connectivity check (BatchMode, no password): `ssh -o BatchMode=yes -p 16126 root@connect.bjb2.seetacloud.com
  'hostname; whoami; nvidia-smi -L'` -> host `autodl-container-edf540a10d-7b2b9f69`, user `root`,
  `nvidia-smi -L` empty.
- Clone command:
  ```
  ssh -p 16126 root@connect.bjb2.seetacloud.com "mkdir -p /root/autodl-tmp && cd /root/autodl-tmp && \
    git clone -b paper-runs git@github.com:eihmuekhine/LGC-Experiment-Results.git LGC_frozen 2>&1 || \
    (cd LGC_frozen && git fetch origin paper-runs && git reset --hard origin/paper-runs)"
  ```
  Output:
  ```
  Cloning into 'LGC_frozen'...
  Warning: Permanently added 'github.com' (ED25519) to the list of known hosts.
  git@github.com: Permission denied (publickey).
  fatal: Could not read from remote repository.

  Please make sure you have the correct access rights
  and the repository exists.
  bash: line 1: cd: LGC_frozen: No such file or directory
  ```

**STOP (E-SPEC):** GitHub auth error (`Permission denied (publickey)`). Per prompt Step 2, STOPPED; no private key
or credential was copied to Beijing. STOP bundle: `reports/stop/B5-pre_step2_20261019-0806.md`.

Steps 3 (transfer) and 4 (verify) were NOT run (blocked by Step 2).

_(The above Step 2 STOP is superseded: H added a read-only GitHub deploy key to Beijing and confirmed
`ssh -T git@github.com` succeeds there. Re-run below, 2026-10-19.)_

## Step 2 (re-run) — git clone on Beijing

- Connectivity check (BatchMode, no password):
  `ssh -o BatchMode=yes -p 16126 root@connect.bjb2.seetacloud.com 'hostname; whoami; nvidia-smi -L; ssh -T -o StrictHostKeyChecking=accept-new git@github.com'`
  -> host `autodl-container-edf540a10d-7b2b9f69`, user `root`,
  `nvidia-smi -L` -> `bash: line 1: /usr/bin/nvidia-smi: Permission denied` (raw; not judged),
  GitHub -> `Hi eihmuekhine/LGC-Experiment-Results! You've successfully authenticated, but GitHub does not provide shell access.`
- Clone command (exactly as prompt Step 2):
  ```
  ssh -p 16126 root@connect.bjb2.seetacloud.com "mkdir -p /root/autodl-tmp && cd /root/autodl-tmp && \
    git clone -b paper-runs git@github.com:eihmuekhine/LGC-Experiment-Results.git LGC_frozen 2>&1 || \
    (cd LGC_frozen && git fetch origin paper-runs && git reset --hard origin/paper-runs)"
  ```
  -> `Cloning into 'LGC_frozen'...` … `Updating files: 100% (579/579), done.` (success; no auth error).
- `ssh ... "cd /root/autodl-tmp/LGC_frozen && git log --oneline -1 && git status --short"`
  -> `56ba062 B5-fix: launcher v2 fix round 1+1b (R1-R7 + test_gates_monitor.py pin), reviewed PASS-WITH-NOTES`
     (HEAD `56ba0625fda83d441b8fdb0a97fcc699319172ee`; `git status --short` empty = clean). Matches expectation.

## Step 3 (re-run) — transfer

1. transfer_include (prompt cmd, run verbatim):
   ```
   rsync -avzP --files-from=reports/raw/B5-pre/transfer_include.txt -e "ssh -p 16126" \
     /root/autodl-tmp/LGC_frozen/ root@connect.bjb2.seetacloud.com:/root/autodl-tmp/LGC_frozen/
   ```
   -> `0 files...34 files to consider` / `total size is 0` / `sent 1,080 bytes`.
   **BUG A:** `--files-from` with directory entries and no trailing slash copies only the empty directory, not
   its contents. Result on Beijing: `Results/` = 4.0K (empty), `models/imagenet` = 0, `models/places365` = 0,
   `splits/kat` = 0 (Harbin: 1.4G / 98M / 198M / 8.0K). ~855M of untracked files NOT transferred.
2. reports/raw/B5-pre/ (prompt cmd, has trailing slash) -> OK, 5 files, `total size is 40,750`.
3. externals loop (prompt cmd `rsync -avzP -l "<path>" ":<same path>"`):
   - `/root/autodl-tmp/TtBA` -> ran, `total size is 1,635,750,039` (~1.6G), but **BUG B:** landed NESTED at
     `/root/autodl-tmp/TtBA/TtBA/` (content one level too deep).
   - `/root/autodl-tmp/ttba_mini_flat` -> ran, `total size is 791,697` (612K), NESTED at
     `/root/autodl-tmp/ttba_mini_flat/ttba_mini_flat/`.
   - STOPPED before the remaining 5 (`models/places365`, `places365_data`, `.cache/robustbench/…/Salman2020Do_R50.pt`,
     `baselines/CGBA`, `baselines/ARS-OPT`).

**STOP (E-SPEC):** two rsync trailing-slash bugs. Local repro (identical flags) confirms both are deterministic
rsync semantics, not network/provisioning:
- Bug A: `--files-from` dir entry w/o trailing slash -> empty dir only (verified: `d1` vs `d1/`).
- Bug B: `rsync -a <dir> :<dir>` (same basename, no trailing slash) -> nests to `<dir>/<dir>/` (verified;
  `--mkpath` does not fix it).
Fix candidates: (A) trailing slash per manifest entry (manifest is sha256-verified, cannot edit) or
`--include-from`/per-line `rsync -a <path>/`; (B) trailing slash on both ends (`rsync -avzP -l "<path>/" ":<same path>/"`)
or destination = parent. STOP bundle: `reports/stop/B5-pre_step3_20261019-0831.md`.

## Step 4 (re-run) — verify (partial; stopped by Step 3 bugs)

- `git rev-parse HEAD` on Beijing -> `56ba0625fda83d441b8fdb0a97fcc699319172ee` (matches).
- `du -sh` on Beijing -> `TtBA` 1.6G (nested), `ttba_mini_flat` 612K (nested), `LGC_frozen` 259M
  (git clone only; ~855M untracked files absent).
- symlink verify path `/root/autodl-tmp/ttba_mini_flat/images` does not exist on Beijing (it is at
  `/root/autodl-tmp/ttba_mini_flat/ttba_mini_flat/images` due to Bug B), so the `-xtype l` count of `0` would be
  misleading.
- `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` -> `bash: line 1: /usr/bin/nvidia-smi: Permission denied` (raw).

## Step 3 (retry 2) — H-approved fixes (2026-10-19 09:15)

Fix A — built `/tmp/transfer_include_fixed.txt` (trailing slash appended only to directory entries; the frozen
manifest was NOT edited). All 29 manifest entries are directories, so every line got a trailing slash:

```
: > /tmp/transfer_include_fixed.txt
while IFS= read -r p; do
  if [ -d "/root/autodl-tmp/LGC_frozen/$p" ]; then printf '%s/\n' "$p"; else printf '%s\n' "$p"; fi
done < reports/raw/B5-pre/transfer_include.txt > /tmp/transfer_include_fixed.txt
```

`rsync -avzP --files-from=/tmp/transfer_include_fixed.txt -e "ssh -p 16126" \
  /root/autodl-tmp/LGC_frozen/ root@connect.bjb2.seetacloud.com:/root/autodl-tmp/LGC_frozen/`
-> completed (exit 0): `xfr#170, to-chk=0/260`, `sent 195,967,782 bytes`, `total size is 292,173,448`.

**STILL INCOMPLETE — new bug:** `--files-from` + trailing slash copies only each directory's immediate
children (subdirectories are created empty; nested files are skipped). Beijing vs Harbin after Fix A:

| path | Beijing | Harbin |
|---|---|---|
| Results | 40K (9 files) | 1.4G (5909 files) |
| models/imagenet | 0 (empty) | 98M (`Linf/Salman2020Do_R50.pt`, 2 levels deep) |
| models/places365 | 198M (OK) | 198M |
| splits/kat | 8.0K (OK) | 8.0K |
| generated/smoke_canary | 82M | 190M |
| reports/raw | 16M | 29M |

Fix B — externals loop NOT run: the trailing-slash loop is correct for the 6 directory entries, but entry 5
(`/root/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt`) is a regular file, so `"$p/"` fails
`Not a directory` and `mkdir -p` would create a spurious empty dir at that path (local repro, rsync 3.2.7).

**STOP (E-SPEC):** Fix A non-recursive; Fix B trailing slash invalid on the `.pt` file entry.
STOP bundle: `reports/stop/B5-pre_step3_retry_20261019-0915.md`.

## Step 3 (retry 3) — H-approved per-entry standalone-rsync loop

Ran the H-approved design verbatim (per-entry `mkdir -p` + `rsync`), first on `transfer_include.txt` (29 entries),
then `externals_manifest.txt` (7 entries). **The loop as given only transferred the FIRST entry of each manifest.**

Root cause (confirmed by controlled repro below): the `ssh` inside `while IFS= read -r p; do …; done < file`
inherits the loop's stdin (the manifest file) and consumes the remaining lines, so the loop sees EOF after its
first iteration.

Controlled repro (`/tmp/ssh_stdin_test.txt` = `line1\nline2\nline3`):

```
while IFS= read -r p; do ssh … "true"; echo "iter: $p"; done < /tmp/ssh_stdin_test.txt
# -> iter: line1            (only first line; ssh ate line2/line3)
while IFS= read -r p; do ssh … "true" < /dev/null; echo "iter: $p"; done < /tmp/ssh_stdin_test.txt
# -> iter: line1 / line2 / line3
```

Beijing `du -sh` after the first (buggy) run confirmed only entry 1 of each manifest landed (`Results/…/targeted`
= 4.0K, all ttba/CGBA/ARS-OPT Results = 0, `ttba_mini_flat` = 0, `places365_data`/`baselines/*` absent).

**Fix (deviation from H's exact wording, but identical design):** add `< /dev/null` to each `ssh` and each `rsync`
so they cannot consume the manifest. Loop logic otherwise unchanged.

Re-ran both loops with the fix, foreground, exit 0, one `total size is` line per entry (29 + 7 = 36):

```
# transfer_include (29 rsync runs, exit 0) — per-entry total size, in manifest order:
8,541,438  8,331,475  8,530,074  8,435,174  16,909,048  8,524,834  8,380,201  16,957,659
16,950,174 16,555,923 16,946,744 16,726,678 16,908,270 16,849,243 16,959,462 16,457,656
16,485,552 16,603,177 16,731,866 16,942,825 16,835,169 16,905,872 16,889,447 16,937,502
196,992,429 102,546,991 207,389,357 29,365,448 1,196
# externals (7 rsync runs, exit 0) — per-entry total size:
1,635,750,039 791,697 207,396,379 632,150,116 102,546,991 15,001,388 20,858,390
```

## Step 4 (retry 3) — thorough verification (all 36 entries)

`du -sh` Harbin vs Beijing, per entry — all 36 match within rounding (only `ttba_mini_flat` differs at the
`du -sh` display level: Harbin 608K vs Beijing 612K; see precise check below):

```
TI Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_non_targeted  8.3M = 8.3M
TI Results/b4e2_smoke_LGC_RC_EGC_Salman2020Do_R50_targeted      8.1M = 8.1M
TI Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_non_targeted 8.3M = 8.3M
TI Results/b4e2_smoke_LGC_RC_EGC_densenet161_places365_targeted 8.2M = 8.2M
TI Results/b4e2_smoke_LGC_RC_EGC_resnet50                      17M  = 17M
TI Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_non_targeted 8.3M = 8.3M
TI Results/b4e2_smoke_LGC_RC_EGC_resnet50_places365_targeted   8.1M = 8.1M
TI Results/b4e2_smoke_LGC_RC_EGC_vgg19                         17M  = 17M
TI Results/b4e2_smoke_LGC_RC_EGC_vit_b_32                      17M  = 17M
TI Results/b4e2_smoke_ttba_Salman2020Do_R50                    16M  = 16M
TI Results/b4e2_smoke_ttba_densenet161_places365               17M  = 17M
TI Results/b4e2_smoke_ttba_resnet50                            17M  = 17M
TI Results/b4e2_smoke_ttba_resnet50_places365                  17M  = 17M
TI Results/b4e2_smoke_ttba_vgg19                               17M  = 17M
TI Results/b4e2_smoke_ttba_vit_b_32                            17M  = 17M
TI Results/b4h_t3_smoke_ARS_OPT_resnet50                       16M  = 16M
TI Results/b4h_t3_smoke_ARS_OPT_vgg19                          16M  = 16M
TI Results/b4h_t3_smoke_ARS_OPT_vit_b_32                       17M  = 17M
TI Results/b4h_t3_smoke_CGBA_Salman2020Do_R50                  17M  = 17M
TI Results/b4h_t3_smoke_CGBA_densenet161_places365             17M  = 17M
TI Results/b4h_t3_smoke_CGBA_resnet50                          17M  = 17M
TI Results/b4h_t3_smoke_CGBA_resnet50_places365                17M  = 17M
TI Results/b4h_t3_smoke_CGBA_vgg19                             17M  = 17M
TI Results/b4h_t3_smoke_CGBA_vit_b_32                          17M  = 17M
TI generated/smoke_canary                                      190M = 190M
TI models/imagenet                                             98M  = 98M
TI models/places365                                            198M = 198M
TI reports/raw                                                 29M  = 29M
TI splits/kat                                                  8.0K = 8.0K
EX /root/autodl-tmp/TtBA                                       1.6G = 1.6G
EX /root/autodl-tmp/ttba_mini_flat                             608K vs 612K (see below)
EX /root/autodl-tmp/models/places365                           198M = 198M
EX /root/autodl-tmp/places365_data                             677M = 677M
EX /root/.cache/robustbench/.../Salman2020Do_R50.pt            98M  = 98M
EX /root/autodl-tmp/baselines/CGBA                              15M  = 15M
EX /root/autodl-tmp/baselines/ARS-OPT                           22M  = 22M
```

Precise checks on the two kinds of residual deltas:

- `ttba_mini_flat` `du -sh` 608K vs 612K: apparent bytes `du -sb` are **identical** (955713 = 955713); dirs
  3/3, symlinks 3923/3923, regular files 3/3. The 4K is filesystem block-allocation rounding, not content.
- `du -sb` byte diff across all 36 entries: 7 entries showed tiny deltas (4–56 bytes; TtBA 24 B, ARS-OPT 56 B,
  CGBA 4 B, four Results dirs 8–32 B). For each of the 7, the **regular-file byte totals are identical**
  (e.g. TtBA files 1,635,750,039 = 1,635,750,039; ARS-OPT 20,858,390 = 20,858,390; CGBA 15,001,388 = 15,001,388).
  These deltas are directory-entry metadata (filesystem-dependent), not missing/different files.

MISSING-file loop (prompt Step 4, `[ -f "$f" ]`): prints **29 `MISSING:` lines** — but only because every one of
the 29 `transfer_include.txt` entries is a **directory** and `-f` tests for regular files. This is a prompt
inconsistency, not missing content. Corrected existence check `[ -e "$f" ]` on Beijing → **no output** (all 29
present).

`find /root/autodl-tmp/ttba_mini_flat/images -xtype l | wc -l` on Beijing → **0** (no broken symlinks).

`nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` on Beijing → raw:
`bash: line 1: /usr/bin/nvidia-smi: Permission denied` (recorded; not fixed, per instructions).

`git rev-parse HEAD` on Beijing → `56ba0625fda83d441b8fdb0a97fcc699319172ee`; `git status --short` →
only `?? reports/raw/B5-pre/` (expected: untracked project files now on Beijing).

### Note on LGC_frozen total size (Harbin 2.0G vs Beijing 1017M)

The whole-tree `du -sh` differs, but this is **expected and not a transfer failure**. `Results/` is git-ignored
(`git ls-files Results/` = 0; `git check-ignore Results` matches) and contains 117 subdirectories on Harbin; the
sha256-verified `transfer_include.txt` (`~855M`) intentionally lists only the 24 current smoke subdirs
(`b4e2_smoke_*`, `b4h_t3_smoke_*`). The other ~93 subdirs (`b3_*`, `b4_smoke_*`, `b4b_*`, `b4c_*`, `b4d_t3_*`,
`b4e_smoke_*`, `b4e_t1_*`, `b4e2_t1/t4/t5/t6/g1a/f1_*`, `retired_b4b_egc_per_victim`, `_bitid_*`) plus untracked
session notes (`*.md`, `reports/stop/*`) are historical/intermediate and are outside the H-approved manifest. The
29 manifest entries sum to ~850M, matching the prompt's stated `~855M`.

```
RESULT
phase: B5-pre migration
head_beijing: 56ba0625fda83d441b8fdb0a97fcc699319172ee
externals:
  /root/autodl-tmp/TtBA -> 1,635,750,039 bytes
  /root/autodl-tmp/ttba_mini_flat -> 791,697 bytes
  /root/autodl-tmp/models/places365 -> 207,396,379 bytes
  /root/autodl-tmp/places365_data -> 632,150,116 bytes
  /root/.cache/robustbench/models/imagenet/Linf/Salman2020Do_R50.pt -> 102,546,991 bytes
  /root/autodl-tmp/baselines/CGBA -> 15,001,388 bytes
  /root/autodl-tmp/baselines/ARS-OPT -> 20,858,390 bytes
verify: transfer_include=0 missing (36/36 entries byte-verified; du -sh within rounding) symlinks=0 broken (3923/3923 match) gpu=Permission denied (/usr/bin/nvidia-smi)
deviations: 2 (H-approved loop ssh-consumed manifest stdin — fixed by adding </dev/null to ssh/rsync, controlled repro in log; prompt [ -f ] MISSING loop tests regular files but all 29 manifest entries are directories — corrected [ -e ] check is empty)
stop_bundles: none (this attempt; prior: reports/stop/B5-pre_step3_20261019-0831.md, reports/stop/B5-pre_step3_retry_20261019-0915.md)
verdict: READY-FOR-B1-PROBE
```
