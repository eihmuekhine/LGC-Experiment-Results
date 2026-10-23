# E9 sources (B5a2-finish part B, B4)

Status: **NOT FOUND**.

The confirm_37 result CSVs required by `contract/paper_contract.yaml` paper element E9_ablation_table
(`source: "confirm_37 result CSVs (path + sha256 recorded in reports/E9_sources.md)"`, D4: ResNet-50 V1,
seed 2718, n = 100, metrics pix_l2 / ssim / lpips-vgg; rows EGC, LGC_RC, EGCI, HT1000) were not found in the
repository, under `Results/`, or anywhere on this host. `reports/E9_sources.md` did not exist before this task.
E9b (operator substitution) is CONDITIONAL and is likewise CUT (its source CSV was not found).

Nothing was rebuilt or re-run for E9.

## Search commands and their output

```
$ find . -maxdepth 4 \( -iname '*confirm_37*' -o -iname '*seed2718*' \)
./confirm_37.py

$ find . -maxdepth 6 \( -iname '*confirm_37*' -o -iname '*seed2718*' \)
./confirm_37.py

$ find . -maxdepth 4 \( -iname '*confirm37*' -o -iname '*confirm_37*' -o -iname '*seed2718*' -o -iname '*sel37*' \)
./confirm_37.py

$ ls -d Results/*confirm* Results/*37*
(no output)

$ ls -d Results/* | grep -iE 'confirm|37|2718'
(no output)

$ find . -maxdepth 5 -iname '*seed2718*'
(no output)

$ find . -iname '*2718*' | grep -v node_modules
(no output)

$ ls /root/autodl-tmp/Latent-Geometric-Chords/
ls: cannot access '/root/autodl-tmp/Latent-Geometric-Chords/': No such file or directory

$ find /root/autodl-tmp -maxdepth 5 \( -iname '*confirm37*' -o -iname '*confirm_37*' -o -iname '*seed2718*' \)
/root/autodl-tmp/LGC_frozen/confirm_37.py

$ find / -maxdepth 6 \( -iname 'confirm_37.csv' -o -iname 'confirm37*' -o -iname '*confirm_37.txt' \)
(no output)
```

The only match is the committed analysis script `confirm_37.py` (which reads `<res>/<mode>/per_checkpoint*.csv`
and writes `<res>/confirm_37.csv`); no `<res>` result directory (e.g. `Results/confirm37_resnet50`) and no
`confirm_37.csv` exists.

## Consequence for the paper

- E9_ablation_table is REQUIRED but its inputs are NOT FOUND; it is marked INCOMPLETE (to be reported to H;
  the ablation rows cannot be regenerated and are never re-run).
- E9b_operator_substitution is CONDITIONAL and is CUT (source CSV not found).
