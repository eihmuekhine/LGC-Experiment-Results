# B5a2 triage (read-only) — 2026-10-20 19:17

Rules honored: no GPU processes, no git writes, no writes under Results/, reports/stop/, reports/raw/B5a2/;
edit only this new file under reports/raw/B5a2_triage/. Python commands prefixed CUDA_VISIBLE_DEVICES="".
adapters/diag/domain_consistency.py was NOT executed (victim use prohibited in this phase); read only.

## T0 — environment

```
date:
Tue Oct 20 19:16:13 CST 2026

pgrep -af "run_b5_parallel|run_benchmark_33b|run_ttba_v2|run_cgba|run_arsopt" | cut -c1-120:
13449 bash run_b5_parallel.sh
81138 python adapters/run_arsopt.py --arch vit_b_32 --mode non_targeted --split_csv splits/imagenet/ttba_github/vit_b_32
82084 python adapters/run_cgba.py --arch vgg19 --mode targeted --split_csv splits/imagenet/ttba_github/vgg19_targeted.cs
82170 python adapters/run_cgba.py --arch vgg19 --mode non_targeted --split_csv splits/imagenet/ttba_github/vgg19_non_tar
83250 python run_benchmark_33b.py --dataset imagenet --mode non_targeted --archs vgg19 --methods LGC_RC_EGC --data_dir /
83430 python run_benchmark_33b.py --dataset imagenet --mode targeted --archs resnet50 --methods LGC_RC_EGC --data_dir /r
(plus the pgrep shell itself)

git rev-parse HEAD:
8cf80567009d7404d5a00f1edcfe5c4a09372f81

sha256sum contract/paper_contract.yaml:
ce101e334a787b78f26c0fef4c5480110b9e84d8dbe0981d3087be7c90e3747c  contract/paper_contract.yaml
```

## T1 — domain-gate outputs of the two stopped cells

Gates logs:

```
$ cat reports/raw/B5a2/e11_ARS_OPT_Salman2020Do_R50_T_gates.log
[gate] partial: PASS (ok) exit=0
[gate] g5: N/A (n/a) not a G5 method (D13)
[gate] domain: FAIL (E-OTHER) X=1 > 0 (stop rule for ARS_OPT)

$ cat reports/raw/B5a2/main_ARS_OPT_vit_b_32_T_gates.log
[gate] partial: PASS (ok) exit=0
[gate] g5: N/A (n/a) not a G5 method (D13)
[gate] domain: FAIL (E-OTHER) X=2 > 0 (stop rule for ARS_OPT)
```

Domain artifacts found (find reports/raw -newer reports/raw/B5a2/prompt.txt -name "*domain*") include, for the two stopped cells:

```
reports/raw/B5a2/gates/e11_ARS_OPT_Salman2020Do_R50_T_domain.log
reports/raw/B5a2/gates/e11_ARS_OPT_Salman2020Do_R50_T_domain.csv
reports/raw/B5a2/gates/main_ARS_OPT_vit_b_32_T_domain.log
reports/raw/B5a2/gates/main_ARS_OPT_vit_b_32_T_domain.csv
```

tools/eval/b5_gate_cell.py — where the domain gate calls domain_consistency and where the CSV is written:

```python
tools/eval/b5_gate_cell.py:325:    csv_out = os.path.join(gates_dir, f"{cell.name}_domain.csv")
tools/eval/b5_gate_cell.py:326:    log_out = os.path.join(gates_dir, f"{cell.name}_domain.log")
tools/eval/b5_gate_cell.py:327:    cmd = [sys.executable, os.path.join(REPO_ROOT, "adapters", "diag", "domain_consistency.py"),
tools/eval/b5_gate_cell.py:328:           "--roots", root, "--out", csv_out]
tools/eval/b5_gate_cell.py:329:    with open(log_out, "w", encoding="utf-8") as lf:
tools/eval/b5_gate_cell.py:330:        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
tools/eval/b5_gate_cell.py:331:    x = count_x(csv_out)
tools/eval/b5_gate_cell.py:332:    stop = cell.method in ("ARS_OPT", "CGBA", "CGBA_H")
tools/eval/b5_gate_cell.py:333:    if x < 0:
tools/eval/b5_gate_cell.py:334:        return {"gate": "domain", "verdict": "FAIL", "code": "E-OTHER",
tools/eval/b5_gate_cell.py:335:                "detail": f"domain_consistency did not produce a CSV (exit={p.returncode})", "x": -1}
tools/eval/b5_gate_cell.py:336:    if stop and x > 0:
tools/eval/b5_gate_cell.py:337:        return {"gate": "domain", "verdict": "FAIL", "code": "E-OTHER",
tools/eval/b5_gate_cell.py:338:                "detail": f"X={x} > 0 (stop rule for {cell.method})", "x": x}
```

The CSV path is therefore `reports/raw/B5a2/gates/<cell_name>_domain.csv` (passed as `--out` at line 328, computed at line 325).

## T2 — class X rows in the domain CSVs (csv.DictReader by column name, no recomputation)

```
FILE: reports/raw/B5a2/gates/e11_ARS_OPT_Salman2020Do_R50_T_domain.csv
COLUMNS: ['method', 'arch', 'mode', 'image', 'budget', 'source_label', 'target_label', 'class', 'm_s1', 'm_s9', 'm_s6', 'adv_s1', 'adv_s9', 'adv_s6', 'adv_uint8']
X rows: 1
{'method': 'ARS_OPT', 'arch': 'Salman2020Do_R50', 'mode': 'targeted', 'image': 'ILSVRC2012_val_00048242', 'budget': '10000', 'source_label': '8', 'target_label': '980', 'class': 'X', 'm_s1': '0.0004200935363769531', 'm_s9': '0.00063323974609375', 'm_s6': '0.005953788757324219', 'adv_s1': '0', 'adv_s9': '0', 'adv_s6': '0', 'adv_uint8': '0'}
```

```
FILE: reports/raw/B5a2/gates/main_ARS_OPT_vit_b_32_T_domain.csv
COLUMNS: ['method', 'arch', 'mode', 'image', 'budget', 'source_label', 'target_label', 'class', 'm_s1', 'm_s9', 'm_s6', 'adv_s1', 'adv_s9', 'adv_s6', 'adv_uint8']
X rows: 2
{'method': 'ARS_OPT', 'arch': 'vit_b_32', 'mode': 'targeted', 'image': 'ILSVRC2012_val_00038119', 'budget': '10000', 'source_label': '126', 'target_label': '121', 'class': 'X', 'm_s1': '8.96453857421875e-05', 'm_s9': '0.00013303756713867188', 'm_s6': '0.01583242416381836', 'adv_s1': '0', 'adv_s9': '0', 'adv_s6': '0', 'adv_uint8': '0'}
{'method': 'ARS_OPT', 'arch': 'vit_b_32', 'mode': 'targeted', 'image': 'ILSVRC2012_val_00011074', 'budget': '10000', 'source_label': '322', 'target_label': '555', 'class': 'X', 'm_s1': '1.52587890625e-05', 'm_s9': '0.0005350112915039062', 'm_s6': '0.0664224624633789', 'adv_s1': '0', 'adv_s9': '0', 'adv_s6': '0', 'adv_uint8': '0'}
```

## T3 — finished ARS_OPT / CGBA / CGBA_H cells (present in parallel.log with Exitval)

parallel.log lines (Exitval column shown): 26 cells with Exitval, of which the following are ARS_OPT/CGBA/CGBA_H.
Rows checked and X count are read from the existing gate outputs — rows checked = number of data rows in
`reports/raw/B5a2/gates/<cell>_domain.csv`; X = count of rows with `class == 'X'` in that CSV (consistent with the
X= value printed in `<cell>_gates.log`).

| seq | cell | Exitval (parallel.log) | rows checked (domain CSV) | X (domain CSV / gates.log) |
|-----|------|------------------------|---------------------------|----------------------------|
| 7  | e10_ARS_OPT_densenet161_places365_NT  | 0 | 1600 | 0 |
| 12 | e10_ARS_OPT_densenet161_places365_T   | 0 | 1600 | 0 |
| 15 | e10_CGBA_H_densenet161_places365_T    | 0 | 1600 | 0 |
| 16 | e10_CGBA_densenet161_places365_NT     | 0 | 1599 | 0 |
| 17 | main_ARS_OPT_vgg19_T                  | 0 | 1600 | 0 |
| 18 | main_ARS_OPT_vgg19_NT                 | 0 | 1600 | 0 |
| 21 | main_ARS_OPT_resnet50_NT              | 0 | 1600 | 0 |
| 23 | e10_ARS_OPT_resnet50_places365_NT     | 0 | 1600 | 0 |
| 24 | e11_ARS_OPT_Salman2020Do_R50_NT       | 0 | 1600 | 0 |
| 25 | main_ARS_OPT_resnet50_T               | 0 | 1600 | 0 |
| 26 | e11_ARS_OPT_Salman2020Do_R50_T        | 1 | 1600 | 1 (domain FAIL, stop rule) |
| 27 | e10_ARS_OPT_resnet50_places365_T      | 0 | 1600 | 0 |
| 29 | main_ARS_OPT_vit_b_32_T               | 1 | 1600 | 2 (domain FAIL, stop rule) |

Verbatim gates.log lines for each of the above:

```
e10_ARS_OPT_densenet161_places365_NT: [gate] domain: PASS (ok) X=0
e10_ARS_OPT_densenet161_places365_T:  [gate] domain: PASS (ok) X=0
e10_CGBA_densenet161_places365_NT:    [gate] domain: PASS (ok) X=0
e10_CGBA_H_densenet161_places365_T:   [gate] domain: PASS (ok) X=0
main_ARS_OPT_vgg19_T:                 [gate] domain: PASS (ok) X=0
main_ARS_OPT_vgg19_NT:                [gate] domain: PASS (ok) X=0
main_ARS_OPT_resnet50_NT:             [gate] domain: PASS (ok) X=0
e10_ARS_OPT_resnet50_places365_NT:    [gate] domain: PASS (ok) X=0
e11_ARS_OPT_Salman2020Do_R50_NT:      [gate] domain: PASS (ok) X=0
main_ARS_OPT_resnet50_T:              [gate] domain: PASS (ok) X=0
e11_ARS_OPT_Salman2020Do_R50_T:       [gate] domain: FAIL (E-OTHER) X=1 > 0 (stop rule for ARS_OPT)
e10_ARS_OPT_resnet50_places365_T:     [gate] domain: PASS (ok) X=0
main_ARS_OPT_vit_b_32_T:              [gate] domain: FAIL (E-OTHER) X=2 > 0 (stop rule for ARS_OPT)
```

(All gates logs also contain `[gate] partial: PASS (ok) exit=0` and a `[gate] g5: N/A (n/a)` line per cell.)

Still running at T0 (no Exitval in parallel.log, not in this table): ARS_OPT vit_b_32 NT (pid 81138),
CGBA vgg19 T (pid 82084), CGBA vgg19 NT (pid 82170).

## T4 — quotes from adapters/diag/domain_consistency.py (read only)

Class definitions (module docstring):

```
adapters/diag/domain_consistency.py:14:  * class: ``C`` if adversarial at s=1; ``Q9`` if not at s=1 but adversarial at s=1+2^-9; ``Q6`` if the first
adapters/diag/domain_consistency.py:15:    adversarial point is s=1+2^-6; ``X`` otherwise.
```

Float16 bound (module docstring):

```
adapters/diag/domain_consistency.py:17: Q9/Q6 are float16-storage effects (||fl16(d) - d|| <= 2^-11 ||d||), not defects.  After the B4h-T3 D19 fix,
```

Implementation:

```
adapters/diag/domain_consistency.py:95: def classify(adv):
adapters/diag/domain_consistency.py:96:     adv_s1, adv_s9, adv_s6 = adv
adapters/diag/domain_consistency.py:97:     if adv_s1:
adapters/diag/domain_consistency.py:98:         return "C"
adapters/diag/domain_consistency.py:99:     if adv_s9:
adapters/diag/domain_consistency.py:100:         return "Q9"
adapters/diag/domain_consistency.py:101:     if adv_s6:
adapters/diag/domain_consistency.py:102:         return "Q6"
adapters/diag/domain_consistency.py:103:     return "X"
```

Scale set used by the measurement:

```
adapters/diag/domain_consistency.py:43: SCALES = (("1", 1.0), ("1+2^-9", 1.0 + 2.0 ** -9), ("1+2^-6", 1.0 + 2.0 ** -6))
```

No interpretation of the rows is made here (T4 requires quoting only).
