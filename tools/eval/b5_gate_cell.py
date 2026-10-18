#!/usr/bin/env python3
"""B5 per-cell gate (T1i).  CPU-safe except the domain gate (needs the study GPU, inherited from the caller).

Runs, records and never short-circuits the gates that apply to one B5 cell:

  * ``partial``  -- ``tools/eval/check_contract.py --stage partial`` on the cell root (L11); NOT for E17.
  * ``g5``       -- ``tools/eval/g5_check.py`` for the main-study (ImageNet) TtBA / CGBA / CGBA_H cells only
                    (contract D13, H3).  E10/E11 cells -> "G5 n/a (no published reference)"; E17 -> "n/a".
  * ``domain``   -- ``adapters/diag/domain_consistency.py`` on the cell root, then the CSV's ``class == "X"``
                    count (by column name).  X > 0 STOPS only ARS_OPT / CGBA / CGBA_H (D19, H1); for LGC_RC_EGC
                    and CGBA_D8only the X count is recorded and never stops.

Exit status: 1 if any *stopping* gate failed, else 0 (a PENDING G5 or a "n/a" is not a failure).  Every gate is
run and recorded even if an earlier one failed.  One line per cell is appended to ``<log_dir>/gates/summary.csv``.
On a stopping failure a skill §7 STOP bundle is written under ``reports/stop/`` (field 4 carries GPU + driver +
torch + cuDNN + backend flags + HEAD + contract sha256; field 5 carries the rows already completed).

Every probe inside the STOP bundle is guarded: on a no-GPU instance nvidia-smi may be missing or a zero-byte
stub (OSError / Exec format error), and the bundle must still be written.

    python tools/eval/b5_gate_cell.py main_CGBA_resnet50_NT \
        --root Results/b4h_t3_smoke_CGBA_resnet50 --ttba_root Results/b4e2_smoke_ttba_resnet50 --gate g5
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT_PATH = os.path.join(REPO_ROOT, "contract", "paper_contract.yaml")
DEFAULT_RESULTS_ROOT = "Results"
DEFAULT_LOG_DIR = os.path.join("reports", "raw", "B5a2")

MAIN_ARCHES = ("resnet50", "vgg19", "vit_b_32")
E10_ARCHES = ("resnet50_places365", "densenet161_places365")
E11_ARCH = "Salman2020Do_R50"
E17_ARCHES = ("resnet50", "vgg19", "vit_b_32", "Salman2020Do_R50")

# s/img from reports/B4e_preflight.md T5 (solo, RTX PRO 6000 / D12 for baselines; LGC_RC_EGC from B4/B4b).
# key: (method_family, arch) -> (NT s/img, T s/img).  CGBA and CGBA_H share the "CGBA" family row.
_MAIN_S = {
    "LGC_RC_EGC": {"resnet50": (22.9, 24.7), "vgg19": (24.8, 29.2), "vit_b_32": (21.5, 23.8)},
    "TtBA":       {"resnet50": (54.8, 54.1), "vgg19": (60.6, 62.9), "vit_b_32": (41.7, 40.6)},
    "CGBA":       {"resnet50": (23.2, 24.3), "vgg19": (27.1, 27.4), "vit_b_32": (23.1, 22.2)},
    "ARS_OPT":    {"resnet50": (32.6, 29.5), "vgg19": (35.3, 35.4), "vit_b_32": (27.8, 28.1)},
}
_EXTRA_S = {
    "LGC_RC_EGC": {"resnet50_places365": (21.7, 24.4), "densenet161_places365": (32.8, 34.5),
                   "Salman2020Do_R50": (22.1, 24.4)},
    "TtBA":       {"resnet50_places365": (48.8, 48.3), "densenet161_places365": (118.3, 118.0),
                   "Salman2020Do_R50": (48.0, 48.3)},
    "CGBA":       {"resnet50_places365": (22.5, 23.5), "densenet161_places365": (36.9, 38.6),
                   "Salman2020Do_R50": (21.8, 23.5)},
    # ARS_OPT E10/E11 absent from T5 -> ASSUMED (see s_per_img).
}

_MODE_IDX = {"non_targeted": 0, "targeted": 1}


@dataclass(frozen=True)
class Cell:
    name: str
    scope: str      # main | e10 | e11 | e17
    method: str     # LGC_RC_EGC | TtBA | CGBA | CGBA_H | ARS_OPT | CGBA_D8only
    arch: str
    mode: str       # non_targeted | targeted


def _family(method: str) -> str:
    return "CGBA" if method in ("CGBA", "CGBA_H", "CGBA_D8only") else method


def _build_cells():
    cells = []

    def add(name, scope, method, arch, mode):
        cells.append(Cell(name, scope, method, arch, mode))

    for arch in MAIN_ARCHES:
        add(f"main_LGC_RC_EGC_{arch}_NT", "main", "LGC_RC_EGC", arch, "non_targeted")
        add(f"main_LGC_RC_EGC_{arch}_T", "main", "LGC_RC_EGC", arch, "targeted")
        add(f"main_TtBA_{arch}_NT", "main", "TtBA", arch, "non_targeted")
        add(f"main_TtBA_{arch}_T", "main", "TtBA", arch, "targeted")
        add(f"main_CGBA_{arch}_NT", "main", "CGBA", arch, "non_targeted")
        add(f"main_CGBA_H_{arch}_T", "main", "CGBA_H", arch, "targeted")
        add(f"main_ARS_OPT_{arch}_NT", "main", "ARS_OPT", arch, "non_targeted")
        add(f"main_ARS_OPT_{arch}_T", "main", "ARS_OPT", arch, "targeted")
    for arch in E10_ARCHES:
        add(f"e10_LGC_RC_EGC_{arch}_NT", "e10", "LGC_RC_EGC", arch, "non_targeted")
        add(f"e10_LGC_RC_EGC_{arch}_T", "e10", "LGC_RC_EGC", arch, "targeted")
        add(f"e10_TtBA_{arch}_NT", "e10", "TtBA", arch, "non_targeted")
        add(f"e10_TtBA_{arch}_T", "e10", "TtBA", arch, "targeted")
        add(f"e10_CGBA_{arch}_NT", "e10", "CGBA", arch, "non_targeted")
        add(f"e10_CGBA_H_{arch}_T", "e10", "CGBA_H", arch, "targeted")
        add(f"e10_ARS_OPT_{arch}_NT", "e10", "ARS_OPT", arch, "non_targeted")
        add(f"e10_ARS_OPT_{arch}_T", "e10", "ARS_OPT", arch, "targeted")
    for mode, suffix in (("non_targeted", "NT"), ("targeted", "T")):
        add(f"e11_LGC_RC_EGC_{E11_ARCH}_{suffix}", "e11", "LGC_RC_EGC", E11_ARCH, mode)
        add(f"e11_TtBA_{E11_ARCH}_{suffix}", "e11", "TtBA", E11_ARCH, mode)
        if mode == "non_targeted":
            add(f"e11_CGBA_{E11_ARCH}_{suffix}", "e11", "CGBA", E11_ARCH, mode)
        else:
            add(f"e11_CGBA_H_{E11_ARCH}_{suffix}", "e11", "CGBA_H", E11_ARCH, mode)
        add(f"e11_ARS_OPT_{E11_ARCH}_{suffix}", "e11", "ARS_OPT", E11_ARCH, mode)
    for arch in E17_ARCHES:
        add(f"e17_CGBA_D8only_{arch}_NT", "e17", "CGBA_D8only", arch, "non_targeted")
    return cells


CELLS = _build_cells()
CELL_BY_NAME = {c.name: c for c in CELLS}


# -------------------------------------------------------------------------------------------------- cost model
def s_per_img(method: str, arch: str, mode: str):
    """Return (s/img float, source label).  ARS_OPT E10/E11 use the resnet50 value (densenet161 x1.6), ASSUMED."""
    fam = _family(method)
    row = _MAIN_S.get(fam, {}).get(arch) or _EXTRA_S.get(fam, {}).get(arch)
    if row is not None:
        return row[_MODE_IDX[mode]], "T5 (reports/B4e_preflight.md)"
    # ARS_OPT on E10/E11 -- absent from T5.
    base = _MAIN_S["ARS_OPT"]["resnet50"][_MODE_IDX[mode]]
    if arch == "densenet161_places365":
        return round(base * 1.6, 4), "ASSUMED (resnet50 value x 1.6)"
    return base, "ASSUMED (resnet50 value)"


def n_images(cell: Cell) -> int:
    return 30 if cell.scope == "e17" else 200


def expected_hours(cell: Cell):
    s, src = s_per_img(cell.method, cell.arch, cell.mode)
    return round(n_images(cell) * s / 3600.0, 4), s, src


def ordered_queue():
    """LPT order: longest expected cell first, E17 last."""
    non_e17 = [c for c in CELLS if c.scope != "e17"]
    e17 = [c for c in CELLS if c.scope == "e17"]
    non_e17.sort(key=lambda c: expected_hours(c)[0], reverse=True)
    return non_e17 + e17


def outdir(cell: Cell, results_root: str = DEFAULT_RESULTS_ROOT) -> str:
    if cell.scope == "e17":
        return os.path.join(results_root, f"appendix_cgba_d8only_{cell.arch}")
    return os.path.join(results_root, f"b5_{cell.name}")


def ttba_partner_cell(cell: Cell):
    """The main-study TtBA cell of the same victim and mode (used to pair CGBA / CGBA_H)."""
    if cell.scope != "main" or cell.method not in ("CGBA", "CGBA_H"):
        return None
    suffix = "NT" if cell.mode == "non_targeted" else "T"
    return CELL_BY_NAME[f"main_TtBA_{cell.arch}_{suffix}"]


# -------------------------------------------------------------------------------------------------- CSV helpers
def count_x(path: str) -> int:
    """Number of rows whose column named ``class`` equals ``"X"`` (domain-consistency CSV)."""
    if not os.path.exists(path):
        return -1
    n = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("class") == "X":
                n += 1
    return n


def error_rows(root: str):
    """Error rows in every runs_*.csv under the cell root, read with the csv module (never grep)."""
    out = []
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            if not fn.startswith("runs") or not fn.endswith(".csv"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("status") == "error":
                        out.append((p, r))
    return out


def count_rows(root: str) -> int:
    """Number of data rows across every runs_*.csv under the cell root (skill §7 field 5)."""
    n = 0
    for dirpath, _dirs, fnames in os.walk(root):
        for fn in fnames:
            if fn.startswith("runs") and fn.endswith(".csv"):
                with open(os.path.join(dirpath, fn), newline="", encoding="utf-8") as fh:
                    n += sum(1 for _ in csv.DictReader(fh))
    return n


# -------------------------------------------------------------------------------------------------- STOP bundle
def _env_block() -> str:
    """Skill §7 field 4: GPU name, driver, torch, backend flags, HEAD, contract sha256.

    Every external probe is guarded: on a no-GPU instance nvidia-smi may be missing or a zero-byte stub
    (``OSError: Exec format error``); the STOP bundle must still be written."""
    try:
        import torch
        torch_v = torch.__version__
        cudnn = torch.backends.cudnn.version()
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
        flags = (f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
                 f"cudnn_tf32={torch.backends.cudnn.allow_tf32} "
                 f"det={torch.backends.cudnn.deterministic} bench={torch.backends.cudnn.benchmark}")
    except Exception as e:  # noqa: BLE001
        torch_v, cudnn, gpu, flags = "?", "?", "?", f"({e})"
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        driver = drv[0] if drv else "?"
    except Exception:  # noqa: BLE001  (no GPU / 0-byte nvidia-smi stub -> OSError Exec format error)
        driver = "?"
    try:
        head = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        head = "?"
    try:
        import hashlib
        with open(CONTRACT_PATH, "rb") as fh:
            contract_sha = hashlib.sha256(fh.read()).hexdigest()
    except Exception:  # noqa: BLE001
        contract_sha = "?"
    return (f"GPU {gpu}; driver {driver}; torch {torch_v}; cuDNN {cudnn}; {flags}; "
            f"HEAD {head}; contract {contract_sha}")


def write_stop_bundle(cell: str, code: str, signature: str, evidence: str, scope: str,
                      candidates: str = "not diagnosed (no fix applied)", proposed_lesson: str = "none") -> str:
    ts = time.strftime("%Y%m%d-%H%M")
    d = os.path.join(REPO_ROOT, "reports", "stop")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"B5a2_{cell}_{ts}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# STOP {cell}\n\n"
                 f"1. code: {code}\n"
                 f"2. signature: {signature}\n"
                 f"3. evidence: {evidence}\n"
                 f"4. env: {_env_block()}\n"
                 f"5. scope: {scope}\n"
                 f"6. candidates: {candidates}\n"
                 f"7. proposed lesson: {proposed_lesson}\n")
    return path


# -------------------------------------------------------------------------------------------------- gates
def gate_partial(cell: Cell, root: str, gates_dir: str):
    if cell.scope == "e17":
        return {"gate": "partial", "verdict": "SKIP", "code": "n/a", "detail": "E17 outside checker roots"}
    os.makedirs(gates_dir, exist_ok=True)
    report = os.path.join(gates_dir, f"{cell.name}_partial.md")
    status = os.path.join(gates_dir, f"{cell.name}_status.md")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "tools", "eval", "check_contract.py"),
           "--contract", CONTRACT_PATH, "--stage", "partial", "--roots", root,
           "--report", report, "--status_out", status]
    p = subprocess.run(cmd, capture_output=True, text=True)
    verdict = "PASS" if p.returncode == 0 else "FAIL"
    return {"gate": "partial", "verdict": verdict, "code": "E-OTHER" if verdict == "FAIL" else "ok",
            "detail": f"exit={p.returncode}", "out": (p.stdout + p.stderr).strip()}


def _g5_applies(cell: Cell) -> bool:
    return cell.scope == "main" and cell.method in ("TtBA", "CGBA", "CGBA_H")


def gate_g5(cell: Cell, root: str, ttba_root: str, gates_dir: str):
    if cell.scope == "e17":
        return {"gate": "g5", "verdict": "N/A", "code": "n/a", "detail": "E17 outside G5 (D13)"}
    if cell.method not in ("TtBA", "CGBA", "CGBA_H"):
        return {"gate": "g5", "verdict": "N/A", "code": "n/a", "detail": "not a G5 method (D13)"}
    if not _g5_applies(cell):
        return {"gate": "g5", "verdict": "N/A", "code": "n/a",
                "detail": "G5 n/a (no published reference)"}
    os.makedirs(gates_dir, exist_ok=True)
    g5 = os.path.join(REPO_ROOT, "tools", "eval", "g5_check.py")
    if cell.method == "TtBA":
        ttba_root = ttba_root if ttba_root else root
        if not ttba_root or not os.path.isdir(ttba_root):
            return {"gate": "g5", "verdict": "PENDING", "code": "pending",
                    "detail": "missing TtBA root"}
        cmd = [sys.executable, g5, "--arch", cell.arch, "--mode", cell.mode, "--ttba_root", ttba_root]
    else:
        if not ttba_root or not os.path.isdir(ttba_root):
            return {"gate": "g5", "verdict": "PENDING", "code": "pending",
                    "detail": "missing TtBA root (partner still running)"}
        cmd = [sys.executable, g5, "--arch", cell.arch, "--mode", cell.mode,
               "--ttba_root", ttba_root, "--cgba_root", root]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout + "\n" + p.stderr).strip()
    # surface the raw output so the caller (and tests) can see the verdict lines
    sys.stdout.write(out + "\n")
    flag_line = next((ln for ln in out.splitlines() if ln.startswith("FLAG")), "")
    if p.returncode == 1:
        return {"gate": "g5", "verdict": "FAIL", "code": "E-G5",
                "detail": "G5 FAIL (CI outside [0.5,2] at a budget)", "out": out}
    if p.returncode == 2:
        return {"gate": "g5", "verdict": "FLAG", "code": "E-OTHER",
                "detail": f"G5 FLAG: {flag_line}", "out": out}
    if "VERDICT: PASS" in out:
        return {"gate": "g5", "verdict": "PASS", "code": "ok", "detail": "VERDICT: PASS", "out": out}
    return {"gate": "g5", "verdict": "PENDING", "code": "pending",
            "detail": "tool check only (n < 30 common images) -> PENDING", "out": out}


def gate_domain(cell: Cell, root: str, gates_dir: str):
    if cell.method not in ("ARS_OPT", "CGBA", "CGBA_H", "LGC_RC_EGC", "CGBA_D8only"):
        return {"gate": "domain", "verdict": "SKIP", "code": "n/a", "detail": "TtBA not in the domain gate"}
    os.makedirs(gates_dir, exist_ok=True)
    csv_out = os.path.join(gates_dir, f"{cell.name}_domain.csv")
    log_out = os.path.join(gates_dir, f"{cell.name}_domain.log")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "adapters", "diag", "domain_consistency.py"),
           "--roots", root, "--out", csv_out]
    with open(log_out, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    x = count_x(csv_out)
    stop = cell.method in ("ARS_OPT", "CGBA", "CGBA_H")
    if x < 0:
        return {"gate": "domain", "verdict": "FAIL", "code": "E-INFRA",
                "detail": f"domain_consistency did not produce a CSV (exit={p.returncode})", "x": -1}
    if stop and x > 0:
        return {"gate": "domain", "verdict": "FAIL", "code": "E-OTHER",
                "detail": f"X={x} > 0 (stop rule for {cell.method})", "x": x}
    verdict = "PASS" if x == 0 else "INFO"
    return {"gate": "domain", "verdict": verdict, "code": "ok",
            "detail": f"X={x}" + ("" if stop else " (reported, not a stop rule)"), "x": x}


# -------------------------------------------------------------------------------------------------- summary
def append_summary(log_dir: str, cell: Cell, results):
    d = os.path.join(log_dir, "gates")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "summary.csv")
    res = {r["gate"]: r for r in results}
    def v(g, key="verdict"):
        return res.get(g, {}).get(key, "")
    row = {
        "cell": cell.name, "scope": cell.scope, "method": cell.method,
        "arch": cell.arch, "mode": cell.mode,
        "partial": v("partial"), "g5": v("g5"), "g5_code": v("g5", "code"),
        "domain_x": v("domain", "x"), "domain": v("domain"),
        "stop": int(any(res.get(g, {}).get("verdict") in ("FAIL", "FLAG")
                        for g in ("partial", "g5", "domain"))),
    }
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    return path


# -------------------------------------------------------------------------------------------------- main
def run_gates(cell: Cell, root: str, ttba_root: str, log_dir: str, only: str) -> int:
    gates_dir = os.path.join(log_dir, "gates")
    results = []
    if only in ("all", "partial"):
        results.append(gate_partial(cell, root, gates_dir))
    if only in ("all", "g5"):
        results.append(gate_g5(cell, root, ttba_root, gates_dir))
    if only in ("all", "domain"):
        results.append(gate_domain(cell, root, gates_dir))

    stopping = [r for r in results if r.get("verdict") in ("FAIL", "FLAG")]
    for r in results:
        print(f"[gate] {r['gate']}: {r['verdict']} ({r.get('code')}) {r.get('detail')}")
    append_summary(log_dir, cell, results)

    for r in stopping:
        code = r.get("code") or "E-OTHER"
        sig = f"{r['gate']} {r['verdict']}: {r.get('detail')}"
        write_stop_bundle(cell.name, code, sig, sig,
                          f"cell {cell.name} ({cell.arch}/{cell.mode}/{cell.method}); "
                          f"{count_rows(root)} rows completed")
    return 1 if stopping else 0


def _print_queue(results_root: str) -> None:
    print("| # | cell | scope | method | arch | mode | n | s/img | source | expected h | outdir |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(ordered_queue(), 1):
        h, s, src = expected_hours(c)
        print(f"| {i} | {c.name} | {c.scope} | {c.method} | {c.arch} | {c.mode} | {n_images(c)} "
              f"| {s} | {src} | {h:.4f} | {outdir(c, results_root)} |")


def _print_order(results_root: str) -> None:
    for c in ordered_queue():
        print(f"{c.name}\t{outdir(c, results_root)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", nargs="?")
    ap.add_argument("--root", default=None, help="override the cell output root (tests)")
    ap.add_argument("--ttba_root", default=None, help="override the TtBA partner root (tests)")
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--gate", choices=["all", "partial", "g5", "domain"], default="all")
    ap.add_argument("--print_queue", action="store_true")
    ap.add_argument("--print_order", action="store_true")
    ap.add_argument("--print_errors", action="store_true")
    a = ap.parse_args(argv)

    if a.print_queue:
        _print_queue(a.results_root)
        return 0
    if a.print_order:
        _print_order(a.results_root)
        return 0
    if a.print_errors:
        root = a.root
        if not root and a.cell:
            root = outdir(CELL_BY_NAME.get(a.cell), a.results_root) if CELL_BY_NAME.get(a.cell) else None
        if not root:
            ap.error("--print_errors needs --root (or a known --cell)")
        errs = error_rows(root)
        print(f"error rows under {root}: {len(errs)}")
        for p, r in errs:
            print(f"  {p}: image={r.get('image')} status={r.get('status')} error={r.get('error')}")
        return 1 if errs else 0
    if not a.cell:
        ap.error("a cell name is required (or use --print_queue / --print_order)")
    cell = CELL_BY_NAME.get(a.cell)
    if cell is None:
        print(f"[!] unknown cell: {a.cell}", file=sys.stderr)
        return 2
    root = a.root if a.root else outdir(cell, a.results_root)
    ttba_root = a.ttba_root
    if ttba_root is None:
        partner = ttba_partner_cell(cell)
        ttba_root = outdir(partner, a.results_root) if partner else None
    return run_gates(cell, root, ttba_root, a.log_dir, a.gate)


if __name__ == "__main__":
    raise SystemExit(main())