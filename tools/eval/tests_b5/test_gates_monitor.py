#!/usr/bin/env python3
"""B5-fix tests (CPU only) — gates, monitor, and the c20cc349 byte-identity diff.

Run from the worktree root:
    CUDA_VISIBLE_DEVICES="" python tools/eval/tests_b5/test_gates_monitor.py

Covers (T3): the three G5 gate scenarios, the domain counter on reports/raw/B5/t0b_smoke.csv,
one b5_monitor.py pass over the B4e2 smoke roots, and the 52-command byte-identity diff (T1g).
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EVAL))
sys.path.insert(0, EVAL)
sys.path.insert(0, REPO)

import b5_gate_cell as G  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def cpu_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# -------------------------------------------------------------------------------------------------- gate tests
def test_g5_gates():
    print("[1] b5_gate_cell.py G5 scenarios")
    tmp = tempfile.mkdtemp(prefix="b5gates_")

    # (a) CGBA cell + TtBA partner smoke -> "tool check only" -> PENDING (never PASS)
    c = G.CELL_BY_NAME["main_CGBA_resnet50_NT"]
    r = G.gate_g5(c, os.path.join(REPO, "Results", "b4h_t3_smoke_CGBA_resnet50"),
                  os.path.join(REPO, "Results", "b4e2_smoke_ttba_resnet50"), tmp)
    check("CGBA G5 on smoke -> PENDING (tool check only, not PASS)",
          r["verdict"] == "PENDING" and r["code"] == "pending", f"got {r['verdict']}/{r.get('detail')}")

    # (b) E10 TtBA places365 -> G5 n/a (no published reference)
    c = G.CELL_BY_NAME["e10_TtBA_resnet50_places365_NT"]
    r = G.gate_g5(c, os.path.join(REPO, "Results", "b4e2_smoke_ttba_resnet50_places365"), None, tmp)
    check("E10 TtBA places365 -> G5 n/a (no published reference)",
          r["verdict"] == "N/A" and "no published reference" in r.get("detail", ""), f"got {r['verdict']}")

    # (c) missing TtBA root -> PENDING (never PASS)
    c = G.CELL_BY_NAME["main_CGBA_resnet50_NT"]
    r = G.gate_g5(c, os.path.join(REPO, "Results", "b4h_t3_smoke_CGBA_resnet50"),
                  os.path.join(tmp, "no_such_ttba"), tmp)
    check("missing TtBA root -> PENDING", r["verdict"] == "PENDING", f"got {r['verdict']}/{r.get('detail')}")


def test_domain_counter():
    print("[2] domain counter (count X by column name)")
    p = os.path.join(REPO, "reports", "raw", "B5", "t0b_smoke.csv")
    x = G.count_x(p)
    total = sum(1 for _ in open(p, encoding="utf-8")) - 1
    check("t0b_smoke.csv has 4 X rows", x == 4, f"X={x}")
    check("t0b_smoke.csv has 1152 data rows", total == 1152, f"rows={total}")


def test_monitor_pass():
    print("[3] one b5_monitor.py pass (CPU-only) over a B4e2 smoke root")
    tmp = tempfile.mkdtemp(prefix="b5mon_")
    log = os.path.join(tmp, "log")
    cmd = [sys.executable, os.path.join(EVAL, "b5_monitor.py"), "--once",
           "--cell", "main_TtBA_resnet50_NT",
           "--root", os.path.join(REPO, "Results", "b4e2_smoke_ttba_resnet50"),
           "--log_dir", log]
    p = subprocess.run(cmd, capture_output=True, text=True, env=cpu_env(), cwd=REPO)
    status_files = []
    if os.path.isdir(os.path.join(log, "monitor")):
        status_files = [f for f in os.listdir(os.path.join(log, "monitor")) if f.startswith("status_")]
    ok = p.returncode == 0 and len(status_files) == 1
    detail = ""
    if status_files:
        txt = open(os.path.join(log, "monitor", status_files[0]), encoding="utf-8").read()
        detail = "row present" if "main_TtBA_resnet50_NT" in txt else "row MISSING"
        ok = ok and "main_TtBA_resnet50_NT" in txt
    check("monitor pass writes one status table with the cell row", ok, detail)


# -------------------------------------------------------------------------------------------------- byte-identity diff
def _gen_new_commands():
    p = subprocess.run(["bash", os.path.join(REPO, "run_b5_parallel.sh"), "--dry_run"],
                       capture_output=True, text=True, cwd=REPO, env=dict(os.environ, NGPU="1"))
    new = {}
    cell = None
    for ln in p.stdout.splitlines():
        if ln.startswith("# cell: "):
            cell = ln[len("# cell: "):].strip()
        elif ln.startswith("python ") and cell is not None:
            new[cell] = ln.strip()
            cell = None
    return new


def _gen_old_commands():
    old = subprocess.run(["git", "-C", REPO, "show", "bb7fa0d:run_b5_parallel.sh"],
                         capture_output=True, text=True).stdout.splitlines()
    cells_lines = old[9:63]          # CELLS=( ... ) (1-based lines 10..63)
    case_lines = old[94:327]         # case patterns (1-based lines 95..327)
    script = [
        "#!/bin/bash",
        'CKPTS="100 200 500 1000 2000 2500 5000 10000"',
        "FLAT=/root/autodl-tmp/ttba_mini_flat",
        "TTBA_REPO=/root/autodl-tmp/TtBA",
        "CGBA_REPO=/root/autodl-tmp/baselines/CGBA",
        "ARS_REPO=/root/autodl-tmp/baselines/ARS-OPT",
        'python() { printf "python"; local a; for a in "$@"; do printf " %s" "$a"; done; printf "\\n"; }',
    ] + cells_lines + [
        'for CELL in "${CELLS[@]}"; do',
        '  OUT="Results/b5_${CELL}"',
        '  printf "%s\\t" "$CELL"',
        '  case "$CELL" in',
    ] + case_lines + [
        "  esac",
        "done",
    ]
    fd, path = tempfile.mkstemp(prefix="old_runner_", suffix=".sh")
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(script) + "\n")
    p = subprocess.run(["bash", path], capture_output=True, text=True)
    os.unlink(path)
    old_map = {}
    for ln in p.stdout.splitlines():
        if "\t" in ln:
            name, cmd = ln.split("\t", 1)
            old_map[name.strip()] = cmd.strip()
    return old_map


def test_command_identity():
    print("[4] 52 attack commands byte-identical to c20cc349 except E17 out_dir (T1g)")
    new = _gen_new_commands()
    old = _gen_old_commands()
    check("52 cells both old and new", len(old) == 52 and len(new) == 52, f"old={len(old)} new={len(new)}")
    if set(old) != set(new):
        check("same cell names", False, f"symdiff={set(old) ^ set(new)}")
        return
    diffs = []
    for name in sorted(new):
        if old[name] != new[name]:
            # the only allowed difference is the E17 --out_dir
            o, n = old[name], new[name]
            diffs.append((name, o, n))
    only_e17 = all(name.startswith("e17_") for name, _, _ in diffs)

    import re
    outdir_only = True
    for name, o, n in diffs:
        o_out = re.search(r"--out_dir (\S+)", o).group(1)
        n_out = re.search(r"--out_dir (\S+)", n).group(1)
        arch = name.replace("e17_CGBA_D8only_", "").replace("_NT", "")
        expect = f"Results/appendix_cgba_d8only_{arch}"
        if o.replace(o_out, n_out) != n:
            outdir_only = False
        if n_out != expect:
            outdir_only = False
    check("differing cells are exactly the 4 E17 cells", only_e17 and len(diffs) == 4, f"n={len(diffs)}")
    check("each E17 diff is only the --out_dir (H4 appendix dir)", outdir_only)


def main():
    test_g5_gates()
    test_domain_counter()
    test_monitor_pass()
    test_command_identity()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
