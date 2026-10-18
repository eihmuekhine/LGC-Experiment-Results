#!/usr/bin/env python3
"""B5 progress monitor (T2).  CPU-only and read-only on RESULTS_ROOT.

Every ``--interval`` minutes (default 10) it scans every B5 cell (the same cell list and output-directory
mapping as ``run_b5_parallel.sh`` / ``b5_gate_cell.py``) and, for each cell that has started (its root
exists), reports:

  * rows by ``status`` column, error rows, ``nonfinite_queries`` sum, median ``wall_time_s``;
  * the partial checker (``check_contract.py --stage partial``) each time a cell gains 10 completed images,
    reported under ``<log_dir>/monitor/``;
  * G5 once a main-study TtBA / CGBA / CGBA_H cell reaches 30 completed images (a CGBA / CGBA_H cell needs 30
    completed by BOTH it and its TtBA partner), evaluated on a *copy* of only the files ``g5_check.py`` reads
    (``<root>/<mode>/per_checkpoint_<stem>.csv``, see ``g5_check.py._load_ck``) restricted to the first 30
    images in ``runs_*.csv`` row order, under ``<log_dir>/monitor/g5early/``;
  * re-evaluation of every G5 PENDING cell (from ``<log_dir>/gates/summary.csv``).

STOP bundles are written (skill §7) but the monitor never kills, signals or edits anything, and never writes
into RESULTS_ROOT:  G5 FAIL -> E-G5; an error row -> E-OTHER; ``nonfinite_queries > 0`` in a CGBA / CGBA_H
cell -> E-NAN.  One status table per pass: ``<log_dir>/monitor/status_<YYYYmmdd-HHMM>.md``.

    python tools/eval/b5_monitor.py [--interval 10] [--once] [--results_root Results]
                                   [--log_dir reports/raw/B5a2]
                                   [--cell main_TtBA_resnet50_NT --root Results/b4e2_smoke_ttba_resnet50
                                    --ttba_root Results/b4e2_smoke_ttba_resnet50]   # test override
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import b5_gate_cell as G  # noqa: E402  (cell list, outdir(), ttba_partner_cell(), STOP bundles)

# -------------------------------------------------------------------------------------------------- CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def subprocess_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


# -------------------------------------------------------------------------------------------------- CSV helpers
def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_runs(root, mode):
    rows = []
    for fn in sorted(glob.glob(os.path.join(root, mode, "runs*.csv"))):
        rows.extend(_read_csv(fn))
    return rows


def _num(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def completed_images(rows):
    """Distinct image stems whose status is ok / budget (an error row is not completed)."""
    stems = []
    for r in rows:
        if r.get("status") in ("ok", "budget") and r.get("image"):
            if r["image"] not in stems:
                stems.append(r["image"])
    return stems


def first_n_images(rows, n=30):
    """The first n distinct image stems in runs_*.csv row order."""
    out = []
    for r in rows:
        img = r.get("image")
        if img and img not in out:
            out.append(img)
        if len(out) >= n:
            break
    return out


def error_rows(rows):
    return [r for r in rows if r.get("status") == "error"]


def nonfinite_sum(rows):
    return sum(int(r.get("nonfinite_queries") or 0) for r in rows)


def median_wall(rows):
    vals = [_num(r.get("wall_time_s")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    if n % 2:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2.0


# -------------------------------------------------------------------------------------------------- partial
def run_partial(cell, root, log_dir, k):
    out_dir = os.path.join(log_dir, "monitor")
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, f"{cell.name}_partial_{k}.md")
    status = os.path.join(out_dir, f"{cell.name}_partial_{k}_status.md")
    cmd = [sys.executable, os.path.join(HERE, "check_contract.py"),
           "--contract", G.CONTRACT_PATH, "--stage", "partial", "--roots", root,
           "--report", report, "--status_out", status]
    p = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env())
    return p.returncode, report, (p.stdout + p.stderr).strip()


# -------------------------------------------------------------------------------------------------- early G5 copy
def _copy_first_n_per_checkpoint(src_root, mode, stem, image_set, dst_root):
    src = os.path.join(src_root, mode, f"per_checkpoint_{stem}.csv")
    if not os.path.exists(src):
        return None
    rows = [r for r in _read_csv(src) if r.get("image") in image_set]
    dst_dir = os.path.join(dst_root, mode)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"per_checkpoint_{stem}.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return dst


def run_early_g5(cell, root, ttba_root, log_dir):
    """Copy the per_checkpoint files g5_check reads, restricted to the first 30 images, and run g5_check."""
    base = os.path.join(log_dir, "monitor", "g5early", cell.name)
    if os.path.exists(base):
        shutil.rmtree(base)  # LOG_DIR, not RESULTS_ROOT
    os.makedirs(base)

    method_stem = "TtBA" if cell.method == "TtBA" else ("CGBA" if cell.method == "CGBA" else "CGBA_H")
    ttba_rows = _read_runs(ttba_root, cell.mode)
    ttba_imgs = set(first_n_images(ttba_rows, 30))
    _copy_first_n_per_checkpoint(ttba_root, cell.mode, "TtBA", ttba_imgs, base)

    if cell.method == "TtBA":
        cmd = [sys.executable, os.path.join(HERE, "g5_check.py"),
               "--arch", cell.arch, "--mode", cell.mode, "--ttba_root", base]
    else:
        own_rows = _read_runs(root, cell.mode)
        own_imgs = set(first_n_images(own_rows, 30))
        _copy_first_n_per_checkpoint(root, cell.mode, method_stem, own_imgs, base)
        cmd = [sys.executable, os.path.join(HERE, "g5_check.py"),
               "--arch", cell.arch, "--mode", cell.mode,
               "--ttba_root", base, "--cgba_root", base]

    p = subprocess.run(cmd, capture_output=True, text=True, env=subprocess_env())
    out = (p.stdout + "\n" + p.stderr).strip()
    return p.returncode, out


def g5_verdict(rc, out):
    if rc == 1:
        return "FAIL", "E-G5", "G5 FAIL (CI outside [0.5,2] at a budget)"
    if rc == 2:
        flag = next((ln for ln in out.splitlines() if ln.startswith("FLAG")), "")
        return "FLAG", "E-OTHER", f"G5 FLAG: {flag}"
    if "VERDICT: PASS" in out:
        return "PASS", "ok", "VERDICT: PASS"
    return "PENDING", "pending", "tool check only (n < 30 common images)"


# -------------------------------------------------------------------------------------------------- state
def _state_path(log_dir):
    d = os.path.join(log_dir, "monitor")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "state.json")


def load_state(log_dir):
    p = _state_path(log_dir)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"partial": {}, "g5": {}, "stops": []}


def save_state(log_dir, state):
    with open(_state_path(log_dir), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)


def read_pending_cells(log_dir):
    """G5 PENDING cells recorded by the launcher's gate summary."""
    pending = set()
    path = os.path.join(log_dir, "gates", "summary.csv")
    for r in _read_csv(path):
        if r.get("g5") == "PENDING":
            pending.add(r.get("cell"))
    return pending


# -------------------------------------------------------------------------------------------------- one pass
def one_pass(results_root, log_dir, cell_filter=None, root_override=None, ttba_override=None):
    os.makedirs(os.path.join(log_dir, "monitor"), exist_ok=True)
    state = load_state(log_dir)
    pending = read_pending_cells(log_dir)

    cells = [c for c in G.CELLS if cell_filter is None or c.name == cell_filter]
    table = []          # list of row dicts for the status markdown table
    stops = []          # (cell_name, code, signature, detail)
    g5_results = []     # (cell_name, verdict, code, detail, out)

    for cell in cells:
        root = root_override if (root_override and cell.name == cell_filter) else G.outdir(cell, results_root)
        if not os.path.isdir(root):
            continue
        rows = _read_runs(root, cell.mode)
        if not rows:
            continue

        status_counts = Counter(r.get("status", "") for r in rows)
        errs = error_rows(rows)
        nf = nonfinite_sum(rows)
        wall = median_wall(rows)
        done = completed_images(rows)
        row = {
            "cell": cell.name, "mode": cell.mode, "rows": len(rows),
            "status": ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
            "error": len(errs), "nonfinite": nf,
            "wall_time_s_median": "" if wall is None else round(wall, 2),
            "completed": len(done),
        }
        table.append(row)

        # --- STOP: error rows (any cell) ---
        if errs:
            sig = f"{len(errs)} error row(s) in {root}/{cell.mode}"
            stops.append((cell.name, "E-OTHER", sig, sig))

        # --- STOP: nonfinite > 0 in CGBA / CGBA_H ---
        if nf > 0 and cell.method in ("CGBA", "CGBA_H"):
            sig = f"nonfinite_queries={nf} in {root}/{cell.mode}"
            stops.append((cell.name, "E-NAN", sig, sig))

        # --- partial every 10 completed images ---
        if cell.scope != "e17":
            k = len(done) // 10
            if k > int(state["partial"].get(cell.name, 0)):
                rc, report, detail = run_partial(cell, root, log_dir, k)
                state["partial"][cell.name] = k
                row["partial"] = f"k={k} exit={rc}"
                if rc != 0:
                    stops.append((cell.name, "E-OTHER",
                                  f"partial checker exit {rc}", f"{detail}\n(report {report})"))
            else:
                row["partial"] = f"k={int(state['partial'].get(cell.name, 0))} (no new 10)"

        # --- early G5 + PENDING re-evaluation ---
        if G._g5_applies(cell):
            own_done = len(done)
            if cell.method == "TtBA":
                ttba_root = ttba_override if (ttba_override and cell.name == cell_filter) else root
                partner_done = own_done
            else:
                partner = G.ttba_partner_cell(cell)
                if partner is None:
                    continue
                ttba_root = (ttba_override if (ttba_override and cell.name == cell_filter)
                             else G.outdir(partner, results_root))
                partner_rows = _read_runs(ttba_root, cell.mode)
                partner_done = len(completed_images(partner_rows))
            enough = own_done >= 30 and partner_done >= 30

            needs = (cell.name in pending) or (state["g5"].get(cell.name) not in ("PASS", "FAIL", "FLAG"))
            if enough and needs and os.path.isdir(ttba_root):
                rc, out = run_early_g5(cell, root, ttba_root, log_dir)
                verdict, code, detail = g5_verdict(rc, out)
                state["g5"][cell.name] = verdict
                row["g5"] = verdict
                g5_results.append((cell.name, verdict, code, detail, out))
                if verdict == "FAIL":
                    stops.append((cell.name, "E-G5", detail, out))
                elif verdict == "FLAG":
                    stops.append((cell.name, "E-OTHER", detail, out))
            else:
                row["g5"] = state["g5"].get(cell.name, "PENDING" if enough else "waiting")

    save_state(log_dir, state)

    ts = time.strftime("%Y%m%d-%H%M")
    _write_status(log_dir, ts, table, g5_results, stops)
    _write_stops(stops, state.setdefault("stops", []))
    save_state(log_dir, state)

    # exit code: 1 if any stop fired (informative only; the monitor is never the killer)
    return 1 if stops else 0


def _write_status(log_dir, ts, table, g5_results, stops):
    out_dir = os.path.join(log_dir, "monitor")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"status_{ts}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# B5 monitor pass {ts}\n\n")
        fh.write(f"cells with data: {len(table)}; G5 runs: {len(g5_results)}; stops: {len(stops)}\n\n")
        if table:
            cols = ["cell", "mode", "rows", "status", "error", "nonfinite",
                    "wall_time_s_median", "completed", "partial", "g5"]
            fh.write("| " + " | ".join(cols) + " |\n")
            fh.write("|" + "|".join("---" for _ in cols) + "|\n")
            for r in table:
                fh.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
            fh.write("\n")
        if g5_results:
            fh.write("## G5\n\n")
            for name, verdict, code, detail, out in g5_results:
                fh.write(f"- {name}: {verdict} ({code}) {detail}\n")
            fh.write("\n")
        if stops:
            fh.write("## STOP bundles written\n\n")
            for name, code, sig, _detail in stops:
                fh.write(f"- {name}: {code} {sig}\n")
    print(f"[monitor] wrote {path}")
    return path


def _write_stops(stops, fired):
    seen = set()
    for name, code, sig, detail in stops:
        key = [name, code]
        if key in fired:
            continue
        if tuple(key) in seen:
            continue
        seen.add(tuple(key))
        G.write_stop_bundle(name, code, sig, detail, f"cell {name}")
        fired.append(key)


# -------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=10, help="minutes between passes (default 10)")
    ap.add_argument("--once", action="store_true", help="run a single pass and exit (tests)")
    ap.add_argument("--results_root", default=G.DEFAULT_RESULTS_ROOT)
    ap.add_argument("--log_dir", default=G.DEFAULT_LOG_DIR)
    ap.add_argument("--cell", default=None, help="monitor only this cell (tests)")
    ap.add_argument("--root", default=None, help="override the cell root (tests)")
    ap.add_argument("--ttba_root", default=None, help="override the TtBA partner root (tests)")
    a = ap.parse_args(argv)

    if a.cell is not None and G.CELL_BY_NAME.get(a.cell) is None:
        ap.error(f"unknown cell: {a.cell}")

    while True:
        rc = one_pass(a.results_root, a.log_dir, cell_filter=a.cell,
                      root_override=a.root, ttba_override=a.ttba_root)
        if a.once:
            return rc
        time.sleep(a.interval * 60)


if __name__ == "__main__":
    raise SystemExit(main())
