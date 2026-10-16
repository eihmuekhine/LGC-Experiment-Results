#!/usr/bin/env python3
"""Contract checker for the LGC-EGC paper (skill lgc-paper-contract, Section 2).

Reads the binding contract (contract/paper_contract.yaml), optionally a column map
(contract/column_map.yaml) and the result folders, and enforces invariants I01-I15 for the given
stage.  It is READ-ONLY: it never writes into any result folder (it only writes the Markdown
report and the element-status file).  The status file is written NEXT TO --report
(<report stem>_status.md) unless --status_out is given; it never overwrites the tracked
reports/contract_status.md when --report points elsewhere.

    python tools/eval/check_contract.py --contract contract/paper_contract.yaml \
        --stage {smoke,partial,final,paper,manuscript} \
        --roots <out dirs> [--column_map contract/column_map.yaml] [--n N] [--metrics_csv path] \
        [--paper_dir paper/v3] [--generated_dir generated] [--split_dir splits/imagenet/ttba_github splits/places365 splits/robust] \
        --report reports/contract_<stage>.md [--status_out reports/contract_status.md]

Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

STAGES: Dict[str, List[str]] = {
    "smoke": ["I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "I11", "I12", "DISK"],
    "partial": ["I01", "I04", "I05", "I06", "I07", "I08", "I11"],
    "final": ["I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "I11", "I12"],
    "paper": ["I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10", "I11", "I12", "I13", "I14"],
    "manuscript": ["I14", "I15"],
}

# severity strings used by the report; "FAIL" -> exit code 1
VERDICT = ("PASS", "FAIL", "WARN", "SKIP")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
SHA_NAME_RE = re.compile(r"\b[0-9a-f]{12}\b")  # engine prefix (12 hex)
GPU_NAME_RE = re.compile(r"(RTX\s*\d{3,4}|Tesla\s+[A-Za-z0-9]+|GeForce\s+[A-Za-z0-9 ]+|A100|A10|V100|H100|L40\S*|A5000|A6000)", re.I)
TORCH_RE = re.compile(r"torch[\s=:]+(\d+\.\d+\.\d+(?:[+][A-Za-z0-9.]+)?)")


def _norm_gpu(name: str) -> str:
    """Canonical GPU name for I10: drop the vendor prefix and normalise whitespace/case.

    ``torch.cuda.get_device_name`` returns ``NVIDIA GeForce RTX 5090`` while ``nvidia-smi`` and the
    ``GPU_NAME_RE`` scan return ``GeForce RTX 5090``; both denote the same single GPU."""
    n = re.sub(r"\bNVIDIA\b", "", name, flags=re.I)
    return re.sub(r"\s+", " ", n).strip().lower()
LEGACY_RE = re.compile(r"(Images_without95%|Without95%|v2_submitted)", re.I)


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0]


def _scalar(z, key: str, default=None):
    """Extract a 0-d npz scalar robustly."""
    if key not in z.files:
        return default
    a = np.asarray(z[key])
    if a.shape == ():
        return a.item()
    a = a.reshape(-1)
    return a[0] if a.size else default


def _rel(path: str, base: str) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


@dataclass
class Row:
    """One per_checkpoint or runs row with its provenance (file:line)."""
    src: str
    d: dict


@dataclass
class Check:
    id: str
    severity: str            # FAIL | WARN | FAIL_for_tracker_best/WARN_for_engine_state (resolved in I08)
    verdict: str = "PASS"
    count: int = 0
    offending: List[str] = field(default_factory=list)
    note: str = ""

    def fail(self, *items: str) -> None:
        self.verdict = "FAIL"
        self.offending.extend(items)

    def warn(self, *items: str) -> None:
        if self.verdict != "FAIL":
            self.verdict = "WARN"
        self.offending.extend(items)


@dataclass
class Report:
    stage: str
    checks: List[Check] = field(default_factory=list)
    elements: Dict[str, str] = field(default_factory=dict)   # E-id -> COMPLETE/INCOMPLETE/CUT
    notes: List[str] = field(default_factory=list)

    def exit_code(self) -> int:
        return 1 if any(c.verdict == "FAIL" for c in self.checks) else 0

    def verdict_of(self, cid: str) -> str:
        for c in self.checks:
            if c.id == cid:
                return c.verdict
        return "SKIP"


# -------------------------------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_resolver(column_map: Optional[dict]) -> Dict[str, Dict[str, List[str]]]:
    """artefact -> contract column -> ordered candidate real names."""
    if not column_map:
        return {}
    art = column_map.get("artefacts", {})
    out: Dict[str, Dict[str, List[str]]] = {}
    for artefact, mapping in art.items():
        m = {}
        for contract_col, cands in mapping.items():
            names = list(cands) if isinstance(cands, (list, tuple)) else [cands]
            m[contract_col] = names + [contract_col]
        out[artefact] = m
    return out


def resolve(header: Sequence[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None


# -------------------------------------------------------------------------------------------------
def read_csv_lines(path: str) -> List[Row]:
    """Read a CSV returning every row with its 1-based line number."""
    rows: List[Row] = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for d in r:
            rows.append(Row(src=f"{path}:{r.line_num}", d=d))
    return rows


def scan_roots(roots: List[str]) -> dict:
    """Find per_checkpoint/runs CSVs, adv_store npz, x0.npy, env files under each root."""
    found = {"ck": [], "runs": [], "npz": [], "x0": [], "env": [], "mismatch": []}
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                if re.match(r"per_checkpoint.*\.csv$", fn):
                    found["ck"].append(p)
                elif re.match(r"runs.*\.csv$", fn):
                    found["runs"].append(p)
                elif re.search(r"_Q\d+\.npz$", fn):
                    found["npz"].append(p)
                elif fn == "x0.npy":
                    found["x0"].append(p)
                elif fn == "x0_mismatch.csv":
                    found["mismatch"].append(p)
                elif fn.startswith("env_") and (fn.endswith(".json") or fn.endswith(".txt")) or fn == "env_frame.txt":
                    found["env"].append(p)
        # our runner's env_frame.txt sits at root (also covered by the walk above)
    return found


def mode_of_ck(path: str, roots: List[str]) -> str:
    """The mode is the per_checkpoint's parent directory name."""
    return os.path.basename(os.path.dirname(path))


def load_ck(files: List[str], resolver) -> List[Row]:
    col = (resolver.get("per_checkpoint") or {})
    rows: List[Row] = []
    for p in files:
        for r in read_csv_lines(p):
            rows.append(r)
    return rows


def field(row: Row, contract_col: str, resolver_col: Dict[str, List[str]]) -> str:
    cands = resolver_col.get(contract_col, [contract_col])
    for c in cands:
        v = row.d.get(c)
        if v not in (None, ""):
            return v
    return row.d.get(contract_col, "")


def to_int(v: str, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def to_float(v: str, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_npz_index(npz_files: List[str]) -> Dict[Tuple, str]:
    """index (arch, mode, image_stem, method, budget) -> path"""
    idx = {}
    for p in npz_files:
        name = os.path.basename(p)
        m = re.match(r"^(.*)_Q(\d+)\.npz$", name)
        if not m:
            continue
        method, budget = m.group(1), int(m.group(2))
        # path .../<mode>/adv_store/<arch>/<image>/<method>_Q<b>.npz
        image = os.path.basename(os.path.dirname(p))
        arch = os.path.basename(os.path.dirname(os.path.dirname(p)))
        mode = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))))
        idx[(arch, mode, _stem(image), method, budget)] = p
    return idx


def load_splits(split_dirs, arch: str, mode: str) -> List[dict]:
    """Load {arch}_{mode}.csv from the first split dir that has it (split_dirs may be a single str)."""
    if isinstance(split_dirs, str):
        split_dirs = [split_dirs]
    for d in split_dirs:
        path = os.path.join(d, f"{arch}_{mode}.csv")
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    return []


# -------------------------------------------------------------------------------------------------
def check(contract: dict, colmap: dict, roots: List[str], stage: str, n: Optional[int],
          metrics_csv: Optional[str], paper_dir: Optional[str], generated_dir: Optional[str],
          split_dir: str, cut: Optional[set] = None) -> Report:
    const = contract.get("constants", {})
    split_dirs = [split_dir] if isinstance(split_dir, str) else list(split_dir)
    checkpoints: List[int] = [int(x) for x in const.get("checkpoints", [])]
    budget = int(const.get("budget", 10000))
    engine_prefix = str(const.get("engine_sha256_prefix", ""))
    split_sha = const.get("split_sha256", {})
    methods_cfg = contract.get("methods", {})
    elements = contract.get("paper_elements", {})
    metrics_offline = const.get("metrics_offline", [])
    asr_l2 = const.get("asr_l2", {})
    adv_dtype = const.get("adv_store", {}).get("dtype", "float16")

    resolver = build_resolver(colmap)
    ck_res = resolver.get("per_checkpoint", {})
    runs_res = resolver.get("runs", {})
    npz_res = resolver.get("npz", {})

    scan = scan_roots(roots)
    ck_rows = load_ck(scan["ck"], resolver)
    run_rows: List[Row] = [r for p in scan["runs"] for r in read_csv_lines(p)]
    npz_idx = load_npz_index(scan["npz"])

    rep = Report(stage=stage)
    rep.notes.append(f"roots={roots} stage={stage} n={n} split_dirs={split_dirs}")
    rep.notes.append(f"per_checkpoint files={len(scan['ck'])} rows={len(ck_rows)}; runs rows={len(run_rows)}; "
                     f"npz={len(scan['npz'])}; env files={len(scan['env'])}")

    def add(cid: str, severity: str) -> Check:
        c = Check(id=cid, severity=severity)
        rep.checks.append(c)
        return c

    # ---- helper: cells present in the data -----------------------------------------------
    cells: Dict[Tuple[str, str], List[Row]] = {}     # (arch, mode) -> per_checkpoint rows
    for r in ck_rows:
        arch = field(r, "arch", ck_res)
        mode = field(r, "mode", ck_res)
        cells.setdefault((arch, mode), []).append(r)

    def completed_pairs(stage: str) -> set:
        """(arch, mode, image, method) with a completed run (status ok/budget)."""
        out = set()
        for r in run_rows:
            arch = field(r, "arch", runs_res)
            mode = field(r, "mode", runs_res)
            img = field(r, "image", runs_res)
            meth = field(r, "method", runs_res)
            st = field(r, "status", runs_res)
            if st in ("ok", "budget"):
                out.add((arch, mode, _stem(img), meth))
        return out

    done_pairs = completed_pairs(stage)

    # ============================================================================================= I01
    c = add("I01", "FAIL")
    # duplicate (mode, image, method, budget) within a cell
    seen: Dict[Tuple, str] = {}
    for r in ck_rows:
        arch = field(r, "arch", ck_res)
        mode = field(r, "mode", ck_res)
        img = field(r, "image", ck_res)
        meth = field(r, "method", ck_res)
        b = field(r, "budget", ck_res)
        key = (arch, mode, _stem(img), meth, to_int(b))
        if key in seen:
            c.fail(f"{seen[key]} and {r.src}: duplicate (image, method, budget)")
        else:
            seen[key] = r.src
    # every (mode, image, method) has exactly the 8 budgets
    by_pair: Dict[Tuple, set] = {}
    for (arch, mode, img, meth, b), _ in seen.items():
        by_pair.setdefault((arch, mode, img, meth), set()).add(b)
    expect = set(checkpoints)
    for pair, budgets in sorted(by_pair.items()):
        arch, mode, img, meth = pair
        # partial stage: only completed images are checked
        if stage == "partial" and pair not in done_pairs:
            continue
        if budgets != expect:
            missing = sorted(expect - budgets)
            extra = sorted(budgets - expect)
            c.fail(f"({arch},{mode},{img},{meth}): budgets {sorted(budgets)} != 8 checkpoints "
                   f"(missing={missing}, extra={extra})")
    c.count = len(c.offending)

    # ============================================================================================= I02
    c = add("I02", "FAIL")
    limit = n if n is not None else 200
    for (arch, mode), rows in sorted(cells.items()):
        if stage == "partial":
            continue
        split_rows = load_splits(split_dirs, arch, mode)
        expected = [_stem(r["source"]) for r in split_rows[:limit]]
        if not expected:
            c.warn(f"no split CSV for ({arch},{mode}) in {split_dirs}")
            continue
        got = {_stem(field(r, "image", ck_res)) for r in rows}
        exp = set(expected)
        if got != exp:
            missing = sorted(exp - got)
            extra = sorted(got - exp)
            c.fail(f"({arch},{mode}): image set != rows 1-{limit} of split "
                   f"(missing={missing[:3]}, extra={extra[:3]})")
    c.count = len(c.offending)

    # ============================================================================================= I03
    c = add("I03", "FAIL")
    for (arch, mode), rows in sorted(cells.items()):
        if mode != "targeted" or stage == "partial":
            continue
        split_rows = load_splits(split_dirs, arch, mode)
        tgt = {_stem(r["source"]): r.get("target_label", "") for r in split_rows}
        for r in rows:
            img = _stem(field(r, "image", ck_res))
            tl = field(r, "target_label", ck_res)
            if tl in ("", None):
                continue
            if tgt.get(img, "") not in ("", None) and str(tl) != str(tgt[img]):
                c.fail(f"{r.src}: target_label {tl} != split target_label {tgt.get(img)}")
    c.count = len(c.offending)

    # ============================================================================================= I04
    c = add("I04", "FAIL")
    for r in ck_rows:
        found = to_int(field(r, "found", ck_res), 0)
        sq = to_int(field(r, "state_query", ck_res), -1)
        b = to_int(field(r, "budget", ck_res), 0)
        if found == 1 and (sq < 0 or sq > b):
            c.fail(f"{r.src}: state_query {sq} > budget {b}")
    for r in run_rows:
        q = to_int(field(r, "queries", runs_res), 0)
        if q > budget:
            c.fail(f"{r.src}: total queries {q} > {budget}")
    c.count = len(c.offending)

    # ============================================================================================= I05
    c = add("I05", "FAIL")
    npz_keys = list(npz_res.keys()) or ["delta", "budget", "state_query", "adv_flag", "l2", "linf",
                                        "method", "image", "arch", "mode"]
    for r in ck_rows:
        found = to_int(field(r, "found", ck_res), 0)
        if found != 1:
            continue
        arch = field(r, "arch", ck_res)
        mode = field(r, "mode", ck_res)
        img = _stem(field(r, "image", ck_res))
        meth = field(r, "method", ck_res)
        b = to_int(field(r, "budget", ck_res))
        p = npz_idx.get((arch, mode, img, meth, b))
        if p is None:
            c.fail(f"{r.src}: no npz for (arch={arch}, image={img}, method={meth}, budget={b})")
            continue
        try:
            z = np.load(p, allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            c.fail(f"{p}: unreadable npz ({e})")
            continue
        missing_keys = [k for k in npz_keys if k not in z.files]
        if missing_keys:
            c.fail(f"{p}: missing keys {missing_keys}")
        n_method = str(_scalar(z, "method", ""))
        n_budget = int(_scalar(z, "budget", -1))
        if n_budget != b or (n_method and n_method != meth):
            c.fail(f"{p}: budget/method ({n_budget}/{n_method}) != filename ({b}/{meth})")
    c.count = len(c.offending)

    # ============================================================================================= I06
    c = add("I06", "FAIL")
    for r in ck_rows:
        found = to_int(field(r, "found", ck_res), 0)
        if found != 1:
            continue
        arch = field(r, "arch", ck_res)
        mode = field(r, "mode", ck_res)
        img = _stem(field(r, "image", ck_res))
        meth = field(r, "method", ck_res)
        b = to_int(field(r, "budget", ck_res))
        pix_l2 = to_float(field(r, "pix_l2", ck_res), float("nan"))
        p = npz_idx.get((arch, mode, img, meth, b))
        if p is None:
            continue
        z = np.load(p, allow_pickle=False)
        l2m = float(_scalar(z, "l2", np.nan))
        if pix_l2 > 0 and not np.isnan(l2m):
            if abs(l2m - pix_l2) / pix_l2 >= 1e-3:
                c.fail(f"{p}: |npz l2 - pix_l2|/pix_l2 = {abs(l2m - pix_l2) / pix_l2:.3e} >= 1e-3")
        if "delta" in z.files:
            delta = z["delta"].astype(np.float64)
            l2r = float(np.linalg.norm(delta.ravel()))
            if l2r > 0:
                rel = abs(l2m - l2r) / l2r
                if rel >= 1e-3:
                    c.fail(f"{p}: |npz l2 - ||delta||2|/||delta||2 = {rel:.3e} >= 1e-3")
    c.count = len(c.offending)

    # ============================================================================================= I07
    c = add("I07", "FAIL")
    if scan["mismatch"]:
        c.fail("x0_mismatch.csv present: " + ", ".join(scan["mismatch"][:3]))
    for p in scan["x0"]:
        try:
            x = np.load(p)
        except Exception as e:  # noqa: BLE001
            c.fail(f"{p}: unreadable ({e})")
            continue
        if x.shape != (3, 224, 224):
            c.fail(f"{p}: x0 shape {x.shape} != (3,224,224)")
        if not np.isfinite(x).all():
            c.fail(f"{p}: x0 not finite")
        if x.min() < -1e-6 or x.max() > 1 + 1e-6:
            c.fail(f"{p}: x0 outside [0,1]")
    c.count = len(c.offending)

    # ============================================================================================= I08
    c = add("I08", "FAIL")   # severity resolved per method below
    # group pix_l2 by (arch, mode, image, method) over budgets
    by_l2: Dict[Tuple, List[Tuple[int, float, str]]] = {}
    for r in ck_rows:
        found = to_int(field(r, "found", ck_res), 0)
        if found != 1:
            continue
        arch = field(r, "arch", ck_res)
        mode = field(r, "mode", ck_res)
        img = _stem(field(r, "image", ck_res))
        meth = field(r, "method", ck_res)
        b = to_int(field(r, "budget", ck_res), 0)
        l2 = to_float(field(r, "pix_l2", ck_res), float("nan"))
        by_l2.setdefault((arch, mode, img, meth), []).append((b, l2, r.src))
    violations = 0
    for pair, vals in sorted(by_l2.items()):
        arch, mode, img, meth = pair
        vals = sorted(vals)
        for (b1, l1, s1), (b2, l2v, s2) in zip(vals, vals[1:]):
            if np.isfinite(l1) and np.isfinite(l2v) and l2v > l1 + 1e-12:
                conv = (methods_cfg.get(meth) or {}).get("convention", "")
                if conv == "tracker_best":
                    c.fail(f"({arch},{mode},{img},{meth}): L2 {l1:.4f}@{b1} -> {l2v:.4f}@{b2} increased")
                else:  # engine_state -> WARN only
                    c.warn(f"({arch},{mode},{img},{meth}): L2 {l1:.4f}@{b1} -> {l2v:.4f}@{b2} increased (engine_state)")
                violations += 1
    c.count = violations

    # ============================================================================================= I09 (WARN)
    c = add("I09", "WARN")
    for (arch, mode), rows in sorted(cells.items()):
        if mode != "targeted":
            continue
        for b in (100, 200):
            f0 = sum(1 for r in rows if to_int(field(r, "budget", ck_res)) == b
                     and to_int(field(r, "found", ck_res), 0) == 0)
            tot = sum(1 for r in rows if to_int(field(r, "budget", ck_res)) == b)
            if f0 > 0:
                c.warn(f"({arch},{mode}): found=0 at budget {b}: {f0}/{tot}")
    c.count = len(c.offending)

    # ============================================================================================= I10
    c = add("I10", "FAIL")
    env_texts = []
    for p in scan["env"]:
        try:
            env_texts.append((p, open(p, encoding="utf-8", errors="replace").read()))
        except OSError:
            continue
    if not env_texts:
        c.warn("no env file found; split/engine sha and GPU/torch not verifiable")
    else:
        env_files: Dict[str, List[Tuple[str, set]]] = {}   # dirname -> [(path, {hex64,...})]
        all_hex: List[str] = []
        gpus: set = set()
        torches: set = set()
        for p, txt in env_texts:
            hexes = set(HEX64.findall(txt))
            if p.endswith(".json"):
                try:
                    d = json.loads(txt)
                except json.JSONDecodeError:
                    d = {}
                for k in ("gpu", "gpu_name"):
                    if d.get(k):
                        gpus.add(_norm_gpu(str(d[k])))
                if d.get("torch"):
                    torches.add(str(d["torch"]))
                if d.get("split_csv_sha256"):
                    hexes.add(str(d["split_csv_sha256"]))
            m = TORCH_RE.search(txt)
            if m:
                torches.add(m.group(1))
            gm = GPU_NAME_RE.search(txt)
            if gm:
                gpus.add(_norm_gpu(gm.group(0)))
            all_hex += list(hexes)
            env_files.setdefault(os.path.dirname(p), []).append((p, hexes))

        # per-cell: the env file(s) in the cell's root or root/<mode> must hold that cell's split sha (no pooling).
        cell_roots: Dict[Tuple[str, str], set] = {}
        for (arch, mode), rows in cells.items():
            rset = set()
            for r in rows:
                path = r.src.rsplit(":", 1)[0]
                rset.add(os.path.dirname(os.path.dirname(path)))
            cell_roots[(arch, mode)] = rset

        for (arch, mode), cell_dirs in sorted(cell_roots.items()):
            key = f"{arch}_{mode}"
            expected = split_sha.get(key, "")
            if not expected:
                c.fail(f"({arch},{mode}): unregistered split (constants.split_sha256['{key}'] missing)")
                continue
            valid_dirs = set(cell_dirs)
            for cd in cell_dirs:
                valid_dirs.add(os.path.join(cd, mode))
            found = False
            for d in valid_dirs:
                for (p, hexes) in env_files.get(d, []):
                    if expected in hexes:
                        found = True
                        break
                if found:
                    break
            if not found:
                c.fail(f"({arch},{mode}): split sha256 {expected[:12]}… not found in any env file at root or root/{mode}")

        if engine_prefix and cells:
            # engine sha is only required for cells that include our method (LGC_RC_EGC)
            has_ours = any(methods_cfg.get(meth, {}).get("role") == "ours" for meth in
                           {field(r, "method", ck_res) for r in ck_rows})
            if has_ours and not any(h.startswith(engine_prefix) for h in all_hex):
                c.fail(f"engine sha with prefix {engine_prefix} not found in env")
        if len(gpus) > 1:
            c.fail(f"multiple GPU names across cells: {sorted(gpus)}")
        if len(torches) > 1:
            c.fail(f"multiple torch versions across cells: {sorted(torches)}")
    c.count = len(c.offending)

    # ============================================================================================= I11
    c = add("I11", "FAIL")
    if not env_texts:
        c.warn("no env file found; ADV_STORE_BUDGETS/dtype not verifiable")
    else:
        for p, txt in env_texts:
            is_ours = os.path.basename(p) == "env_frame.txt"
            m = re.search(r"ADV_STORE_BUDGETS[=\s]+(\S+)", txt)
            if m:
                val = m.group(1).strip()
                if val not in ("", "unset"):
                    c.fail(f"{p}: ADV_STORE_BUDGETS set to {val} (contract I11: must be unset)")
            elif is_ours:
                c.fail(f"{p}: our env file has no ADV_STORE_BUDGETS token (contract I11)")
            if "ADV_STORE_DTYPE" in txt and adv_dtype not in txt:
                c.fail(f"{p}: ADV_STORE_DTYPE not {adv_dtype}")
    c.count = len(c.offending)

    # ============================================================================================= I12
    c = add("I12", "FAIL")
    for p in scan["ck"] + scan["runs"] + scan["env"] + [os.path.abspath(r) for r in roots]:
        if LEGACY_RE.search(p):
            c.fail(f"legacy path token in {p}")
    c.count = len(c.offending)

    # ============================================================================================= DISK (smoke only)
    if stage == "smoke":
        c = add("DISK", "FAIL")
        try:
            free = shutil.disk_usage(os.path.abspath(roots[0]) if roots else ".").free / 2**30
            if free < 25:
                c.fail(f"disk free {free:.1f} GB < 25 GB")
            c.note = f"{free:.1f} GB free"
        except OSError as e:
            c.warn(f"disk usage unavailable: {e}")

    # ============================================================================================= I13
    if stage in ("paper",):
        c = add("I13", "FAIL")
        if not metrics_csv or not os.path.exists(metrics_csv):
            c.fail(f"metrics CSV {metrics_csv} missing")
        else:
            df = pd.read_csv(metrics_csv)
            if len(df) == 0:
                c.fail(f"metrics CSV {metrics_csv} empty")
            # one row per stored npz (dedupe on root,arch,mode,image,method,budget)
            if len(df) != len(df.drop_duplicates(subset=["arch", "mode", "image", "method", "budget"])):
                c.fail("metrics CSV has duplicate (arch,mode,image,method,budget) rows")
            for col in metrics_offline:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if col in df.columns and df[col].isna().any():
                    c.fail(f"metrics column {col} has NaN")
            if "l2_relerr" in df.columns and (df["l2_relerr"] >= 1e-3).any():
                c.fail("l2_relerr >= 1e-3 present")
        c.count = len(c.offending)

    # ============================================================================================= I14
    if stage in ("paper", "manuscript"):
        c = add("I14", "FAIL")
        cut_set = cut if cut is not None else _read_cut_markers()
        has_ck = bool(scan["ck"])
        has_npz = bool(scan["npz"])
        has_runs = bool(scan["runs"])
        has_env = bool(scan["env"])
        has_metrics = bool(metrics_csv and os.path.exists(metrics_csv))
        arches = {a for (a, _m) in cells}
        ctx = {"has_ck": has_ck, "has_npz": has_npz, "has_runs": has_runs,
               "has_env": has_env, "has_metrics": has_metrics, "arches": arches}
        for eid, el in elements.items():
            status = el.get("status", "REQUIRED")
            complete = _element_complete(el, ctx)
            if status == "REQUIRED":
                rep.elements[eid] = "COMPLETE" if complete else "INCOMPLETE"
                if not complete:
                    c.fail(f"{eid} (REQUIRED): inputs absent")
            else:  # CONDITIONAL
                if complete:
                    rep.elements[eid] = "COMPLETE"
                elif _is_cut(eid, cut_set):
                    rep.elements[eid] = "CUT"
                else:
                    rep.elements[eid] = "INCOMPLETE"
                    c.fail(f"{eid} (CONDITIONAL): inputs absent and not marked CUT")
        c.count = len(c.offending)

    # ============================================================================================= I15
    if stage == "manuscript":
        c = add("I15", "FAIL")
        if not paper_dir or not os.path.isdir(paper_dir):
            c.fail(f"paper_dir {paper_dir} missing")
        else:
            used = set()
            for root_, _, fns in os.walk(paper_dir):
                for fn in fns:
                    if fn.endswith(".tex"):
                        txt = open(os.path.join(root_, fn), encoding="utf-8", errors="replace").read()
                        for m in re.findall(r"\\Num[A-Za-z0-9_]+", txt):
                            used.add(m[1:])
            defined: Dict[str, str] = {}
            numbers = os.path.join(generated_dir or "generated", "numbers.tex")
            if os.path.exists(numbers):
                txt = open(numbers, encoding="utf-8", errors="replace").read()
                for m in re.finditer(r"\\(?:newcommand|def)\{\\(Num[A-Za-z0-9_]+)\}\s*\{(.*?)\}", txt):
                    defined[m.group(1)] = m.group(2).strip()
            for macro in sorted(used):
                if macro not in defined:
                    c.fail(f"{macro} used in paper but not defined in numbers.tex")
                elif defined[macro] in ("", "??", "?", "--"):
                    c.fail(f"{macro} = '{defined[macro]}' in numbers.tex")
        c.count = len(c.offending)

    return rep


def _source_present(src: str, ctx: dict) -> bool:
    if src == "metrics_csv":
        return ctx["has_metrics"]
    if src == "per_checkpoint":
        return ctx["has_ck"]
    if src.startswith("adv_store"):
        return ctx["has_npz"]
    if src == "runs":
        return ctx["has_runs"]
    if src in ("env", "run_manifest"):
        return ctx["has_env"]
    if src == "none":
        return True
    # external inputs (confirm_37 CSVs, ttba_table1.json, P2 report, ...): not verified here
    return True


def _element_complete(el: dict, ctx: dict) -> bool:
    inputs = el.get("inputs", {})
    src = inputs.get("source")
    if src:
        if isinstance(src, (list, tuple)):
            return all(_source_present(s, ctx) for s in src)
        return _source_present(str(src), ctx)
    # no source: a text-only element (source == none), or a run-conditioned element (victims)
    victims = inputs.get("victims") or inputs.get("extra_victims")
    if victims:
        names = victims if isinstance(victims, (list, tuple)) else [victims]
        return any(any(v in a for a in ctx["arches"]) for v in names)
    return True


def _is_cut(eid: str, cut_set: set) -> bool:
    return eid in cut_set or eid.split("_")[0] in cut_set


def _read_cut_markers() -> set:
    path = os.path.join("reports", "contract_status.md")
    if not os.path.exists(path):
        return set()
    cut = set()
    for line in open(path, encoding="utf-8", errors="replace"):
        for m in re.findall(r"\bE\d+[a-z]?(?:_\w+)?", line):
            if re.search(r"\bCUT\b", line):
                cut.add(m)
    return cut


# -------------------------------------------------------------------------------------------------
def render_report(rep: Report, report_path: str, contract: dict, status_out: Optional[str] = None) -> None:
    lines = [
        f"# Contract check — stage `{rep.stage}`",
        "",
        f"Exit code: **{rep.exit_code()}**  (0 = no FAIL)",
        "",
    ]
    for note in rep.notes:
        lines.append(f"- {note}")
    lines += ["", "## Checks", "",
              "| check id | severity | PASS/FAIL/WARN | count | first offending rows (file:row) |",
              "|---|---|---|---|---|"]
    for c in rep.checks:
        off = ", ".join(c.offending[:10]) if c.offending else ("—" if not c.note else c.note)
        lines.append(f"| {c.id} | {c.severity} | {c.verdict} | {c.count} | {off} |")
    if rep.elements:
        lines += ["", "## Paper elements", "",
                  "| element | status |", "|---|---|"]
        for eid, st in sorted(rep.elements.items()):
            lines.append(f"| {eid} | {st} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    # element status: next to --report (<stem>_status.md) unless --status_out is given
    status_path = status_out or (os.path.splitext(report_path)[0] + "_status.md")
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    with open(status_path, "w", encoding="utf-8") as fh:
        fh.write("# Contract status (generated by check_contract.py)\n\n")
        fh.write(f"stage: {rep.stage}  (exit code {rep.exit_code()})\n\n| element | status |\n|---|---|\n")
        for eid, st in sorted(rep.elements.items()):
            fh.write(f"| {eid} | {st} |\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", default="contract/paper_contract.yaml")
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--column_map", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--metrics_csv", default=None)
    ap.add_argument("--paper_dir", default="paper/v3")
    ap.add_argument("--generated_dir", default="generated")
    ap.add_argument("--split_dir", nargs="+", default=None,
                    help="one or more split dirs (first dir that contains <arch>_<mode>.csv wins)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--status_out", default=None,
                    help="status file path (default: <report stem>_status.md, next to --report)")
    a = ap.parse_args(argv)

    if not os.path.exists(a.contract):
        print(f"[!] contract not found: {a.contract}", file=sys.stderr)
        return 2
    contract = load_yaml(a.contract)
    colmap = load_yaml(a.column_map) if a.column_map and os.path.exists(a.column_map) else None

    # defaults for split dir / n
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(a.contract)))
    split_dir = a.split_dir or [os.path.join(repo_root, "splits", "imagenet", "ttba_github")]
    if a.n is None:
        a.n = 3 if a.stage == "smoke" else 200
    report_path = a.report or f"reports/contract_{a.stage}.md"

    rep = check(contract, colmap, a.roots, a.stage, a.n, a.metrics_csv, a.paper_dir,
                a.generated_dir, split_dir, cut=_read_cut_markers())
    render_report(rep, report_path, contract, status_out=a.status_out)

    print(f"[*] stage={a.stage} roots={a.roots}")
    for c in rep.checks:
        off = c.offending[:3]
        print(f"    {c.verdict:4s}  {c.id:5s} ({c.severity})  count={c.count}" + (f"  e.g. {off}" if off else ""))
    print(f"[*] report: {report_path}; exit code {rep.exit_code()}")
    return rep.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
