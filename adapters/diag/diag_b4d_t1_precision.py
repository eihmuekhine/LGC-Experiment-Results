#!/usr/bin/env python3
"""B4d T1 — precision parity (DIAGNOSTIC ONLY, changes nothing).

For every paper runner, report the effective values of
  torch.backends.cuda.matmul.allow_tf32
  torch.backends.cudnn.allow_tf32
  torch.backends.cudnn.deterministic
  torch.backends.cudnn.benchmark
  torch.get_float32_matmul_precision()
and WHERE each is set (file:line), by reading the source text (never re-running an attack).

The only place any runner DISABLES TF32 is lgc_bench/env.py:configure_backend (matmul + cudnn
allow_tf32 -> False), called by the two EGC runners.  The baseline adapters do NOT call
configure_backend: they inherit the torch defaults, except TtBA's official dataset.load_dataset_model
sets cudnn.deterministic=True (dataset.py:220).  No runner sets float32_matmul_precision.

    python adapters/diag/diag_b4d_t1_precision.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = "/root/autodl-tmp/LGC_frozen"

RUNNERS = [
    ("run_benchmark_33b.py (EGC, built from run_frame_b1.sh argv)",
     ["run_benchmark_33b.py", "lgc_bench/env.py"], REPO),
    ("ttba_github/run_frame_b1_generic.py (EGC generic E10/E11)",
     ["ttba_github/run_frame_b1_generic.py", "lgc_bench/env.py"], REPO),
    ("adapters/run_cgba.py (CGBA / CGBA_H)",
     ["adapters/run_cgba.py", "adapters/_common.py"], REPO),
    ("adapters/run_ttba_v2.py (TtBA)",
     ["adapters/run_ttba_v2.py", "adapters/_common.py", "/root/autodl-tmp/TtBA/dataset.py"], REPO),
    ("adapters/run_arsopt.py (ARS_OPT)",
     ["adapters/run_arsopt.py", "adapters/_common.py"], REPO),
]

SETTING_RE = re.compile(
    r"(torch\.backends\.(?:cuda\.matmul|cudnn)\.allow_tf32|torch\.backends\.cudnn\.deterministic|"
    r"torch\.backends\.cudnn\.benchmark|float32_matmul_precision|set_float32_matmul_precision|"
    r"configure_backend|allow_tf32|nondeterministic)")

ENV_KEYS = [
    "matmul.allow_tf32",
    "cudnn.allow_tf32",
    "cudnn.deterministic",
    "cudnn.benchmark",
    "float32_matmul_precision",
]


def grep_settings(path: str):
    hits = []
    try:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if SETTING_RE.search(line):
                    hits.append((i, line.rstrip()))
    except OSError:
        pass
    return hits


def defaults() -> dict:
    code = (
        "import torch\n"
        "print(torch.__version__)\n"
        "print(torch.backends.cuda.matmul.allow_tf32)\n"
        "print(torch.backends.cudnn.allow_tf32)\n"
        "print(torch.backends.cudnn.deterministic)\n"
        "print(torch.backends.cudnn.benchmark)\n"
        "print(torch.get_float32_matmul_precision())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    lines = out.stdout.strip().splitlines()
    return {
        "torch": lines[0] if len(lines) > 0 else "?",
        "matmul.allow_tf32": lines[1] if len(lines) > 1 else "?",
        "cudnn.allow_tf32": lines[2] if len(lines) > 2 else "?",
        "cudnn.deterministic": lines[3] if len(lines) > 3 else "?",
        "cudnn.benchmark": lines[4] if len(lines) > 4 else "?",
        "float32_matmul_precision": lines[5] if len(lines) > 5 else "?",
    }


def after_configure_backend() -> dict:
    code = (
        "import torch\n"
        "import sys; sys.path.insert(0, '.')\n"
        "from lgc_bench.env import configure_backend\n"
        "configure_backend(torch.device('cuda'))\n"
        "print(torch.backends.cuda.matmul.allow_tf32)\n"
        "print(torch.backends.cudnn.allow_tf32)\n"
        "print(torch.backends.cudnn.deterministic)\n"
        "print(torch.backends.cudnn.benchmark)\n"
        "print(torch.get_float32_matmul_precision())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    lines = out.stdout.strip().splitlines()
    return {
        "matmul.allow_tf32": lines[0] if len(lines) > 0 else "?",
        "cudnn.allow_tf32": lines[1] if len(lines) > 1 else "?",
        "cudnn.deterministic": lines[2] if len(lines) > 2 else "?",
        "cudnn.benchmark": lines[3] if len(lines) > 3 else "?",
        "float32_matmul_precision": lines[4] if len(lines) > 4 else "?",
    }


def main() -> int:
    d = defaults()
    print("=== torch defaults (fresh process) ===")
    print(f"  torch {d['torch']}")
    for short in ENV_KEYS:
        print(f"  {short:<30s} = {d[short]}")
    print()

    cb = after_configure_backend()
    print("=== lgc_bench/env.py:configure_backend(device)  (called by the EGC runners) ===")
    print("  file:line  lgc_bench/env.py:46-52")
    for short in ENV_KEYS:
        print(f"  {short:<30s} = {cb[short]}")
    print()

    print("=== per-runner settings (static: where each is set) ===")
    for name, files, base in RUNNERS:
        print(f"\n-- {name}")
        for rel in files:
            path = rel if os.path.isabs(rel) else os.path.join(base, rel)
            hits = grep_settings(path)
            relp = os.path.relpath(path, REPO) if not os.path.isabs(rel) else rel
            if hits:
                for ln, text in hits:
                    print(f"    {relp}:{ln}: {text.strip()}")
            else:
                print(f"    {relp}: (no precision setting)")

    print()
    print("=== effective values (deduced) ===")
    print("run_benchmark_33b.py            : TF32 OFF (matmul=F,cudnn=F), deterministic=T, benchmark=F, "
          "fp32_prec=highest (configure_backend @ run_benchmark_33b.py:1157)")
    print("run_frame_b1_generic.py         : TF32 OFF (matmul=F,cudnn=F), deterministic=T, benchmark=F, "
          "fp32_prec=highest (configure_backend @ :95)")
    print("run_cgba.py / run_arsopt.py     : torch defaults — matmul=F, cudnn.allow_tf32=T, deterministic=F, "
          "benchmark=F, fp32_prec=highest (NO precision setting in adapter)")
    print("run_ttba_v2.py                  : cudnn.deterministic=T (official dataset.py:220), cudnn.allow_tf32=T "
          "(default), matmul=F (default), benchmark=F (default), fp32_prec=highest")
    print()
    print("TF32 DISABLED only in lgc_bench/env.py:configure_backend (EGC runners). The TtBA paper env "
          "(torch 2.3.0, RTX 4090) ran with cudnn.allow_tf32=True (default): our TtBA adapter does NOT disable "
          "TF32, so it matches the paper's default. NOTHING changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
