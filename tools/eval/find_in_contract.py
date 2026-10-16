#!/usr/bin/env python3
"""Read-only: print every node of the contract YAML whose key path or scalar value contains a search string.

For a HUMAN checking a contract edit without knowing the exact key layout (e.g. after adding ARS_OPT to E11).

Usage (from the worktree root):
    python tools/eval/find_in_contract.py E11
    python tools/eval/find_in_contract.py ARS_OPT --contract contract/paper_contract.yaml
Prints:  <key path> = <value>   (one line per scalar), then the sha256 of the file and "YAML OK".
A YAML syntax error is printed with its line number and exit code 1 (fix it in the editor, never in bash).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Any, Iterator, Tuple

import yaml


def walk(node: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            label = v.get("id") if isinstance(v, dict) and "id" in v else i
            yield from walk(v, f"{path}[{label}]")
    else:
        yield path, node


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("needle", help="case-sensitive search string, e.g. E11 or ARS_OPT")
    ap.add_argument("--contract", default="contract/paper_contract.yaml")
    a = ap.parse_args()
    raw = open(a.contract, "rb").read()
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        print(f"YAML ERROR: {e}")
        return 1
    hits = [(p, v) for p, v in walk(doc) if a.needle in p or a.needle in str(v)]
    for p, v in hits:
        print(f"{p} = {v!r}")
    print(f"-- {len(hits)} node(s) contain {a.needle!r}")
    print(f"sha256 {hashlib.sha256(raw).hexdigest()}  {a.contract}")
    print("YAML OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
