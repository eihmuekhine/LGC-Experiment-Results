"""G1 gate: contract checkpoints (CRITICAL v7: override evalkit defaults) + shared IO helpers (CPU, no GPU).

Run:  python adapters/tests/test_g1_helpers.py   (exit 0 iff all pass)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C  # noqa: E402

RESULTS = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


def test_contract_checkpoints() -> None:
    print("[1] contract checkpoints override evalkit defaults")
    contract = C.load_contract()
    ck = C.contract_checkpoints(contract)
    check("checkpoints == [100,200,500,1000,2000,2500,5000,10000]",
          ck == [100, 200, 500, 1000, 2000, 2500, 5000, 10000])
    check("budget == 10000", C.contract_budget(contract) == 10000)
    # the evalkit default (6 budgets) MUST NOT be what the adapters use
    check("8 checkpoints (not the evalkit 6-budget default)", len(ck) == 8)


def test_csv_and_seed() -> None:
    print("[2] CSV append / done_images / per-image seed determinism")
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "runs_M.csv")
        C.append_csv(rp, C.RUN_FIELDS, [{"arch": "resnet50", "mode": "non_targeted", "image": "a", "method": "M",
                                         "queries": 5, "status": "ok", "wall_time_s": 1.0, "error": ""}])
        check("done_images reads ok/budget", C.done_images(rp) == {"a"})
        C.append_csv(rp, C.RUN_FIELDS, [{"arch": "resnet50", "mode": "non_targeted", "image": "b", "method": "M",
                                         "queries": 5, "status": "error", "wall_time_s": 1.0, "error": "x"}])
        check("error status is NOT 'done' (will retry)", C.done_images(rp) == {"a"})

    C.seed_all(123)
    import torch
    a = torch.randn(5)
    C.seed_all(123)
    b = torch.randn(5)
    check("per-image seed reproducible", bool((a == b).all()))


def test_env_manifest_roundtrip() -> None:
    print("[3] env + manifest + resume refusal on mismatch")
    import types

    args = types.SimpleNamespace(repo="/tmp/x", split_csv="/tmp/s.csv")
    contract = C.load_contract()
    ck = C.contract_checkpoints(contract)
    env = C.build_env("CGBA", "resnet50", "non_targeted", args, contract, {"a": 1}, {"a": "s"},
                      "abc123", "resnet50-11ad3fa6.pth", "def456", "aaa", "bbb", "ccc", ck)
    manifest = C.build_manifest(env, contract)
    check("manifest has split sha / weight sha / repo commit / adapter+tracker sha",
          manifest["split_csv_sha256"] == "abc123" and manifest["victim_weight_sha256"] == "def456"
          and manifest["repo_commit"] == "ccc" and manifest["adapter_sha256"] == "aaa")
    check("manifest checkpoints == contract checkpoints", manifest["checkpoints"] == ck)
    with tempfile.TemporaryDirectory() as d:
        json.dump(manifest, open(os.path.join(d, "run_manifest.json"), "w"))
        C.check_manifest(d, manifest)  # identical -> no raise
        mism = dict(manifest, split_csv_sha256="DIFFERENT")
        try:
            C.check_manifest(d, mism)
            ok = False
        except SystemExit:
            ok = True
        check("resume refuses on manifest mismatch", ok)


def main() -> int:
    test_contract_checkpoints()
    test_csv_and_seed()
    test_env_manifest_roundtrip()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
