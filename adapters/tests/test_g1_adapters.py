"""G1 gate (adapter smoke): wire every adapter's make_tracker + run_official against a dummy victim.

Each (method, mode) runs in its OWN subprocess because the four official repositories share top-level module
names (``utils``, ``config``, ``dataset``, ...) that would collide in one interpreter.  The dummy victim and image
live on cuda because CGBA (``proposed_attack.py:38``) and Sign-OPT (``Sign_OPT.py`` hard-coded ``.cuda()``) force
the cuda device when a GPU is present; the real run is on cuda too.  Tiny image/budget keeps each case under a
second.

This smoke test verifies the adapter WIRING (the attack imports, the tracker counts queries, only decisions reach
the attack, and the budget raises BudgetExhausted).  The numbers are NOT checked here: the fidelity gate G3 compares
adapter vs official on the real victims in a later phase.

Run:  python adapters/tests/test_g1_adapters.py   (exit 0 iff all pass)
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ADAPTERS)
EVAL = os.path.join(REPO_ROOT, "tools", "eval")

BASELINES = {
    "CGBA": "/root/autodl-tmp/baselines/CGBA",
    "HSJA": "/root/autodl-tmp/baselines/HSJA",
    "Sign-OPT": "/root/autodl-tmp/baselines/Sign-OPT",
    "ARS-OPT": "/root/autodl-tmp/baselines/ARS-OPT",
}

CASES = [
    ("CGBA", False),
    ("CGBA", True),
    ("HSJA", False),
    ("HSJA", True),
    ("Sign-OPT", False),
    ("Sign-OPT", True),
    ("ARS-OPT", False),
    ("ARS-OPT", True),
]


def worker(name: str, targeted: bool) -> str:
    """Run one adapter against a dummy cuda victim with a tiny budget.  Prints one result line."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DummyVictim(torch.nn.Module):
        def __init__(self, ncls: int = 10, hw: int = 16):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.W = torch.nn.Parameter(torch.randn(ncls, 3 * hw * hw, generator=g))

        def forward(self, x):
            return x.flatten(1) @ self.W.T

    def predict(v, x):
        with torch.no_grad():
            return int(v(x).argmax(1).item())

    sys.path.insert(0, ADAPTERS)
    sys.path.insert(0, EVAL)
    sys.path.insert(0, REPO_ROOT)
    import _common as C  # noqa: F401
    from hardlabel_tracker import BudgetExhausted  # noqa: F401

    repo = BASELINES[name]
    victim = DummyVictim().to(device)
    x0 = torch.rand(1, 3, 16, 16, device=device)
    y_src = predict(victim, x0)
    x_tgt, y_tgt = None, None
    if targeted:
        x_tgt = torch.rand(1, 3, 16, 16, device=device)
        y_tgt = predict(victim, x_tgt)
        while y_tgt == y_src:
            x_tgt = torch.rand(1, 3, 16, 16, device=device)
            y_tgt = predict(victim, x_tgt)

    if name == "CGBA":
        import run_cgba as A
        params = dict(A.PARAMS, attack_method=("CGBA_H" if targeted else "CGBA"), iteration=2, initial_query=5)
    elif name == "HSJA":
        import run_hsja as A
        params = dict(A.PARAMS, num_iterations=2)
    elif name == "Sign-OPT":
        import run_signopt as A
        params = dict(A.PARAMS, iterations=2, query_limit=50, k=5)
    else:  # ARS-OPT
        import run_arsopt as A
        A._ARCH = "resnet50"
        A._TARGETED = targeted
        params = dict(A.BASE_PARAMS, lr=(0.2 if targeted else 0.4), iterations=2, k=5)

    # repo MUST sit in front of REPO_ROOT (which has a bare `utils/`) so the official `from utils...`/`from config`
    # imports resolve to the clone, not to LGC_frozen.
    sys.path.insert(0, repo)

    tracker = A.make_tracker(victim, x0, y_src, y_tgt, x_tgt, [25, 50], 50)
    status = "ok"
    try:
        A.run_official(tracker, x0, y_src, x_tgt, y_tgt, params)
    except BudgetExhausted:
        status = "budget"
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {e}"
    if tracker.queries <= 0:
        return "FAIL queries not counted"
    return f"{status} q={tracker.queries}"


def main() -> int:
    results = []
    for name, targeted in CASES:
        label = f"{name} {'targeted' if targeted else 'non_targeted'}"
        if name == "Sign-OPT" and targeted:
            # the adapter must refuse targeted mode (Sign-OPT is untargeted-only)
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", name, "1"],
                capture_output=True, text=True, timeout=180)
            out = (p.stdout + p.stderr).strip().splitlines()
            ok = p.returncode != 0 and any("targeted" in ln.lower() and ("refuse" in ln.lower() or "no targeted" in ln.lower()) for ln in out)
            if ok:
                print(f"   PASS  {label} (correctly refuses: Sign-OPT untargeted-only)")
                results.append(True)
            else:
                print(f"   FAIL  {label}: expected a clear refusal, got rc={p.returncode} {out[-3:]}")
                results.append(False)
            continue
        try:
            p = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--worker", name, "1" if targeted else "0"],
                capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            print(f"   FAIL  {label}: TIMEOUT")
            results.append(False)
            continue
        out = (p.stdout + p.stderr).strip()
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")][-1] if "RESULT " in out else ""
        if line:
            msg = line[len("RESULT "):]
            ok = not (msg.startswith("ERROR") or msg.startswith("FAIL"))
            print(f"   {'PASS' if ok else 'FAIL'}  {label}: {msg}")
            results.append(ok)
        else:
            print(f"   FAIL  {label}: no RESULT line (rc={p.returncode})")
            print("   ---- stdout/stderr tail ----")
            for ln in out.splitlines()[-6:]:
                print("   | " + ln)
            results.append(False)
    n_pass = sum(results)
    print(f"\n{'ALL PASS' if n_pass == len(results) else 'FAILED'}  ({n_pass}/{len(results)})")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        print("RESULT " + worker(sys.argv[2], sys.argv[3] == "1"))
    else:
        raise SystemExit(main())
