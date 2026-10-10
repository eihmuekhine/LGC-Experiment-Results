"""Add the evaluation-only flag --save_adv to run_benchmark_33b.py (adversarial-image store at every checkpoint).

    python apply_advstore_patch.py --runner run_benchmark_33b.py            # dry run: shows the diff
    python apply_advstore_patch.py --runner run_benchmark_33b.py --apply    # writes it (backup kept)

The patch only inserts lines; it never edits existing ones.  With --save_adv OFF the runner is unchanged in
behaviour.  With --save_adv ON it copies the image that the runner already evaluates at each checkpoint into
<run_dir>/adv_store/ (advstore.py, same folder as the runner).  No victim query, no RNG draw, no state change.
Anchors are matched exactly; if one is missing or ambiguous the patch aborts without writing anything.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import py_compile
import re
import shutil
import sys

ARG_ANCHOR = 'p.add_argument("--save_images"'
LOOP_ANCHOR = "if args.save_images and cp == args.checkpoints[-1]:"
NEEDED_PATTERNS = {  # variables the hook uses, all assigned inside the same function before the anchor
    "img": r"\bimg\s*=", "x0": r"\bx0\s*=", "stem": r"\bstem\s*=", "run_dir": r"\brun_dir\s*=",
    "sq": r"\bz,\s*sq\s*=", "m": r"\bm\s*=\s*image_metrics\(", "spec": r"\bfor spec in\b",
}
NEEDED_NAMES = tuple(NEEDED_PATTERNS)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="run_benchmark_33b.py")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src = open(a.runner, encoding="utf-8").read()
    if "--save_adv" in src:
        print(f"[=] {a.runner} already has --save_adv (sha {sha(src)}); nothing to do")
        return 0
    lines = src.splitlines(keepends=True)
    arg_idx = [i for i, l in enumerate(lines) if ARG_ANCHOR in l]
    loop_idx = [i for i, l in enumerate(lines) if l.strip() == LOOP_ANCHOR]
    if len(arg_idx) != 1 or len(loop_idx) != 1:
        print(f"[!] anchors not found exactly once (argparse {len(arg_idx)}, loop {len(loop_idx)}); aborting")
        return 1
    li = loop_idx[0]
    fdef = max(i for i in range(li) if re.match(r"def \w+\(", lines[i]))  # enclosing top-level function
    window = "".join(lines[fdef: li + 1])
    missing = [n for n in NEEDED_NAMES if not re.search(NEEDED_PATTERNS[n], window)]
    if missing:
        print(f"[!] names expected near the checkpoint loop are missing: {missing}; aborting")
        return 1
    ai = arg_idx[0]
    a_ind = re.match(r"\s*", lines[ai]).group(0)
    l_ind = re.match(r"\s*", lines[li]).group(0)
    arg_line = (f'{a_ind}p.add_argument("--save_adv", action="store_true", help="evaluation-only: store the '
                f'adversarial image at every checkpoint in <run_dir>/adv_store (advstore.py)")\n')
    hook = [
        f"{l_ind}if getattr(args, \"save_adv\", False):  # evaluation-only (advstore): no query, no RNG\n",
        f"{l_ind}    from advstore import store_adv\n",
        f"{l_ind}    store_adv(os.path.join(run_dir, \"adv_store\"), arch, stem, spec.name, cp, img, x0,\n",
        f"{l_ind}              state_query=sq, adv_flag=m.get(\"adv_float\", -1), mode=args.mode)\n",
    ]
    new_lines = lines[: ai + 1] + [arg_line] + lines[ai + 1: li] + hook + lines[li:]
    new = "".join(new_lines)
    sys.stdout.writelines(difflib.unified_diff(lines, new_lines, a.runner, a.runner + " (patched)", n=1))
    print(f"\n[*] sha256 before {sha(src)}  after {sha(new)}")
    if not a.apply:
        print("[*] dry run only; add --apply to write")
        return 0
    backup = f"{a.runner}.bak_{sha(src)}"
    shutil.copy2(a.runner, backup)
    with open(a.runner, "w", encoding="utf-8") as fh:
        fh.write(new)
    py_compile.compile(a.runner, doraise=True)
    print(f"[+] patched {a.runner} (backup {backup}); compiles OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
