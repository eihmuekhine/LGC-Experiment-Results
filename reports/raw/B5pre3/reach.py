#!/usr/bin/env python3
"""B5-pre3 Step 2a/2b (v3): per-entry reachability of third-party modules.

Static import-closure simulation with entry-specific sys.path:
  seeds  = clone dirs actually inserted at runtime (evidence: run_b5_parallel.sh
           repo vars + the entry's function-scoped sys.path.insert of --repo/--ttba_repo)
           + the entry file's own directory (python adds it automatically)
           + module-level sys.path literals of every file reached by that entry.
Any import that cannot be resolved to a local .py file on that entry's path is
recorded as third-party (top-level vs lazy). Stdlib imports are ignored.
"""
import ast
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/LGC_frozen")
ENTRIES = [
    ("run_benchmark_33b.py", REPO / "run_benchmark_33b.py"),
    ("adapters/run_ttba_v2.py", REPO / "adapters" / "run_ttba_v2.py"),
    ("adapters/run_cgba.py", REPO / "adapters" / "run_cgba.py"),
    ("adapters/run_arsopt.py", REPO / "adapters" / "run_arsopt.py"),
    ("ttba_github/run_frame_b1_generic.py", REPO / "ttba_github" / "run_frame_b1_generic.py"),
    ("tools/eval/b5_gate_cell.py", REPO / "tools" / "eval" / "b5_gate_cell.py"),
    ("tools/eval/b5_monitor.py", REPO / "tools" / "eval" / "b5_monitor.py"),
    ("lgc_bench/metrics.py", REPO / "lgc_bench" / "metrics.py"),
    ("tools/eval/eval_metrics_offline.py", REPO / "tools" / "eval" / "eval_metrics_offline.py"),
]
# Clone dirs inserted at runtime via function-scoped sys.path inserts of
# --repo/--ttba_repo (run_b5_parallel.sh: TTBA_REPO=/root/autodl-tmp/TtBA,
# CGBA_REPO=/root/autodl-tmp/baselines/CGBA, ARS_REPO=/root/autodl-tmp/baselines/ARS-OPT).
EXTRAS = {
    "adapters/run_ttba_v2.py": ["/root/autodl-tmp/TtBA"],
    "adapters/run_cgba.py": ["/root/autodl-tmp/baselines/CGBA"],
    "adapters/run_arsopt.py": ["/root/autodl-tmp/baselines/ARS-OPT"],
}
OUT = REPO / "reports" / "raw" / "B5pre3"

STDLIB = set(sys.stdlib_module_names)

ALIAS_DIST = {
    "pil": ["pillow"],
    "skimage": ["scikit-image"],
    "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "yaml": ["pyyaml"],
    "bs4": ["beautifulsoup4"],
    "sklearn": ["scikit-learn"],
}

parse_cache = {}
edge_cache = {}


def read_src(path):
    try:
        return path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None


def parse(path):
    if path in parse_cache:
        return parse_cache[path]
    src = read_src(path)
    tree = None
    if src is not None:
        try:
            tree = ast.parse(src, filename=str(path))
        except (SyntaxError, ValueError, MemoryError):
            pass
    parse_cache[path] = tree
    return tree


def inside_function(node, parent_map):
    p = parent_map.get(node)
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return True
        p = parent_map.get(p)
    return False


def const_literal(node, known):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                s = const_literal(v.value, known)
                if s is None:
                    return None
                parts.append(s)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = const_literal(node.left, known)
        b = const_literal(node.right, known)
        if a is not None and b is not None:
            return a + b
    return None


def _is_ospath_call(f):
    if isinstance(f, ast.Attribute):
        base = f.value
        if isinstance(base, ast.Name) and base.id == "os" and f.attr == "path":
            return True
        if (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "os"
            and base.attr == "path"
        ):
            return True
    return False


def eval_path(node, known, file):
    """Best-effort static evaluation of path expressions (str or None)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(Path(file).resolve())
        if node.id in known:
            return known[node.id]
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = eval_path(node.left, known, file)
        b = eval_path(node.right, known, file)
        if a is not None and b is not None:
            return a + b
        return None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                s = eval_path(v.value, known, file)
                if s is None:
                    return None
                parts.append(s)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.Attribute):
        base = eval_path(node.value, known, file)
        if base is None:
            return None
        p = Path(base)
        if node.attr == "parent":
            return str(p.parent)
        if node.attr == "resolve":
            return str(p.resolve())
        if node.attr == "absolute":
            return str(p.absolute())
        return None
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("str", "Path"):
            if node.args:
                return eval_path(node.args[0], known, file)
            return None
        if _is_ospath_call(f):
            fn = f.attr
            args = [eval_path(a, known, file) for a in node.args]
            if any(a is None for a in args):
                return None
            if fn == "join":
                return os.path.join(*args)
            if fn == "dirname":
                return os.path.dirname(args[0])
            if fn == "abspath":
                return os.path.abspath(args[0])
            if fn == "expanduser":
                return os.path.expanduser(args[0])
            if fn == "realpath":
                return os.path.realpath(args[0])
            return None
        if (
            isinstance(f, ast.Attribute)
            and f.attr in ("parent", "resolve", "absolute", "abspath")
            and f.attr in ("abspath",)
        ):
            return None
        if isinstance(f, ast.Attribute) and f.attr in ("parent", "absolute"):
            base = eval_path(f.value, known, file)
            if base is None:
                return None
            if f.attr == "parent":
                return str(Path(base).parent)
            return str(Path(base).absolute())
    return None


def extract_edges(path):
    """Return (path_stmts [(path, scope)], edges).
    edge = (kind, dotted, scope, lineno[, rel_base])"""
    if path in edge_cache:
        return edge_cache[path]
    tree = parse(path)
    result = ([], [])
    if tree is None:
        edge_cache[path] = result
        return result

    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    known = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    v = eval_path(node.value, known, path)
                    if isinstance(v, str):
                        known[t.id] = v
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            v = eval_path(node.value, known, path)
            if isinstance(v, str):
                known[node.target.id] = v

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr in ("insert", "append")
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "path"
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "sys"
        ):
            if node.args:
                raw = node.args[-1]
                val = eval_path(raw, known, path)
                if val is None:
                    val = const_literal(raw, known)
                if val is not None:
                    if not os.path.isabs(val):
                        val = os.path.join(str(path.parent), val)
                    scope = "lazy" if inside_function(node, parent_map) else "top"
                    result[0].append((str(Path(val)), scope))

    for node in ast.walk(tree):
        scope = "lazy" if inside_function(node, parent_map) else "top"
        if isinstance(node, ast.Import):
            for alias in node.names:
                result[1].append(("import", alias.name, scope, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            mod_parts = node.module.split(".") if node.module else []
            if node.level and node.level > 0:
                base = path.parent
                for _ in range(node.level - 1):
                    base = base.parent
                for alias in node.names:
                    target = ".".join(mod_parts + [alias.name])
                    result[1].append(("from_rel", target, scope, node.lineno, str(base)))
            else:
                for alias in node.names:
                    target = ".".join(mod_parts + [alias.name])
                    result[1].append(("from", target, scope, node.lineno))
        elif isinstance(node, ast.Call):
            f = node.func
            fname = None
            if isinstance(f, ast.Name) and f.id in ("import_module", "__import__"):
                fname = f.id
            elif isinstance(f, ast.Attribute) and f.attr == "import_module":
                fname = "import_module"
            if fname and node.args:
                lit = const_literal(node.args[0], known)
                if isinstance(lit, str):
                    result[1].append(("dyn", lit, scope, node.lineno))

    edge_cache[path] = result
    return result


def candidate_files(dotted, roots, rel_base=None):
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return []
    found = []
    bases = [Path(rel_base)] if rel_base is not None else list(roots)
    for b in bases:
        p = b
        for i, part in enumerate(parts):
            last = (i == len(parts) - 1)
            pyd = p / f"{part}.py"
            ini = p / part / "__init__.py"
            if last:
                if pyd.is_file():
                    found.append(pyd)
                if ini.is_file():
                    found.append(ini)
                break
            if ini.is_file():
                found.append(ini)
                p = p / part
            elif pyd.is_file():
                found.append(pyd)
                break
            elif (p / part).is_dir():
                p = p / part
            else:
                break
    return found


def resolve(edge, roots):
    if edge[0] == "from_rel":
        return candidate_files(edge[1], roots, rel_base=edge[4])
    return candidate_files(edge[1], roots)


def norm_top(dotted):
    parts = [p for p in dotted.split(".") if p]
    return parts[0] if parts else ""


def main():
    per_entry = {}
    for label, path in ENTRIES:
        seeds = list(EXTRAS.get(label, []))
        if str(path.parent) not in [str(Path(x)) for x in seeds]:
            seeds.append(str(path.parent))
        roots = [Path(x) for x in seeds]
        entry_file = str(path)
        reached = {entry_file}
        processed = set()
        progress = True
        rounds = 0
        while progress:
            progress = False
            rounds += 1
            for f in sorted(reached - processed):
                pathst, _ = extract_edges(Path(f))
                for val, scope in pathst:
                    if scope == "top" and val and str(Path(val)) not in [str(r) for r in roots]:
                        roots.append(Path(val))
                        progress = True
                processed.add(f)
            for f in sorted(processed):
                for e in extract_edges(Path(f))[1]:
                    for c in resolve(e, roots):
                        if str(c) not in reached:
                            reached.add(str(c))
                            progress = True
        per_entry[label] = {"reachable": reached, "roots": roots, "rounds": rounds}

    union = set()
    for v in per_entry.values():
        union |= v["reachable"]

    records = {}
    for label, v in per_entry.items():
        for f in sorted(v["reachable"]):
            edges = extract_edges(Path(f))[1]
            for e in edges:
                top = norm_top(e[1])
                if not top or top in STDLIB:
                    continue
                if resolve(e, v["roots"]):
                    continue
                rec = records.setdefault(top, {"scopes": set(), "hits": []})
                rec["scopes"].add(e[2])
                rec["hits"].append({"entry": label, "file": f, "line": e[3], "scope": e[2],
                                    "stmt": e[0] + " " + e[1]})

    harbin = {}
    with open(OUT / "pip_freeze_harbin.txt") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if i == 0 and line.startswith("Python "):
                continue
            if not line:
                continue
            if "==" in line:
                nm, ver = line.split("==", 1)
                harbin[nm.strip().lower().replace("_", "-")] = ver.strip()

    def harbin_lookup(mod):
        norm = mod.lower().replace("_", "-")
        if norm in harbin:
            return harbin[norm]
        for cand in ALIAS_DIST.get(norm, []):
            if cand in harbin:
                return harbin[cand]
        return ""

    def installed_check(mod):
        import importlib.util
        try:
            spec = importlib.util.find_spec(mod)
            return "yes" if spec is not None else "no"
        except Exception:
            return "no"

    rows = []
    for mod in sorted(records):
        r = records[mod]
        entries = sorted({h["entry"] for h in r["hits"]})
        first = min((h for h in r["hits"]), key=lambda h: sorted(entries).index(h["entry"]))
        top_or_lazy = "top" if "top" in r["scopes"] else "lazy"
        rows.append([mod, top_or_lazy, "|".join(entries), f"{first['file']}:{first['line']}",
                     installed_check(mod), harbin_lookup(mod)])

    with open(OUT / "reachability.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["module", "top_or_lazy", "starting_points", "first file:line",
                    "installed_on_beijing", "version_in_harbin_freeze"])
        for r in rows:
            w.writerow(r)

    with open(OUT / "reach_local_files.txt", "w") as fh:
        fh.write(f"union closure size: {len(union)} files\n\n")
        for label, v in per_entry.items():
            fh.write(f"### entry: {label}  (rounds={v['rounds']}, files={len(v['reachable'])})\n")
            fh.write(f"roots:\n")
            for r in v["roots"]:
                fh.write(f"  {r}\n")
            fh.write(f"reachable files:\n")
            for f in sorted(v["reachable"]):
                fh.write(f"  {f}\n")
            fh.write("\n")

    json_out = {}
    for mod, r in records.items():
        json_out[mod] = {"scopes": sorted(r["scopes"]), "hits": r["hits"]}
    with open(OUT / "reach_records.json", "w") as fh:
        json.dump(json_out, fh, indent=1, sort_keys=True)

    print("per-entry rounds/files:")
    for label, v in per_entry.items():
        print(f"  {label}: rounds={v['rounds']} files={len(v['reachable'])}")
    print(f"union files: {len(union)}")
    print(f"third-party modules: {len(records)}")
    print(f"top: {sum(1 for r in records.values() if 'top' in r['scopes'])}, lazy: {sum(1 for r in records.values() if 'top' not in r['scopes'])}")


if __name__ == "__main__":
    main()