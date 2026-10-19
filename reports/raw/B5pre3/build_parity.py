import csv
import re
import sys

HIGHLIGHT = {"torch", "torchvision", "numpy", "pillow", "scipy", "pandas", "triton"}


def norm_name(name):
    return name.lower().replace("_", "-")


def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("-e ") or line.startswith("--editable"):
        return None
    m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
    if not m:
        return None
    name = norm_name(m.group(1))
    if "==" in line:
        version = line.split("==", 1)[1]
    elif " @ " in line:
        version = "(direct-url) " + line.split(" @ ", 1)[1]
    elif re.match(r"^[A-Za-z0-9_.\-]+$", line):
        version = "(unpinned)"
    else:
        version = line
    return name, version


def load(path):
    d = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0 and line.strip().startswith("Python "):
                continue
            p = parse_line(line)
            if p:
                d[p[0]] = p[1]
    return d


def is_highlight(name):
    if name in HIGHLIGHT:
        return "X"
    if name.startswith("nvidia-"):
        return "X"
    return ""


harbin = load(sys.argv[1])
beijing = load(sys.argv[2])

rows = []
keys = sorted(set(harbin) | set(beijing))
for k in keys:
    h, b = harbin.get(k), beijing.get(k)
    if h is None:
        status = "ONLY_ON_BEIJING"
    elif b is None:
        status = "MISSING_ON_BEIJING"
    elif h == b:
        status = "SAME"
    else:
        status = "VERSION_DIFF"
    rows.append([k, h or "", b or "", status, is_highlight(k)])

rows.sort(key=lambda r: (r[3] != "SAME", r[0]))

with open(sys.argv[3], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["package", "harbin_version", "beijing_version", "status", "highlight"])
    for r in rows:
        w.writerow(r)
print("rows:", len(rows))