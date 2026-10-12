#!/usr/bin/env python3
"""T7: disk usage of every smoke cell (n=3), scaled to 200 images, total vs df.

    python adapters/diag/diag_t7_disk.py
"""
from __future__ import annotations

import os
import shutil

REPO = "/root/autodl-tmp/LGC_frozen"
R = os.path.join(REPO, "Results")

MAIN_ARCHS = ["resnet50", "vgg19", "vit_b_32"]
E10E11_EGC = ["resnet50_places365_non_targeted", "resnet50_places365_targeted",
              "densenet161_places365_non_targeted", "densenet161_places365_targeted",
              "Salman2020Do_R50_non_targeted", "Salman2020Do_R50_targeted"]
E10E11_BASE = ["resnet50_places365", "densenet161_places365", "Salman2020Do_R50"]


def du(dirpath: str) -> int:
    total = 0
    for dp, _dn, fns in os.walk(dirpath):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
    return total


def gib(b: int) -> str:
    return f"{b / 2**30:.3f} GiB"


def main() -> int:
    cells = []  # (name, bytes)
    for arch in MAIN_ARCHS:
        for mode in ("non_targeted", "targeted"):
            cells.append((f"egc {arch} {mode}", du(os.path.join(R, f"b4_smoke_egc_{arch}", mode))))
            cells.append((f"ttba {arch} {mode}", du(os.path.join(R, f"b4_smoke_ttba_{arch}", mode))))
            cells.append((f"arsopt {arch} {mode}", du(os.path.join(R, f"b4_smoke_arsopt_{arch}", mode))))
        cells.append((f"cgba {arch} non_targeted", du(os.path.join(R, f"b4_smoke_cgba_{arch}", "non_targeted"))))
        cells.append((f"cgba_h {arch} targeted", du(os.path.join(R, f"b4_smoke_cgba_{arch}", "targeted"))))

    for c in E10E11_EGC:
        cells.append((f"egc {c}", du(os.path.join(R, f"b4b_smoke_LGC_RC_EGC_{c}"))))
    for v in E10E11_BASE:
        cells.append((f"ttba {v} non_targeted", du(os.path.join(R, f"b4b_smoke_TtBA_{v}", "non_targeted"))))
        cells.append((f"ttba {v} targeted", du(os.path.join(R, f"b4b_smoke_TtBA_{v}", "targeted"))))
        cells.append((f"cgba {v} non_targeted", du(os.path.join(R, f"b4b_smoke_CGBA_{v}", "non_targeted"))))
        cells.append((f"cgba_h {v} targeted", du(os.path.join(R, f"b4b_smoke_CGBA_H_{v}", "targeted"))))

    n_main = 0
    n_e = 0
    tot_smoke = 0
    tot_scaled = 0
    print(f"{'cell':48s} {'n=3':>10s} {'x200':>10s}")
    print("-" * 72)
    for name, b in cells:
        is_main = name.split()[0] in ("egc", "ttba", "cgba", "cgba_h", "arsopt") and name.split()[1] in MAIN_ARCHS
        if is_main:
            n_main += 1
        else:
            n_e += 1
        tot_smoke += b
        scaled = b * 200 // 3
        tot_scaled += scaled
        print(f"{name:48s} {gib(b):>10s} {gib(scaled):>10s}")

    print("-" * 72)
    print(f"cells: main={n_main} E10/E11={n_e} total={n_main + n_e}")
    print(f"smoke total (n=3):   {gib(tot_smoke)}")
    print(f"scaled total (n=200): {gib(tot_scaled)}")
    du_ = shutil.disk_usage(R)
    print(f"\ndf: total={gib(du_.total)} used={gib(du_.used)} free={gib(du_.free)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
