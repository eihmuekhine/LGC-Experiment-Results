"""CPU unit tests for make_low_query_table.py (paper element E8_low_query_table).

Synthetic per_checkpoint + metrics CSVs with known exact answers.  Run:
    python tools/eval/tests_lowq/test_lowq.py   (exit 0 iff all pass)
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)

import make_low_query_table as T  # noqa: E402

RESULTS = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((name, bool(cond)))
    print(f"   {'PASS' if cond else 'FAIL'}  {name}")


def _ckpt_frame() -> pd.DataFrame:
    # arch=A, mode=non_targeted; methods O (ours) and B; images img0/img1/img2; budgets 100/200.
    rows = [
        # O: img0 found both; img1 found only at 200; img2 never found
        ("A", "non_targeted", "img0", "O", 100, 1, 1.0),
        ("A", "non_targeted", "img0", "O", 200, 1, 0.5),
        ("A", "non_targeted", "img1", "O", 100, 0, np.nan),
        ("A", "non_targeted", "img1", "O", 200, 1, 10.0),
        ("A", "non_targeted", "img2", "O", 100, 0, np.nan),
        ("A", "non_targeted", "img2", "O", 200, 0, np.nan),
        # B: img0 found both; img1 found both (huge); img2 never
        ("A", "non_targeted", "img0", "B", 100, 1, 2.0),
        ("A", "non_targeted", "img0", "B", 200, 1, 1.0),
        ("A", "non_targeted", "img1", "B", 100, 1, 10000.0),
        ("A", "non_targeted", "img1", "B", 200, 1, 10000.0),
        ("A", "non_targeted", "img2", "B", 100, 0, np.nan),
        ("A", "non_targeted", "img2", "B", 200, 0, np.nan),
    ]
    return pd.DataFrame(rows, columns=T.CKPT_COLS)


def _metrics_frame() -> pd.DataFrame:
    rows = [
        ("A", "non_targeted", "img0", "O", 100, 1, 0.10),
        ("A", "non_targeted", "img0", "O", 200, 1, 0.05),
        ("A", "non_targeted", "img1", "O", 200, 1, 0.20),
        ("A", "non_targeted", "img0", "B", 100, 1, 0.15),
        ("A", "non_targeted", "img0", "B", 200, 1, 0.04),
        ("A", "non_targeted", "img1", "B", 100, 1, 0.30),
        ("A", "non_targeted", "img1", "B", 200, 1, 0.25),
    ]
    return pd.DataFrame(rows, columns=T.METRIC_COLS)


def _close(a, b, tol=1e-9) -> bool:
    return bool(np.isclose(float(a), float(b), atol=tol, rtol=1e-12))


def test_compute() -> None:
    df = T.compute(_ckpt_frame(), _metrics_frame(), [100, 200], [0.1, 0.05, 0.01])

    def row(method, budget):
        r = df[(df.method == method) & (df.budget == budget)]
        assert len(r) == 1, f"{method}@{budget}: {len(r)} rows"
        return r.iloc[0]

    o100 = row("O", 100)
    check("O@100 n", int(o100["n"]) == 3)
    check("O@100 found_rate 33.33", _close(o100["found_rate"], 100.0 * 1 / 3))
    check("O@100 asr_1/asr_2/asr_3 all 33.33", all(_close(o100[f"asr_{i}"], 100.0 / 3) for i in (1, 2, 3)))
    check("O@100 med_l2 1.0", _close(o100["med_l2"], 1.0))
    check("O@100 med_lpips 0.10", _close(o100["med_lpips"], 0.10))

    o200 = row("O", 200)
    check("O@200 found_rate 66.67", _close(o200["found_rate"], 100.0 * 2 / 3))
    check("O@200 asr_1/asr_2 66.67", _close(o200["asr_1"], 200.0 / 3) and _close(o200["asr_2"], 200.0 / 3))
    check("O@200 asr_3 33.33 (10.0 pixel fails RMSE 0.01)", _close(o200["asr_3"], 100.0 / 3))
    check("O@200 med_l2 5.25", _close(o200["med_l2"], 5.25))
    check("O@200 med_lpips 0.125", _close(o200["med_lpips"], 0.125))

    b100 = row("B", 100)
    check("B@100 found_rate 66.67", _close(b100["found_rate"], 200.0 / 3))
    check("B@100 asr all 33.33 (10000 pixel fails every threshold)", all(_close(b100[f"asr_{i}"], 100.0 / 3) for i in (1, 2, 3)))
    check("B@100 med_l2 5001.0", _close(b100["med_l2"], 5001.0))
    check("B@100 med_lpips 0.225", _close(b100["med_lpips"], 0.225))

    b200 = row("B", 200)
    check("B@200 med_l2 5000.5", _close(b200["med_l2"], 5000.5))
    check("B@200 med_lpips 0.145", _close(b200["med_lpips"], 0.145))


def test_loading_and_stemming() -> None:
    # per_checkpoint uses ".JPEG" image names (ours) while the metrics CSV uses stems; loading must normalise.
    ckpt = _ckpt_frame().copy()
    ckpt["image"] = ckpt["image"].map(lambda s: s + ".JPEG" if s == "img0" or s == "img2" else s)
    with tempfile.TemporaryDirectory() as d:
        ck_path = os.path.join(d, "per_checkpoint.csv")
        met_path = os.path.join(d, "metrics.csv")
        ckpt.to_csv(ck_path, index=False)
        _metrics_frame().to_csv(met_path, index=False)
        ck = T.load_per_checkpoint([os.path.join(d, "..")])  # root = parent of the CSV dir is NOT used; use direct
        # load_per_checkpoint globs <root>/*/per_checkpoint*.csv, so lay the CSV under <d>/mode/.
        md = os.path.join(d, "non_targeted")
        os.makedirs(md, exist_ok=True)
        os.replace(ck_path, os.path.join(md, "per_checkpoint.csv"))
        os.remove(met_path)
        _metrics_frame().to_csv(os.path.join(d, "metrics.csv"), index=False)
        ck = T.load_per_checkpoint([d])
        met = T.load_metrics([os.path.join(d, "metrics.csv")])
    check("load_per_checkpoint stems .JPEG -> img0", "img0" in set(ck["image"]) and "img0.JPEG" not in set(ck["image"]))
    check("load_per_checkpoint has 12 rows", len(ck) == 12)
    df = T.compute(ck, met, [100, 200], [0.1, 0.05, 0.01])
    o200 = df[(df.method == "O") & (df.budget == 200)].iloc[0]
    check("stemmed load gives O@200 med_l2 5.25", _close(o200["med_l2"], 5.25))


def test_adv_flag_zero_is_failure() -> None:
    ckpt = _ckpt_frame().copy()
    met = _metrics_frame().copy()
    # make img0 (found=1, small pixel L2) adv_flag=0 for O@100 -> must count as a failure for ASR
    met.loc[(met.image == "img0") & (met.method == "O") & (met.budget == 100), "adv_flag"] = 0
    df = T.compute(ckpt, met, [100, 200], [0.1, 0.05, 0.01])
    o100 = df[(df.method == "O") & (df.budget == 100)].iloc[0]
    check("O@100 adv_flag=0 -> asr_1 = 0", _close(o100["asr_1"], 0.0))
    check("O@100 found_rate still 33.33", _close(o100["found_rate"], 100.0 / 3))
    check("O@100 med_l2 still 1.0 (found rows only)", _close(o100["med_l2"], 1.0))


def test_render_order_and_bold() -> None:
    df = T.compute(_ckpt_frame(), _metrics_frame(), [100, 200], [0.1, 0.05, 0.01])
    tex = T.render(df, ["O", "B"], {}, [100, 200], [0.1, 0.05, 0.01], "A", "non_targeted")
    check("ours listed first", tex.splitlines()[0].startswith(r"\begin{table*}[t]") and
          "O &" in tex and tex.find("O &") < tex.find("B &"))
    check("caption prints n", "n = O:3, B:3" in tex)
    # direction-aware bolding: O has the smaller med_l2 at 100 (1.0 < 5001.0) -> O cell bolded
    check("med_l2 best (min) bolded for O@100", r"\textbf{1.00}" in tex)
    # found_rate at 200: O and B tie at 66.7% -> the max picks one (bolded); the value must appear bolded once
    check("a found-rate value is bolded at 200", r"\textbf{66.7\%}" in tex)


def main() -> int:
    test_compute()
    test_loading_and_stemming()
    test_adv_flag_zero_is_failure()
    test_render_order_and_bold()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{'ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad)}  ({len(RESULTS) - len(bad)}/{len(RESULTS)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
