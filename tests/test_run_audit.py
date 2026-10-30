import pandas as pd

from chartaudit.prices import PriceError
from chartaudit.schema import ChartReading, Drawing
from run_audit import audit_charts, summarize, wilson


def fake_bars(symbol):
    if symbol == "NOPE":
        raise PriceError("no data")
    n = 200
    close = pd.Series([100 + (i % 10) for i in range(n)], dtype=float)  # saw tooth between 100 and 109
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"open": close.values, "high": close.values + 0.5, "low": close.values - 0.5, "close": close.values}, index=idx)


READING = ChartReading(drawings=[
    Drawing(type="horizontal_line", price=100.0),
    Drawing(type="horizontal_line", price=None),
    Drawing(type="position_box", entry=104, stop=101, target=109),
    Drawing(type="zone", top=105, bottom=103),
])


def test_audit_rows_trader_levels_boxes_and_baselines():
    charts = [{"id": "A1", "ticker": "AAA", "asof": "2026-03-02"}]
    rows, skipped = audit_charts({"A1": READING}, charts, fake_bars, horizon=20, lookback=60, random_per_level=5, seed=0)
    assert skipped == []
    by = lambda src, kind: [r for r in rows if r["source"] == src and r["kind"] == kind]  # noqa: E731
    assert len(by("trader", "level")) == 1 and by("trader", "level")[0]["price"] == 100.0
    assert len(by("trader", "box")) == 1
    assert len(by("pivot", "level")) == 1 and len(by("random", "level")) == 5
    assert all(r["chart"] == "A1" and r["asof"] == "2026-03-02" for r in rows)


def test_charts_without_reading_or_prices_are_skipped_with_reason():
    charts = [{"id": "A1", "ticker": "NOPE", "asof": "2026-03-02"}, {"id": "B2", "ticker": "AAA", "asof": "2026-03-02"}]
    rows, skipped = audit_charts({"A1": READING}, charts, fake_bars)
    assert rows == []
    assert [s["id"] for s in skipped] == ["A1", "B2"]
    assert "no data" in skipped[0]["reason"] and "no reading" in skipped[1]["reason"]


def test_wilson_interval():
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert wilson(0, 0) == (None, None)


def test_summary_reports_hold_rate_by_source_and_touch_bucket():
    rows = [
        {"source": "trader", "kind": "level", "outcome": "held", "touches_before": 3},
        {"source": "trader", "kind": "level", "outcome": "broke", "touches_before": 1},
        {"source": "random", "kind": "level", "outcome": "untested", "touches_before": 0},
        {"source": "trader", "kind": "box", "outcome": "win", "r": 2.0},
    ]
    text = summarize(rows, skipped=[{"id": "X", "reason": "no data"}])
    assert "| trader | 2 | 1 | 1 | 50% |" in text
    assert "| 3+ | 1 | 1 | 100% |" in text
    assert "Boxes: 1 win, 0 loss" in text
    assert "X: no data" in text
