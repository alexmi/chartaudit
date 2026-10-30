import pandas as pd

from chartaudit.calibrate import check_axis, implied_scale
from chartaudit.schema import ChartReading, Drawing


def bars(closes, start="2026-09-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c})


def test_equal_steps_are_linear_and_equal_ratios_are_log():
    assert implied_scale(["110", "105", "100", "95", "90"]) == "linear"
    assert implied_scale(["160", "80", "40", "20", "10"]) == "log"
    assert implied_scale(["100"]) is None


def test_reading_that_matches_real_prices_passes():
    r = ChartReading(y_axis_scale="linear", y_axis_ticks=["110", "105", "100", "95"],
                     drawings=[Drawing(type="horizontal_line", price=101.0)])
    out = check_axis(r, bars([98, 100, 102, 104, 103]), asof="2026-09-07")
    assert out["ok"] and out["flags"] == []


def test_log_axis_called_linear_is_flagged():
    r = ChartReading(y_axis_scale="linear", y_axis_ticks=["160", "80", "40", "20"])
    assert "scale: ticks look log but the reading says linear" in check_axis(r)["flags"]


def test_real_close_outside_the_axis_is_flagged():
    r = ChartReading(y_axis_scale="linear", y_axis_ticks=["30", "25", "20"])
    out = check_axis(r, bars([98, 100, 102]), asof="2026-09-03")
    assert not out["ok"] and any(f.startswith("price: real close 102") for f in out["flags"])


def test_drawing_off_the_axis_is_flagged():
    r = ChartReading(y_axis_ticks=["110", "105", "100"], drawings=[Drawing(type="zone", top=180.0, bottom=170.0)])
    assert any(f.startswith("drawing: zone") for f in check_axis(r)["flags"])


def test_unreadable_ticks_are_flagged():
    assert "ticks: fewer than 3 numeric labels" in check_axis(ChartReading(y_axis_ticks=["a", "1"]))["flags"]


def test_level_tags_mixed_into_ticks_do_not_change_the_scale():
    ticks = ["116.00", "112.56", "112.00", "108.00", "105.39", "104.00", "100.00", "96.00", "92.00"]
    assert implied_scale(ticks) == "linear"


def test_narrow_linear_axis_with_a_missing_label_is_linear():
    assert implied_scale(["115.00", "112.50", "110.00", "102.50", "100.00", "97.50"]) == "linear"
