import pytest

from chartaudit.extraction import score_reading, timeframe_equal
from chartaudit.schema import ChartReading, Drawing

TRUTH = ChartReading(
    ticker="SYN0001",
    timeframe="1D",
    y_axis_scale="linear",
    y_axis_ticks=["110", "105", "100", "95"],
    drawings=[
        Drawing(type="position_box", entry=100, stop=95, target=110),
        Drawing(type="horizontal_line", price=97.5),
        Drawing(type="horizontal_line", price=106.25),
    ],
)


def _with(drawings):
    return TRUTH.model_copy(update={"drawings": drawings})


def test_perfect_reading():
    s = score_reading(TRUTH, TRUTH)
    assert s["ticker_ok"] and s["timeframe_ok"] and s["ticks_all_ok"]
    assert s["found"] == s["drawn"] == 3
    assert s["worst_err_ticks"] == 0 and s["within_one_tick"] == s["prices_total"] == 5


def test_numeric_tick_compare():
    got = TRUTH.model_copy(update={"y_axis_ticks": ["29,868.00", "29,800"]})
    truth = TRUTH.model_copy(update={"y_axis_ticks": ["29868", "29800.0"]})
    assert score_reading(truth, got)["ticks_all_ok"]


def test_one_wrong_tick_fails_ticks():
    got = TRUTH.model_copy(update={"y_axis_ticks": ["110", "105", "100", "96"]})
    assert not score_reading(TRUTH, got)["ticks_all_ok"]


def test_box_off_by_three_ticks():
    got = _with([Drawing(type="position_box", entry=100.03, stop=95, target=110), *TRUTH.drawings[1:]])
    s = score_reading(TRUTH, got)
    assert s["worst_err_ticks"] == pytest.approx(3)
    assert s["within_one_tick"] == 4


def test_missing_and_null_fields():
    got = _with([Drawing(type="position_box", entry=100, stop=None, target=110), TRUTH.drawings[2]])
    s = score_reading(TRUTH, got)
    assert s["found"] == 2 and s["drawn"] == 3 and s["prices_total"] == 5
    assert s["within_one_tick"] == 3


def test_nothing_found():
    s = score_reading(TRUTH, _with([]))
    assert s["found"] == 0 and s["worst_err_ticks"] is None and s["within_one_tick"] == 0


def test_ticker_prefix_and_case():
    truth = TRUTH.model_copy(update={"ticker": "GC1!"})
    got = TRUTH.model_copy(update={"ticker": "COMEX:gc1!"})
    assert score_reading(truth, got)["ticker_ok"]


def test_timeframe_aliases():
    for a, b in [("1D", "D"), ("1D", "daily"), ("4h", "240"), ("1W", "weekly"), ("30m", "30")]:
        assert timeframe_equal(a, b)
    assert not timeframe_equal("1D", "4h")
    assert not timeframe_equal("1D", None)


def test_far_off_drawing_is_not_found():
    got = _with([Drawing(type="horizontal_line", price=500.0), Drawing(type="position_box")])
    truth = _with([Drawing(type="horizontal_line", price=100.0), TRUTH.drawings[0]])
    s = score_reading(truth, got)
    assert s["found"] == 0 and s["extra"] == 2


def test_extra_drawings_are_counted():
    got = _with([*TRUTH.drawings, Drawing(type="horizontal_line", price=101.0)])
    assert score_reading(TRUTH, got)["extra"] == 1


def test_level_tags_in_read_ticks_are_ignored():
    got = TRUTH.model_copy(update={"y_axis_ticks": ["110", "106.25", "105", "100", "97.50", "95"]})
    assert score_reading(TRUTH, got)["ticks_all_ok"]


def test_more_ticker_and_timeframe_forms():
    t = TRUTH.model_copy(update={"ticker": "BINANCE:BTCUSDT.P"})
    assert score_reading(t, TRUTH.model_copy(update={"ticker": "BTCUSDT"}))["ticker_ok"]
    for a, b in [("1D", "1 day"), ("4h", "4 hours"), ("1W", "1 week"), ("1D", "D1"), ("5h", "5h (300m)")]:
        assert timeframe_equal(a, b), (a, b)
    assert not timeframe_equal("1M", "1m")
