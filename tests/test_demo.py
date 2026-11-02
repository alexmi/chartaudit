import io

import pytest
from PIL import Image

from chartaudit.demo import analyze, outcome_table, render_overlay
from chartaudit.prices import PriceError
from chartaudit.reader import ReadError
from chartaudit.schema import ChartReading, Drawing
from chartaudit.synth import make_bars

BARS = make_bars(seed=1, n=160, start_date="2026-03-02")
ASOF = BARS.index[119].strftime("%Y-%m-%d")
TODAY = BARS.index[-1].strftime("%Y-%m-%d")
CLOSE = float(BARS.close.iloc[119])
TICKS = [f"{CLOSE * k:.2f}" for k in (1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7)]


def _reading(**kw):
    base = dict(ticker="TEST", last_visible_date=ASOF, y_axis_scale="linear", y_axis_ticks=TICKS,
                drawings=[Drawing(type="horizontal_line", price=CLOSE * 0.95), Drawing(type="horizontal_line", price=CLOSE * 1.05)])
    return ChartReading(**(base | kw))


def _read(reading):
    return lambda png: reading


def test_analyze_audits_trader_levels_beside_baselines():
    out = analyze(b"png", _read(_reading()), lambda t: BARS, today=TODAY)
    assert out["ticker"] == "TEST" and out["asof"] == ASOF and out["axis"]["ok"]
    assert {r["source"] for r in out["rows"]} == {"trader", "pivot", "matched", "random"}
    assert sum(r["source"] == "trader" for r in out["rows"]) == 2


def test_visitor_ticker_and_date_override_the_reading():
    seen = []
    out = analyze(b"png", _read(_reading(ticker="WRONG", last_visible_date=None)),
                  lambda t: seen.append(t) or BARS, ticker="nasdaq:test", asof=ASOF, today=TODAY)
    assert seen == ["TEST"] and out["asof"] == ASOF and out["rows"]


def test_unsafe_ticker_never_reaches_the_price_lookup():
    seen = []
    out = analyze(b"png", _read(_reading(ticker="../../etc/passwd")), lambda t: seen.append(t) or BARS, today=TODAY)
    assert seen == [] and out["rows"] == []
    assert any("ticker" in n for n in out["notes"])


def test_price_errors_become_notes_and_the_reading_still_shows():
    def fail(t):
        raise PriceError("daily price lookup limit reached; showing the reading without the audit")

    out = analyze(b"png", _read(_reading()), fail, today=TODAY)
    assert out["reading"].drawings and out["rows"] == []
    assert any("daily price lookup limit" in n for n in out["notes"])


def test_unexpected_price_failures_show_a_generic_note():
    def boom(t):
        raise RuntimeError("https://www.alphavantage.co/query?apikey=SECRET timed out")

    out = analyze(b"png", _read(_reading()), boom, today=TODAY)
    assert out["rows"] == [] and not any("SECRET" in n for n in out["notes"])


def test_unreadable_chart_raises_read_error_for_the_page_to_show():
    def bad(png):
        raise ReadError("no JSON object in model output")

    with pytest.raises(ReadError):
        analyze(b"png", bad, lambda t: BARS, today=TODAY)


def test_outcome_table_counts_each_source():
    out = analyze(b"png", _read(_reading()), lambda t: BARS, today=TODAY)
    table = outcome_table(out["rows"])
    trader = next(r for r in table if r["source"] == "trader")
    assert trader["levels"] == 2 and trader["held"] + trader["broke"] + trader["untested"] == 2


@pytest.mark.parametrize("bars,asof", [(BARS, ASOF), (None, None)])
def test_render_overlay_returns_a_png(bars, asof):
    png = render_overlay(_reading(), bars, asof)
    assert Image.open(io.BytesIO(png)).format == "PNG"


def test_render_overlay_handles_an_empty_reading():
    png = render_overlay(ChartReading(), None, None)
    assert Image.open(io.BytesIO(png)).format == "PNG"
