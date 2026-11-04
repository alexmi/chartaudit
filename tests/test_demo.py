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


# the page's request handler

from chartaudit.demo import handle_upload  # noqa: E402
from chartaudit.live import Limits  # noqa: E402


def _upload(tmp_path, data=None):
    p = tmp_path / "up.png"
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buf, "PNG")
    p.write_bytes(data if data is not None else buf.getvalue())
    return p


def _limits(**kw):
    return Limits(**({"per_visitor": 5, "window_s": 3600, "per_day": 100} | kw))


def test_handle_upload_returns_overlay_table_reading_and_notes(tmp_path):
    overlay, table, reading, notes = handle_upload(
        str(_upload(tmp_path)), "", "", "v", _read(_reading()), lambda t: BARS, _limits(), today=TODAY)
    assert Image.open(io.BytesIO(overlay)).format == "PNG"
    assert table[0]["source"] == "trader" and reading["ticker"] == "TEST"


def test_handle_upload_deletes_the_uploaded_file(tmp_path):
    p = _upload(tmp_path)
    handle_upload(str(p), "", "", "v", _read(_reading()), lambda t: BARS, _limits(), today=TODAY)
    assert not p.exists()


def test_bad_uploads_do_not_use_the_reading_allowance(tmp_path):
    lim, calls = _limits(per_day=1), []
    reader = lambda png: calls.append(1) or _reading()  # noqa: E731
    *_, notes = handle_upload(str(_upload(tmp_path, b"not an image")), "", "", "v", reader, lambda t: BARS, lim, today=TODAY)
    assert "not an image" in notes and calls == []
    handle_upload(str(_upload(tmp_path)), "", "", "v", reader, lambda t: BARS, lim, today=TODAY)
    assert calls == [1]


def test_over_the_limit_the_model_is_not_called(tmp_path):
    lim, calls = _limits(per_day=0), []
    reader = lambda png: calls.append(1) or _reading()  # noqa: E731
    overlay, _, _, notes = handle_upload(str(_upload(tmp_path)), "", "", "v", reader, lambda t: BARS, lim, today=TODAY)
    assert overlay is None and calls == [] and "allowance" in notes


def test_model_failures_show_a_plain_message(tmp_path):
    def boom(png):
        raise RuntimeError("401 Unauthorized for key sk-SECRET")

    overlay, _, _, notes = handle_upload(str(_upload(tmp_path)), "", "", "v", boom, lambda t: BARS, _limits(), today=TODAY)
    assert overlay is None and "SECRET" not in notes and "try again" in notes


def test_no_file_asks_for_one():
    *_, notes = handle_upload(None, "", "", "v", _read(_reading()), lambda t: BARS, _limits())
    assert "Upload" in notes


def test_box_results_are_described_in_the_notes():
    from chartaudit.demo import box_notes
    rows = [{"source": "trader", "kind": "box", "side": "long", "outcome": "win", "r": 2.0, "bars_to_exit": 7},
            {"source": "trader", "kind": "box", "side": "short", "outcome": "not_triggered", "r": None, "bars_to_exit": None},
            {"source": "trader", "kind": "level", "outcome": "held"}]
    assert box_notes(rows) == ["Position box (long): hit the target after 7 bars, +2.0R.",
                               "Position box (short): entry never reached in the 20 bars after the chart."]


def test_overlay_has_a_legend_for_what_was_drawn():
    from chartaudit.demo import overlay_figure
    reading = _reading(drawings=[Drawing(type="horizontal_line", price=CLOSE * 0.95), Drawing(type="horizontal_line", price=CLOSE),
                                 Drawing(type="position_box", entry=CLOSE, stop=CLOSE * 0.97, target=CLOSE * 1.06)])
    fig = overlay_figure(reading, BARS, ASOF)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["close", "level read", "box entry", "box stop", "box target"]
    assert "chart date" not in fig.axes[0].get_title()
