import io

import pandas as pd
import pytest
from PIL import Image

from chartaudit.live import (
    FetchBudget, Limits, UploadError, clean_date, clean_ticker, prepare_image, visitor_id,
)
from chartaudit.prices import PriceError


def _png(w=200, h=100, fmt="PNG", **save_kw) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, fmt, **save_kw)
    return buf.getvalue()


# uploads

def test_prepare_image_returns_a_png():
    out = prepare_image(_png(fmt="JPEG"))
    assert Image.open(io.BytesIO(out)).format == "PNG"


def test_prepare_image_rejects_files_over_the_size_limit():
    with pytest.raises(UploadError, match="too large"):
        prepare_image(_png(), max_bytes=100)


def test_prepare_image_rejects_non_images():
    with pytest.raises(UploadError, match="not an image"):
        prepare_image(b"<script>alert(1)</script>")


def test_prepare_image_rejects_other_formats():
    with pytest.raises(UploadError, match="PNG, JPEG or WebP"):
        prepare_image(_png(fmt="GIF"))


def test_prepare_image_rejects_huge_pixel_counts():
    with pytest.raises(UploadError, match="too many pixels"):
        prepare_image(_png(400, 400), max_pixels=100_000)


def test_prepare_image_shrinks_to_max_side():
    out = Image.open(io.BytesIO(prepare_image(_png(3000, 1000), max_side=1500)))
    assert out.size == (1500, 500)


def test_prepare_image_drops_metadata():
    exif = Image.Exif()
    exif[0x010E] = "secret location"  # ImageDescription
    out = Image.open(io.BytesIO(prepare_image(_png(fmt="JPEG", exif=exif))))
    assert not out.getexif() and not out.info.get("exif")


# tickers and dates become file names and API parameters

@pytest.mark.parametrize("raw,want", [("aapl", "AAPL"), (" BRK.B ", "BRK.B"), ("NASDAQ:NVDA", "NVDA"), ("$TSLA", "TSLA")])
def test_clean_ticker_accepts_plain_symbols(raw, want):
    assert clean_ticker(raw) == want


@pytest.mark.parametrize("raw", [None, "", "../../etc/passwd", "A/B", "A B", "X" * 11, "a;rm"])
def test_clean_ticker_rejects_anything_else(raw):
    assert clean_ticker(raw) is None


def test_clean_date_parses_printed_dates():
    assert clean_date("28-Oct-2026", today="2026-11-02") == "2026-10-28"


@pytest.mark.parametrize("raw", [None, "", "not a date", "2026-12-01", "1990-01-01"])
def test_clean_date_rejects_future_ancient_and_garbage(raw):
    assert clean_date(raw, today="2026-11-02") is None


# limits

class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_limits_cap_each_visitor_per_window():
    clock = Clock()
    lim = Limits(per_visitor=2, window_s=3600, per_day=100, clock=clock)
    assert lim.take("a") is None and lim.take("a") is None
    assert "per hour" in lim.take("a")
    assert lim.take("b") is None  # other visitors are not affected
    clock.t += 3601
    assert lim.take("a") is None


def test_limits_cap_the_whole_page_per_day():
    clock = Clock()
    lim = Limits(per_visitor=100, window_s=3600, per_day=2, clock=clock)
    assert lim.take("a") is None and lim.take("b") is None
    assert "today" in lim.take("c")
    clock.t += 86400
    assert lim.take("c") is None


def test_limits_turned_off_refuses_everyone():
    lim = Limits(per_visitor=5, window_s=3600, per_day=100, clock=Clock(), enabled=False)
    assert "off" in lim.take("a")


def test_visitor_id_uses_the_first_forwarded_address():
    assert visitor_id({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.9") == "1.2.3.4"
    assert visitor_id({}, "5.6.7.8") == "5.6.7.8"
    assert visitor_id({}, None) == "unknown"


# price lookups share one small daily allowance

def _bars():
    return pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=pd.to_datetime(["2026-10-28"]))


def test_fetch_budget_stops_after_the_daily_allowance():
    calls, clock = [], Clock()
    budget = FetchBudget(lambda s: calls.append(s) or _bars(), per_day=2, clock=clock)
    budget("A"), budget("B")
    with pytest.raises(PriceError, match="daily price lookup limit"):
        budget("C")
    assert calls == ["A", "B"]
    clock.t += 86400
    budget("C")
    assert calls == ["A", "B", "C"]


def test_fetch_budget_stops_for_the_day_when_the_provider_says_rate_limit():
    calls = []

    def fetch(s):
        calls.append(s)
        raise PriceError("Alpha Vantage rate limit or plan: 25 requests per day")

    budget = FetchBudget(fetch, per_day=10, clock=Clock())
    with pytest.raises(PriceError):
        budget("A")
    with pytest.raises(PriceError, match="daily price lookup limit"):
        budget("B")
    assert calls == ["A"]
