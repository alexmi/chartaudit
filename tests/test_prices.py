import os

import pandas as pd
import pytest

from chartaudit.prices import PriceError, from_alpha_vantage, get_bars, load_csv

AV = {
    "Meta Data": {},
    "Time Series (Daily)": {
        "2026-10-28": {"1. open": "11", "2. high": "12", "3. low": "10", "4. close": "11.5", "5. volume": "9"},
        "2026-10-27": {"1. open": "10", "2. high": "11", "3. low": "9", "4. close": "10.5", "5. volume": "9"},
    },
}


def test_alpha_vantage_json_becomes_sorted_bars():
    b = from_alpha_vantage(AV)
    assert list(b.columns) == ["open", "high", "low", "close"]
    assert b.index[0] == pd.Timestamp("2026-10-27") and b.close.iloc[-1] == 11.5


def test_alpha_vantage_limit_message_raises():
    with pytest.raises(PriceError, match="rate limit"):
        from_alpha_vantage({"Information": "Our standard API rate limit is 25 requests per day."})


def test_load_csv_accepts_broker_style_headers(tmp_path):
    p = tmp_path / "X.csv"
    p.write_text("Date,Open,High,Low,Close,Volume\n10/28/2026,11,12,10,11.5,9\n10/27/2026,10,11,9,10.5,9\n")
    b = load_csv(p)
    assert b.index.is_monotonic_increasing and b.high.iloc[0] == 11


def test_get_bars_uses_cache_then_fetches_once(tmp_path):
    calls = []

    def fetch(symbol):
        calls.append(symbol)
        return from_alpha_vantage(AV)

    a = get_bars("aapl", tmp_path, fetch=fetch)
    b = get_bars("AAPL", tmp_path, fetch=fetch)
    assert calls == ["AAPL"] and (tmp_path / "AAPL.csv").exists()
    assert a.equals(b)


def test_stale_cache_is_refetched_and_merged(tmp_path):
    old = tmp_path / "AAPL.csv"
    old.write_text("date,open,high,low,close\n2026-10-20,9,10,8,9.5\n2026-10-27,1,1,1,1\n")
    os.utime(old, (0, 0))  # very old file
    b = get_bars("AAPL", tmp_path, fetch=lambda s: from_alpha_vantage(AV))
    assert list(b.index.strftime("%m-%d")) == ["10-20", "10-27", "10-28"]
    assert b.close.loc["2026-10-27"] == 10.5  # fresh data wins on overlap


def test_get_bars_without_cache_or_fetcher_explains(tmp_path):
    with pytest.raises(PriceError, match="no price file"):
        get_bars("ZZZ", tmp_path, fetch=None)


def test_error_messages_never_carry_the_key():
    msg = {"Information": "We have detected your API key as SECRET123 and our standard API rate limit is 25 requests per day."}
    with pytest.raises(PriceError) as e:
        from_alpha_vantage(msg, redact="SECRET123")
    assert "SECRET123" not in str(e.value) and "rate limit" in str(e.value)
