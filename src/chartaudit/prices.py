"""Daily price history for the audit: local CSV files first, Alpha Vantage only to fill gaps.

Any CSV with Date, Open, High, Low, Close columns works (a broker export, for example).
Drop it in the price folder as <TICKER>.csv. Fetched history is saved there too, so each
ticker is fetched once (the free Alpha Vantage key allows about 25 calls a day).
"""
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

AV_URL = "https://www.alphavantage.co/query"
COLS = ["open", "high", "low", "close"]


class PriceError(Exception):
    pass


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].astype(float).sort_index()
    df.index.name = "date"
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.index = pd.to_datetime(df.pop("date"))
    return _clean(df)


def from_alpha_vantage(payload: dict) -> pd.DataFrame:
    series = payload.get("Time Series (Daily)")
    if not series:
        msg = payload.get("Information") or payload.get("Note") or payload.get("Error Message") or "no data"
        kind = "rate limit or plan" if ("rate limit" in msg or "premium" in msg) else "error"
        raise PriceError(f"Alpha Vantage {kind}: {msg}")
    df = pd.DataFrame.from_dict(series, orient="index").rename(columns=lambda c: c.split(". ")[1])
    df.index = pd.to_datetime(df.index)
    return _clean(df)


def fetch_alpha_vantage(symbol: str) -> pd.DataFrame:
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise PriceError("ALPHAVANTAGE_API_KEY not set in .env")
    query = urllib.parse.urlencode(
        {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full", "apikey": key}
    )
    with urllib.request.urlopen(f"{AV_URL}?{query}", timeout=30) as r:
        return from_alpha_vantage(json.load(r))


def get_bars(
    ticker: str, price_dir: Path, fetch: Callable[[str], pd.DataFrame] | None = fetch_alpha_vantage
) -> pd.DataFrame:
    symbol = ticker.strip().upper()
    path = Path(price_dir) / f"{symbol}.csv"
    if path.exists():
        return load_csv(path)
    if fetch is None:
        raise PriceError(f"no price file {path} and fetching is off")
    bars = fetch(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path)
    return load_csv(path)
