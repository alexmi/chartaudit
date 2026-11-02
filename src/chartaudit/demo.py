"""One chart, end to end, for the demo page: read it, check the axis, audit it if prices are at hand,
and draw what was read over the real prices (or over the chart's own axis when there are none).

The reader and the price lookup are passed in, so the page decides keys and limits, and tests need neither.
"""
import io
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from chartaudit.audit import HORIZON, LOOKBACK, audit_charts
from chartaudit.calibrate import _numbers, check_axis
from chartaudit.live import Limits, UploadError, clean_date, clean_ticker, prepare_image
from chartaudit.prices import PriceError
from chartaudit.reader import ReadError
from chartaudit.schema import ChartReading

SOURCES = ("trader", "pivot", "matched", "random")
log = logging.getLogger(__name__)


def analyze(png: bytes, read: Callable[[bytes], ChartReading], bars_for: Callable[[str], pd.DataFrame],
            ticker: str | None = None, asof: str | None = None, today: str | None = None) -> dict:
    reading = read(png)
    out = {"reading": reading, "ticker": None, "asof": None, "axis": check_axis(reading), "bars": None, "rows": [], "notes": []}
    t = clean_ticker(ticker) or clean_ticker(reading.ticker)
    d = clean_date(asof, today) or clean_date(reading.last_visible_date, today)
    out["ticker"], out["asof"] = t, d
    if not t:
        out["notes"].append("No usable ticker: type it in (letters, digits, dot or dash) to run the audit.")
    if not d:
        out["notes"].append("No usable chart date: type it in (within the last two years) to run the audit.")
    if not (t and d):
        return out
    try:
        bars = bars_for(t)
    except PriceError as e:
        out["notes"].append(f"No price data for {t}: {e}")
        return out
    except Exception:  # never show raw errors: they can carry URLs and keys
        out["notes"].append(f"No price data for {t} right now.")
        return out
    out["bars"] = bars
    out["axis"] = check_axis(reading, bars, d)
    rows, skipped = audit_charts({"upload": reading}, [{"id": "upload", "ticker": t, "asof": d}], lambda _: bars)
    out["rows"] = rows
    out["notes"] += [s["reason"] for s in skipped]
    if rows and all(r.get("outcome") == "pending" for r in rows if r["source"] == "trader"):
        out["notes"].append("The chart is too recent: no bars after it yet, so every level is pending.")
    return out


def outcome_table(rows: list[dict]) -> list[dict]:
    table = []
    for src in SOURCES:
        levels = [r for r in rows if r["source"] == src and r["kind"] == "level"]
        if not levels:
            continue
        n = {k: sum(r["outcome"] == k for r in levels) for k in ("held", "broke", "untested", "pending")}
        tested = n["held"] + n["broke"]
        table.append({"source": src, "levels": len(levels), **n,
                      "hold rate": f"{n['held'] / tested:.0%}" if tested else "n/a"})
    return table


def render_overlay(reading: ChartReading, bars: pd.DataFrame | None, asof: str | None, horizon: int = HORIZON) -> bytes:
    fig = Figure(figsize=(9, 4.5), dpi=110)
    ax = fig.add_subplot()
    if bars is not None and asof is not None:
        end = len(bars.loc[:asof])
        win = bars.iloc[max(0, end - LOOKBACK): end + horizon]
        ax.plot(win.index, win.close, color="#444", lw=1)
        ax.axvline(pd.Timestamp(asof), color="#888", ls="--", lw=1)
        ax.text(pd.Timestamp(asof), 1.0, " chart date", transform=ax.get_xaxis_transform(), va="bottom", fontsize=8, color="#666")
        ax.set_title("What the model read, drawn on real prices (scored only after the dashed line)", fontsize=10)
    else:
        ticks = _numbers(reading.y_axis_ticks)
        if len(ticks) >= 2:
            ax.set_ylim(min(ticks), max(ticks))
        ax.set_xticks([])
        ax.set_title("What the model read (no price data, so no audit)", fontsize=10)
    for d in reading.drawings:
        if d.type == "horizontal_line" and d.price is not None:
            ax.axhline(d.price, color="#1f6feb", lw=1.5, ls=":" if d.estimated else "-")
        elif d.type == "zone" and None not in (d.top, d.bottom):
            ax.axhspan(min(d.top, d.bottom), max(d.top, d.bottom), color="#1f6feb", alpha=0.15)
        elif d.type == "position_box" and None not in (d.entry, d.stop, d.target):
            ax.axhline(d.entry, color="#555", lw=1)
            ax.axhline(d.stop, color="#d1242f", lw=1)
            ax.axhline(d.target, color="#1a7f37", lw=1)
    if not reading.drawings:
        ax.text(0.5, 0.5, "no drawings found", transform=ax.transAxes, ha="center", color="#888")
    fig.tight_layout()
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()


def handle_upload(path: str | None, ticker: str, asof: str, visitor: str,
                  read: Callable[[bytes], ChartReading], bars_for: Callable[[str], pd.DataFrame],
                  limits: Limits, today: str | None = None) -> tuple[bytes | None, list[dict], dict | None, str]:
    """The page's one action: (overlay PNG, outcome table, reading, notes). The upload is deleted at once,
    checked before it counts against the reading allowance, and errors are shown without their details."""
    if not path:
        return None, [], None, "Upload a chart screenshot (PNG, JPEG or WebP)."
    try:
        data = Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)
    try:
        png = prepare_image(data)
    except UploadError as e:
        return None, [], None, f"Upload refused: {e}."
    refused = limits.take(visitor)
    if refused:
        return None, [], None, refused
    try:
        out = analyze(png, read, bars_for, ticker=ticker or None, asof=asof or None, today=today)
    except ReadError:
        return None, [], None, "The model could not read this as a price chart; try a clearer screenshot."
    except Exception as e:
        log.warning("reading failed: %s", type(e).__name__)  # the type only: messages can carry keys
        return None, [], None, "Reading failed on our side; please try again later."
    notes = out["notes"] + [f"Axis check: {f}" for f in out["axis"].get("flags", [])]
    overlay = render_overlay(out["reading"], out["bars"], out["asof"])
    return overlay, outcome_table(out["rows"]), out["reading"].model_dump(), "\n".join(notes) or "Done."
