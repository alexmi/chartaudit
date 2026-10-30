"""Sanity checks on a chart reading before it is scored: is the price axis self-consistent, and does it
match the real price history for that ticker and date?

A misread axis (log taken for linear, wrong ticker, wrong date, split-adjusted data) would put every
drawn price in the wrong place, so a flagged chart is set aside instead of scored.
"""
import math
import re
import statistics

import pandas as pd

from chartaudit.schema import ChartReading

PRICE_FIELDS = ("price", "top", "bottom", "entry", "stop", "target")
STEADY = 0.08  # coefficient of variation below which steps (or ratios) count as equal
MARGIN = 0.05  # share of the axis span allowed outside it


def _numbers(ticks: list[str]) -> list[float]:
    out = []
    for t in ticks:
        m = re.search(r"-?\d[\d,]*\.?\d*", t or "")
        if m:
            out.append(float(m.group().replace(",", "")))
    return sorted(set(out))


def _steady(xs: list[float]) -> bool:
    mean = statistics.fmean(xs)
    return mean != 0 and statistics.pstdev(xs) / abs(mean) < STEADY


def _grid(v: list[float]) -> list[float]:
    """Drop price tags of drawn levels that sit between the axis labels: keep values on the most common step."""
    diffs = [round(b - a, 6) for a, b in zip(v, v[1:])]
    step = max(set(diffs), key=diffs.count)
    if diffs.count(step) < 2:
        return v
    base = next(a for a, d in zip(v, diffs) if d == step)
    return [x for x in v if abs((x - base) / step - round((x - base) / step)) < 0.02]


def implied_scale(ticks: list[str]) -> str | None:
    """'linear' if the labels step evenly, 'log' if they step by an even ratio, else None.
    Labels alone can't reveal a log axis printed at round values (88, 92, 96); that needs pixel positions."""
    v = _numbers(ticks)
    if len(v) >= 3:
        v = _grid(v)
    if len(v) < 3:
        return None
    diffs = [b - a for a, b in zip(v, v[1:])]
    step = min(diffs)
    if _steady(diffs):
        return "linear"
    if v[0] > 0 and _steady([math.log(b / a) for a, b in zip(v, v[1:])]):
        return "log"
    on_step = [abs(d / step - round(d / step)) < 0.02 for d in diffs]
    if all(on_step) and sum(abs(d - step) < 0.02 * step for d in diffs) * 2 >= len(diffs):
        return "linear"  # even steps with a label or two skipped
    return None


def check_axis(reading: ChartReading, bars: pd.DataFrame | None = None, asof=None) -> dict:
    flags = []
    v = _numbers(reading.y_axis_ticks)
    if len(v) < 3:
        return {"ok": False, "flags": ["ticks: fewer than 3 numeric labels"]}
    lo, hi = v[0], v[-1]
    pad = MARGIN * (hi - lo)
    scale = implied_scale(reading.y_axis_ticks)
    if scale and reading.y_axis_scale and scale != reading.y_axis_scale:
        flags.append(f"scale: ticks look {scale} but the reading says {reading.y_axis_scale}")
    for d in reading.drawings:
        for f in PRICE_FIELDS:
            p = getattr(d, f)
            if p is not None and not lo - pad <= p <= hi + pad:
                flags.append(f"drawing: {d.type} {f} {p:g} is off the axis {lo:g} to {hi:g}")
    if bars is not None and asof is not None:
        seen = bars.loc[:asof]
        if seen.empty:
            flags.append("price: no real bars on or before the chart date")
        else:
            close = float(seen.close.iloc[-1])
            if not lo - pad <= close <= hi + pad:
                flags.append(f"price: real close {close:g} on the chart date is off the axis {lo:g} to {hi:g}")
    return {"ok": not flags, "flags": flags, "implied_scale": scale}
