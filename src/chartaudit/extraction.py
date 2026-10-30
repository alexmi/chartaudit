"""Score a ChartReading against the truth: did the model read the chart and the drawings correctly?"""
import re

from chartaudit.schema import ChartReading, Drawing

PRICE_FIELDS = {
    "position_box": ("entry", "stop", "target"),
    "horizontal_line": ("price",),
    "zone": ("top", "bottom"),
}
UNITS = {
    "": 1, "m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
    "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60, "hourly": 60,
    "d": 1440, "day": 1440, "days": 1440, "daily": 1440,
    "w": 10080, "wk": 10080, "week": 10080, "weeks": 10080, "weekly": 10080,
    "mo": 43200, "month": 43200, "months": 43200, "monthly": 43200,
}
MATCH_TOL = 0.02  # a read drawing counts as found only within 2% of the true price


def _ticker(t: str | None) -> str | None:
    if not t:
        return None
    s = t.strip().split(":")[-1].replace("!", "").upper()
    return s[:-2] if s.endswith(".P") else s


def _minutes(tf: str | None) -> int | None:
    """'4h', '240', '1 day', 'D1', 'weekly', '5h (300m)' -> minutes. 'M' alone means month, as on TradingView."""
    if not tf:
        return None
    s = tf.strip().split("(")[0].strip()
    m = re.fullmatch(r"(\d*)\s*([A-Za-z]*)", s) or re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if not m:
        return None
    num, unit = m.groups() if m.group(1).isdigit() or m.group(1) == "" else m.groups()[::-1]
    unit = "mo" if unit == "M" else unit.lower()
    return int(num or 1) * UNITS[unit] if unit in UNITS else None


def timeframe_equal(a: str | None, b: str | None) -> bool:
    ma, mb = _minutes(a), _minutes(b)
    return ma is not None and ma == mb


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None


def _ticks_equal(truth: list[str], got: list[str], tags: list[float], tick: float) -> bool:
    """Axis labels match as numbers; level tags the model listed as axis labels are ignored."""
    a = [_num(x) for x in truth]
    b = [
        v for v in (_num(x) for x in got)
        if v is None or v in a or not any(abs(v - t) <= tick for t in tags)
    ]
    return len(a) == len(b) and None not in a + b and all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def _key(d: Drawing) -> float | None:
    return next((getattr(d, f) for f in PRICE_FIELDS.get(d.type, ()) if getattr(d, f) is not None), None)


def _match(truth: list[Drawing], got: list[Drawing]) -> list[tuple[Drawing, Drawing]]:
    """Pair truth and read drawings of the same type, closest key prices first."""
    pairs = []
    for ti, t in enumerate(truth):
        for gi, g in enumerate(got):
            if t.type == g.type:
                kt, kg = _key(t), _key(g)
                if kt is not None and kg is not None and abs(kt - kg) <= MATCH_TOL * abs(kt):
                    pairs.append((abs(kt - kg), ti, gi))
    used_t, used_g, out = set(), set(), []
    for _, ti, gi in sorted(pairs):
        if ti not in used_t and gi not in used_g:
            used_t.add(ti)
            used_g.add(gi)
            out.append((truth[ti], got[gi]))
    return out


def score_reading(truth: ChartReading, got: ChartReading, tick: float = 0.01) -> dict:
    matched = _match(truth.drawings, got.drawings)
    pairs = [
        (getattr(t, f), getattr(g, f), bool(g.estimated))
        for t, g in matched
        for f in PRICE_FIELDS.get(t.type, ())
        if getattr(t, f) is not None and getattr(g, f) is not None
    ]
    errs = [abs(tv - gv) / tick for tv, gv, _ in pairs]
    return {
        "ticker_ok": _ticker(truth.ticker) is not None and _ticker(truth.ticker) == _ticker(got.ticker),
        "timeframe_ok": timeframe_equal(truth.timeframe, got.timeframe),
        "ticks_all_ok": _ticks_equal(
            truth.y_axis_ticks, got.y_axis_ticks,
            [d.price for d in truth.drawings if d.type == "horizontal_line" and d.price is not None], tick,
        ),
        "drawn": len(truth.drawings),
        "found": len(matched),
        "extra": len(got.drawings) - len(matched),
        "worst_err_ticks": max(errs) if errs else None,
        "within_one_tick": sum(e <= 1 + 1e-9 for e in errs),
        "within_one_pct": sum(abs(tv - gv) <= 0.01 * abs(tv) for tv, gv, _ in pairs),
        "estimated_prices": sum(est for _, _, est in pairs),
        "prices_total": sum(getattr(d, f) is not None for d in truth.drawings for f in PRICE_FIELDS.get(d.type, ())),
    }
