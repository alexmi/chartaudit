"""Score drawn lines and position boxes on price action after the chart, never before it.

Bars are a DataFrame with open, high, low, close and a date index. `asof` is the chart's
date: bars up to and including it are what the trader could see; only later bars score.

A touch is a bar whose range comes within `tol` of the level (tol = tol_atr x ATR, so it
scales with volatility). Touching bars closer than `min_gap` bars to the previous touch
are one visit. A level breaks when a bar closes through it by more than `tol`.
"""
import numpy as np
import pandas as pd

ATR_BARS = 14
TOL_ATR = 0.25
MIN_GAP = 5
HORIZON = 20


def atr(bars: pd.DataFrame, n: int = ATR_BARS) -> float:
    prev = bars.close.shift(1)
    tr = pd.concat([bars.high - bars.low, (bars.high - prev).abs(), (bars.low - prev).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def touches(bars: pd.DataFrame, price: float, tol: float, min_gap: int = MIN_GAP) -> list[int]:
    """Positions of the first bar of each visit to the level."""
    hit = np.flatnonzero((bars.low.to_numpy() - tol <= price) & (bars.high.to_numpy() + tol >= price))
    out, last = [], None
    for i in hit:
        if last is None or i - last >= min_gap:
            out.append(int(i))
        last = i
    return out


def _split(bars: pd.DataFrame, asof, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = bars.loc[:asof]
    return before, bars.iloc[len(before) : len(before) + horizon]


def score_level(
    bars: pd.DataFrame, price: float, asof, horizon: int = HORIZON, tol_atr: float = TOL_ATR, min_gap: int = MIN_GAP
) -> dict:
    before, after = _split(bars, asof, horizon)
    tol = tol_atr * atr(before)
    support = before.close.iloc[-1] > price
    out = {
        "role": "support" if support else "resistance",
        "touches_before": len(touches(before, price, tol, min_gap)),
        "holds": 0,
        "bars_to_break": None,
        "bars_seen": len(after),
    }
    if after.empty:
        return out | {"outcome": "pending"}
    broke = (after.close < price - tol) if support else (after.close > price + tol)
    if broke.any():
        k = int(np.argmax(broke.to_numpy()))
        out["bars_to_break"] = k + 1
        after = after.iloc[:k]
    out["holds"] = len(touches(after, price, tol, min_gap))
    outcome = "broke" if out["bars_to_break"] else ("held" if out["holds"] else "untested")
    return out | {"outcome": outcome}


def score_box(bars: pd.DataFrame, entry: float, stop: float, target: float, asof, horizon: int = HORIZON) -> dict:
    """Long if the target is above entry. Fills on the first bar that trades through entry;
    if stop and target fall in the same bar, it counts as a loss."""
    long = target > entry
    out = {"side": "long" if long else "short", "outcome": "invalid", "r": None, "bars_to_exit": None}
    if (stop >= entry) if long else (stop <= entry):
        return out
    risk = abs(entry - stop)
    _, after = _split(bars, asof, horizon)
    filled = (after.low <= entry) & (after.high >= entry)
    if not filled.any():
        return out | {"outcome": "not_triggered"}
    for k in range(int(np.argmax(filled.to_numpy())), len(after)):
        bar = after.iloc[k]
        hit_stop = bar.low <= stop if long else bar.high >= stop
        hit_target = bar.high >= target if long else bar.low <= target
        if hit_stop or hit_target:
            won = hit_target and not hit_stop
            return out | {"outcome": "win" if won else "loss", "r": abs(target - entry) / risk if won else -1, "bars_to_exit": k + 1}
    last = float(after.close.iloc[-1])
    return out | {"outcome": "open", "r": (last - entry if long else entry - last) / risk}


def pivot_levels(before: pd.DataFrame, n: int = 5, k: int = 5) -> list[float]:
    """The n most recent swing highs and lows: a high above every other high within k bars, likewise lows."""
    hi, lo = before.high.to_numpy(), before.low.to_numpy()
    found = []
    for i in range(k, len(before) - k):
        win = slice(i - k, i + k + 1)
        others_hi = np.delete(hi[win], k)
        others_lo = np.delete(lo[win], k)
        if hi[i] > others_hi.max():
            found.append((i, float(hi[i])))
        if lo[i] < others_lo.min():
            found.append((i, float(lo[i])))
    levels = []
    for _, p in sorted(found, reverse=True):
        if p not in levels:
            levels.append(p)
    return levels[:n]


def random_levels(before: pd.DataFrame, n: int, seed: int) -> list[float]:
    """Levels drawn uniformly across the visible price range."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.uniform(before.low.min(), before.high.max(), n)]
