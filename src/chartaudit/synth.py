"""Synthetic candlestick charts with drawn levels at known prices.

Prices are a zero-drift random walk (a martingale), so no drawn line has a real edge.
That makes these charts both an exact answer key for chart reading and a null test
for scoring (see EVAL_IDEAS.md).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullFormatter  # noqa: E402

from chartaudit.schema import ChartReading, Drawing  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#131722", "grid": "#e0e3eb", "up": "#089981", "down": "#f23645"},
    "dark": {"bg": "#131722", "fg": "#d1d4dc", "grid": "#2a2e39", "up": "#089981", "down": "#f23645"},
}
TICK_FMT = "%.2f"
BOX_BARS = 30


def make_bars(seed: int, n: int = 120, start_price: float = 100.0, start_date: str = "2025-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    open_ = np.concatenate([[start_price], close[:-1]])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.0, 0.005, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.0, 0.005, n)))
    idx = pd.bdate_range(start_date, periods=n)
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close}, index=idx)


@dataclass
class SynthSpec:
    seed: int
    n_bars: int = 120
    theme: Literal["light", "dark"] = "light"
    log_scale: bool = False
    n_levels: int = 2
    box: Literal["long", "short"] | None = "long"
    label_levels: bool = True


def _truth_drawings(bars: pd.DataFrame, spec: SynthSpec) -> list[Drawing]:
    closes = bars.close.to_numpy()
    out = [
        Drawing(type="horizontal_line", price=round(float(np.percentile(closes, q)), 2))
        for q in np.linspace(25, 75, spec.n_levels)
    ]
    if spec.box:
        i = len(bars) - BOX_BARS
        entry = round(float(closes[i]), 2)
        risk = 2 * float((bars.high - bars.low).iloc[i - 20 : i].mean())
        sign = 1 if spec.box == "long" else -1
        out.append(
            Drawing(type="position_box", entry=entry, stop=round(entry - sign * risk, 2), target=round(entry + 2 * sign * risk, 2))
        )
    return out


def _draw(ax, bars: pd.DataFrame, drawings: list[Drawing], spec: SynthSpec, th: dict) -> None:
    for x, (o, h, l, c) in enumerate(bars[["open", "high", "low", "close"]].to_numpy()):
        col = th["up"] if c >= o else th["down"]
        ax.vlines(x, l, h, color=col, linewidth=1)
        ax.add_patch(Rectangle((x - 0.35, min(o, c)), 0.7, max(abs(c - o), 1e-9), color=col))
    n = len(bars)
    for d in drawings:
        if d.type == "horizontal_line":
            ax.axhline(d.price, color="#2962ff", linewidth=1.2)
            if spec.label_levels:
                ax.annotate(TICK_FMT % d.price, xy=(1.0, d.price), xycoords=("axes fraction", "data"), fontsize=9,
                            color="white", va="center", bbox={"boxstyle": "square,pad=0.2", "fc": "#2962ff", "ec": "none"})
        elif d.type == "position_box":
            x0, w = n - BOX_BARS, BOX_BARS + 8
            ax.add_patch(Rectangle((x0, min(d.entry, d.target)), w, abs(d.target - d.entry), color="#089981", alpha=0.25))
            ax.add_patch(Rectangle((x0, min(d.entry, d.stop)), w, abs(d.stop - d.entry), color="#f23645", alpha=0.25))


def _set_axes(ax, bars: pd.DataFrame, drawings: list[Drawing], spec: SynthSpec, th: dict) -> None:
    prices = [bars.low.min(), bars.high.max()] + [v for d in drawings for v in (d.price, d.entry, d.stop, d.target) if v is not None]
    lo, hi = min(prices), max(prices)
    pad = (hi - lo) * 0.05
    ax.set_xlim(-1, len(bars) + 10)
    if spec.log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(MaxNLocator(10).tick_values(lo - pad, hi + pad)))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_formatter(FormatStrFormatter(TICK_FMT))
    ax.yaxis.tick_right()
    step = 20
    ax.set_xticks(range(0, len(bars), step), [d.strftime("%b %d") for d in bars.index[::step]])
    ax.set_facecolor(th["bg"])
    ax.grid(True, color=th["grid"], linewidth=0.6)
    ax.tick_params(colors=th["fg"], which="both")
    for s in ax.spines.values():
        s.set_color(th["grid"])


def _printed_ticks(fig, ax, covered: list[float]) -> list[str]:
    """Tick labels a reader can actually see: inside the axis and not hidden under a level tag."""
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    to_frac = lambda v: ax.transAxes.inverted().transform(ax.transData.transform((0, v)))[1]  # noqa: E731
    hidden = [to_frac(p) for p in covered]
    labels = [(t.get_position()[1], t.get_text()) for t in ax.get_yticklabels()]
    return [
        txt for v, txt in sorted(labels, reverse=True)
        if lo <= v <= hi and txt and all(abs(to_frac(v) - h) > 0.02 for h in hidden)
    ]


def render(spec: SynthSpec, out_dir: Path) -> tuple[Path, ChartReading]:
    bars = make_bars(spec.seed, spec.n_bars)
    drawings = _truth_drawings(bars, spec)
    th = THEMES[spec.theme]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100, facecolor=th["bg"])
    try:
        _draw(ax, bars, drawings, spec, th)
        _set_axes(ax, bars, drawings, spec, th)
        ticker = f"SYN{spec.seed:04d}"
        fig.text(0.01, 0.97, f"{ticker} · 1D", color=th["fg"], fontsize=14, va="top")
        tags = [d.price for d in drawings if d.type == "horizontal_line"] if spec.label_levels else []
        ticks = _printed_ticks(fig, ax, tags)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ticker}.png"
        fig.savefig(path, facecolor=th["bg"])
    finally:
        plt.close(fig)
    truth = ChartReading(
        ticker=ticker,
        timeframe="1D",
        last_visible_date=bars.index[-1].strftime("%Y-%m-%d"),
        y_axis_scale="log" if spec.log_scale else "linear",
        y_axis_ticks=ticks,
        drawings=drawings,
    )
    return path, truth
