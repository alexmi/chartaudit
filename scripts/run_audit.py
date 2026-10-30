"""Audit: score each chart's drawn levels and boxes on the bars after the chart, beside baselines.

    python scripts/run_audit.py --readings runs/2026-10-27/results.jsonl \
        --charts data/private/breakpoint_2026-10-27/charts.csv --out runs/audit_2026-10-29

charts.csv gives the truth the model can't read reliably: id (image file stem), ticker, asof
(the chart's date, YYYY-MM-DD). Baselines on the same visible bars: pivot levels (one per
trader level) and random levels (random_per_level per trader level). Zones get the same
baselines as bands of the trader zone's width: centred on a pivot, or placed at random.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

from chartaudit.audit import HORIZON, pivot_levels, random_levels, random_zones, score_box, score_level, score_zone
from chartaudit.calibrate import check_axis
from chartaudit.prices import PriceError, fetch_alpha_vantage, get_bars
from chartaudit.schema import ChartReading

LOOKBACK = 120  # bars a daily chart typically shows
BUCKETS = (("0 to 1", 0, 1), ("2", 2, 2), ("3+", 3, 10**9))


def audit_charts(readings, charts, bars_for, horizon=HORIZON, lookback=LOOKBACK, random_per_level=5, seed=0):
    rows, skipped = [], []
    for c in charts:
        reading = readings.get(c["id"])
        if reading is None:
            skipped.append({"id": c["id"], "reason": "no reading"})
            continue
        try:
            bars = bars_for(c["ticker"])
        except Exception as e:  # one bad ticker must not stop the audit
            skipped.append({"id": c["id"], "reason": f"{type(e).__name__}: {e}"})
            continue
        end = len(bars.loc[: c["asof"]])
        if end == 0:
            skipped.append({"id": c["id"], "reason": "no price history before the chart date"})
            continue
        if reading.y_axis_ticks:  # a misread axis would misplace every drawing, so set the chart aside
            check = check_axis(reading, bars, c["asof"])
            if not check["ok"]:
                skipped.append({"id": c["id"], "reason": "axis check: " + "; ".join(check["flags"])})
                continue
        window = bars.iloc[max(0, end - lookback) :]
        visible = window.loc[: c["asof"]]
        base = {"chart": c["id"], "ticker": c["ticker"], "asof": c["asof"]}

        levels = [d.price for d in reading.drawings if d.type == "horizontal_line" and d.price is not None]
        sources = [("trader", levels)]
        if levels:
            sources += [
                ("pivot", pivot_levels(visible, n=len(levels))),
                ("random", random_levels(visible, n=random_per_level * len(levels), seed=seed)),
            ]
        for source, prices in sources:
            for p in prices:
                rows.append(base | {"source": source, "kind": "level", "price": p} | score_level(window, p, c["asof"], horizon))
        for i, d in enumerate(z for z in reading.drawings if z.type == "zone" and None not in (z.top, z.bottom)):
            top, bottom = max(d.top, d.bottom), min(d.top, d.bottom)
            w = top - bottom
            bands = [("trader", [(top, bottom)])]
            pivots = pivot_levels(visible, n=i + 1)
            if len(pivots) > i:
                bands.append(("pivot", [(pivots[i] + w / 2, pivots[i] - w / 2)]))
            bands.append(("random", random_zones(visible, n=random_per_level, width=w, seed=seed + i)))
            for source, zs in bands:
                for t, b in zs:
                    rows.append(base | {"source": source, "kind": "zone", "top": t, "bottom": b} | score_zone(window, t, b, c["asof"], horizon))
        for d in reading.drawings:
            if d.type == "position_box" and None not in (d.entry, d.stop, d.target):
                box = {"entry": d.entry, "stop": d.stop, "target": d.target}
                rows.append(base | {"source": "trader", "kind": "box"} | box | score_box(window, **box, asof=c["asof"], horizon=horizon))
    return rows, skipped


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _rate(k, n):
    if n == 0:
        return "n/a", ""
    lo, hi = wilson(k, n)
    return f"{k / n:.0%}", f"{lo:.0%} to {hi:.0%}"


def _hold_table(rows, kind: str, noun: str) -> list[str]:
    out = [f"| source | {noun} | held | broke | hold rate | 95% CI | untested, pending or inside |", "|---|---|---|---|---|---|---|"]
    for src in ("trader", "pivot", "random"):
        ls = [r for r in rows if r["kind"] == kind and r["source"] == src]
        held = sum(r["outcome"] == "held" for r in ls)
        broke = sum(r["outcome"] == "broke" for r in ls)
        rate, ci = _rate(held, held + broke)
        out.append(f"| {src} | {len(ls)} | {held} | {broke} | {rate} | {ci} | {len(ls) - held - broke} |")
    return out


def summarize(rows, skipped) -> str:
    levels = [r for r in rows if r["kind"] == "level"]
    out = ["# Audit", "", "Hold rate = held / (held + broke), over lines and zones price reached after the chart.", "",
           "**Levels**", "", *_hold_table(rows, "level", "levels")]
    out += ["", "**Trader levels by touches before the chart**", "",
            "| touches | tested | held | hold rate | 95% CI |", "|---|---|---|---|---|"]
    trader = [r for r in levels if r["source"] == "trader" and r["outcome"] in ("held", "broke")]
    for name, lo, hi in BUCKETS:
        b = [r for r in trader if lo <= r["touches_before"] <= hi]
        held = sum(r["outcome"] == "held" for r in b)
        rate, ci = _rate(held, len(b))
        out.append(f"| {name} | {len(b)} | {held} | {rate} | {ci} |")
    if any(r["kind"] == "zone" for r in rows):
        out += ["", "**Zones**", "", *_hold_table(rows, "zone", "zones")]
    boxes = [r for r in rows if r["kind"] == "box"]
    count = {o: sum(r["outcome"] == o for r in boxes) for o in ("win", "loss", "open", "not_triggered", "invalid")}
    rs = [r["r"] for r in boxes if r["outcome"] in ("win", "loss")]
    avg = f"{sum(rs) / len(rs):.2f}" if rs else "n/a"
    out += ["", f"Boxes: {count['win']} win, {count['loss']} loss, {count['open']} open, "
            f"{count['not_triggered']} not triggered, {count['invalid']} invalid; average R on closed trades {avg}."]
    if skipped:
        out += ["", f"Skipped {len(skipped)} charts:"] + [f"* {s['id']}: {s['reason']}" for s in skipped]
    return "\n".join(out) + "\n"


def load_readings(path: Path) -> dict[str, ChartReading]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("reading"):
            out[r["id"]] = ChartReading.model_validate(r["reading"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readings", type=Path, required=True)
    ap.add_argument("--charts", type=Path, required=True)
    ap.add_argument("--prices", type=Path, default=Path("data/prices"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--lookback", type=int, default=LOOKBACK)
    ap.add_argument("--no-fetch", action="store_true", help="use only CSVs already in --prices")
    a = ap.parse_args()
    charts = list(csv.DictReader(a.charts.open()))
    limited = []

    def polite_fetch(symbol):  # spread live calls out; after the daily limit, stop calling
        if limited:
            raise PriceError("daily fetch limit reached; rerun tomorrow to fill this ticker")
        time.sleep(2)
        try:
            return fetch_alpha_vantage(symbol)
        except PriceError as e:
            if "rate limit" in str(e):
                limited.append(symbol)
            raise

    fetch_kw = {"fetch": None if a.no_fetch else polite_fetch}
    rows, skipped = audit_charts(
        load_readings(a.readings), charts, lambda t: get_bars(t, a.prices, **fetch_kw), a.horizon, a.lookback
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (a.out / "summary.md").write_text(summarize(rows, skipped))
    print((a.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
