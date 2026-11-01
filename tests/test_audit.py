import pandas as pd
import pytest

from chartaudit.audit import atr, pivot_levels, random_levels, random_zones, score_box, score_level, score_zone, touches


def bars(rows, start="2026-01-05"):
    """rows of (open, high, low, close), one business day each."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def flat(n, price=110.0, half_range=1.0):
    return [(price, price + half_range, price - half_range, price)] * n


def test_atr_is_mean_true_range():
    b = bars([(10, 11, 9, 10), (10, 12, 10, 11), (11, 11, 8, 9)])
    # true ranges: 2, max(2, |12-10|, |10-10|)=2, max(3, |11-11|, |8-11|)=3
    assert atr(b, n=3) == pytest.approx(7 / 3)


def test_touches_collapse_bars_closer_than_min_gap():
    rows = flat(30)
    for i in (3, 4, 5, 15, 27):  # 3,4,5 are one visit; 15 and 27 are separate
        rows[i] = (110, 111, 100.2, 110)
    assert touches(bars(rows), 100.0, tol=0.5, min_gap=5) == [3, 15, 27]


def test_support_that_holds_after_the_chart():
    rows = flat(40)
    rows[10] = rows[25] = (110, 111, 100.1, 109)  # two touches before the chart
    rows[33] = (110, 111, 100.3, 108)  # tested after, closes back above
    b = bars(rows)
    s = score_level(b, 100.0, asof=b.index[29], horizon=10, tol_atr=0.25)
    assert s["role"] == "support"
    assert s["touches_before"] == 2
    assert s["outcome"] == "held" and s["holds"] == 1


def test_resistance_that_breaks():
    rows = flat(40, price=90.0)
    rows[32] = (90, 101.5, 89, 101)  # closes above 100 by more than the tolerance: broken
    b = bars(rows)
    s = score_level(b, 100.0, asof=b.index[29], horizon=10, tol_atr=0.25)
    assert s["role"] == "resistance"
    assert s["outcome"] == "broke" and s["bars_to_break"] == 3


def test_level_never_reached_is_untested():
    b = bars(flat(40))
    s = score_level(b, 50.0, asof=b.index[29], horizon=10)
    assert s["outcome"] == "untested" and s["bars_seen"] == 10


def test_level_with_no_bars_after_is_pending():
    b = bars(flat(30))
    assert score_level(b, 100.0, asof=b.index[-1])["outcome"] == "pending"


def test_long_box_hits_target_first():
    rows = flat(30, price=100.0, half_range=0.5)
    rows[22] = (100, 106, 99.5, 105)
    b = bars(rows)
    s = score_box(b, entry=100, stop=97, target=105, asof=b.index[19], horizon=10)
    assert s == {"side": "long", "outcome": "win", "r": pytest.approx(5 / 3), "bars_to_exit": 3}


def test_short_box_stopped_out():
    rows = flat(30, price=100.0, half_range=0.5)
    rows[21] = (100, 103.5, 99.6, 103)
    b = bars(rows)
    s = score_box(b, entry=100, stop=103, target=90, asof=b.index[19], horizon=10)
    assert s["side"] == "short" and s["outcome"] == "loss" and s["r"] == -1


def test_stop_and_target_in_one_bar_counts_as_loss():
    rows = flat(30, price=100.0, half_range=0.5)
    rows[21] = (100, 106, 96, 101)
    b = bars(rows)
    assert score_box(b, entry=100, stop=97, target=105, asof=b.index[19])["outcome"] == "loss"


def test_box_entry_never_reached():
    b = bars(flat(30, price=120.0, half_range=0.5))
    s = score_box(b, entry=100, stop=97, target=105, asof=b.index[19], horizon=10)
    assert s["outcome"] == "not_triggered"


def test_box_open_at_horizon_is_marked_to_market():
    b = bars(flat(30, price=101.0, half_range=1.0))  # range touches entry 100, never 97 or 105
    s = score_box(b, entry=100, stop=97, target=105, asof=b.index[19], horizon=5)
    assert s["outcome"] == "open" and s["r"] == pytest.approx(1 / 3)


def test_box_with_stop_and_target_on_one_side_is_invalid():
    b = bars(flat(30))
    assert score_box(b, entry=100, stop=105, target=110, asof=b.index[19])["outcome"] == "invalid"


def test_baselines_come_only_from_bars_up_to_the_chart():
    rows = flat(30)
    rows[10] = (110, 125, 109, 110)  # swing high inside the visible window
    rows[28] = (110, 500, 1, 110)  # after the chart: must never leak into baselines
    b = bars(rows)
    before = b.loc[: b.index[20]]
    assert 125 in pivot_levels(before, n=4, k=3)
    r = random_levels(before, n=50, seed=1)
    assert len(r) == 50 and min(r) >= 109 and max(r) <= 125
    assert random_levels(before, n=5, seed=1) == random_levels(before, n=5, seed=1)


def test_demand_zone_that_holds():
    rows = flat(40)
    rows[12] = (110, 111, 101.5, 109)  # dips into the zone before the chart
    rows[34] = (110, 111, 100.5, 107)  # enters the zone after, closes above it
    b = bars(rows)
    s = score_zone(b, top=102.0, bottom=100.0, asof=b.index[29], horizon=10, tol_atr=0.25)
    assert s["role"] == "support" and s["touches_before"] == 1
    assert s["outcome"] == "held" and s["holds"] == 1


def test_supply_zone_breaks_on_close_above_its_top():
    rows = flat(40, price=100.0)
    rows[33] = (100, 112, 100, 111)  # closes well above the 108 top
    b = bars(rows)
    s = score_zone(b, top=108.0, bottom=106.0, asof=b.index[29], horizon=10, tol_atr=0.25)
    assert s["role"] == "resistance" and s["outcome"] == "broke" and s["bars_to_break"] == 4


def test_close_inside_the_zone_is_not_a_break():
    rows = flat(40)
    rows[33] = (110, 111, 100.5, 101.0)  # closes inside the 100 to 102 zone
    b = bars(rows)
    s = score_zone(b, top=102.0, bottom=100.0, asof=b.index[29], horizon=10, tol_atr=0.25)
    assert s["outcome"] == "held"


def test_price_inside_the_zone_at_the_chart_date_is_not_scored():
    b = bars(flat(40, price=101.0))
    s = score_zone(b, top=102.0, bottom=100.0, asof=b.index[29], horizon=10)
    assert s["role"] == "inside" and s["outcome"] == "inside"


def test_zone_given_upside_down_is_normalized():
    b = bars(flat(40))
    assert score_zone(b, top=100.0, bottom=102.0, asof=b.index[29])["role"] == "support"


def test_random_zones_keep_the_width_and_stay_in_range():
    b = bars(flat(40))
    zs = random_zones(b, n=20, width=2.0, seed=1)
    assert len(zs) == 20 and all(abs((t - lo) - 2.0) < 1e-9 for t, lo in zs)
    assert all(b.low.min() <= lo and t <= b.high.max() for t, lo in zs)


def test_matched_random_levels_keep_side_and_distance():
    from chartaudit.audit import matched_random_levels
    before = pd.DataFrame({"open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5, "close": [100.0] * 5},
                          index=pd.date_range("2026-01-01", periods=5))
    levels = [110.0, 92.0]
    out = matched_random_levels(before, levels, per_level=50, seed=1, spread=0.25)
    assert len(out) == 100
    above, below = out[:50], out[50:]
    assert all(107.5 <= x <= 112.5 for x in above)   # resistance 10 above close: same side, distance within 25%
    assert all(90.0 <= x <= 94.0 for x in below)     # support 8 below close
    assert len(set(out)) > 90
