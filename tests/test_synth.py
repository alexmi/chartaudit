import numpy as np

from chartaudit.synth import SynthSpec, make_bars, render


def test_bars_are_reproducible_and_valid():
    a, b = make_bars(7), make_bars(7)
    assert a.equals(b) and len(a) == 120
    assert (a.high >= a[["open", "close"]].max(axis=1)).all()
    assert (a.low <= a[["open", "close"]].min(axis=1)).all()


def test_random_walk_has_no_drift():
    rets = np.concatenate([np.diff(np.log(make_bars(s).close)) for s in range(200)])
    assert abs(rets.mean()) < 3 * rets.std() / np.sqrt(len(rets))


def test_render_long_box_truth(tmp_path):
    path, truth = render(SynthSpec(seed=1, box="long"), tmp_path)
    assert path.exists() and path.name == "SYN0001.png"
    assert truth.ticker == "SYN0001" and truth.timeframe == "1D"
    box = [d for d in truth.drawings if d.type == "position_box"][0]
    assert box.stop < box.entry < box.target
    assert len([d for d in truth.drawings if d.type == "horizontal_line"]) == 2


def test_short_box_and_dark_theme(tmp_path):
    _, truth = render(SynthSpec(seed=2, box="short", theme="dark"), tmp_path)
    box = [d for d in truth.drawings if d.type == "position_box"][0]
    assert box.target < box.entry < box.stop


def test_log_scale_ticks_are_the_printed_labels(tmp_path):
    _, truth = render(SynthSpec(seed=3, log_scale=True), tmp_path)
    assert truth.y_axis_scale == "log"
    vals = [float(t.replace(",", "")) for t in truth.y_axis_ticks]
    assert vals == sorted(vals, reverse=True) and len(vals) >= 3


def test_ticks_hidden_by_level_tags_are_not_in_truth(tmp_path):
    _, truth = render(SynthSpec(seed=2, box="short", theme="dark"), tmp_path)
    assert "104.00" not in truth.y_axis_ticks and "100.00" not in truth.y_axis_ticks
    _, plain = render(SynthSpec(seed=2, box="short", theme="dark", label_levels=False), tmp_path)
    assert "104.00" in plain.y_axis_ticks
