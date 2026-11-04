import json

from chartaudit.audit import HORIZON
from chartaudit.examples import list_examples, load_example, run_example, save_example
from chartaudit.synth import SynthSpec, make_bars, render


def _make(tmp_path, name="held"):
    bars = make_bars(seed=5, n=120 + HORIZON)
    png, truth = render(SynthSpec(seed=5, box=None), tmp_path / "img", bars=bars.iloc[:120])
    save_example(name, png, truth, bars, "Levels that held", tmp_path)
    return bars, truth


def test_saved_example_round_trips(tmp_path):
    bars, truth = _make(tmp_path)
    ex = load_example("held", tmp_path)
    assert ex["reading"] == truth and ex["title"] == "Levels that held"
    assert ex["png"][:4] == b"\x89PNG"
    assert len(ex["bars"]) == len(bars) and abs(ex["bars"].close.iloc[-1] - bars.close.iloc[-1]) < 1e-3
    assert list_examples(tmp_path) == ["held"]
    assert json.loads((tmp_path / "held.json").read_text())["asof"] == truth.last_visible_date


def test_run_example_audits_without_model_or_price_lookup(tmp_path):
    _make(tmp_path)
    overlay, table, reading, notes = run_example("held", tmp_path)
    assert overlay[:4] == b"\x89PNG"
    assert [r["source"] for r in table][0] == "trader"
    assert sum(r["held"] + r["broke"] + r["untested"] for r in table if r["source"] == "trader") == 2
    assert reading["ticker"] == "SYN0005"
    assert "saved" in notes.lower()


def test_unknown_example_is_refused(tmp_path):
    _make(tmp_path)
    overlay, table, reading, notes = run_example("../etc/passwd", tmp_path)
    assert overlay is None and table == [] and "unknown example" in notes.lower()
