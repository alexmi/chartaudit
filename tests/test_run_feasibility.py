import json
from pathlib import Path

from chartaudit.reader import ReadError
from chartaudit.synth import render
from run_feasibility import parse_private_truth, run, synthetic_specs


def _truth_for(tmp: Path):
    specs = {f"SYN{s.seed:04d}": s for s in synthetic_specs(10)}
    return lambda path: render(specs[path.stem], tmp / "truth")[1]


def test_run_continues_after_errors(tmp_path):
    truth_for = _truth_for(tmp_path)
    calls = {"n": 0}

    def flaky(path, client, model):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ReadError("429 Too Many Requests")
        return truth_for(path), "{}"

    run(3, None, tmp_path / "out", reader=flaky, client=(None, "m"))
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and sum(r["error"] is not None for r in rows) == 1
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "2 of 2 charts with every axis label right" in summary
    assert "1 errors" in summary


def test_unexpected_exception_is_recorded_not_raised(tmp_path):
    def boom(path, client, model):
        raise TimeoutError("read timed out")

    run(1, None, tmp_path, reader=boom, client=(None, "m"))
    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert "timed out" in row["error"] and row["score"] is None


def test_specs_cover_the_variants():
    specs = synthetic_specs(4)
    assert {s.theme for s in specs} == {"light", "dark"}
    assert any(s.log_scale for s in specs) and any(s.box is None for s in specs)
    assert {s.box for s in specs} >= {"long", "short"}


def test_parse_private_truth(tmp_path):
    (tmp_path / "TO_SAVE.md").write_text(
        "- [ ] G1 `G1_GC1.png`: [GOLD Short](https://www.tradingview.com/chart/GC1!/x/) (COMEX:GC1! / 30m)\n"
        "- [ ] H2 `H2_IWM.png`: [Trend](https://www.tradingview.com/chart/IWM/y/) (AMEX:IWM / **78-min**)\n"
        "- [ ] D1 `D1_AAPL.png`: [Mean](https://www.tradingview.com/chart/AAPL/z/) (AAPL 1D)\n"
    )
    t = parse_private_truth(tmp_path / "TO_SAVE.md")
    assert t["G1"].ticker == "COMEX:GC1!" and t["G1"].timeframe == "30m"
    assert t["H2"].timeframe == "78m"
    assert t["D1"].ticker == "AAPL" and t["D1"].timeframe == "1D"


def test_parse_private_truth_odd_rows(tmp_path):
    (tmp_path / "TO_SAVE.md").write_text(
        "- [ ] F1 `F1_EUR.png`: [Fx](https://www.tradingview.com/chart/EURUSD/q/) (OANDA:EUR/USD / 4h)\n"
        "- [ ] F2 `F2_AAPL.png`: [Solo](https://www.tradingview.com/chart/AAPL/r/) (AAPL)\n"
    )
    t = parse_private_truth(tmp_path / "TO_SAVE.md")
    assert t["F1"].ticker == "OANDA:EUR/USD" and t["F1"].timeframe == "4h"
    assert t["F2"].ticker == "AAPL" and t["F2"].timeframe is None


def test_raw_text_is_stored_on_error(tmp_path):
    def bad(path, client, model):
        raise ReadError("invalid chart JSON", raw="not json")

    run(1, None, tmp_path, reader=bad, client=(None, "m"))
    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert row["raw"] == "not json"


def test_cache_skips_charts_already_read(tmp_path):
    truth_for = _truth_for(tmp_path)
    calls = {"n": 0}

    def counting(path, client, model):
        calls["n"] += 1
        return truth_for(path), "{}"

    cache = tmp_path / "cache.jsonl"
    run(2, None, tmp_path / "a", reader=counting, client=(None, "m"), cache_path=cache)
    run(3, None, tmp_path / "b", reader=counting, client=(None, "m"), cache_path=cache)
    assert calls["n"] == 3  # 2 new, then only the 1 chart not seen before
    rows = [json.loads(line) for line in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert [r["cached"] for r in rows] == [True, True, False]
    assert all(r["score"]["ticks_all_ok"] for r in rows)


def test_cache_ignores_errors_and_other_models(tmp_path):
    truth_for = _truth_for(tmp_path)
    calls = {"n": 0}

    def fail_once(path, client, model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ReadError("429")
        return truth_for(path), "{}"

    cache = tmp_path / "cache.jsonl"
    run(1, None, tmp_path / "a", reader=fail_once, client=(None, "m"), cache_path=cache)
    run(1, None, tmp_path / "b", reader=fail_once, client=(None, "m"), cache_path=cache)
    run(1, None, tmp_path / "c", reader=fail_once, client=(None, "other"), cache_path=cache)
    assert calls["n"] == 3  # error not cached; a different model is a different key


def test_private_subfolders_are_read(tmp_path):
    priv = tmp_path / "private"
    (priv / "batch").mkdir(parents=True)
    (priv / "G1_X.png").write_bytes(b"a")
    (priv / "batch" / "BP_NVDA.png").write_bytes(b"b")
    seen = []

    def rec(path, client, model):
        seen.append(path.name)
        raise ReadError("skip")

    run(0, priv, tmp_path / "out", reader=rec, client=(None, "m"))
    assert sorted(seen) == ["BP_NVDA.png", "G1_X.png"]


def test_seed_offset_gives_fresh_charts_with_the_same_variants():
    base, fresh = synthetic_specs(5), synthetic_specs(5, seed_offset=100)
    assert [s.seed for s in fresh] == [101, 102, 103, 104, 105]
    assert not {s.seed for s in base} & {s.seed for s in fresh}
    assert [(s.box, s.log_scale) for s in base] == [(s.box, s.log_scale) for s in fresh]
