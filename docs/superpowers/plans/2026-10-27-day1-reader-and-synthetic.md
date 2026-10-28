# ChartAudit Day 1: Synthetic Charts and Muse Spark Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repo that renders synthetic charts with known drawn levels, reads any chart screenshot into structured JSON with Muse Spark, and reports how accurate the reading is (the Appendix B feasibility number).

**Architecture:** Three small modules around one shared schema: `synth` renders random-walk candlestick charts plus position boxes and horizontal lines and returns the exact truth; `reader` sends an image to Muse Spark through the OpenAI-compatible Meta Model API and validates the JSON; `extraction` compares a reading to the truth. A script runs both on N synthetic charts and the 29 private TradingView charts and writes a summary.

**Tech Stack:** Python 3.14 (`/opt/homebrew/bin/python3`), numpy, pandas, matplotlib, pydantic v2, openai>=2 (pointed at the Meta Model API), python-dotenv, pytest.

**Spec:** the hackathon application (sections 3, 6 Day 1, 9, Appendix B) and `EVAL_IDEAS.md`.

## Global Constraints

- Personal accounts only.
- New work only (built Oct 27 to Nov 4); no code reused from earlier projects.
- Repo root: `chartaudit/`. Becomes the public GitHub repo `chartaudit`.
- `data/private/` (copyrighted TradingView screenshots and the fetch script), `.env`, `.venv/`, `runs/` are gitignored and never committed.
- `data/public/` holds only Commons images with `CREDITS.md`; CC BY-SA files stay in `data/public/cc-by-sa/`.
- API config from env: `MODEL_API_KEY` (Alex enters it himself), `MODEL_API_BASE_URL` default `https://api.meta.ai/v1`, `MODEL_API_MODEL` default `muse-spark-1.3`.
- Venv must be created with `/opt/homebrew/bin/python3` (the `/usr/local/bin/python3` on PATH has no SSL).
- Reader prompt is the Appendix B prompt verbatim; nulls allowed, "Do not guess".
- Synthetic prices follow a zero-drift random walk (martingale null, EVAL_IDEAS.md section 1).
- No dashes as punctuation in user-facing copy (README).

## Review Focus

1. Model returns JSON wrapped in ```json fences or with prose before it: parse it anyway (test in Task 2).
2. Model returns a drawing missing its prices (e.g. a position_box with null stop): keep it, count it as found, exclude null fields from the price error (test in Task 3).
3. Axis tick labels with thousands separators or 5-digit futures prices ("29,868.00"): compare as numbers, not strings (test in Task 3).
4. Log-scale synthetic charts: truth tick labels must be the labels actually printed, not the linear guess (test in Task 1).
5. One API failure (timeout, 429, refusal) must not kill a 40-chart run: record the error for that chart and continue (test in Task 4).

---

### Task 1: Repo, schema and synthetic chart renderer

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`
- Create: `src/chartaudit/__init__.py`, `src/chartaudit/schema.py`, `src/chartaudit/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces (`schema.py`):
  - `class Drawing(BaseModel)`: `type: Literal["position_box","horizontal_line","zone","trendline","other"]`; optional floats `entry, stop, target, price, top, bottom` (default None).
  - `class ChartReading(BaseModel)`: `ticker: str | None`, `timeframe: str | None`, `last_visible_date: str | None` (ISO date), `y_axis_scale: Literal["linear","log"] | None`, `y_axis_ticks: list[str]`, `drawings: list[Drawing]`.
- Produces (`synth.py`):
  - `make_bars(seed: int, n: int = 120, start_price: float = 100.0, start_date: str = "2025-01-02") -> pd.DataFrame` with columns `open, high, low, close`, business-day DatetimeIndex.
  - `@dataclass SynthSpec`: `seed: int`, `n_bars: int = 120`, `theme: Literal["light","dark"] = "light"`, `log_scale: bool = False`, `n_levels: int = 2`, `box: Literal["long","short"] | None = "long"`, `label_levels: bool = True`.
  - `render(spec: SynthSpec, out_dir: Path) -> tuple[Path, ChartReading]`: writes `out_dir/SYN{seed:04d}.png`, returns the path and the truth.

- [ ] **Step 1: Scaffold.** `git init`; `pyproject.toml` (package `chartaudit` under `src/`, deps above, `[tool.pytest.ini_options] pythonpath=["src"]`); `.gitignore` with `data/private/`, `.env`, `.venv/`, `runs/`, `__pycache__/`, `reports/`, `research_notes/`; `.env.example` with the three env names and empty key; README one paragraph: what ChartAudit is, "Built during the Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026", data rights note. Create venv: `/opt/homebrew/bin/python3 -m venv .venv && .venv/bin/pip install -e . pytest`.

- [ ] **Step 2: Write the failing tests** in `tests/test_synth.py`:

```python
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
```

- [ ] **Step 3: Run** `.venv/bin/pytest tests/test_synth.py -v`. Expected: FAIL (import error).

- [ ] **Step 4: Implement `schema.py` and `synth.py`.** Bars: log returns `N(0, 0.015)` from `np.random.default_rng(seed)`, open = previous close, high/low = max/min of open and close times `1 + |N(0, 0.005)|` / `1 - |N(0, 0.005)|`. Render with plain matplotlib (no mplfinance): candles as rectangles plus wicks, price axis on the right, title `"SYN{seed:04d} · 1D"`, x labels as dates, figure 1600x900 px, dark theme background `#131722`. Levels: prices at the 25th and 75th percentile of closes rounded to 0.01, drawn as full-width lines, with a price tag on the right axis when `label_levels`. Box: from bar `n_bars - 30` to the right edge, entry = close of that bar, risk = 2 × mean(high - low) of the last 20 bars, stop/target at 1R/2R (long: stop below), all rounded to 0.01; green target rectangle and red stop rectangle like TradingView. Truth ticks: after `fig.canvas.draw()`, the y tick labels whose value lies inside `ax.get_ylim()`, top to bottom, text exactly as rendered. `last_visible_date` = last bar date.

- [ ] **Step 5: Run** the tests. Expected: 5 passed. Open `SYN0001.png` once to eyeball it looks like a trader chart.

- [ ] **Step 6: Commit** `git add -A && git commit -m "feat: synthetic charts with known levels"`

### Task 2: Muse Spark chart reader

**Files:**
- Create: `src/chartaudit/reader.py`
- Test: `tests/test_reader.py`

**Interfaces:**
- Consumes: `ChartReading`, `Drawing` from Task 1.
- Produces:
  - `PROMPT: str` (Appendix B prompt verbatim).
  - `class ReadError(Exception)`.
  - `parse_reading(text: str) -> ChartReading`: raises `ReadError` on unparseable or invalid JSON.
  - `make_client() -> tuple[OpenAI, str]`: loads `.env`, returns client and model name; raises `ReadError("MODEL_API_KEY not set")` when empty.
  - `read_chart(image_path: Path, client: OpenAI, model: str) -> tuple[ChartReading, str]`: returns the reading and the raw model text.

- [ ] **Step 1: Write the failing tests** (no network):

```python
GOOD = '{"ticker":"META","timeframe":"4h","last_visible_date":"2025-05-16","y_axis_scale":"linear","y_axis_ticks":["640","620"],"drawings":[{"type":"position_box","entry":621.11,"stop":612.2,"target":735.1}]}'

def test_parse_plain_json():
    r = parse_reading(GOOD)
    assert r.ticker == "META" and r.drawings[0].entry == 621.11

def test_parse_fenced_json_with_prose():
    assert parse_reading("Here you go:\n```json\n" + GOOD + "\n```").timeframe == "4h"

def test_parse_nulls_allowed():
    r = parse_reading('{"ticker":null,"timeframe":null,"last_visible_date":null,"y_axis_scale":null,"y_axis_ticks":[],"drawings":[{"type":"position_box","entry":10,"stop":null,"target":null}]}')
    assert r.ticker is None and r.drawings[0].stop is None

def test_parse_garbage_raises():
    with pytest.raises(ReadError):
        parse_reading("I cannot read this chart.")

def test_read_chart_sends_image(tmp_path, monkeypatch):
    # fake client records the request and returns GOOD
    ...
    reading, raw = read_chart(img, fake, "muse-spark-1.3")
    assert sent["model"] == "muse-spark-1.3" and sent["temperature"] == 0
    assert sent["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_reader.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `reader.py`.** Extract the first `{` to the last `}` before `json.loads`; validate with `ChartReading.model_validate`. `read_chart` uses `client.chat.completions.create(model=..., temperature=0, messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": data_url}}]}])`; mime from file suffix.

- [ ] **Step 4: Run** the tests. Expected: 5 passed.

- [ ] **Step 5: Live smoke test (needs Alex's key in `.env`).** `.venv/bin/python -c "from pathlib import Path; from chartaudit.reader import *; c,m=make_client(); print(read_chart(Path('data/private/G3_META.png'), c, m)[1])"`. Expected: JSON mentioning META. If the API rejects image input or the model name, stop and report the exact error to Alex.

- [ ] **Step 6: Commit** `git commit -am "feat: Muse Spark chart reader"` (add new files first).

### Task 3: Extraction scoring

**Files:**
- Create: `src/chartaudit/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: `ChartReading`, `Drawing`.
- Produces: `score_reading(truth: ChartReading, got: ChartReading, tick: float = 0.01) -> dict` with keys `ticker_ok: bool`, `timeframe_ok: bool`, `ticks_all_ok: bool`, `drawn: int`, `found: int`, `worst_err_ticks: float | None`, `within_one_tick: int` (truth prices matched within 1 tick), `prices_total: int`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_perfect_reading():
    s = score_reading(TRUTH, TRUTH)
    assert s["ticker_ok"] and s["ticks_all_ok"] and s["found"] == s["drawn"] == 3
    assert s["worst_err_ticks"] == 0 and s["within_one_tick"] == s["prices_total"] == 5

def test_numeric_tick_compare():
    got = TRUTH.model_copy(update={"y_axis_ticks": ["29,868.00", "29,800"]})
    truth = TRUTH.model_copy(update={"y_axis_ticks": ["29868", "29800.0"]})
    assert score_reading(truth, got)["ticks_all_ok"]

def test_box_off_by_three_ticks():
    got = TRUTH with box entry + 0.03
    assert score_reading(TRUTH, got)["worst_err_ticks"] == pytest.approx(3)

def test_missing_and_null_fields():
    got = TRUTH with one horizontal line removed and box stop=None
    s = score_reading(TRUTH, got)
    assert s["found"] == 2 and s["drawn"] == 3 and s["prices_total"] == 5

def test_timeframe_aliases():
    for a, b in [("1D", "D"), ("1D", "daily"), ("4h", "240"), ("1W", "weekly")]:
        assert timeframe_equal(a, b)
```

`TRUTH`: ticker `SYN0001`, timeframe `1D`, ticks `["110","105","100","95"]`, one long box (100/95/110) and two horizontal lines (97.5, 106.25).

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_extraction.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.** Match greedily per type: for each truth drawing, the unused got drawing of the same type with the smallest difference on its key price (`entry` for boxes, `price` for lines, `top` for zones). Price errors only over fields non-null in both. Ticker compare is case-insensitive after stripping an `EXCHANGE:` prefix and `!`. `timeframe_equal(a: str, b: str) -> bool` normalizes to minutes (D=1440, W=10080, "240"=240, "4h"=240).

- [ ] **Step 4: Run** the tests. Expected: 5 passed.

- [ ] **Step 5: Commit** `git commit -m "feat: score chart readings against truth"`

### Task 4: Feasibility run and summary

**Files:**
- Create: `scripts/run_feasibility.py`
- Test: `tests/test_run_feasibility.py`

**Interfaces:**
- Consumes: `render`, `SynthSpec`, `read_chart`, `make_client`, `ReadError`, `score_reading`.
- Produces: `run(n_synthetic: int, private_dir: Path | None, out_dir: Path, reader=read_chart, client: tuple[OpenAI, str] | None = None) -> Path` writing `out_dir/results.jsonl` (one row per chart: `id, source, reading|null, error|null, score|null`) and `out_dir/summary.md`. CLI: `python scripts/run_feasibility.py --synthetic 20 --private data/private --out runs/2026-10-27`.

- [ ] **Step 1: Write the failing tests** with a fake reader (no network):

```python
def test_run_continues_after_errors(tmp_path):
    calls = {"n": 0}
    def flaky(path, client, model):
        calls["n"] += 1
        if calls["n"] == 2: raise ReadError("429")
        return TRUTH_FOR(path), "{}"
    out = run(3, None, tmp_path, reader=flaky, client=(None, "m"))
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and sum(r["error"] is not None for r in rows) == 1
    assert "of 2 charts with every axis label right" in (tmp_path / "summary.md").read_text()
```

(`run` takes an optional `client: tuple | None`; when None it calls `make_client()`. `TRUTH_FOR` returns the synthetic truth so scores are perfect.)

- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_run_feasibility.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.** Synthetic specs cycle through: long box light, short box dark, log scale, no box with 3 levels. Private charts: every `*.png` in `private_dir`; truth limited to ticker and timeframe parsed from `TO_SAVE.md` rows (e.g. `COMEX:GC1! / 30m`), so they get `ticker_ok` and `timeframe_ok` only; drawings recorded for review. Summary: the Appendix B line for synthetic ("N of M charts with every axis label right; drawn levels within one tick on K of P prices; worst error X ticks"), mean found/drawn, ticker and timeframe accuracy on real charts, error count, total API calls.

- [ ] **Step 4: Run** the tests. Expected: passed. Then the full suite: `.venv/bin/pytest -v`, all green.

- [ ] **Step 5: Live run (Alex's key):** `.venv/bin/python scripts/run_feasibility.py --synthetic 20 --private data/private --out runs/2026-10-27`. Expected: 49 rows, summary written. About 50 API calls, well under $1. Report the summary line to Alex as is, even if weak.

- [ ] **Step 6: Commit** code only (`runs/` is ignored): `git commit -m "feat: feasibility run over synthetic and real charts"`
