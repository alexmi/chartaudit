"""Built-in examples for the demo page: a chart, the reading saved from an earlier model call, and the bars.

Examples are synthetic (scripts/make_examples.py), so they are safe to publish and need neither a model
call nor a price lookup: anyone can try the page without a key or an allowance.
"""
import io
import json
from pathlib import Path

import pandas as pd

from chartaudit.demo import analyze, box_notes, outcome_table, render_overlay
from chartaudit.schema import ChartReading

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
NOTE = "Saved example: the reading was made earlier by the model, and the prices are synthetic (a random walk)."


def save_example(name: str, png: Path, reading: ChartReading, bars: pd.DataFrame, title: str, out_dir: Path = EXAMPLES_DIR,
                 asof: str | None = None) -> None:
    """asof is the chart's date when known (as a visitor would type it); else the date the model read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.png").write_bytes(Path(png).read_bytes())
    bars.round(4).to_csv(out_dir / f"{name}.csv", index_label="date")
    meta = {"title": title, "ticker": reading.ticker, "asof": asof or reading.last_visible_date, "reading": reading.model_dump()}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")


def list_examples(out_dir: Path = EXAMPLES_DIR) -> list[str]:
    return sorted(p.stem for p in out_dir.glob("*.json"))


def load_example(name: str, out_dir: Path = EXAMPLES_DIR) -> dict:
    meta = json.loads((out_dir / f"{name}.json").read_text())
    bars = pd.read_csv(out_dir / f"{name}.csv", index_col="date", parse_dates=True)
    return {"title": meta["title"], "ticker": meta["ticker"], "asof": meta["asof"],
            "reading": ChartReading.model_validate(meta["reading"]), "bars": bars,
            "png": (out_dir / f"{name}.png").read_bytes(), "png_path": out_dir / f"{name}.png"}


def run_example(name: str, out_dir: Path = EXAMPLES_DIR) -> tuple[bytes | None, list[dict], dict | None, str]:
    """Same outputs as demo.handle_upload, from saved files only."""
    if name not in list_examples(out_dir):
        return None, [], None, "Unknown example."
    ex = load_example(name, out_dir)
    today = ex["bars"].index[-1].strftime("%Y-%m-%d")  # synthetic dates never go stale
    out = analyze(ex["png"], lambda _: ex["reading"], lambda _: ex["bars"], ticker=ex["ticker"], asof=ex["asof"], today=today)
    notes = [NOTE] + box_notes(out["rows"]) + out["notes"] + [f"Axis check: {f}" for f in out["axis"].get("flags", [])]
    overlay = render_overlay(out["reading"], out["bars"], out["asof"])
    return overlay, outcome_table(out["rows"]), out["reading"].model_dump(), "\n".join(notes)
