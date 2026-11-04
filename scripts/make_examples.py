"""Build the demo page's built-in examples: synthetic charts read once by the model, saved with their bars.

    python scripts/make_examples.py            # reads each chart with Muse Spark (3 calls)
    python scripts/make_examples.py --truth    # saves the answer key instead (no model call)

Each example has a chart of 120 bars and 20 more bars after it, so the audit can score what was drawn.
Seeds are searched so the answer key shows the named outcome; the saved reading is the model's own.
The chart's date is saved from the answer key, as a visitor would type it: the synthetic x axis has no year.

The box example's seed is pinned. On 2026-11-03 Muse Spark read the first winning box chart (seed 183)
upside down (stop and target swapped, so a long box became a short one); it read the next three
(193, 234, 257) correctly to within 0.4. Seed 234 is used. Shown here so the example is not taken
as better reading than the model gives.
"""
import argparse
import tempfile
from pathlib import Path

from chartaudit.audit import HORIZON, score_box, score_level
from chartaudit.calibrate import check_axis
from chartaudit.examples import EXAMPLES_DIR, save_example
from chartaudit.reader import make_client, read_chart
from chartaudit.synth import SynthSpec, make_bars, render

N_BARS = 120
EXAMPLES = [
    {"name": "1_levels_held", "title": "Levels that held", "spec": {"box": None, "n_levels": 2}, "want": "held"},
    {"name": "2_levels_broke", "title": "Levels that broke", "spec": {"box": None, "n_levels": 2, "theme": "dark"}, "want": "broke"},
    {"name": "3_position_box", "title": "A long position box that hit its target", "spec": {"box": "long", "n_levels": 1}, "want": "win", "seed": 234},
]


def _chart(ex: dict, seed: int):
    bars = make_bars(seed, N_BARS + HORIZON)
    return bars, SynthSpec(seed=seed, **ex["spec"])


def trader_outcomes(ex: dict, seed: int) -> list[str]:
    """Outcomes of the answer key's drawings: levels for level examples, the box for the box example."""
    bars, spec = _chart(ex, seed)
    with tempfile.TemporaryDirectory() as d:
        _, truth = render(spec, Path(d), bars=bars.iloc[:N_BARS])
    asof = truth.last_visible_date
    if not check_axis(truth, bars, asof)["ok"]:  # every drawing must sit inside the printed axis labels
        return []
    if ex["want"] == "win":
        return [score_box(bars, d.entry, d.stop, d.target, asof)["outcome"] for d in truth.drawings if d.type == "position_box"]
    return [score_level(bars, d.price, asof)["outcome"] for d in truth.drawings if d.type == "horizontal_line"]


def find_seed(ex: dict, start: int = 100, tries: int = 400) -> int:
    seeds = [ex["seed"]] if "seed" in ex else range(start, start + tries)
    for seed in seeds:
        outcomes = trader_outcomes(ex, seed)
        if outcomes and all(o == ex["want"] for o in outcomes):
            return seed
    raise RuntimeError(f"no seed for {ex['name']} in {tries} tries")


def build(out_dir: Path, read) -> None:
    """read(png_path, answer_key) returns the reading to save."""
    with tempfile.TemporaryDirectory() as d:
        for ex in EXAMPLES:
            seed = find_seed(ex)
            bars, spec = _chart(ex, seed)
            png, truth = render(spec, Path(d), bars=bars.iloc[:N_BARS])
            save_example(ex["name"], png, read(png, truth), bars, ex["title"], out_dir, asof=truth.last_visible_date)
            print(f"{ex['name']}: seed {seed}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    ap.add_argument("--truth", action="store_true", help="save the answer key instead of a model reading")
    args = ap.parse_args()
    if args.truth:
        build(args.out, lambda png, truth: truth)
    else:
        client, model = make_client("meta")
        build(args.out, lambda png, truth: read_chart(png, client, model)[0])
