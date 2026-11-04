"""ChartAudit demo page (Gradio; runs as a Hugging Face Space or locally with `python app.py`).

Keys come from the environment only (Space Secrets, or .env locally): MODEL_API_KEY for the reader,
ALPHAVANTAGE_API_KEY optional for prices. Settings, all optional:
  LIVE_READS=off            turn model reading off without a redeploy
  READS_PER_DAY=100         model reads for the whole page per day
  READS_PER_VISITOR_HOUR=5  model reads per visitor per hour
  PRICE_LOOKUPS_PER_DAY=20  Alpha Vantage calls per day (the free key allows 25, shared with other use)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

from chartaudit.demo import handle_upload  # noqa: E402
from chartaudit.examples import list_examples, load_example, run_example  # noqa: E402
from chartaudit.live import FetchBudget, Limits, visitor_id  # noqa: E402
from chartaudit.prices import fetch_alpha_vantage, get_bars  # noqa: E402
from chartaudit.reader import make_client, read_chart  # noqa: E402

PRICE_DIR = Path(os.getenv("PRICE_DIR", Path(tempfile.gettempdir()) / "chartaudit_prices"))

LIMITS = Limits(
    per_visitor=int(os.getenv("READS_PER_VISITOR_HOUR", "5")),
    per_day=int(os.getenv("READS_PER_DAY", "100")),
    enabled=os.getenv("LIVE_READS", "on").lower() != "off",
)
FETCH = FetchBudget(fetch_alpha_vantage, per_day=int(os.getenv("PRICE_LOOKUPS_PER_DAY", "20"))) \
    if os.getenv("ALPHAVANTAGE_API_KEY") else None

INTRO = """# ChartAudit
Upload a chart screenshot with lines drawn on it. Muse Spark reads the price levels, and ChartAudit scores
each level on the prices **after** the chart's date, beside levels picked by simple rules and at random.

Please upload only charts you made or may share. Uploads are not stored. This is a research demo, not investment advice.
"""


def read_png(png: bytes):
    client, model = make_client("meta")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(png)
        f.flush()
        return read_chart(Path(f.name), client, model)[0]


def bars_for(ticker: str):
    return get_bars(ticker, PRICE_DIR, fetch=FETCH)


EXAMPLES = {load_example(n)["title"]: n for n in list_examples()}


def show(overlay, table, reading, notes):
    if overlay is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(overlay)
            overlay = f.name
    rows = [[r["source"], r["levels"], r["held"], r["broke"], r["untested"], r["pending"], r["hold rate"]] for r in table]
    return overlay, rows, reading, notes


def run(path, ticker, asof, request: gr.Request):
    visitor = visitor_id(dict(request.headers), request.client.host if request.client else None)
    return show(*handle_upload(path, ticker, asof, visitor, read_png, bars_for, LIMITS))


def run_saved(title):
    """A built-in example: saved reading and synthetic prices, so no model call and no allowance used."""
    name = EXAMPLES.get(title)
    chart = str(load_example(name)["png_path"]) if name else None
    return (chart, *show(*run_example(name))) if name else (None, None, [], None, "Pick an example.")


with gr.Blocks(title="ChartAudit", analytics_enabled=False, delete_cache=(1800, 1800)) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            upload = gr.File(label="Chart screenshot", file_types=[".png", ".jpg", ".jpeg", ".webp"], type="filepath")
            ticker = gr.Textbox(label="Ticker (optional, overrides what is read)", max_length=16)
            asof = gr.Textbox(label="Chart date, YYYY-MM-DD (optional)", max_length=20)
            go = gr.Button("Read and audit", variant="primary")
            gr.Markdown("**Or try a built-in example** (synthetic chart, saved reading, no upload needed):")
            pick = gr.Radio(list(EXAMPLES), label="Examples", value=None)
            chart = gr.Image(label="Example chart", type="filepath", interactive=False)
        with gr.Column():
            overlay = gr.Image(label="What was read", type="filepath", interactive=False)
            notes = gr.Textbox(label="Notes", lines=3, interactive=False)
    table = gr.Dataframe(headers=["source", "levels", "held", "broke", "untested", "pending", "hold rate"],
                         label="Levels after the chart date", interactive=False)
    reading = gr.JSON(label="Model reading")
    go.click(run, [upload, ticker, asof], [overlay, table, reading, notes],
             api_visibility="private", concurrency_limit=2)
    pick.change(run_saved, [pick], [chart, overlay, table, reading, notes], api_visibility="private", concurrency_limit=4)

demo.queue(max_size=20, api_open=False)

if __name__ == "__main__":
    demo.launch(max_file_size="5mb", footer_links=[], show_error=False)
