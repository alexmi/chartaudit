"""Read a chart screenshot into a ChartReading with Muse Spark (Meta Model API, OpenAI-compatible)."""
import base64
import json
import mimetypes
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from chartaudit.schema import ChartReading

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
TIMEOUT_S = 60

PROMPT = """You are reading a trading chart screenshot. Return JSON only, with:
- ticker
- timeframe
- last_visible_date
- y_axis_scale: "linear" or "log"
- y_axis_ticks: every price label on the right axis, exactly as printed, top to bottom
- drawings: a list of objects, each with
    type: position_box, horizontal_line, zone, trendline or other
    position_box: entry, stop, target prices
    horizontal_line: price
    zone: top and bottom prices
If a value is not readable, use null. Do not guess."""


class ReadError(Exception):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


def parse_reading(text: str) -> ChartReading:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ReadError(f"no JSON object in model output: {text[:120]!r}")
    try:
        return ChartReading.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ReadError(f"invalid chart JSON: {e}") from e


def make_client() -> tuple[OpenAI, str]:
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("MODEL_API_KEY", "")
    if not key:
        raise ReadError("MODEL_API_KEY not set (copy .env.example to .env and add your key)")
    base = os.getenv("MODEL_API_BASE_URL") or DEFAULT_BASE_URL
    return OpenAI(api_key=key, base_url=base, timeout=TIMEOUT_S), os.getenv("MODEL_API_MODEL") or DEFAULT_MODEL


def read_chart(image_path: Path, client: OpenAI, model: str) -> tuple[ChartReading, str]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    data_url = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
    )
    raw = resp.choices[0].message.content or ""
    try:
        return parse_reading(raw), raw
    except ReadError as e:
        e.raw = raw
        raise
