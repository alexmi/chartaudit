"""Pull a trader's stated calls and general rules out of a newsletter transcript, with a text model.

A call is a specific, checkable statement about one ticker ("buy SPY at the new demand zone");
a rule is a general claim the audit can test across charts ("first test of supply is always sold").
"""
import json
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

from chartaudit.schema import _to_float

Action = Literal["buy", "sell", "watch", "target", "avoid", "other"]

PROMPT = """Below is the transcript of a trading newsletter. Return JSON only, with:
- date: the newsletter's date as YYYY-MM-DD if stated
- calls: every specific statement about one ticker, each with
    ticker, action (buy, sell, watch, target, avoid or other), kind (level, zone, box or none),
    price (for a level), top and bottom (for a zone), entry, stop, target (for a trade),
    timeframe (e.g. daily, 60m), condition (when it applies), quote (the exact words, under 25 words)
- rules: every general trading rule the speaker states, each with rule (one short sentence) and quote
Use numbers only when the speaker says them; otherwise null. Do not invent levels.

Transcript:
"""


class Call(BaseModel):
    ticker: str | None = None
    action: Action = "other"
    kind: str | None = None
    price: float | None = None
    top: float | None = None
    bottom: float | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    timeframe: str | None = None
    condition: str | None = None
    quote: str | None = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, v):
        return str(v).strip().upper() if v else None

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, v):
        v = str(v).strip().lower() if v else "other"
        return v if v in get_args(Action) else "other"

    @field_validator("price", "top", "bottom", "entry", "stop", "target", mode="before")
    @classmethod
    def _num(cls, v):
        return _to_float(v)


class Rule(BaseModel):
    rule: str
    quote: str | None = None


class CallSheet(BaseModel):
    date: str | None = None
    calls: list[Call] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)


def parse_calls(text: str) -> CallSheet:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:120]!r}")
    return CallSheet.model_validate(json.loads(text[start : end + 1]))


def extract_calls(transcript: str, client, model: str) -> tuple[CallSheet, str]:
    resp = client.chat.completions.create(
        model=model, temperature=0, messages=[{"role": "user", "content": PROMPT + transcript}]
    )
    raw = resp.choices[0].message.content or ""
    return parse_calls(raw), raw
