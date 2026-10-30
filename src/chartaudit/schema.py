"""Structured reading of one chart screenshot: what the model returns and what synthetic truth looks like.

Validation is lenient on purpose: a model's format quirks (numbers as strings, "Linear",
unknown drawing types, null lists) must not throw away an otherwise good reading.
"""
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator

DrawingType = Literal["position_box", "horizontal_line", "zone", "trendline", "other"]
PRICE_KEYS = ("entry", "stop", "target", "price", "top", "bottom")


SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}


def _to_float(v):
    if isinstance(v, str):
        v = v.replace(",", "").replace("$", "").strip()
        if not v:
            return None
        mult = SUFFIX.get(v[-1].lower())
        return float(v[:-1]) * mult if mult else float(v)
    return v


class Drawing(BaseModel):
    type: DrawingType
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    price: float | None = None
    top: float | None = None
    bottom: float | None = None
    estimated: bool | None = None  # True when the prices were read off the axis, not from a printed label

    @field_validator("estimated", mode="before")
    @classmethod
    def _flag(cls, v):
        return str(v).strip().lower() in ("true", "1", "yes") if isinstance(v, str) else v

    @field_validator("type", mode="before")
    @classmethod
    def _known_type(cls, v):
        v = str(v).strip().lower() if v is not None else "other"
        return v if v in get_args(DrawingType) else "other"

    @field_validator(*PRICE_KEYS, mode="before")
    @classmethod
    def _price(cls, v):
        return _to_float(v)


class ChartReading(BaseModel):
    ticker: str | None = None
    timeframe: str | None = None
    last_visible_date: str | None = None
    y_axis_scale: Literal["linear", "log"] | None = None
    y_axis_ticks: list[str] = Field(default_factory=list)
    drawings: list[Drawing] = Field(default_factory=list)

    @field_validator("ticker", "timeframe", "last_visible_date", mode="before")
    @classmethod
    def _text(cls, v):
        return None if v is None else str(v)

    @field_validator("y_axis_scale", mode="before")
    @classmethod
    def _scale(cls, v):
        v = str(v).strip().lower() if v is not None else None
        return v if v in ("linear", "log") else None

    @field_validator("y_axis_ticks", mode="before")
    @classmethod
    def _ticks(cls, v):
        return [str(t) for t in v if t is not None] if v else []

    @field_validator("drawings", mode="before")
    @classmethod
    def _drawings(cls, v):
        return v or []
