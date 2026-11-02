"""Guards for the public demo page: what a stranger can send in, and how much they can spend.

Every upload is decoded and re-saved as a plain PNG (no metadata, bounded size). Tickers and dates
read off a chart become file names and API parameters, so they are checked against strict patterns.
Model reads and price lookups each have a daily allowance; the per visitor limit keys on the
forwarded address, which a client can fake, so the daily caps are the real spending bound.
"""
import io
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable

import pandas as pd
from PIL import Image, UnidentifiedImageError

from chartaudit.prices import PriceError

MAX_BYTES = 5_000_000
MAX_PIXELS = 40_000_000
MAX_SIDE = 1600
FORMATS = {"PNG", "JPEG", "WEBP"}
TICKER = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,9}")
MAX_AGE_DAYS = 2 * 365


class UploadError(Exception):
    pass


def prepare_image(data: bytes, max_bytes: int = MAX_BYTES, max_side: int = MAX_SIDE, max_pixels: int = MAX_PIXELS) -> bytes:
    if len(data) > max_bytes:
        raise UploadError(f"file too large: {len(data) / 1e6:.1f} MB (limit {max_bytes / 1e6:.0f} MB)")
    try:
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as e:
        raise UploadError("not an image") from e
    if img.format not in FORMATS:
        raise UploadError(f"upload a PNG, JPEG or WebP image (got {img.format})")
    if img.width * img.height > max_pixels:  # checked before decoding, so a pixel bomb never loads
        raise UploadError("image has too many pixels")
    try:
        img = img.convert("RGB")
    except (OSError, Image.DecompressionBombError) as e:
        raise UploadError("not an image") from e
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "PNG")  # a fresh file: no EXIF, no text chunks
    return out.getvalue()


def clean_ticker(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip().upper().split(":")[-1].lstrip("$")
    return s if TICKER.fullmatch(s) else None


def clean_date(raw: str | None, today: str | None = None) -> str | None:
    if not raw:
        return None
    try:
        d = pd.Timestamp(raw).normalize()
    except (ValueError, TypeError):
        return None
    now = pd.Timestamp(today) if today else pd.Timestamp.now().normalize()
    if d > now or (now - d).days > MAX_AGE_DAYS:
        return None
    return d.strftime("%Y-%m-%d")


def visitor_id(headers: dict, client_host: str | None) -> str:
    fwd = {k.lower(): v for k, v in headers.items()}.get("x-forwarded-for", "")
    first = fwd.split(",")[0].strip()
    return first or client_host or "unknown"


class Limits:
    """Model reads: at most per_visitor per window_s for each visitor, and per_day for the page."""

    def __init__(self, per_visitor: int = 5, window_s: float = 3600, per_day: int = 100,
                 clock: Callable[[], float] = time.time, enabled: bool = True):
        self.per_visitor, self.window_s, self.per_day = per_visitor, window_s, per_day
        self.clock, self.enabled = clock, enabled
        self.seen: dict[str, deque] = defaultdict(deque)
        self.day, self.used = None, 0

    def take(self, visitor: str) -> str | None:
        """None if the read may go ahead (and it is counted); otherwise the reason to show."""
        if not self.enabled:
            return "Live reading is off right now; try the examples."
        now = self.clock()
        day = int(now // 86400)
        if day != self.day:
            self.day, self.used = day, 0
        if self.used >= self.per_day:
            return "The page has used today's reading allowance; try the examples, or come back tomorrow."
        q = self.seen[visitor]
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if len(q) >= self.per_visitor:
            return f"Limit of {self.per_visitor} readings per hour reached; try again later."
        q.append(now)
        self.used += 1
        return None


class FetchBudget:
    """Wraps a price fetch so the page makes at most per_day calls, and none after a rate limit reply."""

    def __init__(self, fetch: Callable[[str], pd.DataFrame], per_day: int = 20, clock: Callable[[], float] = time.time):
        self.fetch, self.per_day, self.clock = fetch, per_day, clock
        self.day, self.used, self.blocked = None, 0, False

    def __call__(self, symbol: str) -> pd.DataFrame:
        day = int(self.clock() // 86400)
        if day != self.day:
            self.day, self.used, self.blocked = day, 0, False
        if self.blocked or self.used >= self.per_day:
            raise PriceError("daily price lookup limit reached; showing the reading without the audit")
        self.used += 1
        try:
            return self.fetch(symbol)
        except PriceError as e:
            if "rate limit" in str(e):
                self.blocked = True
            raise
