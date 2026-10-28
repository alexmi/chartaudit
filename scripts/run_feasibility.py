"""Feasibility run: read synthetic charts (exact answers) and real screenshots with Muse Spark, then summarize.

    python scripts/run_feasibility.py --synthetic 20 --private data/private --out runs/2026-10-27
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from chartaudit.extraction import score_reading
from chartaudit.reader import PROMPT, make_client, read_chart
from chartaudit.schema import ChartReading
from chartaudit.synth import SynthSpec, render

VARIANTS = [
    {"box": "long", "theme": "light"},
    {"box": "short", "theme": "dark"},
    {"box": "long", "log_scale": True},
    {"box": None, "n_levels": 3, "theme": "dark"},
]
ROW = re.compile(r"^- \[[ x]\] (\w+) `[^`]+`: \[[^\]]*\]\([^)]+\) \(([^)]+)\)", re.M)


def synthetic_specs(n: int) -> list[SynthSpec]:
    return [SynthSpec(seed=i + 1, **VARIANTS[i % len(VARIANTS)]) for i in range(n)]


def parse_private_truth(to_save: Path) -> dict[str, ChartReading]:
    """Ticker and timeframe for each saved screenshot, from the TO_SAVE.md checklist."""
    out = {}
    for ident, meta in ROW.findall(to_save.read_text()):
        meta = meta.replace("*", "").strip()
        if " / " in meta:
            ticker, tf = (p.strip() for p in meta.split(" / ", 1))
        else:
            ticker, tf = (meta.split() + [None])[:2]
        if tf:
            tf = re.sub(r"(\d+)-?min\b", r"\1m", tf.split()[0])
        out[ident] = ChartReading(ticker=ticker, timeframe=tf)
    return out


def _cache_key(path: Path, model: str) -> str:
    """Same image bytes, model and prompt give the same reading, so it is read once."""
    h = hashlib.sha256(path.read_bytes())
    h.update(model.encode())
    h.update(PROMPT.encode())
    return h.hexdigest()


def _load_cache(cache_path: Path | None) -> dict:
    if not cache_path or not cache_path.exists():
        return {}
    entries = (json.loads(line) for line in cache_path.read_text().splitlines() if line.strip())
    return {e["key"]: e for e in entries}


def _read_one(path, truth, source, reader, client, model, cache=None) -> dict:
    row = {"id": path.stem, "source": source, "reading": None, "raw": None, "error": None, "score": None, "cached": False}
    hit = cache.get(_cache_key(path, model)) if cache is not None else None
    try:
        if hit:
            reading, row["raw"], row["cached"] = ChartReading.model_validate(hit["reading"]), hit["raw"], True
        else:
            reading, row["raw"] = reader(path, client, model)
        row["reading"] = reading.model_dump()
        row["score"] = score_reading(truth, reading) if truth else None
    except Exception as e:  # one failed chart must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
        row["raw"] = getattr(e, "raw", None)
    return row


def _summary(rows: list[dict]) -> str:
    syn = [r for r in rows if r["source"] == "synthetic" and r["score"]]
    real = [r for r in rows if r["source"] == "private" and r["score"]]
    errors = [r for r in rows if r["error"]]
    lines = ["# Feasibility run", ""]
    if syn:
        ticks_ok = sum(r["score"]["ticks_all_ok"] for r in syn)
        within = sum(r["score"]["within_one_tick"] for r in syn)
        total = sum(r["score"]["prices_total"] for r in syn)
        found = sum(r["score"]["found"] for r in syn)
        drawn = sum(r["score"]["drawn"] for r in syn)
        worst = max((r["score"]["worst_err_ticks"] for r in syn if r["score"]["worst_err_ticks"] is not None), default=None)
        lines += [
            f"**Synthetic:** {ticks_ok} of {len(syn)} charts with every axis label right; "
            f"drawn levels within one tick on {within} of {total} prices; worst error {worst} ticks.",
            f"Drawings found: {found} of {drawn}.",
            "",
        ]
    if real:
        lines += [
            f"**Real screenshots:** ticker right on {sum(r['score']['ticker_ok'] for r in real)} of {len(real)}, "
            f"timeframe right on {sum(r['score']['timeframe_ok'] for r in real)} of {len(real)}. "
            "Drawings recorded in results.jsonl for review.",
            "",
        ]
    lines.append(f"{len(rows)} charts ({sum(r['cached'] for r in rows)} from cache), {len(errors)} errors.")
    lines += [f"* {r['id']}: {r['error']}" for r in errors]
    return "\n".join(lines) + "\n"


def run(
    n_synthetic: int, private_dir: Path | None, out_dir: Path, reader=read_chart, client=None,
    delay_s: float = 0.0, cache_path: Path | None = None,
) -> Path:
    client, model = client or make_client()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(*render(s, out_dir / "synthetic"), "synthetic") for s in synthetic_specs(n_synthetic)]
    if private_dir:
        truths = parse_private_truth(private_dir / "TO_SAVE.md") if (private_dir / "TO_SAVE.md").exists() else {}
        jobs += [(p, truths.get(p.stem.split("_")[0]), "private") for p in sorted(private_dir.rglob("*.png"))]
    cache = _load_cache(cache_path) if cache_path else None
    rows, called = [], False
    with (out_dir / "results.jsonl").open("w") as f:
        for path, truth, source in jobs:
            if called and delay_s and not (cache and _cache_key(path, model) in cache):
                time.sleep(delay_s)
            row = _read_one(path, truth, source, reader, client, model, cache)
            called = called or not row["cached"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            if cache_path and not row["cached"] and not row["error"]:
                entry = {"key": _cache_key(path, model), "id": row["id"], "reading": row["reading"], "raw": row["raw"]}
                cache[entry["key"]] = entry
                with cache_path.open("a") as c:
                    c.write(json.dumps(entry) + "\n")
            status = "ERROR " + row["error"] if row["error"] else ("cached" if row["cached"] else "ok")
            print(f"{row['id']}: {status}", flush=True)
    (out_dir / "summary.md").write_text(_summary(rows))
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=20)
    ap.add_argument("--private", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--cache", type=Path, default=Path("runs/cache.jsonl"), help="reuse earlier reads; 'none' to disable")
    a = ap.parse_args()
    out = run(a.synthetic, a.private, a.out, delay_s=a.delay, cache_path=None if str(a.cache) == "none" else a.cache)
    print((out / "summary.md").read_text())
