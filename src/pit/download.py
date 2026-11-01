"""Download every published vintage of FRED/ALFRED series.

ALFRED keeps each historical version of a series. Asking the FRED API for the full real-time range
(realtime_start=1776-07-04, realtime_end=9999-12-31) returns one row per (observation date, vintage
period): the value, plus the date range during which that value was the published one.

Usage (on your own machine; the API key never leaves it):
    export FRED_API_KEY=...            # or put FRED_API_KEY=... in a local .env file (gitignored)
    python -m pit.download             # all series in configs/series.yaml -> data/raw/<ID>.parquet

Security: the key is read from the environment, sent only to api.stlouisfed.org, and redacted from
every error message (URLs in HTTP errors would otherwise contain it). It is never written to disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import polars as pl
import yaml

API = "https://api.stlouisfed.org/fred/series/observations"
PAGE = 100_000                     # API maximum per request
ALL_TIME = ("1776-07-04", "9999-12-31")

Fetch = Callable[[str], dict]      # url -> parsed JSON; injectable for tests


class DownloadError(RuntimeError):
    pass


def load_api_key(env_file: str | Path = ".env") -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise DownloadError("FRED_API_KEY is not set (environment variable or .env file)")
    return key


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def http_fetch(url: str, retries: int = 4) -> dict:
    """GET with exponential backoff on rate limits and server errors."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise AssertionError("unreachable")


def fetch_vintages(series_id: str, api_key: str, fetch: Fetch = http_fetch) -> pl.DataFrame:
    """All vintages of one series, paginated. Missing values ('.') become nulls."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        params = dict(series_id=series_id, api_key=api_key, file_type="json",
                      realtime_start=ALL_TIME[0], realtime_end=ALL_TIME[1], limit=PAGE, offset=offset)
        url = f"{API}?{urllib.parse.urlencode(params)}"
        try:
            page = fetch(url)
        except Exception as e:                       # never let the key escape in a traceback
            raise DownloadError(_redact(f"{series_id}: {type(e).__name__}: {e}", api_key)) from None
        if "error_message" in page:
            raise DownloadError(_redact(f"{series_id}: {page['error_message']}", api_key))
        total = int(page["count"])
        batch = page.get("observations", [])
        if not batch and offset < total:
            raise DownloadError(f"{series_id}: API returned an empty page at offset {offset} of {total}")
        rows.extend(batch)
        offset += len(batch)

    if len(rows) != total:
        raise DownloadError(f"{series_id}: expected {total} rows, got {len(rows)}")
    return pl.DataFrame(
        {
            "series_id": [series_id] * len(rows),
            "observation_date": [r["date"] for r in rows],
            "value": [None if r["value"] == "." else float(r["value"]) for r in rows],
            "realtime_start": [r["realtime_start"] for r in rows],
            "realtime_end": [r["realtime_end"] for r in rows],
        },
        schema={"series_id": pl.Utf8, "observation_date": pl.Utf8, "value": pl.Float64,
                "realtime_start": pl.Utf8, "realtime_end": pl.Utf8},
    ).with_columns(pl.col("observation_date", "realtime_start", "realtime_end").str.to_date())


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Download all vintages of the configured FRED series.")
    p.add_argument("--config", default="configs/series.yaml")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--series", nargs="*", help="only these series ids")
    args = p.parse_args(argv)

    try:
        key = load_api_key()
    except DownloadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    ids = args.series or list(yaml.safe_load(Path(args.config).read_text())["series"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        try:
            df = fetch_vintages(sid, key)
        except DownloadError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        tmp = out / f"{sid}.parquet.tmp"
        df.write_parquet(tmp)
        tmp.rename(out / f"{sid}.parquet")
        vintages = df["realtime_start"].n_unique()
        print(f"{sid}: {df.height:,} rows, {vintages:,} vintages, "
              f"observations {df['observation_date'].min()} -> {df['observation_date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
