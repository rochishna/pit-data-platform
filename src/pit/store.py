"""Bitemporal store: every published value, with both of its time axes.

Each row says: for `observation_date` (the period the number describes), the value `value` was the
published figure from `realtime_start` through `realtime_end` (inclusive; 9999-12-31 = still current).

    valid time      = observation_date   ("what period is this about?")
    knowledge time  = realtime_start..end ("when did the world believe this?")

The one rule everything here enforces: a query "as of date D" may only see rows with realtime_start <= D.
Anything published after D is the future, and using it is lookahead bias.

Usage:
    python -m pit.store build           # data/raw/*.parquet -> data/pit.duckdb, validated
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

DDL = """
CREATE TABLE IF NOT EXISTS observations (
    series_id        VARCHAR NOT NULL,
    observation_date DATE    NOT NULL,
    value            DOUBLE,              -- NULL = published as missing
    realtime_start   DATE    NOT NULL,
    realtime_end     DATE    NOT NULL,
    PRIMARY KEY (series_id, observation_date, realtime_start)
);
CREATE TABLE IF NOT EXISTS releases (     -- one row per (series, vintage date): when each version came out
    series_id    VARCHAR NOT NULL,
    vintage_date DATE    NOT NULL,
    PRIMARY KEY (series_id, vintage_date)
);
"""


class IntegrityError(ValueError):
    pass


def validate(df: pl.DataFrame) -> None:
    """Checks the bitemporal invariants before anything is stored. Each failure names the series."""
    bad = df.filter(pl.col("realtime_end") < pl.col("realtime_start"))
    if bad.height:
        raise IntegrityError(f"{bad.height} rows end before they start, e.g. {bad.row(0, named=True)}")

    # For one observation, the published versions must tile knowledge time with no overlap:
    # each version ends the day before the next one starts. An overlap would make "as of D" ambiguous.
    seq = df.sort("series_id", "observation_date", "realtime_start").with_columns(
        pl.col("realtime_start").shift(-1).over("series_id", "observation_date").alias("next_start"))
    overlaps = seq.filter(pl.col("next_start").is_not_null() & (pl.col("realtime_end") >= pl.col("next_start")))
    if overlaps.height:
        r = overlaps.row(0, named=True)
        raise IntegrityError(f"{overlaps.height} overlapping versions, e.g. {r['series_id']} {r['observation_date']}: "
                             f"one runs to {r['realtime_end']} but the next starts {r['next_start']}")

    dupes = df.group_by("series_id", "observation_date", "realtime_start").len().filter(pl.col("len") > 1)
    if dupes.height:
        raise IntegrityError(f"{dupes.height} duplicate (series, observation, realtime_start) keys")


def build(raw_dir: str | Path = "data/raw", db: str | Path = "data/pit.duckdb") -> dict[str, int]:
    """Load every raw file into the store. Each series is replaced in one transaction (idempotent)."""
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {raw_dir}")
    loaded = {}
    with duckdb.connect(str(db)) as con:
        con.execute(DDL)
        for f in files:
            df = pl.read_parquet(f)
            validate(df)
            sid = df["series_id"][0]
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("DELETE FROM observations WHERE series_id = ?", [sid])
                con.execute("DELETE FROM releases WHERE series_id = ?", [sid])
                con.register("incoming", df.to_arrow())
                con.execute("""INSERT INTO observations SELECT series_id, observation_date, value,
                               realtime_start, realtime_end FROM incoming""")
                con.execute("INSERT INTO releases SELECT DISTINCT series_id, realtime_start FROM incoming")
                con.unregister("incoming")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            loaded[sid] = df.height
    return loaded


class PointInTime:
    """Read-only access that can only ever see the past relative to the as-of date."""

    def __init__(self, db: str | Path = "data/pit.duckdb"):
        self.con = duckdb.connect(str(db), read_only=True)

    def as_of(self, series_id: str, as_of: date | str) -> pl.DataFrame:
        """The full history of a series exactly as published on `as_of`: one value per observation date,
        from the version in force that day. Observations not yet released by then are absent."""
        return self.con.sql("""
            SELECT observation_date, value, realtime_start AS published
            FROM observations
            WHERE series_id = ? AND realtime_start <= CAST(? AS DATE) AND realtime_end >= CAST(? AS DATE)
            ORDER BY observation_date""", params=[series_id, str(as_of), str(as_of)]).pl()

    def latest(self, series_id: str) -> pl.DataFrame:
        """Today's revised history. Fine for describing the past; using it to TEST a forecast is lookahead."""
        return self.con.sql("""
            SELECT observation_date, value, realtime_start AS published FROM observations
            WHERE series_id = ? AND realtime_end = DATE '9999-12-31' ORDER BY observation_date""",
            params=[series_id]).pl()

    def first_release(self, series_id: str) -> pl.DataFrame:
        """The first published value of each observation: what the market saw on release day."""
        return self.con.sql("""
            SELECT observation_date, arg_min(value, realtime_start) AS value, min(realtime_start) AS published
            FROM observations WHERE series_id = ? GROUP BY 1 ORDER BY 1""", params=[series_id]).pl()

    def vintages(self, series_id: str) -> list[date]:
        return [r[0] for r in self.con.execute(
            "SELECT vintage_date FROM releases WHERE series_id = ? ORDER BY 1", [series_id]).fetchall()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the point-in-time store.")
    p.add_argument("command", choices=["build"])
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--db", default="data/pit.duckdb")
    args = p.parse_args(argv)
    for sid, n in build(args.raw, args.db).items():
        print(f"{sid}: {n:,} rows stored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
