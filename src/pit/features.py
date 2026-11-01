"""Forecast dataset: one row per industrial-production month, built honestly or leakily.

For target month t, the forecast is made on F_t = the day BEFORE industrial production for t is first released.

honest mode   every input is read as published on F_t (its as-of version). The target is the FIRST-RELEASE
              growth of month t, computed within the vintage published on release day.
leaky mode    every input and the target come from today's fully revised data: the common mistake.

Growth is always computed from two values of the SAME vintage. Index levels from different vintages are never
compared: the Nov 2025 benchmark revision lowered 2022+ levels by up to 2.5% while leaving 2021 and earlier alone.

Each honest row also records `max_input_published`: the newest publication date among all its inputs.
It must be <= forecast_date; tests and the experiment both assert it.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

# (feature name, series, months before target, kind)
#   growth: % change from month t-k-1 to t-k      diff: change in level (percentage points)
FEATURES = [
    ("ip_g1",  "INDPRO", 1, "growth"),
    ("ip_g2",  "INDPRO", 2, "growth"),
    ("ip_g3",  "INDPRO", 3, "growth"),
    ("tcu_d1", "TCU",    1, "diff"),
    ("rs_g0",  "RSAFS",  0, "growth"),   # retail sales for month t: often out a day or two before IP
    ("rs_g1",  "RSAFS",  1, "growth"),
    ("pay_g0", "PAYEMS", 0, "growth"),   # jobs report for month t: out ~2 weeks before IP
    ("pay_g1", "PAYEMS", 1, "growth"),
    ("ur_d0",  "UNRATE", 0, "diff"),
]
LATEST = "9999-12-31"


def _version(alias: str, series: str, month_expr: str, mode: str, date_col: str) -> str:
    """JOIN clause picking one version of one observation: as of `date_col` (honest) or today's (leaky)."""
    when = (f"{alias}.realtime_start <= {date_col} AND {alias}.realtime_end >= {date_col}" if mode == "honest"
            else f"{alias}.realtime_end = DATE '{LATEST}'")
    return (f"LEFT JOIN observations {alias} ON {alias}.series_id = '{series}' "
            f"AND {alias}.observation_date = {month_expr} AND {when}")


def build(db: str | Path, mode: str, start: str = "2001-07-01") -> pl.DataFrame:
    if mode not in ("honest", "leaky"):
        raise ValueError(mode)
    joins, cols, pubs = [], [], []
    for name, series, k, kind in FEATURES:
        a, b = f"{name}_a", f"{name}_b"     # a = month t-k, b = month t-k-1, same version rule
        joins.append(_version(a, series, f"t.month - INTERVAL {k} MONTH", mode, "t.forecast_date"))
        joins.append(_version(b, series, f"t.month - INTERVAL {k + 1} MONTH", mode, "t.forecast_date"))
        expr = (f"100 * ({a}.value / {b}.value - 1)" if kind == "growth" else f"{a}.value - {b}.value")
        cols.append(f"{expr} AS {name}")
        pubs += [f"{a}.realtime_start", f"{b}.realtime_start"]

    # Target: growth of month t vs t-1. Honest = within the vintage published on release day (first release).
    joins.append(_version("y_a", "INDPRO", "t.month", mode, "t.release_date"))
    joins.append(_version("y_b", "INDPRO", "t.month - INTERVAL 1 MONTH", mode, "t.release_date"))

    sql = f"""
        WITH t AS (
            SELECT observation_date AS month, min(realtime_start) AS release_date,
                   min(realtime_start) - INTERVAL 1 DAY AS forecast_date
            FROM observations WHERE series_id = 'INDPRO' AND observation_date >= DATE '{start}'
            GROUP BY 1
        )
        SELECT t.month, CAST(t.release_date AS DATE) AS release_date, CAST(t.forecast_date AS DATE) AS forecast_date,
               {', '.join(cols)},
               100 * (y_a.value / y_b.value - 1) AS target,
               CAST(greatest({', '.join(f'coalesce({p}, DATE {chr(39)}1900-01-01{chr(39)})' for p in pubs)}) AS DATE)
                   AS max_input_published
        FROM t {' '.join(joins)}
        ORDER BY t.month
    """
    with duckdb.connect(str(db), read_only=True) as con:
        return con.sql(sql).pl()


FEATURE_NAMES = [f[0] for f in FEATURES]
