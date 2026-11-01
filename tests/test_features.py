"""Feature builder and experiment guards: honest rows can only use what was published by the forecast day."""
from datetime import date

import polars as pl
import pytest

from pit import experiment, features, store

FOREVER = date(9999, 12, 31)


def obs(series, month, value, start, end=FOREVER):
    return (series, month, value, start, end)


def write_store(tmp_path, rows):
    raw = tmp_path / "raw"
    raw.mkdir()
    df = pl.DataFrame(rows, schema={"series_id": pl.Utf8, "observation_date": pl.Date, "value": pl.Float64,
                                    "realtime_start": pl.Date, "realtime_end": pl.Date}, orient="row")
    for sid in df["series_id"].unique():
        df.filter(pl.col("series_id") == sid).write_parquet(raw / f"{sid}.parquet")
    db = tmp_path / "pit.duckdb"
    store.build(raw, db)
    return db


@pytest.fixture
def db(tmp_path):
    # INDPRO: Jan..Apr 2025. February is first published Mar 14 as 101, revised Apr 15 to 99.
    # March first published Apr 15 (forecast day = Apr 14). April first published May 15.
    ip = [obs("INDPRO", date(2025, 1, 1), 100.0, date(2025, 2, 14)),
          obs("INDPRO", date(2025, 2, 1), 101.0, date(2025, 3, 14), date(2025, 4, 14)),
          obs("INDPRO", date(2025, 2, 1), 99.0, date(2025, 4, 15)),
          obs("INDPRO", date(2025, 3, 1), 99.99, date(2025, 4, 15)),
          obs("INDPRO", date(2025, 4, 1), 100.5, date(2025, 5, 15))]
    # Retail sales for March is published Apr 16: AFTER the forecast day for March IP (Apr 14).
    rs = [obs("RSAFS", date(2025, 2, 1), 500.0, date(2025, 3, 16)),
          obs("RSAFS", date(2025, 3, 1), 510.0, date(2025, 4, 16))]
    return write_store(tmp_path, ip + rs)


def row(df, month):
    return df.filter(pl.col("month") == month).row(0, named=True)


def test_honest_uses_version_known_on_forecast_day(db):
    h = features.build(db, "honest", start="2025-01-01")
    r = row(h, date(2025, 3, 1))
    assert r["forecast_date"] == date(2025, 4, 14)
    assert r["ip_g1"] == pytest.approx(1.0)              # Feb as known Apr 14: 101 vs 100
    assert r["rs_g0"] is None                             # March retail not out until Apr 16
    assert r["target"] == pytest.approx(1.0)              # first release: 99.99 vs 99 (both from Apr 15 vintage)


def test_leaky_uses_todays_revised_data(db):
    lk = features.build(db, "leaky", start="2025-01-01")
    r = row(lk, date(2025, 3, 1))
    assert r["ip_g1"] == pytest.approx(-1.0)              # Feb as revised later: 99 vs 100
    assert r["rs_g0"] == pytest.approx(2.0)               # retail for March: a value that did not exist yet


def test_no_honest_input_postdates_its_forecast(db):
    h = features.build(db, "honest", start="2025-01-01")
    assert h.filter(pl.col("max_input_published") > pl.col("forecast_date")).height == 0


def test_walk_forward_refuses_leaked_honest_rows(db):
    h = features.build(db, "honest", start="2025-01-01").with_columns(
        pl.col("forecast_date").dt.offset_by("-60d"))       # pretend forecasts were made earlier: now inputs leak
    with pytest.raises(AssertionError, match="published after"):
        experiment.walk_forward(h, features.build(db, "leaky", start="2025-01-01"))


def test_skill_is_relative_to_baseline_on_same_answer_key():
    import numpy as np
    y = np.array([1.0, -1.0, 1.0, -1.0])
    s = experiment.scores(y, y * 0.5, np.zeros(4))
    assert s["skill"] == pytest.approx(0.5)
    assert s["direction"] == 1.0
