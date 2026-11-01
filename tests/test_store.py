"""Bitemporal store tests. The central guarantee: an as-of query never sees the future."""
import random
from datetime import date, timedelta

import polars as pl
import pytest

from pit import store

FOREVER = date(9999, 12, 31)


def rows(*specs):
    """specs: (observation_date, value, realtime_start, realtime_end)"""
    return pl.DataFrame(
        {"series_id": ["X"] * len(specs),
         "observation_date": [s[0] for s in specs], "value": [s[1] for s in specs],
         "realtime_start": [s[2] for s in specs], "realtime_end": [s[3] for s in specs]},
        schema={"series_id": pl.Utf8, "observation_date": pl.Date, "value": pl.Float64,
                "realtime_start": pl.Date, "realtime_end": pl.Date})


# March is first published Apr 16 as 100, revised May 15 to 101, then rebased Nov 24 to 98.
# April is first published May 15.
HISTORY = rows(
    (date(2025, 3, 1), 100.0, date(2025, 4, 16), date(2025, 5, 14)),
    (date(2025, 3, 1), 101.0, date(2025, 5, 15), date(2025, 11, 23)),
    (date(2025, 3, 1), 98.0, date(2025, 11, 24), FOREVER),
    (date(2025, 4, 1), 102.0, date(2025, 5, 15), FOREVER),
)


@pytest.fixture
def pit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    HISTORY.write_parquet(raw / "X.parquet")
    db = tmp_path / "pit.duckdb"
    store.build(raw, db)
    return store.PointInTime(db)


def values(df):
    return dict(zip(df["observation_date"].to_list(), df["value"].to_list()))


def test_as_of_returns_the_version_in_force(pit):
    assert values(pit.as_of("X", "2025-04-20")) == {date(2025, 3, 1): 100.0}
    assert values(pit.as_of("X", "2025-06-01")) == {date(2025, 3, 1): 101.0, date(2025, 4, 1): 102.0}
    assert values(pit.as_of("X", "2026-01-01")) == {date(2025, 3, 1): 98.0, date(2025, 4, 1): 102.0}


def test_nothing_exists_before_first_publication(pit):
    assert pit.as_of("X", "2025-04-15").height == 0


def test_boundary_days_are_inclusive(pit):
    assert values(pit.as_of("X", "2025-05-14"))[date(2025, 3, 1)] == 100.0     # last day of version 1
    assert values(pit.as_of("X", "2025-05-15"))[date(2025, 3, 1)] == 101.0     # release day of version 2


def test_never_sees_the_future_randomized(pit):
    """For 500 random as-of dates, every returned value was published on or before that date."""
    rng = random.Random(7)
    for _ in range(500):
        d = date(2025, 1, 1) + timedelta(days=rng.randrange(0, 500))
        df = pit.as_of("X", d)
        assert all(p <= d for p in df["published"].to_list()), d
        assert df["observation_date"].n_unique() == df.height          # one version per observation


def test_latest_and_first_release(pit):
    assert values(pit.latest("X")) == {date(2025, 3, 1): 98.0, date(2025, 4, 1): 102.0}
    assert values(pit.first_release("X")) == {date(2025, 3, 1): 100.0, date(2025, 4, 1): 102.0}


def test_vintages(pit):
    assert pit.vintages("X") == [date(2025, 4, 16), date(2025, 5, 15), date(2025, 11, 24)]


def test_rebuild_is_idempotent(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    HISTORY.write_parquet(raw / "X.parquet")
    db = tmp_path / "pit.duckdb"
    store.build(raw, db)
    assert store.build(raw, db) == {"X": 4}
    assert store.PointInTime(db).con.execute("SELECT count(*) FROM observations").fetchone()[0] == 4


# ---------- integrity checks refuse broken histories ----------

def test_overlapping_versions_rejected():
    bad = rows((date(2025, 3, 1), 100.0, date(2025, 4, 16), date(2025, 5, 20)),     # runs past the next start
               (date(2025, 3, 1), 101.0, date(2025, 5, 15), FOREVER))
    with pytest.raises(store.IntegrityError, match="overlapping"):
        store.validate(bad)


def test_end_before_start_rejected():
    with pytest.raises(store.IntegrityError, match="end before"):
        store.validate(rows((date(2025, 3, 1), 1.0, date(2025, 5, 1), date(2025, 4, 1))))


def test_duplicate_key_rejected():
    dup = rows((date(2025, 3, 1), 1.0, date(2025, 4, 16), FOREVER),
               (date(2025, 3, 1), 2.0, date(2025, 4, 16), FOREVER))
    with pytest.raises(store.IntegrityError):
        store.validate(dup)
