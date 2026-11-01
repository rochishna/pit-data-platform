# Point-in-Time Data Platform

Stores every published version of US economic data, so any question can be answered **as of** a past date
using only what was known then, and measures how much lookahead bias inflates a forecast that ignores this.

> **Result:** a forecast trained and tested on revised data claims **+23.9%** skill over a historical-average
> guess. Given only what was published on each forecast day, the same model delivers **+14.3%**; trained honestly,
> **+2.9%**. Outside the 2020 shock, nothing beats the average, which a leaky backtest would have hidden.
> Full write-up: [docs/LOOKAHEAD.md](docs/LOOKAHEAD.md).

## Why

Economic statistics are revised after release, sometimes repeatedly. A backtest built on today's revised
numbers uses information nobody had at the time (lookahead bias) and looks better than reality.

## Series

Target: **industrial production** (INDPRO). Inputs include **retail sales** (RSAFS), capacity utilization,
unemployment and payrolls. See [`configs/series.yaml`](configs/series.yaml).

## Getting the data

The FRED API key stays on your machine: set it as an environment variable (or in a gitignored `.env`).
The downloader reads it at runtime and redacts it from every error message.

```bash
pip install -e ".[dev]"
export FRED_API_KEY=your_key_here          # Windows PowerShell: $env:FRED_API_KEY="your_key_here"
python -m pit.download                     # every vintage of every series -> data/raw/<ID>.parquet
```

## Storage and as-of queries

```bash
python -m pit.store build        # data/raw -> data/pit.duckdb, after integrity checks
```

Each stored value carries two time axes: the period it describes (`observation_date`) and the span during which
it was the published figure (`realtime_start`..`realtime_end`). `PointInTime.as_of(series, date)` returns a
series exactly as it was published on that date, and nothing published later.

| Guarantee | Proof |
|---|---|
| An as-of query never returns a value published after its date | `test_never_sees_the_future_randomized` (500 dates); on real data, 1,000 random queries returned 643,682 values with 0 leaks |
| Release-day boundaries are exact | `test_boundary_days_are_inclusive` |
| Broken histories are refused: overlapping versions, end-before-start, duplicate keys | `test_overlapping_versions_rejected` and siblings |
| Rebuilding is idempotent | `test_rebuild_is_idempotent` |

Letting the query see even one day ahead fails three tests.

## What the real data shows

**Revisions are routine, and can be large.** Industrial production for March 2025 was first published as 103.89
(Apr 16, 2025), revised five times to 103.54, then reported as 101.04 from Nov 24, 2025. Retail sales for the same
month went from $734.9B on release, to $722.0B nine days later, to $694.8B in the Sep 28, 2026 vintage.

**Release dates are not a fixed schedule.** No series published anything in October 2025; every one jumped from
mid-September to late November, consistent with the October–November 2025 federal shutdown. A backtest that
assumes data arrives mid-month would have used numbers that did not exist yet. This store uses actual release dates.

## The lookahead experiment

```bash
python -m pit.experiment         # writes docs/LOOKAHEAD.md
```

Each month, forecast industrial production growth the day before its first release, and score it against that
first release, the number known and traded on release day. Three setups share one model and one set of months
(Jan 2008 – Sep 2026, walk-forward, yearly refit):

| Setup | Trained on | Inputs at forecast time | Scored against |
|---|---|---|---|
| Leaky, as reported | today's revised data | today's revised data | revised growth |
| Leaky, run for real | today's revised data | as published that day | first release |
| Honest | as published, first-release targets already out | as published that day | first release |

| Ridge regression | All months | Excluding Mar–Jul 2020 |
|---|---:|---:|
| Leaky, as reported | +23.9% | +1.9% |
| Leaky, run for real | +14.3% | −2.4% |
| Honest | +2.9% | −0.9% |

Skill = 1 − RMSE(model) / RMSE(historical average), on the same answer key. Gradient-boosted trees were run first
and overfit the small sample (~80–250 rows per refit); their results are reported alongside. Honest rows record
the newest publication date among their inputs, and the run aborts if any postdates the forecast day.
