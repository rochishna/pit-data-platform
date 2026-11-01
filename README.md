# Point-in-Time Data Platform

Stores every published version of US economic data, so any question can be answered **as of** a past date
using only what was known then, and measures how much lookahead bias inflates a forecast that ignores this.

> **Status:** Stage 1, the vintage downloader. Bitemporal storage, as-of queries and the forecast come next.

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
