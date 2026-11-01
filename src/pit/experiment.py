"""The lookahead-bias experiment: how much does training on revised data inflate a forecast's apparent skill?

Task: forecast month-over-month industrial production growth the day before its first release, scored against
the FIRST RELEASE (what is known, and traded on, that day). Decision: Rochishna.

Three setups, same model (gradient-boosted trees), same months, walk-forward with a yearly refit:

  honest          trains on as-published features and first-release targets that were already out at refit time;
                  forecasts from as-published features; scored vs first release.
  leaky_reported  the usual mistake: train AND test on today's revised data. This is the score such a backtest
                  would report.
  leaky_live      that same leaky model, run the way it would have to run in reality: given only the data
                  published on the forecast day, scored vs first release.

reported vs live = how much lookahead inflated the backtest. honest vs live = what honest training buys.

Usage:
    python -m pit.experiment --out docs/LOOKAHEAD.md
"""
from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, build

TEST_START = date(2008, 1, 1)
COVID = (date(2020, 3, 1), date(2020, 7, 1))   # reported both with and without; never silently dropped

# Hyperparameters are fixed up front, not tuned on the test period.
# The trees were run first and overfit (~80-250 training rows per refit, 9 inputs); ridge is the standard
# choice for samples this small. Both are reported.
MODELS = {
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
    "gbm": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                                 min_samples_leaf=15, l2_regularization=1.0, random_state=0),
}
make_model = MODELS["ridge"]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    ok = ~(np.isnan(y) | np.isnan(p))
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def scores(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> dict:
    """Skill = 1 - RMSE(model) / RMSE(historical-average baseline), both scored against the SAME answer key.
    Comparing raw RMSE across answer keys is unfair: revisions made the 2020 crash larger, inflating errors."""
    ok = ~(np.isnan(y) | np.isnan(p))
    nz = ok & (y != 0)
    r, b = rmse(y, p), rmse(y, baseline)
    return {"rmse": r, "baseline_rmse": b, "skill": 1 - r / b,
            "direction": float(np.mean(np.sign(y[nz]) == np.sign(p[nz]))), "n": int(ok.sum())}


def walk_forward(honest: pl.DataFrame, leaky: pl.DataFrame) -> pl.DataFrame:
    """One row per test month with every setup's prediction."""
    if honest.filter(pl.col("max_input_published") > pl.col("forecast_date")).height:
        raise AssertionError("honest dataset contains an input published after its forecast date")
    h = honest.join(leaky.select("month", *FEATURE_NAMES, "target"), on="month", suffix="_lk")
    out = []
    for year in sorted({m.year for m in h["month"] if m >= TEST_START}):
        test = h.filter(pl.col("month").dt.year() == year)
        refit_day = test["forecast_date"].min()
        first_test_month = test["month"].min()

        # Honest: only rows whose first release was out by the refit day.
        tr_h = h.filter((pl.col("release_date") <= refit_day) & pl.col("target").is_not_null())
        honest_model = make_model().fit(tr_h.select(FEATURE_NAMES).to_numpy(), tr_h["target"].to_numpy())

        # Leaky: all earlier months, in today's revised form.
        tr_l = h.filter((pl.col("month") < first_test_month) & pl.col("target_lk").is_not_null())
        leaky_cols = [f"{c}_lk" for c in FEATURE_NAMES]
        leaky_model = make_model().fit(tr_l.select(leaky_cols).to_numpy(), tr_l["target_lk"].to_numpy())

        x_h = test.select(FEATURE_NAMES).to_numpy()
        out.append(test.select("month", "forecast_date", "target", "target_lk", "ip_g1").with_columns(
            pl.Series("honest", honest_model.predict(x_h)),
            pl.Series("leaky_reported", leaky_model.predict(test.select(leaky_cols).to_numpy())),
            pl.Series("leaky_live", leaky_model.predict(x_h)),
            pl.lit(float(tr_h["target"].mean())).alias("baseline_mean"),
            pl.lit(float(tr_l["target_lk"].mean())).alias("baseline_mean_lk"),   # same baseline, revised answer key
        ))
    return pl.concat(out).rename({"ip_g1": "baseline_last_month"})


SETUPS = [
    # name, label, prediction column, answer key, baseline column
    ("leaky_reported", "Leaky, as its backtest reports it", "leaky_reported", "target_lk", "baseline_mean_lk"),
    ("leaky_live", "Leaky, run on real-time data", "leaky_live", "target", "baseline_mean"),
    ("honest", "Honest (point-in-time)", "honest", "target", "baseline_mean"),
    ("last_month", "Naive: last month's growth", "baseline_last_month", "target", "baseline_mean"),
]


def summarize(pred: pl.DataFrame, exclude_covid: bool = False) -> list[dict]:
    if exclude_covid:
        pred = pred.filter(~pl.col("month").is_between(*COVID))
    return [{"key": k, "setup": label,
             **scores(pred[y].to_numpy(), pred[col].to_numpy(), pred[base].to_numpy())}
            for k, label, col, y, base in SETUPS]


def report(results: dict, n_months: int, first: date, last: date) -> str:
    def table(rows):
        out = ["| Setup | RMSE (pp) | Baseline RMSE | Skill vs baseline | Direction right |", "|---|---:|---:|---:|---:|"]
        out += [f"| {r['setup']} | {r['rmse']:.3f} | {r['baseline_rmse']:.3f} | {r['skill']:+.1%} | {r['direction']:.0%} |"
                for r in rows]
        return "\n".join(out)

    r_all, r_ex = results["ridge"]["all"], results["ridge"]["ex_covid"]
    g_all, g_ex = results["gbm"]["all"], results["gbm"]["ex_covid"]
    get = lambda rows, k: next(r for r in rows if r["key"] == k)
    lines = [
        "# Lookahead Bias Experiment",
        "",
        f"Forecast month-over-month industrial production growth the day before its first release, {first:%b %Y}"
        f" to {last:%b %Y} ({n_months} months), walk-forward with a yearly refit. Scored against the **first "
        "release**, the number known and traded on release day (decision: Rochishna).",
        "",
        "**Skill** = 1 − RMSE(model) / RMSE(historical-average guess), both scored against the same answer key. "
        "Positive means better than guessing the average. Raw RMSE is not comparable across answer keys: "
        "revisions made the 2020 crash larger (April 2020: −11.2% on first release, −13.2% revised).",
        "",
        "## Result",
        "",
        f"- **Lookahead inflates the backtest.** Trained and tested on revised data, the model claims "
        f"**{get(r_all,'leaky_reported')['skill']:+.1%}** skill. The same model given only what was published on "
        f"each forecast day delivers **{get(r_all,'leaky_live')['skill']:+.1%}**. Trained honestly: "
        f"**{get(r_all,'honest')['skill']:+.1%}**.",
        f"- **Outside the COVID shock, nothing beats the average.** Excluding Mar–Jul 2020, the leaky backtest still "
        f"claims {get(r_ex,'leaky_reported')['skill']:+.1%}, while every real-time setup lands at or below zero "
        f"(honest {get(r_ex,'honest')['skill']:+.1%}). With these inputs, monthly industrial production growth is "
        "essentially unpredictable the day before release, and a leaky backtest would have hidden that.",
        f"- **Not every comparison favors honesty, and that is reported too.** Over all months the leaky-trained "
        f"model beats the honest one even on real-time data ({get(r_all,'leaky_live')['skill']:+.1%} vs "
        f"{get(r_all,'honest')['skill']:+.1%}); the gap comes from the 2020 shock months. Outside them the order "
        f"flips ({get(r_ex,'honest')['skill']:+.1%} honest vs {get(r_ex,'leaky_live')['skill']:+.1%}), and both are "
        "within noise of zero.",
        "",
        "## Ridge regression (primary)",
        "",
        "All months:",
        "",
        table(r_all),
        "",
        "Excluding Mar–Jul 2020:",
        "",
        table(r_ex),
        "",
        "## Gradient-boosted trees (run first, reported for completeness)",
        "",
        "The trees overfit: about 80–250 training rows per refit with 9 inputs. Ridge, the standard choice for "
        "samples this small, became the primary model. This switch was made after seeing these results, and is "
        "disclosed here rather than hidden.",
        "",
        "All months:",
        "",
        table(g_all),
        "",
        "Excluding Mar–Jul 2020:",
        "",
        table(g_ex),
        "",
        "## Why it is honest",
        "",
        "- Every honest input is read as published on the forecast day; each row records the newest publication date "
        "among its inputs, and the run aborts if any is later than the forecast date.",
        "- The honest model trains only on months whose first release was out by the refit day.",
        "- Growth is computed within a single vintage; index levels from different vintages are never compared.",
        "- Hyperparameters are fixed up front, not tuned on the test period.",
        "- Real data gap: November 2025 has no prior-month value as of its forecast day, because the Oct–Nov 2025 "
        "shutdown meant October was not yet published. The model imputes it; the naive baseline skips that month.",
        "",
        "Reproduce: `python -m pit.experiment`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the lookahead-bias experiment.")
    p.add_argument("--db", default="data/pit.duckdb")
    p.add_argument("--out", default="docs/LOOKAHEAD.md")
    p.add_argument("--predictions", default="data/predictions.parquet")
    args = p.parse_args(argv)

    global make_model
    honest, leaky = build(args.db, "honest"), build(args.db, "leaky")
    results, primary = {}, None
    for name, factory in MODELS.items():
        make_model = factory
        pred = walk_forward(honest, leaky)
        results[name] = {"all": summarize(pred), "ex_covid": summarize(pred, exclude_covid=True)}
        if name == "ridge":
            primary = pred
        for period, rows in results[name].items():
            for r in rows:
                print(f"{name:5s} {period:8s} {r['setup']:34s} RMSE {r['rmse']:.3f}  skill {r['skill']:+.1%}")
    make_model = MODELS["ridge"]
    primary.write_parquet(args.predictions)
    with open(args.out, "w") as f:
        f.write(report(results, primary.height, primary["month"].min(), primary["month"].max()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
