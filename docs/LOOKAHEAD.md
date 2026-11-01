# Lookahead Bias Experiment

Forecast month-over-month industrial production growth the day before its first release, Jan 2008 to Sep 2026 (225 months), walk-forward with a yearly refit. Scored against the **first release**, the number known and traded on release day (decision: Rochishna).

**Skill** = 1 − RMSE(model) / RMSE(historical-average guess), both scored against the same answer key. Positive means better than guessing the average. Raw RMSE is not comparable across answer keys: revisions made the 2020 crash larger (April 2020: −11.2% on first release, −13.2% revised).

## Result

- **Lookahead inflates the backtest.** Trained and tested on revised data, the model claims **+23.9%** skill. The same model given only what was published on each forecast day delivers **+14.3%**. Trained honestly: **+2.9%**.
- **Outside the COVID shock, nothing beats the average.** Excluding Mar–Jul 2020, the leaky backtest still claims +1.9%, while every real-time setup lands at or below zero (honest -0.9%). With these inputs, monthly industrial production growth is essentially unpredictable the day before release, and a leaky backtest would have hidden that.
- **Not every comparison favors honesty, and that is reported too.** Over all months the leaky-trained model beats the honest one even on real-time data (+14.3% vs +2.9%); the gap comes from the 2020 shock months. Outside them the order flips (-0.9% honest vs -2.4%), and both are within noise of zero.

## Ridge regression (primary)

All months:

| Setup | RMSE (pp) | Baseline RMSE | Skill vs baseline | Direction right |
|---|---:|---:|---:|---:|
| Leaky, as its backtest reports it | 0.983 | 1.291 | +23.9% | 58% |
| Leaky, run on real-time data | 0.976 | 1.140 | +14.3% | 60% |
| Honest (point-in-time) | 1.106 | 1.140 | +2.9% | 63% |
| Naive: last month's growth | 1.392 | 1.140 | -22.1% | 52% |

Excluding Mar–Jul 2020:

| Setup | RMSE (pp) | Baseline RMSE | Skill vs baseline | Direction right |
|---|---:|---:|---:|---:|
| Leaky, as its backtest reports it | 0.740 | 0.755 | +1.9% | 57% |
| Leaky, run on real-time data | 0.670 | 0.655 | -2.4% | 60% |
| Honest (point-in-time) | 0.661 | 0.655 | -0.9% | 62% |
| Naive: last month's growth | 0.913 | 0.655 | -39.5% | 52% |

## Gradient-boosted trees (run first, reported for completeness)

The trees overfit: about 80–250 training rows per refit with 9 inputs. Ridge, the standard choice for samples this small, became the primary model. This switch was made after seeing these results, and is disclosed here rather than hidden.

All months:

| Setup | RMSE (pp) | Baseline RMSE | Skill vs baseline | Direction right |
|---|---:|---:|---:|---:|
| Leaky, as its backtest reports it | 1.230 | 1.291 | +4.7% | 57% |
| Leaky, run on real-time data | 1.083 | 1.140 | +5.0% | 57% |
| Honest (point-in-time) | 1.096 | 1.140 | +3.8% | 64% |
| Naive: last month's growth | 1.392 | 1.140 | -22.1% | 52% |

Excluding Mar–Jul 2020:

| Setup | RMSE (pp) | Baseline RMSE | Skill vs baseline | Direction right |
|---|---:|---:|---:|---:|
| Leaky, as its backtest reports it | 0.806 | 0.755 | -6.9% | 56% |
| Leaky, run on real-time data | 0.686 | 0.655 | -4.7% | 57% |
| Honest (point-in-time) | 0.699 | 0.655 | -6.7% | 63% |
| Naive: last month's growth | 0.913 | 0.655 | -39.5% | 52% |

## Why it is honest

- Every honest input is read as published on the forecast day; each row records the newest publication date among its inputs, and the run aborts if any is later than the forecast date.
- The honest model trains only on months whose first release was out by the refit day.
- Growth is computed within a single vintage; index levels from different vintages are never compared.
- Hyperparameters are fixed up front, not tuned on the test period.
- Real data gap: November 2025 has no prior-month value as of its forecast day, because the Oct–Nov 2025 shutdown meant October was not yet published. The model imputes it; the naive baseline skips that month.

Reproduce: `python -m pit.experiment`
