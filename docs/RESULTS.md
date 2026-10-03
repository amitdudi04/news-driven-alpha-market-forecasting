# Results and Current Empirical Status

This file summarizes the results that are supported by the **current public data and code** in this repository. It separates descriptive evidence from model-performance claims so that a reader can see what has been established and what still requires more data.

## 1. Research question

The project asks whether financial-news sentiment contains incremental information for forecasting the **next CSI 300 trading-session direction** beyond market-only variables such as momentum and volatility.

The empirical comparison is therefore:

- **Market-only model:** volatility, momentum, momentum acceleration and the ex-ante volatility indicator.
- **Market + sentiment model:** the same market variables plus rolling FinBERT sentiment, sentiment dispersion, news intensity and sentiment-volatility interaction terms.

The finance question is not simply whether XGBoost can fit the data. It is whether the alternative-data signal adds out-of-sample information after a market-only benchmark is already present.

## 2. Current clean data sample

The committed clean news/sentiment history begins on **22 April 2026**. Earlier non-empirical development seed rows are not part of the research sample.

| Item | Current committed evidence |
|---|---:|
| Daily news observations | 49 |
| Daily sentiment observations | 49 |
| Clean news period | 22 Apr 2026 – 6 Jul 2026 |
| Article-level sentiment inputs | 6,919 |
| Mean articles per news day | 141.2 |
| Median articles per news day | 150 |
| Mean daily FinBERT sentiment | +0.0551 |
| Positive-sentiment days | 39 |
| Negative-sentiment days | 10 |
| CSI 300 observations in committed market file | 221 |
| CSI 300 market-file period | 4 Aug 2025 – 3 Jul 2026 |

The 49 clean news days and 49 sentiment days contain no duplicate dates, and the committed-data tests verify that the known development seed patterns are absent.

## 3. Market environment during the clean news window

For the CSI 300 trading sessions from **22 April 2026 through 3 July 2026**:

| Market statistic | Value |
|---|---:|
| Trading sessions | 49 |
| Cumulative CSI 300 return | +1.56% |
| Annualized realized return volatility | 20.78% |
| Mean 20-day daily volatility estimate | 1.13% |

These are descriptive market statistics. They are not strategy returns and should not be interpreted as evidence that the sentiment model produced alpha.

## 4. Trading-session alignment result

The cleaned pipeline maps calendar-day news to actual CSI 300 trading sessions. It does not create weekend market rows and it does not treat missing news coverage as neutral sentiment.

Within the clean market window:

| Alignment statistic | Value |
|---|---:|
| CSI 300 trading sessions considered | 49 |
| Trading sessions with news assigned | 31 |
| Trading sessions with missing news coverage | 18 |

Weekend and holiday news can be included in the next available trading-session information set. A trading interval with no committed news observation remains missing.

## 5. Feature-set availability

The current feature design uses rolling 5-day, 10-day and 20-day sentiment statistics together with market momentum and volatility.

The repository's CI smoke test currently produces:

| Feature-status measure | Value |
|---|---:|
| Usable labelled feature rows | 8 |
| Initial training rows required before public OOS evaluation | 60 |
| Minimum rows required for first genuine OOS prediction | 61 |
| Current shortfall | 53 |

The latest GitHub Actions research run successfully built the feature dataset and reported **8 labelled rows**, then stopped before model evaluation because the public minimum-sample condition was not met.

This stop is intentional. The code does not manufacture a performance table from an undersized sample.

## 6. Current model-performance result

**No headline out-of-sample forecasting or trading-performance result is currently claimed.**

The repository therefore does not publish:

- directional accuracy;
- balanced accuracy;
- Brier score;
- log loss;
- strategy cumulative return;
- Sharpe ratio;
- maximum drawdown;
- active return versus the CSI 300;
- turnover-adjusted profitability;

for the corrected clean sample at this stage.

An earlier development trace contained unusually strong short-window figures. Those numbers are not carried forward as research evidence because the earlier period overlapped non-empirical development seed data and preceded the current time-series safeguards.

Current conclusion:

> **The research pipeline is validated at the data-alignment and methodology level, but the clean sample is not yet large enough to support a credible claim about persistent predictive alpha or trading profitability.**

## 7. Finance interpretation

### Alternative data and price discovery

The project tests whether unstructured financial news contributes information beyond standard market variables. This connects directly to price discovery, market efficiency and the use of alternative data in investment research.

A positive future result would mean that the **market + sentiment** model improves out-of-sample forecasts relative to the **market-only** model on the tested sample. It would not by itself prove causality or a permanent market inefficiency.

### Benchmark and active return

The CSI 300 is the economic benchmark.

For future strategy reporting:

[
	ext{Active Return} = R_{	ext{strategy}} - R_{	ext{CSI300}}
]

The repository uses **News-Driven Alpha** as the project name, but benchmark-relative return should be described as **active return** unless a formal asset-pricing regression is used to estimate statistical alpha. The project does not currently claim Jensen's alpha.

### Risk-adjusted performance

Once sufficient clean observations exist, the strategy evaluation will report:

- cumulative strategy return;
- CSI 300 benchmark return;
- active return;
- annualized Sharpe ratio;
- maximum drawdown;
- average turnover;
- transaction-cost drag.

The paper-strategy rule includes a **10 bps proportional transaction cost per unit of turnover** and caps absolute position size at **1.0x**.

### GARCH risk overlay

GARCH(1,1) is used for **one-step-ahead volatility forecasting and position scaling**. It is not treated as the source of directional alpha.

This distinction matters financially:

- XGBoost asks **which direction?**
- GARCH asks **how much risk is forecast?**
- the signal engine asks **how large should the paper position be?**

## 8. Result table that will be populated when the sample is sufficient

The research pipeline is designed to generate the following comparison.

### Directional-model evaluation

| Metric | Market only | Market + sentiment | Interpretation |
|---|---:|---:|---|
| Accuracy | Pending | Pending | Fraction of correct next-session signs |
| Balanced accuracy | Pending | Pending | Directional accuracy adjusted for class imbalance |
| Brier score | Pending | Pending | Probability calibration error; lower is better |
| Log loss | Pending | Pending | Penalizes confident wrong probabilities; lower is better |

The important research quantity is the **incremental improvement from adding sentiment**, not the standalone complexity of the model.

### Paper-strategy evaluation

| Metric | Strategy | CSI 300 benchmark |
|---|---:|---:|
| Cumulative return | Pending | Pending |
| Annualized Sharpe | Pending | Pending |
| Maximum drawdown | Pending | Pending |
| Active return | Pending | — |
| Average turnover | Pending | — |
| Transaction-cost drag | Pending | — |

These values will only be published after the corrected expanding-window evaluation has enough clean labelled observations.

## 9. Reproducibility

Run:

```bash
python run_research_pipeline.py
```

The current committed sample builds the feature dataset and then exits before publishing OOS results because the minimum-sample requirement has not yet been reached.

When sufficient history is available, the same command will generate:

```text
outputs/model_evaluation.csv
outputs/oos_predictions.csv
outputs/garch_oos_forecasts.csv
outputs/oos_backtest.csv
outputs/oos_backtest_metrics.csv
```

Generated result files remain outside version control so that an old local run cannot be mistaken for the current research result.

## 10. Current conclusion

At this stage, the project provides a **time-aligned alternative-data research design** for Chinese equity forecasting:

- GDELT provides the news stream;
- FinBERT converts text into financial-tone features;
- CSI 300 market variables provide the conventional information set;
- expanding-window XGBoost is designed to test whether sentiment adds predictive information;
- GARCH provides a separate volatility forecast for risk scaling;
- turnover costs are incorporated into the paper-strategy design.

The economic hypothesis remains open until a longer clean news history produces enough out-of-sample observations. That limitation is reported directly rather than replaced with an overstated short-sample performance claim.
