# Results

## Research question

The project asks whether **China-focused economic and financial news sentiment contains incremental information for forecasting the next CSI 300 trading-session direction** beyond market-only variables.

The planned predictive comparison is:

- **Market-only model:** volatility, momentum, momentum acceleration and an ex-ante volatility indicator.
- **Market + sentiment model:** the same market variables plus rolling FinBERT sentiment, sentiment dispersion, news intensity and sentiment-volatility interactions.

Before estimating that comparison, the repository reports descriptive evidence directly from the committed sample.

## Sample and market environment

| Item | Current evidence |
|---|---:|
| News days | 49 |
| Sentiment days | 49 |
| News period | 22 Apr 2026 – 6 Jul 2026 |
| GDELT title/headline observations scored by FinBERT | 6,919 |
| Mean title/headline observations per news day | 141.2 |
| Median title/headline observations per news day | 150 |
| Mean daily FinBERT sentiment | +0.0551 |
| CSI 300 observations in committed market file | 221 |

For the **49 CSI 300 trading sessions from 22 April through 3 July 2026**:

| Market statistic | Value |
|---|---:|
| Up sessions | 25 |
| Down sessions | 24 |
| Compounded return from the stored session log returns | **+1.14%** |
| First-close to last-close price change | **+0.89%** |
| Annualized realized volatility | **20.78%** |
| Mean 20-day daily volatility estimate | **1.13%** |

The two return figures use different boundaries: the compounded session-return figure includes the stored return for 22 April, which is measured from the preceding trading close, whereas the first-close to last-close figure begins at the 22 April closing level.

## Trading-session alignment

The end-of-day alignment produces:

| Alignment statistic | Value |
|---|---:|
| CSI 300 sessions in the aligned window | 49 |
| Sessions with usable news assigned | 31 |
| Sessions without usable news coverage | 18 |
| Sentiment / next-session-return pairs | 31 |

Weekend and holiday news is assigned to the next available trading-session information set. Missing news coverage remains missing rather than being converted to neutral sentiment.

## Descriptive sentiment-return evidence

Using the 31 aligned observations, the **Spearman time-series rank correlation** between session-level sentiment and the next-session CSI 300 return is:

**Spearman correlation = -0.348**

The corresponding linear correlation is:

**Pearson correlation = -0.246**

These are descriptive statistics from a short sample; no statistical-significance or causal claim is made.

The negative association means that higher raw sentiment did not correspond monotonically to higher next-session CSI 300 returns in this sample.

### Sentiment-sorted next-session returns

| Sentiment group | Observations | Mean sentiment | Mean next-session CSI 300 return | Next session positive |
|---|---:|---:|---:|---:|
| Lowest-sentiment tercile | 10 | +0.0177 | **+0.305%** | **70.0%** |
| Middle tercile | 10 | +0.0610 | **+0.011%** | **50.0%** |
| Highest-sentiment tercile | 11 | +0.1236 | **-0.161%** | **27.3%** |

The descriptive high-minus-low return difference is approximately **-0.466 percentage points**.

A simple rule that predicts next-session direction only from the sign of raw sentiment is correct on **45.2%** of the 31 aligned observations.

Taken together, these statistics do not support a simple rule that more positive news is followed by a higher next-session market return. They motivate, rather than establish, the hypothesis that news may be more informative when considered jointly with momentum, volatility and attention.

## Predictive-model evaluation protocol

The feature set uses rolling sentiment measures, sentiment momentum and dispersion, news intensity, market momentum, sentiment-volatility interactions and an ex-ante volatility indicator.

After rolling-feature construction and next-session target formation, the current sample contains **8 labelled model rows**. This is insufficient for the pre-specified walk-forward evaluation.

The research configuration separates two thresholds:

- **60 labelled observations** are required for the initial expanding training window.
- At least **30 genuine OOS forecasts** are required before model-performance statistics are published.

The 30-observation rule is a conservative reporting threshold, not a claim that 30 forecasts are sufficient to establish a durable effect. With the current sample, the repository therefore does not report XGBoost accuracy, Sharpe ratio, active return or strategy profitability.

When the threshold is reached, the primary model question is whether the **market + sentiment** XGBoost model improves on the **market-only** XGBoost baseline out of sample.

## Risk and economic evaluation

GARCH(1,1) is estimated separately from the directional model and provides one-step-ahead volatility forecasts for position scaling.

The paper-strategy rule uses:

- p(up) >= 0.55 → LONG;
- p(up) <= 0.45 → SHORT;
- otherwise → NO TRADE;
- absolute position size capped at **1.0x**;
- **10 bps** proportional transaction cost per unit of turnover.

Market log returns are converted to simple returns before position weighting, transaction costs and wealth compounding.

When enough OOS observations exist, the evaluation reports:

- cumulative strategy return;
- CSI 300 benchmark return;
- benchmark-relative active return;
- annualized Sharpe ratio under a zero risk-free-rate convention;
- maximum drawdown;
- directional hit rate on active signals;
- average turnover;
- transaction-cost drag.

The project title uses “Alpha,” but **Jensen's alpha is not claimed** without an explicit asset-pricing regression.

## Reproducibility

Run:

~~~bash
python run_research_pipeline.py
~~~

With the current committed sample, the command regenerates:

~~~text
outputs/descriptive_summary.csv
outputs/sentiment_terciles.csv
data/final_dataset.csv
~~~

The generated output files are excluded from version control so that stale local outputs are not presented as current evidence.

After the pre-specified OOS reporting threshold is reached, the same pipeline also produces:

~~~text
outputs/model_evaluation.csv
outputs/oos_predictions.csv
outputs/garch_oos_forecasts.csv
outputs/oos_backtest.csv
outputs/oos_backtest_metrics.csv
~~~

## Conclusion

The current evidence supports a limited descriptive conclusion: **raw China-focused news sentiment is not a simple monotonic positive next-session signal for the CSI 300 in the committed sample**.

The negative rank association and sentiment-sorted return pattern motivate a pre-specified test of whether rolling sentiment, investor-attention proxies and sentiment-volatility interactions add incremental predictive information beyond market-only variables. No persistent forecasting advantage, causal effect or profitable trading strategy is claimed from the current sample.
