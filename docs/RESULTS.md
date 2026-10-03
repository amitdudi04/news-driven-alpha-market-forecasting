# Results

## Research question

The project tests whether financial-news sentiment contains useful information for forecasting the **next CSI 300 trading-session direction** beyond market-only variables such as momentum and volatility.

The finance comparison is therefore:

- **Market-only information set:** volatility, momentum, momentum acceleration and an ex-ante volatility indicator.
- **Market + sentiment information set:** the same market variables plus rolling FinBERT sentiment, sentiment dispersion, news intensity and sentiment-volatility interaction terms.

The important question is not whether XGBoost can fit a short sample. It is whether the alternative-data signal adds information beyond a conventional market baseline.

## Clean research sample

The committed empirical news/sentiment sample begins on **22 April 2026**.

| Item | Current evidence |
|---|---:|
| Daily news observations | 49 |
| Daily sentiment observations | 49 |
| Clean news period | 22 Apr 2026 – 6 Jul 2026 |
| Article/headline inputs scored by FinBERT | 6,919 |
| Mean articles per news day | 141.2 |
| Median articles per news day | 150 |
| Mean daily FinBERT sentiment | +0.0551 |
| CSI 300 observations in committed market file | 221 |

Earlier development seed rows are not included in these figures.

## CSI 300 market environment

For CSI 300 trading sessions from **22 April 2026 through 3 July 2026**:

| Market statistic | Value |
|---|---:|
| Trading sessions | 49 |
| Up sessions | 25 |
| Down sessions | 24 |
| Cumulative CSI 300 return | **+1.56%** |
| Annualized realized volatility | **20.78%** |
| Mean 20-day daily volatility estimate | **1.13%** |

This was a mildly positive but volatile market window rather than a one-directional rally.

## News-to-market alignment

The trading-session mapping produced:

| Alignment statistic | Value |
|---|---:|
| CSI 300 sessions in the clean window | 49 |
| Sessions with usable news assigned | 31 |
| Sessions without usable news coverage | 18 |
| Usable sentiment / next-session-return pairs | 31 |

Weekend and holiday news is assigned to the next available trading-session information set. Missing news intervals remain missing rather than being converted to neutral sentiment.

## Exploratory finance result: raw sentiment information coefficient

Before applying XGBoost, the project can ask a simpler finance question:

> Does stronger daily news sentiment rank with a higher next-session CSI 300 return?

Using the 31 aligned observations, the **Spearman rank correlation (raw-sentiment information coefficient)** between the trading-session sentiment score and the next-session CSI 300 return is:

**Rank IC = -0.348**

The corresponding Pearson correlation is:

**Pearson correlation = -0.246**

These are descriptive sample statistics, not statistical-significance claims.

The negative rank relationship means that, in this short clean window, more positive raw sentiment did **not** translate monotonically into a higher next-session market return.

That result is useful for the finance interpretation of the project: the value of news sentiment, if any, may depend on its interaction with momentum, volatility, attention and regime conditions rather than on a simple rule such as “positive news means buy.”

## Sentiment-sorted next-session returns

To make the raw relationship easier to interpret, the 31 aligned observations can be divided into three equal-sized sentiment groups.

| Sentiment group | Observations | Mean sentiment | Mean next-session CSI 300 return | Next session positive |
|---|---:|---:|---:|---:|
| Lowest-sentiment tercile | 10 | +0.0177 | **+0.305%** | **70.0%** |
| Middle tercile | 10 | +0.0610 | **+0.011%** | **50.0%** |
| Highest-sentiment tercile | 11 | +0.1236 | **-0.161%** | **27.3%** |

The descriptive high-minus-low next-session return difference is approximately **-0.466 percentage points**.

This is not presented as a tradable strategy or a causal result. The sample is small and the tercile thresholds are descriptive. However, it is a real empirical finding from the cleaned data and suggests that the simple relationship between news tone and next-session return is more complex than a direct positive-sentiment effect.

## Simple directional benchmark

A naive rule that predicts the next-session direction from only the sign of the raw sentiment score is correct on:

**45.2% of the 31 aligned observations.**

That is below 50% in this sample.

This negative benchmark is important because it prevents the project from claiming that raw FinBERT sentiment alone is already an alpha signal. The research question is whether **engineered sentiment features and market interactions** add information beyond the market-only baseline under a proper walk-forward evaluation.

## Why the project uses interaction features

The descriptive results above motivate the feature design used in the research pipeline:

- rolling sentiment rather than only same-day sentiment;
- sentiment momentum;
- sentiment dispersion;
- news intensity;
- sentiment × volatility;
- sentiment-z-score × volatility;
- market momentum;
- an ex-ante volatility regime indicator.

From a finance perspective, this tests whether the market response to information depends on the state in which that information arrives.

## Current model-evaluation status

The cleaned feature design uses 5-day, 10-day and 20-day rolling sentiment statistics together with market variables.

With the current committed sample, only **8 labelled rows** remain after all rolling features and the next-session target are available.

The public expanding-window design requires:

- **60 observations** for the initial training window; and
- at least **1 additional unseen observation** for the first genuine out-of-sample forecast.

For that reason, the repository does **not** report accuracy, Sharpe ratio or strategy profitability from the corrected model yet. Reporting those numbers from eight labelled observations would be economically and statistically uninformative.

This is different from saying that the project has “no result.” The current empirical result is:

1. the cleaned news sample is measurable and aligned to actual trading sessions;
2. raw daily sentiment does not show a simple positive next-session relationship;
3. the observed rank IC is negative in this short sample;
4. the sentiment-sorted returns are consistent with a possible contrarian or state-dependent relationship;
5. the nonlinear market-plus-sentiment hypothesis therefore remains an open test rather than a pre-decided conclusion.

## Finance interpretation

### Alternative data and price discovery

The project examines whether unstructured news adds information beyond standard market variables. That connects directly to:

- price discovery;
- market efficiency;
- investor attention;
- information processing;
- alternative data in investment research.

The current descriptive evidence does not support a simple “positive news leads to positive next-day return” rule.

### Benchmark and active return

The economic benchmark is the **CSI 300**.

When the corrected paper strategy has enough observations, benchmark-relative performance will be reported as:

**Active Return = Strategy Return - CSI 300 Return**

The project name uses “Alpha,” but the repository does not claim Jensen's alpha without an explicit asset-pricing regression.

### Risk-adjusted performance

The eventual strategy evaluation is designed to report:

- cumulative strategy return;
- CSI 300 benchmark return;
- active return;
- annualized Sharpe ratio;
- maximum drawdown;
- average turnover;
- transaction-cost drag.

The paper strategy includes a **10 bps proportional transaction cost per unit of turnover** and caps absolute position size at **1.0x**.

### GARCH risk overlay

GARCH(1,1) is used for **one-step-ahead volatility forecasting and position scaling**.

Its role is separate from directional forecasting:

- XGBoost estimates **direction probability**;
- GARCH estimates **forecast risk**;
- the signal engine converts both into a bounded paper position.

GARCH is therefore a risk-management component, not the source of directional alpha.

## What a finance reviewer can conclude today

The current repository supports the following conclusions without relying on the earlier development trace:

1. **The alternative-data pipeline is economically defined.** News is mapped to a specific tradable benchmark, the CSI 300, and to a next-session target.
2. **The clean sample contains 6,919 FinBERT-scored headline/article inputs across 49 news days.**
3. **The CSI 300 gained 1.56% over the matched clean market window with 20.78% annualized realized volatility.**
4. **Raw sentiment has a negative rank IC of -0.348 with next-session returns across 31 aligned observations.**
5. **The lowest-sentiment tercile was followed by a +0.305% average next-session return, while the highest-sentiment tercile was followed by -0.161%.**
6. **A naive raw-sentiment-sign rule achieved only 45.2% directional accuracy, so the project does not treat raw sentiment as a proven trading signal.**
7. **The main finance hypothesis is therefore incremental and conditional:** whether rolling sentiment, attention and sentiment-volatility interactions improve a market-only forecasting model out of sample.

## Reproducibility

Run:

```bash
python run_research_pipeline.py
```

The current sample builds the feature dataset and stops before model-level out-of-sample evaluation because the minimum training requirement has not yet been reached.

Once the clean sample is long enough, the same command will create:

```text
outputs/model_evaluation.csv
outputs/oos_predictions.csv
outputs/garch_oos_forecasts.csv
outputs/oos_backtest.csv
outputs/oos_backtest_metrics.csv
```

Generated outputs are kept outside version control so that an old local run cannot be mistaken for the current research result.

## Current conclusion

The project already produces a defensible finance result at the **descriptive alternative-data stage**: raw news tone is not a simple monotonic next-session signal in the clean CSI 300 sample, and the observed relationship is sufficiently state-dependent to justify testing rolling, interaction and risk-conditioned features.

The stronger question—whether those engineered sentiment features improve the market-only model and generate positive active return after costs—remains to be answered with a longer clean history and genuine expanding-window out-of-sample observations.
