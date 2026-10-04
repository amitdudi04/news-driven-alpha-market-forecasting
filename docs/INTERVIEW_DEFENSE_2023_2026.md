# Interview Defense — News-Driven Alpha 2023–2026

## 30-second project summary

I studied whether China-focused financial-news sentiment adds **incremental next-session forecasting information** for the CSI 300 beyond market variables.

The main methodological issue was timing. I reconstructed 2023–2026 GDELT headline data, scored it with FinBERT, and assigned every headline to the first CSI 300 close after its GDELT seen timestamp using a 15:00 Shanghai cutoff. I then compared market-only and market-plus-sentiment logistic regression and XGBoost models under a frozen chronological design: 2023 training, 2024 expanding validation, untouched 2025 holdout, and locked-model 2026 robustness.

The result is mixed. Sentiment sometimes improves forecasts—most notably XGBoost in 2026—but paired block-bootstrap intervals still include zero, so I do not claim statistically resolved sentiment alpha.

## Five-minute explanation

### 1. Research question

The question is not “can XGBoost predict the market?”

It is:

> Does timestamp-safe financial-news sentiment improve next-session CSI 300 forecasts relative to the same model family using market information alone?

That is why every main model has a paired market-only baseline.

### 2. Data and timing

I built a 2023–2026 English-language China-finance headline sample from GDELT and scored headlines with pretrained ProsusAI/FinBERT.

The alignment layer is crucial. A forecast is defined at the CSI 300 close at 15:00 Asia/Shanghai.

A headline is usable for trading session t only if:

```text
previous trading close < GDELT seen time <= current trading close
```

If news appears after the close, on a weekend, or during a holiday, it moves forward to the next genuine market close.

The full audit contained 292,373 scored headline observations. 291,973 could be assigned to completed market windows. After normalized within-session de-duplication, 276,960 unique headlines were used for pooled sentiment.

The independent timing audit found zero causal-window violations.

### 3. Master dataset

The master dataset contains 908 CSI 300 sessions.

Market variables include log return, 20-session volatility, momentum, momentum acceleration, and a causal volatility-regime indicator.

Sentiment variables include unique headline count, pooled FinBERT mean and dispersion, strict 5/10/20-session rolling sentiment, news intensity, and sentiment-volatility interactions.

There is one genuine no-news trading session, 20 Jun 2025. I leave sentiment missing rather than replacing it with neutral sentiment.

The next-session direction target is positive if the next genuine CSI 300 log return is greater than zero.

After requiring the full feature set and a known target, 867 rows are model-ready.

### 4. Frozen split

The split is based on the year of the **target session**, not the feature-row year.

- 2023: 221 initial training targets
- 2024: 242 targets in 12 monthly expanding validation folds
- 2025: 223 untouched holdout targets
- 2026: 181 post-sample robustness targets

The split, feature sets, candidate grids, threshold, and metric rules were committed before model fitting.

### 5. Models and results

I compare four models:

- logistic market-only;
- logistic market + sentiment;
- XGBoost market-only;
- XGBoost market + sentiment.

In the 2025 untouched holdout, the result is mixed.

Logistic + sentiment improves balanced accuracy from about 48.3% to 51.3%, but slightly worsens Brier score and log loss.

XGBoost + sentiment slightly worsens balanced accuracy from about 51.1% to 50.8%, but improves Brier, log loss, and AUC.

In 2026, the strongest point result appears for XGBoost + sentiment:

- balanced accuracy: 55.68% versus 50.20% market-only;
- Brier: 0.2502 versus 0.2531;
- log loss: 0.6935 versus 0.6992.

But I do not stop at point estimates.

### 6. Uncertainty

I froze a paired circular moving-block bootstrap:

- 10-session blocks;
- 5,000 resamples;
- identical sampled rows for each market-only/sentiment pair.

For XGBoost 2026 balanced accuracy, the point improvement is +5.48 percentage points, but the 95% interval is approximately -0.20 to +11.83 percentage points.

Because the interval includes zero, I call the effect **promising but statistically unresolved**.

A useful negative result also appears: in 2024 logistic validation, sentiment significantly worsens Brier and log loss under the same frozen bootstrap rule.

### 7. GARCH and simulation

GARCH(1,1) is separate from direction prediction. It forecasts next-session volatility only for risk scaling.

The fixed simulation uses:

- p(up) >= 0.55 long;
- p(up) <= 0.45 short;
- otherwise flat;
- max absolute position 1;
- 10 bps transaction cost per unit turnover.

There are 907 GARCH forecasts with zero convergence failures.

For XGBoost + sentiment with GARCH scaling:

- 2024 net return: +5.54% versus benchmark +14.68%;
- 2025: -7.27% versus benchmark +11.52%;
- 2026: +3.12% versus benchmark -5.88%.

So the strategy does not dominate the market. I treat this as a risk-aware simulation, not proof of alpha.

## Likely interview questions

### Why did you use a market-only baseline?

Because the research question is incremental information. Without the paired market-only model, a positive XGBoost result would not tell me whether news sentiment adds anything beyond standard market state variables.

### Why split by target-session year?

A predictor row at year-end can forecast a return whose target session is in January of the next year. Partitioning by target date prevents a boundary observation from leaking across the intended train/holdout split.

### Why was 2025 untouched?

I used 2024 for model selection and chronological validation. Holding 2025 untouched gives a cleaner estimate of generalization after the model specification is fixed.

### Why did you not retrain on 2025 before testing 2026?

I wanted 2026 to be a locked-model robustness check. Using the same 2023–2024 fit isolates temporal robustness rather than letting the model adapt after seeing the holdout.

### Why use logistic regression if the project uses XGBoost?

Logistic regression is an interpretable benchmark. If sentiment only works in a flexible nonlinear model but not in a simpler model, that is important evidence about how stable and model-dependent the effect may be.

### Why balanced accuracy?

The classes are fairly balanced, but balanced accuracy still gives equal weight to positive and non-positive sessions and avoids a misleading result if class frequencies shift by year.

### Why Brier score and log loss?

Accuracy throws away probability information. Brier and log loss evaluate the quality of the full predicted probability, which matters because the simulation uses probability thresholds.

### What did calibration show?

Calibration was weak overall. For example, XGBoost + sentiment in 2026 had a calibration slope around 0.56 rather than the ideal 1. This is one reason I do not interpret p(up)=0.60 as a precisely calibrated 60% event probability.

### Why block bootstrap instead of an IID bootstrap?

Financial time series can have serial dependence. Resampling contiguous 10-session blocks preserves more local dependence than drawing individual days independently.

### Why 10-session blocks?

It was frozen before the bootstrap run as a modest two-trading-week dependence window. I did not choose it after observing which interval produced the desired conclusion.

### Did sentiment significantly improve the 2025 holdout?

No. All paired 2025 incremental intervals include zero under the frozen block-bootstrap analysis.

### What is your strongest positive result?

The strongest point result is XGBoost + sentiment in 2026: balanced accuracy improves by about 5.48 percentage points. But the 95% block-bootstrap interval still crosses zero, so I call it suggestive rather than statistically resolved.

### Did you get any statistically resolved result?

Yes, a negative development result. In 2024 logistic validation, adding sentiment worsened Brier score and log loss with 95% block-bootstrap intervals below zero for the sentiment improvement metric.

### Why is a negative result useful?

Because the purpose is to test whether sentiment adds information, not to force a positive finding. A model-dependent or period-dependent result is more credible if negative evidence is retained.

### Why use GDELT seen time instead of article publication time?

That is the timestamp consistently available in the historical GDELT source. I explicitly label it as GDELT seen time and do not claim it is the publisher's exact publication timestamp.

### What is the biggest timing risk?

If after-close news were assigned to the same session, the model would use information not available at the forecast cutoff. The timestamp-safe alignment prevents that by moving it to the first eligible later close.

### Why de-duplicate within trading sessions?

The same syndicated headline can appear on multiple calendar days, especially across weekends or holidays. After those days roll into one trading window, duplicates could overweight one story. I therefore apply normalized session-level title de-duplication.

### Why not fill missing sentiment with zero?

Zero is a sentiment value. A session with no usable news is an information-availability state, not neutral sentiment. Imputing zero would mix missingness with a meaningful FinBERT score.

### What exactly does GARCH do?

It forecasts next-session volatility. The directional model decides long, short, or flat from p(up). GARCH only reduces position size when forecast volatility is high relative to its prior forecast history.

### Did GARCH improve the strategy?

It generally reduced drawdown magnitude, but it did not always improve total return. That is consistent with its role as a risk overlay rather than an alpha generator.

### Did the strategy beat the CSI 300?

Not consistently. The XGBoost + sentiment fixed-rule simulation underperformed the CSI 300 in 2024 and 2025. It had a positive active-return difference in the 2026 robustness period because the benchmark declined. I do not interpret this as durable alpha.

### Is “active return” Jensen alpha?

No. Here active total return is simply strategy cumulative return minus benchmark cumulative return. Jensen alpha would require an explicit asset-pricing regression.

### What would you improve next?

I would test alternative sentiment models/languages only under a new pre-specified experiment, improve publication-time provenance where possible, add formal probability calibration learned only on development data, test more economic volatility models, and evaluate a longer untouched future period.

## Safe final conclusion for interview

> My result is not “news predicts the CSI 300.” The result is that timestamp-safe news sentiment sometimes adds incremental information, especially in the 2026 XGBoost robustness period, but the paired uncertainty intervals remain wide and include zero. The project contribution is the auditable timing-safe research design and the disciplined comparison against market-only baselines, not a claim of guaranteed trading alpha.
