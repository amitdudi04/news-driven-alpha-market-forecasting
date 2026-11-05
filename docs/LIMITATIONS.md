# Research Limitations and Validity Boundaries

This document distinguishes limitations of information availability, statistical inference, data reconstruction, and economic implementation. These qualifications apply to all numerical findings in [Results](RESULTS.md) and do not disappear when the repository's tests pass.

## 1. News-source coverage

The historical query uses English-language headlines about China-related macroeconomic and financial topics. It does not represent the full Chinese-language information set available to domestic investors.

Title-based filtering also creates relevance and recall trade-offs.

## 2. Timestamp source

Session alignment uses the GDELT seen timestamp. That time can differ from the publisher's original publication time.

## 3. Headline-level sentiment

FinBERT is applied to titles rather than full article bodies. The pretrained model is not specialized for contemporary Chinese-market English news.

## 4. Syndication and duplication

Normalized-title duplicates are removed within each assigned trading-session window. Semantically equivalent stories with different wording can remain.

## 5. Missing news

The no-news research session remains missing for sentiment features. Complete-window rolling features therefore lose additional observations around that date.

## 6. Model scope

The principal directional models are logistic regression and shallow XGBoost. Other model classes may produce different results.

## 7. Sample size and uncertainty

The strict model-ready sample contains **867** observations:

- 221 target-year 2023 rows;
- 242 target-year 2024 rows;
- 223 target-year 2025 rows;
- 181 target-year 2026 rows.

Principal 2025 and 2026 paired bootstrap intervals for incremental sentiment effects include zero.

## 8. Probability calibration

Calibration slopes are generally below 1, so predicted probabilities are not perfectly calibrated event probabilities.

## 9. GARCH specification

The volatility overlay uses a zero-mean Normal GARCH(1,1). Alternative volatility models, innovation distributions, and structural-break treatments are outside the present scope.

## 10. Transaction-cost simulation

The simulation uses fixed signal thresholds and a 10 bps turnover cost.

It does not model:

- market impact;
- slippage;
- financing;
- shorting constraints;
- taxes;
- execution latency;
- capacity;
- operational failures.

## 11. Scope of inference

The study covers one equity index, one news-source construction, one sentiment model, and the 2023-2026 period.

Results may differ in other markets, languages, data sources, or time periods.

## 12. Experimental chronology and execution assumptions

The public configuration declares that experiment specifications were frozen before model fitting, but independently dated preregistration evidence or proof of complete noninspection of the 2025 holdout is unavailable. The released code uses chronological validation and excludes 2025 and 2026 from the final model fit; this is a verified implementation property, not external certification of blindness.

The study observes features at the 15:00 Shanghai market close, then uses the next close-to-close index return for its simulated strategy. Realistic entry at the same closing price after observing closing features has not been demonstrated. The strategy is an idealized signal-based return calculation, not a next-executable-price backtest.

## 13. Interpretation boundary

The project does not establish causality, Jensen alpha, guaranteed trading profitability, or persistence outside the studied sample.
