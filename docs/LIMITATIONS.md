# Project Limitations

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

## 12. Interpretation boundary

The project does not establish causality, Jensen alpha, guaranteed trading profitability, or persistence outside the studied sample.
