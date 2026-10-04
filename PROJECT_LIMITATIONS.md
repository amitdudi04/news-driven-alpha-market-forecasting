# Project Limitations

## 1. News-source selection

The historical news query is English-language and title-based. It captures international English coverage of China-related macroeconomic and financial topics, not the full Chinese-language information set available to domestic investors.

The query is deliberately transparent but cannot guarantee perfect relevance or recall.

## 2. Timestamp precision

The alignment uses the **GDELT seen timestamp**. This is the time GDELT observed the item and is not guaranteed to equal the publisher's original publication timestamp.

The project therefore claims **timestamp-safe alignment relative to the observed GDELT timestamp**, not perfect reconstruction of first public dissemination.

## 3. Headline-level sentiment

FinBERT is applied to titles/headlines rather than full article bodies.

The pretrained ProsusAI/FinBERT model was not trained specifically for contemporary Chinese-market news and is not fine-tuned in this project. Sentiment scores can therefore contain domain and language-distribution error.

## 4. Syndication and duplication

Exact normalized-title duplicates are removed within each assigned market-session information window.

Semantically identical stories with changed wording can remain. The resulting headline count is therefore a cleaned observation count, not a count of economically independent information events.

## 5. Missing news

Missing sentiment is not filled with zero or neutral sentiment.

The one genuine no-news research session remains missing, which propagates through strict rolling sentiment windows. This reduces sample size but avoids conflating missing information with neutral tone.

## 6. Feature and model scope

The principal directional models are logistic regression and shallow XGBoost.

The feature set was frozen before final holdout evaluation. This reduces post-hoc flexibility, but it does not prove that the specification is economically optimal.

## 7. OOS sample size and uncertainty

The strict modeling sample contains 867 observations:

- 221 target-year 2023 rows;
- 242 target-year 2024 validation rows;
- 223 target-year 2025 holdout rows;
- 181 target-year 2026 robustness rows.

The paired 10-session moving-block bootstrap shows wide uncertainty. Principal 2025/2026 incremental sentiment effects remain unresolved at the 95% level.

A high fraction of positive bootstrap draws is not treated as a classical p-value.

## 8. Probability calibration

Directional probabilities are imperfectly calibrated. Calibration slopes are often below 1.

Probability outputs are therefore interpreted primarily as forecasting/ranking scores rather than literal perfectly calibrated event probabilities.

No post-hoc calibration model is fitted on the 2025 holdout or 2026 robustness periods.

## 9. GARCH specification

The risk overlay uses a simple zero-mean Normal GARCH(1,1).

It provides a transparent one-step volatility forecast but does not exhaust alternative volatility models, distributions, realized-volatility estimators, or structural breaks.

GARCH does not generate directional alpha.

## 10. Transaction-cost simulation

The simulation uses fixed conventions:

- long at p(up) >= 0.55;
- short at p(up) <= 0.45;
- otherwise flat;
- maximum absolute position 1;
- 10 bps cost per unit turnover.

These values were not optimized on 2025 or 2026 outcomes.

The simulation does not model slippage, market impact, financing, shorting constraints, taxes, execution latency, capacity, or live operational failures.

## 11. Interpretation

The project does not establish:

- causality;
- statistically resolved sentiment alpha;
- Jensen alpha;
- persistent live profitability;
- economic scalability;
- transferability to other markets or time periods.

Null and negative evidence is retained as part of the result.
