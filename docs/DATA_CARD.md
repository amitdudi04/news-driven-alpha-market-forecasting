# Data Card

## Research scope

The project combines CSI 300 market data with English-language China-focused financial and economic news.

### Historical periods

- Market warm-up: **4 Jan 2022 onward**
- Research/news period: **1 Jan 2023 - 3 Oct 2026**
- Last completed CSI 300 market close in the study: **30 Sep 2026**

2022 is used only for rolling market features and GARCH warm-up.

## News source

Historical headlines are retrieved from the **GDELT 2.0 Global Article List (GAL)** in BigQuery.

The query:

- restricts language to English;
- requires a China identifier in the title;
- requires at least one macroeconomic or financial term;
- retains headline text rather than full article bodies.

The resulting sample therefore represents English-language international coverage of China-related finance and economics, not the complete domestic Chinese-language information set.

## Timestamp handling

The available GDELT time field is treated as the **GDELT seen timestamp**.

Headlines are mapped to CSI 300 trading sessions using Shanghai-local time and the 15:00 market close. News observed after the close, on weekends, or during market holidays is carried forward to the next eligible trading session.

Alignment summary:

| Item | Count |
|---|---:|
| Headline observations entering alignment | 292,373 |
| Assigned to completed CSI 300 sessions | 291,973 |
| Remaining after the final completed market close | 400 |
| Article/FinBERT unmatched rows | 0 |
| Shanghai-date mismatches | 0 |
| Causal-window timing violations | 0 |

The 400 remaining observations occur after the final completed CSI 300 close and are not assigned backward.

## Sentiment

Headlines are scored with pretrained **ProsusAI/FinBERT**.

Before session-level pooling, normalized titles are de-duplicated within each assigned trading-session window.

| Item | Count |
|---|---:|
| Assigned headline observations | 291,973 |
| Session-unique normalized headlines | 276,960 |
| Repeated observations removed | 15,013 |

Missing news is not converted to a neutral score.

## Market data

The CSI 300 historical series is built from the **China Securities Index** feed exposed through AkShare and cross-checked against Sina history.

- total market sessions: **1,150**
- 2022 warm-up sessions: **242**
- research sessions from 2023: **908**
- cross-source overlapping sessions: **1,150**
- maximum close difference: **0.005 index points**

## Master session dataset

The final local master dataset contains **908 CSI 300 trading sessions**.

Feature groups include:

- current log return;
- 20-session realized volatility;
- 5/20 momentum;
- momentum acceleration;
- causal volatility-regime indicator;
- unique headline count;
- pooled FinBERT sentiment mean and dispersion;
- positive, negative, and neutral FinBERT shares;
- complete-window 5/10/20-session rolling sentiment;
- news intensity relative to the prior 20 sessions;
- sentiment x volatility interactions.

The no-news research session is **20 Jun 2025**. Sentiment remains missing for that session.

Targets are defined from the next genuine CSI 300 trading session:

- next-session log return;
- next-session direction = 1 for a positive return, otherwise 0.

There are **907** known next-session targets and **867** rows with the full feature set used by the principal models.

## Data limitations

- English-only news selection does not represent the full Chinese-language information set.
- GDELT seen time may differ from the publisher's original publication time.
- Title-based filtering can miss relevant stories or include imperfectly relevant items.
- Normalized-title de-duplication does not eliminate all semantic syndication.
- FinBERT is not specifically trained for Chinese-market English news.
- The study ends before a completed market session is available for the final 400 late-Sep/Oct 2026 headline observations.
