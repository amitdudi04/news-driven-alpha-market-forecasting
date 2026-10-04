# Master Dataset Data Dictionary

The final modeling table is generated locally as:

`data/master_session_dataset_2023_2026.csv`

It contains one row per CSI 300 trading session.

## Session and timing columns

| Column | Definition |
|---|---|
| `date` | Predictor/session date for the CSI 300 close. |
| `window_start_shanghai` | Previous CSI 300 close timestamp in Asia/Shanghai. |
| `window_end_shanghai` | Current CSI 300 close timestamp in Asia/Shanghai. |
| `target_session_date` | Next CSI 300 trading session whose return is the target. |
| `target_available` | 1 when a next-session target exists, otherwise 0. |
| `strict_model_ready` | 1 when all model features and the next-session target are available. |

## Market columns

| Column | Definition |
|---|---|
| `close` | CSI 300 closing index level on session t. |
| `return_t` | Close-to-close natural-log return on session t. |
| `volatility_20_t` | Sample standard deviation of log returns over the trailing 20 trading sessions. |
| `momentum_5_20_t` | Five-session mean close divided by 20-session mean close, minus 1. |
| `momentum_acceleration_t` | First difference of `momentum_5_20_t`. |
| `regime_dummy_t` | 1 when current 20-session volatility exceeds the expanding median of prior volatility observations. |

## News availability and count columns

| Column | Definition |
|---|---|
| `unique_headline_count_t` | Number of normalized unique titles in the session information window. |
| `headline_observation_count_t` | Number of assigned headline observations before within-session de-duplication. |
| `has_news_t` | 1 when at least one unique headline is available; otherwise 0. |
| `prior_20_session_mean_unique_headlines` | Mean unique-headline count over the prior 20 sessions, excluding the current session. |
| `news_intensity_20` | Current unique-headline count divided by the prior-20-session mean. |

## FinBERT sentiment columns

| Column | Definition |
|---|---|
| `sentiment_mean_t` | Mean FinBERT sentiment score in the session information window. |
| `sentiment_std_t` | Sample standard deviation of session sentiment scores; 0 when exactly one unique headline is present. |
| `positive_share_t` | Share of session-unique headlines classified positive. |
| `negative_share_t` | Share classified negative. |
| `neutral_share_t` | Share classified neutral. |
| `sentiment_roll_5` | Five-session rolling mean of `sentiment_mean_t`, requiring a complete window. |
| `sentiment_roll_10` | Ten-session rolling mean of `sentiment_mean_t`, requiring a complete window. |
| `sentiment_roll_20` | Twenty-session rolling mean of `sentiment_mean_t`, requiring a complete window. |

Missing news is not converted to a neutral or zero sentiment value.

## Interaction columns

| Column | Definition |
|---|---|
| `sentiment_x_volatility` | `sentiment_mean_t * volatility_20_t`. |
| `sentiment_roll_5_x_volatility` | `sentiment_roll_5 * volatility_20_t`. |
| `sentiment_roll_20_x_volatility` | `sentiment_roll_20 * volatility_20_t`. |

## Target columns

| Column | Definition |
|---|---|
| `target_return_t_plus_1` | Natural-log return of the next CSI 300 trading session. |
| `target_direction_t_plus_1` | 1 when the next-session return is positive; 0 otherwise; missing when no next session is available. |

## Final counts

- **908** trading-session rows
- **907** known next-session targets
- **867** model-ready rows
- one no-news research session: **20 Jun 2025**

The experiment configuration records the SHA-256 of the master dataset so the modeling script can verify that it is running on the intended data version.
