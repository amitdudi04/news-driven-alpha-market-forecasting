# Master Dataset Data Dictionary

The final modeling table is generated locally as:

`data/master_session_dataset_2023_2026.csv`

It contains one row per CSI 300 trading session. Generated data are excluded from Git; column definitions are committed here so the modeling specification remains auditable.

## Session and timing columns

| Column | Definition |
|---|---|
| `date` | Predictor/session date for the CSI 300 trading close. |
| `window_start_shanghai` | Previous genuine CSI 300 close timestamp in Asia/Shanghai. |
| `window_end_shanghai` | Current genuine CSI 300 close timestamp in Asia/Shanghai; the forecasting information cutoff. |
| `target_session_date` | Next genuine CSI 300 trading session whose return is the prediction target. |
| `target_available` | 1 when a next-session target exists, otherwise 0. |
| `strict_model_ready` | 1 when all frozen model features and the next-session return target are available. |

## Market columns

| Column | Definition |
|---|---|
| `close` | CSI 300 closing index level on session `t`. |
| `return_t` | Close-to-close natural-log return on session `t`. |
| `volatility_20_t` | Sample standard deviation of log returns over the trailing 20 trading sessions, including session `t`. |
| `momentum_5_20_t` | Five-session mean close divided by 20-session mean close, minus 1. |
| `momentum_acceleration_t` | First difference of `momentum_5_20_t`. |
| `regime_dummy_t` | 1 when current 20-session volatility exceeds the expanding median of prior volatility observations; otherwise 0 once the reference is available. |

## News availability and count columns

| Column | Definition |
|---|---|
| `unique_headline_count_t` | Number of normalized unique titles inside the timestamp-safe session information window. Cross-calendar-day repetitions that land in the same market window are de-duplicated. |
| `headline_observation_count_t` | Number of assigned headline observations before the final within-session normalized-title de-duplication. |
| `has_news_t` | 1 when at least one unique headline is available in the session window; otherwise 0. |
| `prior_20_session_mean_unique_headlines` | Mean `unique_headline_count_t` over the prior 20 trading sessions, excluding the current session. |
| `news_intensity_20` | Current unique-headline count divided by `prior_20_session_mean_unique_headlines`. A genuine zero-news session can therefore have an intensity of 0 without imputing sentiment. |

## FinBERT sentiment columns

FinBERT is applied to individual retained English-language headlines. The pooled statistics below use session-unique normalized titles.

| Column | Definition |
|---|---|
| `sentiment_mean_t` | Mean FinBERT sentiment score in the session window. |
| `sentiment_std_t` | Sample standard deviation of FinBERT sentiment scores in the session window; 0 when exactly one unique headline is present. |
| `positive_share_t` | Share of session-unique headlines whose predicted FinBERT label is positive. |
| `negative_share_t` | Share whose predicted FinBERT label is negative. |
| `neutral_share_t` | Share whose predicted FinBERT label is neutral. |
| `sentiment_roll_5` | Strict five-session rolling mean of `sentiment_mean_t`. Missing if any required session has missing sentiment. |
| `sentiment_roll_10` | Strict ten-session rolling mean of `sentiment_mean_t`. |
| `sentiment_roll_20` | Strict 20-session rolling mean of `sentiment_mean_t`. |

Missing news is not converted to a neutral or zero sentiment score.

## Interaction columns

| Column | Definition |
|---|---|
| `sentiment_x_volatility` | `sentiment_mean_t × volatility_20_t`. |
| `sentiment_roll_5_x_volatility` | `sentiment_roll_5 × volatility_20_t`. |
| `sentiment_roll_20_x_volatility` | `sentiment_roll_20 × volatility_20_t`. |

## Target columns

| Column | Definition |
|---|---|
| `target_return_t_plus_1` | Natural-log return of the next genuine CSI 300 trading session. |
| `target_direction_t_plus_1` | 1 when `target_return_t_plus_1 > 0`; 0 when the return is zero or negative; missing when no next session is available. |

## Final counts

The validated local master dataset contains:

- 908 trading-session rows;
- 907 known next-session targets;
- 867 strict model-ready rows;
- one genuine no-news session: 20 Jun 2025.

The frozen dataset SHA-256 used by the directional experiment is recorded in `config/directional_experiment_2023_2026.json`. The modeling script refuses to run if the local master file does not match that frozen hash.
