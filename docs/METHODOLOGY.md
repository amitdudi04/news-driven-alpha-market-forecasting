# Research Methodology

**Study:** Incremental predictive information from financial-news sentiment for CSI 300 returns  
**Empirical horizon:** January 2023–September 2026 for completed market sessions; 2022 market-history warm-up. News observations extend into October 2026 but are not assigned to later completed market closes.  
**Unit:** One CSI 300 trading session

## Research question and analytical design

This study examines whether China-focused financial-news sentiment adds incremental predictive information about the next-session **return direction** for the CSI 300 relative to market-only information. Each market-plus-sentiment model is compared with its matching market-only model **within the same model family**. The four principal specifications comprise logistic regression and XGBoost, each fitted with and without news-sentiment variables. Calibration and paired uncertainty analyses assess the directional predictions; GARCH(1,1) forecasts and a costed position simulation are **separate** risk and economic experiments.

## Data sources, measurement and timing

The CSI 300 market series covers **4 January 2022–30 September 2026**, with 2022 used for market-feature and volatility-model warm-up. The retained AkShare/China Securities Index feed was cross-checked against Sina on **1,150** overlapping trading days; the maximum observed closing-level discrepancy was **0.005 index points**.

English-language, China-related finance and macroeconomic headlines were collected from the **GDELT 2.0 Global Article List** through BigQuery, then classified with pretrained **ProsusAI/FinBERT**. Article times are interpreted as **GDELT seen timestamps**, not as independently established first-publication times. The English-language headline selection does not represent the entire Chinese-language information environment.

Forecast information becomes available at the **15:00 Asia/Shanghai close** for session `t`. A headline enters that session's news window only when:

```text
previous CSI 300 close < GDELT seen timestamp <= current CSI 300 close
```

After-close, weekend and holiday items are carried **forward** to the next trading-session window. Normalized identical titles are deduplicated within each session before sentiment pooling.

| Archived data-processing result | Count |
|---|---:|
| Scored headline observations entering alignment | 292,373 |
| Assigned to completed CSI 300 windows | 291,973 |
| Not assigned after the last completed market close | 400 |
| Session-unique normalized headlines | 276,960 |
| Repeated observations removed | 15,013 |
| Article/FinBERT unmatched rows | 0 |
| Shanghai-date mismatches | 0 |
| Session-window timing violations | 0 |

A recorded absence of news is **not** evidence of neutral sentiment. The no-news session of **20 June 2025** retains missing sentiment rather than artificial zero sentiment.

## Feature engineering and target

The master table contains **908** research-period sessions, **907** known next-session targets and **867** fully model-ready rows. Market features are the current natural-log return (`return_t`), trailing 20-session return volatility (`volatility_20_t`), 5-versus-20-session price momentum (`momentum_5_20_t`), momentum acceleration and the trailing-volatility regime indicator (`regime_dummy_t`).

The sentiment specification adds unique-headline counts, session mean and dispersion of FinBERT scores, positive/negative/neutral shares, complete-window rolling mean sentiment over **5, 10 and 20** sessions, headline intensity relative to the **previous** 20 sessions, and sentiment × volatility interactions. All rolling and relative-volume calculations use the recorded current or earlier information windows, never the next-session target.

The continuous target `target_return_t_plus_1` is the **following actual trading session's natural-log close-to-close return**. The binary target `target_direction_t_plus_1` is one for a strictly positive return and zero otherwise. Missing next-session targets are not assigned a class. The split is by **`target_session_date`**, not the predictor-row calendar year, avoiding target-year contamination at year boundaries.

## Chronological experiment

| Target period | Analytical role | Eligible rows |
|---|---|---:|
| 2023 | Initial training | 221 |
| 2024 | Twelve expanding monthly validation folds | 242 |
| 2025 | Designated holdout excluded from fitting | 223 |
| 2026 | Locked-model temporal robustness | 181 |

Within each 2024 validation fold, the training observations have target dates **strictly earlier** than the first validation target. Logistic preprocessing is fitted only to that fold's training rows. Candidate models are ranked by aggregate 2024 **Brier score**, then log loss and candidate identifier as deterministic tie-breakers.

The logistic model uses L2 regularization with `lbfgs` and a candidate `C` grid of **0.1, 1, 10**; both variants selected **0.1**. Both selected XGBoost variants use **150 trees**, maximum depth **2**, learning rate **0.03**, minimum child weight **5**, row and column subsampling of **0.9**, L2 regularization **5**, and L1 regularization **0**.

After 2024 validation-based model selection, the selected models are fitted on eligible **2023–2024** data and applied **without refitting on 2025** to the 2025 and 2026 periods. The released code supports this chronological separation. The configuration declares a pre-fitting specification freeze, but independent contemporaneous preregistration evidence and complete noninspection of 2025 are **not established publicly**.

## Predictive evaluation and statistical inference

Balanced accuracy, accuracy, **Brier score**, log loss and ROC AUC describe discrimination and probability quality. Calibration intercept, calibration slope and five-bin expected calibration error characterize probability calibration.

For each matched market-only/sentiment pair, the study reports incremental differences using a **paired circular moving-block bootstrap** with **5,000 draws**, **10-session blocks** and **95% percentile intervals**. Identical block indices are applied to both model variants. Positive differences favor the sentiment specification; for losses this means *market-only loss minus sentiment loss*. Intervals including zero are described as statistically unresolved. These intervals do not fully correct for specification selection across years, model families or evaluation metrics.

## Volatility and illustrative strategy

A separate expanding-window, zero-mean **Gaussian GARCH(1,1)** forecasts next-session conditional volatility; it does **not** generate the directional signal. The saved release contains **907** forecasts with **no recorded convergence failures**.

The strategy is long for predicted up probabilities at least **0.55**, short for probabilities at most **0.45**, and flat otherwise. GARCH scaling reduces absolute exposure but does not change its direction. Costs are **10 bps per unit change in target exposure**, and the reported Sharpe ratio uses a zero risk-free rate. These thresholds and costs are research conventions, not optimized or independently validated execution parameters.

The simulated position earns the next close-to-close CSI 300 return after signals are constructed using information observed at the current close. **Execution at the same close after observing closing features is not demonstrated.** The simplified strategy excludes several financing, shorting, implementation and market-impact frictions. Cumulative net strategy return minus cumulative index return is an *active-return difference*, not Jensen alpha. Drawdown calculations include starting unit wealth.

## Variable dictionary and evidence provenance

| Field | Definition |
|---|---|
| `date`, `target_session_date` | Predictor session date and following actual market-session target date |
| `window_start_shanghai`, `window_end_shanghai` | Previous and current market-close boundaries |
| `strict_model_ready`, `target_available` | Complete-feature/target eligibility and existence of next-session target |
| `return_t` | Current close-to-close natural-log return |
| `volatility_20_t` | Trailing 20-session sample standard deviation of returns |
| `momentum_5_20_t`, `momentum_acceleration_t` | Short-to-long moving-average price ratio minus one, and its change |
| `regime_dummy_t` | Current volatility above expanding median of prior observed volatility |
| `unique_headline_count_t` | Normalized distinct titles in current session's information window |
| `headline_observation_count_t`, `has_news_t` | Pre-deduplication observations and availability flag |
| `prior_20_session_mean_unique_headlines` | Trailing headline-volume mean **excluding** current session |
| `news_intensity_20` | Current unique count relative to preceding-20-session mean |
| `sentiment_mean_t`, `sentiment_std_t` | Pooled score mean and dispersion |
| `positive_share_t`, `negative_share_t`, `neutral_share_t` | Session-level FinBERT label shares |
| `sentiment_roll_5`, `sentiment_roll_10`, `sentiment_roll_20` | Complete-window rolling sentiment means |
| `sentiment_x_volatility`, `sentiment_roll_5_x_volatility`, `sentiment_roll_20_x_volatility` | Sentiment–volatility interactions |
| `target_return_t_plus_1`, `target_direction_t_plus_1` | Following-session return and binary sign |

The source master-dataset identifier and selected model settings are stored in `config/` and `results/results_manifest.json`; compact evidence is under `results/`. See [Results](RESULTS.md), [Limitations](LIMITATIONS.md), and [Reproducibility](REPRODUCIBILITY.md) for the numerical results, validity boundaries, and execution steps. No statistically resolved positive sentiment-alpha effect is claimed.
