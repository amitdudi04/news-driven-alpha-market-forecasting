# News-Driven Alpha: Timestamp-Safe Financial Sentiment and CSI 300 Forecasting

This project tests whether **China-focused financial-news sentiment adds incremental next-session forecasting information beyond market-only variables for the CSI 300**.

The repository now contains a completed 2023–2026 historical research pipeline: GDELT headline acquisition, FinBERT scoring, timestamp-safe trading-session alignment, CSI 300 market features, four frozen directional models, paired block-bootstrap uncertainty, a separate GARCH(1,1) risk overlay, and a fixed-rule transaction-cost simulation.

## Main empirical result

The evidence is **mixed rather than uniformly positive**.

- The strict modeling sample contains **867** fully usable session/target rows.
- Hyperparameters were selected using **2024 monthly expanding-window validation only**.
- **2025 (223 targets)** was held untouched until model selection was complete.
- **2026 (181 targets)** is a locked-model post-sample robustness period; the 2023–2024 fit was carried forward without retraining on 2025.
- In the 2025 holdout, sentiment modestly improves some metrics but worsens others.
- The strongest point result appears for **XGBoost in 2026**, where balanced accuracy rises from **50.20% to 55.68%** and Brier score improves from **0.25310 to 0.25017**.
- However, the paired 10-session block-bootstrap 95% interval for that balanced-accuracy improvement is approximately **-0.20 to +11.83 percentage points**, so the effect is **not statistically resolved at 95%**.
- A resolved negative result also exists: in 2024 logistic validation, adding sentiment worsens Brier score and log loss under the frozen bootstrap design.

The defensible conclusion is therefore:

> Timestamp-safe financial-news sentiment shows **period- and model-dependent incremental forecasting information**, but the current OOS evidence does not establish a stable, statistically resolved predictive advantage.

Full results are in [docs/RESULTS.md](docs/RESULTS.md).

## Data construction

### News

The historical news layer covers **1 Jan 2023 through 3 Oct 2026**.

- **292,373** FinBERT-scored headline observations entered the timestamp-alignment audit.
- **291,973** could be assigned to a completed CSI 300 information window.
- **400** late-Sep/Oct 2026 headlines remain pending because the last completed CSI 300 close in the market file is 30 Sep 2026.
- After normalized within-session de-duplication, **276,960 session-unique headlines** remain for pooled sentiment estimation.

GDELT's timestamp is treated as a **GDELT seen timestamp**, not asserted to be the publisher's original publication timestamp.

### Market

CSI 300 history is built from **4 Jan 2022 through 30 Sep 2026**.

- 2022 is warm-up only.
- The research sample begins in 2023.
- Market history contains **1,150 trading sessions**.
- The primary historical source is the China Securities Index feed exposed through AkShare, independently checked against Sina data.
- Across 1,150 overlapping sessions, the maximum close discrepancy is only **0.005 index points**.

## Timestamp-safe information set

Every directional forecast is defined at the **15:00 Asia/Shanghai CSI 300 close**.

A headline is assigned to the first market close satisfying:

```text
previous CSI 300 trading close < GDELT seen timestamp <= current CSI 300 trading close
```

This prevents after-close news from leaking backward into the same trading day's prediction. Weekend and holiday news flows forward to the next genuine CSI 300 trading close.

Independent streaming audit:

- **0** article/FinBERT join mismatches
- **0** Shanghai-calendar-date mismatches
- **0** causal-window timing violations

## Master session dataset

The master dataset contains **908 CSI 300 sessions** from 3 Jan 2023 through 30 Sep 2026.

Features include:

- current log return;
- 20-session volatility;
- 5/20 momentum and momentum acceleration;
- causal volatility-regime indicator;
- unique headline count;
- pooled FinBERT sentiment mean and dispersion;
- positive / negative / neutral FinBERT shares;
- strict 5/10/20-session rolling sentiment;
- news intensity relative to the **prior** 20 sessions;
- sentiment × volatility interactions.

Missing sentiment is **not neutralized**. The one genuine no-news market session, **20 Jun 2025**, remains missing for sentiment features.

The genuine target is the **next CSI 300 trading-session return sign**. Of 908 sessions, **907** have a known next-session target, and **867** rows satisfy the strict full-feature modeling rule.

## Frozen experiment

The split is based on **target_session_date**, not predictor-row calendar year.

| Target period | Role | Rows |
|---|---|---:|
| 2023 | initial development training | 221 |
| 2024 | 12 monthly expanding validation folds | 242 |
| 2025 | untouched final holdout | 223 |
| 2026 | locked-model post-sample robustness | 181 |

The experiment specification was committed **before model fitting**.

Principal models:

1. Logistic regression — market only
2. Logistic regression — market + sentiment
3. XGBoost — market only
4. XGBoost — market + sentiment

The research question is **incremental sentiment value within the same model family**, not simply whether XGBoost can predict the market.

## OOS model results

| Model | Period | Balanced accuracy | Brier | ROC AUC |
|---|---|---:|---:|---:|
| Logistic market | 2025 holdout | 0.483 | 0.2562 | 0.509 |
| Logistic + sentiment | 2025 holdout | 0.513 | 0.2568 | 0.513 |
| XGB market | 2025 holdout | 0.511 | 0.2624 | 0.506 |
| XGB + sentiment | 2025 holdout | 0.508 | 0.2604 | 0.519 |
| Logistic market | 2026 robustness | 0.481 | 0.2563 | 0.486 |
| Logistic + sentiment | 2026 robustness | 0.514 | 0.2563 | 0.493 |
| XGB market | 2026 robustness | 0.502 | 0.2531 | 0.542 |
| XGB + sentiment | 2026 robustness | **0.557** | **0.2502** | **0.546** |

Calibration remains imperfect. Probability forecasts should therefore not be interpreted as precisely calibrated event probabilities.

## Block-bootstrap uncertainty

Incremental sentiment differences are evaluated with a **paired circular moving-block bootstrap**:

- 5,000 resamples
- 10-session blocks
- paired market-only / sentiment rows
- 95% percentile intervals

For the 2025 untouched holdout, all incremental sentiment intervals include zero.

For XGBoost 2026 balanced accuracy:

```text
point improvement: +5.48 percentage points
95% block-bootstrap interval: approximately [-0.20, +11.83] percentage points
```

This is promising but statistically unresolved.

## GARCH risk overlay and fixed-rule simulation

GARCH(1,1) is estimated **separately** from the directional models. It forecasts next-session volatility and only changes position size; it does not determine direction.

The frozen simulation uses:

- p(up) >= 0.55 → LONG
- p(up) <= 0.45 → SHORT
- otherwise → FLAT
- max absolute position = 1.0
- transaction cost = **10 bps per unit of turnover**
- GARCH scale = prior expanding median forecast volatility / current forecast volatility, clipped to [0, 1]

There are **907** one-step GARCH forecasts and **0 convergence failures**.

The GARCH overlay generally reduces drawdown magnitude, but it does **not** consistently improve total return. The simulation is descriptive research, not live-trading evidence.

For XGBoost + sentiment with GARCH scaling:

| Period | Strategy net return | CSI 300 benchmark | Active-return difference |
|---|---:|---:|---:|
| 2024 validation | +5.54% | +14.68% | -9.14% |
| 2025 holdout | -7.27% | +11.52% | -18.79% |
| 2026 robustness | +3.12% | -5.88% | +9.00% |

“Active return” here is only the difference in cumulative total return. It is **not Jensen alpha**.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

The completed historical workflow is intentionally split into auditable stages:

```bash
python build_historical_market.py
python build_timestamp_safe_alignment.py
python build_master_session_dataset.py
python run_directional_experiment.py
python evaluate_oos_uncertainty.py
python run_garch_oos_simulation.py
```

Important frozen specifications:

```text
config/directional_experiment_2023_2026.json
config/oos_uncertainty_spec.json
config/garch_simulation_spec.json
```

Launch the read-only research dashboard:

```bash
python -m streamlit run app.py
```

## Core outputs

```text
data/master_session_dataset_2023_2026.csv
data/master_session_dataset_2023_2026_summary.json

outputs/directional_experiment_v1/
  metrics.csv
  predictions.csv
  incremental_sentiment_comparison.csv
  selected_hyperparameters_from_2024.json
  uncertainty/
    oos_calibration_metrics.csv
    paired_block_bootstrap_summary.csv

outputs/garch_simulation_v1/
  garch_forecasts.csv
  garch_forecast_metrics.csv
  simulation_metrics.csv
  paired_sentiment_simulation_comparison.csv
```

Generated historical data and result files are excluded from version control so stale local artifacts are not mistaken for source code.

## Interpretation guardrails

This project does **not** claim:

- causal news effects;
- statistically resolved sentiment alpha;
- Jensen alpha;
- live-trading profitability;
- persistence outside the studied sample;
- that FinBERT probabilities or directional probabilities are perfectly calibrated.

A null or negative sentiment result is retained rather than optimized away.

For interview-ready project defense, see [docs/INTERVIEW_DEFENSE_2023_2026.md](docs/INTERVIEW_DEFENSE_2023_2026.md).
