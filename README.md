# News-Driven Alpha: Financial Sentiment and CSI 300 Forecasting

This repository studies whether China-focused financial-news sentiment adds incremental next-session forecasting information for the CSI 300 beyond market-only predictors.

The final research design is a timestamp-safe, multi-year out-of-sample study with paired market-only versus market-plus-sentiment models, calibration diagnostics, paired moving-block-bootstrap uncertainty, a separate GARCH(1,1) volatility overlay, and a fixed-rule transaction-cost simulation.

## Main finding

The evidence is mixed rather than uniformly positive.

- The strict modeling sample contains **867** fully usable session/target rows.
- Hyperparameters are selected using **2024 monthly expanding-window validation only**.
- **2025 (223 targets)** is the untouched holdout.
- **2026 (181 targets)** is a locked-model temporal robustness period using the same 2023-2024 final fit.
- In 2025, sentiment improves some metrics but worsens or leaves others unchanged.
- The strongest point result appears for **XGBoost in 2026**, where balanced accuracy rises from **50.20% to 55.68%** and Brier score improves from **0.25310 to 0.25017**.
- The paired 10-session moving-block-bootstrap 95% interval for that balanced-accuracy improvement is approximately **[-0.20, +11.83] percentage points**, so the improvement is not statistically resolved at the 95% level.
- A resolved negative development result is also retained: in 2024 logistic validation, sentiment worsens Brier score and log loss under the frozen bootstrap design.

The defensible conclusion is:

> Timestamp-safe financial-news sentiment shows model- and period-dependent incremental information, but the current out-of-sample evidence does not establish a stable, statistically resolved predictive advantage.

See [docs/RESULTS.md](docs/RESULTS.md) for the full empirical record.

## Data and timing

### News

Historical English-language China-finance headlines are collected from GDELT GAL through BigQuery and scored with pretrained ProsusAI/FinBERT.

- **292,373** scored headline observations enter the timestamp-alignment audit.
- **291,973** observations map to completed CSI 300 trading-session windows.
- **276,960** normalized session-unique headlines are used for pooled sentiment features.
- **400** late-Sep/Oct 2026 observations occur after the last completed CSI 300 close in the study and remain unassigned.

GDELT's time field is treated as a **GDELT seen timestamp**, not asserted to be the publisher's exact original publication time.

### Market

CSI 300 market history covers **4 Jan 2022 through 30 Sep 2026**.

- 2022 is warm-up only.
- Total trading sessions: **1,150**.
- Research sessions from 2023 onward: **908**.
- The historical series is sourced from the China Securities Index feed exposed through AkShare and cross-checked against Sina history.
- Across 1,150 overlapping sessions, the maximum close discrepancy is **0.005 index points**.

### Timestamp-safe information set

Every forecast is defined at the **15:00 Asia/Shanghai CSI 300 close**.

A headline is assigned to the first market close satisfying:

```text
previous CSI 300 trading close < GDELT seen timestamp <= current CSI 300 trading close
```

After-close, weekend, and holiday news therefore moves forward to the next eligible CSI 300 session.

Audit results:

- article/FinBERT unmatched rows: **0**
- Shanghai-date mismatches: **0**
- causal-window timing violations: **0**

## Master session dataset

The master dataset contains **908 CSI 300 sessions** from 3 Jan 2023 through 30 Sep 2026.

Feature groups include:

- current log return;
- 20-session realized volatility;
- 5/20 momentum and momentum acceleration;
- causal volatility-regime indicator;
- unique headline count;
- pooled FinBERT sentiment mean and dispersion;
- positive, negative, and neutral FinBERT shares;
- strict 5/10/20-session rolling sentiment;
- news intensity relative to the prior 20 sessions;
- sentiment x volatility interactions.

Missing sentiment is not neutralized. The one genuine no-news research session, **20 Jun 2025**, remains missing for sentiment features.

The target is the return sign of the next genuine CSI 300 trading session. Of 908 sessions, **907** have a known next-session target and **867** satisfy the strict full-feature modeling rule.

## Frozen experiment

Partitioning is based on **target_session_date**, not predictor-row calendar year.

| Target period | Role | Model-ready rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | 12 monthly expanding validation folds | 242 |
| 2025 | untouched holdout | 223 |
| 2026 | locked-model temporal robustness | 181 |

The experiment specification was frozen before fitting.

Principal models:

1. Logistic regression - market only
2. Logistic regression - market + sentiment
3. XGBoost - market only
4. XGBoost - market + sentiment

The research question is the **incremental value of sentiment within the same model family**, not whether a flexible model can predict the market in isolation.

## Genuine OOS results

### 2025 holdout

| Model | Balanced accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|
| Logistic market | 0.483 | 0.2562 | 0.7057 | 0.509 |
| Logistic + sentiment | 0.513 | 0.2568 | 0.7070 | 0.513 |
| XGBoost market | 0.511 | 0.2624 | 0.7191 | 0.506 |
| XGBoost + sentiment | 0.508 | 0.2604 | 0.7146 | 0.519 |

The holdout result is mixed: sentiment improves some classification or probability metrics, but not all of them simultaneously.

### 2026 temporal robustness

| Model | Balanced accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|
| Logistic market | 0.481 | 0.2563 | 0.7059 | 0.486 |
| Logistic + sentiment | 0.514 | 0.2563 | 0.7063 | 0.493 |
| XGBoost market | 0.502 | 0.2531 | 0.6992 | 0.542 |
| XGBoost + sentiment | **0.557** | **0.2502** | **0.6935** | **0.546** |

The 2026 XGBoost point estimates are stronger, but the principal paired bootstrap intervals still include zero.

## Calibration and uncertainty

Calibration is evaluated with intercept, slope, and five-bin expected calibration error. Slopes are generally below 1, so the model probabilities should not be interpreted as perfectly calibrated event probabilities.

Incremental sentiment differences use a **paired circular moving-block bootstrap**:

- 5,000 resamples;
- 10-session blocks;
- paired market-only / market-plus-sentiment rows;
- 95% percentile intervals.

For XGBoost 2026 balanced accuracy:

```text
point improvement: +5.48 percentage points
95% interval: approximately [-0.20, +11.83] percentage points
```

This is suggestive, not statistically resolved.

## GARCH risk overlay and fixed-rule simulation

GARCH(1,1) is estimated separately from the directional models. It forecasts next-session volatility for risk scaling only; it does not determine direction.

Frozen simulation conventions:

- p(up) >= 0.55 -> long
- p(up) <= 0.45 -> short
- otherwise -> flat
- maximum absolute position = 1.0
- transaction cost = 10 bps per unit of turnover
- GARCH risk scale is capped at 1.0

There are **907** one-step GARCH forecasts and **0 convergence failures**.

For XGBoost + sentiment with GARCH scaling:

| Period | Net strategy return | CSI 300 benchmark | Active-return difference |
|---|---:|---:|---:|
| 2024 validation | +5.54% | +14.68% | -9.14% |
| 2025 holdout | -7.27% | +11.52% | -18.79% |
| 2026 robustness | +3.12% | -5.88% | +9.00% |

"Active-return difference" is only cumulative strategy return minus cumulative benchmark return. It is **not Jensen alpha**.

The simulation is historical research, not evidence of live-capital profitability.

## Repository structure

```text
README.md
MODEL_CARD.md
ACADEMIC_DISCLOSURE.md
SECURITY.md
requirements.txt

config/                 frozen experiment and simulation specifications
docs/                   research design, data, results, limitations, reproducibility
src/data_pipeline/      historical GDELT and FinBERT implementation utilities
src/utils/              shared helpers
scripts/                release/repository validation
results/                compact tracked empirical evidence
tests/                  methodology tests

build_historical_market.py
build_session_alignment.py
build_master_session_dataset.py
run_directional_experiment.py
evaluate_oos_uncertainty.py
evaluate_garch_risk_overlay.py
app.py
```

The visible top-level research scripts correspond to the empirical stages a reviewer may want to inspect. Operational acquisition/scoring utilities are kept under `src/` rather than mixed into the repository root.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

The low-level historical news reconstruction utilities are under `src/data_pipeline/`. The main research stages are:

```bash
python build_historical_market.py
python build_session_alignment.py
python build_master_session_dataset.py
python run_directional_experiment.py
python evaluate_oos_uncertainty.py
python evaluate_garch_risk_overlay.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

Run release validation:

```bash
python -m unittest discover -s tests -v
python scripts/validate_repository.py
```

A fresh clone uses the compact tracked files under `results/` for the principal OOS, calibration, uncertainty, GARCH, and simulation evidence. Large historical datasets, fitted model binaries, bootstrap draws, and full simulation paths remain local/generated.

## Documentation

- [Research Scope](docs/RESEARCH_SCOPE.md)
- [Data Card](docs/DATA_CARD.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [Experiment Design](docs/EXPERIMENT_DESIGN.md)
- [Results](docs/RESULTS.md)
- [Limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Interview Defense Guide](docs/INTERVIEW_DEFENSE_2023_2026.md)
- [Model Card](MODEL_CARD.md)
- [Academic Disclosure](ACADEMIC_DISCLOSURE.md)
- [Empirical Result Files](results/README.md)

## Interpretation guardrails

This project does **not** claim:

- a causal effect of news sentiment on CSI 300 returns;
- statistically resolved sentiment alpha;
- Jensen alpha;
- live-trading profitability;
- persistence outside the studied period;
- perfectly calibrated directional probabilities.

Null and negative results are retained rather than optimized away.
