# News-Driven Alpha: Financial Sentiment and CSI 300 Forecasting

This project studies whether China-focused financial-news sentiment adds incremental next-session forecasting information for the CSI 300 beyond market-only predictors.

The research workflow covers historical news collection, FinBERT sentiment scoring, trading-session alignment, CSI 300 market features, logistic-regression and XGBoost benchmarks, out-of-sample evaluation, calibration, block-bootstrap uncertainty, a separate GARCH(1,1) volatility overlay, and a transaction-cost simulation.

## Study design

The modeling sample is organized by the year of the **target trading session**:

| Target period | Role | Model-ready rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | holdout evaluation | 223 |
| 2026 | temporal robustness | 181 |

The final directional models are trained on 2023-2024 after model selection. The same 2023-2024 fit is then evaluated on 2025 and carried forward unchanged for the 2026 robustness period.

## Data

### News

Historical English-language China-finance headlines are collected from GDELT GAL through BigQuery and scored with pretrained ProsusAI/FinBERT.

- 292,373 scored headline observations enter the session-alignment stage.
- 291,973 observations map to completed CSI 300 trading-session windows.
- 276,960 normalized session-unique headlines are used for pooled sentiment features.
- 400 late-Sep/Oct 2026 observations occur after the final completed CSI 300 close in the market file and remain unassigned.

Headlines are aligned using Shanghai-local timestamps and the 15:00 CSI 300 market close. After-close, weekend, and holiday news is moved forward to the next eligible trading session.

### Market

CSI 300 history covers 4 Jan 2022 through 30 Sep 2026.

- 2022 is used as warm-up for rolling market variables and GARCH history.
- 1,150 trading sessions are available in total.
- The research sample contains 908 sessions from 2023 onward.
- Market history is sourced from the China Securities Index feed through AkShare and cross-checked against Sina history.

## Master session dataset

Each CSI 300 trading-session row contains:

- current log return;
- 20-session realized volatility;
- 5/20 momentum;
- momentum acceleration;
- volatility-regime indicator;
- unique headline count;
- pooled FinBERT sentiment mean and dispersion;
- positive, negative, and neutral FinBERT shares;
- complete-window 5/10/20-session rolling sentiment;
- prior-20-session news intensity;
- sentiment x volatility interactions;
- next-session return and direction targets.

The no-news research session, 20 Jun 2025, retains missing sentiment rather than being converted to a neutral score.

The final table contains 908 sessions, 907 known next-session targets, and 867 rows with the full feature set used by the principal models.

## Models

Four directional models are compared:

1. Logistic regression - market only
2. Logistic regression - market + sentiment
3. XGBoost - market only
4. XGBoost - market + sentiment

The central comparison is within model family: the market-plus-sentiment specification is evaluated against the corresponding market-only baseline.

## Out-of-sample results

### 2025 holdout

| Model | Balanced accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|
| Logistic market | 0.483 | 0.2562 | 0.7057 | 0.509 |
| Logistic + sentiment | 0.513 | 0.2568 | 0.7070 | 0.513 |
| XGBoost market | 0.511 | 0.2624 | 0.7191 | 0.506 |
| XGBoost + sentiment | 0.508 | 0.2604 | 0.7146 | 0.519 |

The 2025 results are mixed: sentiment improves some classification or probability metrics, but not all of them simultaneously.

### 2026 temporal robustness

| Model | Balanced accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|
| Logistic market | 0.481 | 0.2563 | 0.7059 | 0.486 |
| Logistic + sentiment | 0.514 | 0.2563 | 0.7063 | 0.493 |
| XGBoost market | 0.502 | 0.2531 | 0.6992 | 0.542 |
| XGBoost + sentiment | **0.557** | **0.2502** | **0.6935** | **0.546** |

The largest point improvement appears for XGBoost in 2026. Its balanced-accuracy difference is approximately +5.48 percentage points. The paired 10-session moving-block-bootstrap interval is approximately -0.20 to +11.83 percentage points, so the interval includes zero.

## Calibration and uncertainty

Calibration is evaluated with intercept, slope, and five-bin expected calibration error. Probability calibration is imperfect, with slopes generally below 1.

Incremental model differences are evaluated with a paired circular moving-block bootstrap:

- 5,000 resamples;
- 10-session blocks;
- 95% percentile intervals;
- identical sampled indices for each market-only / market-plus-sentiment pair.

The 2025 incremental intervals include zero across the principal comparisons. The 2026 XGBoost point estimates are stronger, but their principal intervals also include zero.

## GARCH volatility overlay

GARCH(1,1) is estimated separately from the directional models and is used only for position scaling.

- 907 one-step volatility forecasts
- 0 convergence failures
- expanding return history
- zero-mean Normal GARCH(1,1)

The position scale compares the current forecast volatility with the prior forecast-volatility history and is capped at 1.0.

## Transaction-cost simulation

The simulation uses the saved out-of-sample directional probabilities with:

- long when p(up) >= 0.55;
- short when p(up) <= 0.45;
- otherwise flat;
- maximum absolute position = 1;
- 10 bps transaction cost per unit of turnover.

For XGBoost + sentiment with GARCH scaling:

| Period | Net strategy return | CSI 300 benchmark |
|---|---:|---:|
| 2024 validation | +5.54% | +14.68% |
| 2025 holdout | -7.27% | +11.52% |
| 2026 robustness | +3.12% | -5.88% |

The simulation is reported separately from the directional forecasting metrics.

## Repository structure

```text
config/                 experiment and simulation specifications
docs/                   methodology, data, results, and limitations
results/                compact empirical result tables
tests/                  methodology tests
build_historical_market.py
build_session_alignment.py
build_master_session_dataset.py
run_directional_experiment.py
evaluate_oos_uncertainty.py
run_garch_oos_simulation.py
app.py
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the historical research stages:

```bash
python build_historical_market.py
python build_session_alignment.py
python build_master_session_dataset.py
python run_directional_experiment.py
python evaluate_oos_uncertainty.py
python run_garch_oos_simulation.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

Run repository validation:

```bash
python -m unittest discover -s tests -v
python validate_repository.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Data Card](docs/DATA_CARD.md)
- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [Experiment Design](docs/EXPERIMENT_DESIGN.md)
- [Results](docs/RESULTS.md)
- [Model Card](docs/MODEL_CARD.md)
- [Limitations](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Research Scope](docs/RESEARCH_SCOPE.md)
- [Academic Disclosure](docs/ACADEMIC_DISCLOSURE.md)
- [Oral Defense Guide](docs/ORAL_DEFENSE_GUIDE.md)
- [Empirical Result Files](results/README.md)
