# Reproducibility and Research Pipeline

The canonical workflow is the 2023-2026 historical research pipeline.

## 1. Historical news acquisition and FinBERT scoring

The historical reconstruction uses GDELT GAL in BigQuery with resumable local checkpoints.

Relevant source files:

```text
bigquery_gdelt_backfill.py
finbert_backfill.py
consolidate_bigquery_backfill.py
consolidate_finbert_backfill.py
run_fast_one_year_rebuild.py
run_fast_two_year_rebuild.py
run_fast_2026_rebuild.py
```

The rebuild runners process bounded date ranges and save checkpoints so interrupted collection can resume without restarting the full history.

BigQuery credentials remain local and are not stored in the repository.

## 2. CSI 300 market history

```bash
python build_historical_market.py
```

This generates CSI 300 market history and features beginning in 2022. The 2022 observations are used as warm-up.

## 3. News-to-session alignment

```bash
python build_session_alignment.py
```

Headlines are mapped to CSI 300 sessions using Shanghai-local timestamps and the 15:00 market close. After-close and non-trading-day news is carried forward to the next eligible session.

Processing is month-by-month with compressed checkpoints.

## 4. Master session dataset

```bash
python build_master_session_dataset.py
```

The master table includes:

- market variables;
- session-unique FinBERT sentiment;
- rolling sentiment;
- prior-session news intensity;
- sentiment-volatility interactions;
- next-session targets.

## 5. Directional experiment

Experiment definition:

```text
config/directional_experiment_2023_2026.json
```

Run:

```bash
python run_directional_experiment.py
```

The script verifies the SHA-256 of the generated master dataset before fitting. The directional split and candidate grid were fixed before model fitting; the bootstrap settings and simulation conventions were likewise fixed before their respective analyses. The public release history was later consolidated, while the configuration files retain the settings used to generate the reported results.

Design:

- 2023 target sessions: initial training;
- 2024 target sessions: monthly expanding validation;
- 2025 target sessions: holdout evaluation;
- 2026 target sessions: temporal robustness using the same 2023-2024 fitted model.

Models:

- logistic market-only;
- logistic market + sentiment;
- XGBoost market-only;
- XGBoost market + sentiment.

## 6. OOS uncertainty and calibration

Configuration:

```text
config/oos_uncertainty_spec.json
```

Run:

```bash
python evaluate_oos_uncertainty.py
```

Outputs include:

- calibration intercept and slope;
- five-bin ECE and reliability tables;
- paired circular moving-block bootstrap;
- 5,000 resamples;
- 10-session blocks;
- 95% percentile intervals.

## 7. GARCH risk overlay and transaction simulation

Configuration:

```text
config/garch_simulation_spec.json
```

Run:

```bash
python run_garch_oos_simulation.py
```

GARCH(1,1) forecasts next-session volatility using expanding history and is used for position scaling.

The simulation uses saved out-of-sample directional probabilities.

## 8. Dashboard

```bash
python -m streamlit run app.py
```

The dashboard reads generated result files and does not retrain models.

## 9. Validation

```bash
python -m unittest discover -s tests -v
python validate_repository.py
```

CI runs compile checks, methodology tests, and repository validation.

Large historical datasets, fitted model binaries, bootstrap replication files, and full simulation paths are excluded from Git. Compact empirical result tables are stored under `results/`.
