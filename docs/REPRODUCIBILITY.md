# Reproducibility

This document describes the cleaned 2023-2026 historical research workflow.

Large raw datasets and fitted model binaries are intentionally not tracked. Compact empirical evidence is published under `results/`.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

The historical news pipeline requires Google BigQuery credentials with access to the public GDELT datasets. Credentials remain local and are not stored in this repository.

## 2. Historical news acquisition

Operational news-acquisition utilities are kept under `src/data_pipeline/` rather than at repository root.

### Collect GDELT GAL headlines

```bash
python -m src.data_pipeline.collect_news \
  --project <GCP_PROJECT> \
  --start 2023-01-01 \
  --end 2026-10-03 \
  --output-dir data/gdelt_headlines
```

The collector checkpoints each day so an interrupted run can resume without rebuilding completed dates.

### Consolidate the daily headline checkpoints

```bash
python -m src.data_pipeline.consolidate_news \
  --start 2023-01-01 \
  --end 2026-10-03 \
  --input-dir data/gdelt_headlines \
  --output data/news_daily_historical.csv \
  --summary data/gdelt_headlines_summary.json
```

### Score headlines with FinBERT

```bash
python -m src.data_pipeline.score_sentiment \
  --input data/news_daily_historical.csv \
  --output-dir data/finbert_scores \
  --start 2023-01-01 \
  --end 2026-10-03
```

Use an appropriate batch size for the available CPU/GPU environment.

### Consolidate sentiment checkpoints

```bash
python -m src.data_pipeline.consolidate_sentiment \
  --news data/news_daily_historical.csv \
  --input-dir data/finbert_scores \
  --output data/sentiment_features_historical.csv \
  --summary data/finbert_scores_summary.json
```

## 3. CSI 300 market history

```bash
python build_historical_market.py
```

This generates CSI 300 history and market-only features beginning in 2022. The 2022 observations are warm-up only.

## 4. Timestamp-safe news-to-session alignment

```bash
python build_session_alignment.py \
  --article-dir data/gdelt_headlines/daily \
  --score-dir data/finbert_scores/headline_scores
```

Headlines are mapped to CSI 300 sessions using Shanghai-local timestamps and the 15:00 market close. After-close and non-trading-day news is carried forward to the next eligible session.

Processing is month-by-month with compressed checkpoints.

## 5. Master session dataset

```bash
python build_master_session_dataset.py
```

The master table combines market variables, session-unique FinBERT sentiment, rolling sentiment, news intensity, interaction features, and next-session targets.

## 6. Directional experiment

Frozen specification:

```text
config/directional_experiment_2023_2026.json
```

Run:

```bash
python run_directional_experiment.py
```

The script verifies the master-dataset SHA-256 before fitting.

The split is:

- 2023 target sessions: initial training;
- 2024 target sessions: 12 monthly expanding validation folds;
- 2025 target sessions: untouched holdout;
- 2026 target sessions: temporal robustness using the same 2023-2024 fit.

Principal models:

- logistic market-only;
- logistic market + sentiment;
- XGBoost market-only;
- XGBoost market + sentiment.

## 7. OOS uncertainty and calibration

Frozen specification:

```text
config/oos_uncertainty_spec.json
```

Run:

```bash
python evaluate_oos_uncertainty.py
```

Outputs include calibration diagnostics and the paired 5,000-replication, 10-session moving-block bootstrap.

## 8. GARCH risk overlay and transaction simulation

Frozen specification:

```text
config/garch_simulation_spec.json
```

Run:

```bash
python evaluate_garch_risk_overlay.py
```

GARCH(1,1) forecasts next-session volatility using expanding history and is used for position scaling only.

The transaction-cost simulation uses saved out-of-sample directional probabilities.

## 9. Dashboard

```bash
python -m streamlit run app.py
```

The dashboard reads generated local artifacts when available and falls back to compact tracked result tables for the principal evidence.

## 10. Validation

```bash
python -m unittest discover -s tests -v
python scripts/validate_repository.py
```

CI runs compile checks, methodology tests, and repository validation.

## Published versus local artifacts

Tracked under `results/`:

- directional OOS metrics;
- incremental sentiment comparisons;
- OOS prediction probabilities;
- selected hyperparameters;
- calibration metrics;
- paired block-bootstrap summaries;
- GARCH forecasts and period diagnostics;
- simulation metrics;
- paired simulation comparisons;
- result manifest.

Kept local/generated:

- full GDELT headline history;
- per-headline FinBERT checkpoint files;
- full timestamp-aligned article table;
- master-session CSV;
- fitted model binaries;
- bootstrap replication draws;
- full simulation paths.
