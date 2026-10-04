# Pipeline

The canonical pipeline is the **2023–2026 timestamp-safe historical workflow**. The earlier short-sample prototype pipeline has been removed.

## 1. Historical news acquisition and FinBERT scoring

The historical build uses GDELT GAL in BigQuery plus resumable local checkpoints.

Relevant source files:

~~~text
bigquery_gdelt_backfill.py
finbert_backfill.py
consolidate_bigquery_backfill.py
consolidate_finbert_backfill.py
run_fast_one_year_rebuild.py
run_fast_two_year_rebuild.py
run_fast_2026_rebuild.py
~~~

The fast runners were designed to process bounded weekly ranges and save durable checkpoints so failures do not require restarting the entire history.

The physical checkpoint directories retain the original `*_2023_2025` baseline name because the 2026 extension was deliberately appended to that already-validated store. Final consolidated outputs use `2023_2026` names. The directory label is therefore historical provenance, not a claim that 2026 data are absent.

BigQuery credentials are local environment/application credentials and are never committed.

## 2. CSI 300 market history

~~~bash
python build_historical_market.py
~~~

Creates locally generated market history/features beginning in 2022.

2022 is warm-up only.

## 3. Timestamp-safe news alignment

~~~bash
python build_timestamp_safe_alignment.py
~~~

The cutoff is 15:00 Asia/Shanghai.

Each headline is assigned to:

~~~text
(previous genuine CSI 300 close, current genuine CSI 300 close]
~~~

Processing is month-by-month with compressed checkpoints.

## 4. Master session dataset

~~~bash
python build_master_session_dataset.py
~~~

Builds one row per CSI 300 session with:

- market state;
- session-unique FinBERT sentiment;
- strict rolling sentiment;
- prior-only news intensity;
- sentiment-volatility interactions;
- genuine next-session targets.

## 5. Frozen directional experiment

Experiment definition:

~~~text
config/directional_experiment_2023_2026.json
~~~

Run:

~~~bash
python run_directional_experiment.py
~~~

The script verifies the SHA-256 of the locally generated master dataset before fitting.

Design:

- target-year 2023: initial development;
- target-year 2024: monthly expanding validation;
- target-year 2025: untouched holdout;
- target-year 2026: locked-model robustness.

Models:

- logistic market-only;
- logistic market + sentiment;
- XGBoost market-only;
- XGBoost market + sentiment.

## 6. OOS uncertainty and calibration

Frozen definition:

~~~text
config/oos_uncertainty_spec.json
~~~

Run:

~~~bash
python evaluate_oos_uncertainty.py
~~~

Reports:

- calibration intercept/slope;
- 5-bin ECE/reliability tables;
- paired circular moving-block bootstrap;
- 5,000 resamples;
- 10-session blocks;
- 95% percentile intervals.

## 7. GARCH risk overlay and transaction simulation

Frozen definition:

~~~text
config/garch_simulation_spec.json
~~~

Run:

~~~bash
python run_garch_oos_simulation.py
~~~

GARCH(1,1) forecasts next-session volatility using expanding history. It only scales positions and never generates direction.

The fixed simulation uses the saved genuine OOS probabilities.

## 8. Dashboard

~~~bash
python -m streamlit run app.py
~~~

The dashboard is read-only with respect to research results. It visualizes saved local artifacts; it does not silently retrain models.

## 9. Validation

~~~bash
python -m unittest discover -s tests -v
python validate_repository.py
~~~

CI runs compile checks, the synthetic methodology suite, and repository validation.

Large historical datasets and generated outputs are intentionally excluded from Git, so CI validates source methodology rather than attempting to rebuild the full BigQuery/FinBERT history.
