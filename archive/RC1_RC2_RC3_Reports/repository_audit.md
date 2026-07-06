# Repository Audit Report

## PRODUCTION (Do Not Delete)
- `app.py`: Read-only operational dashboard.
- `run_daily_pipeline.py`: Main orchestration script.
- `module1_news_extraction.py` to `module14_live_monitoring.py`: Core execution modules.
- `config/institutional_config.py`: Immutable configuration bounds.
- `models/model_live.pkl`: Production model.
- `models/scaler.pkl`: Production scaler.
- `deployment/`: Containerization manifests.

## RESEARCH (Keep for Documentation)
- `module15_advanced_modeling.py` to `module19_explainability.py`.
- `history/`: Frozen historical dataset and partitions.
- `models/model_candidate.pkl`: Audited candidate model.

## ARCHIVE (Keep for Lineage)
- `archive/`: Old audits, disaster recovery backups.
- `logs/`: Execution logs.

## TEMPORARY / SCRATCH (Recommend Deletion)
- `scratch/`: AI subagent scratchpads.
- `err.txt`, `test_out.txt`: Temporary command outputs.
- `__pycache__`, `.venv`, `venv`: Local execution caches.
- `fill_gap.py`: Deprecated synthetic data script.
