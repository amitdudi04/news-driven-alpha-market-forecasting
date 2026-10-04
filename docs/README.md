# Research Documentation

This directory contains the research documentation for the completed 2023–2026 CSI 300 news-sentiment study.

The recommended reading order is:

1. [Architecture](ARCHITECTURE.md) — system boundaries and separation between sentiment, direction forecasting, uncertainty, GARCH, and simulation.
2. [Data Card](DATA_CARD.md) — data sources, date coverage, timestamp semantics, de-duplication, and known data limitations.
3. [Data Dictionary](DATA_DICTIONARY.md) — definitions for the final 908-session master dataset.
4. [Experiment Design](EXPERIMENT_DESIGN.md) — frozen split, feature groups, model families, selection rules, and holdout policy.
5. [Results](RESULTS.md) — genuine OOS directional, calibration, bootstrap, GARCH, and fixed-rule simulation results.
6. [Model Card](MODEL_CARD.md) — intended use, model scope, evaluation, calibration, and limitations.
7. [Limitations](LIMITATIONS.md) — threats to validity and claims the project does not make.
8. [Reproducibility](REPRODUCIBILITY.md) — exact build/evaluation sequence and local-artifact policy.
9. [Academic Disclosure](ACADEMIC_DISCLOSURE.md) — concise statement of scope and research claims.
10. [Oral Defense Guide](ORAL_DEFENSE_GUIDE.md) — concise explanations of the methodological choices and likely technical questions.

The repository intentionally keeps the large historical datasets, fitted model binaries, bootstrap-draw files, and full simulation paths out of Git. A compact review snapshot of the reported OOS results is committed under `results/`. The source code, frozen specifications, tests, documentation, and committed result snapshot define the auditable research release; compatible local artifacts can be checked with `python validate_repository.py`.
