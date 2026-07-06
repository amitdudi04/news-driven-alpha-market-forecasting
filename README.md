# News-Driven Alpha: Chinese Equity Markets
![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.13-blue)

An institutional-grade paper trading infrastructure utilizing NLP-derived sentiment (ProsusAI FinBERT) applied to GDELT global news streams to forecast directional volatility regimes in the CSI 300 index.

## Motivation
Traditional asset pricing models often fail to rapidly assimilate unstructured geopolitical and macroeconomic news flow. This platform serves as a deterministic execution engine that algorithmically translates raw linguistic sentiment into statistically governed, walk-forward evaluated trading signals.

## Architecture & Pipeline
The architecture consists of 14 strictly governed sequential modules:
1. **News Extraction** (GDELT API)
2. **Sentiment Analysis** (FinBERT)
3. **Market Data** (yfinance)
4. **Feature Engineering** (Rolling percentiles, IC filtering)
5. **Multi-Model Inference** (XGBoost Regime-Switching)
6. **Execution Engine** (SAFE MODE, Entropy Thresholding)

## Quick Start
```bash
# Clone repository
git clone https://github.com/username/news-driven-alpha.git
cd news-driven-alpha

# Install requirements
pip install -r requirements.txt

# Run deterministic pipeline
python run_daily_pipeline.py
```

## Known Limitations
The walk-forward evaluation is constrained by API ingestion limits restricting the pure chronological holdout to 104 trading days (21 out-of-sample days). Extreme statistical limitations apply. **Authorized for Paper Trading Only.**
