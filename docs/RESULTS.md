# Results — 2023–2026 Historical Study

## Research question

Does **timestamp-safe China-focused financial-news sentiment** improve next-session CSI 300 directional forecasts beyond market-only information?

The primary comparison is paired within model family:

- logistic market-only vs logistic market + sentiment;
- XGBoost market-only vs XGBoost market + sentiment.

The study separates directional forecasting from the GARCH risk overlay.

## 1. Historical sample

### News layer

| Item | Result |
|---|---:|
| News period | 1 Jan 2023 – 3 Oct 2026 |
| FinBERT-scored headline observations entering alignment | 292,373 |
| Assigned to completed market-close windows | 291,973 |
| Pending after last known CSI 300 close | 400 |
| Session-unique normalized headlines used for pooled sentiment | 276,960 |
| Within-session repeats removed | 15,013 |

The 400 pending observations occur after the 30 Sep 2026 close and are not assigned backward.

### CSI 300 market layer

| Item | Result |
|---|---:|
| Market history | 4 Jan 2022 – 30 Sep 2026 |
| Total trading sessions | 1,150 |
| 2022 warm-up sessions | 242 |
| Research sessions from 2023 | 908 |
| Primary / cross-check overlap | 1,150 sessions |
| Max close discrepancy | 0.005 index points |

2022 is used only for rolling-feature and GARCH warm-up.

## 2. Timestamp-safe alignment

Forecast cutoff: **15:00 Asia/Shanghai**.

Information window:

```text
(previous CSI 300 trading close, current CSI 300 trading close]
```

Headline assignment:

| Relationship | Headlines |
|---|---:|
| same trading day at/before close | 130,233 |
| after-close trading-day news moved to next session | 102,814 |
| non-trading-day news moved to next session | 58,926 |
| pending after last known close | 400 |

Audit:

- article/FinBERT unmatched rows: **0**
- Shanghai-date mismatches: **0**
- causal-window violations: **0**

## 3. Master session dataset

| Item | Result |
|---|---:|
| CSI 300 master sessions | 908 |
| Known next-session targets | 907 |
| Strict model-ready rows | 867 |
| Genuine no-news sessions | 1 |
| No-news date | 20 Jun 2025 |

The no-news session retains missing sentiment; it is not converted to neutral sentiment.

Target balance among all 907 labeled sessions:

- up: 450
- down/flat: 457

## 4. Frozen experimental split

The split uses **target_session_date**.

| Target year | Role | Strict rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | untouched holdout | 223 |
| 2026 | locked-model robustness | 181 |

There are 12 expanding monthly validation folds in 2024 for each of four models. Audit found **0 fold leakages**.

Hyperparameters are selected by **lowest aggregate 2024 Brier score**, with log loss and candidate ID as tie-breakers.

Selected specifications:

- Logistic market-only: C = 0.1
- Logistic + sentiment: C = 0.1
- XGBoost market-only: 150 trees, depth 2, learning rate 0.03, min child weight 5, subsample/column sample 0.9, L2 = 5
- XGBoost + sentiment: same XGBoost configuration

## 5. Directional OOS performance

### 2024 chronological development validation

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4996 | 0.5041 | 0.2703 | 0.7469 | 0.5080 |
| Logistic + sentiment | 0.4923 | 0.4959 | 0.2770 | 0.7724 | 0.5151 |
| XGB market | 0.5188 | 0.5207 | 0.2623 | 0.7209 | 0.5354 |
| XGB + sentiment | 0.5301 | 0.5331 | 0.2665 | 0.7306 | 0.5065 |

Development evidence is mixed.

### 2025 untouched holdout

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4829 | 0.4709 | 0.2562 | 0.7057 | 0.5086 |
| Logistic + sentiment | 0.5127 | 0.5022 | 0.2568 | 0.7070 | 0.5132 |
| XGB market | 0.5106 | 0.4888 | 0.2624 | 0.7191 | 0.5057 |
| XGB + sentiment | 0.5078 | 0.4843 | 0.2604 | 0.7146 | 0.5194 |

Interpretation:

- Logistic + sentiment improves threshold classification but slightly worsens Brier/log loss.
- XGB + sentiment slightly worsens threshold classification but improves Brier, log loss, and AUC.
- There is no uniform sentiment advantage.

### 2026 locked-model robustness

The same final 2023–2024 fits are used; 2025 is **not** added to training.

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4806 | 0.4807 | 0.2563 | 0.7059 | 0.4856 |
| Logistic + sentiment | 0.5144 | 0.5138 | 0.2563 | 0.7063 | 0.4929 |
| XGB market | 0.5020 | 0.5028 | 0.2531 | 0.6992 | 0.5424 |
| XGB + sentiment | **0.5568** | **0.5580** | **0.2502** | **0.6935** | **0.5458** |

The strongest point evidence occurs for XGB + sentiment in 2026.

## 6. Calibration

Ideal calibration intercept = 0 and slope = 1.

Selected examples:

| Model | Period | Intercept | Slope | 5-bin ECE |
|---|---|---:|---:|---:|
| Logistic market | 2025 | 0.201 | 0.310 | 0.0840 |
| Logistic + sentiment | 2025 | 0.191 | 0.271 | 0.0760 |
| XGB market | 2025 | 0.189 | 0.159 | 0.0984 |
| XGB + sentiment | 2025 | 0.225 | 0.309 | 0.1084 |
| XGB market | 2026 | 0.032 | 0.384 | 0.0963 |
| XGB + sentiment | 2026 | 0.074 | 0.556 | **0.0485** |

Calibration slopes are generally far below 1. The probabilities should therefore be treated as ranking/forecast scores rather than perfectly calibrated event probabilities.

## 7. Paired block-bootstrap uncertainty

Frozen procedure:

- circular moving-block bootstrap;
- 10 trading-session blocks;
- 5,000 paired resamples;
- identical bootstrap indices for market-only and market+sentiment;
- 95% percentile intervals.

A positive increment favors sentiment.

### 2025 holdout

| Family | Metric | Point increment | 95% interval | Resolution |
|---|---|---:|---:|---|
| Logistic | balanced accuracy | +0.0299 | [-0.0127, +0.0737] | unresolved |
| Logistic | Brier improvement | -0.00058 | [-0.00544, +0.00436] | unresolved |
| XGB | balanced accuracy | -0.0028 | [-0.0495, +0.0440] | unresolved |
| XGB | Brier improvement | +0.00199 | [-0.00569, +0.00951] | unresolved |
| XGB | log-loss improvement | +0.00451 | [-0.01150, +0.02037] | unresolved |
| XGB | AUC improvement | +0.01375 | [-0.04464, +0.07103] | unresolved |

All 2025 incremental effects are statistically unresolved at the 95% block-bootstrap level.

### 2026 robustness

| Family | Metric | Point increment | 95% interval | Resolution |
|---|---|---:|---:|---|
| Logistic | balanced accuracy | +0.0338 | [-0.0132, +0.0810] | unresolved |
| XGB | balanced accuracy | **+0.0548** | **[-0.0020, +0.1183]** | unresolved |
| XGB | Brier improvement | +0.00293 | [-0.00532, +0.01126] | unresolved |
| XGB | log-loss improvement | +0.00578 | [-0.01113, +0.02280] | unresolved |
| XGB | ECE improvement | +0.04779 | [-0.04995, +0.09233] | unresolved |
| XGB | AUC improvement | +0.00336 | [-0.05697, +0.06506] | unresolved |

For XGB 2026 balanced accuracy, 97.0% of bootstrap draws are positive, but the 95% interval still crosses zero. This is suggestive, not resolved.

### Resolved negative development result

In 2024 logistic validation:

- Brier increment = **-0.00670**, 95% CI **[-0.01341, -0.00045]**
- log-loss increment = **-0.02550**, 95% CI **[-0.04985, -0.00466]**

Under the frozen interpretation rule, sentiment is resolved as **worse** on these two probabilistic metrics in that development period.

## 8. GARCH(1,1) volatility overlay

GARCH is separate from direction prediction.

- expanding one-step forecasts;
- zero mean;
- Normal innovations;
- returns scaled ×100 during fitting;
- 907 forecasts from 3 Jan 2023 through 29 Sep 2026;
- **0 convergence failures**.

OOS GARCH diagnostics:

| Period | N | Mean forecast vol | Mean |r next| | Mean QLIKE | Mean risk scale |
|---|---:|---:|---:|---:|---:|
| 2024 validation | 242 | 1.114% | 0.855% | -8.015 | 0.889 |
| 2025 holdout | 223 | 1.031% | 0.685% | -8.095 | 0.927 |
| 2026 robustness | 181 | 1.117% | 0.908% | -7.921 | 0.876 |

QLIKE is reported as `log(h) + r²/h`; only relative comparisons are meaningful because additive constants are omitted.

## 9. Fixed-rule transaction-cost simulation

Frozen conventions:

- long if p(up) >= 0.55;
- short if p(up) <= 0.45;
- otherwise flat;
- max absolute exposure 1.0;
- 10 bps cost per unit of turnover;
- each reporting period starts flat;
- GARCH position scale is capped at 1;
- market log returns are converted with `expm1` before strategy compounding.

### XGBoost models — GARCH-scaled simulation

| Period | Model | Net return | Benchmark | Active-return difference | Net Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| 2024 | market only | +0.52% | +14.68% | -14.16% | 0.11 | -10.25% |
| 2024 | + sentiment | **+5.54%** | +14.68% | -9.14% | **0.49** | -11.02% |
| 2025 | market only | -10.71% | +11.52% | -22.23% | -0.94 | -19.77% |
| 2025 | + sentiment | **-7.27%** | +11.52% | -18.79% | **-0.57** | **-17.98%** |
| 2026 | market only | -0.45% | -5.88% | +5.43% | 0.02 | **-8.25%** |
| 2026 | + sentiment | **+3.12%** | -5.88% | **+9.00%** | **0.40** | -10.85% |

The sentiment XGB strategy outperforms the paired market-only strategy in cumulative return in all three reporting periods under this fixed simulation. This **does not** establish profitable alpha:

- both XGB strategies materially underperform the benchmark in 2024 and 2025;
- simulation thresholds are fixed conventions rather than optimized policies;
- directional probabilities are imperfectly calibrated;
- no uncertainty interval is attached to simulated-return differences here;
- the simulation is historical and not live-capital evidence.

### GARCH overlay effect

The volatility scale generally reduces maximum drawdown magnitude relative to the unscaled signal, but total-return effects are mixed. This is consistent with its intended role as a **risk overlay**, not an alpha generator.

## 10. Final conclusion

The historical reconstruction materially changes the project's evidentiary status.

The project is no longer a 49-day feasibility sample. It is now a multi-year, timestamp-safe OOS study with:

- genuine 2025 holdout;
- locked 2026 robustness;
- paired model-family ablations;
- calibration diagnostics;
- block-bootstrap uncertainty;
- separate volatility overlay;
- transaction-cost simulation.

The central result remains cautious:

> Sentiment sometimes improves CSI 300 directional forecasts and fixed-rule simulated performance, especially for XGBoost in 2026, but the incremental directional effects are not statistically resolved at the 95% block-bootstrap level. The evidence supports a mixed incremental-information conclusion, not a durable-alpha claim.
