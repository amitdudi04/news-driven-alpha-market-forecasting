# Results

## 1. Sample

### News

| Item | Result |
|---|---:|
| News period | 1 Jan 2023 – 3 Oct 2026 |
| FinBERT-scored headline observations | 292,373 |
| Assigned to completed CSI 300 session windows | 291,973 |
| Remaining after final completed market close | 400 |
| Session-unique normalized headlines | 276,960 |
| Repeated within-session observations removed | 15,013 |

### Market

| Item | Result |
|---|---:|
| Market history | 4 Jan 2022 – 30 Sep 2026 |
| Total trading sessions | 1,150 |
| 2022 warm-up sessions | 242 |
| Research sessions from 2023 | 908 |
| Cross-source overlap | 1,150 sessions |
| Maximum close difference | 0.005 index points |

## 2. Session alignment

Headlines are aligned to CSI 300 trading sessions using Shanghai-local timestamps and the 15:00 market close.

| Assignment category | Headlines |
|---|---:|
| Same trading day by the close | 130,233 |
| After-close trading-day news carried to next session | 102,814 |
| Non-trading-day news carried to next session | 58,926 |
| Remaining after final completed close | 400 |

All assigned headline observations have corresponding FinBERT records and a completed-session assignment.

## 3. Master dataset

| Item | Result |
|---|---:|
| CSI 300 sessions | 908 |
| Known next-session targets | 907 |
| Model-ready rows | 867 |
| No-news sessions | 1 |
| No-news date | 20 Jun 2025 |

Target distribution among the 907 labeled sessions:

- positive next-session return: 450
- zero/negative next-session return: 457

## 4. Chronological evaluation design

| Target year | Role | Model-ready rows |
|---|---|---:|
| 2023 | initial training | 221 |
| 2024 | monthly expanding validation | 242 |
| 2025 | holdout evaluation | 223 |
| 2026 | temporal robustness | 181 |

The 2024 validation period contains 12 expanding monthly folds per model.

Selected specifications:

- Logistic market-only: C = 0.1
- Logistic + sentiment: C = 0.1
- XGBoost market-only: 150 trees, depth 2, learning rate 0.03, min child weight 5, subsample 0.9, column sample 0.9, L2 = 5
- XGBoost + sentiment: same XGBoost specification

## 5. Directional performance

### 2024 chronological validation

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4996 | 0.5041 | 0.2703 | 0.7469 | 0.5080 |
| Logistic + sentiment | 0.4923 | 0.4959 | 0.2770 | 0.7724 | 0.5151 |
| XGBoost market | 0.5188 | 0.5207 | 0.2623 | 0.7209 | 0.5354 |
| XGBoost + sentiment | 0.5301 | 0.5331 | 0.2665 | 0.7306 | 0.5065 |

### 2025 holdout

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4829 | 0.4709 | 0.2562 | 0.7057 | 0.5086 |
| Logistic + sentiment | 0.5127 | 0.5022 | 0.2568 | 0.7070 | 0.5132 |
| XGBoost market | 0.5106 | 0.4888 | 0.2624 | 0.7191 | 0.5057 |
| XGBoost + sentiment | 0.5078 | 0.4843 | 0.2604 | 0.7146 | 0.5194 |

The holdout evidence is mixed. Logistic sentiment improves threshold classification while slightly worsening probability loss. XGBoost sentiment improves Brier score, log loss, and AUC while slightly reducing balanced accuracy.

### 2026 temporal robustness

The 2026 evaluation uses the same 2023–2024 fitted models.

| Model | Balanced accuracy | Accuracy | Brier | Log loss | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic market | 0.4806 | 0.4807 | 0.2563 | 0.7059 | 0.4856 |
| Logistic + sentiment | 0.5144 | 0.5138 | 0.2563 | 0.7063 | 0.4929 |
| XGBoost market | 0.5020 | 0.5028 | 0.2531 | 0.6992 | 0.5424 |
| XGBoost + sentiment | **0.5568** | **0.5580** | **0.2502** | **0.6935** | **0.5458** |

The largest point improvement appears for XGBoost + sentiment in 2026.

## 6. Calibration

Ideal calibration has intercept 0 and slope 1.

| Model | Period | Intercept | Slope | 5-bin ECE |
|---|---|---:|---:|---:|
| Logistic market | 2025 | 0.201 | 0.310 | 0.0840 |
| Logistic + sentiment | 2025 | 0.191 | 0.271 | 0.0760 |
| XGBoost market | 2025 | 0.189 | 0.159 | 0.0984 |
| XGBoost + sentiment | 2025 | 0.225 | 0.309 | 0.1084 |
| XGBoost market | 2026 | 0.032 | 0.384 | 0.0963 |
| XGBoost + sentiment | 2026 | 0.074 | 0.556 | 0.0485 |

Calibration slopes are generally below 1.

## 7. Paired moving-block bootstrap

Bootstrap settings:

- 5,000 replications;
- 10-session blocks;
- paired market-only / market-plus-sentiment sampling;
- 95% percentile intervals.

### 2025 holdout

| Family | Metric | Point difference | 95% interval |
|---|---|---:|---:|
| Logistic | balanced accuracy | +0.0299 | [-0.0127, +0.0737] |
| Logistic | Brier improvement | -0.00058 | [-0.00544, +0.00436] |
| XGBoost | balanced accuracy | -0.0028 | [-0.0495, +0.0440] |
| XGBoost | Brier improvement | +0.00199 | [-0.00569, +0.00951] |
| XGBoost | log-loss improvement | +0.00451 | [-0.01150, +0.02037] |
| XGBoost | AUC improvement | +0.01375 | [-0.04464, +0.07103] |

All principal 2025 intervals shown above include zero.

### 2026 temporal robustness

| Family | Metric | Point difference | 95% interval |
|---|---|---:|---:|
| Logistic | balanced accuracy | +0.0338 | [-0.0132, +0.0810] |
| XGBoost | balanced accuracy | **+0.0548** | **[-0.0020, +0.1183]** |
| XGBoost | Brier improvement | +0.00293 | [-0.00532, +0.01126] |
| XGBoost | log-loss improvement | +0.00578 | [-0.01113, +0.02280] |
| XGBoost | ECE improvement | +0.04779 | [-0.04995, +0.09233] |
| XGBoost | AUC improvement | +0.00336 | [-0.05697, +0.06506] |

The 2026 XGBoost point estimates are stronger, although the displayed intervals include zero.

### 2024 logistic probability metrics

For logistic regression in 2024:

- Brier improvement: -0.00670, 95% interval [-0.01341, -0.00045]
- log-loss improvement: -0.02550, 95% interval [-0.04985, -0.00466]

These intervals are below zero, indicating worse probability-loss performance for the sentiment specification in that development period.

## 8. GARCH(1,1)

The volatility model uses expanding historical returns and produces one-step-ahead forecasts.

- 907 forecasts
- 0 convergence failures

| Period | N | Mean forecast vol | Mean absolute next return | Mean QLIKE | Mean risk scale |
|---|---:|---:|---:|---:|---:|
| 2024 validation | 242 | 1.114% | 0.855% | -8.015 | 0.889 |
| 2025 holdout | 223 | 1.031% | 0.685% | -8.095 | 0.927 |
| 2026 robustness | 181 | 1.117% | 0.908% | -7.921 | 0.876 |

## 9. Transaction-cost simulation

Simulation settings:

- long if p(up) ≥ 0.55;
- short if p(up) ≤ 0.45;
- otherwise flat;
- maximum absolute exposure 1;
- 10 bps cost per unit turnover.

### XGBoost with GARCH scaling

| Period | Model | Net return | Benchmark | Net Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|
| 2024 | market only | +0.52% | +14.68% | 0.11 | -10.25% |
| 2024 | + sentiment | **+5.54%** | +14.68% | **0.49** | -11.02% |
| 2025 | market only | -10.71% | +11.52% | -0.94 | -19.77% |
| 2025 | + sentiment | **-7.27%** | +11.52% | **-0.57** | **-17.98%** |
| 2026 | market only | -0.45% | -5.88% | 0.02 | **-8.25%** |
| 2026 | + sentiment | **+3.12%** | -5.88% | **0.40** | -10.85% |

The GARCH scale generally reduces exposure during higher-volatility periods. Return effects vary across periods.

## 10. Summary

Sentiment contributes differently across model families and time periods.

- The 2025 holdout results are mixed.
- The 2026 XGBoost sentiment specification has the strongest point results.
- Principal 2025 and 2026 bootstrap intervals shown above include zero.
- GARCH changes the risk profile but does not consistently improve return.
- The costed simulation does not dominate the CSI 300 benchmark across all periods.
