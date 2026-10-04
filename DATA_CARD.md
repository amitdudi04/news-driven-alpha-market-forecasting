# Data Card

## Research scope

The project studies the CSI 300 and China-focused financial/economic news.

### Historical periods

- **Market warm-up:** 4 Jan 2022 onward
- **Research/news period:** 1 Jan 2023 – 3 Oct 2026
- **Last completed CSI 300 market close in the study:** 30 Sep 2026

2022 is used only to initialize rolling market features and GARCH history.

## News source

Historical headlines are retrieved from the **GDELT 2.0 Global Article List (GAL)** in BigQuery.

The query:

- restricts language to English;
- requires a China identifier in the title (for example China, Chinese, PBOC, Beijing);
- requires at least one macro/finance term;
- retains title/headline text rather than full article bodies.

The query therefore samples **English-language international coverage of China-related finance/economics**. It is not a complete archive of all information available to Chinese-market investors.

## Timestamp semantics

The GDELT timestamp is treated as a **GDELT seen timestamp**.

It is not claimed to be the publisher's exact original publication time.

Forecast cutoff:

**15:00 Asia/Shanghai on genuine CSI 300 trading days**

Headline assignment rule:

~~~text
previous CSI 300 trading close < GDELT seen timestamp <= current CSI 300 trading close
~~~

After-close, weekend, and holiday news is moved forward to the first eligible market close.

Final audit:

- headline observations entering alignment: **292,373**
- assigned to completed CSI 300 windows: **291,973**
- pending after the last known market close: **400**
- causal-window timing violations: **0**

## Sentiment

Headlines are scored with pretrained **ProsusAI/FinBERT**.

FinBERT is not fine-tuned for China-specific news in this project.

Before session-level pooling, normalized titles are de-duplicated within each assigned trading-session information window.

- assigned observations: **291,973**
- session-unique normalized headlines: **276,960**
- repeated observations removed: **15,013**

## Market data

The CSI 300 historical series is built from the China Securities Index feed exposed through AkShare and independently checked against Sina history.

- total market sessions: **1,150**
- 2022 warm-up sessions: **242**
- research sessions from 2023: **908**
- independent-source overlapping sessions: **1,150**
- maximum close difference: **0.005 index points**

The project does not use Yahoo Finance as the authoritative historical source because the tested Yahoo endpoint returned an incomplete CSI 300 history during reconstruction.

## Master session dataset

The master dataset contains **908 genuine CSI 300 sessions**.

Features include:

- current log return;
- 20-session realized volatility;
- 5/20 momentum;
- momentum acceleration;
- causal volatility-regime indicator;
- unique headline count;
- pooled FinBERT sentiment mean and dispersion;
- positive/negative/neutral FinBERT shares;
- strict 5/10/20-session rolling sentiment;
- current news intensity relative to the prior 20 sessions;
- sentiment-volatility interactions.

Missing sentiment remains missing. The one genuine no-news research session is **20 Jun 2025**.

Targets:

- next genuine CSI 300 trading-session log return;
- next-session direction = 1 if that return is positive, otherwise 0.

There are **907** known next-session targets and **867** rows satisfying the strict full-feature modeling rule.

## Version-control policy

Large generated historical datasets and model outputs are intentionally excluded from Git.

Source code, frozen specifications, documentation, tests, and small configuration files are committed. This prevents large/stale generated artifacts from being mistaken for source-of-truth code.

## Limitations

- English-only news selection creates coverage bias.
- GDELT seen time is not exact publisher publication time.
- Title filtering may miss relevant stories or include imperfectly relevant ones.
- Exact normalized-title de-duplication does not eliminate all semantic syndication.
- FinBERT is not China-specific.
- The last completed market close precedes some late-Sep/Oct 2026 news, so 400 observations remain correctly unassigned.
