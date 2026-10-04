"""Build validated CSI 300 market history for the historical experiment.

Primary source: AkShare Eastmoney CSI 300 index history (sh000300).
Cross-check source: AkShare Sina CSI 300 index history (sh000300).

2022 is warm-up only. The research sample begins in 2023. Outputs are kept
separate from the repository's original short-sample market file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

SYMBOL = "sh000300"
DISPLAY_TICKER = "000300.SS"
WARMUP_START = dt.date(2022, 1, 1)
RESEARCH_START = dt.date(2023, 1, 1)
DEFAULT_END = dt.date(2026, 10, 4)


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_index_history(
    frame: pd.DataFrame,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Index data missing columns: {sorted(missing)}"
        )

    out = frame.copy()
    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    ).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    keep = [
        col
        for col in [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        if col in out.columns
    ]
    out = out[keep]
    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    out = out[
        (out["date"].dt.date >= start)
        & (out["date"].dt.date <= end)
    ].copy()
    # Mainland exchanges do not trade on weekends. Some provider
    # boundary queries can emit a synthetic carry-forward weekend row.
    out = out[out["date"].dt.dayofweek < 5].reset_index(drop=True)
    return out


def _retry_download(label: str, func, attempts: int = 3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"{label} failed after {attempts} attempts: {last_error}"
    )


def _download_csindex(
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    frame = _retry_download(
        "China Securities Index CSI 300",
        lambda: ak.stock_zh_index_hist_csindex(
            symbol="000300",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        ),
    )
    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交金额": "amount",
    }
    frame = frame.rename(columns=rename)
    return _normalize_index_history(frame, start, end)


def download_primary(
    start: dt.date,
    end: dt.date,
) -> tuple[pd.DataFrame, str]:
    try:
        return (
            _download_csindex(start, end),
            "AkShare stock_zh_index_hist_csindex / "
            "China Securities Index Company",
        )
    except Exception as csindex_error:
        frame = _retry_download(
            "Sina CSI 300 fallback",
            lambda: ak.stock_zh_index_daily(symbol=SYMBOL),
        )
        return (
            _normalize_index_history(frame, start, end),
            "AkShare stock_zh_index_daily / Sina fallback "
            f"(CSI error: {type(csindex_error).__name__})",
        )


def download_crosscheck(
    start: dt.date,
    end: dt.date,
    primary_source: str,
) -> tuple[pd.DataFrame | None, str | None]:
    candidates = []
    if "Sina" not in primary_source:
        candidates.append(
            (
                "AkShare stock_zh_index_daily / Sina",
                lambda: _normalize_index_history(
                    ak.stock_zh_index_daily(symbol=SYMBOL),
                    start,
                    end,
                ),
            )
        )
    if "China Securities Index" not in primary_source:
        candidates.append(
            (
                "AkShare stock_zh_index_hist_csindex / "
                "China Securities Index Company",
                lambda: _download_csindex(start, end),
            )
        )

    for label, func in candidates:
        try:
            frame = _retry_download(
                f"{label} cross-check",
                func,
                attempts=2,
            )
            return frame, label
        except Exception:
            continue
    return None, None


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy().sort_values("date").reset_index(drop=True)

    data["return"] = np.log(
        data["close"] / data["close"].shift(1)
    )
    data["volatility"] = (
        data["return"].rolling(20, min_periods=20).std(ddof=1)
    )

    ma5 = data["close"].rolling(5, min_periods=5).mean()
    ma20 = data["close"].rolling(20, min_periods=20).mean()
    data["momentum"] = (ma5 / ma20) - 1.0
    data["momentum_acceleration"] = data["momentum"].diff()

    prior_median_vol = (
        data["volatility"]
        .shift(1)
        .expanding(min_periods=20)
        .median()
    )
    data["regime_dummy"] = np.where(
        prior_median_vol.notna(),
        (data["volatility"] > prior_median_vol).astype(float),
        np.nan,
    )

    data["is_research_sample"] = (
        data["date"].dt.date >= RESEARCH_START
    ).astype(int)

    return data[
        [
            "date",
            "close",
            "return",
            "volatility",
            "momentum",
            "momentum_acceleration",
            "regime_dummy",
            "is_research_sample",
        ]
    ].copy()


def validate(
    raw: pd.DataFrame,
    crosscheck: pd.DataFrame | None,
    features: pd.DataFrame,
    primary_source: str,
    crosscheck_source: str | None,
) -> dict:
    if raw.empty or features.empty:
        raise ValueError("Market history is empty.")
    if len(raw) < 1000:
        raise ValueError(
            f"Market history is unexpectedly short: {len(raw)} rows"
        )
    if raw["date"].min().date() > dt.date(2022, 1, 10):
        raise ValueError(
            "Warm-up history does not genuinely cover early January 2022."
        )
    if raw["date"].max().date() < dt.date(2026, 9, 30):
        raise ValueError(
            "Market history does not reach the latest completed "
            "pre-holiday CSI 300 session."
        )
    if raw["date"].duplicated().any():
        raise ValueError("Duplicate market dates detected.")
    if not raw["date"].is_monotonic_increasing:
        raise ValueError("Market dates are not strictly sorted.")
    if (raw["close"] <= 0).any():
        raise ValueError("Non-positive CSI 300 close detected.")

    weekend_rows = int(
        (raw["date"].dt.dayofweek >= 5).sum()
    )
    if weekend_rows:
        raise ValueError(
            f"Unexpected weekend market rows: {weekend_rows}"
        )

    expected_return = np.log(
        raw["close"] / raw["close"].shift(1)
    )
    valid = expected_return.notna() & features["return"].notna()
    return_error = float(
        (
            expected_return.loc[valid]
            - features.loc[valid, "return"]
        ).abs().max()
    )
    if return_error > 1e-12:
        raise ValueError(
            f"Return recomputation mismatch: {return_error}"
        )

    expected_vol = expected_return.rolling(
        20,
        min_periods=20,
    ).std(ddof=1)
    valid_vol = (
        expected_vol.notna()
        & features["volatility"].notna()
    )
    vol_error = float(
        (
            expected_vol.loc[valid_vol]
            - features.loc[valid_vol, "volatility"]
        ).abs().max()
    )
    if vol_error > 1e-12:
        raise ValueError(
            f"Volatility recomputation mismatch: {vol_error}"
        )

    comparison = pd.DataFrame()
    max_abs_close_diff = None
    max_rel_close_diff = None
    median_abs_close_diff = None
    if crosscheck is not None:
        comparison = raw[
            ["date", "close"]
        ].merge(
            crosscheck[["date", "close"]],
            on="date",
            how="inner",
            suffixes=("_primary", "_crosscheck"),
        )
        if len(comparison) < 1000:
            raise ValueError(
                "Insufficient independent-source overlap for cross-check."
            )
        comparison["abs_close_diff"] = (
            comparison["close_primary"]
            - comparison["close_crosscheck"]
        ).abs()
        comparison["rel_close_diff"] = (
            comparison["abs_close_diff"]
            / comparison["close_primary"].abs()
        )
        max_abs_close_diff = float(
            comparison["abs_close_diff"].max()
        )
        max_rel_close_diff = float(
            comparison["rel_close_diff"].max()
        )
        median_abs_close_diff = float(
            comparison["abs_close_diff"].median()
        )
        if max_rel_close_diff > 1e-4:
            raise ValueError(
                "Independent CSI 300 close sources disagree materially: "
                f"max relative diff={max_rel_close_diff}"
            )

    research = features[
        features["is_research_sample"] == 1
    ].copy()
    warmup = features[
        features["is_research_sample"] == 0
    ].copy()
    complete_market_features = research.dropna(
        subset=[
            "return",
            "volatility",
            "momentum",
            "momentum_acceleration",
            "regime_dummy",
        ]
    )

    yearly_sessions = (
        raw.assign(year=raw["date"].dt.year)
        .groupby("year")
        .size()
        .astype(int)
        .to_dict()
    )

    return {
        "display_ticker": DISPLAY_TICKER,
        "source_symbol": SYMBOL,
        "primary_source": primary_source,
        "crosscheck_source": crosscheck_source,
        "warmup_start": WARMUP_START.isoformat(),
        "research_start": RESEARCH_START.isoformat(),
        "downloaded_first_session": raw["date"].min().strftime(
            "%Y-%m-%d"
        ),
        "downloaded_last_session": raw["date"].max().strftime(
            "%Y-%m-%d"
        ),
        "total_trading_sessions": int(len(raw)),
        "warmup_trading_sessions_2022": int(len(warmup)),
        "research_trading_sessions": int(len(research)),
        "research_rows_with_all_market_features": int(
            len(complete_market_features)
        ),
        "yearly_session_counts": {
            str(key): int(value)
            for key, value in yearly_sessions.items()
        },
        "weekend_market_rows": weekend_rows,
        "duplicate_market_dates": int(
            raw["date"].duplicated().sum()
        ),
        "crosscheck_overlapping_sessions": int(
            len(comparison)
        ),
        "crosscheck_max_abs_close_difference": max_abs_close_diff,
        "crosscheck_median_abs_close_difference": (
            median_abs_close_diff
        ),
        "crosscheck_max_relative_close_difference": (
            max_rel_close_diff
        ),
        "max_abs_return_recompute_error": return_error,
        "max_abs_volatility_recompute_error": vol_error,
        "feature_definitions": {
            "return": "close-to-close natural log return",
            "volatility": "20-session sample std of log returns",
            "momentum": (
                "5-session mean close / 20-session mean close - 1"
            ),
            "momentum_acceleration": (
                "first difference of momentum"
            ),
            "regime_dummy": (
                "1 when current 20-session volatility exceeds the "
                "expanding median of prior volatility observations"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=parse_date,
        default=WARMUP_START,
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=DEFAULT_END,
    )
    parser.add_argument(
        "--raw-output",
        default="data/csi300_history_2022_2026_raw.csv",
    )
    parser.add_argument(
        "--feature-output",
        default="data/csi300_market_features_2022_2026.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/csi300_market_2022_2026_summary.json",
    )
    args = parser.parse_args()

    raw, primary_source = download_primary(
        args.start,
        args.end,
    )
    crosscheck, crosscheck_source = download_crosscheck(
        args.start,
        args.end,
        primary_source,
    )
    features = build_features(raw)
    summary = validate(
        raw,
        crosscheck,
        features,
        primary_source,
        crosscheck_source,
    )

    raw_out = Path(args.raw_output)
    feature_out = Path(args.feature_output)
    summary_out = Path(args.summary_output)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    feature_out.parent.mkdir(parents=True, exist_ok=True)

    raw_save = raw.copy()
    raw_save["date"] = raw_save["date"].dt.strftime("%Y-%m-%d")
    feature_save = features.copy()
    feature_save["date"] = feature_save["date"].dt.strftime(
        "%Y-%m-%d"
    )

    raw_save.to_csv(raw_out, index=False)
    feature_save.to_csv(feature_out, index=False)
    summary_out.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Raw market history: {raw_out}")
    print(f"Market features: {feature_out}")
    print(f"Validation summary: {summary_out}")


if __name__ == "__main__":
    main()
