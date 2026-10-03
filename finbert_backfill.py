"""Resumable FinBERT scoring for historical BigQuery headline data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"


def parse_date(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def split_headlines(
    raw_text: str,
    headlines_json: str | None = None,
) -> list[str]:
    """Parse headlines losslessly, preferring JSON over the legacy delimiter."""
    if headlines_json is not None and not pd.isna(headlines_json):
        value = str(headlines_json).strip()
        if value:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("headlines_json must decode to a list")
            return [
                str(item).strip()
                for item in parsed
                if len(str(item).strip()) > 15
            ]

    return [
        item.strip()
        for item in str(raw_text).split(" || ")
        if len(item.strip()) > 15
    ]


def score_headlines(
    headlines: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    id2label = {
        int(key): str(value).lower()
        for key, value in model.config.id2label.items()
    }

    for start in range(0, len(headlines), batch_size):
        batch = headlines[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        for offset, (headline, values) in enumerate(
            zip(batch, probabilities)
        ):
            probability = {
                id2label[index]: float(value)
                for index, value in enumerate(values)
            }
            positive = probability.get("positive", 0.0)
            negative = probability.get("negative", 0.0)
            neutral = probability.get("neutral", 0.0)
            label = max(
                ("positive", positive),
                ("negative", negative),
                ("neutral", neutral),
                key=lambda item: item[1],
            )[0]
            rows.append(
                {
                    "headline_index": start + offset,
                    "title": headline,
                    "positive_probability": positive,
                    "negative_probability": negative,
                    "neutral_probability": neutral,
                    "predicted_label": label,
                    "sentiment_score": positive - negative,
                }
            )

    return pd.DataFrame(rows)


def score_day(
    row: pd.Series,
    output_dir: Path,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    force: bool,
) -> dict:
    date_value = str(row["date"])
    label = date_value.replace("-", "")
    score_path = output_dir / "headline_scores" / f"scores_{label}.csv.gz"
    daily_path = output_dir / "daily" / f"sentiment_{label}.csv"
    manifest_path = output_dir / "manifests" / f"manifest_{label}.json"

    expected_hash = str(row["headline_hash"])
    if (
        not force
        and score_path.exists()
        and daily_path.exists()
        and manifest_path.exists()
    ):
        existing = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            existing.get("headline_hash") == expected_hash
            and existing.get("model_name") == MODEL_NAME
            and int(existing.get("max_length", -1)) == max_length
        ):
            return {
                "date": date_value,
                "status": "skipped",
                "manifest": str(manifest_path),
            }

    headlines = split_headlines(
        row["raw_text"],
        row.get("headlines_json"),
    )
    declared_count = int(row["article_count"])
    if len(headlines) != declared_count:
        raise ValueError(
            f"{date_value}: article_count={declared_count}, "
            f"but parsed {len(headlines)} headlines"
        )

    started = time.perf_counter()
    scored = score_headlines(
        headlines,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
    )
    elapsed = time.perf_counter() - started

    if len(scored) != declared_count:
        raise ValueError(
            f"{date_value}: scored {len(scored)} of "
            f"{declared_count} headlines"
        )

    scored.insert(0, "date", date_value)
    daily = pd.DataFrame(
        [
            {
                "date": date_value,
                "sentiment_mean": float(
                    scored["sentiment_score"].mean()
                ),
                "sentiment_std": float(
                    scored["sentiment_score"].std(ddof=1)
                    if len(scored) > 1
                    else 0.0
                ),
                "article_count": int(len(scored)),
                "positive_share": float(
                    (scored["predicted_label"] == "positive").mean()
                ),
                "negative_share": float(
                    (scored["predicted_label"] == "negative").mean()
                ),
                "neutral_share": float(
                    (scored["predicted_label"] == "neutral").mean()
                ),
                "headline_hash": expected_hash,
            }
        ]
    )

    for directory in (
        score_path.parent,
        daily_path.parent,
        manifest_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    scored.to_csv(score_path, index=False, compression="gzip")
    daily.to_csv(daily_path, index=False)

    manifest = {
        "date": date_value,
        "model_name": MODEL_NAME,
        "device": str(device),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "headline_hash": expected_hash,
        "headlines_scored": int(len(scored)),
        "elapsed_seconds": round(float(elapsed), 3),
        "headlines_per_second": round(
            float(len(scored) / elapsed) if elapsed > 0 else 0.0,
            3,
        ),
        "score_file": str(score_path),
        "daily_file": str(daily_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return {"status": "complete", **manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/news_daily_bigquery_2023_2025.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/finbert_backfill_2023_2025",
    )
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError("--end must be on or after --start")

    news = pd.read_csv(args.input)
    news["date"] = news["date"].astype(str)
    selected = news[
        (news["date"] >= args.start.isoformat())
        & (news["date"] <= args.end.isoformat())
    ].sort_values("date").reset_index(drop=True)
    if selected.empty:
        raise ValueError("No candidate news rows in requested period.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"Loading {MODEL_NAME} on {device}; "
        f"days={len(selected)}, batch_size={args.batch_size}"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    except OSError:
        print(
            "Cached FinBERT files unavailable; falling back to "
            "Hugging Face download.",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    results: list[dict] = []
    for index, row in selected.iterrows():
        result = score_day(
            row=row,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            force=args.force,
        )
        results.append(result)
        print(
            f"[{index + 1}/{len(selected)}] "
            f"{row['date']} {result['status']} "
            f"headlines={result.get('headlines_scored', '-')}, "
            f"seconds={result.get('elapsed_seconds', '-')}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_progress.jsonl").open(
        "a",
        encoding="utf-8",
    ) as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
