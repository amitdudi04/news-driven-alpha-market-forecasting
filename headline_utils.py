"""Lightweight lossless headline serialization helpers."""

from __future__ import annotations

import json


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def split_headlines(
    raw_text: str,
    headlines_json: str | None = None,
) -> list[str]:
    """Parse headlines losslessly, preferring JSON over the legacy delimiter."""
    if not _is_missing(headlines_json):
        value = str(headlines_json).strip()
        if value:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(
                    "headlines_json must decode to a list"
                )
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
