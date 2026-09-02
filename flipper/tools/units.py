"""Deterministic unit and schema QC. No model in this file."""

from typing import Any
# Allowed units for audio processing and analysis
ALLOWED_UNITS = {
    "khz", "hz", "db", "db re 1 µpa", "db/hz", "s", "ms", "bits",
}


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    ok, rejected = [], []
    for i, row in enumerate(rows):
        unit = str(row.get("unit") or "").strip().lower()
        value = row.get("value")
        wav_id = row.get("wav_id")
        if not wav_id:
            rejected.append(f"row {i}: missing wav_id")
            continue
        if value is not None and unit and unit not in ALLOWED_UNITS:
            rejected.append(f"row {i} ({wav_id}): unknown unit {unit!r}")
            continue
        ok.append(row)
    return ok, rejected
