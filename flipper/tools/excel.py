"""Load dolphin Excel workbooks into notes for the LangGraph extract node.

Does not read WAV files. Numbers stay as they appear in the sheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SHEET = "Audio Data"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_sheet(path: Path, sheet: str = DEFAULT_SHEET) -> pd.DataFrame:
    """Return one sheet as a DataFrame with stripped column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path, sheet_name=sheet)
    df = _normalize_columns(df)
    df = df.dropna(how="all")
    return df


def list_sheets(path: Path) -> list[str]:
    path = Path(path)
    xl = pd.ExcelFile(path)
    return list(xl.sheet_names)


def rows_as_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit > 0:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in rec.items():
            if pd.isna(value):
                continue
            if hasattr(value, "isoformat"):
                clean[key] = str(value)[:10]
            else:
                clean[key] = value
        if clean:
            records.append(clean)
    return records


def records_to_notes(
    records: list[dict[str, Any]],
    *,
    source: str,
    sheet: str,
) -> str:
    """Render sheet rows as text for Grok to summarize. No invented numbers."""
    lines = [
        "Please summarize the following Excel data.",
        "Do not invent numbers. Use only values present below.",
        f"Source: {source}",
        f"Sheet: {sheet}",
        f"Rows: {len(records)}",
        "",
    ]
    if not records:
        lines.append("(no rows)")
        return "\n".join(lines)

    keys: list[str] = []
    for rec in records:
        for key in rec:
            if key not in keys:
                keys.append(key)
    lines.append("\t".join(keys))
    for rec in records:
        lines.append("\t".join(_cell(rec.get(k)) for k in keys))
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def excel_to_notes(
    path: Path,
    *,
    sheet: str = DEFAULT_SHEET,
    limit: int | None = 20,
) -> str:
    """Load a dolphin Excel sheet and return text for Grok to summarize."""
    df = load_sheet(path, sheet=sheet)
    records = rows_as_records(df, limit=limit)
    return records_to_notes(records, source=str(path), sheet=sheet)
