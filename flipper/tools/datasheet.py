"""The only module allowed to persist official measurement rows."""

import csv
from pathlib import Path
from typing import Any

from flipper.config import settings

FIELDS = ["wav_id", "measurement", "value", "unit", "notes", "model_used"]


def write_rows(rows: list[dict[str, Any]], thread_id: str, model_used: str) -> Path:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / f"{thread_id}.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        for row in rows:
            payload = {k: row.get(k, "") for k in FIELDS}
            payload["model_used"] = model_used
            writer.writerow(payload)
    return path
