"""Tests for dolphin Excel loading."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from flipper.tools.excel import excel_to_notes, load_sheet, records_to_notes, rows_as_records


class ExcelLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "dolphin.xlsx"
        df = pd.DataFrame(
            [
                {
                    "Date": "2025-07-28",
                    "Dolphin": "KOD",
                    "Session ": 1,
                    "Trial": "KOD_20250728_S1_T1",
                    "BP": 1,
                    "Delta Time (s)": 0.5071,
                    "Center Freq (Hz)": 4875,
                }
            ]
        )
        df.to_excel(self.path, sheet_name="Audio Data", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_strips_column_whitespace(self):
        df = load_sheet(self.path)
        self.assertIn("Session", df.columns)
        self.assertNotIn("Session ", df.columns)

    def test_notes_include_trial_and_numbers(self):
        notes = excel_to_notes(self.path, limit=5)
        self.assertIn("KOD_20250728_S1_T1", notes)
        self.assertIn("0.5071", notes)
        self.assertIn("Do not invent numbers", notes)
        self.assertIn("Please summarize", notes)
        self.assertNotIn("Preferred measurements", notes)

    def test_limit(self):
        df = load_sheet(self.path)
        records = rows_as_records(df, limit=1)
        notes = records_to_notes(records, source="t.xlsx", sheet="Audio Data")
        self.assertIn("Rows: 1", notes)


if __name__ == "__main__":
    unittest.main()
