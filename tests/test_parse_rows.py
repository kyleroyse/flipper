"""Tests for extract JSON parsing."""

import unittest

from flipper.graphs.analysis_session import parse_rows


class ParseRowsTests(unittest.TestCase):
    def test_object_with_rows(self):
        text = 'prefix {"rows": [{"wav_id": "a.wav", "value": 1, "unit": "kHz"}]} suffix'
        rows = parse_rows(text)
        self.assertEqual(rows[0]["wav_id"], "a.wav")

    def test_empty_when_no_json(self):
        self.assertEqual(parse_rows("no structured data"), [])


if __name__ == "__main__":
    unittest.main()
