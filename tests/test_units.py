"""Tests for deterministic row validation."""

import unittest

from flipper.tools.units import validate_rows


class ValidateRowsTests(unittest.TestCase):
    def test_accepts_known_unit(self):
        ok, rejected = validate_rows(
            [{"wav_id": "a.wav", "measurement": "peak", "value": 12.4, "unit": "kHz"}]
        )
        self.assertEqual(len(ok), 1)
        self.assertEqual(rejected, [])

    def test_rejects_missing_wav_id(self):
        ok, rejected = validate_rows([{"measurement": "peak", "value": 1, "unit": "hz"}])
        self.assertEqual(ok, [])
        self.assertTrue(rejected[0].startswith("row 0"))

    def test_rejects_unknown_unit(self):
        ok, rejected = validate_rows(
            [{"wav_id": "a.wav", "value": 1, "unit": "furlongs"}]
        )
        self.assertEqual(ok, [])
        self.assertIn("unknown unit", rejected[0])

    def test_null_value_skips_unit_check(self):
        ok, rejected = validate_rows(
            [{"wav_id": "a.wav", "value": None, "unit": "not-a-unit", "notes": "contour"}]
        )
        self.assertEqual(len(ok), 1)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
