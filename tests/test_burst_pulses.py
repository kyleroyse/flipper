"""Synthetic burst-pulse extraction tests. Outputs must be named *_Flipper*."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from flipper.extract_burst_pulses import (
    ensure_flipper_filename,
    match_wav,
    parse_wav_dolphin_date,
    raven_trial_label,
    run_pipeline,
)
from flipper.services.burst_pulses import (
    detect_in_array,
    group_pulse_trains,
    merge_overlapping_detections,
    parse_clock_to_seconds,
    BurstTrain,
)


def _synthetic_train(
    fs: int = 192000,
    n_pulses: int = 200,
    ici_s: float = 0.0022,
    t0: float = 1.0,
    duration: float = 3.0,
) -> np.ndarray:
    n = int(duration * fs)
    x = np.zeros(n, dtype=np.float64)
    click_n = int(0.00015 * fs)
    t = np.arange(click_n) / fs
    click = np.sin(2 * np.pi * 25000 * t) * np.hanning(click_n)
    for i in range(n_pulses):
        start = int((t0 + i * ici_s) * fs)
        end = start + click_n
        if end < n:
            x[start:end] += click
    return x


class ParseClockTests(unittest.TestCase):
    def test_hms_string(self):
        self.assertAlmostEqual(parse_clock_to_seconds("0:01:07"), 67.0)
        self.assertAlmostEqual(parse_clock_to_seconds("0:01:39.088"), 99.088, places=3)

    def test_timedelta(self):
        self.assertAlmostEqual(parse_clock_to_seconds(pd.Timedelta("0 days 00:01:07")), 67.0)

    def test_time_obj(self):
        import datetime as dt

        self.assertAlmostEqual(parse_clock_to_seconds(dt.time(0, 1, 7)), 67.0)

    def test_numeric(self):
        self.assertEqual(parse_clock_to_seconds(67.47), 67.47)

    def test_nan(self):
        self.assertIsNone(parse_clock_to_seconds(np.nan))
        self.assertIsNone(parse_clock_to_seconds(None))


class GroupTrainTests(unittest.TestCase):
    def test_groups_on_max_ici(self):
        # 10 peaks at 2.2 ms, then a gap, then 10 more
        a = np.arange(10) * 0.0022
        b = 0.05 + np.arange(10) * 0.0022
        trains = group_pulse_trains(np.concatenate([a, b]), min_pulses=8)
        self.assertEqual(len(trains), 2)
        self.assertAlmostEqual(trains[0].mean_ici_ms, 2.2, places=5)


class MergeTests(unittest.TestCase):
    def test_merge_within_20ms(self):
        a = BurstTrain(1.0, 1.4, 10, 2.2, 454.5, 1, event="touch")
        b = BurstTrain(1.005, 1.402, 10, 2.2, 454.5, 1, event="bridge")
        merged = merge_overlapping_detections([a, b])
        self.assertEqual(len(merged), 1)


class WavNameTests(unittest.TestCase):
    def test_obj_name(self):
        p = Path("JULY28/KOD/OBJ_25-07-28_KOD_TC_yse-bsw.wav")
        keys = parse_wav_dolphin_date(p)
        self.assertIn(("KOD", "2025-07-28"), keys)

    def test_match_single(self):
        wavs = [Path("/tmp/OBJ_25-07-28_KOD_TC_yse-bsw.wav")]
        path, reason = match_wav(wavs, dolphin="KOD", date_iso="2025-07-28", session=1)
        self.assertEqual(path, wavs[0])
        self.assertIsNone(reason)

    def test_flipper_name(self):
        p = ensure_flipper_filename(Path("Audio_Data.csv"))
        self.assertEqual(p.name, "Audio_Data_Flipper.csv")
        p2 = ensure_flipper_filename(Path("KOD_20250728_S1_Flipper.selections.txt"))
        self.assertEqual(p2.name, "KOD_20250728_S1_Flipper.selections.txt")


class RavenLabelTests(unittest.TestCase):
    def test_suffix(self):
        self.assertEqual(raven_trial_label(6, 0), "6")
        self.assertEqual(raven_trial_label(6, 1), "6.2")
        self.assertEqual(raven_trial_label(6, 2), "6.3")


class SyntheticPipelineTests(unittest.TestCase):
    def test_detect_ici_and_flipper_outputs(self):
        fs = 192000
        ici = 0.0022
        x = _synthetic_train(fs=fs, n_pulses=200, ici_s=ici, t0=1.0, duration=3.0)
        stereo = np.column_stack([x * 0.05, x])
        trains, ch = detect_in_array(stereo, fs, t0=0.0)
        self.assertEqual(ch, 2)
        self.assertGreaterEqual(len(trains), 1)
        best = max(trains, key=lambda t: t.pulses)
        self.assertGreaterEqual(best.pulses, 150)
        self.assertAlmostEqual(best.mean_ici_ms, 2.2, delta=0.3)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        audio_dir = root / "audio"
        audio_dir.mkdir()
        wav_path = audio_dir / "OBJ_25-07-28_KOD_TC_yse-bsw.wav"
        pcm = np.clip(stereo / (np.max(np.abs(stereo)) + 1e-12) * 20000, -32767, 32767)
        wavfile.write(str(wav_path), fs, pcm.astype(np.int16))

        excel = root / "fake.xlsx"
        df = pd.DataFrame(
            [
                {
                    "Date": "2025-07-28",
                    "Dolphin": "KOD",
                    "Session ": 1,
                    "Trial": "KOD_20250728_S1_T1",
                    "Touch Obj Time": "0:00:01.200",
                    "Touch Obj Time (s)": 1.2,
                    "Bridge": pd.NaT,
                    "Bridge (s)": np.nan,
                },
                {
                    "Date": "2025-07-28",
                    "Dolphin": "KOD",
                    "Session ": 1,
                    "Trial": "CLANG",
                    "Touch Obj Time": "0:00:01.200",
                    "Touch Obj Time (s)": 1.2,
                },
            ]
        )
        df.to_excel(excel, sheet_name="Data", index=False)

        out = root / "out"
        result = run_pipeline(
            excel=excel,
            audio_dir=audio_dir,
            out=out,
            window=3.0,
            dolphin="KOD",
            date="2025-07-28",
        )
        self.assertGreaterEqual(result["n_bp"], 1)
        csv_path = Path(result["csv"])
        self.assertTrue(csv_path.exists())
        self.assertIn("Flipper", csv_path.name)
        self.assertTrue(csv_path.name.endswith("Flipper.csv") or csv_path.stem.endswith("Flipper"))
        raven = list(out.glob("*Flipper.selections.txt"))
        self.assertTrue(raven, "expected a Raven table named *Flipper.selections.txt")
        self.assertTrue(all("Flipper" in p.name for p in raven))
        text = raven[0].read_text(encoding="utf-8")
        self.assertIn("Selection\tView\tChannel", text)
        self.assertIn("Waveform 1", text)
        csv = pd.read_csv(csv_path)
        self.assertIn("Event", csv.columns)
        self.assertIn("Wav", csv.columns)
        self.assertIn("WindowStart", csv.columns)
        self.assertIn("WindowEnd", csv.columns)
        self.assertIn("SNR (dB)", csv.columns)
        self.assertGreaterEqual(float(csv["SNR (dB)"].iloc[0]), 10.0)
        self.assertAlmostEqual(float(csv["mean_ici_ms"].iloc[0]), 2.2, delta=0.3)
        self.assertGreaterEqual(int(csv["Pulses"].iloc[0]), 150)



class SnrGateTests(unittest.TestCase):
    def test_snr_db_loud_vs_quiet(self):
        from flipper.services.burst_pulses import snr_db

        sig = np.ones(1000) * 0.5
        noise = np.ones(1000) * 0.01
        self.assertGreater(snr_db(sig, noise), 10.0)

    def test_low_snr_rejected(self):
        fs = 192000
        rng = np.random.default_rng(0)
        duration = 3.0
        noise = rng.normal(0, 0.4, int(duration * fs))
        click_n = int(0.00015 * fs)
        t = np.arange(click_n) / fs
        click = np.sin(2 * np.pi * 25000 * t) * np.hanning(click_n) * 0.02
        x = noise.copy()
        ici_s = 0.0022
        for i in range(200):
            start = int((1.0 + i * ici_s) * fs)
            end = start + click_n
            if end < x.size:
                x[start:end] += click
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        audio_dir = root / "audio"
        audio_dir.mkdir()
        wav_path = audio_dir / "OBJ_25-07-28_KOD_TC_yse-bsw.wav"
        pcm = np.clip(x / (np.max(np.abs(x)) + 1e-12) * 20000, -32767, 32767)
        wavfile.write(str(wav_path), fs, pcm.astype(np.int16))
        excel = root / "fake.xlsx"
        df = pd.DataFrame(
            [
                {
                    "Date": "2025-07-28",
                    "Dolphin": "KOD",
                    "Session ": 1,
                    "Trial": "KOD_20250728_S1_T1",
                    "Touch Obj Time": "0:00:01.200",
                    "Touch Obj Time (s)": 1.2,
                }
            ]
        )
        df.to_excel(excel, sheet_name="Data", index=False)
        out = root / "out"
        result = run_pipeline(
            excel=excel,
            audio_dir=audio_dir,
            out=out,
            window=3.0,
            dolphin="KOD",
            date="2025-07-28",
            min_snr=10.0,
        )
        rejected = Path(result["rejected"])
        self.assertTrue(rejected.exists())
        self.assertIn("Flipper", rejected.name)
        # Either no detections, or any detections that survived must have SNR >= 10
        csv = pd.read_csv(Path(result["csv"])) if Path(result["csv"]).stat().st_size else pd.DataFrame()
        if len(csv):
            self.assertTrue((csv["SNR (dB)"] >= 10.0).all())
        rej = pd.read_csv(rejected)
        if len(rej):
            self.assertTrue((rej["reason"] == "low_snr").all())

if __name__ == "__main__":
    unittest.main()
