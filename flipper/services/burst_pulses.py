"""Burst-pulse ICI detector and Raven-like spectral metrics.

Original count_burst_pulses.py used duration/(n-1) as ICI, which is wrong
inside a fixed video window. Here successive peaks are grouped into trains
where ICI <= max_ici_ms.
"""

from __future__ import annotations

import datetime as dt
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

DEFAULT_MIN_ICI_MS = 1.2
DEFAULT_MAX_ICI_MS = 10.0
DEFAULT_MIN_PULSES = 25
DEFAULT_PROMINENCE_FACTOR = 1.25
DEFAULT_MAX_TRAIN_S = 2.5
SAVGOL_WINDOW = 11
SAVGOL_POLY = 2
LOWCUT_HZ = 500.0
MERGE_TOL_S = 0.020
DEFAULT_MIN_SNR_DB = 3.0
SNR_EPS = 1e-20
RAVEN_NPERSEG = 512
RAVEN_NOVERLAP = 256  # 50%
RAVEN_WINDOW = "hann"
MIN_TRAIN_PEAK_TO_MEDIAN = 8.0
MAX_MEAN_ICI_FOR_BP_MS = 5.0  # packed BP vs spaced echolocation clicks
WHISTLE_GAP_MAX_S = 0.080


def parse_clock_to_seconds(value: Any) -> float | None:
    """Parse Excel clock / timedelta / time / numeric seconds to float seconds."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (np.floating, float, np.integer, int)):
        v = float(value)
        return v if np.isfinite(v) else None

    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, np.timedelta64):
        return float(pd.Timedelta(value).total_seconds())
    if isinstance(value, dt.timedelta):
        return float(value.total_seconds())

    if isinstance(value, dt.datetime):
        value = value.time()
    if isinstance(value, dt.time):
        return (
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1e6
        )

    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "nat", "none"}:
            return None
        if s.lower().startswith("0 days"):
            try:
                return float(pd.Timedelta(s).total_seconds())
            except (ValueError, TypeError):
                pass
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, sec = parts
                return int(h) * 3600 + int(m) * 60 + float(sec)
            if len(parts) == 2:
                m, sec = parts
                return int(m) * 60 + float(sec)
            return float(s)
        except ValueError:
            return None
    return None


def prefer_seconds(row: dict[str, Any], raw_key: str, sec_key: str) -> float | None:
    """Use a filled `(s)` column if present, otherwise parse the clock string."""
    sec = parse_clock_to_seconds(_row_get(row, sec_key))
    if sec is not None:
        return sec
    return parse_clock_to_seconds(_row_get(row, raw_key))


def _row_get(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    stripped = {str(k).strip(): v for k, v in row.items()}
    return stripped.get(key.strip())


def wav_info(path: Path) -> tuple[int, int, int, int]:
    """Return nchannels, sampwidth, framerate, nframes (header only)."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def wav_duration_s(path: Path) -> float:
    ch, sw, fs, n = wav_info(path)
    if fs <= 0:
        return 0.0
    return n / float(fs)


def _pcm24_to_int32(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8)
    n = b.size // 3
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    b = b[: n * 3]
    out = (
        b[0::3].astype(np.int32)
        | (b[1::3].astype(np.int32) << 8)
        | (b[2::3].astype(np.int32) << 16)
    )
    out = np.where(out >= 0x800000, out - 0x1000000, out)
    return out.astype(np.int32)


def read_wav_window(path: Path, start_s: float, end_s: float) -> tuple[int, np.ndarray]:
    """Read one time window as float64 array shape (n,) or (n, ch). Does not load the file."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fs = wf.getframerate()
        nframes = wf.getnframes()
        start = max(0, int(round(start_s * fs)))
        end = min(nframes, int(round(end_s * fs)))
        if end <= start:
            empty = np.zeros((0, nch) if nch > 1 else (0,), dtype=np.float64)
            return fs, empty
        wf.setpos(start)
        raw = wf.readframes(end - start)
    n_samp_expected = (end - start) * nch
    if sw == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    elif sw == 3:
        data = _pcm24_to_int32(raw).astype(np.float64)
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
    else:
        raise ValueError(f"Unsupported sampwidth {sw} for {path}")
    if nch > 1:
        usable = (data.size // nch) * nch
        data = data[:usable].reshape(-1, nch)
    return fs, data


def _bandpass(x: np.ndarray, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    highcut = min(96000.0, 0.9 * nyq)
    if highcut <= LOWCUT_HZ or nyq <= LOWCUT_HZ:
        return x
    b, a = signal.butter(4, [LOWCUT_HZ / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def _envelope(x: np.ndarray, fs: int) -> np.ndarray:
    if x.size < SAVGOL_WINDOW:
        return np.asarray(x, dtype=np.float64)
    filtered = _bandpass(np.asarray(x, dtype=np.float64), fs)
    analytic = signal.hilbert(filtered)
    env = np.abs(analytic)
    return signal.savgol_filter(env, SAVGOL_WINDOW, SAVGOL_POLY)


def envelope_and_peaks(
    x: np.ndarray,
    fs: int,
    *,
    prominence_factor: float = DEFAULT_PROMINENCE_FACTOR,
    min_ici_ms: float = DEFAULT_MIN_ICI_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (envelope, peak_indices) for a mono float channel."""
    env = _envelope(x, fs)
    if env.size < SAVGOL_WINDOW:
        return env, np.zeros(0, dtype=int)
    min_distance = max(1, int(min_ici_ms / 1000.0 * fs))
    threshold = float(np.median(env) * prominence_factor)
    peaks, _ = signal.find_peaks(
        env,
        distance=min_distance,
        height=threshold,
        prominence=threshold * 0.4,
    )
    return env, peaks


def _peaks_at_threshold(
    env: np.ndarray,
    fs: int,
    threshold: float,
    min_ici_ms: float,
) -> np.ndarray:
    min_distance = max(1, int(min_ici_ms / 1000.0 * fs))
    peaks, _ = signal.find_peaks(
        env,
        distance=min_distance,
        height=threshold,
        prominence=threshold * 0.4,
    )
    return peaks


def trains_from_envelope(
    env: np.ndarray,
    fs: int,
    *,
    t0: float = 0.0,
    prominence_factor: float = DEFAULT_PROMINENCE_FACTOR,
    min_ici_ms: float = DEFAULT_MIN_ICI_MS,
    max_ici_ms: float = DEFAULT_MAX_ICI_MS,
    min_pulses: int = DEFAULT_MIN_PULSES,
    max_train_s: float = DEFAULT_MAX_TRAIN_S,
) -> list[BurstTrain]:
    """Group peaks into trains. If a train fills a long video window (noise
    peaks every ~min ICI), raise the height gate until trains are <= max_train_s.

    Short clips (her original use-case) keep median * prominence_factor.
    """
    if env.size < SAVGOL_WINDOW:
        return []
    med = float(np.median(env))
    thr = med * float(prominence_factor)
    if not np.isfinite(thr) or thr <= 0:
        return []

    def _group(threshold: float) -> list[BurstTrain]:
        peaks = _peaks_at_threshold(env, fs, threshold, min_ici_ms)
        times = t0 + peaks.astype(np.float64) / float(fs)
        return group_pulse_trains(
            times,
            min_ici_ms=min_ici_ms,
            max_ici_ms=max_ici_ms,
            min_pulses=min_pulses,
        )

    trains = _group(thr)
    longest = max((t.end_s - t.begin_s for t in trains), default=0.0)
    # Clip-based files (short, loud) keep median*1.25. A 6 s video window
    # of water noise makes that gate stitch peaks into one window-long train;
    # raise the floor to the 99th percentile of the envelope in that case.
    if longest > max_train_s:
        p99 = float(np.percentile(env, 99))
        trains = _group(max(thr, p99))
    kept = [t for t in trains if (t.end_s - t.begin_s) <= max_train_s]
    if not kept:
        return []
    best = max(t.pulses for t in kept)
    return [t for t in kept if t.pulses >= max(min_pulses, int(0.15 * best))]


@dataclass
class BurstTrain:
    begin_s: float
    end_s: float
    pulses: int
    mean_ici_ms: float
    prr: float
    channel: int  # 1-indexed
    event: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
    peak_times_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    loudness: float = 0.0


def group_pulse_trains(
    peak_times_s: np.ndarray,
    *,
    min_ici_ms: float = DEFAULT_MIN_ICI_MS,
    max_ici_ms: float = DEFAULT_MAX_ICI_MS,
    min_pulses: int = DEFAULT_MIN_PULSES,
) -> list[BurstTrain]:
    """Group successive peaks with ICI <= max_ici_ms; drop trains failing the ICI gate."""
    times = np.asarray(peak_times_s, dtype=np.float64)
    if times.size < min_pulses:
        return []
    trains: list[BurstTrain] = []
    start = 0
    for i in range(1, times.size + 1):
        split = i == times.size
        if not split:
            ici_ms = (times[i] - times[i - 1]) * 1000.0
            split = ici_ms > max_ici_ms
        if not split:
            continue
        chunk = times[start:i]
        start = i
        if chunk.size < min_pulses:
            continue
        icis = np.diff(chunk) * 1000.0
        if icis.size == 0:
            continue
        mean_ici = float(np.mean(icis))
        if mean_ici < min_ici_ms or mean_ici > max_ici_ms:
            continue
        prr = 1000.0 / mean_ici if mean_ici > 0 else 0.0
        trains.append(
            BurstTrain(
                begin_s=float(chunk[0]),
                end_s=float(chunk[-1]),
                pulses=int(chunk.size),
                mean_ici_ms=mean_ici,
                prr=prr,
                channel=1,
                peak_times_s=chunk.copy(),
            )
        )
    return trains


def _channel_score(trains: list[BurstTrain], env: np.ndarray) -> tuple[int, float]:
    n_pulses = sum(t.pulses for t in trains)
    energy = float(np.mean(env ** 2)) if env.size else 0.0
    return n_pulses, energy


def detect_in_array(
    data: np.ndarray,
    fs: int,
    *,
    t0: float = 0.0,
    prominence_factor: float = DEFAULT_PROMINENCE_FACTOR,
    min_ici_ms: float = DEFAULT_MIN_ICI_MS,
    max_ici_ms: float = DEFAULT_MAX_ICI_MS,
    min_pulses: int = DEFAULT_MIN_PULSES,
    max_train_s: float = DEFAULT_MAX_TRAIN_S,
) -> tuple[list[BurstTrain], int]:
    """Detect on every channel, then keep the louder hydrophone per burst.

    Burst pulses are directional, so the dolphin faces one phone or the other.
    Channel is 1-indexed. `data` is (n,) or (n, ch).
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        channels = [arr]
    else:
        channels = [arr[:, c] for c in range(arr.shape[1])]

    all_trains: list[BurstTrain] = []
    for idx, ch in enumerate(channels):
        env = _envelope(ch, fs)
        med = float(np.median(env)) if env.size else 0.0
        trains = trains_from_envelope(
            env,
            fs,
            t0=t0,
            prominence_factor=prominence_factor,
            min_ici_ms=min_ici_ms,
            max_ici_ms=max_ici_ms,
            min_pulses=min_pulses,
            max_train_s=max_train_s,
        )
        for tr in trains:
            tr.channel = idx + 1
            i0 = max(0, int(round((tr.begin_s - t0) * fs)))
            i1 = min(env.size, int(round((tr.end_s - t0) * fs)))
            slice_env = env[i0:i1] if i1 > i0 else env
            peak = float(np.max(slice_env)) if slice_env.size else 0.0
            tr.loudness = rms(slice_env) if slice_env.size else 0.0
            if med > 0 and peak < MIN_TRAIN_PEAK_TO_MEDIAN * med:
                continue  # background click soup, not a directional BP
            all_trains.append(tr)

    merged = merge_overlapping_detections(all_trains)
    best_ch = merged[0].channel if merged else 1
    return merged, best_ch


def _interval_overlap(a: BurstTrain, b: BurstTrain) -> float:
    return max(0.0, min(a.end_s, b.end_s) - max(a.begin_s, b.begin_s))


def merge_overlapping_detections(
    trains: list[BurstTrain], *, tol_s: float = MERGE_TOL_S
) -> list[BurstTrain]:
    """Keep one box when two detections are the same burst (any real overlap).

    Prefer the louder channel, then more pulses, then longer duration.
    """
    ordered = sorted(
        trains,
        key=lambda t: (t.loudness, t.pulses, t.end_s - t.begin_s),
        reverse=True,
    )
    kept: list[BurstTrain] = []
    for t in ordered:
        dup = False
        for i, k in enumerate(kept):
            ov = _interval_overlap(k, t)
            shorter = min(k.end_s - k.begin_s, t.end_s - t.begin_s)
            close = abs(k.begin_s - t.begin_s) <= max(tol_s, 0.05)
            if (shorter > 0 and ov / shorter >= 0.4) or (close and ov > 0):
                dup = True
                break
        if not dup:
            kept.append(t)
    kept.sort(key=lambda t: t.begin_s)
    return kept


def _mean_psd(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(RAVEN_NPERSEG, max(32, x.size // 2 * 2))
    if nperseg < 32 or x.size < nperseg:
        nperseg = min(256, x.size) if x.size >= 16 else x.size
    if nperseg < 16:
        freqs = np.fft.rfftfreq(x.size, d=1.0 / fs) if x.size else np.array([0.0])
        spec = np.abs(np.fft.rfft(x)) ** 2 if x.size else np.array([0.0])
        return freqs, spec
    freqs, psd = signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        window=RAVEN_WINDOW, scaling="density",
    )
    return freqs, psd


def _spectrogram_entropy_bits(x: np.ndarray, fs: int) -> float:
    nperseg = min(RAVEN_NPERSEG, max(32, x.size // 4 * 2))
    if nperseg < 32:
        return float("nan")
    noverlap = nperseg // 2
    freqs, times, sxx = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
        window=RAVEN_WINDOW, scaling="density", mode="psd",
    )
    if sxx.size == 0:
        return float("nan")
    entropies = []
    for i in range(sxx.shape[1]):
        frame = sxx[:, i]
        total = float(np.sum(frame))
        if total <= 0:
            continue
        p = frame / total
        p = p[p > 0]
        entropies.append(float(-np.sum(p * np.log2(p))))
    if not entropies:
        return float("nan")
    return float(np.mean(entropies))


def _bw50_around_peak(freqs: np.ndarray, psd: np.ndarray) -> float:
    total = float(np.sum(psd))
    if total <= 0 or freqs.size == 0:
        return float("nan")
    peak = int(np.argmax(psd))
    left = right = peak
    captured = float(psd[peak])
    target = 0.5 * total
    n = psd.size
    while captured < target and (left > 0 or right < n - 1):
        take_left = False
        if left > 0 and right < n - 1:
            take_left = psd[left - 1] >= psd[right + 1]
        elif left > 0:
            take_left = True
        if take_left:
            left -= 1
            captured += float(psd[left])
        else:
            right += 1
            captured += float(psd[right])
    return float(freqs[right] - freqs[left])


def measure_raven_metrics(
    x: np.ndarray,
    fs: int,
    *,
    begin_s: float,
    end_s: float,
) -> dict[str, float]:
    """Raven-like band/entropy/power metrics on a mono slice. Low/High are 5/95% power."""
    x = np.asarray(x, dtype=np.float64)
    freqs, psd = _mean_psd(x, fs)
    total = float(np.sum(psd))
    if total <= 0 or freqs.size == 0:
        low = high = center = bw50 = np.nan
        power_db = np.nan
    else:
        c = np.cumsum(psd)
        c = c / c[-1]
        low = float(freqs[min(int(np.searchsorted(c, 0.05)), freqs.size - 1)])
        high = float(freqs[min(int(np.searchsorted(c, 0.95)), freqs.size - 1)])
        center = float(np.sum(freqs * psd) / total)
        bw50 = _bw50_around_peak(freqs, psd)
        power_db = float(10.0 * np.log10(float(np.mean(psd)) + 1e-20))
        high_band = freqs >= 50000.0
        frac_above_50k = float(np.sum(psd[high_band]) / total) if np.any(high_band) else 0.0
        low_band = freqs <= 20000.0
        frac_below_20k = float(np.sum(psd[low_band]) / total) if np.any(low_band) else 0.0
    entropy = _spectrogram_entropy_bits(x, fs)
    if total <= 0 or freqs.size == 0:
        frac_above_50k = float("nan")
        frac_below_20k = float("nan")
    return {
        "Begin Time (s)": float(begin_s),
        "End Time (s)": float(end_s),
        "Delta Time (s)": float(max(0.0, end_s - begin_s)),
        "Low Freq (Hz)": low,
        "High Freq (Hz)": high,
        "BW 50% (Hz)": bw50,
        "Avg Entropy (bits)": entropy,
        "Center Freq (Hz)": center,
        "Avg Power Density (dB/Hz)": power_db,
        "Nyquist (Hz)": float(fs) / 2.0,
        "Frac power >50 kHz": frac_above_50k,
        "Frac power <20 kHz": frac_below_20k,
    }


def slice_channel(data: np.ndarray, channel_1indexed: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr
    idx = max(0, min(arr.shape[1] - 1, channel_1indexed - 1))
    return arr[:, idx]


def clip_window(center: float | None, half: float, duration: float) -> tuple[float, float] | None:
    if center is None or not np.isfinite(center) or duration <= 0:
        return None
    start = max(0.0, center - half)
    end = min(duration, center + half)
    if end <= start:
        return None
    return start, end


def rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def snr_db(signal_x: np.ndarray, noise_x: np.ndarray, *, eps: float = SNR_EPS) -> float:
    """20 log10(signal_rms / noise_rms). Empty noise -> +inf so the gate keeps it."""
    s = rms(signal_x)
    n = rms(noise_x)
    if n <= 0:
        return float("inf") if s > 0 else float("nan")
    return float(20.0 * np.log10((s + eps) / (n + eps)))


def noise_from_window(
    window: np.ndarray,
    fs: int,
    w0: float,
    begin_s: float,
    end_s: float,
    *,
    pad_s: float = 0.005,
    min_noise_s: float = 0.020,
) -> np.ndarray:
    """Local ambient for SNR: 50–150 ms before the burst, then after, then leftover.

    Do not use the rest of a 6 s video window as noise — it often contains the
    other burst in the same trial and tanks SNR.
    """
    w = np.asarray(window, dtype=np.float64)
    n = w.size
    if n == 0 or fs <= 0:
        return w
    min_n = max(1, int(round(min_noise_s * fs)))

    def _slice(t0: float, t1: float) -> np.ndarray:
        j0 = max(0, int(round((t0 - w0) * fs)))
        j1 = max(0, min(n, int(round((t1 - w0) * fs))))
        if j1 > j0:
            return w[j0:j1]
        return w[:0]

    i0 = max(0, int(round((begin_s - w0) * fs)))
    i1 = min(n, int(round((end_s - w0) * fs)))
    pad = int(round(pad_s * fs))
    mask = np.ones(n, dtype=bool)
    mask[max(0, i0 - pad) : min(n, i1 + pad)] = False
    absx = np.abs(w)
    # Quiet samples in the window (ignore other BPs sitting at high amplitude).
    outside = absx[mask]
    if outside.size < min_n:
        outside = absx
    q = float(np.percentile(outside, 25))
    quiet = w[mask & (absx <= max(q, 1e-12))]
    if quiet.size >= min_n:
        return quiet
    quiet_all = w[absx <= max(q, 1e-12)]
    if quiet_all.size >= min_n:
        return quiet_all
    pre = _slice(begin_s - 0.150, begin_s - 0.050)
    if pre.size:
        return pre
    return w[mask] if mask.any() else w


def non_bp_reason(metrics: dict) -> str | None:
    """Whistle (narrow/tonal) or hydro-hit (fuzzy high-frequency energy only)."""
    bw = metrics.get("BW 50% (Hz)")
    ent = metrics.get("Avg Entropy (bits)")
    low = metrics.get("Low Freq (Hz)")
    high = metrics.get("High Freq (Hz)")
    frac_hi = metrics.get("Frac power >50 kHz")
    frac_lo = metrics.get("Frac power <20 kHz")
    try:
        bw_f = float(bw)
    except (TypeError, ValueError):
        bw_f = float("nan")
    try:
        ent_f = float(ent)
    except (TypeError, ValueError):
        ent_f = float("nan")
    # Whistles: skinny spectral peak, low entropy
    pulses = metrics.get("Pulses")
    try:
        n_pulses = int(pulses)
    except (TypeError, ValueError):
        n_pulses = 0
    # Whistle: narrow AND low entropy. A click train can look low-entropy on a 512-pt FFT.
    if np.isfinite(ent_f) and ent_f < 3.2:
        return "whistle"
    if n_pulses < 30 and np.isfinite(bw_f) and np.isfinite(ent_f) and bw_f < 2000.0 and ent_f < 4.0:
        return "whistle"
    ici = metrics.get("mean_ici_ms")
    try:
        ici_f = float(ici)
    except (TypeError, ValueError):
        ici_f = float("nan")
    # Waveform ICI < 10 ms is necessary; spectrogram/packed ICI ~2–5 ms is the BP look.
    if np.isfinite(ici_f) and ici_f > MAX_MEAN_ICI_FOR_BP_MS and n_pulses < 80:
        return "background_clicks"
    try:
        low_f = float(low)
    except (TypeError, ValueError):
        low_f = float("nan")
    try:
        hi_f = float(high)
    except (TypeError, ValueError):
        hi_f = float("nan")
    try:
        fhi = float(frac_hi)
    except (TypeError, ValueError):
        fhi = float("nan")
    try:
        flo = float(frac_lo)
    except (TypeError, ValueError):
        flo = float("nan")
    # Hydro scrape: energy lives up high, little low-frequency click body
    # Hydro scrape alone (fuzzy top of spectrogram). Keep BP+hydro edge cases
    # when there is still a low-frequency click body.
    if np.isfinite(flo) and flo >= 0.10:
        return None
    if np.isfinite(low_f) and low_f >= 25000.0:
        return "hydro_hit"
    if np.isfinite(fhi) and np.isfinite(flo) and fhi >= 0.55 and flo < 0.08:
        return "hydro_hit"
    return None


def gap_looks_like_whistle(x: np.ndarray, fs: int) -> bool:
    """True if a short gap is tonal (whistle) rather than a real ICI break."""
    if x.size < RAVEN_NPERSEG // 2:
        return False
    m = measure_raven_metrics(x, fs, begin_s=0.0, end_s=x.size / float(fs))
    ent = m.get("Avg Entropy (bits)")
    bw = m.get("BW 50% (Hz)")
    try:
        if float(ent) < 4.0:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if float(bw) < 2500.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def merge_whistle_split_trains(
    trains: list[BurstTrain],
    gap_is_whistle=None,
) -> list[BurstTrain]:
    """Join two ICI trains if a whistle sits in a small gap (same channel).

    `gap_is_whistle(cur, nxt)` should return True only when the spectrogram
    of the gap is tonal. A silent ICI break > 10 ms stays two BPs.
    """
    if len(trains) < 2:
        return trains
    ordered = sorted(trains, key=lambda t: (t.channel, t.begin_s))
    out: list[BurstTrain] = []
    i = 0
    while i < len(ordered):
        cur = ordered[i]
        j = i + 1
        while j < len(ordered):
            nxt = ordered[j]
            if nxt.channel != cur.channel:
                break
            gap = nxt.begin_s - cur.end_s
            if gap <= 0 or gap > WHISTLE_GAP_MAX_S:
                break
            if gap_is_whistle is not None and not gap_is_whistle(cur, nxt):
                break
            # merge: one box spanning both (whistle in the middle)
            peaks = np.concatenate([cur.peak_times_s, nxt.peak_times_s]) if cur.peak_times_s.size else nxt.peak_times_s
            icis = np.diff(np.sort(peaks)) * 1000.0 if peaks.size > 1 else np.array([cur.mean_ici_ms])
            mean_ici = float(np.mean(icis[icis > 0])) if icis.size else cur.mean_ici_ms
            cur = BurstTrain(
                begin_s=cur.begin_s,
                end_s=nxt.end_s,
                pulses=int(cur.pulses + nxt.pulses),
                mean_ici_ms=mean_ici,
                prr=(1000.0 / mean_ici) if mean_ici else cur.prr,
                channel=cur.channel,
                event=cur.event or nxt.event,
                window_start=min(cur.window_start, nxt.window_start),
                window_end=max(cur.window_end, nxt.window_end),
                peak_times_s=np.sort(peaks) if peaks.size else cur.peak_times_s,
                loudness=max(cur.loudness, nxt.loudness),
            )
            j += 1
        out.append(cur)
        i = j if j > i + 1 else i + 1
    return out
