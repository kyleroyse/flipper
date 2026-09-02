"""Video-timestamp → ±window → ICI trains → Raven/_Flipper outputs.

Never writes the original Excel workbook or WAV files. All outputs include
Flipper in the filename before the extension.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from flipper.services.burst_pulses import (
    DEFAULT_MIN_SNR_DB,
    BurstTrain,
    clip_window,
    detect_in_array,
    measure_raven_metrics,
    merge_overlapping_detections,
    gap_looks_like_whistle,
    merge_whistle_split_trains,
    noise_from_window,
    non_bp_reason,
    prefer_seconds,
    read_wav_window,
    rms,
    slice_channel,
    snr_db,
    wav_duration_s,
)

DEFAULT_EXCEL = Path(
    "/Users/christiannadewind/Desktop/UCSD/projects/burst pulse/"
    "dolphin_data_template_v3.xlsx"
)
DEFAULT_AUDIO_DIR = Path("/Users/christiannadewind/Desktop/UCSD/projects/burst pulse")
DEFAULT_OUT = Path("data/processed")
DATA_SHEET = "Data"
AUDIO_DATA_COLUMNS = [
    "Date",
    "Dolphin",
    "Session",
    "Trial",
    "BP",
    "Selection in Raven",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Delta Time (s)",
    "Avg Power Density (dB/Hz)",
    "BW 50% (Hz)",
    "Agg Entropy (bits)",
    "Center Freq (Hz)",
    "Pulses",
    "mean_ici_ms",
    "Pulse Repitition Rate",
]
CSV_EXTRA = ["Event", "Wav", "WindowStart", "WindowEnd", "SNR (dB)"]
RAVEN_COLUMNS = [
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Delta Time (s)",
    "Low Freq (Hz)",
    "BW 50% (Hz)",
    "Avg Entropy (bits)",
    "Center Freq (Hz)",
    "High Freq (Hz)",
    "Avg Power Density (dB/Hz)",
    "Trial",
]
OBJ_RE = re.compile(
    r"OBJ_(\d{2})-(\d{2})-(\d{2})_([A-Za-z]{2,6})_",
    re.IGNORECASE,
)
DOTDATE_RE = re.compile(r"([A-Za-z]{2,6})\.(\d{6})\b")
UNDATE_RE = re.compile(r"([A-Za-z]{2,6})_(\d{8})")
TRIAL_T_RE = re.compile(r"_T(\d+)\s*$", re.IGNORECASE)
SESSION_RE = re.compile(r"(?:^|[_\s-])S(\d+)(?:[_\s-]|$)", re.IGNORECASE)


def ensure_flipper_filename(path: Path) -> Path:
    """Require Flipper in the filename before the extension. Never overwrite sources.

    Raven tables use ``*_Flipper.selections.txt`` (Flipper is not the last stem
    segment), so any existing ``Flipper`` token is left unchanged.
    """
    if "Flipper" in path.name:
        return path
    return path.with_name(f"{path.stem}_Flipper{path.suffix}")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip column whitespace for lookups; keep a Session column."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_data_sheet(excel: Path) -> pd.DataFrame:
    df = pd.read_excel(excel, sheet_name=DATA_SHEET)
    return _norm_cols(df)


def _date_iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        s = str(value).strip()
        return s[:10] if len(s) >= 10 else None
    return ts.strftime("%Y-%m-%d")


def _date_compact(iso: str) -> str:
    return iso.replace("-", "")


def _yy_mm_dd(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{y[2:]}-{m}-{d}"


def parse_trial_number(trial: str) -> int | None:
    m = TRIAL_T_RE.search(str(trial))
    if m:
        return int(m.group(1))
    return None


def raven_trial_label(trial_num: int, index_in_trial: int) -> str:
    """First BP in a trial is `6`; extras are `6.2`, `6.3`, ..."""
    if index_in_trial <= 0:
        return str(trial_num)
    return f"{trial_num}.{index_in_trial + 1}"


def _year_from_yy(yy: int) -> int:
    return 2000 + yy if yy < 70 else 1900 + yy


def parse_wav_dolphin_date(path: Path) -> list[tuple[str, str]]:
    """Possible (DOLPHIN, YYYY-MM-DD) keys encoded in name or parents."""
    found: list[tuple[str, str]] = []
    text = path.name
    m = OBJ_RE.search(text)
    if m:
        yy, mm, dd, dol = m.groups()
        iso = f"{_year_from_yy(int(yy)):04d}-{mm}-{dd}"
        found.append((dol.upper(), iso))
    m = DOTDATE_RE.search(text)
    if m:
        dol, ymd = m.groups()
        yy, mm, dd = int(ymd[0:2]), ymd[2:4], ymd[4:6]
        iso = f"{_year_from_yy(yy):04d}-{mm}-{dd}"
        found.append((dol.upper(), iso))
    m = UNDATE_RE.search(text)
    if m:
        dol, ymd = m.groups()
        iso = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
        found.append((dol.upper(), iso))
    return found


def _session_tokens(path: Path) -> set[str]:
    blob = str(path).lower()
    toks = set(SESSION_RE.findall(blob))
    parent = path.parent.name.lower()
    toks.update(SESSION_RE.findall(parent))
    return {str(int(t)) for t in toks}


def score_wav_for_session(path: Path, session: str | int | None) -> int:
    score = 0
    name = path.name
    if name.upper().startswith("OBJ_"):
        score += 10
    if re.search(r"\(\d+\)", name):
        score -= 5
    if session is None or str(session).strip() == "":
        return score
    sess = str(int(float(session)))
    tokens = _session_tokens(path)
    if sess in tokens:
        score += 8
    if re.search(rf"s{sess}(?:\D|$)", str(path), re.IGNORECASE):
        score += 3
    return score


def index_wavs(audio_dirs: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in audio_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() != ".wav":
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return out


def match_wav(
    wavs: list[Path],
    *,
    dolphin: str,
    date_iso: str,
    session: Any,
) -> tuple[Path | None, str | None]:
    """Return (path, skip_reason). skip_reason set when unmatched or ambiguous."""
    dol = str(dolphin).strip().upper()
    candidates: list[Path] = []
    for p in wavs:
        keys = parse_wav_dolphin_date(p)
        if any(k[0] == dol and k[1] == date_iso for k in keys):
            candidates.append(p)
            continue
        parent_blob = str(p.parent).upper()
        if dol in parent_blob.split("/") and date_iso.replace("-", "")[-4:] in parent_blob:
            # weak parent match e.g. JULY28 + KOD folder
            if dol in p.parent.name.upper() or dol in p.name.upper():
                yymmdd = _yy_mm_dd(date_iso)
                if yymmdd in p.name or date_iso in str(p):
                    candidates.append(p)
    # de-dupe
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    if not uniq:
        return None, f"no wav for {dol} {date_iso}"
    if len(uniq) == 1:
        return uniq[0], None
    scored = [(score_wav_for_session(p, session), p) for p in uniq]
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0][0]
    top = [p for s, p in scored if s == best]
    if len(top) == 1:
        return top[0], None
    names = ", ".join(p.name for p in top[:6])
    return None, f"ambiguous wavs for {dol} {date_iso} S{session}: {names}"


def is_clang(trial: Any) -> bool:
    return "CLANG" in str(trial).upper()


def row_to_trial(row: dict[str, Any]) -> dict[str, Any] | None:
    trial = row.get("Trial")
    if trial is None or is_clang(trial):
        return None
    dolphin = row.get("Dolphin")
    date_iso = _date_iso(row.get("Date"))
    if not dolphin or not date_iso:
        return None
    session = row.get("Session")
    try:
        if session is not None and not (isinstance(session, float) and np.isnan(session)):
            session = int(float(session))
        else:
            session = None
    except (TypeError, ValueError):
        pass
    touch = prefer_seconds(row, "Touch Obj Time", "Touch Obj Time (s)")
    bridge = prefer_seconds(row, "Bridge", "Bridge (s)")
    return {
        "Date": date_iso,
        "Dolphin": str(dolphin).strip().upper(),
        "Session": session,
        "Trial": str(trial).strip(),
        "touch_s": touch,
        "bridge_s": bridge,
        "trial_num": parse_trial_number(str(trial)),
    }


def load_trials(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    trials = []
    for rec in records:
        t = row_to_trial(rec)
        if t:
            trials.append(t)
    return trials


def session_key(t: dict[str, Any]) -> tuple[str, str, Any]:
    return t["Dolphin"], t["Date"], t["Session"]


def process_trial_windows(
    trial: dict[str, Any],
    wav: Path,
    duration: float,
    *,
    half_window: float,
    min_snr: float = DEFAULT_MIN_SNR_DB,
    **detect_kw: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[tuple[str, float]] = []
    if trial["touch_s"] is not None:
        events.append(("touch", trial["touch_s"]))
    if trial["bridge_s"] is not None:
        events.append(("bridge", trial["bridge_s"]))
    found: list[BurstTrain] = []
    for event, center in events:
        win = clip_window(center, half_window, duration)
        if win is None:
            continue
        w0, w1 = win
        fs, data = read_wav_window(wav, w0, w1)
        trains, _ch = detect_in_array(data, fs, t0=w0, **detect_kw)
        for tr in trains:
            tr.event = event
            tr.window_start = w0
            tr.window_end = w1
            found.append(tr)
    def _gap_whistle(a, b):
        fs_g, data_g = read_wav_window(wav, a.end_s, b.begin_s)
        if data_g.size == 0:
            return False
        mono_g = slice_channel(data_g, a.channel)
        return gap_looks_like_whistle(mono_g, fs_g)

    merged = merge_whistle_split_trains(
        merge_overlapping_detections(found), gap_is_whistle=_gap_whistle
    )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for tr in merged:
        fs, data = read_wav_window(wav, tr.begin_s, tr.end_s)
        if data.size == 0:
            continue
        arr = data
        if arr.ndim == 2 and arr.shape[1] > 1:
            loud = [rms(arr[:, c]) for c in range(arr.shape[1])]
            tr.channel = int(max(range(len(loud)), key=lambda i: loud[i]) + 1)
        mono = slice_channel(data, tr.channel)
        metrics = measure_raven_metrics(mono, fs, begin_s=tr.begin_s, end_s=tr.end_s)
        fs_w, data_w = read_wav_window(wav, tr.window_start, tr.window_end)
        noise = noise_from_window(
            slice_channel(data_w, tr.channel),
            fs_w,
            tr.window_start,
            tr.begin_s,
            tr.end_s,
        )
        snr = snr_db(mono, noise)
        rec = {
            **metrics,
            "Channel": tr.channel,
            "Pulses": tr.pulses,
            "mean_ici_ms": tr.mean_ici_ms,
            "Pulse Repetition Rate": tr.prr,
            "Pulse Repitition Rate": tr.prr,
            "Event": tr.event,
            "WindowStart": tr.window_start,
            "WindowEnd": tr.window_end,
            "Wav": str(wav),
            "SNR (dB)": snr,
        }
        # inf SNR (silent remainder) is a keep; nan or below threshold is a drop
        if rec.get("Delta Time (s)", 0) < 0.05 and rec.get("Pulses", 0) < 40:
            rec["reason"] = "too_short"
            rejected.append(rec)
            continue
        if not (snr >= min_snr):
            rec["reason"] = "low_snr"
            rejected.append(rec)
            continue
        kind = non_bp_reason(rec)
        if kind:
            rec["reason"] = kind
            rejected.append(rec)
            continue
        rows.append(rec)
    rows.sort(key=lambda r: r["Begin Time (s)"])
    return rows, rejected


def write_raven_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_flipper_filename(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(RAVEN_COLUMNS)]
    for i, r in enumerate(rows, start=1):
        vals = [
            str(i),
            "Waveform 1",
            str(int(r["Channel"])),
            _fmt(r["Begin Time (s)"], 7),
            _fmt(r["End Time (s)"], 7),
            _fmt(r["Delta Time (s)"], 4),
            "0.0",
            _fmt(r["BW 50% (Hz)"], 2),
            _fmt(r["Avg Entropy (bits)"], 3),
            _fmt(r["Center Freq (Hz)"], 2),
            _fmt(r.get("Nyquist (Hz)", 96000.0), 1),
            _fmt(r["Avg Power Density (dB/Hz)"], 2),
            str(r["RavenTrial"]),
        ]
        lines.append("\t".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, ndigits: int) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{ndigits}f}"


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_flipper_filename(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(columns=AUDIO_DATA_COLUMNS + CSV_EXTRA).to_csv(path, index=False)
        return
    df = pd.DataFrame(rows)
    for col in AUDIO_DATA_COLUMNS + CSV_EXTRA:
        if col not in df.columns:
            df[col] = np.nan
    # Agg Entropy shares Avg Entropy
    if "Agg Entropy (bits)" not in df.columns or df["Agg Entropy (bits)"].isna().all():
        if "Avg Entropy (bits)" in df.columns:
            df["Agg Entropy (bits)"] = df["Avg Entropy (bits)"]
    df = df[AUDIO_DATA_COLUMNS + CSV_EXTRA]
    df.to_csv(path, index=False)


def write_rejected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = ensure_flipper_filename(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = AUDIO_DATA_COLUMNS + CSV_EXTRA + ["reason"]
    if not rows:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    if "Agg Entropy (bits)" not in df.columns or df["Agg Entropy (bits)"].isna().all():
        if "Avg Entropy (bits)" in df.columns:
            df["Agg Entropy (bits)"] = df["Avg Entropy (bits)"]
    df = df[cols]
    df.to_csv(path, index=False)


def run_pipeline(
    *,
    excel: Path,
    audio_dir: Path,
    out: Path,
    window: float = 3.0,
    limit: int | None = None,
    dolphin: str | None = None,
    date: str | None = None,
    extra_audio_dirs: list[Path] | None = None,
    min_snr: float = DEFAULT_MIN_SNR_DB,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data_sheet(excel)
    trials = load_trials(df)
    if dolphin:
        d = dolphin.strip().upper()
        trials = [t for t in trials if t["Dolphin"] == d]
    if date:
        iso = _date_iso(date) or date
        trials = [t for t in trials if t["Date"] == iso]

    grouped: dict[tuple[str, str, Any], list[dict[str, Any]]] = defaultdict(list)
    order: list[tuple[str, str, Any]] = []
    for t in trials:
        key = session_key(t)
        if key not in grouped:
            order.append(key)
        grouped[key].append(t)
    if limit is not None and limit > 0:
        order = order[:limit]

    search_roots = [Path(audio_dir)]
    if extra_audio_dirs:
        search_roots.extend(extra_audio_dirs)
    wavs = index_wavs(search_roots)

    unmatched: list[str] = []
    all_csv_rows: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    raven_files: list[str] = []
    n_bp = 0

    wav_cache: dict[tuple[str, str, Any], tuple[Path | None, str | None, float]] = {}

    for key in order:
        dol, date_iso, sess = key
        if key not in wav_cache:
            path, reason = match_wav(wavs, dolphin=dol, date_iso=date_iso, session=sess)
            dur = wav_duration_s(path) if path else 0.0
            wav_cache[key] = (path, reason, dur)
            if reason:
                print(f"WAV MATCH: {reason}")
            elif path:
                print(f"WAV MATCH: {dol} {date_iso} S{sess} -> {path}")
        path, reason, dur = wav_cache[key]
        session_rows: list[dict[str, Any]] = []
        for trial in grouped[key]:
            if path is None:
                unmatched.append(
                    f"{trial['Trial']} ({reason or 'unmatched'})"
                )
                continue
            if trial["touch_s"] is None and trial["bridge_s"] is None:
                unmatched.append(f"{trial['Trial']} (no touch/bridge times)")
                continue
            bp_rows, rejected_rows = process_trial_windows(
                trial, path, dur, half_window=window, min_snr=min_snr
            )
            tnum = trial["trial_num"] if trial["trial_num"] is not None else 0
            for r in rejected_rows:
                r["Date"] = date_iso
                r["Dolphin"] = dol
                r["Session"] = sess
                r["Trial"] = trial["Trial"]
                r["Agg Entropy (bits)"] = r.get("Avg Entropy (bits)")
                all_rejected.append(r)
            for i, r in enumerate(bp_rows):
                r["Date"] = date_iso
                r["Dolphin"] = dol
                r["Session"] = sess
                r["Trial"] = trial["Trial"]
                r["BP"] = i + 1
                r["RavenTrial"] = raven_trial_label(tnum, i)
                r["Selection in Raven"] = None  # filled per session below
                r["Agg Entropy (bits)"] = r.get("Avg Entropy (bits)")
                session_rows.append(r)
        session_rows.sort(key=lambda r: r["Begin Time (s)"])
        for i, r in enumerate(session_rows, start=1):
            r["Selection in Raven"] = i
        if session_rows:
            sess_label = sess if sess is not None else "NA"
            raven_name = f"{dol}_{_date_compact(date_iso)}_S{sess_label}_Flipper.selections.txt"
            raven_path = out / raven_name
            write_raven_table(raven_path, session_rows)
            raven_files.append(str(raven_path))
            n_bp += len(session_rows)
            all_csv_rows.extend(session_rows)

    csv_path = ensure_flipper_filename(out / "Audio_Data_Flipper.csv")
    write_metrics_csv(csv_path, all_csv_rows)
    rejected_path = ensure_flipper_filename(out / "Audio_Data_Flipper_rejected.csv")
    write_rejected_csv(rejected_path, all_rejected)
    if unmatched:
        print("Unmatched / skipped trials:")
        for line in unmatched:
            print(f"  {line}")
    print(f"Wrote {csv_path} ({len(all_csv_rows)} burst rows)")
    print(f"Wrote {rejected_path} ({len(all_rejected)} rejected, SNR < {min_snr} dB)")
    for f in raven_files:
        print(f"Wrote {f}")
    return {
        "csv": str(csv_path),
        "raven_files": raven_files,
        "n_bp": n_bp,
        "unmatched": unmatched,
        "n_sessions": len(order),
        "rejected": str(rejected_path),
        "n_rejected": len(all_rejected),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract burst pulses around video touch/bridge times (Flipper)."
    )
    p.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    p.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window", type=float, default=3.0, help="Half-window seconds (default 3).")
    p.add_argument("--limit", type=int, default=None, help="Max sessions to process.")
    p.add_argument("--dolphin", type=str, default=None)
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument(
        "--min-snr",
        type=float,
        default=DEFAULT_MIN_SNR_DB,
        help="Drop detections with SNR below this (dB). Default 0 (keep all; SNR still recorded).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_pipeline(
        excel=args.excel,
        audio_dir=args.audio_dir,
        out=args.out,
        window=args.window,
        limit=args.limit,
        dolphin=args.dolphin,
        date=args.date,
        min_snr=args.min_snr,
    )


if __name__ == "__main__":
    main()
