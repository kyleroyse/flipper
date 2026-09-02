# Flipper

Minimal Python scaffold. Prints a short status line so you can confirm the environment is working.

Requires Python 3.10+. Includes audio processing and spectrogram analysis libraries.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

A `.venv` may already exist if you set this project up on this machine. You can skip `python3 -m venv .venv` and just activate it.

### Dependencies

This project includes audio processing libraries (librosa, torch, torchaudio, etc.). On macOS, you may need to install CMake for building some native extensions:

```bash
brew install cmake
```

Alternatively, use `uv` for faster dependency resolution and installation:

```bash
uv venv .venv
uv pip install -e .
```

## Run

With the venv active:

```bash
python main.py
```

After `pip install -e .`, you can also run:

```bash
flipper
```

You should see:

```
Flipper is running.
Environment setup complete.
```

## Layout

Audio domain code stays under `flipper/core`, `flipper/models`, `flipper/services`, and `flipper/utils`. The agent only calls that code through `flipper/tools`.

```text
flipper/
  core/ models/ services/ utils/
  agents/     # loop, Message / ToolCall / ToolResult
  tools/      # Tool protocol, registry, audio wrappers
  memory/     # in-memory conversation store
  prompts/    # system.md and optional skills
  llm/        # Grok 4.6 primary, OpenAI backup, router
  graphs/     # LangGraph extract → validate → approve → write
  config.py state.py
data/raw data/processed runs/
main.py
tests/
```

## Run the agent

With the venv active:

```bash
python main.py --task "what formats do you support?"
python main.py --task "process a silent clip"
```

Or after `pip install -e .`:

```bash
flipper --task "what formats do you support?"
```

`python main.py` with no arguments still prints the environment check.

## Tests

```bash
python -m unittest discover -s tests
```

## LangGraph analysis session

Grok 4.6 is the primary model. OpenAI ChatGPT runs only if Grok fails (timeout, 429, 5xx). Graph nodes never import `xai_sdk` or `openai` directly; they call `flipper.llm.router.complete`. Local Python validates units and writes the datasheet. You approve before any official row is written.

```text
notes  →  extract (router)  →  validate (local)  →  interrupt (you)
                                                      ↓ approve
                                                 write CSV
```

```bash
cp .env.example .env   # fill XAI_API_KEY and OPENAI_API_KEY
python -m flipper.graphs.analysis_session \
  --notes data/raw/session_notes.txt \
  --thread burst-2026-08-28
```

The graph pauses at the human gate. After you review the draft rows:

```bash
python -m flipper.graphs.analysis_session \
  --thread burst-2026-08-28 \
  --resume approve
```

Same `thread_id` reloads the sqlite checkpoint in `runs/`. The CSV in `data/processed/` records `model_used` so you know whether a row came from Grok or the OpenAI backup.

Do not send WAV files to the model. Code measures; the model proposes labels and drafts text.

## Summarize dolphin Excel

Grok 4.6 summarizes the workbook. It does not extract measurement rows from Excel.

```bash
python -m flipper.graphs.analysis_session --limit 20
```

Uses `DOLPHIN_XLSX` from `.env`, sheet `Audio Data`. Pass `--excel path.xlsx` to override. `--limit 0` sends every row.

## Video timestamps → burst pulses (`_Flipper` files)

Local (no LLM) pipeline: read the Excel **Data** sheet, skip CLANG rows, parse
touch/bridge clocks, cut **±3 s** windows on the matching session WAV, detect
burst-pulse trains with Hilbert-envelope ICI grouping, and write Raven tables
plus a metrics CSV. Original workbooks and WAVs are never modified. Output
filenames always contain `Flipper` immediately before the extension
(`Audio_Data_Flipper.csv`, `KOD_20250728_S1_Flipper.selections.txt`).

ICI is **not** `duration/(n-1)`. Successive envelope peaks are grouped into a
train while ICI ≤ 8 ms (default); a train needs ≥ 8 pulses and a mean ICI
between 1.2 and 8 ms. SNR (dB) is written on each row. Pass `--min-snr 10` to drop weak detections into `Audio_Data_Flipper_rejected.csv`. Overlapping touch/bridge windows are de-duplicated
(begin/end within 20 ms).

```bash
python -m flipper.extract_burst_pulses \
  --excel "/Users/christiannadewind/Desktop/UCSD/projects/burst pulse/dolphin_data_template_v3.xlsx" \
  --audio-dir "/Users/christiannadewind/Desktop/UCSD/projects/burst pulse" \
  --out data/processed \
  --dolphin KOD --date 2025-07-28 --limit 1
```

After `pip install -e .` you can also run `flipper-extract` with the same flags.
WAV matching uses dolphin + date in names such as `OBJ_25-07-28_KOD_...wav`;
unmatched or ambiguous trials are printed and skipped.
