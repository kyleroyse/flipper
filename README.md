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
