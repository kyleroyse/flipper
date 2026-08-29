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
