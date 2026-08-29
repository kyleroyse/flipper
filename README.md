# Flipper

Minimal Python scaffold. Prints a short status line so you can confirm the environment is working.

Requires Python 3.10+.

## Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

A `.venv` may already exist if you set this project up on this machine. You can skip `python3 -m venv .venv` and just activate it.

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
