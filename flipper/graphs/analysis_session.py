"""Extract (Grok then OpenAI) -> validate (local) -> human gate -> write CSV."""

from __future__ import annotations

import argparse
import json
import pprint
import re
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from flipper.config import settings
from flipper.llm.grok import SUMMARY_SYSTEM
from flipper.llm.router import complete
from flipper.state import AnalysisState
from flipper.tools.datasheet import write_rows
from flipper.tools.excel import DEFAULT_SHEET, excel_to_notes
from flipper.tools.units import validate_rows

_saver: SqliteSaver | None = None
_conn: sqlite3.Connection | None = None


def extract(state: AnalysisState) -> dict:
    result = complete(state["notes"])
    rows = parse_rows(result.text)
    return {
        "draft_rows": rows,
        "model_used": f"{result.provider}:{result.model}",
        "log": [f"extract via {result.provider}:{result.model} ({len(rows)} rows)"],
    }


def validate(state: AnalysisState) -> dict:
    ok, rejected = validate_rows(state["draft_rows"])
    return {
        "valid_rows": ok,
        "rejected": rejected,
        "log": [f"validate: {len(ok)} ok, {len(rejected)} rejected"],
    }


def human_gate(state: AnalysisState) -> dict:
    decision = interrupt(
        {
            "valid_rows": state["valid_rows"],
            "rejected": state["rejected"],
            "model_used": state["model_used"],
            "ask": "Reply approve or reject",
        }
    )
    approved = str(decision).strip().lower() == "approve"
    return {"approved": approved, "log": [f"human: {decision}"]}


def persist(state: AnalysisState) -> dict:
    if not state["approved"]:
        return {"log": ["persist skipped"]}
    return {"log": ["persist armed"]}


def _get_saver() -> SqliteSaver:
    global _saver, _conn
    if _saver is None:
        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
        _saver = SqliteSaver(_conn)
        if hasattr(_saver, "setup"):
            _saver.setup()
    return _saver


def build_app():
    g = StateGraph(AnalysisState)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("human_gate", human_gate)
    g.add_node("persist", persist)
    g.set_entry_point("extract")
    g.add_edge("extract", "validate")
    g.add_edge("validate", "human_gate")
    g.add_edge("human_gate", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=_get_saver())


def parse_rows(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return []
    payload = json.loads(match.group(0))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return rows if isinstance(rows, list) else []


def _load_notes(args: argparse.Namespace) -> str:
    if args.notes:
        return args.notes.read_text()
    xlsx = args.excel or settings.dolphin_xlsx
    if not xlsx:
        raise SystemExit("--notes, --excel, or DOLPHIN_XLSX is required on first run")
    limit = None if args.limit == 0 else args.limit
    notes = excel_to_notes(xlsx, sheet=args.sheet, limit=limit)
    print(f"Loaded Excel {xlsx} sheet={args.sheet!r} limit={limit}")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flipper analysis session: extract, validate, approve, write"
    )
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--excel", type=Path, help="Dolphin .xlsx (Audio Data by default)")
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max Excel rows to send to the model; 0 means all",
    )
    parser.add_argument("--thread", help="Required for --notes extract / --resume")
    parser.add_argument("--resume", choices=["approve", "reject"])
    args = parser.parse_args()

    excel_mode = args.excel is not None or (
        args.notes is None and settings.dolphin_xlsx is not None and not args.resume
    )
    if excel_mode and not args.resume:
        notes = _load_notes(args)
        result = complete(notes, system=SUMMARY_SYSTEM)
        print(result.text)
        print(f"({result.provider}:{result.model})")
        return

    if args.resume and not args.thread:
        raise SystemExit("--thread is required with --resume")
    if not args.thread:
        raise SystemExit("--thread is required for notes extract")

    app = build_app()
    config = {"configurable": {"thread_id": args.thread}}

    if args.resume:
        result = app.invoke(Command(resume=args.resume), config)
    else:
        result = app.invoke(
            {
                "notes": _load_notes(args),
                "draft_rows": [],
                "valid_rows": [],
                "rejected": [],
                "approved": False,
                "model_used": "",
                "log": [],
            },
            config,
        )

    snapshot = app.get_state(config)
    if snapshot.next:
        values = snapshot.values or {}
        print("PAUSED at", snapshot.next)
        print("model_used:", values.get("model_used"))
        print("valid_rows:")
        pprint.pp(values.get("valid_rows") or [])
        rejected = values.get("rejected") or []
        if rejected:
            print("rejected:")
            pprint.pp(rejected)
        print("Resume with: --thread", args.thread, "--resume approve")
        return

    if result.get("approved"):
        path = write_rows(result["valid_rows"], args.thread, result["model_used"])
        print("Wrote", path)
    else:
        print("Not written. log:", result.get("log"))


if __name__ == "__main__":
    main()
