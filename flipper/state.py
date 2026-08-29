"""LangGraph state schema for one analysis session."""

from typing import Annotated, Any
import operator

from typing_extensions import TypedDict


class AnalysisState(TypedDict):
    notes: str
    draft_rows: list[dict[str, Any]]
    valid_rows: list[dict[str, Any]]
    rejected: list[str]
    approved: bool
    model_used: str
    log: Annotated[list[str], operator.add]
