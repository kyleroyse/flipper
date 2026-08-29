"""Main entry point for Flipper application."""

from __future__ import annotations

import argparse

from flipper import __version__
from flipper.agents.agent import Agent, load_system_prompt
from flipper.memory.store import MemoryStore
from flipper.tools.audio import ListAudioFormatsTool, ProcessAudioTool
from flipper.tools.registry import ToolRegistry
from flipper.utils import setup_logging


def build_agent() -> Agent:
    """Wire registry, memory, and the default agent loop."""
    registry = ToolRegistry()
    registry.register(ListAudioFormatsTool())
    registry.register(ProcessAudioTool())
    return Agent(
        registry=registry,
        memory=MemoryStore(),
        system_prompt=load_system_prompt(),
    )


def main() -> None:
    """Run the Flipper application."""
    parser = argparse.ArgumentParser(description="Flipper audio agent")
    parser.add_argument(
        "--task",
        "-t",
        help="Run one agent task and print the result",
    )
    args = parser.parse_args()

    logger = setup_logging()

    if args.task:
        result = build_agent().run(args.task)
        print(result)
        logger.info("Agent task completed")
        return

    print(f"Flipper v{__version__} is running.")
    print("Object-oriented audio processing framework initialized.")
    print("Environment setup complete.")
    print("Pass --task to run the agent, for example:")
    print('  python main.py --task "what formats do you support?"')
    print("Analysis session (Grok 4.6, ChatGPT fallback):")
    print("  python -m flipper.graphs.analysis_session --notes data/raw/session_notes.txt --thread burst-2026-08-28")

    logger.info("Flipper application started successfully")


if __name__ == "__main__":
    main()
