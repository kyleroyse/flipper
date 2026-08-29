"""Main entry point for Flipper application."""

from flipper import __version__
from flipper.utils import setup_logging


def main() -> None:
    """Run the Flipper application."""
    logger = setup_logging()

    print(f"Flipper v{__version__} is running.")
    print("Object-oriented audio processing framework initialized.")
    print("Environment setup complete.")

    logger.info("Flipper application started successfully")


if __name__ == "__main__":
    main()
