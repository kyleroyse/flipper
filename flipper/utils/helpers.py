"""Helper utilities for the Flipper project."""

import logging
from typing import Optional


def setup_logging(
    level: int = logging.INFO, name: Optional[str] = None
) -> logging.Logger:
    """Set up logging for the application.

    Args:
        level: Logging level (default: INFO)
        name: Logger name (default: __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
