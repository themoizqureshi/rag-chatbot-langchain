"""Shared utilities: logging setup and environment validation."""

import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_api_key() -> None:
    """Raise EnvironmentError if GOOGLE_API_KEY is not set."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. "
            "Get a free key at https://aistudio.google.com/apikey "
            "then add it to your .env file."
        )
