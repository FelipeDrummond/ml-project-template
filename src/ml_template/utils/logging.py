"""Stdlib logging setup. Use logging, not print()."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str | int = "INFO") -> None:
    """Configure root logger with a single stderr handler. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
