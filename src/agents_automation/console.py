"""Console helpers for the CLIs."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def quiet_package_logs(level: int = logging.ERROR) -> Iterator[None]:
    """Hide the package's warnings while a CLI prints a transcript that already shows them."""
    logger = logging.getLogger("agents_automation")
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)
