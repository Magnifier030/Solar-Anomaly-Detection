"""Immediate terminal progress mirrored to an append-only console.log."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER = logging.getLogger("research_training.progress")


def report(message: str) -> None:
    """Emit a status line; StreamHandler flushes immediately, including piped output."""
    LOGGER.info(message)


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@contextmanager
def console_output(directory: Path) -> Iterator[None]:
    """Scope handlers to one run so queues and tests never duplicate or leak handlers."""
    previous_handlers = LOGGER.handlers[:]
    previous_level, previous_propagate = LOGGER.level, LOGGER.propagate
    stream = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(directory / "console.log", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = [stream, file_handler]
    for handler in handlers:
        handler.setFormatter(formatter)
    LOGGER.handlers = handlers
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    try:
        yield
    finally:
        for handler in LOGGER.handlers:
            handler.close()
        LOGGER.handlers = previous_handlers
        LOGGER.setLevel(previous_level)
        LOGGER.propagate = previous_propagate


class BatchProgress:
    """Report first/last batches and bounded intermediate progress for long phases."""

    def __init__(self, phase: str, total: int) -> None:
        self.phase, self.total = phase, total
        self.started = self.last = time.perf_counter()

    def update(self, completed: int) -> None:
        now = time.perf_counter()
        if (
            completed == 1
            or completed == self.total
            or completed % 10 == 0
            or now - self.last >= 10
        ):
            elapsed = now - self.started
            eta = elapsed / max(1, completed) * max(0, self.total - completed)
            report(
                f"{self.phase} | batch {completed}/{self.total} | "
                f"{duration(elapsed)} elapsed | ETA={duration(eta)}"
            )
            self.last = now
