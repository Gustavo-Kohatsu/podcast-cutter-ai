"""Centralized logging configuration.

Call ``setup_logging(settings)`` once at application startup (in main.py).
All other modules should use ``get_logger(__name__)`` to obtain their logger.

This approach ensures:
- Consistent log format across the entire application
- Logs written to both stdout and a persistent file
- Module-level logger names for easy log filtering (e.g. "app.services.transcriber")
"""

import logging
import sys
from pathlib import Path

from app.config.settings import Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(settings: Settings) -> None:
    """Configure root logger with stdout and file handlers.

    Must be called exactly once at application startup, before any
    other module calls ``get_logger()``.

    Args:
        settings: Application settings containing log level and log directory.
    """
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "pipeline.log"

    handlers: list[logging.Handler] = [
        _build_stream_handler(),
        _build_file_handler(log_file),
    ]

    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module.

    Args:
        name: Module name — always pass ``__name__`` from the calling module.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting download...")
    """
    return logging.getLogger(name)


def _build_stream_handler() -> logging.StreamHandler:
    """Build a stdout stream handler with color-friendly formatting."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_file_handler(log_file: Path) -> logging.FileHandler:
    """Build a file handler that appends to the pipeline log file."""
    handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler
