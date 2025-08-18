# src/utils/logger.py
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "app.log"


class _Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


_LEVEL_TO_COLOR = {
    logging.DEBUG: _Color.CYAN,
    logging.INFO: _Color.GREEN,
    logging.WARNING: _Color.YELLOW,
    logging.ERROR: _Color.RED,
    logging.CRITICAL: _Color.MAGENTA,
}


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_TO_COLOR.get(record.levelno, _Color.RESET)
        base = f"{_Color.DIM}{record.asctime}{_Color.RESET} {color}{record.levelname:<8}{_Color.RESET} {_Color.BOLD}{record.name}{_Color.RESET}: {record.getMessage()}"
        if record.exc_info:
            base += f"\n{_Color.GRAY}{self.formatException(record.exc_info)}{_Color.RESET}"
        return base


def _make_console_handler(level: int) -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(level)
    fmt = _ConsoleFormatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    h.setFormatter(fmt)
    return h


def _make_file_handler(level: int) -> logging.Handler:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    h.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [%(module)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    h.setFormatter(fmt)
    return h


def get_logger(name: Optional[str] = None, level: Optional[str | int] = None) -> logging.Logger:
    """
    Get a configured logger. Idempotent: calling multiple times won't add duplicate handlers.
    """
    logger = logging.getLogger(name if name else "app")
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), logging.INFO)
    elif isinstance(level, int):
        lvl = level
    else:
        lvl = getattr(logging, _DEFAULT_LEVEL, logging.INFO)

    if not getattr(logger, "_initialized", False):
        logger.setLevel(lvl)
        logger.addHandler(_make_console_handler(lvl))
        logger.addHandler(_make_file_handler(lvl))
        logger.propagate = False
        logger._initialized = True  # type: ignore[attr-defined]

    return logger
