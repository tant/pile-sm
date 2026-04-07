"""Logging setup — separate files for inference and app logs."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from pile.config import settings

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_inference_logger: logging.Logger | None = None
_app_logger_initialized = False


def _make_file_handler(filepath: str, level: int) -> RotatingFileHandler:
    """Create a rotating file handler (50MB, 7 backups)."""
    handler = RotatingFileHandler(
        filepath, maxBytes=50 * 1024 * 1024, backupCount=7, encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    return handler


def _make_console_handler(level: int) -> logging.StreamHandler:
    """Create a console handler."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    return handler


def setup_inference_logger() -> logging.Logger:
    """Set up pile.inference logger → inference.log + terminal."""
    global _inference_logger

    logger = logging.getLogger("pile.inference")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    log_dir = os.path.expanduser(settings.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    logger.addHandler(_make_file_handler(os.path.join(log_dir, "inference.log"), level))
    logger.addHandler(_make_console_handler(level))

    _inference_logger = logger
    return logger


def setup_app_logger() -> None:
    """Set up pile.* loggers (except inference) → app.log + terminal.

    Attaches a file handler to the 'pile' root logger so all pile.models,
    pile.router, pile.context, etc. are captured in app.log.
    pile.inference is excluded via its own logger (does not propagate).
    """
    global _app_logger_initialized
    if _app_logger_initialized:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    log_dir = os.path.expanduser(settings.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    app_logger = logging.getLogger("pile")
    app_logger.setLevel(level)

    if app_logger.handlers:
        app_logger.handlers.clear()

    app_logger.addHandler(_make_file_handler(os.path.join(log_dir, "app.log"), level))
    app_logger.addHandler(_make_console_handler(level))

    # Prevent pile.inference from duplicating into app.log
    inference_logger = logging.getLogger("pile.inference")
    inference_logger.propagate = False

    _app_logger_initialized = True


def get_inference_logger() -> logging.Logger:
    """Return the inference logger, setting up if needed."""
    global _inference_logger
    if _inference_logger is None:
        return setup_inference_logger()
    return _inference_logger


def log_inference_call(
    *,
    role: str,
    latency_ms: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    status: str = "ok",
    tool_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Log an inference call at INFO level."""
    logger = get_inference_logger()
    parts = [f"role={role}", f"latency={latency_ms}ms"]
    if input_tokens is not None:
        parts.append(f"input_tokens={input_tokens}")
    if output_tokens is not None:
        parts.append(f"output_tokens={output_tokens}")
    if tool_calls is not None:
        parts.append(f"tool_calls={tool_calls}")
    parts.append(f"status={status}")
    if error:
        parts.append(f"error={error}")
    logger.info(" ".join(parts))


def log_inference_detail(
    *,
    role: str,
    direction: str,
    content: str,
) -> None:
    """Log full prompt/response at DEBUG level for troubleshooting."""
    logger = get_inference_logger()
    logger.debug("role=%s %s:\n%s", role, direction, content)
