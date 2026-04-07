"""Inference logging — structured logs for LLM calls with file rotation."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from pile.config import settings

_logger: logging.Logger | None = None


def setup_inference_logger() -> logging.Logger:
    """Set up the inference logger with file rotation and console output."""
    global _logger

    logger = logging.getLogger("pile.inference")
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    log_dir = os.path.expanduser(settings.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # File handler — rotating, 50MB max, keep 7 files
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "inference.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_inference_logger() -> logging.Logger:
    """Return the inference logger, setting up if needed."""
    global _logger
    if _logger is None:
        return setup_inference_logger()
    return _logger


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
