"""Tests for inference logging setup."""

import logging
import os
from unittest.mock import patch

import pytest


def test_setup_creates_log_dir(tmp_path):
    """Logger setup should create log directory."""
    from pile.models.logging import setup_inference_logger

    log_dir = str(tmp_path / "logs")
    with patch("pile.models.logging.settings") as mock_settings:
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = log_dir
        logger = setup_inference_logger()

    assert os.path.isdir(log_dir)
    assert logger.name == "pile.inference"


def test_log_level_from_config(tmp_path):
    """Logger should use level from config."""
    from pile.models.logging import setup_inference_logger

    with patch("pile.models.logging.settings") as mock_settings:
        mock_settings.log_level = "DEBUG"
        mock_settings.log_dir = str(tmp_path)
        logger = setup_inference_logger()

    assert logger.level == logging.DEBUG


def test_log_inference_call(tmp_path, caplog):
    """log_inference_call should log at INFO level."""
    from pile.models.logging import setup_inference_logger, log_inference_call

    with patch("pile.models.logging.settings") as mock_settings:
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = str(tmp_path)
        setup_inference_logger()

    with caplog.at_level(logging.INFO, logger="pile.inference"):
        log_inference_call(
            role="agent",
            latency_ms=3200,
            input_tokens=1200,
            output_tokens=85,
            status="ok",
            tool_calls=2,
        )

    assert "role=agent" in caplog.text
    assert "latency=3200ms" in caplog.text
