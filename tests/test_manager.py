"""Tests for model manager — download detection and model loading."""

from pathlib import Path
from unittest.mock import patch

from pile.models.registry import MODELS


def test_is_model_downloaded_false(tmp_path):
    """Model not downloaded yet."""
    from pile.models.manager import is_model_downloaded

    with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
        assert is_model_downloaded("agent") is False


def test_is_model_downloaded_true(tmp_path):
    """Model file exists."""
    from pile.models.manager import is_model_downloaded

    model_dir = tmp_path / "agent"
    model_dir.mkdir()
    (model_dir / MODELS["agent"]["filename"]).write_bytes(b"fake-gguf")

    with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
        assert is_model_downloaded("agent") is True


def test_get_missing_models_all_missing(tmp_path):
    """All models missing on fresh install."""
    from pile.models.manager import get_missing_models

    with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
        missing = get_missing_models()
        assert set(missing) == {"agent", "router", "embedding"}


def test_get_missing_models_none_missing(tmp_path):
    """All models present."""
    from pile.models.manager import get_missing_models

    for role, info in MODELS.items():
        d = tmp_path / role
        d.mkdir()
        (d / info["filename"]).write_bytes(b"fake")

    with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
        assert get_missing_models() == []
