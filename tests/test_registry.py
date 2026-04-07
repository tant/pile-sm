"""Tests for model registry."""

from pile.models.registry import MODELS, MODELS_DIR, get_model_path


def test_registry_has_three_roles():
    assert set(MODELS.keys()) == {"agent", "router", "embedding"}


def test_each_model_has_required_fields():
    for role, info in MODELS.items():
        assert "repo" in info, f"{role} missing repo"
        assert "filename" in info, f"{role} missing filename"
        assert "size_gb" in info, f"{role} missing size_gb"
        assert info["filename"].endswith(".gguf"), f"{role} filename must be .gguf"


def test_get_model_path():
    path = get_model_path("agent")
    assert "agent" in str(path)
    assert str(path).endswith(".gguf")


def test_get_model_path_invalid_role():
    import pytest
    with pytest.raises(KeyError):
        get_model_path("nonexistent")
