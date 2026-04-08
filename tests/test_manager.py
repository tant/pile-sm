"""Tests for model manager — download detection and model loading."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pile.models.registry import MODELS


# ---------------------------------------------------------------------------
# is_model_downloaded / get_missing_models (existing tests)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _download_single
# ---------------------------------------------------------------------------

class TestDownloadSingle:
    @patch("huggingface_hub.hf_hub_download")
    def test_calls_hf_hub_download(self, mock_dl, tmp_path):
        from pile.models.manager import _download_single

        with patch("pile.models.manager.MODELS_DIR", str(tmp_path)):
            _download_single("agent")

        info = MODELS["agent"]
        mock_dl.assert_called_once_with(
            repo_id=info["repo"],
            filename=info["filename"],
            local_dir=str(tmp_path / "agent"),
        )


# ---------------------------------------------------------------------------
# download_models
# ---------------------------------------------------------------------------

class TestDownloadModels:
    @patch("pile.models.manager._download_single")
    @patch("pile.models.manager.get_missing_models")
    def test_no_roles_does_nothing(self, mock_missing, mock_dl):
        from pile.models.manager import download_models

        mock_missing.return_value = []
        download_models()
        mock_dl.assert_not_called()

    @patch("pile.models.manager._download_single")
    def test_downloads_specified_roles(self, mock_dl):
        from pile.models.manager import download_models

        download_models(roles=["agent", "router"])
        assert mock_dl.call_count == 2

    @patch("pile.models.manager._download_single")
    def test_raises_on_failure(self, mock_dl):
        from pile.models.manager import download_models

        mock_dl.side_effect = RuntimeError("download failed")
        with pytest.raises(RuntimeError, match="download failed"):
            download_models(roles=["agent"])


# ---------------------------------------------------------------------------
# _detect_gpu_layers
# ---------------------------------------------------------------------------

class TestDetectGpuLayers:
    @patch("pile.models.manager.platform.system", return_value="Darwin")
    def test_macos_returns_minus_one(self, _):
        from pile.models.manager import _detect_gpu_layers
        assert _detect_gpu_layers() == -1

    @patch("subprocess.run")
    @patch("pile.models.manager.platform.system", return_value="Linux")
    def test_linux_with_nvidia(self, _, mock_run):
        from pile.models.manager import _detect_gpu_layers
        mock_run.return_value = MagicMock(returncode=0)
        assert _detect_gpu_layers() == -1

    @patch("subprocess.run")
    @patch("pile.models.manager.platform.system", return_value="Linux")
    def test_linux_no_nvidia(self, _, mock_run):
        from pile.models.manager import _detect_gpu_layers
        mock_run.side_effect = FileNotFoundError
        assert _detect_gpu_layers() == 0

    @patch("pile.models.manager.platform.system", return_value="Windows")
    def test_windows_returns_zero(self, _):
        from pile.models.manager import _detect_gpu_layers
        assert _detect_gpu_layers() == 0


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    @patch("pile.models.manager._detect_gpu_layers", return_value=-1)
    @patch("pile.models.manager.Llama")
    def test_load_agent(self, mock_llama, _, tmp_path):
        from pile.models.manager import load_model

        model_file = tmp_path / "agent" / MODELS["agent"]["filename"]
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"fake")

        with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
            load_model("agent")

        mock_llama.assert_called_once()
        kwargs = mock_llama.call_args[1]
        assert kwargs["n_gpu_layers"] == -1
        assert kwargs["verbose"] is False
        assert "embedding" not in kwargs

    @patch("pile.models.manager._detect_gpu_layers", return_value=0)
    @patch("pile.models.manager.Llama")
    def test_load_router(self, mock_llama, _, tmp_path):
        from pile.models.manager import load_model

        model_file = tmp_path / "router" / MODELS["router"]["filename"]
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"fake")

        with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
            load_model("router")

        mock_llama.assert_called_once()
        kwargs = mock_llama.call_args[1]
        assert kwargs["n_gpu_layers"] == 0

    @patch("pile.models.manager._detect_gpu_layers", return_value=0)
    @patch("pile.models.manager.Llama")
    def test_load_embedding(self, mock_llama, _, tmp_path):
        from pile.models.manager import load_model

        model_file = tmp_path / "embedding" / MODELS["embedding"]["filename"]
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"fake")

        with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
            load_model("embedding")

        mock_llama.assert_called_once()
        kwargs = mock_llama.call_args[1]
        assert kwargs["embedding"] is True
        assert kwargs["n_ctx"] == 2048

    def test_load_unknown_role(self, tmp_path):
        from pile.models.manager import load_model

        model_file = tmp_path / "unknown" / "fake.gguf"
        model_file.parent.mkdir(parents=True)
        model_file.write_bytes(b"fake")

        with pytest.raises((ValueError, KeyError)):
            load_model("unknown")

    def test_load_missing_file(self, tmp_path):
        from pile.models.manager import load_model

        with patch("pile.models.registry.MODELS_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError, match="Model not found"):
                load_model("agent")


# ---------------------------------------------------------------------------
# unload_model / unload_all
# ---------------------------------------------------------------------------

class TestUnloadModel:
    def test_unload_agent(self):
        import pile.models.manager as mgr

        fake = MagicMock()
        old = mgr._agent_model
        try:
            mgr._agent_model = fake
            mgr.unload_model("agent")
            assert mgr._agent_model is None
        finally:
            mgr._agent_model = old

    def test_unload_router(self):
        import pile.models.manager as mgr

        fake = MagicMock()
        old = mgr._router_model
        try:
            mgr._router_model = fake
            mgr.unload_model("router")
            assert mgr._router_model is None
        finally:
            mgr._router_model = old

    def test_unload_embedding(self):
        import pile.models.manager as mgr

        fake = MagicMock()
        old = mgr._embed_model
        try:
            mgr._embed_model = fake
            mgr.unload_model("embedding")
            assert mgr._embed_model is None
        finally:
            mgr._embed_model = old

    def test_unload_already_none(self):
        import pile.models.manager as mgr

        old = mgr._agent_model
        try:
            mgr._agent_model = None
            mgr.unload_model("agent")
            assert mgr._agent_model is None
        finally:
            mgr._agent_model = old


class TestUnloadAll:
    @patch("gc.collect")
    def test_unloads_all_models(self, mock_gc):
        import pile.models.manager as mgr

        originals = (mgr._agent_model, mgr._router_model, mgr._embed_model)
        try:
            mgr._agent_model = MagicMock()
            mgr._router_model = MagicMock()
            mgr._embed_model = MagicMock()
            mgr.unload_all()
            assert mgr._agent_model is None
            assert mgr._router_model is None
            assert mgr._embed_model is None
            mock_gc.assert_called_once()
        finally:
            mgr._agent_model, mgr._router_model, mgr._embed_model = originals


# ---------------------------------------------------------------------------
# ensure_models
# ---------------------------------------------------------------------------

class TestEnsureModels:
    @patch("pile.models.manager.get_embed_model")
    @patch("pile.models.manager.get_router_model")
    @patch("pile.models.manager.get_agent_model")
    @patch("pile.models.manager.download_models")
    @patch("pile.models.manager.get_missing_models")
    def test_downloads_missing_then_loads(self, mock_missing, mock_dl,
                                           mock_agent, mock_router, mock_embed):
        from pile.models.manager import ensure_models

        mock_missing.return_value = ["agent"]
        ensure_models()
        mock_dl.assert_called_once_with(["agent"])
        mock_agent.assert_called_once()
        mock_router.assert_called_once()
        mock_embed.assert_called_once()

    @patch("pile.models.manager.get_embed_model")
    @patch("pile.models.manager.get_router_model")
    @patch("pile.models.manager.get_agent_model")
    @patch("pile.models.manager.download_models")
    @patch("pile.models.manager.get_missing_models")
    def test_no_missing_skips_download(self, mock_missing, mock_dl,
                                        mock_agent, mock_router, mock_embed):
        from pile.models.manager import ensure_models

        mock_missing.return_value = []
        ensure_models()
        mock_dl.assert_not_called()
        mock_agent.assert_called_once()


# ---------------------------------------------------------------------------
# get_*_model singletons
# ---------------------------------------------------------------------------

class TestGetModelSingletons:
    @patch("pile.models.manager.load_model")
    def test_get_agent_model_caches(self, mock_load):
        import pile.models.manager as mgr

        fake = MagicMock()
        mock_load.return_value = fake
        old = mgr._agent_model
        try:
            mgr._agent_model = None
            result = mgr.get_agent_model()
            assert result is fake
            mock_load.assert_called_once_with("agent")
            result2 = mgr.get_agent_model()
            assert result2 is fake
            assert mock_load.call_count == 1
        finally:
            mgr._agent_model = old

    @patch("pile.models.manager.load_model")
    def test_get_router_model_caches(self, mock_load):
        import pile.models.manager as mgr

        fake = MagicMock()
        mock_load.return_value = fake
        old = mgr._router_model
        try:
            mgr._router_model = None
            result = mgr.get_router_model()
            assert result is fake
            mock_load.assert_called_once_with("router")
        finally:
            mgr._router_model = old

    @patch("pile.models.manager.load_model")
    def test_get_embed_model_caches(self, mock_load):
        import pile.models.manager as mgr

        fake = MagicMock()
        mock_load.return_value = fake
        old = mgr._embed_model
        try:
            mgr._embed_model = None
            result = mgr.get_embed_model()
            assert result is fake
            mock_load.assert_called_once_with("embedding")
        finally:
            mgr._embed_model = old
