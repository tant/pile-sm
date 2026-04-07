"""Model manager — download GGUF models from HuggingFace and load via llama-cpp."""

from __future__ import annotations

import logging
import platform
from pathlib import Path

from llama_cpp import Llama

from pile.config import settings
from pile.models.registry import MODELS, MODELS_DIR, get_model_path

logger = logging.getLogger("pile.models")

# Singleton model instances
_agent_model: Llama | None = None
_router_model: Llama | None = None
_embed_model: Llama | None = None


def is_model_downloaded(role: str) -> bool:
    """Check if a model file exists locally."""
    return get_model_path(role).exists()


def get_missing_models() -> list[str]:
    """Return list of model roles that haven't been downloaded yet."""
    return [role for role in MODELS if not is_model_downloaded(role)]


def download_models(roles: list[str] | None = None) -> None:
    """Download missing models from HuggingFace with progress display."""
    from huggingface_hub import hf_hub_download
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

    if roles is None:
        roles = get_missing_models()
    if not roles:
        return

    total = len(roles)
    logger.info("Downloading %d model(s) for first-time setup...", total)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
    ) as progress:
        for i, role in enumerate(roles, 1):
            info = MODELS[role]
            dest_dir = Path(MODELS_DIR) / role
            dest_dir.mkdir(parents=True, exist_ok=True)

            task = progress.add_task(
                f"[{i}/{total}] {info['filename']} ({info['size_gb']:.1f} GB)",
                total=None,
            )
            hf_hub_download(
                repo_id=info["repo"],
                filename=info["filename"],
                local_dir=str(dest_dir),
            )
            progress.update(task, completed=True)

    logger.info("All models downloaded.")


def _detect_gpu_layers() -> int:
    """Auto-detect GPU availability. Returns n_gpu_layers value."""
    system = platform.system()
    if system == "Darwin":
        return -1  # macOS Metal
    if system == "Linux":
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return -1  # CUDA available
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return 0  # CPU fallback


def load_model(role: str) -> Llama:
    """Load a GGUF model into memory. Uses appropriate settings per role."""
    path = get_model_path(role)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run setup first.")

    gpu_layers = _detect_gpu_layers()
    logger.info("Loading %s model from %s (gpu_layers=%d)", role, path, gpu_layers)

    if role == "agent":
        return Llama(
            model_path=str(path),
            n_ctx=settings.agent_max_tokens,
            n_gpu_layers=gpu_layers,
            chat_format="chatml-function-calling",
            verbose=False,
        )
    elif role == "router":
        return Llama(
            model_path=str(path),
            n_ctx=settings.router_max_tokens,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
    elif role == "embedding":
        return Llama(
            model_path=str(path),
            n_ctx=2048,
            n_gpu_layers=gpu_layers,
            embedding=True,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model role: {role}")


def get_agent_model() -> Llama:
    """Return the singleton agent model, loading if needed."""
    global _agent_model
    if _agent_model is None:
        _agent_model = load_model("agent")
    return _agent_model


def get_router_model() -> Llama:
    """Return the singleton router model, loading if needed."""
    global _router_model
    if _router_model is None:
        _router_model = load_model("router")
    return _router_model


def get_embed_model() -> Llama:
    """Return the singleton embedding model, loading if needed."""
    global _embed_model
    if _embed_model is None:
        _embed_model = load_model("embedding")
    return _embed_model


def ensure_models() -> None:
    """Download missing models, then load all. Called on app startup."""
    missing = get_missing_models()
    if missing:
        download_models(missing)
    get_agent_model()
    get_router_model()
    get_embed_model()
    logger.info("All models loaded and ready.")
