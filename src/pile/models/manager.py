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


def _download_single(role: str) -> None:
    """Download a single model from HuggingFace. Supports auto-resume."""
    from huggingface_hub import hf_hub_download

    info = MODELS[role]
    dest_dir = Path(MODELS_DIR) / role
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s (%s, %.1f GB)...", role, info["filename"], info["size_gb"])
    hf_hub_download(
        repo_id=info["repo"],
        filename=info["filename"],
        local_dir=str(dest_dir),
    )
    logger.info("Downloaded %s.", role)


def download_models(roles: list[str] | None = None) -> None:
    """Download missing models from HuggingFace in parallel with progress display."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rich.console import Console

    if roles is None:
        roles = get_missing_models()
    if not roles:
        return

    total = len(roles)
    console = Console()
    console.print(f"[bold blue]Downloading {total} model(s) in parallel...[/]")
    for role in roles:
        info = MODELS[role]
        console.print(f"  - {role}: {info['filename']} ({info['size_gb']:.1f} GB)")

    with ThreadPoolExecutor(max_workers=len(roles)) as pool:
        futures = {pool.submit(_download_single, role): role for role in roles}
        for future in as_completed(futures):
            role = futures[future]
            try:
                future.result()
                console.print(f"  [green]✓[/] {role} done")
            except Exception as e:
                console.print(f"  [red]✗[/] {role} failed: {e}")
                raise

    console.print("[bold green]All models downloaded.[/]")


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


def unload_model(role: str) -> None:
    """Unload a model from memory. Next get_*_model() call will reload it."""
    global _agent_model, _router_model, _embed_model
    if role == "agent" and _agent_model is not None:
        del _agent_model
        _agent_model = None
        logger.info("Unloaded agent model")
    elif role == "router" and _router_model is not None:
        del _router_model
        _router_model = None
        logger.info("Unloaded router model")
    elif role == "embedding" and _embed_model is not None:
        del _embed_model
        _embed_model = None
        logger.info("Unloaded embedding model")


def unload_all() -> None:
    """Unload all models from memory."""
    for role in ("agent", "router", "embedding"):
        unload_model(role)
    import gc
    gc.collect()
    logger.info("All models unloaded")


def ensure_models() -> None:
    """Download missing models, then load all. Called on app startup."""
    missing = get_missing_models()
    if missing:
        download_models(missing)
    get_agent_model()
    get_router_model()
    get_embed_model()
    logger.info("All models loaded and ready.")
