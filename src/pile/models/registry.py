"""Fixed model registry — maps each role to a specific GGUF on HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = os.path.expanduser("~/.pile/models")

MODELS: dict[str, dict] = {
    "agent": {
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "size_gb": 2.55,
    },
    "router": {
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "size_gb": 2.89,
    },
    "embedding": {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "filename": "nomic-embed-text-v1.5.Q8_0.gguf",
        "size_gb": 0.14,
    },
}


def get_model_path(role: str) -> Path:
    """Return the local file path for a model role. Raises KeyError if role unknown."""
    info = MODELS[role]
    return Path(MODELS_DIR) / role / info["filename"]
