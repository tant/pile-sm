# Self-Contained Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Ollama/LM Studio dependency with built-in llama-cpp-python inference. App downloads GGUF models from HuggingFace and runs them locally.

**Architecture:** New `src/pile/models/` module handles model registry, download, loading, and inference. A custom `LlamaCppClient` implements MAF's `BaseChatClient` interface so the agent/workflow layer stays untouched. `client.py`, `memory/store.py`, `config.py`, and `health.py` are updated to use the new local engine.

**Tech Stack:** llama-cpp-python (inference), huggingface-hub (download), rich (progress bars)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/pile/models/__init__.py` | Create | Package init |
| `src/pile/models/registry.py` | Create | Fixed model definitions (repo, filename, size) |
| `src/pile/models/manager.py` | Create | Download from HF + load llama-cpp instances |
| `src/pile/models/engine.py` | Create | Inference functions (chat, router, embed) |
| `src/pile/models/llm_client.py` | Create | MAF-compatible LlamaCppClient |
| `src/pile/models/logging.py` | Create | Inference logger setup (file rotation, levels) |
| `src/pile/config.py` | Modify | Remove LLM provider fields, add max_tokens + log settings |
| `src/pile/client.py` | Modify | Use LlamaCppClient + local router |
| `src/pile/memory/store.py` | Modify | Use local embedding function |
| `src/pile/health.py` | Modify | Check model files instead of endpoints |
| `pyproject.toml` | Modify | Swap dependencies |
| `.env` | Modify | Remove LLM config, add max_tokens + log_level |
| `tests/test_registry.py` | Create | Registry tests |
| `tests/test_manager.py` | Create | Manager tests |
| `tests/test_engine.py` | Create | Engine tests |
| `tests/test_llm_client.py` | Create | LlamaCppClient tests |
| `tests/test_logging_inference.py` | Create | Logging tests |

---

### Task 1: Dependencies — pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update dependencies**

Replace LLM provider packages with local inference packages:

```toml
dependencies = [
    "llama-cpp-python>=0.3",
    "huggingface-hub>=0.25",
    "rich>=13.0",
    "agent-framework-orchestrations>=1.0.0b260402",
    "chainlit>=2.10.1",
    "httpx>=0.27",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
    "chromadb>=0.6",
    "pymupdf>=1.25",
    "playwright>=1.40",
    "plotly>=5.0",
]
```

Removed: `agent-framework-ollama`, `agent-framework-openai`.
Added: `llama-cpp-python`, `huggingface-hub`, `rich`.

- [ ] **Step 2: Install dependencies**

Run: `cd /Users/tantran/works/gg && uv sync`
Expected: dependencies resolve and install successfully.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: swap ollama/openai for llama-cpp-python and huggingface-hub"
```

---

### Task 2: Config — remove LLM provider fields, add new settings

**Files:**
- Modify: `src/pile/config.py`
- Modify: `.env`
- Test: `tests/test_config.py` (if exists, otherwise skip)

- [ ] **Step 1: Write test for new config fields**

Create `tests/test_config.py`:

```python
"""Tests for config changes — new fields, removed fields."""

import os
import pytest


def test_new_fields_have_defaults():
    """New config fields should have sensible defaults."""
    from pile.config import Settings

    s = Settings(
        _env_file=None,
        jira_base_url="https://x.atlassian.net",
        jira_email="a@b.com",
        jira_api_token="tok",
        jira_project_key="X",
    )
    assert s.agent_max_tokens == 32768
    assert s.router_max_tokens == 4096
    assert s.log_level == "INFO"
    assert s.log_dir == "~/.pile/logs"


def test_removed_fields_absent():
    """Old LLM provider fields should no longer exist."""
    from pile.config import Settings

    s = Settings(
        _env_file=None,
        jira_base_url="https://x.atlassian.net",
        jira_email="a@b.com",
        jira_api_token="tok",
        jira_project_key="X",
    )
    assert not hasattr(s, "llm_provider")
    assert not hasattr(s, "ollama_host")
    assert not hasattr(s, "ollama_model_id")
    assert not hasattr(s, "openai_base_url")
    assert not hasattr(s, "openai_model")
    assert not hasattr(s, "openai_api_key")
    assert not hasattr(s, "router_model")
    assert not hasattr(s, "embedding_model_id")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_config.py -v`
Expected: FAIL — old fields still exist, new fields missing.

- [ ] **Step 3: Update config.py**

Replace the Settings class in `src/pile/config.py`. Remove all LLM provider fields. Add new fields:

```python
class Settings(BaseSettings):
    # Model context limits
    agent_max_tokens: int = 32768
    router_max_tokens: int = 4096

    # Logging
    log_level: str = "INFO"
    log_dir: str = "~/.pile/logs"

    # Jira
    jira_base_url: str = "https://your-instance.atlassian.net"
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # Git repositories
    git_repos: str = ""
    git_repos_json: str = ""

    # Memory / RAG
    memory_enabled: bool = True
    memory_store_path: str = "~/.pile/chromadb"

    # Agent limits
    agent_max_iterations: int = 5
    agent_max_function_calls: int = 15

    # Default board
    default_board_id: int = 0

    # Browser
    browser_enabled: bool = True
    browser_profile_path: str = "~/.pile/browser"
    browser_jira_email: str = ""
    browser_jira_password: str = ""
    browser_github_username: str = ""
    browser_github_password: str = ""
    browser_gitlab_username: str = ""
    browser_gitlab_password: str = ""

    # UI
    chainlit_host: str = "0.0.0.0"
    chainlit_port: int = 8000

    # ... keep all existing properties (git_repo_list, git_repo_paths, get_git_repo)
    # ... keep model_config

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

Fields removed: `llm_provider`, `ollama_host`, `ollama_model_id`, `openai_base_url`, `openai_model`, `openai_api_key`, `router_model`, `embedding_model_id`.

- [ ] **Step 4: Update .env**

Remove LLM provider section. Add new fields:

```env
# --- Model Context Limits ---
AGENT_MAX_TOKENS=32768
ROUTER_MAX_TOKENS=4096

# --- Logging ---
LOG_LEVEL=INFO
LOG_DIR=~/.pile/logs
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pile/config.py .env tests/test_config.py
git commit -m "config: remove LLM provider fields, add model context limits and logging"
```

---

### Task 3: Model Registry

**Files:**
- Create: `src/pile/models/__init__.py`
- Create: `src/pile/models/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Create package init**

Create `src/pile/models/__init__.py`:

```python
"""Local model management — download, load, and run GGUF models."""
```

- [ ] **Step 2: Write registry tests**

Create `tests/test_registry.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_registry.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement registry**

Create `src/pile/models/registry.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/pile/models/__init__.py src/pile/models/registry.py tests/test_registry.py
git commit -m "feat: add model registry with fixed GGUF model definitions"
```

---

### Task 4: Model Manager — Download & Load

**Files:**
- Create: `src/pile/models/manager.py`
- Test: `tests/test_manager.py`

- [ ] **Step 1: Write manager tests**

Create `tests/test_manager.py`:

```python
"""Tests for model manager — download detection and model loading."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pile.models.registry import MODELS, get_model_path


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_manager.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement manager**

Create `src/pile/models/manager.py`:

```python
"""Model manager — download GGUF models from HuggingFace and load via llama-cpp."""

from __future__ import annotations

import logging
import os
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
        # macOS — Apple Silicon Metal
        return -1
    if system == "Linux":
        # Check for CUDA
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                return -1
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/models/manager.py tests/test_manager.py
git commit -m "feat: add model manager with download and load lifecycle"
```

---

### Task 5: Inference Logging

**Files:**
- Create: `src/pile/models/logging.py`
- Test: `tests/test_logging_inference.py`

- [ ] **Step 1: Write logging tests**

Create `tests/test_logging_inference.py`:

```python
"""Tests for inference logging setup."""

import logging
import os
from unittest.mock import patch

import pytest


def test_setup_creates_log_dir(tmp_path):
    """Logger setup should create log directory."""
    from pile.models.logging import setup_inference_logger

    log_dir = str(tmp_path / "logs")
    with patch("pile.config.settings") as mock_settings:
        mock_settings.log_level = "INFO"
        mock_settings.log_dir = log_dir
        logger = setup_inference_logger()

    assert os.path.isdir(log_dir)
    assert logger.name == "pile.inference"


def test_log_level_from_config(tmp_path):
    """Logger should use level from config."""
    from pile.models.logging import setup_inference_logger

    with patch("pile.config.settings") as mock_settings:
        mock_settings.log_level = "DEBUG"
        mock_settings.log_dir = str(tmp_path)
        logger = setup_inference_logger()

    assert logger.level == logging.DEBUG


def test_log_inference_call(tmp_path, caplog):
    """log_inference_call should log at INFO level."""
    from pile.models.logging import setup_inference_logger, log_inference_call

    with patch("pile.config.settings") as mock_settings:
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_logging_inference.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement logging module**

Create `src/pile/models/logging.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_logging_inference.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/models/logging.py tests/test_logging_inference.py
git commit -m "feat: add inference logging with rotation and structured output"
```

---

### Task 6: Inference Engine

**Files:**
- Create: `src/pile/models/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write engine tests**

Create `tests/test_engine.py`:

```python
"""Tests for inference engine functions."""

from unittest.mock import patch, MagicMock

import pytest


def _mock_llama_response(content="hello", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a fake llama-cpp response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "test-id",
        "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@patch("pile.models.engine._get_agent_model")
def test_chat_completion_basic(mock_get):
    """chat_completion should return response dict with content."""
    from pile.models.engine import chat_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("test answer")
    mock_get.return_value = mock_model

    result = chat_completion([{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "test answer"
    mock_model.create_chat_completion.assert_called_once()


@patch("pile.models.engine._get_router_model")
def test_router_completion(mock_get):
    """router_completion should return stripped text."""
    from pile.models.engine import router_completion

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = _mock_llama_response("jira_query")
    mock_get.return_value = mock_model

    result = router_completion("Pick an agent for: find issues")

    assert result == "jira_query"


@patch("pile.models.engine._get_embed_model")
def test_embed(mock_get):
    """embed should return list of float vectors."""
    from pile.models.engine import embed

    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_get.return_value = mock_model

    result = embed(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_engine.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement engine**

Create `src/pile/models/engine.py`:

```python
"""Inference engine — wraps llama-cpp-python for chat, routing, and embedding."""

from __future__ import annotations

import json
import time
from typing import Any

from pile.models.logging import log_inference_call, log_inference_detail
from pile.models.manager import get_agent_model, get_router_model, get_embed_model


def _get_agent_model():
    return get_agent_model()


def _get_router_model():
    return get_router_model()


def _get_embed_model():
    return get_embed_model()


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict:
    """Run chat completion on the agent model. Returns OpenAI-compatible dict."""
    model = _get_agent_model()
    start = time.monotonic()

    log_inference_detail(
        role="agent",
        direction="request",
        content=json.dumps({"messages": messages, "tools": [t["function"]["name"] for t in tools] if tools else None}, ensure_ascii=False, indent=2),
    )

    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    result = model.create_chat_completion(**kwargs)

    latency_ms = int((time.monotonic() - start) * 1000)
    usage = result.get("usage", {})
    choice = result["choices"][0] if result.get("choices") else {}
    message = choice.get("message", {})
    tool_call_count = len(message.get("tool_calls", []))

    log_inference_call(
        role="agent",
        latency_ms=latency_ms,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        status="ok",
        tool_calls=tool_call_count if tool_call_count else None,
    )

    log_inference_detail(
        role="agent",
        direction="response",
        content=json.dumps(message, ensure_ascii=False, indent=2),
    )

    return result


def router_completion(prompt: str, max_tokens: int = 20) -> str | None:
    """Run classification on the router model. Returns response text or None."""
    model = _get_router_model()
    start = time.monotonic()

    log_inference_detail(role="router", direction="request", content=prompt)

    try:
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        usage = result.get("usage", {})
        text = result["choices"][0]["message"]["content"].strip()

        log_inference_call(
            role="router",
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            status="ok",
        )
        log_inference_detail(role="router", direction="response", content=text)

        return text
    except Exception as e:
        latency_ms = int((time.monotonic() - start) * 1000)
        log_inference_call(role="router", latency_ms=latency_ms, status="error", error=str(e))
        return None


def embed(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = _get_embed_model()
    start = time.monotonic()

    result = model.embed(texts)

    latency_ms = int((time.monotonic() - start) * 1000)
    log_inference_call(role="embedding", latency_ms=latency_ms, status="ok")

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/models/engine.py tests/test_engine.py
git commit -m "feat: add inference engine wrapping llama-cpp-python"
```

---

### Task 7: MAF-Compatible LlamaCppClient

**Files:**
- Create: `src/pile/models/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write client tests**

Create `tests/test_llm_client.py`:

```python
"""Tests for LlamaCppClient — MAF compatibility."""

import asyncio
from unittest.mock import patch, MagicMock

import pytest


def _mock_llama_response(content="hello", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "id": "resp-1",
        "choices": [{"message": message, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "model": "test-model",
        "created": 1700000000,
    }


@patch("pile.models.engine.chat_completion")
def test_client_returns_chat_response(mock_chat):
    """LlamaCppClient should return a valid ChatResponse."""
    from pile.models.llm_client import LlamaCppClient
    from agent_framework import Message

    mock_chat.return_value = _mock_llama_response("test reply")

    client = LlamaCppClient()
    messages = [Message(role="user", contents="hello")]
    response = asyncio.get_event_loop().run_until_complete(
        client.get_response(messages)
    )

    assert response.messages is not None
    assert len(response.messages) > 0


@patch("pile.models.engine.chat_completion")
def test_client_handles_tool_calls(mock_chat):
    """LlamaCppClient should parse tool calls into Content objects."""
    from pile.models.llm_client import LlamaCppClient
    from agent_framework import Message

    mock_chat.return_value = _mock_llama_response(
        content=None,
        tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "jira_search", "arguments": '{"jql": "project=TETRA"}'},
        }],
    )

    client = LlamaCppClient()
    messages = [Message(role="user", contents="find issues")]
    response = asyncio.get_event_loop().run_until_complete(
        client.get_response(messages)
    )

    assert response.messages is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_llm_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement LlamaCppClient**

Create `src/pile/models/llm_client.py`:

```python
"""MAF-compatible LLM client wrapping local llama-cpp-python inference."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from agent_framework._tools import FunctionInvocationLayer
from agent_framework._clients import ChatMiddlewareLayer, ChatTelemetryLayer

from pile.models.engine import chat_completion


class LlamaCppClient(
    FunctionInvocationLayer,
    ChatMiddlewareLayer,
    ChatTelemetryLayer,
    BaseChatClient,
):
    """Local LLM client using llama-cpp-python, compatible with MAF orchestration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ChatResponse | ResponseStream[ChatResponseUpdate, ChatResponse]:
        """Convert MAF messages to llama-cpp format, call engine, convert back."""
        llama_messages = self._convert_messages(messages)
        tools = self._convert_tools(options.get("tools"))

        if stream:
            # Streaming not supported with tool calling in llama-cpp
            # Fall back to non-streaming
            pass

        async def _get() -> ChatResponse:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: chat_completion(llama_messages, tools=tools),
            )
            return self._parse_response(result)

        return _get()

    def _convert_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Convert MAF Message objects to llama-cpp dict format."""
        converted = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role}

            # Extract text content
            text_parts = []
            tool_calls = []
            tool_result = None

            for content in (msg.contents or []):
                if isinstance(content, str):
                    text_parts.append(content)
                elif hasattr(content, "type"):
                    if content.type == "text":
                        text_parts.append(content.text)
                    elif content.type == "function_call":
                        tool_calls.append({
                            "id": content.call_id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": content.arguments if isinstance(content.arguments, str) else str(content.arguments),
                            },
                        })
                    elif content.type == "function_result":
                        tool_result = content.result if isinstance(content.result, str) else str(content.result)

            if msg.role == "tool" and tool_result is not None:
                entry["content"] = tool_result
                # Find the call_id from function_result content
                for content in (msg.contents or []):
                    if hasattr(content, "type") and content.type == "function_result":
                        entry["tool_call_id"] = content.call_id
                        break
            elif tool_calls:
                entry["content"] = "\n".join(text_parts) if text_parts else None
                entry["tool_calls"] = tool_calls
            else:
                entry["content"] = "\n".join(text_parts) if text_parts else ""

            converted.append(entry)
        return converted

    def _convert_tools(self, tools: Any) -> list[dict] | None:
        """Convert MAF tool definitions to OpenAI-compatible format."""
        if not tools:
            return None
        converted = []
        for tool in tools:
            if hasattr(tool, "to_dict"):
                converted.append(tool.to_dict())
            elif isinstance(tool, dict):
                converted.append(tool)
        return converted if converted else None

    def _parse_response(self, result: dict) -> ChatResponse:
        """Convert llama-cpp response dict to MAF ChatResponse."""
        choice = result["choices"][0]
        message = choice["message"]
        usage = result.get("usage", {})

        contents: list[Content] = []

        # Text content
        if message.get("content"):
            contents.append(Content.from_text(text=message["content"]))

        # Tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                contents.append(Content.from_function_call(
                    call_id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ))

        finish_reason = choice.get("finish_reason", "stop")
        if message.get("tool_calls"):
            finish_reason = "tool_calls"

        return ChatResponse(
            messages=[Message(role="assistant", contents=contents)],
            response_id=result.get("id", str(uuid.uuid4())),
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            usage_details={
                "input_token_count": usage.get("prompt_tokens", 0),
                "output_token_count": usage.get("completion_tokens", 0),
                "total_token_count": usage.get("total_tokens", 0),
            },
            model="local-llama-cpp",
            finish_reason=finish_reason,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/test_llm_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pile/models/llm_client.py tests/test_llm_client.py
git commit -m "feat: add MAF-compatible LlamaCppClient"
```

---

### Task 8: Integrate — client.py

**Files:**
- Modify: `src/pile/client.py`

- [ ] **Step 1: Rewrite client.py**

Replace the entire `src/pile/client.py` with:

```python
"""LLM client factory — uses local llama-cpp-python inference."""

from __future__ import annotations

from pile.models.llm_client import LlamaCppClient
from pile.models.engine import router_completion

from pile.config import settings


def create_client() -> LlamaCppClient:
    """Create a LlamaCppClient with function invocation limits from config."""
    client = LlamaCppClient()
    client.function_invocation_configuration["max_iterations"] = settings.agent_max_iterations
    client.function_invocation_configuration["max_function_calls"] = settings.agent_max_function_calls
    return client


def call_router_model(prompt: str, max_tokens: int = 20) -> str | None:
    """Call the lightweight router model for classification/compression."""
    return router_completion(prompt, max_tokens=max_tokens)
```

- [ ] **Step 2: Verify imports work**

Run: `cd /Users/tantran/works/gg && uv run python -c "from pile.client import create_client, call_router_model; print('OK')"`
Expected: `OK` (may warn about models not downloaded, that's fine)

- [ ] **Step 3: Commit**

```bash
git add src/pile/client.py
git commit -m "refactor: client.py uses local LlamaCppClient instead of external providers"
```

---

### Task 9: Integrate — memory/store.py

**Files:**
- Modify: `src/pile/memory/store.py`

- [ ] **Step 1: Replace embedding function**

In `src/pile/memory/store.py`, replace `_embedding_fn()` to use local engine:

```python
def _embedding_fn() -> EmbeddingFunction:
    """Return a cached embedding function using local llama-cpp inference."""
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn

    from pile.models.engine import embed as local_embed

    class LocalEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: list[str]) -> list[list[float]]:
            return local_embed(input)

    _embed_fn = LocalEmbeddingFunction()
    return _embed_fn
```

Remove the old Ollama/OpenAI imports (`OllamaEmbeddingFunction`, `OpenAIEmbeddingFunction`).
Remove the `settings.llm_provider` branching logic.

- [ ] **Step 2: Verify import works**

Run: `cd /Users/tantran/works/gg && uv run python -c "from pile.memory.store import _embedding_fn; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pile/memory/store.py
git commit -m "refactor: memory store uses local embedding instead of Ollama/OpenAI API"
```

---

### Task 10: Integrate — health.py

**Files:**
- Modify: `src/pile/health.py`

- [ ] **Step 1: Rewrite health.py**

Replace `src/pile/health.py` — remove all Ollama/OpenAI checks, add model file checks:

```python
"""Health checks for dependencies."""

from __future__ import annotations

import httpx

from pile.config import settings
from pile.models.registry import MODELS, get_model_path


def check_models() -> str | None:
    """Check if all required model files are downloaded."""
    missing = []
    for role in MODELS:
        if not get_model_path(role).exists():
            missing.append(role)
    if missing:
        return f"Missing model files for: {', '.join(missing)}. Run `pile` to download."
    return None


def check_jira() -> str | None:
    """Check if Jira is reachable with valid credentials."""
    if not settings.jira_email or not settings.jira_api_token:
        return "JIRA_EMAIL and JIRA_API_TOKEN must be set in .env"
    try:
        resp = httpx.get(
            f"{settings.jira_base_url}/rest/api/3/myself",
            auth=(settings.jira_email, settings.jira_api_token),
            headers={"Accept": "application/json"},
            timeout=10.0,
        )
        if resp.status_code == 401:
            return "Jira authentication failed. Check JIRA_EMAIL and JIRA_API_TOKEN."
        if resp.status_code == 403:
            return "Jira access forbidden. Check your API token permissions."
        resp.raise_for_status()
        return None
    except httpx.ConnectError:
        return f"Cannot connect to Jira at {settings.jira_base_url}."
    except Exception as e:
        return f"Jira health check failed: {e}"


def check_browser() -> str | None:
    """Check if Playwright Firefox browser is installed."""
    if not settings.browser_enabled:
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-c", "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); p.firefox; p.stop()"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return "Playwright Firefox not installed. Run: playwright install firefox"
        return None
    except FileNotFoundError:
        return "Playwright not installed. Run: uv sync && playwright install firefox"
    except Exception as e:
        return f"Browser health check failed: {e}"


def run_health_checks() -> list[str]:
    """Run all health checks."""
    errors = []

    err = check_models()
    if err:
        errors.append(err)

    err = check_jira()
    if err:
        errors.append(err)

    if settings.browser_enabled:
        err = check_browser()
        if err:
            errors.append(err)

    return errors
```

- [ ] **Step 2: Verify import works**

Run: `cd /Users/tantran/works/gg && uv run python -c "from pile.health import run_health_checks; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/pile/health.py
git commit -m "refactor: health checks verify model files instead of external endpoints"
```

---

### Task 11: Integrate — startup (ensure_models on first run)

**Files:**
- Modify: `src/pile/ui/cli.py`
- Modify: `src/pile/ui/chainlit_app.py`

- [ ] **Step 1: Read current startup code**

Read `src/pile/ui/cli.py` and `src/pile/ui/chainlit_app.py` to find where startup/initialization happens.

- [ ] **Step 2: Add ensure_models() call at startup**

In both entry points, add before any agent/workflow initialization:

```python
from pile.models.manager import ensure_models

# At the start of main() or startup function:
ensure_models()  # Downloads missing models on first run, then loads all
```

This must run before `create_client()` or any agent creation.

- [ ] **Step 3: Add logging setup at startup**

```python
from pile.models.logging import setup_inference_logger

setup_inference_logger()
```

- [ ] **Step 4: Test startup manually**

Run: `cd /Users/tantran/works/gg && uv run pile --help`
Expected: Should work without Ollama/LM Studio running (models need to be downloaded first time).

- [ ] **Step 5: Commit**

```bash
git add src/pile/ui/cli.py src/pile/ui/chainlit_app.py
git commit -m "feat: auto-download models on first startup"
```

---

### Task 12: Update .env.sample and cleanup

**Files:**
- Modify: `.env` (remove old LLM fields if not done in Task 2)
- Create or modify: `.env.sample`

- [ ] **Step 1: Create .env.sample**

Create `.env.sample` reflecting the new config (no LLM provider fields):

```env
# ============================================================
# Pile Configuration
# ============================================================

# --- Model Context Limits ---
AGENT_MAX_TOKENS=32768
ROUTER_MAX_TOKENS=4096

# --- Logging ---
# Levels: ERROR, WARNING, INFO, DEBUG
# DEBUG logs full LLM prompts and responses for troubleshooting
LOG_LEVEL=INFO
LOG_DIR=~/.pile/logs

# --- Jira ---
JIRA_BASE_URL=https://your-instance.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your-api-token
JIRA_PROJECT_KEY=YOUR_KEY

# --- Git (optional) ---
# Simple: comma-separated paths
# GIT_REPOS=/path/to/repo1,/path/to/repo2
# Advanced: JSON with credentials
# GIT_REPOS_JSON=[{"path":"/repo","url":"https://...","token":"ghp_xxx"}]

# --- Memory / RAG ---
MEMORY_ENABLED=true
MEMORY_STORE_PATH=~/.pile/chromadb

# --- Browser (optional) ---
BROWSER_ENABLED=true
BROWSER_PROFILE_PATH=~/.pile/browser
# BROWSER_JIRA_EMAIL=
# BROWSER_JIRA_PASSWORD=
# BROWSER_GITHUB_USERNAME=
# BROWSER_GITHUB_PASSWORD=

# --- UI ---
CHAINLIT_HOST=0.0.0.0
CHAINLIT_PORT=8000
```

- [ ] **Step 2: Update .env — remove old fields, add new ones**

Remove from `.env`:
- `LLM_PROVIDER`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`
- `OLLAMA_HOST`, `OLLAMA_MODEL_ID` (commented or not)
- `ROUTER_MODEL`, `EMBEDDING_MODEL_ID`

Add to `.env`:
- `AGENT_MAX_TOKENS=32768`
- `ROUTER_MAX_TOKENS=4096`
- `LOG_LEVEL=INFO`
- `LOG_DIR=~/.pile/logs`

- [ ] **Step 3: Commit**

```bash
git add .env .env.sample
git commit -m "config: update .env and .env.sample for self-contained inference"
```

---

### Task 13: End-to-end smoke test

**Files:** None (testing only)

- [ ] **Step 1: Run all unit tests**

Run: `cd /Users/tantran/works/gg && uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run ruff**

Run: `cd /Users/tantran/works/gg && uv run ruff check src/pile/`
Expected: No errors.

- [ ] **Step 3: Test CLI startup (without models — should trigger download)**

Run: `cd /Users/tantran/works/gg && uv run pile`
Expected: If models not present, shows download progress. If present, starts normally.

- [ ] **Step 4: Test a simple query**

In the running app, try: `Sprint hiện tại thế nào?`
Expected: Routes to scrum agent, calls Jira tools, returns response — all using local inference.

- [ ] **Step 5: Check inference log**

Run: `cat ~/.pile/logs/inference.log`
Expected: See INFO lines with role, latency, token counts for each LLM call made.

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: smoke test fixes for self-contained inference"
```
