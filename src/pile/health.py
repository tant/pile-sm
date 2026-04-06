"""Health checks for external dependencies."""

from __future__ import annotations

import httpx

from pile.config import settings


def check_ollama() -> str | None:
    """Check if Ollama server is reachable and model available."""
    host = settings.ollama_host
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=15.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        model_id = settings.ollama_model_id
        if model_id not in models and f"{model_id}:latest" not in models:
            return f"Model '{model_id}' not found on {host}. Available: {', '.join(models)}"
        return None
    except httpx.ConnectError:
        return f"Cannot connect to Ollama at {host}. Is the server running?"
    except Exception as e:
        return f"Ollama health check failed: {e}"


def _list_openai_models() -> list[str] | None:
    """Fetch model IDs from the OpenAI-compatible endpoint. Returns None on error."""
    base_url = settings.openai_base_url.rstrip("/")
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=15.0,
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return None


def check_openai() -> str | None:
    """Check if OpenAI-compatible endpoint is reachable and required models are loaded."""
    base_url = settings.openai_base_url.rstrip("/")
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=15.0,
        )
        if resp.status_code == 401:
            return f"OpenAI endpoint auth failed at {base_url}. Check OPENAI_API_KEY."
        resp.raise_for_status()
        loaded = [m["id"] for m in resp.json().get("data", [])]
        model_id = settings.openai_model
        if model_id not in loaded:
            return (
                f"LLM model '{model_id}' not loaded at {base_url}. "
                f"Available: {', '.join(loaded)}"
            )
        return None
    except httpx.ConnectError:
        return f"Cannot connect to OpenAI endpoint at {base_url}. Is the server running?"
    except Exception as e:
        return f"OpenAI health check failed: {e}"


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


def check_embedding_model() -> str | None:
    """Check if the embedding model is available on the configured provider."""
    if not settings.memory_enabled:
        return None

    model_id = settings.embedding_model_id

    if settings.llm_provider == "openai":
        base_url = settings.openai_base_url.rstrip("/")
        loaded = _list_openai_models()
        if loaded is None:
            return f"Cannot check embedding model — endpoint at {base_url} unreachable."
        if model_id not in loaded:
            return (
                f"Embedding model '{model_id}' not loaded at {base_url}. "
                f"Available: {', '.join(loaded)}"
            )
        return None

    # Ollama provider: check /api/tags
    host = settings.ollama_host
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=15.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_id not in models and f"{model_id}:latest" not in models:
            return (
                f"Embedding model '{model_id}' not found on {host}. "
                f"Run: ollama pull {model_id}"
            )
        return None
    except httpx.ConnectError:
        return f"Cannot check embedding model — Ollama at {host} unreachable."
    except Exception as e:
        return f"Embedding model health check failed: {e}"


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
    """Run all health checks based on configured providers."""
    errors = []

    # LLM — check the provider that's actually configured
    if settings.llm_provider in ("ollama", "ollama-native"):
        err = check_ollama()
        if err:
            errors.append(err)
    elif settings.llm_provider == "openai":
        err = check_openai()
        if err:
            errors.append(err)

    # Jira
    err = check_jira()
    if err:
        errors.append(err)

    # Router model — verify it's loaded if configured
    if settings.router_model:
        if settings.llm_provider == "openai":
            loaded = _list_openai_models()
            if loaded and settings.router_model not in loaded:
                errors.append(
                    f"Router model '{settings.router_model}' not loaded. "
                    f"Available: {', '.join(loaded)}"
                )
        elif settings.llm_provider in ("ollama", "ollama-native"):
            err = check_ollama()
            if err is None:
                # Ollama reachable, check model exists
                try:
                    resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=15.0)
                    models = [m["name"] for m in resp.json().get("models", [])]
                    mid = settings.router_model
                    if mid not in models and f"{mid}:latest" not in models:
                        errors.append(
                            f"Router model '{mid}' not found on Ollama. "
                            f"Run: ollama pull {mid}"
                        )
                except Exception:
                    pass

    # Embedding — check the configured provider
    if settings.memory_enabled:
        err = check_embedding_model()
        if err:
            errors.append(err)

    # Browser
    if settings.browser_enabled:
        err = check_browser()
        if err:
            errors.append(err)

    return errors
