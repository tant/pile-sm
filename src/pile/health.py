"""Health checks for external dependencies."""

from __future__ import annotations

import httpx

from pile.config import settings


def check_ollama() -> str | None:
    """Check if Ollama server is reachable. Returns error message or None."""
    host = settings.ollama_host
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if settings.ollama_model_id not in models:
            return f"Model '{settings.ollama_model_id}' not found on {host}. Available: {', '.join(models)}"
        return None
    except httpx.ConnectError:
        return f"Cannot connect to Ollama at {host}. Is the server running?"
    except Exception as e:
        return f"Ollama health check failed: {e}"


def check_jira() -> str | None:
    """Check if Jira is reachable with valid credentials. Returns error message or None."""
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
    """Check if the embedding model is available on Ollama. Returns error message or None."""
    if not settings.memory_enabled:
        return None
    host = settings.embedding_ollama_host
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if settings.embedding_model_id not in models:
            return (
                f"Embedding model '{settings.embedding_model_id}' not found on {host}. "
                f"Run: ollama pull {settings.embedding_model_id}"
            )
        return None
    except httpx.ConnectError:
        return f"Cannot check embedding model — Ollama at {host} unreachable."
    except Exception as e:
        return f"Embedding model health check failed: {e}"


def check_browser() -> str | None:
    """Check if Playwright Firefox browser is installed. Returns error message or None."""
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
    """Run all health checks. Returns list of error messages (empty = all OK)."""
    errors = []

    # Only check Ollama if using ollama provider
    if settings.llm_provider in ("ollama", "ollama-native"):
        err = check_ollama()
        if err:
            errors.append(err)

    err = check_jira()
    if err:
        errors.append(err)

    err = check_embedding_model()
    if err:
        errors.append(err)

    err = check_browser()
    if err:
        errors.append(err)

    return errors
