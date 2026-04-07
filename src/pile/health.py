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
