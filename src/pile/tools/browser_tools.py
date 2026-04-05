"""Browser tools using Playwright + Firefox for web scraping when APIs are unavailable.

Playwright sync API cannot run inside an asyncio event loop (which the agent framework uses).
Solution: run all Playwright operations in a dedicated background thread via _run_in_browser_thread().
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
from pathlib import Path
from typing import Annotated

from pydantic import Field

from agent_framework import tool
from pile.config import settings

logger = logging.getLogger("pile.tools.browser")

MAX_CONTENT = 4000

# Dedicated thread pool for Playwright (sync API needs its own non-async thread)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")

# Playwright state (lives in the background thread)
_playwright_instance = None
_browser_context = None
_current_page = None


def _run_in_browser_thread(fn, *args, **kwargs):
    """Run a function in the dedicated Playwright thread and wait for result."""
    future = _executor.submit(fn, *args, **kwargs)
    return future.result(timeout=60)


def _safe_browser_call(func):
    """Decorator to handle common browser errors gracefully."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception("Browser tool error in %s", func.__name__)
            return f"Error: {type(e).__name__}: {e}"
    return wrapper


# --- Browser lifecycle (singleton persistent context, runs in background thread) ---


def _get_context(headed: bool = False):
    """Get or create a persistent Firefox browser context. Must be called from browser thread."""
    global _playwright_instance, _browser_context

    profile_path = str(Path(settings.browser_profile_path).expanduser())
    os.makedirs(profile_path, exist_ok=True)

    if _browser_context is not None:
        return _browser_context

    from playwright.sync_api import sync_playwright

    if _playwright_instance is None:
        _playwright_instance = sync_playwright().start()

    _browser_context = _playwright_instance.firefox.launch_persistent_context(
        user_data_dir=profile_path,
        headless=not headed,
        viewport={"width": 1280, "height": 720},
        locale="en-US",
    )
    return _browser_context


def _get_page():
    """Get the current page, creating one if needed. Must be called from browser thread."""
    global _current_page
    ctx = _get_context()
    if _current_page is None or _current_page.is_closed():
        pages = ctx.pages
        _current_page = pages[0] if pages else ctx.new_page()
    return _current_page


def _close_context():
    """Close browser context. Must be called from browser thread."""
    global _browser_context, _current_page
    if _browser_context:
        try:
            _browser_context.close()
        except Exception:
            pass
        _browser_context = None
        _current_page = None


# --- Auto-login detection and handling ---


_LOGIN_PATTERNS = {
    "atlassian": {
        "url_contains": ["id.atlassian.com"],
        "email_selector": "#username",
        "password_selector": "#password",
        "submit_selector": "#login-submit",
        "email_key": "browser_jira_email",
        "password_key": "browser_jira_password",
        "two_step": True,
    },
    "github": {
        "url_contains": ["github.com/login", "github.com/session"],
        "email_selector": "#login_field",
        "password_selector": "#password",
        "submit_selector": "input[type='submit']",
        "email_key": "browser_github_username",
        "password_key": "browser_github_password",
        "two_step": False,
    },
    "gitlab": {
        "url_contains": ["/users/sign_in"],
        "email_selector": "#user_login",
        "password_selector": "#user_password",
        "submit_selector": "input[type='submit'], button[type='submit']",
        "email_key": "browser_gitlab_username",
        "password_key": "browser_gitlab_password",
        "two_step": False,
    },
}


def _detect_login_page(url: str) -> dict | None:
    """Check if URL matches a known login page pattern."""
    for pattern in _LOGIN_PATTERNS.values():
        for fragment in pattern["url_contains"]:
            if fragment in url:
                return pattern
    return None


def _try_auto_login(page, pattern: dict) -> bool:
    """Attempt to auto-login using credentials from .env. Returns True if attempted."""
    email = getattr(settings, pattern["email_key"], "")
    password = getattr(settings, pattern["password_key"], "")

    if not email or not password:
        return False

    try:
        if pattern.get("two_step"):
            page.fill(pattern["email_selector"], email)
            page.click(pattern["submit_selector"])
            page.wait_for_load_state("domcontentloaded", timeout=10000)
            page.wait_for_selector(pattern["password_selector"], timeout=10000)
            page.fill(pattern["password_selector"], password)
            page.click(pattern["submit_selector"])
        else:
            page.fill(pattern["email_selector"], email)
            page.fill(pattern["password_selector"], password)
            page.click(pattern["submit_selector"])

        page.wait_for_load_state("domcontentloaded", timeout=15000)
        logger.info("Auto-login successful for %s", page.url)
        return True
    except Exception as e:
        logger.warning("Auto-login failed: %s", e)
        return False


def _handle_login_if_needed(page) -> str | None:
    """Check if on a login page and attempt auto-login."""
    pattern = _detect_login_page(page.url)
    if pattern is None:
        return None

    if _try_auto_login(page, pattern):
        new_pattern = _detect_login_page(page.url)
        if new_pattern:
            return "Auto-login failed (still on login page). Use browser_login for manual login."
        return None

    return "Login required but no credentials configured. Use browser_login for manual login."


# --- Playwright operations (run in browser thread) ---


def _do_open(url: str, selector: str | None) -> str:
    page = _get_page()
    page.goto(url, wait_until="domcontentloaded", timeout=30000)

    login_msg = _handle_login_if_needed(page)
    if login_msg:
        return login_msg

    title = page.title()
    if selector:
        el = page.query_selector(selector)
        text = el.inner_text() if el else f"Selector '{selector}' not found on page."
    else:
        text = page.inner_text("body")

    text = text[:MAX_CONTENT]
    return f"**{title}**\nURL: {page.url}\n\n{text}"


def _do_read(selector: str) -> str:
    page = _get_page()
    el = page.query_selector(selector)
    if not el:
        return f"No element found for selector: {selector}"
    text = el.inner_text()
    return text[:MAX_CONTENT]


def _do_click(selector: str | None, text: str | None) -> str:
    page = _get_page()
    if text:
        page.get_by_text(text, exact=False).first.click()
    elif selector:
        page.click(selector)
    else:
        return "Error: provide either selector or text."
    page.wait_for_load_state("domcontentloaded", timeout=10000)
    return f"Clicked. Now on: {page.title()} ({page.url})"


def _do_fill(selector: str, value: str) -> str:
    page = _get_page()
    page.fill(selector, value)
    return f"Filled '{selector}' with value."


def _do_login(url: str) -> str:
    _close_context()
    ctx = _get_context(headed=True)
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    import sys
    print(f"\n  Browser opened at {url}")
    print("  Log in manually, then press Enter here when done...")
    sys.stdin.readline()

    final_url = page.url
    _close_context()
    return f"Login session saved. Last URL: {final_url}"


def _do_screenshot(save_path: str) -> str:
    page = _get_page()
    page.screenshot(path=save_path)
    return f"Screenshot saved to {save_path}. Page: {page.title()} ({page.url})"


# --- Tools (called by agents, delegate to browser thread) ---


@_safe_browser_call
def browser_open(
    url: Annotated[str, Field(description="URL to navigate to")],
    selector: Annotated[str | None, Field(description="CSS selector to extract specific section (optional)")] = None,
) -> str:
    """Open a URL in Firefox and return page title + text content. Auto-logs in if credentials are configured."""
    return _run_in_browser_thread(_do_open, url, selector)


@_safe_browser_call
def browser_read(
    selector: Annotated[str, Field(description="CSS selector to extract text from (e.g. '#main-content', '.issue-detail', 'table')")],
) -> str:
    """Extract text from an element on the current page using a CSS selector."""
    return _run_in_browser_thread(_do_read, selector)


@_safe_browser_call
def browser_click(
    selector: Annotated[str | None, Field(description="CSS selector of element to click")] = None,
    text: Annotated[str | None, Field(description="Visible text of element to click")] = None,
) -> str:
    """Click an element on the current page by CSS selector or visible text."""
    return _run_in_browser_thread(_do_click, selector, text)


@_safe_browser_call
def browser_fill(
    selector: Annotated[str, Field(description="CSS selector of the input field")],
    value: Annotated[str, Field(description="Text to type into the field")],
) -> str:
    """Fill a form field on the current page."""
    return _run_in_browser_thread(_do_fill, selector, value)


@tool(approval_mode="always_require")
@_safe_browser_call
def browser_login(
    url: Annotated[str, Field(description="URL to open for manual login (e.g. https://github.com/login)")],
) -> str:
    """Open a VISIBLE Firefox browser for manual login. Session is saved after login. Requires approval."""
    return _run_in_browser_thread(_do_login, url)


@_safe_browser_call
def browser_screenshot(
    path: Annotated[str | None, Field(description="File path to save screenshot (optional)")] = None,
) -> str:
    """Take a screenshot of the current page for debugging."""
    save_path = path or str(Path(settings.browser_profile_path).expanduser() / "screenshot.png")
    return _run_in_browser_thread(_do_screenshot, save_path)
