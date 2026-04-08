"""Tests for browser_tools with mocked Playwright."""

from unittest.mock import MagicMock, patch, call

import pytest

from pile.tools.browser_tools import (
    MAX_CONTENT,
    _detect_login_page,
    _do_click,
    _do_fill,
    _do_login,
    _do_open,
    _do_read,
    _do_screenshot,
    _handle_login_if_needed,
    _run_in_browser_thread,
    _safe_browser_call,
    _try_auto_login,
    browser_click,
    browser_fill,
    browser_login,
    browser_open,
    browser_read,
    browser_screenshot,
)


# ---------------------------------------------------------------------------
# _run_in_browser_thread
# ---------------------------------------------------------------------------


class TestRunInBrowserThread:
    @patch("pile.tools.browser_tools._executor")
    def test_submits_and_waits(self, mock_executor):
        future = MagicMock()
        future.result.return_value = "done"
        mock_executor.submit.return_value = future

        result = _run_in_browser_thread(lambda: "hello")

        mock_executor.submit.assert_called_once()
        future.result.assert_called_once_with(timeout=60)
        assert result == "done"

    @patch("pile.tools.browser_tools._executor")
    def test_passes_args_and_kwargs(self, mock_executor):
        future = MagicMock()
        future.result.return_value = "ok"
        mock_executor.submit.return_value = future

        def my_fn(a, b, key=None):
            pass

        _run_in_browser_thread(my_fn, 1, 2, key="val")
        mock_executor.submit.assert_called_once_with(my_fn, 1, 2, key="val")


# ---------------------------------------------------------------------------
# _get_context
# ---------------------------------------------------------------------------


class TestGetContext:
    @patch("pile.tools.browser_tools.os.makedirs")
    @patch("pile.tools.browser_tools.settings")
    def test_returns_existing_context(self, mock_settings, mock_makedirs):
        mock_settings.browser_profile_path = "/tmp/browser"
        import pile.tools.browser_tools as mod

        fake_ctx = MagicMock()
        old_ctx = mod._browser_context
        try:
            mod._browser_context = fake_ctx
            from pile.tools.browser_tools import _get_context
            result = _get_context()
            assert result is fake_ctx
        finally:
            mod._browser_context = old_ctx

    @patch("pile.tools.browser_tools.os.makedirs")
    @patch("pile.tools.browser_tools.settings")
    def test_creates_new_context(self, mock_settings, mock_makedirs):
        mock_settings.browser_profile_path = "/tmp/browser"
        import pile.tools.browser_tools as mod

        old_ctx = mod._browser_context
        old_pw = mod._playwright_instance
        try:
            mod._browser_context = None
            mod._playwright_instance = None

            mock_pw = MagicMock()
            mock_ctx = MagicMock()
            mock_pw.firefox.launch_persistent_context.return_value = mock_ctx

            with patch("pile.tools.browser_tools.sync_playwright", create=True) as mock_sync_pw:
                # Patch the import inside the function
                mock_sync_pw_instance = MagicMock()
                mock_sync_pw_instance.start.return_value = mock_pw
                with patch.dict("sys.modules", {"playwright.sync_api": MagicMock(sync_playwright=lambda: mock_sync_pw_instance)}):
                    from importlib import reload
                    # Direct test: simulate what _get_context does
                    from pile.tools.browser_tools import _get_context

                    # Manually set playwright instance to skip the import
                    mod._playwright_instance = mock_pw

                    result = _get_context()

                    assert result is mock_ctx
                    mock_pw.firefox.launch_persistent_context.assert_called_once()
                    call_kwargs = mock_pw.firefox.launch_persistent_context.call_args.kwargs
                    assert call_kwargs["headless"] is True
        finally:
            mod._browser_context = old_ctx
            mod._playwright_instance = old_pw

    @patch("pile.tools.browser_tools.os.makedirs")
    @patch("pile.tools.browser_tools.settings")
    def test_creates_headed_context(self, mock_settings, mock_makedirs):
        mock_settings.browser_profile_path = "/tmp/browser"
        import pile.tools.browser_tools as mod

        old_ctx = mod._browser_context
        old_pw = mod._playwright_instance
        try:
            mod._browser_context = None
            mock_pw = MagicMock()
            mock_ctx = MagicMock()
            mock_pw.firefox.launch_persistent_context.return_value = mock_ctx
            mod._playwright_instance = mock_pw

            from pile.tools.browser_tools import _get_context
            result = _get_context(headed=True)

            call_kwargs = mock_pw.firefox.launch_persistent_context.call_args.kwargs
            assert call_kwargs["headless"] is False
        finally:
            mod._browser_context = old_ctx
            mod._playwright_instance = old_pw


# ---------------------------------------------------------------------------
# _get_page
# ---------------------------------------------------------------------------


class TestGetPage:
    def test_creates_new_page_when_none(self):
        import pile.tools.browser_tools as mod

        old_page = mod._current_page
        old_ctx = mod._browser_context
        try:
            mod._current_page = None
            mock_ctx = MagicMock()
            mock_new_page = MagicMock()
            mock_ctx.pages = []
            mock_ctx.new_page.return_value = mock_new_page

            with patch("pile.tools.browser_tools._get_context", return_value=mock_ctx):
                from pile.tools.browser_tools import _get_page
                result = _get_page()
                assert result is mock_new_page
        finally:
            mod._current_page = old_page
            mod._browser_context = old_ctx

    def test_reuses_existing_page(self):
        import pile.tools.browser_tools as mod

        old_page = mod._current_page
        try:
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mod._current_page = mock_page

            with patch("pile.tools.browser_tools._get_context") as mock_ctx:
                from pile.tools.browser_tools import _get_page
                result = _get_page()
                assert result is mock_page
        finally:
            mod._current_page = old_page

    def test_uses_first_existing_page(self):
        import pile.tools.browser_tools as mod

        old_page = mod._current_page
        try:
            mod._current_page = None
            mock_ctx = MagicMock()
            existing_page = MagicMock()
            mock_ctx.pages = [existing_page]

            with patch("pile.tools.browser_tools._get_context", return_value=mock_ctx):
                from pile.tools.browser_tools import _get_page
                result = _get_page()
                assert result is existing_page
        finally:
            mod._current_page = old_page

    def test_creates_new_when_closed(self):
        import pile.tools.browser_tools as mod

        old_page = mod._current_page
        try:
            mock_page = MagicMock()
            mock_page.is_closed.return_value = True
            mod._current_page = mock_page

            mock_ctx = MagicMock()
            new_page = MagicMock()
            mock_ctx.pages = []
            mock_ctx.new_page.return_value = new_page

            with patch("pile.tools.browser_tools._get_context", return_value=mock_ctx):
                from pile.tools.browser_tools import _get_page
                result = _get_page()
                assert result is new_page
        finally:
            mod._current_page = old_page


# ---------------------------------------------------------------------------
# _close_context
# ---------------------------------------------------------------------------


class TestCloseContext:
    def test_closes_existing_context(self):
        import pile.tools.browser_tools as mod

        old_ctx = mod._browser_context
        old_page = mod._current_page
        try:
            mock_ctx = MagicMock()
            mod._browser_context = mock_ctx
            mod._current_page = MagicMock()

            from pile.tools.browser_tools import _close_context
            _close_context()

            mock_ctx.close.assert_called_once()
            assert mod._browser_context is None
            assert mod._current_page is None
        finally:
            mod._browser_context = old_ctx
            mod._current_page = old_page

    def test_no_context_noop(self):
        import pile.tools.browser_tools as mod

        old_ctx = mod._browser_context
        old_page = mod._current_page
        try:
            mod._browser_context = None
            mod._current_page = None

            from pile.tools.browser_tools import _close_context
            _close_context()

            assert mod._browser_context is None
            assert mod._current_page is None
        finally:
            mod._browser_context = old_ctx
            mod._current_page = old_page

    def test_close_exception_suppressed(self):
        import pile.tools.browser_tools as mod

        old_ctx = mod._browser_context
        old_page = mod._current_page
        try:
            mock_ctx = MagicMock()
            mock_ctx.close.side_effect = Exception("close failed")
            mod._browser_context = mock_ctx
            mod._current_page = MagicMock()

            from pile.tools.browser_tools import _close_context
            _close_context()

            assert mod._browser_context is None
            assert mod._current_page is None
        finally:
            mod._browser_context = old_ctx
            mod._current_page = old_page


# ---------------------------------------------------------------------------
# _do_login
# ---------------------------------------------------------------------------


class TestDoLogin:
    @patch("pile.tools.browser_tools._close_context")
    @patch("pile.tools.browser_tools._get_context")
    def test_login_flow(self, mock_get_context, mock_close_context):
        mock_ctx = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://github.com/dashboard"
        mock_ctx.pages = [mock_page]
        mock_get_context.return_value = mock_ctx

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            result = _do_login("https://github.com/login")

        # Should close context first
        assert mock_close_context.call_count == 2
        # Should open headed context
        mock_get_context.assert_called_once_with(headed=True)
        # Should navigate
        mock_page.goto.assert_called_once_with(
            "https://github.com/login", wait_until="domcontentloaded", timeout=60000
        )
        assert "Login session saved" in result
        assert "https://github.com/dashboard" in result

    @patch("pile.tools.browser_tools._close_context")
    @patch("pile.tools.browser_tools._get_context")
    def test_login_creates_new_page_if_none(self, mock_get_context, mock_close_context):
        mock_ctx = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        mock_ctx.pages = []
        mock_ctx.new_page.return_value = mock_page
        mock_get_context.return_value = mock_ctx

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            result = _do_login("https://example.com/login")

        mock_ctx.new_page.assert_called_once()
        assert "Login session saved" in result


# ---------------------------------------------------------------------------
# browser_login (public tool)
# ---------------------------------------------------------------------------


class TestBrowserLogin:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "Login session saved. Last URL: https://example.com"
        result = browser_login("https://example.com/login")
        mock_thread.assert_called_once()
        assert "Login session saved" in result

    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_handles_exception(self, mock_thread):
        mock_thread.side_effect = RuntimeError("browser died")
        result = browser_login("https://example.com/login")
        assert "Error" in result
        assert "RuntimeError" in result


# ---------------------------------------------------------------------------
# _detect_login_page
# ---------------------------------------------------------------------------


class TestDetectLoginPage:
    def test_atlassian(self):
        result = _detect_login_page("https://id.atlassian.com/login")
        assert result is not None
        assert result["email_key"] == "browser_jira_email"

    def test_github(self):
        result = _detect_login_page("https://github.com/login")
        assert result is not None
        assert result["email_key"] == "browser_github_username"

    def test_gitlab(self):
        result = _detect_login_page("https://gitlab.com/users/sign_in")
        assert result is not None
        assert result["email_key"] == "browser_gitlab_username"

    def test_no_match(self):
        result = _detect_login_page("https://example.com/page")
        assert result is None


# ---------------------------------------------------------------------------
# _try_auto_login
# ---------------------------------------------------------------------------


class TestTryAutoLogin:
    def test_no_credentials(self, monkeypatch):
        from pile.config import Settings

        s = Settings(git_repos="", git_repos_json="", browser_jira_email="", browser_jira_password="")
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        pattern = {"email_key": "browser_jira_email", "password_key": "browser_jira_password"}
        page = MagicMock()
        assert _try_auto_login(page, pattern) is False

    def test_one_step_login(self, monkeypatch):
        from pile.config import Settings

        s = Settings(
            git_repos="", git_repos_json="",
            browser_github_username="user@test.com",
            browser_github_password="secret",
        )
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        pattern = {
            "email_key": "browser_github_username",
            "password_key": "browser_github_password",
            "email_selector": "#login_field",
            "password_selector": "#password",
            "submit_selector": "input[type='submit']",
            "two_step": False,
        }
        page = MagicMock()
        result = _try_auto_login(page, pattern)
        assert result is True
        page.fill.assert_any_call("#login_field", "user@test.com")
        page.fill.assert_any_call("#password", "secret")
        page.click.assert_called_once()

    def test_two_step_login(self, monkeypatch):
        from pile.config import Settings

        s = Settings(
            git_repos="", git_repos_json="",
            browser_jira_email="user@test.com",
            browser_jira_password="secret",
        )
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        pattern = {
            "email_key": "browser_jira_email",
            "password_key": "browser_jira_password",
            "email_selector": "#username",
            "password_selector": "#password",
            "submit_selector": "#login-submit",
            "two_step": True,
        }
        page = MagicMock()
        result = _try_auto_login(page, pattern)
        assert result is True
        assert page.click.call_count == 2

    def test_login_exception(self, monkeypatch):
        from pile.config import Settings

        s = Settings(
            git_repos="", git_repos_json="",
            browser_github_username="user",
            browser_github_password="pass",
        )
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        pattern = {
            "email_key": "browser_github_username",
            "password_key": "browser_github_password",
            "email_selector": "#login_field",
            "password_selector": "#password",
            "submit_selector": "input[type='submit']",
            "two_step": False,
        }
        page = MagicMock()
        page.fill.side_effect = Exception("element not found")
        result = _try_auto_login(page, pattern)
        assert result is False


# ---------------------------------------------------------------------------
# _handle_login_if_needed
# ---------------------------------------------------------------------------


class TestHandleLoginIfNeeded:
    def test_not_login_page(self):
        page = MagicMock()
        page.url = "https://example.com/dashboard"
        result = _handle_login_if_needed(page)
        assert result is None

    def test_login_no_credentials(self, monkeypatch):
        from pile.config import Settings

        s = Settings(
            git_repos="", git_repos_json="",
            browser_github_username="", browser_github_password="",
        )
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        page = MagicMock()
        page.url = "https://github.com/login"
        result = _handle_login_if_needed(page)
        assert "no credentials" in result.lower() or "browser_login" in result

    def test_login_success(self, monkeypatch):
        from pile.config import Settings

        s = Settings(
            git_repos="", git_repos_json="",
            browser_github_username="user",
            browser_github_password="pass",
        )
        monkeypatch.setattr("pile.tools.browser_tools.settings", s)

        page = MagicMock()
        page.url = "https://github.com/login"

        with patch("pile.tools.browser_tools._try_auto_login", return_value=True):
            with patch("pile.tools.browser_tools._detect_login_page", side_effect=[
                {"email_key": "browser_github_username", "password_key": "browser_github_password"},
                None,
            ]):
                result = _handle_login_if_needed(page)
                assert result is None

    def test_login_still_on_login_page(self, monkeypatch):
        page = MagicMock()
        page.url = "https://github.com/login"

        login_pattern = {
            "url_contains": ["github.com/login"],
            "email_key": "browser_github_username",
            "password_key": "browser_github_password",
        }

        with patch("pile.tools.browser_tools._try_auto_login", return_value=True):
            with patch("pile.tools.browser_tools._detect_login_page", return_value=login_pattern):
                result = _handle_login_if_needed(page)
                assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# _do_open
# ---------------------------------------------------------------------------


class TestDoOpen:
    @patch("pile.tools.browser_tools._get_page")
    @patch("pile.tools.browser_tools._handle_login_if_needed", return_value=None)
    def test_basic_open(self, mock_login, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Example"
        page.url = "https://example.com"
        page.inner_text.return_value = "Hello World"
        mock_get_page.return_value = page

        result = _do_open("https://example.com", None)
        assert "**Example**" in result
        assert "Hello World" in result
        page.goto.assert_called_once()

    @patch("pile.tools.browser_tools._get_page")
    @patch("pile.tools.browser_tools._handle_login_if_needed", return_value=None)
    def test_with_selector(self, mock_login, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Page"
        page.url = "https://example.com"
        el = MagicMock()
        el.inner_text.return_value = "Specific content"
        page.query_selector.return_value = el
        mock_get_page.return_value = page

        result = _do_open("https://example.com", "#main")
        assert "Specific content" in result

    @patch("pile.tools.browser_tools._get_page")
    @patch("pile.tools.browser_tools._handle_login_if_needed", return_value=None)
    def test_selector_not_found(self, mock_login, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Page"
        page.url = "https://example.com"
        page.query_selector.return_value = None
        mock_get_page.return_value = page

        result = _do_open("https://example.com", "#missing")
        assert "not found" in result

    @patch("pile.tools.browser_tools._get_page")
    @patch("pile.tools.browser_tools._handle_login_if_needed")
    def test_login_required(self, mock_login, mock_get_page):
        page = MagicMock()
        mock_get_page.return_value = page
        mock_login.return_value = "Login required"

        result = _do_open("https://example.com", None)
        assert result == "Login required"

    @patch("pile.tools.browser_tools._get_page")
    @patch("pile.tools.browser_tools._handle_login_if_needed", return_value=None)
    def test_content_truncation(self, mock_login, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Page"
        page.url = "https://example.com"
        page.inner_text.return_value = "x" * (MAX_CONTENT + 500)
        mock_get_page.return_value = page

        result = _do_open("https://example.com", None)
        body_text = result.split("\n\n", 1)[1]
        assert len(body_text) <= MAX_CONTENT


# ---------------------------------------------------------------------------
# _do_read
# ---------------------------------------------------------------------------


class TestDoRead:
    @patch("pile.tools.browser_tools._get_page")
    def test_element_found(self, mock_get_page):
        page = MagicMock()
        el = MagicMock()
        el.inner_text.return_value = "Content here"
        page.query_selector.return_value = el
        mock_get_page.return_value = page

        result = _do_read("#main")
        assert result == "Content here"

    @patch("pile.tools.browser_tools._get_page")
    def test_element_not_found(self, mock_get_page):
        page = MagicMock()
        page.query_selector.return_value = None
        mock_get_page.return_value = page

        result = _do_read("#missing")
        assert "No element found" in result

    @patch("pile.tools.browser_tools._get_page")
    def test_truncation(self, mock_get_page):
        page = MagicMock()
        el = MagicMock()
        el.inner_text.return_value = "y" * (MAX_CONTENT + 100)
        page.query_selector.return_value = el
        mock_get_page.return_value = page

        result = _do_read("body")
        assert len(result) == MAX_CONTENT


# ---------------------------------------------------------------------------
# _do_click
# ---------------------------------------------------------------------------


class TestDoClick:
    @patch("pile.tools.browser_tools._get_page")
    def test_click_by_selector(self, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Next Page"
        page.url = "https://example.com/next"
        mock_get_page.return_value = page

        result = _do_click("#btn", None)
        page.click.assert_called_once_with("#btn")
        assert "Clicked" in result

    @patch("pile.tools.browser_tools._get_page")
    def test_click_by_text(self, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Page"
        page.url = "https://example.com"
        text_locator = MagicMock()
        text_locator.first = MagicMock()
        page.get_by_text.return_value = text_locator
        mock_get_page.return_value = page

        result = _do_click(None, "Submit")
        page.get_by_text.assert_called_once_with("Submit", exact=False)
        assert "Clicked" in result

    @patch("pile.tools.browser_tools._get_page")
    def test_click_no_args(self, mock_get_page):
        page = MagicMock()
        mock_get_page.return_value = page
        result = _do_click(None, None)
        assert "Error" in result


# ---------------------------------------------------------------------------
# _do_fill
# ---------------------------------------------------------------------------


class TestDoFill:
    @patch("pile.tools.browser_tools._get_page")
    def test_fill(self, mock_get_page):
        page = MagicMock()
        mock_get_page.return_value = page

        result = _do_fill("#input", "hello")
        page.fill.assert_called_once_with("#input", "hello")
        assert "Filled" in result


# ---------------------------------------------------------------------------
# _do_screenshot
# ---------------------------------------------------------------------------


class TestDoScreenshot:
    @patch("pile.tools.browser_tools._get_page")
    def test_screenshot(self, mock_get_page):
        page = MagicMock()
        page.title.return_value = "Page"
        page.url = "https://example.com"
        mock_get_page.return_value = page

        result = _do_screenshot("/tmp/shot.png")
        page.screenshot.assert_called_once_with(path="/tmp/shot.png")
        assert "Screenshot saved" in result
        assert "/tmp/shot.png" in result


# ---------------------------------------------------------------------------
# _safe_browser_call decorator
# ---------------------------------------------------------------------------


class TestSafeBrowserCall:
    def test_wraps_exception(self):
        @_safe_browser_call
        def failing():
            raise RuntimeError("boom")

        result = failing()
        assert "Error" in result
        assert "RuntimeError" in result
        assert "boom" in result

    def test_passes_through_on_success(self):
        @_safe_browser_call
        def succeeding():
            return "all good"

        result = succeeding()
        assert result == "all good"


# ---------------------------------------------------------------------------
# Public tool functions (wrapper layer via _run_in_browser_thread)
# ---------------------------------------------------------------------------


class TestBrowserOpen:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "page content"
        result = browser_open("https://example.com")
        mock_thread.assert_called_once()
        assert result == "page content"

    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_handles_exception(self, mock_thread):
        mock_thread.side_effect = Exception("thread error")
        result = browser_open("https://example.com")
        assert "Error" in result


class TestBrowserRead:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "text"
        result = browser_read("#main")
        assert result == "text"


class TestBrowserClick:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "Clicked. Now on: Page (url)"
        result = browser_click(selector="#btn")
        assert "Clicked" in result


class TestBrowserFill:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "Filled '#input' with value."
        result = browser_fill("#input", "val")
        assert "Filled" in result


class TestBrowserScreenshot:
    @patch("pile.tools.browser_tools._run_in_browser_thread")
    def test_delegates(self, mock_thread):
        mock_thread.return_value = "Screenshot saved"
        result = browser_screenshot("/tmp/test.png")
        assert "Screenshot saved" in result
