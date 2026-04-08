"""Tests for health checks — all external dependencies are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# check_models
# ---------------------------------------------------------------------------

class TestCheckModels:
    def _call(self):
        from pile.health import check_models
        return check_models()

    @patch("pile.health.get_model_path")
    def test_all_present(self, mock_path):
        mock_p = MagicMock()
        mock_p.exists.return_value = True
        mock_path.return_value = mock_p
        assert self._call() is None

    @patch("pile.health.get_model_path")
    def test_some_missing(self, mock_path):
        def side_effect(role):
            m = MagicMock()
            m.exists.return_value = role != "agent"
            return m
        mock_path.side_effect = side_effect
        result = self._call()
        assert result is not None
        assert "agent" in result

    @patch("pile.health.get_model_path")
    def test_all_missing(self, mock_path):
        mock_p = MagicMock()
        mock_p.exists.return_value = False
        mock_path.return_value = mock_p
        result = self._call()
        assert result is not None
        assert "Missing model files" in result


# ---------------------------------------------------------------------------
# check_jira
# ---------------------------------------------------------------------------

class TestCheckJira:
    def _call(self):
        from pile.health import check_jira
        return check_jira()

    @patch("pile.health.settings")
    def test_missing_credentials(self, mock_settings):
        mock_settings.jira_email = ""
        mock_settings.jira_api_token = ""
        result = self._call()
        assert "JIRA_EMAIL" in result

    @patch("pile.health.settings")
    @patch("pile.health.httpx.get")
    def test_success(self, mock_get, mock_settings):
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token123"
        mock_settings.jira_base_url = "https://test.atlassian.net"
        resp = MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp
        assert self._call() is None

    @patch("pile.health.settings")
    @patch("pile.health.httpx.get")
    def test_auth_failed(self, mock_get, mock_settings):
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "bad"
        mock_settings.jira_base_url = "https://test.atlassian.net"
        resp = MagicMock()
        resp.status_code = 401
        mock_get.return_value = resp
        result = self._call()
        assert "authentication failed" in result.lower()

    @patch("pile.health.settings")
    @patch("pile.health.httpx.get")
    def test_forbidden(self, mock_get, mock_settings):
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        mock_settings.jira_base_url = "https://test.atlassian.net"
        resp = MagicMock()
        resp.status_code = 403
        mock_get.return_value = resp
        result = self._call()
        assert "forbidden" in result.lower()

    @patch("pile.health.settings")
    @patch("pile.health.httpx.get")
    def test_connect_error(self, mock_get, mock_settings):
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        mock_settings.jira_base_url = "https://test.atlassian.net"
        mock_get.side_effect = httpx.ConnectError("refused")
        result = self._call()
        assert "Cannot connect" in result

    @patch("pile.health.settings")
    @patch("pile.health.httpx.get")
    def test_generic_error(self, mock_get, mock_settings):
        mock_settings.jira_email = "user@example.com"
        mock_settings.jira_api_token = "token"
        mock_settings.jira_base_url = "https://test.atlassian.net"
        mock_get.side_effect = RuntimeError("boom")
        result = self._call()
        assert "health check failed" in result


# ---------------------------------------------------------------------------
# check_browser
# ---------------------------------------------------------------------------

class TestCheckBrowser:
    def _call(self):
        from pile.health import check_browser
        return check_browser()

    @patch("pile.health.settings")
    def test_browser_disabled(self, mock_settings):
        mock_settings.browser_enabled = False
        assert self._call() is None

    @patch("subprocess.run")
    @patch("pile.health.settings")
    def test_browser_ok(self, mock_settings, mock_run):
        mock_settings.browser_enabled = True
        mock_run.return_value = MagicMock(returncode=0)
        assert self._call() is None

    @patch("subprocess.run")
    @patch("pile.health.settings")
    def test_browser_not_installed(self, mock_settings, mock_run):
        mock_settings.browser_enabled = True
        mock_run.return_value = MagicMock(returncode=1)
        result = self._call()
        assert "not installed" in result.lower()

    @patch("subprocess.run")
    @patch("pile.health.settings")
    def test_playwright_missing(self, mock_settings, mock_run):
        mock_settings.browser_enabled = True
        mock_run.side_effect = FileNotFoundError("no playwright")
        result = self._call()
        assert "not installed" in result.lower()

    @patch("subprocess.run")
    @patch("pile.health.settings")
    def test_generic_error(self, mock_settings, mock_run):
        mock_settings.browser_enabled = True
        mock_run.side_effect = RuntimeError("boom")
        result = self._call()
        assert "health check failed" in result


# ---------------------------------------------------------------------------
# run_health_checks
# ---------------------------------------------------------------------------

class TestRunHealthChecks:
    def _call(self):
        from pile.health import run_health_checks
        return run_health_checks()

    @patch("pile.health.check_browser")
    @patch("pile.health.check_jira")
    @patch("pile.health.check_models")
    @patch("pile.health.settings")
    def test_all_pass(self, mock_settings, mock_models, mock_jira, mock_browser):
        mock_settings.browser_enabled = True
        mock_models.return_value = None
        mock_jira.return_value = None
        mock_browser.return_value = None
        assert self._call() == []

    @patch("pile.health.check_browser")
    @patch("pile.health.check_jira")
    @patch("pile.health.check_models")
    @patch("pile.health.settings")
    def test_errors_collected(self, mock_settings, mock_models, mock_jira,
                               mock_browser):
        mock_settings.browser_enabled = True
        mock_models.return_value = "models err"
        mock_jira.return_value = "jira err"
        mock_browser.return_value = "browser err"
        errors = self._call()
        assert "models err" in errors
        assert "jira err" in errors
        assert "browser err" in errors

    @patch("pile.health.check_browser")
    @patch("pile.health.check_jira")
    @patch("pile.health.check_models")
    @patch("pile.health.settings")
    def test_browser_disabled_skips_check(self, mock_settings, mock_models,
                                           mock_jira, mock_browser):
        mock_settings.browser_enabled = False
        mock_models.return_value = None
        mock_jira.return_value = None
        errors = self._call()
        assert errors == []
        mock_browser.assert_not_called()

    @patch("pile.health.check_browser")
    @patch("pile.health.check_jira")
    @patch("pile.health.check_models")
    @patch("pile.health.settings")
    def test_only_models_fail(self, mock_settings, mock_models, mock_jira,
                               mock_browser):
        mock_settings.browser_enabled = False
        mock_models.return_value = "missing models"
        mock_jira.return_value = None
        errors = self._call()
        assert errors == ["missing models"]
