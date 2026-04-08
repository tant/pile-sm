"""Tests for git_tools tool functions with mocked subprocess calls."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pile.tools.git_tools import (
    _git_env,
    _run_git,
    _validate_path,
    _validate_ref,
    _validate_repo,
    git_blame,
    git_branch_list,
    git_diff,
    git_log,
    git_show,
)


@pytest.fixture(autouse=True)
def _allow_repo(monkeypatch):
    """Configure settings to allow /test/repo for all tests."""
    from pile.config import Settings

    s = Settings(git_repos="/test/repo", git_repos_json="")
    monkeypatch.setattr("pile.tools.git_tools.settings", s)


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["git"], stdout=stdout, stderr=stderr, returncode=returncode,
    )


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------


class TestRunGit:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = _completed(stdout="ok output")
        result = _run_git("/test/repo", "status")
        assert result == "ok output"

    @patch("pile.tools.git_tools.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = _completed(stdout="")
        result = _run_git("/test/repo", "status")
        assert result == "No output."

    @patch("pile.tools.git_tools.subprocess.run")
    def test_truncation(self, mock_run):
        long_output = "x" * 5000
        mock_run.return_value = _completed(stdout=long_output)
        result = _run_git("/test/repo", "log")
        assert result.endswith("... (truncated)")
        assert len(result) < 5000

    @patch("pile.tools.git_tools.subprocess.run")
    def test_nonzero_returncode(self, mock_run):
        mock_run.return_value = _completed(stderr="fatal: bad ref", returncode=128)
        result = _run_git("/test/repo", "show", "badref")
        assert "Git error" in result
        assert "bad ref" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        result = _run_git("/test/repo", "log")
        assert "timed out" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_git_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run_git("/test/repo", "log")
        assert "git not found" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_unexpected_error(self, mock_run):
        mock_run.side_effect = OSError("disk error")
        result = _run_git("/test/repo", "log")
        assert "Error" in result

    def test_disallowed_repo(self):
        result = _run_git("/not/allowed", "status")
        assert "not allowed" in result


# ---------------------------------------------------------------------------
# _git_env
# ---------------------------------------------------------------------------


class TestGitEnv:
    def test_no_credentials(self, monkeypatch):
        from pile.config import Settings

        s = Settings(git_repos="/test/repo", git_repos_json="")
        monkeypatch.setattr("pile.tools.git_tools.settings", s)
        env = _git_env("/test/repo")
        assert env is None

    def test_with_credentials(self, monkeypatch):
        import json

        from pile.config import Settings

        repos_json = json.dumps([{
            "path": "/test/repo",
            "url": "https://github.com/org/repo.git",
            "token": "ghp_secret",
        }])
        s = Settings(git_repos="", git_repos_json=repos_json)
        monkeypatch.setattr("pile.tools.git_tools.settings", s)
        env = _git_env("/test/repo")
        assert env is not None
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_CONFIG_COUNT"] == "1"


# ---------------------------------------------------------------------------
# git_log
# ---------------------------------------------------------------------------


class TestGitLog:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_basic(self, mock_run):
        mock_run.return_value = _completed(stdout="abc123 | Author | 2026-04-01 | msg")
        result = git_log("/test/repo")
        assert "abc123" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_with_author_and_since(self, mock_run):
        mock_run.return_value = _completed(stdout="abc | Alice | 2026-04-01 | fix")
        result = git_log("/test/repo", author="Alice", since="2 days ago")
        call_args = mock_run.call_args[0][0]
        assert "--author" in call_args
        assert "--since" in call_args

    @patch("pile.tools.git_tools.subprocess.run")
    def test_with_branch(self, mock_run):
        mock_run.return_value = _completed(stdout="output")
        result = git_log("/test/repo", branch="develop")
        call_args = mock_run.call_args[0][0]
        assert "develop" in call_args

    def test_invalid_branch(self):
        result = git_log("/test/repo", branch=";rm -rf /")
        assert "Invalid" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_count_capped_at_100(self, mock_run):
        mock_run.return_value = _completed(stdout="output")
        git_log("/test/repo", count=999)
        call_args = mock_run.call_args[0][0]
        assert "-100" in call_args


# ---------------------------------------------------------------------------
# git_diff
# ---------------------------------------------------------------------------


class TestGitDiff:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_stat_only(self, mock_run):
        mock_run.return_value = _completed(stdout=" file.py | 5 +++++")
        result = git_diff("/test/repo", ref1="main", ref2="HEAD")
        call_args = mock_run.call_args[0][0]
        assert "--stat" in call_args
        assert "file.py" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_full_diff(self, mock_run):
        mock_run.return_value = _completed(stdout="+added line")
        result = git_diff("/test/repo", ref1="main", ref2="HEAD", stat_only=False)
        call_args = mock_run.call_args[0][0]
        assert "--stat" not in call_args

    def test_invalid_ref1(self):
        result = git_diff("/test/repo", ref1="$(evil)", ref2="HEAD")
        assert "Invalid" in result

    def test_invalid_ref2(self):
        result = git_diff("/test/repo", ref1="main", ref2="$(evil)")
        assert "Invalid" in result


# ---------------------------------------------------------------------------
# git_show
# ---------------------------------------------------------------------------


class TestGitShow:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_valid_commit(self, mock_run):
        mock_run.return_value = _completed(stdout="commit abc123\nAuthor: A\n\nmsg")
        result = git_show("/test/repo", commit_hash="abc123")
        assert "abc123" in result

    def test_invalid_commit(self):
        result = git_show("/test/repo", commit_hash=";evil")
        assert "Invalid" in result


# ---------------------------------------------------------------------------
# git_branch_list
# ---------------------------------------------------------------------------


class TestGitBranchList:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_local_only(self, mock_run):
        mock_run.return_value = _completed(stdout="main origin/main 2026-04-01")
        result = git_branch_list("/test/repo")
        call_args = mock_run.call_args[0][0]
        assert "-a" not in call_args
        assert "main" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_include_remote(self, mock_run):
        mock_run.return_value = _completed(stdout="main\norigin/develop")
        result = git_branch_list("/test/repo", remote=True)
        call_args = mock_run.call_args[0][0]
        assert "-a" in call_args


# ---------------------------------------------------------------------------
# git_blame
# ---------------------------------------------------------------------------


class TestGitBlame:
    @patch("pile.tools.git_tools.subprocess.run")
    def test_basic(self, mock_run):
        mock_run.return_value = _completed(stdout="abc123 (Author 2026-04-01  1) line")
        result = git_blame("/test/repo", file_path="src/main.py")
        assert "abc123" in result

    @patch("pile.tools.git_tools.subprocess.run")
    def test_with_line_range(self, mock_run):
        mock_run.return_value = _completed(stdout="blame output")
        result = git_blame("/test/repo", file_path="src/main.py", line_start=10, line_end=20)
        call_args = mock_run.call_args[0][0]
        assert "-L" in call_args
        assert "10,20" in call_args

    @patch("pile.tools.git_tools.subprocess.run")
    def test_with_line_start_only(self, mock_run):
        mock_run.return_value = _completed(stdout="blame output")
        result = git_blame("/test/repo", file_path="src/main.py", line_start=5)
        call_args = mock_run.call_args[0][0]
        assert "-L" in call_args
        assert "5," in call_args

    def test_invalid_path(self):
        result = git_blame("/test/repo", file_path="../etc/passwd")
        assert "Invalid" in result
