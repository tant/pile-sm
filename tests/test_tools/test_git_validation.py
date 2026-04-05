"""Tests for git_tools validation functions."""

from pile.tools.git_tools import _validate_path, _validate_ref, _validate_repo


class TestValidateRepo:
    def test_no_repos_configured(self, monkeypatch):
        from pile.config import Settings
        s = Settings(git_repos="", git_repos_json="")
        monkeypatch.setattr("pile.tools.git_tools.settings", s)
        assert _validate_repo("/some/path") is not None
        assert "No git repositories" in _validate_repo("/some/path")

    def test_allowed_repo(self, monkeypatch):
        from pile.config import Settings
        s = Settings(git_repos="/allowed/repo", git_repos_json="")
        monkeypatch.setattr("pile.tools.git_tools.settings", s)
        assert _validate_repo("/allowed/repo") is None

    def test_disallowed_repo(self, monkeypatch):
        from pile.config import Settings
        s = Settings(git_repos="/allowed/repo", git_repos_json="")
        monkeypatch.setattr("pile.tools.git_tools.settings", s)
        result = _validate_repo("/other/repo")
        assert result is not None
        assert "not allowed" in result


class TestValidateRef:
    def test_valid_branch(self):
        assert _validate_ref("main") is None
        assert _validate_ref("feature/foo-bar") is None
        assert _validate_ref("v1.0.0") is None
        assert _validate_ref("HEAD~3") is None
        assert _validate_ref("abc123") is None

    def test_empty_ref(self):
        assert _validate_ref("") is not None

    def test_starts_with_dash(self):
        assert _validate_ref("-evil") is not None

    def test_invalid_characters(self):
        assert _validate_ref("branch;rm -rf") is not None
        assert _validate_ref("$(whoami)") is not None
        assert _validate_ref("branch`cmd`") is not None


class TestValidatePath:
    def test_valid_paths(self):
        assert _validate_path("src/main.py") is None
        assert _validate_path("README.md") is None
        assert _validate_path("path with spaces/file.txt") is None

    def test_empty_path(self):
        assert _validate_path("") is not None

    def test_path_traversal(self):
        assert _validate_path("../etc/passwd") is not None
        assert _validate_path("foo/../../bar") is not None

    def test_absolute_path(self):
        assert _validate_path("/etc/passwd") is not None

    def test_starts_with_dash(self):
        assert _validate_path("-evil") is not None

    def test_invalid_characters(self):
        assert _validate_path("file;rm") is not None
