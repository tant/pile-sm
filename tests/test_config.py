"""Tests for pile.config — settings and GitRepo parsing."""

from pile.config import GitRepo, Settings


class TestGitRepo:
    def test_no_credentials(self):
        repo = GitRepo(path="/tmp/repo")
        assert not repo.has_credentials
        assert repo.auth_url is None

    def test_token_credentials(self):
        repo = GitRepo(path="/tmp/repo", url="https://github.com/org/repo.git", token="ghp_abc")
        assert repo.has_credentials
        assert "ghp_abc@" in repo.auth_url

    def test_username_password_credentials(self):
        repo = GitRepo(path="/tmp/repo", url="https://github.com/org/repo.git", username="user", password="pass")
        assert repo.has_credentials
        assert "user:pass@" in repo.auth_url

    def test_auth_url_encodes_special_chars(self):
        repo = GitRepo(path="/tmp/repo", url="https://github.com/org/repo.git", token="tok@en/special")
        assert repo.has_credentials
        url = repo.auth_url
        assert "@" not in url.split("@", 1)[0].replace("https://", "").replace("tok%40en%2Fspecial", "")

    def test_no_url_returns_none(self):
        repo = GitRepo(path="/tmp/repo", token="ghp_abc")
        assert repo.auth_url is None


class TestSettings:
    def test_git_repos_simple(self):
        s = Settings(git_repos="/repo1, /repo2", git_repos_json="")
        repos = s.git_repo_list
        assert len(repos) == 2
        assert repos[0].path == "/repo1"
        assert repos[1].path == "/repo2"

    def test_git_repos_json(self):
        import json
        data = [{"path": "/repo1", "token": "tok"}, "/repo2"]
        s = Settings(git_repos="", git_repos_json=json.dumps(data))
        repos = s.git_repo_list
        assert len(repos) == 2
        assert repos[0].token == "tok"
        assert repos[1].path == "/repo2"

    def test_git_repos_json_takes_priority(self):
        import json
        s = Settings(
            git_repos="/simple",
            git_repos_json=json.dumps(["/json_repo"]),
        )
        repos = s.git_repo_list
        assert len(repos) == 1
        assert repos[0].path == "/json_repo"

    def test_git_repos_invalid_json(self):
        s = Settings(git_repos="", git_repos_json="not valid json")
        assert s.git_repo_list == []

    def test_git_repos_empty(self):
        s = Settings(git_repos="", git_repos_json="")
        assert s.git_repo_list == []

    def test_get_git_repo_found(self):
        s = Settings(git_repos="/repo1,/repo2", git_repos_json="")
        repo = s.get_git_repo("/repo1")
        assert repo is not None
        assert repo.path == "/repo1"

    def test_get_git_repo_not_found(self):
        s = Settings(git_repos="/repo1", git_repos_json="")
        assert s.get_git_repo("/other") is None

    def test_git_repo_list_not_cached(self):
        """Verify git_repo_list is a regular property (not cached), allowing config changes."""
        s = Settings(git_repos="/repo1", git_repos_json="")
        assert len(s.git_repo_list) == 1
        s.git_repos = "/repo1,/repo2"
        assert len(s.git_repo_list) == 2
