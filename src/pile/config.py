"""Application settings loaded from .env file."""

from __future__ import annotations

import json
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


@dataclass
class GitRepo:
    """A git repository with optional credentials for private repos."""

    path: str
    url: str | None = None  # remote URL (for clone/fetch if needed)
    username: str | None = None
    password: str | None = None  # password or API token
    token: str | None = None  # alternative: standalone API token (e.g. GitHub PAT)

    @property
    def has_credentials(self) -> bool:
        return bool(self.token or (self.username and self.password))

    @property
    def auth_url(self) -> str | None:
        """Build authenticated URL for git operations."""
        from urllib.parse import quote
        if not self.url:
            return None
        if self.token:
            return self.url.replace("https://", f"https://{quote(self.token, safe='')}@")
        if self.username and self.password:
            return self.url.replace("https://", f"https://{quote(self.username, safe='')}:{quote(self.password, safe='')}@")
        return self.url


class Settings(BaseSettings):
    # LLM Provider: "ollama" (default), "openai", or "ollama-native"
    llm_provider: str = "ollama"

    # Ollama (used when llm_provider is "ollama" or "ollama-native")
    ollama_host: str = "http://localhost:11434"
    ollama_model_id: str = "qwen3.5:9b"

    # OpenAI-compatible (used when llm_provider is "openai")
    openai_base_url: str = "http://localhost:1234/v1"
    openai_model: str = "qwen3.5:9b"
    openai_api_key: str = "lm-studio"

    # Jira
    jira_base_url: str = "https://your-instance.atlassian.net"
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    # Git repositories
    # Simple: comma-separated paths (public repos or already-cloned local repos)
    git_repos: str = ""
    # Advanced: JSON array for repos with credentials (see .env.sample)
    git_repos_json: str = ""

    # Memory / RAG
    memory_enabled: bool = True
    memory_store_path: str = "~/.pile/chromadb"
    embedding_model_id: str = "nomic-embed-text"

    # Agent limits (prevent tool call loops, tune per model capability)
    agent_max_iterations: int = 5
    agent_max_function_calls: int = 15

    # Default board (auto-detected on startup if empty)
    default_board_id: int = 0

    # Browser (Playwright + Firefox)
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

    @property
    def git_repo_list(self) -> list[GitRepo]:
        """Parse git repos from config. Supports both simple and JSON formats."""
        repos: list[GitRepo] = []

        # Parse JSON format first (takes priority if both set)
        if self.git_repos_json:
            try:
                parsed = json.loads(self.git_repos_json)
            except json.JSONDecodeError as e:
                import logging
                logging.getLogger("pile.config").error("Invalid GIT_REPOS_JSON: %s", e)
                return repos
            for item in parsed:
                if isinstance(item, str):
                    repos.append(GitRepo(path=item))
                elif isinstance(item, dict):
                    repos.append(GitRepo(**item))

        # Parse simple comma-separated paths
        if self.git_repos and not repos:
            for p in self.git_repos.split(","):
                p = p.strip()
                if p:
                    repos.append(GitRepo(path=p))

        return repos

    @property
    def git_repo_paths(self) -> list[str]:
        """Get just the paths for simple lookups."""
        return [r.path for r in self.git_repo_list]

    def get_git_repo(self, path: str) -> GitRepo | None:
        """Look up a repo config by path."""
        for r in self.git_repo_list:
            if r.path == path:
                return r
        return None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
