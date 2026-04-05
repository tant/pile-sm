"""Git CLI tools for the Git Agent. All read-only operations via subprocess."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Annotated

from pydantic import Field

from pile.config import settings

logger = logging.getLogger("pile.tools.git")

MAX_OUTPUT = 4000

# Safe patterns for git parameters
_SAFE_REF = re.compile(r"^[a-zA-Z0-9_./@\-~^]+$")
_SAFE_PATH = re.compile(r"^[a-zA-Z0-9_./@\- ]+$")


def _validate_repo(repo_path: str) -> str | None:
    """Validate repo_path is in the allowed list. Returns error message or None."""
    allowed = settings.git_repo_paths
    if not allowed:
        return "No git repositories configured. Set GIT_REPOS in .env."
    if repo_path not in allowed:
        return f"Repository not allowed. Available repos: {', '.join(allowed)}"
    return None


def _validate_ref(ref: str) -> str | None:
    """Validate a git ref (branch, tag, commit hash). Returns error or None."""
    if not ref or not _SAFE_REF.match(ref):
        return f"Invalid git reference: {ref}"
    if ref.startswith("-"):
        return f"Invalid git reference (cannot start with -): {ref}"
    return None


def _validate_path(path: str) -> str | None:
    """Validate a file path. Returns error or None."""
    if not path or ".." in path or path.startswith("/") or path.startswith("-"):
        return f"Invalid file path: {path}"
    if not _SAFE_PATH.match(path):
        return f"Invalid file path characters: {path}"
    return None


def _git_env(repo_path: str) -> dict[str, str] | None:
    """Build environment with credentials for private repos."""
    repo = settings.get_git_repo(repo_path)
    if not repo or not repo.has_credentials:
        return None
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if repo.url and repo.auth_url:
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = f"url.{repo.auth_url}.insteadOf"
        env["GIT_CONFIG_VALUE_0"] = repo.url
    return env


def _run_git(repo_path: str, *args: str) -> str:
    """Run a git command and return stdout, truncated if too long."""
    error = _validate_repo(repo_path)
    if error:
        return error
    cmd = ["git", "-C", repo_path, *args]
    env = _git_env(repo_path)
    logger.info("git %s", " ".join(args[:3]))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out (30s)."
    except FileNotFoundError:
        return "Error: git not found. Is git installed?"
    except Exception as e:
        logger.exception("Unexpected git error")
        return f"Error: {e}"
    if result.returncode != 0:
        return f"Git error: {result.stderr.strip()}"
    output = result.stdout.strip() or "No output."
    if len(output) > MAX_OUTPUT:
        return output[:MAX_OUTPUT] + "\n... (truncated)"
    return output


def git_log(
    repo_path: Annotated[str, Field(description="Absolute path to git repository")],
    count: Annotated[int, Field(description="Number of commits to show")] = 20,
    author: Annotated[str | None, Field(description="Filter by author name")] = None,
    since: Annotated[str | None, Field(description="Show commits since date, e.g. '2 days ago', '2026-04-01'")] = None,
    branch: Annotated[str | None, Field(description="Branch name, defaults to current branch")] = None,
) -> str:
    """Get git commit history."""
    args = ["log", f"-{min(count, 100)}", "--format=%h | %an | %ad | %s", "--date=short"]
    if author:
        args.extend(["--author", author])
    if since:
        args.extend(["--since", since])
    if branch:
        err = _validate_ref(branch)
        if err:
            return err
        args.append(branch)
    args.append("--")
    return _run_git(repo_path, *args)


def git_diff(
    repo_path: Annotated[str, Field(description="Absolute path to git repository")],
    ref1: Annotated[str, Field(description="First reference (branch name or commit hash)")],
    ref2: Annotated[str, Field(description="Second reference")] = "HEAD",
    stat_only: Annotated[bool, Field(description="Show only file change summary, not full diff")] = True,
) -> str:
    """Compare changes between two git references."""
    for ref in (ref1, ref2):
        err = _validate_ref(ref)
        if err:
            return err
    args = ["diff"]
    if stat_only:
        args.append("--stat")
    args.extend([f"{ref1}..{ref2}", "--"])
    return _run_git(repo_path, *args)


def git_branch_list(
    repo_path: Annotated[str, Field(description="Absolute path to git repository")],
    remote: Annotated[bool, Field(description="Include remote branches")] = False,
) -> str:
    """List git branches."""
    args = ["branch", "--format=%(refname:short) %(upstream:short) %(committerdate:short)"]
    if remote:
        args.insert(1, "-a")
    return _run_git(repo_path, *args)


def git_show(
    repo_path: Annotated[str, Field(description="Absolute path to git repository")],
    commit_hash: Annotated[str, Field(description="Commit hash or reference")],
) -> str:
    """Show details of a specific commit."""
    err = _validate_ref(commit_hash)
    if err:
        return err
    return _run_git(repo_path, "show", "--stat", "--format=full", commit_hash, "--")


def git_blame(
    repo_path: Annotated[str, Field(description="Absolute path to git repository")],
    file_path: Annotated[str, Field(description="Relative file path within the repository")],
    line_start: Annotated[int | None, Field(description="Start line number")] = None,
    line_end: Annotated[int | None, Field(description="End line number")] = None,
) -> str:
    """Show who last modified each line of a file."""
    err = _validate_path(file_path)
    if err:
        return err
    args = ["blame", "--date=short"]
    if line_start is not None:
        end = line_end if line_end is not None else ""
        args.extend(["-L", f"{line_start},{end}"])
    args.extend(["--", file_path])
    return _run_git(repo_path, *args)
