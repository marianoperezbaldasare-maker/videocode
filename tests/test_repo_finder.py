"""Unit tests for videocode.repo_finder."""

from __future__ import annotations

import pytest

from videocode.repo_finder import (
    RepoCandidate,
    find_repos_in_text,
)


class TestFindReposInText:
    def test_extracts_basic_github_url(self) -> None:
        text = "Source code: https://github.com/myorg/myrepo here"
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
        assert repos[0].owner == "myorg"
        assert repos[0].repo == "myrepo"
        assert repos[0].url == "https://github.com/myorg/myrepo"
        assert repos[0].host == "github"
        assert repos[0].source == "desc"

    def test_extracts_gitlab_url(self) -> None:
        text = "Mirror at https://gitlab.com/myorg/myrepo too"
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
        assert repos[0].host == "gitlab"

    def test_deduplicates_same_repo(self) -> None:
        text = (
            "Repo: https://github.com/x/y\n"
            "Also see https://github.com/x/y/blob/main/README.md"
        )
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1

    def test_filters_non_repo_paths(self) -> None:
        text = (
            "https://github.com/sponsors/someone "
            "https://github.com/marketplace/foo "
            "https://github.com/myorg/myrepo"
        )
        repos = find_repos_in_text(text, "desc")
        urls = [r.url for r in repos]
        assert "https://github.com/myorg/myrepo" in urls
        assert all("sponsors" not in u for u in urls)
        assert all("marketplace" not in u for u in urls)

    def test_handles_trailing_dot_git(self) -> None:
        text = "Clone: https://github.com/foo/bar.git"
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
        assert repos[0].repo == "bar"

    def test_handles_trailing_punctuation(self) -> None:
        text = "See (https://github.com/foo/bar) for code."
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
        assert repos[0].repo == "bar"

    def test_multiple_distinct_repos(self) -> None:
        text = (
            "Backend: https://github.com/team/backend\n"
            "Frontend: https://github.com/team/frontend"
        )
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 2
        repo_names = sorted(r.repo for r in repos)
        assert repo_names == ["backend", "frontend"]

    def test_empty_text(self) -> None:
        assert find_repos_in_text("", "desc") == []
        assert find_repos_in_text(None, "desc") == []

    def test_no_github_in_text(self) -> None:
        text = "Check the website at https://example.com"
        assert find_repos_in_text(text, "desc") == []

    def test_case_insensitive_matching(self) -> None:
        text = "Repo: HTTPS://GitHub.com/Foo/Bar"
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
        assert repos[0].owner == "Foo"
        assert repos[0].repo == "Bar"

    def test_dedup_is_case_insensitive(self) -> None:
        text = "https://github.com/Foo/Bar and https://github.com/foo/bar"
        repos = find_repos_in_text(text, "desc")
        assert len(repos) == 1
