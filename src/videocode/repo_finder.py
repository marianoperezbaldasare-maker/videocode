"""Find the source-code repository for a tutorial video.

Most coding tutorials link their accompanying GitHub repo in the video
description (or pinned comment, or channel about page). When that link
exists, returning the *actual* repo to the user is far more useful than
reconstructing code frame-by-frame: the code is exact, complete, and
already runnable.

This module discovers those references using ``yt-dlp`` metadata —
which is already a dependency — and a small regex. No new packages,
no API keys.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

try:
    import yt_dlp  # type: ignore[import-untyped]

    _HAS_YT_DLP = True
except ImportError:
    yt_dlp = None  # type: ignore[misc,assignment]
    _HAS_YT_DLP = False

logger = logging.getLogger(__name__)


# Match a repo URL on github.com / gitlab.com. The trailing boundary
# is intentionally permissive — paths, branches, and trailing
# punctuation are all stripped from the captured repo name.
_REPO_RE = re.compile(
    r"https?://(?P<host>github|gitlab)\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:[/?#)\]\s]|$|\.git)",
    re.IGNORECASE,
)

# GitHub URL segments that aren't repos. Stops "github.com/sponsors/x"
# or "github.com/orgs/foo" from leaking through as candidates.
_NON_REPO_OWNERS = frozenset({
    "sponsors", "marketplace", "topics", "trending", "explore",
    "settings", "notifications", "issues", "pulls", "search",
    "orgs", "users", "features", "pricing", "enterprise",
    "collections", "events", "about", "site", "security",
})


@dataclass
class RepoCandidate:
    """A repository URL discovered in a video's metadata."""

    url: str
    """Canonical URL: ``https://<host>.com/<owner>/<repo>``."""

    owner: str
    """Repo owner (user or org)."""

    repo: str
    """Repo name (with no trailing ``.git``)."""

    host: str
    """``"github"`` or ``"gitlab"``."""

    source: str
    """Where the URL was found: ``description``, ``title``,
    ``uploader_url``, ``channel_url``, etc."""


def find_repos_in_text(text: str | None, source: str = "text") -> list[RepoCandidate]:
    """Extract all GitHub/GitLab repo references from a blob of *text*.

    Deduplicates by (owner, repo) — the same repo mentioned several
    times in a description only returns one candidate.
    """
    if not text:
        return []

    candidates: list[RepoCandidate] = []
    seen: set[tuple[str, str, str]] = set()

    for match in _REPO_RE.finditer(text):
        host = match.group("host").lower()
        owner = match.group("owner")
        repo = match.group("repo").rstrip(".")
        if not repo:
            continue
        if host == "github" and owner.lower() in _NON_REPO_OWNERS:
            continue

        key = (host, owner.lower(), repo.lower())
        if key in seen:
            continue
        seen.add(key)

        candidates.append(
            RepoCandidate(
                url=f"https://{host}.com/{owner}/{repo}",
                owner=owner,
                repo=repo,
                host=host,
                source=source,
            )
        )

    return candidates


def find_repos_for_url(url: str, *, timeout: float = 15.0) -> list[RepoCandidate]:
    """Discover source-code repos referenced from a video *url*.

    Uses yt-dlp to fetch metadata only (no download) and scans the
    title, description, and channel URLs. Returned list is ordered by
    trust: description hits first, then title, then channel-level.

    Raises:
        RuntimeError: If yt-dlp is not installed.
    """
    if not _HAS_YT_DLP or yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is required for repo discovery. "
            "Install with: pip install yt-dlp"
        )

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": timeout,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        logger.warning("yt-dlp metadata fetch failed for %s: %s", url, exc)
        return []

    # Order matters: things in the description are almost always the
    # author's intended repo link; channel-level fields are weaker signal.
    sources: list[tuple[str, str]] = [
        ("description", info.get("description") or ""),
        ("title", info.get("title") or ""),
        ("uploader_url", info.get("uploader_url") or ""),
        ("channel_url", info.get("channel_url") or ""),
    ]

    all_candidates: list[RepoCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for source_name, text in sources:
        for cand in find_repos_in_text(text, source_name):
            key = (cand.host, cand.owner.lower(), cand.repo.lower())
            if key in seen:
                continue
            seen.add(key)
            all_candidates.append(cand)

    logger.info(
        "Found %d unique repo candidates for %s", len(all_candidates), url
    )
    return all_candidates


__all__ = ["RepoCandidate", "find_repos_in_text", "find_repos_for_url"]
