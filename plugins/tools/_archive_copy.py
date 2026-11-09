"""Find a saved copy of a page the live web did not deliver.

A live page can fail in ways a renderer cannot fix: a security check, or a page
that still loads after its data has gone. The Internet Archive's lookup and its
saved pages need no key, so trying one costs a moment and one extraction.

Off unless a profile sets ``archive_reads``. ``convergence_run`` opens the
allowance for one run, with the same inheritance as the run itself, so helpers
share it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from plugins.tools._render_check import _visible_body_text

logger = logging.getLogger(__name__)

_LOOKUP_URL = "https://archive.org/wayback/available"
_LOOKUP_TIMEOUT_S = 15
_STAMP = re.compile(r"\d{14}")
_ARCHIVE_HOSTS = ("archive.org", "archive.ph", "archive.today", "archive.is")
# A saved search result answers nothing, and an image holds no text to extract.
_SEARCH_QUERY = re.compile(r"[?&](?:q|query|s|search|srsearch)=", re.IGNORECASE)
_IMAGE_PATH = re.compile(r"\.(?:jpe?g|png|gif|webp|svg|ico)$", re.IGNORECASE)
# The extractor's own words for "this page does not have it". It opens with them
# when it has nothing, and a page that merely lacks one detail says so further on.
_MISSING_HEAD_CHARS = 400
_REPORTS_MISSING = re.compile(
    r"not available|not (?:found|provided|present|included|listed|stated|mentioned|given|contained)"
    r"|does(?: not|n't) (?:contain|include|mention|provide|state|list)"
    r"|no (?:information|mention|data|details|relevant)"
    r"|cannot be (?:found|determined)|unable to (?:find|locate)",
    re.IGNORECASE,
)


def worth_a_saved_copy(url: str) -> bool:
    """Skip search results, images and pages that already are saved copies."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if any(host == name or host.endswith(f".{name}") for name in _ARCHIVE_HOSTS):
        return False
    return not (_SEARCH_QUERY.search(url) or _IMAGE_PATH.search(parts.path))


def reports_missing(extracted: str | None) -> bool:
    """Whether the extractor gave nothing or opens by saying the page lacks it."""
    text = (extracted or "").strip()
    return not text or bool(_REPORTS_MISSING.search(text[:_MISSING_HEAD_CHARS]))


def copy_url(stamp: str, url: str) -> str:
    """The saved page itself: ``id_`` leaves out the archive's own toolbar."""
    return f"https://web.archive.org/web/{stamp}id_/{url}"


def saved_on(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def page_text(html: str) -> str:
    """The saved page's text, by whichever reading keeps more of it.

    The table-aware extractor keeps rows that the plain reading runs together,
    and it drops the short panels of figures that the plain reading keeps. Both
    are far smaller than the HTML, which is what the extraction model pays for.
    """
    visible = _visible_body_text(html)
    try:
        import trafilatura
    except ImportError:
        return visible
    structured = trafilatura.extract(html, include_tables=True) or ""
    return structured if len(structured) > len(visible) else visible


async def newest_snapshot(url: str) -> str | None:
    """Timestamp of the newest saved copy. A failed lookup counts as no copy."""
    try:
        async with httpx.AsyncClient(timeout=_LOOKUP_TIMEOUT_S) as client:
            response = await client.get(_LOOKUP_URL, params={"url": url})
        response.raise_for_status()
        closest = (response.json().get("archived_snapshots") or {}).get("closest") or {}
    except Exception as exc:
        logger.warning("Archive lookup failed for %s: %s", url[:80], exc)
        return None
    stamp = str(closest.get("timestamp") or "")
    return stamp if closest.get("available") and _STAMP.fullmatch(stamp) else None


@dataclass
class ArchiveReads:
    """One run's allowance of saved pages, shared with its helpers."""

    limit: int
    on_read: Callable[[], None]
    used: int = 0
    _snapshots: dict[str, str | None] = field(default_factory=dict)

    @property
    def spent(self) -> bool:
        return self.used >= self.limit

    async def snapshot(self, url: str) -> str | None:
        """Look each page up once: the answer does not change within a run."""
        if url not in self._snapshots:
            self._snapshots[url] = await newest_snapshot(url)
        return self._snapshots[url]

    def take(self) -> bool:
        """Spend one read. Like any page read, it counts as one light call."""
        if self.spent:
            return False
        self.used += 1
        self.on_read()
        return True


_allowance: ContextVar[ArchiveReads | None] = ContextVar("archive_reads", default=None)


def current_archive_reads() -> ArchiveReads | None:
    return _allowance.get()


def open_archive_reads(allowance: ArchiveReads) -> Token:
    """Allow saved pages in this context; pair with :func:`close_archive_reads`."""
    return _allowance.set(allowance)


def close_archive_reads(token: Token) -> None:
    _allowance.reset(token)
