"""Reuse raw search responses only within one explicitly enabled research run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Hashable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from plugins.tools._search_contract import SEARCH_ERROR_KEY


@dataclass(slots=True)
class _RunCache:
    results: dict[Hashable, dict[str, Any]] = field(default_factory=dict)
    active: bool = True


_run_cache: ContextVar[_RunCache | None] = ContextVar(
    "search_run_cache", default=None,
)


def _has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_content(item) for item in value)
    return False


@contextmanager
def search_run_cache(*, enabled: bool = False) -> Iterator[None]:
    """Create a fresh run cache; children inherit it and nested runs isolate it."""
    cache = _RunCache() if enabled is True else None
    token = _run_cache.set(cache)
    try:
        yield
    finally:
        if cache is not None:
            # Children may outlive cancellation of the root. Invalidate their
            # inherited reference too, including calls already awaiting I/O.
            cache.active = False
            cache.results.clear()
        _run_cache.reset(token)


async def cached_search_result(
    key: Hashable,
    search: Callable[[], Awaitable[dict[str, Any]]],
    *,
    on_cache_hit: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Reuse a completed raw response; leave single-flight and retries to search.

    The caller supplies the complete provider/request key without semantic
    normalization. ``on_cache_hit`` observes only completed-cache reuse;
    existing single-flight followers retain their own accounting.
    """
    cache = _run_cache.get()
    if cache is None or not cache.active:
        return await search()
    if key in cache.results:
        if on_cache_hit is not None:
            on_cache_hit()
        return deepcopy(cache.results[key])
    result = await search()
    if cache.active and SEARCH_ERROR_KEY not in result and not result.get("error") and any(
        _has_content(result.get(field))
        for field in (
            "organic", "answerBox", "knowledgeGraph", "peopleAlsoAsk", "relatedSearches",
        )
    ):
        cache.results[key] = deepcopy(result)
    return deepcopy(result)
