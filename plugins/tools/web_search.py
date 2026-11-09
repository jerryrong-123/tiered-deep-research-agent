"""Web search tool using Serper API (direct or via proxy)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextvars import ContextVar, Token
from functools import lru_cache
from urllib.parse import unquote, urlsplit

import httpx

from frontier_agent.core.tool import tool
from frontier_agent.infra.config import get_config
from frontier_agent.infra.usage_meter import record_api_request
from plugins.tools._coerce import coerce_json_list
from plugins.tools._parallel_search import (
    parallel_search_batch,
    selected_search_provider,
    valid_search_provider,
)
from plugins.tools._search_contract import (
    SEARCH_ERROR_KEY as SEARCH_ERROR_KEY,
)
from plugins.tools._search_contract import (
    _SEARCH_ERROR_REASONS as _SEARCH_ERROR_REASONS,
)
from plugins.tools._search_contract import (
    _format_search_error,
    _search_failure,
)
from plugins.tools._search_run_cache import cached_search_result
from plugins.tools._single_flight import SingleFlightCoalescer

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

# Collapses concurrent identical Serper bursts (8 sibling runs firing the same
# query at once) into one upstream call; a caching search proxy in front of
# Serper handles cross-task persistence (24h TTL).
# See :mod:`plugins.tools._single_flight`.
_search_coalescer = SingleFlightCoalescer("web_search", meter_provider="serper")

# Such a proxy keys its cache on ``hash(url + body)`` — so a query whose
# only difference is ``num`` (one agent asks 8, another 10) produces a distinct
# key and misses the warm cache. Snapping ``num`` up to a coarse bucket makes
# otherwise-identical sibling queries share one cache entry. Bucketing *up*
# only ever returns >= the requested count, so the tool can slice down safely.
_NUM_BUCKETS = (10, 20, 50, 100)
_FREE_PLAN_REFUSAL = "not allowed for free accounts"
_FREE_PLAN_OPERATOR_NUM = 10


def _canonical_num(n: int) -> int:
    """Snap ``num`` up to the nearest cache-friendly bucket (max 100)."""
    n = max(1, min(int(n), 100))
    for bucket in _NUM_BUCKETS:
        if n <= bucket:
            return bucket
    return 100

# Domains with low information density for research tasks.
# Filtered from organic results before formatting (search-display only —
# web_fetch can still access them when the agent has a direct URL).
_DOMAIN_BLACKLIST: frozenset[str] = frozenset({
    "youtube.com", "youtu.be",
    "tiktok.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "pinterest.com",
    "twitter.com", "x.com",
    "linkedin.com",
    "quora.com",
    "amazon.com",
    "ebay.com",
    "walmart.com",
    "etsy.com",
    "aliexpress.com",
})


def _normalise_entry(entry: str) -> str:
    """Normalise a blocklist entry (bare domain or URL prefix).

    Strips scheme, ``#fragment``, leading ``www.`` and trailing ``/``,
    and percent-decodes the result — so an entry pasted straight from a
    browser address bar compares cleanly against search-result URLs
    normalised the same way, whichever encoding either side uses
    (``telegra.ph/%E5%A6%82...`` and ``telegra.ph/如何...`` are the
    same entry after normalisation).
    """
    e = entry.strip()
    e = e.split("://", 1)[-1]
    # Fragment split BEFORE unquote so an encoded ``%23`` in the path
    # isn't mistaken for a fragment marker.
    e = e.split("#", 1)[0]
    e = unquote(e).lower()
    e = e.lstrip(".")
    if e.startswith("www."):
        e = e[4:]
    return e.rstrip("/")


@lru_cache(maxsize=8)
def _parse_domain_list(raw: str) -> frozenset[str]:
    """Parse a comma-separated entry list into a normalised frozenset.

    Entries are either bare domains (``example.com`` — hostname suffix
    match, blocks the whole site) or URL prefixes
    (``zhihu.com/question/123`` — blocks that path and its sub-paths
    only, the rest of the site stays reachable).
    """
    return frozenset(
        _normalise_entry(d)
        for d in raw.split(",")
        if d.strip()
    )


# ── Profile-driven override ───────────────────────────────────────────
# Workflows install the profile's
# ``web_domain_blacklist:`` list at task start; ``_extra_blacklist()``
# UNIONS it with env ``WEB_DOMAIN_BLACKLIST_EXTRA`` — env entries are
# additive on top of the profile, never replaced.
#
# ContextVar — not module global — so concurrent tasks on different
# profiles don't trample each other (same pattern as
# ``frontier_agent/infra/summary_llm.py``).
_domain_blacklist_override: ContextVar[frozenset[str] | None] = ContextVar(
    "_domain_blacklist_override", default=None,
)


def set_domain_blacklist_extra(domains: list[str] | str | None) -> Token:
    """Install a profile-derived blocked-domain list; returns a reset token.

    Accepts a YAML list or a comma-separated string. Pass ``None`` /
    empty to clear (env-only for the rest of the context). Pair with
    :func:`reset_domain_blacklist_extra` in a ``finally`` block.
    """
    if not domains:
        return _domain_blacklist_override.set(None)
    raw = domains if isinstance(domains, str) else ",".join(str(d) for d in domains)
    return _domain_blacklist_override.set(_parse_domain_list(raw))


def reset_domain_blacklist_extra(token: Token) -> None:
    """Restore the previous override (paired with ``set_domain_blacklist_extra``)."""
    _domain_blacklist_override.reset(token)


def _extra_blacklist() -> frozenset[str]:
    """Operator-configured hard-block domains.

    Union of the profile override (``web_domain_blacklist:`` block,
    installed by the workflow) and env ``WEB_DOMAIN_BLACKLIST_EXTRA``.
    """
    env_domains = _parse_domain_list(get_config().web_domain_blacklist_extra)
    override = _domain_blacklist_override.get()
    return env_domains if override is None else override | env_domains


def _domain_matches(url: str, domains: frozenset[str]) -> bool:
    """True if the URL matches any blocklist entry.

    Two entry shapes, distinguished by the presence of a path/query:

    - Bare domain (``x.com``) — hostname suffix match: blocks ``x.com``
      and ``m.x.com`` but not ``netflix.com`` or a path containing
      ``x.com``.
    - URL prefix (``zhihu.com/question/123``) — normalised-prefix match
      on a path-segment boundary: blocks that page and sub-paths
      (``.../123/answer/456``) but not ``.../1234`` and not the rest of
      the site.
    """
    if not domains:
        return False
    target = url if "://" in url else f"https://{url}"
    host = (urlsplit(target).hostname or "").lower()
    if not host:
        return False
    norm = _normalise_entry(url)
    for d in domains:
        if "/" in d or "?" in d:
            # URL-prefix entry. Entries carrying a query string match by
            # plain prefix (the query itself is the discriminator);
            # path-only entries require a segment boundary so an id that
            # happens to prefix a longer id can't false-positive.
            if norm == d or (
                norm.startswith(d)
                if "?" in d
                else norm.startswith((d + "/", d + "?"))
            ):
                return True
        elif host == d or host.endswith("." + d):
            return True
    return False


def is_domain_blocked(url: str) -> bool:
    """Operator hard-block check — shared with ``web_fetch``.

    Only consults ``WEB_DOMAIN_BLACKLIST_EXTRA`` (manual blocks), not the
    built-in low-value filter, so fetch behaviour for the built-in list
    is unchanged.
    """
    return _domain_matches(url, _extra_blacklist())


_low_value_results_visible: ContextVar[bool] = ContextVar(
    "_low_value_results_visible", default=False,
)


def set_low_value_results_visible(visible: bool) -> Token:
    """Show the built-in low-value domains in search display for this context.

    Operator hard blocks still apply. Pair with
    :func:`reset_low_value_results_visible` in a ``finally`` block.
    """
    return _low_value_results_visible.set(visible)


def reset_low_value_results_visible(token: Token) -> None:
    """Restore the previous visibility (paired with the setter above)."""
    _low_value_results_visible.reset(token)


def _combined_blacklist() -> frozenset[str]:
    """Built-in low-value filter ∪ operator extra — hoist out of
    per-result loops (the union rebuilds a set each call)."""
    builtin = frozenset() if _low_value_results_visible.get() else _DOMAIN_BLACKLIST
    return builtin | _extra_blacklist()


# ── Domain-scoped snippet/title content filter ────────────────────────
# A URL blacklist needs every offending page enumerated; the same
# narrative often resurfaces on new sub-pages of one site (e.g. fresh
# zhihu ``/people/<id>`` profiles or ``/question/<id>`` threads). This
# filter drops a result whose host matches a configured domain AND whose
# title/snippet contains a configured phrase — catching the long tail
# without touching URLs the operator never listed. Scoped to named
# domains so it can never black-hole the open web. Search-display only;
# web_fetch is unaffected (it has no snippet, and the URL blacklist
# already gates it).


@lru_cache(maxsize=8)
def _parse_phrase_list(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated phrase list into lowercased, non-empty terms."""
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def _snippet_content_blocked(host: str, title: str, snippet: str) -> bool:
    """True if ``host`` is a configured domain and the title/snippet hits a phrase."""
    cfg = get_config()
    phrases = _parse_phrase_list(cfg.web_snippet_block_phrases)
    if not phrases:
        return False
    domains = _parse_domain_list(cfg.web_snippet_block_domains)
    if not domains or not host:
        return False
    # Bare-domain (hostname-suffix) match only — path entries are
    # meaningless for a content filter keyed on the host.
    if not any(
        host == d or host.endswith("." + d) for d in domains if "/" not in d
    ):
        return False
    hay = f"{title}\n{snippet}".lower()
    return any(p in hay for p in phrases)


def is_snippet_blocked_result(result: dict) -> bool:
    """Apply the env-configured domain + title/snippet phrase filter.

    This public seam lets alternate ``web_search`` implementations preserve
    their own URL filtering and output format while still honoring
    ``WEB_SNIPPET_BLOCK_DOMAINS`` + ``WEB_SNIPPET_BLOCK_PHRASES``.
    """
    link = result.get("link", "")
    target = link if "://" in link else f"https://{link}"
    host = (urlsplit(target).hostname or "").lower()
    return _snippet_content_blocked(
        host,
        result.get("title", ""),
        result.get("snippet", ""),
    )


def _result_filtered(r: dict, blocked: frozenset[str]) -> bool:
    """True if an organic result must be hidden from search display.

    Combines the URL blacklist (built-in low-value ∪ operator extra) with
    the domain-scoped snippet/title content filter.
    """
    link = r.get("link", "")
    if _domain_matches(link, blocked):
        return True
    return is_snippet_blocked_result(r)


async def raw_web_search(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    tbs: str = "",
) -> dict:
    """Execute search and return full Serper response dict.

    Returns dict with keys: organic, answerBox, knowledgeGraph, peopleAlsoAsk, etc.
    Failures return the reserved ``SEARCH_ERROR_KEY`` metadata described above;
    an empty successful response has no error field. No failure is raised to
    raw dictionary consumers.
    Retries on 429/5xx with exponential backoff.

    ``num_results`` is snapped up to a coarse bucket so that otherwise-identical
    sibling queries hit the same proxy cache entry, and concurrent identical
    bursts are coalesced into one upstream call. Callers that need an exact
    count slice the returned ``organic`` list themselves.
    """
    num = _canonical_num(num_results)
    # Key mirrors the canonical request body — NEVER add control fields here or
    # to the body itself: the proxy cache key is ``hash(url + body)``.
    key = f"{query}\x1f{num}\x1f{gl}\x1f{hl}\x1f{tbs}"
    config = get_config()
    cache_key = (
        "serper", config.serper_base_url,
        hashlib.sha256((config.serper_api_key or "").encode()).digest(),
        query, num, gl, hl, tbs,
    )
    return await cached_search_result(
        cache_key,
        lambda: _search_coalescer.run(
            key, lambda: _do_web_search(query, num, gl, hl, tbs),
        ),
        on_cache_hit=lambda: record_api_request("serper", requests=0, cache_hits=1),
    )


async def _do_web_search(
    query: str,
    num_results: int,
    gl: str,
    hl: str,
    tbs: str,
) -> dict:
    """Single Serper round-trip with retry. See :func:`raw_web_search`."""
    config = get_config()

    if not config.serper_api_key:
        return _search_failure("configuration_error")

    url = f"{config.serper_base_url}/search"
    headers = {"X-API-KEY": config.serper_api_key, "Content-Type": "application/json"}

    payload: dict = {
        "q": query,
        "num": num_results,
        "gl": gl,
        "hl": hl,
    }
    if tbs:
        payload["tbs"] = tbs

    last_failure = _search_failure("unexpected_error")
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, headers=headers, json=payload)

                if resp.status_code == 429:
                    last_failure = _search_failure("rate_limited", resp.status_code)
                    wait = 2 ** attempt
                    logger.warning("Serper 429 rate limited, retrying in %ds (attempt %d)", wait, attempt + 1)
                    record_api_request("serper", requests=0, retries=1)
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    last_failure = _search_failure("server_error", resp.status_code)
                    wait = 2 ** attempt
                    logger.warning("Serper %d server error, retrying in %ds (attempt %d)", resp.status_code, wait, attempt + 1)
                    record_api_request("serper", requests=0, retries=1)
                    await asyncio.sleep(wait)
                    continue

                if (resp.status_code == 400 and payload["num"] > _FREE_PLAN_OPERATOR_NUM
                        and _FREE_PLAN_REFUSAL in resp.text):
                    # Free accounts refuse quoted or site: queries above this
                    # size, so only the requested size changes on this retry.
                    payload["num"] = _FREE_PLAN_OPERATOR_NUM
                    last_failure = _search_failure("http_error", resp.status_code)
                    logger.warning("Serper refused this query size, retrying with num=%d", _FREE_PLAN_OPERATOR_NUM)
                    record_api_request("serper", requests=0, retries=1)
                    continue

                resp.raise_for_status()
                # Count one answered, billable Serper query.
                record_api_request("serper")
                try:
                    data = resp.json()
                except ValueError:
                    logger.error("Serper search failed (invalid_response)")
                    return _search_failure("invalid_response")
                if not isinstance(data, dict):
                    logger.error("Serper search failed (invalid_response)")
                    return _search_failure("invalid_response")
                provider_status = data.get("statusCode")
                if data.get("error") or (
                    type(provider_status) is int and 400 <= provider_status <= 599
                ):
                    # A provider may reject a query inside an HTTP 200 payload.
                    # Keep the actual transport status and discard its message.
                    logger.error("Serper search failed (provider_error; HTTP %d)", resp.status_code)
                    record_api_request("serper", requests=0, errors=1)
                    return _search_failure("provider_error", resp.status_code)
                return data

        except httpx.TimeoutException:
            logger.warning("Serper search failed (timeout; attempt %d)", attempt + 1)
            record_api_request("serper", requests=0, errors=1)
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            return _search_failure("timeout")

        except httpx.HTTPStatusError as e:
            logger.error("Serper search failed (http_error; HTTP %d)", e.response.status_code)
            record_api_request("serper", requests=0, errors=1)
            return _search_failure("http_error", e.response.status_code)

        except httpx.RequestError:
            logger.error("Serper search failed (transport_error)")
            record_api_request("serper", requests=0, errors=1)
            return _search_failure("transport_error")

        except Exception:
            logger.error("Serper search failed (unexpected_error)")
            record_api_request("serper", requests=0, errors=1)
            return _search_failure("unexpected_error")

    logger.error("Serper search exhausted retries")
    return last_failure


def _format_results(data: dict, max_organic: int | None = None) -> str:
    """Format Serper response into readable text with all result types.

    ``max_organic`` caps the number of organic results displayed — used to
    honour the caller's requested ``num_results`` after the upstream call was
    widened to a cache-friendly bucket (see :func:`raw_web_search`).
    """
    if SEARCH_ERROR_KEY in data:
        return _format_search_error(data)
    parts = []

    # Answer Box (direct answer — highest priority)
    answer_box = data.get("answerBox")
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        title = answer_box.get("title", "")
        if answer:
            parts.append(f"## Direct Answer\n**{title}**\n{answer}")

    # Knowledge Graph (entity info)
    kg = data.get("knowledgeGraph")
    if kg:
        kg_title = kg.get("title", "")
        kg_type = kg.get("type", "")
        kg_desc = kg.get("description", "")
        kg_parts = [f"## Knowledge Graph: {kg_title}"]
        if kg_type:
            kg_parts.append(f"Type: {kg_type}")
        if kg_desc:
            kg_parts.append(kg_desc)
        for key in ("founded", "headquarters", "ceo", "revenue", "employees",
                     "website", "born", "died", "nationality"):
            if key in kg:
                kg_parts.append(f"{key.title()}: {kg[key]}")
        parts.append("\n".join(kg_parts))

    # Organic Results (filtered: skip low-value domains)
    organic = data.get("organic", [])
    if organic:
        results = []
        filtered_count = 0
        idx = 0
        blocked = _combined_blacklist()
        for r in organic:
            if max_organic is not None and idx >= max_organic:
                break
            link = r.get("link", "")
            if _result_filtered(r, blocked):
                filtered_count += 1
                continue
            idx += 1
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            results.append(f"[{idx}] **{title}**\n{snippet}\nURL: {link}")
        if results:
            header = "## Search Results"
            if filtered_count:
                header += f"\n({filtered_count} low-value results filtered)"
            parts.append(header + "\n" + "\n\n".join(results))
        elif filtered_count:
            parts.append(
                f"## Search Results\nAll {filtered_count} results were from "
                f"low-value domains (video/social). Try a more specific query."
            )

    return "\n\n---\n\n".join(parts) if parts else "No results found."


@tool
async def web_search(
    q: str | list[str],
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    tbs: str = "",
) -> str:
    """Search the web for information about a topic.

    ``q`` accepts a single string or a list of strings. List queries
    dispatch in parallel and merge into one result, deduplicated by URL —
    one turn can cover multiple angles without burning extra steps.

    Args:
        q: A query string, or a list of strings for parallel dispatch.
        num_results: Organic results per query (default 10, max 100).
        gl: Country/region code (default "us"). Examples: "cn", "uk", "de".
        hl: Language code (default "en"). Examples: "zh", "es", "fr".
        tbs: Time filter. Examples: "qdr:h" / "qdr:d" / "qdr:w" / "qdr:m" / "qdr:y".

    Returns:
        Formatted results with answers, knowledge graph, and organic links.
        Multi-query calls return one block per query separated by ``---``.
    """
    queries = _normalise_queries(q)
    if not queries:
        return "Error: search query cannot be empty."
    num_results = max(1, min(num_results, 100))

    from plugins.tools._overflow import maybe_overflow

    provider = selected_search_provider()
    if not valid_search_provider(provider):
        return "Error: WEB_SEARCH_PROVIDER must be 'serper' or 'parallel'."
    if provider == "parallel":
        datas = await parallel_search_batch(
            queries,
            num_results=num_results,
            gl=gl,
            hl=hl,
            tbs=tbs,
        )
        if isinstance(datas, str):
            return f"Error: {datas}"
        if not datas:
            return f"No results found for: {queries[0]}"
        if len(datas) == 1:
            return maybe_overflow(
                "web_search",
                _format_results(
                    {**datas[0], "organic": _dedupe_display_organic(
                        datas[0].get("organic") or [], set(), num_results,
                    )},
                    max_organic=num_results,
                ),
            )
        return maybe_overflow(
            "web_search",
            _format_parallel_query_results(queries, datas, num_results),
        )

    if len(queries) == 1:
        data = await raw_web_search(queries[0], num_results, gl, hl, tbs)
        if not data:
            return f"No results found for: {queries[0]}"
        return maybe_overflow("web_search", _format_results(data, max_organic=num_results))

    return maybe_overflow(
        "web_search",
        await _run_parallel_queries(queries, num_results, gl, hl, tbs),
    )


def _format_parallel_query_results(
    queries: list[str], datas: list[dict], max_results: int,
) -> str:
    """Keep the canonical query-labelled output for Parallel MCP batches."""
    seen_urls: set[str] = set()
    blocks: list[str] = []
    for query, data in zip(queries, datas, strict=False):
        if not data:
            blocks.append(f"## Query: {query}\n\nNo results found.")
            continue
        organic = _dedupe_display_organic(
            data.get("organic") or [], seen_urls, max_results,
        )
        blocks.append(
            f"## Query: {query}\n\n"
            f"{_format_results({**data, 'organic': organic}, max_organic=max_results)}"
        )
    return "\n\n---\n\n".join(blocks)


def _normalise_queries(query: str | list[str]) -> list[str]:
    """Collapse the LangChain payload into a clean list of non-empty strings.

    Handles three shapes: a single string, a list, and the JSON-encoded
    list some models emit when the schema is ``str | list[str]``.
    """
    coerced = coerce_json_list(query) if isinstance(query, str) else query
    if isinstance(coerced, list):
        return [q.strip() for q in coerced if isinstance(q, str) and q.strip()]
    if isinstance(coerced, str) and coerced.strip():
        return [coerced.strip()]
    return []


async def _run_parallel_queries(
    queries: list[str], num_results: int, gl: str, hl: str, tbs: str,
) -> str:
    """Dispatch every query in parallel and merge with URL de-duplication."""
    datas = await asyncio.gather(
        *(raw_web_search(q, num_results, gl, hl, tbs) for q in queries),
        return_exceptions=True,
    )
    seen_urls: set[str] = set()
    blocks: list[str] = []
    for q, data in zip(queries, datas, strict=False):
        if isinstance(data, BaseException):
            blocks.append(f"## Query: {q}\n\n{_format_search_error(_search_failure('unexpected_error'))}")
            continue
        deduped = _dedupe_display_organic(data.get("organic") or [], seen_urls, num_results)
        blocks.append(
            f"## Query: {q}\n\n"
            f"{_format_results({**data, 'organic': deduped}, max_organic=num_results)}"
        )
    return "\n\n---\n\n".join(blocks)


def _dedupe_display_organic(
    organic: list[dict],
    seen_urls: set[str],
    max_organic: int,
) -> list[dict]:
    """Deduplicate only the organic results that can be displayed."""
    deduped = []
    displayed = 0
    blocked = _combined_blacklist()
    for item in organic:
        if displayed >= max_organic:
            break

        if _result_filtered(item, blocked):
            deduped.append(item)
            continue

        url = item.get("link") or item.get("url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped.append(item)
        displayed += 1
    return deduped
