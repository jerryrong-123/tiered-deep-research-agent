"""Offline checks for saved copies of pages and for single-word quotes."""
from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from local_runner.answer_label_guard import FETCH_WITHHELD, MARKER
from workflows.agent_team.convergence import guard_tools
from workflows.agent_team.progressive import ProgressivePolicy, unquote_single_words

try:
    from tests.test_tiered_research_v1 import TIER, _run
except ImportError:
    from test_tiered_research_v1 import TIER, _run

archive = importlib.import_module("plugins.tools._archive_copy")
fetch = importlib.import_module("plugins.tools.web_fetch_aligned")

STAMP = "20260730060433"
PAGE = "The Harbor Lane table lists every player, team and block count for the tournament. " * 12
FOUND = {"success": True, "extracted_info": "Leon blocked 4 of 27.", "error": ""}
MISSING = {"success": True, "extracted_info": "The block table is not available in the provided content.", "error": ""}


def _scraped(content):
    return {"success": True, "content": content, "error": ""}


FAILED = {"success": False, "content": "", "error": "Client error '403 Forbidden'"}


@pytest.fixture
def web(monkeypatch):
    """A page that answers from the given reader body; nothing leaves the process."""
    monkeypatch.setattr(fetch, "non_public_url_error", AsyncMock(return_value=""))
    monkeypatch.setattr(archive, "newest_snapshot", AsyncMock(return_value=STAMP))
    monkeypatch.setattr(fetch, "_scrape_url_with_python", AsyncMock(return_value=_scraped(f"<html><body><p>{PAGE}</p></body></html>")))
    monkeypatch.setattr(fetch, "_scrape_url_with_jina", AsyncMock(return_value=_scraped("Title: Short\n\nShort page.")))
    monkeypatch.setattr(fetch, "_extract_info_with_llm", AsyncMock(return_value=FOUND))
    return fetch


async def _read(url, focus="the block table", **config):
    with _run(**TIER, archive_reads=10, **config) as state:
        result = await fetch._fetch_single(url, focus)
    return result, state


@pytest.mark.asyncio
async def test_without_the_setting_nothing_changes(web):
    with _run(**TIER):
        result = await web._fetch_single("https://example.org/off", "the block table")
    assert result.startswith("[POSSIBLY NOT RENDERED]")
    archive.newest_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_outside_a_run_nothing_changes(web):
    result = await web._fetch_single("https://example.org/outside", "the block table")
    assert result.startswith("[POSSIBLY NOT RENDERED]")
    archive.newest_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_short_page_is_replaced_by_its_saved_copy(web):
    result, state = await _read("https://example.org/short")
    assert result.startswith("Leon blocked 4 of 27.")
    assert "[ARCHIVE COPY]" in result and "saved on 2026-07-30" in result
    assert f"https://web.archive.org/web/{STAMP}id_/https://example.org/short" in result
    assert "POSSIBLY NOT RENDERED" not in result
    assert state.snapshot()["archive_reads"] == 1 and state.light_calls == 1


@pytest.mark.asyncio
async def test_a_failed_page_is_replaced_by_its_saved_copy(web):
    web._scrape_url_with_jina.return_value = FAILED
    web._scrape_url_with_python.side_effect = [FAILED, _scraped(f"<html><body><p>{PAGE}</p></body></html>")]
    result, _ = await _read("https://example.org/failed")
    assert result.startswith("Leon blocked 4 of 27.") and "[ARCHIVE COPY]" in result


@pytest.mark.asyncio
async def test_a_failed_page_with_no_copy_keeps_its_error(web):
    web._scrape_url_with_jina.return_value = FAILED
    web._scrape_url_with_python.return_value = FAILED
    archive.newest_snapshot.return_value = None
    result, _ = await _read("https://example.org/gone")
    assert result.startswith("[ERROR]: Scraping failed")


@pytest.mark.asyncio
async def test_the_extractor_saying_the_fact_is_missing_tries_the_copy(web):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    web._extract_info_with_llm.side_effect = [MISSING, FOUND]
    result, _ = await _read("https://example.org/blank-data")
    assert result.startswith("Leon blocked 4 of 27.") and "[ARCHIVE COPY]" in result


@pytest.mark.asyncio
async def test_a_copy_that_also_lacks_the_fact_leaves_the_live_result(web):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    web._extract_info_with_llm.side_effect = [MISSING, MISSING]
    result, state = await _read("https://example.org/also-blank")
    assert result == MISSING["extracted_info"]
    assert state.snapshot()["archive_reads"] == 1


@pytest.mark.asyncio
async def test_a_copy_of_a_failed_page_stands_even_without_the_fact(web):
    web._extract_info_with_llm.side_effect = [MISSING, MISSING]
    result, _ = await _read("https://example.org/blocked")
    assert result.startswith(MISSING["extracted_info"]) and "[ARCHIVE COPY]" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["", "   ", None])
async def test_an_empty_extraction_of_a_copy_is_never_delivered(web, empty):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    web._extract_info_with_llm.side_effect = [MISSING, {"success": True, "extracted_info": empty, "error": ""}]
    result, state = await _read(f"https://example.org/empty-{empty!r}")
    assert result == MISSING["extracted_info"]
    assert state.snapshot()["archive_reads"] == 1


@pytest.mark.asyncio
async def test_an_empty_extraction_of_a_copy_leaves_a_failed_page_flagged(web):
    live = {"success": True, "extracted_info": "A short page.", "error": ""}
    web._extract_info_with_llm.side_effect = [live, {"success": True, "extracted_info": "", "error": ""}]
    result, _ = await _read("https://example.org/empty-copy")
    assert result.startswith("[POSSIBLY NOT RENDERED]") and "[ARCHIVE COPY]" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["", None])
async def test_an_extractor_that_gave_nothing_tries_the_copy(web, empty):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    web._extract_info_with_llm.side_effect = [{"success": True, "extracted_info": empty, "error": ""}, FOUND]
    result, _ = await _read(f"https://example.org/nothing-{empty!r}")
    assert result.startswith("Leon blocked 4 of 27.") and "[ARCHIVE COPY]" in result


@pytest.mark.asyncio
async def test_without_the_setting_a_null_extraction_and_a_null_focus_stay_harmless(web):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    web._extract_info_with_llm.return_value = {"success": True, "extracted_info": None, "error": ""}
    with _run(**TIER):
        assert await web._fetch_single("https://example.org/null-extraction", "the table") is None
    web._extract_info_with_llm.return_value = FOUND
    assert await web._fetch_single("https://example.org/null-focus", None) == FOUND["extracted_info"]


@pytest.mark.asyncio
async def test_a_page_that_answers_needs_no_copy(web):
    web._scrape_url_with_jina.return_value = _scraped(PAGE)
    result, state = await _read("https://example.org/fine")
    assert result == FOUND["extracted_info"]
    archive.newest_snapshot.assert_not_awaited()
    assert "archive_reads" not in state.snapshot()


@pytest.mark.asyncio
async def test_a_raw_read_has_no_extractor_to_report_a_miss(web):
    raw = "This item is not available in your region. " + PAGE
    web._scrape_url_with_jina.return_value = _scraped(raw)
    web._extract_info_with_llm.return_value = {"success": True, "extracted_info": raw, "error": ""}
    result, _ = await _read("https://example.org/raw", focus="")
    assert result == raw
    archive.newest_snapshot.assert_not_awaited()


@pytest.mark.parametrize("url", [
    "https://html.duckduckgo.com/html/?q=first+novel",
    "https://www.mojeek.com/search?q=first+novel",
    "https://en.wikipedia.org/w/index.php?search=novel",
    "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=novel",
    "https://blog.example.org/?s=grief",
    "https://digital.example.org/iiif/full/0/default.jpg",
    "https://web.archive.org/web/2021/https://example.org/a",
    "http://archive.org/wayback/available?url=example.org",
])
def test_search_results_images_and_saved_copies_are_not_tried(url):
    assert not archive.worth_a_saved_copy(url)


@pytest.mark.parametrize("url", [
    "https://en.volleyballworld.com/volleyball/competitions/olympics-2020/statistics/men/best-blockers/",
    "https://goodluckcharlie.fandom.com/wiki/Go_Teddy!",
    "https://example.org/item.php?id=12",
    "https://example.org/thesis.pdf",
])
def test_ordinary_pages_are_tried(url):
    assert archive.worth_a_saved_copy(url)


@pytest.mark.asyncio
async def test_a_run_stops_at_its_limit_and_looks_each_page_up_once(web):
    with _run(**TIER, archive_reads=1) as state:
        first = await web._fetch_single("https://example.org/first", "the block table")
        second = await web._fetch_single("https://example.org/second", "the block table")
    assert "[ARCHIVE COPY]" in first and second.startswith("[POSSIBLY NOT RENDERED]")
    assert archive.newest_snapshot.await_count == 1 and state.snapshot()["archive_reads"] == 1


@pytest.mark.asyncio
async def test_no_snapshot_is_remembered_within_the_run(web):
    archive.newest_snapshot.return_value = None
    with _run(**TIER, archive_reads=10) as state:
        for _ in range(2):
            await web._fetch_single("https://example.org/twice", "the block table")
    assert archive.newest_snapshot.await_count == 1 and "archive_reads" not in state.snapshot()


@pytest.mark.asyncio
async def test_a_thin_copy_is_not_used(web):
    web._scrape_url_with_python.return_value = _scraped("<html><body><p>Just a moment.</p></body></html>")
    result, _ = await _read("https://example.org/thin")
    assert result.startswith("[POSSIBLY NOT RENDERED]")


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [FAILED, _scraped("%PDF-1.6 raw bytes " + "x" * 2000)])
async def test_a_saved_pdf_goes_through_the_reader(web, direct):
    web._scrape_url_with_python.return_value = direct
    web._scrape_url_with_jina.side_effect = [_scraped("Title: Short\n\nShort."), _scraped("Title: Short\n\nShort."),
                                             _scraped(PAGE)]
    result, _ = await _read("https://example.org/thesis.pdf")
    assert "[ARCHIVE COPY]" in result
    assert web._scrape_url_with_jina.await_args.args[0] == f"https://web.archive.org/web/{STAMP}id_/https://example.org/thesis.pdf"
    assert web._extract_info_with_llm.await_args.args[0] == PAGE


@pytest.mark.asyncio
async def test_a_hung_copy_gives_up_and_keeps_the_live_result(web, monkeypatch):
    monkeypatch.setattr(fetch, "_ARCHIVE_READ_TIMEOUT_S", 0.01)

    async def hang(*_):
        import asyncio
        await asyncio.sleep(1)

    web._scrape_url_with_python = hang
    monkeypatch.setattr(fetch, "_scrape_url_with_python", hang)
    result, _ = await _read("https://example.org/hung")
    assert result.startswith("[POSSIBLY NOT RENDERED]")


@pytest.mark.asyncio
async def test_a_withheld_copy_still_reads_as_withheld_to_the_guard(web):
    web._extract_info_with_llm.return_value = {"success": True, "extracted_info": MARKER, "error": ""}
    with _run(**TIER, archive_reads=10):
        result = await web.web_fetch_aligned.func(url="https://example.org/withheld", info_to_extract="the table")
    assert FETCH_WITHHELD.findall(result) == ["https://example.org/withheld"]


@pytest.mark.asyncio
async def test_the_lookup_reads_the_newest_snapshot(monkeypatch):
    body = {"archived_snapshots": {"closest": {"status": "200", "available": True, "timestamp": STAMP,
                                               "url": "http://web.archive.org/x"}}}

    async def send(client, request, **_):
        assert request.url.host == "archive.org" and request.url.params["url"] == "https://example.org/a"
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    assert await archive.newest_snapshot("https://example.org/a") == STAMP


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [
    httpx.Response(200, json={"archived_snapshots": {}}),
    httpx.Response(200, json={"archived_snapshots": {"closest": {"available": False, "timestamp": STAMP}}}),
    httpx.Response(200, json={"archived_snapshots": {"closest": {"available": True, "timestamp": "2026"}}}),
    httpx.Response(200, text="Internet Archive services are temporarily offline."),
    httpx.Response(503, text="offline"),
])
async def test_anything_but_a_clear_snapshot_counts_as_no_copy(monkeypatch, reply):
    async def send(client, request, **_):
        reply.request = request
        return reply

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    assert await archive.newest_snapshot("https://example.org/a") is None


@pytest.mark.asyncio
async def test_a_blocked_network_counts_as_no_copy(monkeypatch):
    monkeypatch.setattr(httpx.AsyncClient, "send", AsyncMock(side_effect=RuntimeError("offline_network_blocked")))
    assert await archive.newest_snapshot("https://example.org/a") is None


def test_page_text_keeps_a_table_and_a_short_panel():
    rows = "".join(f"<tr><td>{n}</td><td>Player {n}</td><td>{n * 3}</td></tr>" for n in range(1, 40))
    table = f"<html><body><article><h1>Best blockers</h1><table>{rows}</table></article></body></html>"
    assert "Player 39" in archive.page_text(table)
    panel = "<html><body><div><span>Block Points 4</span><span>Success 14.81%</span></div></body></html>"
    assert "Success 14.81%" in archive.page_text(panel)


# Single-word quotes ---------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ('"physiotherapist" "completed her first novel" 2005', 'physiotherapist "completed her first novel" 2005'),
    ('"completed her first novel" 2005 physiotherapist', '"completed her first novel" 2005 physiotherapist'),
    ('"novelist" "blog" grief', "novelist blog grief"),
    ('"well-known" poet', "well-known poet"),
    ('"Wootton" "2005" "14.81"', '"Wootton" "2005" "14.81"'),
    ('"first novel" site:example.org', '"first novel" site:example.org'),
    ("no quotes here", "no quotes here"),
])
def test_only_single_ordinary_words_lose_their_quotes(query, expected):
    assert unquote_single_words(query) == expected


def test_each_query_in_a_list_is_rewritten():
    assert unquote_single_words(['"a" b', '"c d"']) == ["a b", '"c d"']


def _search_tool():
    from frontier_agent.core.tool import tool

    @tool
    async def web_search(q: str | list[str]) -> str:
        """Return what was searched."""
        return repr(q)

    return web_search


@pytest.mark.asyncio
async def test_the_switch_rewrites_the_search_and_keeps_the_count():
    with _run(**TIER, unquote_single_words=True) as state:
        search = guard_tools([_search_tool()], coordinator=False)[0]
        assert await search.func(q=['"word" one', '"two words"']) == repr(["word one", '"two words"'])
        assert await search.func(q='"word" three') == repr("word three")
        assert state.progressive.search_queries == 3


@pytest.mark.asyncio
async def test_a_search_sent_as_json_text_is_rewritten_too():
    with _run(**TIER, unquote_single_words=True):
        search = guard_tools([_search_tool()], coordinator=False)[0]
        assert await search.func(q='["\\"word\\" one"]') == repr(["word one"])


@pytest.mark.asyncio
async def test_without_the_switch_the_search_is_left_alone():
    with _run(**TIER):
        search = guard_tools([_search_tool()], coordinator=False)[0]
        assert await search.func(q='"word" one') == repr('"word" one')


# Settings -------------------------------------------------------------------

@pytest.mark.parametrize("config", [
    {"archive_reads": -1}, {"archive_reads": True}, {"archive_reads": "10"}, {"unquote_single_words": 1},
])
def test_invalid_settings_are_refused(config):
    with pytest.raises(ValueError, match="invalid_tier_config"):
        ProgressivePolicy({**TIER, **config})


def test_both_rules_are_off_by_default():
    tier = ProgressivePolicy(TIER).tier
    assert tier["archive_reads"] == 0 and tier["unquote_single_words"] is False


def test_the_allowance_lasts_for_its_run_only():
    with _run(**TIER, archive_reads=3) as state:
        assert archive.current_archive_reads() is state.archive_reads
        assert state.archive_reads.limit == 3
    assert archive.current_archive_reads() is None
    with _run(**TIER) as state:
        assert archive.current_archive_reads() is None and state.archive_reads is None
