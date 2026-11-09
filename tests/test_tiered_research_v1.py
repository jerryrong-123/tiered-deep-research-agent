"""Offline checks for opt-in tiered research: light path first, team when needed."""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from local_runner.answer_label_guard import MARKER, SEEK_MARKER, AnswerLabelGuard, install_guard
from workflows.agent_team.convergence import (
    ESCALATE_STOP, ConvergenceBlocked, convergence_run, current_convergence, guard_tools,
)
from workflows.agent_team.observers.bare_text_finalize import BareTextFinalizeObserver
from workflows.agent_team.observers.convergence import ConvergenceObserver
from workflows.agent_team.prompts import get_main_system_prompt

QUESTION = ("A jointly written essay appeared in 2021 about an animated web series. "
            "One author joined a college founded in 1927. What is the title of the episode?")
TIER = {"tiered_escalation": True, "escalate_after_gate_rejections": 2}
SOURCE = ("By Ana Reyes and Bo Lin, published January 2021, on the animated series Harbor Lane. "
          "Ana Reyes joined Northgate College, founded in 1927. "
          "They discuss the episode *Salt Water* of the series.")


def _run(**config):
    return convergence_run(enabled=True, progressive_research_v1=True, progressive_config=config,
                           bus_task_id="tiered-offline", question=QUESTION)


def _source(state, text=SOURCE, name="web_fetch", call="call-1", **kwargs):
    return state.observe_main_tool(tool_call_id=call, tool_name=name, arguments={"url": "https://x.invalid/a"},
                                   result=text, **kwargs)


def _ledger(receipt_id):
    return [
        {"question_span": "A jointly written essay appeared in 2021 about an animated web series",
         "receipt_id": receipt_id, "evidence_quote": "published January 2021, on the animated series Harbor Lane"},
        {"question_span": "One author joined a college founded in 1927",
         "receipt_id": receipt_id, "evidence_quote": "Ana Reyes joined Northgate College, founded in 1927"},
        {"question_span": "What is the title of the episode",
         "receipt_id": receipt_id, "evidence_quote": "They discuss the episode Salt Water"},
    ]


def _finalize(receipt_id, ledger=None):
    args = {"action": "finalize", "receipt_ids": [receipt_id], "final_answer": "Salt Water [1]."}
    return {**args, "requirements": ledger} if ledger is not None else args


# Answer guard ---------------------------------------------------------------

LEAKED = ("## Query: q\n\n## Search Results\n"
          "[1] **FineVerify: Scaling Test-Time Compute**\nCandidate Answer: Some Book. Rationale: The finding aid.\n"
          "URL: https://arxiv.org/pdf/2606.00660\n\n---\n\n"
          "[2] **Ordinary page**\nAn ordinary snippet about a library.\nURL: https://example.org/a\n")


def test_labelled_snippet_is_withheld_and_other_results_survive(tmp_path):
    filtered = AnswerLabelGuard(QUESTION, tmp_path).filter_search(LEAKED)
    assert MARKER in filtered
    assert "Candidate Answer" not in filtered and "Rationale" not in filtered
    assert "Ordinary page" in filtered
    assert (tmp_path / "ANSWER_LABEL_GUARD.jsonl").exists()


def test_explanation_label_is_withheld(tmp_path):
    entry = ("[1] **DR-DCI: Scaling Direct Corpus Interaction**\nby Y Lu — Explanation: The collection record says it.\n"
             "URL: https://openreview.net/pdf?id=x\n")
    assert AnswerLabelGuard(QUESTION, tmp_path).filter_search(entry).strip() == MARKER


def test_withheld_source_stays_quarantined_for_label_free_snippets(tmp_path):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    guard.filter_search(LEAKED)
    same_title = "[1] **FineVerify: Scaling Test-Time Compute**\nA snippet with no label.\nURL: https://google.invalid/goto\n"
    assert guard.filter_search(same_title).strip() == MARKER
    assert guard.screen("web_fetch", "page body", {"url": "https://arxiv.org/pdf/2606.00660"}) == MARKER


def test_fetch_hit_quarantines_later_snippets_of_that_url(tmp_path):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    fetched = f"[1] URL: https://arxiv.org/pdf/1\n    Info: {MARKER}"
    assert guard.screen("web_fetch", fetched, {"url": "https://arxiv.org/pdf/1"}) == fetched
    later = "[1] **Different title**\nNo label here.\nURL: https://www.arxiv.org/pdf/1/\n"
    assert guard.filter_search(later).strip() == MARKER


@pytest.mark.parametrize("request_text", [
    "the stated gold/correct answer", "Quote the stated correct (gold) answer", "ground truth for this item",
    "the benchmark question", "BrowseComp answer key",
])
def test_answer_seeking_requests_are_withheld(tmp_path, request_text):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    assert guard.screen("web_fetch", "body", {"url": "https://x.invalid", "info_to_extract": request_text}) == SEEK_MARKER


@pytest.mark.parametrize("text", [
    "Olympic gold medal winners in 2011", "Q: what time is it? A: noon.", "Answer: the museum opens at nine.",
])
def test_ordinary_words_are_not_treated_as_evaluation_material(tmp_path, text):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    assert not guard.seeks_answer(text)
    entry = f"[1] **Ordinary**\n{text}\nURL: https://example.org/b\n"
    assert guard.filter_search(entry) == entry


def test_answer_seeking_extraction_never_reaches_the_summary_model(tmp_path):
    module = importlib.import_module("plugins.tools.web_fetch_aligned")
    model = AsyncMock(return_value={"success": True, "extracted_info": "leak", "error": ""})
    with patch.object(module, "_extract_info_with_llm", model):
        _, restore = install_guard(QUESTION, tmp_path)
        try:
            result = asyncio.run(module._extract_info_with_llm("page", "the stated gold answer"))
        finally:
            restore()
    assert result["extracted_info"] == SEEK_MARKER
    model.assert_not_awaited()


def test_receipts_store_screened_text_for_helpers_and_the_team(tmp_path):
    _, restore = install_guard(QUESTION, tmp_path)
    try:
        with _run(**TIER) as state:
            receipt = _source(state, LEAKED, name="web_search")
    finally:
        restore()
    assert "Candidate Answer" not in receipt["report"]
    assert "Ordinary page" in receipt["report"]


# Pro as the reasoning model ---------------------------------------------------

def test_flash_accounting_is_unchanged_and_pro_uses_its_own_rates():
    from decimal import Decimal

    from local_runner import trial_helpers as helpers
    from local_runner.budget_adapter import reservation_upper

    usage = {"prompt_tokens": 1_000_000, "prompt_cache_hit_tokens": 400_000, "completion_tokens": 100_000}
    assert helpers.cost(usage, Decimal(1)) == Decimal("1.008")
    assert helpers.cost(usage, Decimal(1), "deepseek-v4-pro") == Decimal("4.11")
    assert helpers.reservation_amount("deepseek-flash", 32768) == reservation_upper(32768)
    assert helpers.reservation_amount("deepseek-v4-pro", 32768) < Decimal("3.3")


@pytest.mark.asyncio
async def test_pro_call_settles_at_pro_rates_within_its_reservation():
    from decimal import Decimal

    from local_runner import trial_helpers as helpers

    helpers.HERE = Path(__import__("tempfile").mkdtemp())
    meter = helpers.Meter()
    ident = await meter.reserve({"model": "deepseek-v4-pro", "max_tokens": 32768}, "native_stream")
    assert meter.entries[ident]["model"] == "deepseek-v4-pro"
    await meter.settle(ident, {"prompt_tokens": 200_000, "prompt_cache_hit_tokens": 190_000, "completion_tokens": 30_000})
    assert meter.reserved == 0 and meter.spent > Decimal("0.4")


def test_prepare_pins_the_model_only_for_paid_modes(tmp_path):
    from local_runner import prepare as local_prepare

    question = tmp_path / "q.json"
    text = "A synthetic question for the model override."
    import hashlib

    question.write_text(json.dumps({"id": "x", "question": text, "question_sha256": hashlib.sha256(text.encode()).hexdigest()}))
    local_prepare.prepare(question, tmp_path / "trial", model="deepseek-v4-pro")
    launch = (tmp_path / "trial" / "launch.ps1").read_text(encoding="utf-8")
    assert "'--env-file',$envFile,'-e','OPENAI_MODEL=deepseek-v4-pro'" in launch
    assert "OPENAI_MODEL=deepseek-flash" in launch
    with pytest.raises(ValueError):
        local_prepare.prepare(question, tmp_path / "other", model="unknown-model")


def test_extraction_stays_on_flash_when_pro_reasons(monkeypatch):
    from frontier_agent.infra.config import get_config
    from frontier_agent.infra import summary_llm

    monkeypatch.setenv("OPENAI_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-unused")
    monkeypatch.setenv("SUMMARY_LLM_MODEL_NAME", "deepseek-flash")
    monkeypatch.setenv("SUMMARY_LLM_BASE_URL", "https://api.deepseek.com/chat/completions")
    monkeypatch.setenv("SUMMARY_LLM_API_KEY", "offline-unused")
    get_config(force_reload=True)
    try:
        candidates = summary_llm.summary_llm_candidates()
        assert [item["model"] for item in candidates] == ["deepseek-flash"]
    finally:
        monkeypatch.undo()
        get_config(force_reload=True)


# Polluted DNS at the address check -------------------------------------------

def _resolves_to(monkeypatch, *addresses):
    from plugins.tools import _download_runner as runner

    monkeypatch.setattr(runner.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", (address, 443)) for address in addresses])
    return runner


def test_poison_answer_is_refused_unless_the_switch_is_on(monkeypatch):
    runner = _resolves_to(monkeypatch, "157.240.17.35", "2001::1")
    monkeypatch.delenv("FRONTIER_AGENT_TOLERATE_DNS_POISON", raising=False)
    with pytest.raises(runner.DownloadError, match="non-public"):
        runner._validate_public_url("https://en.wikipedia.org/wiki/Example")
    monkeypatch.setenv("FRONTIER_AGENT_TOLERATE_DNS_POISON", "1")
    assert runner._validate_public_url("https://en.wikipedia.org/wiki/Example") == ("157.240.17.35",)


@pytest.mark.parametrize("addresses", [
    ("2001::1",), ("157.240.17.35", "127.0.0.1"), ("157.240.17.35", "192.168.1.1"),
    ("157.240.17.35", "169.254.169.254"), ("157.240.17.35", "fd00::1"), ("157.240.17.35", "::1"),
    ("157.240.17.35", "fe80::1"), ("157.240.17.35", "2001::1", "10.0.0.1"),
])
def test_switch_never_admits_internal_addresses(monkeypatch, addresses):
    runner = _resolves_to(monkeypatch, *addresses)
    monkeypatch.setenv("FRONTIER_AGENT_TOLERATE_DNS_POISON", "1")
    with pytest.raises(runner.DownloadError, match="non-public"):
        runner._validate_public_url("https://example.org/")


# Requirement gate -----------------------------------------------------------

@pytest.mark.asyncio
async def test_progressive_v1_without_the_flag_is_unchanged():
    with _run() as state:
        receipt = _source(state)
        schema = state.review_tool().to_openai_schema()["function"]["parameters"]
        assert "requirements" not in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["continue", "finalize"]
        escalated = await state.review_tool().ainvoke(
            {"action": "escalate", "receipt_ids": [], "disposition": "reason"})
        assert escalated["reason"] == "invalid_review_action"
        assert (await state.review_tool().ainvoke(_finalize(receipt["receipt_id"])))["ok"]


@pytest.mark.asyncio
async def test_finalize_without_a_ledger_names_the_uncovered_question():
    with _run(**TIER) as state:
        receipt = _source(state)
        result = await state.review_tool().ainvoke(_finalize(receipt["receipt_id"]))
    assert not result["ok"] and result["reason"] == "requirements_not_supported"
    assert result["coverage_pct"] == 0
    assert "what is the title" in " ".join(result["uncovered_question_parts"])


@pytest.mark.asyncio
async def test_traceable_ledger_finalizes_and_is_saved():
    with _run(**TIER) as state:
        receipt = _source(state)
        result = await state.review_tool().ainvoke(_finalize(receipt["receipt_id"], _ledger(receipt["receipt_id"])))
        assert result["ok"]
        assert state.phase == "finalizing"
        assert len(state.snapshot()["decisions"][0]["requirements"]) == 3


@pytest.mark.asyncio
async def test_partial_coverage_lists_only_the_missing_condition():
    with _run(**TIER) as state:
        receipt = _source(state)
        ledger = [row for row in _ledger(receipt["receipt_id"]) if "college" not in row["question_span"]]
        result = await state.review_tool().ainvoke(_finalize(receipt["receipt_id"], ledger))
    assert not result["ok"]
    assert result["uncovered_question_parts"] == ["one author joined a college founded in 1927"]


@pytest.mark.parametrize("field,value,problem", [
    ("evidence_quote", "This sentence is not in the receipt at all", "evidence_quote_not_in_receipt"),
    ("evidence_quote", "Salt", "evidence_quote_too_short"),
    ("question_span", "a condition the question never states", "question_span_not_verbatim"),
])
@pytest.mark.asyncio
async def test_untraceable_rows_are_rejected(field, value, problem):
    with _run(**TIER) as state:
        receipt = _source(state)
        ledger = _ledger(receipt["receipt_id"])
        ledger[0][field] = value
        result = await state.review_tool().ainvoke(_finalize(receipt["receipt_id"], ledger))
    assert not result["ok"]
    assert result["row_problems"] == [{"row": 0, "problem": problem}]


@pytest.mark.asyncio
async def test_single_oversized_span_cannot_cover_the_whole_question():
    with _run(**TIER, max_requirement_words=10) as state:
        receipt = _source(state)
        ledger = [{"question_span": QUESTION, "receipt_id": receipt["receipt_id"],
                   "evidence_quote": "They discuss the episode Salt Water"}]
        result = await state.review_tool().ainvoke(_finalize(receipt["receipt_id"], ledger))
    assert result["row_problems"] == [{"row": 0, "problem": "question_span_length"}]


@pytest.mark.asyncio
async def test_failed_tool_receipt_cannot_support_a_condition():
    with _run(**TIER) as state:
        good = _source(state)
        bad = _source(state, "Only the failed fetch contains this sentence.", call="call-2",
                      is_error=True, error_kind="fetch_reported_failure")
        ledger = _ledger(good["receipt_id"])
        ledger[1].update(receipt_id=bad["receipt_id"], evidence_quote="Only the failed fetch contains this sentence.")
        result = await state.review_tool().ainvoke(_finalize(good["receipt_id"], ledger))
    assert result["row_problems"] == [{"row": 1, "problem": "evidence_quote_not_in_receipt"}]


@pytest.mark.parametrize("cited", ["receipt-unknown", "other"])
@pytest.mark.asyncio
async def test_real_quote_under_a_wrong_receipt_id_passes_and_is_recorded(cited):
    with _run(**TIER) as state:
        good = _source(state)
        other = _source(state, "An unrelated page.", call="call-2")
        ledger = _ledger(good["receipt_id"])
        wrong = other["receipt_id"] if cited == "other" else cited
        ledger[1]["receipt_id"] = wrong
        result = await state.review_tool().ainvoke(_finalize(good["receipt_id"], ledger))
        assert result["ok"], result
        saved = state.snapshot()["decisions"][0]["receipt_corrections"]
    assert saved == [{"row": 1, "cited": wrong, "found": good["receipt_id"]}]


def test_quoted_text_in_a_withheld_receipt_is_not_evidence():
    with _run(**TIER) as state:
        withheld = _source(state, MARKER)
        ledger = _ledger(withheld["receipt_id"])
        verdict = state.progressive.check_requirements(QUESTION, ledger, {withheld["receipt_id"]: withheld})
    assert not verdict["ok"]
    assert {item["problem"] for item in verdict["row_problems"]} == {"evidence_quote_not_in_receipt"}


def test_quote_match_tolerates_markdown_case_and_ellipsis():
    with _run(**TIER) as state:
        receipt = _source(state)
        ledger = _ledger(receipt["receipt_id"])
        ledger[2]["evidence_quote"] = "THEY DISCUSS the episode \"Salt Water\" ... of the series"
        verdict = state.progressive.check_requirements(QUESTION, ledger, {receipt["receipt_id"]: receipt})
    assert verdict["ok"], verdict


# Escalation control ---------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_gate_rejection_hands_the_uncovered_parts_to_the_team():
    with _run(**TIER) as state:
        receipt = _source(state)
        tool = state.review_tool()
        assert not (await tool.ainvoke(_finalize(receipt["receipt_id"])))["escalated"]
        second = await tool.ainvoke(_finalize(receipt["receipt_id"]))
        assert second["escalated"] and state.escalation_pending
        assert state.escalation["reason"] == "requirement_gate_exhausted"
        ctx = SimpleNamespace(task_id="t", turn=1, tool_calls=[], ai_text="", finish_reason="stop", metadata={})
        assert (await ConvergenceObserver().on_turn_end(ctx)).stop_reason == ESCALATE_STOP


@pytest.mark.asyncio
async def test_coordinator_escalation_needs_a_reason_and_closes_tools():
    with _run(**TIER) as state:
        receipt = _source(state)
        tool = state.review_tool()
        missing = await tool.ainvoke({"action": "escalate", "receipt_ids": []})
        assert missing["reason"] == "escalation_reason_required"
        done = await tool.ainvoke({"action": "escalate", "receipt_ids": [receipt["receipt_id"]],
                                   "disposition": "Several candidates need parallel checks.",
                                   "gaps": ["Which college fits?"]})
        assert done["escalated"]

        async def web_search(query: str) -> str:
            return "unreachable"

        from frontier_agent.core.tool import Tool

        guarded = guard_tools([Tool(name="web_search", description="d", func=web_search,
                                    parameters={"type": "object", "properties": {"query": {"type": "string"}}})],
                              coordinator=True)[0]
        with pytest.raises(ConvergenceBlocked):
            await guarded.ainvoke({"query": "more"})


@pytest.mark.asyncio
async def test_turn_budget_escalates_without_a_model_decision():
    with _run(**TIER, escalate_after_main_turns=2) as state:
        observer = ConvergenceObserver()
        ctx = SimpleNamespace(task_id="t", turn=0, tool_calls=[{}], ai_text="", finish_reason="tool_calls", metadata={})
        assert await observer.on_turn_end(ctx) is None
        ctx.turn = 1
        assert (await observer.on_turn_end(ctx)).stop_reason == ESCALATE_STOP
        assert state.escalation["reason"] == "light_turn_budget"


@pytest.mark.asyncio
async def test_answer_with_no_research_is_refused_then_escalated():
    with _run(**TIER) as state:
        ctx = SimpleNamespace(tool_calls=[], ai_text="From memory: X.", task_id="t", turn=0, max_turns=200, metadata={})
        first = await BareTextFinalizeObserver().on_llm_response(ctx)
        assert first.continue_to_next_turn and "final_answer" not in ctx.metadata
        second = await BareTextFinalizeObserver().on_llm_response(ctx)
        assert second.stop_reason == ESCALATE_STOP
        assert state.escalation["reason"] == "unsourced_answer"


@pytest.mark.asyncio
async def test_progress_can_be_acknowledged_before_any_gap_exists():
    with _run(**TIER) as state:
        state.observe_main_tool(tool_call_id="s", tool_name="web_search", arguments={"q": "x"}, result="hits")
        receipt = _source(state)
        assert state.progressive.cycles_since_progress == 1
        result = await state.review_tool().ainvoke({
            "action": "continue", "receipt_ids": [], "progress_summary": "Found the article.",
            "progress_receipt_ids": [receipt["receipt_id"]],
        })
        assert result["ok"]
        assert state.progressive.cycles_since_progress == 0
        assert not state._continuation_waiting_for_turn


def test_tiered_no_progress_notice_is_spaced_out():
    with _run(**TIER) as state:
        notices = 0
        for turn in range(12):
            state.observe_main_tool(tool_call_id=f"s{turn}", tool_name="web_search", arguments={"q": str(turn)}, result="r")
            state.observe_main_tool(tool_call_id=f"f{turn}", tool_name="web_fetch", arguments={"url": str(turn)}, result="r")
            notices += bool(state.progressive.on_main_turn("t", turn, effective=True))
    assert notices == 6


def test_handoff_is_bounded_and_seeds_the_team_gaps():
    with _run(**TIER) as state:
        for index in range(30):
            state.observe_main_tool(tool_call_id=f"f{index}", tool_name="web_fetch",
                                    arguments={"url": f"https://x.invalid/{index}"}, result="body " * 400)
        state.request_escalation("light_turn_budget", ["Which college fits?"])
        message = state.begin_team_stage()
        assert state.progressive is None and not state.escalation_pending
        assert state.approved_gaps == ["Which college fits?"]
        assert message.startswith(QUESTION)
        assert len(message) < 14000
        assert "omitted" in message
        saved = state.snapshot()["escalation"]
        assert saved["light_stage"]["tier"]["tiered_escalation"] is True


class _Model:
    model = "offline"

    async def chat(self, *args, **kwargs):
        return "ok"


async def _helper_call(state):
    from workflows.agent_team.convergence import _ConvergenceLLM

    return await _ConvergenceLLM(_Model(), state, False).chat([])


@pytest.mark.asyncio
async def test_helper_and_page_work_count_against_the_light_budget():
    from frontier_agent.core.tool import Tool

    async def web_fetch(url):
        return "page"

    with _run(**TIER) as state:
        fetch = guard_tools([Tool(name="web_fetch", description="d", func=web_fetch, parameters={
            "type": "object", "properties": {"url": {"type": "array"}}})], coordinator=False)[0]
        await fetch.ainvoke({"url": ["https://x.invalid/a", "https://x.invalid/b"]})
        await _helper_call(state)
        assert state.light_calls == 3


@pytest.mark.asyncio
async def test_soft_limit_warns_once_and_stops_new_helpers_only():
    from frontier_agent.core.tool import Tool

    async def assign_task(tasks):
        return "assigned"

    with _run(**TIER, light_soft_calls=2, light_hard_calls=4) as state:
        state.count_light_calls(2)
        assert state.light_soft_reached and not state.light_hard_reached
        assign = guard_tools([Tool(name="assign_task", description="d", func=assign_task, parameters={
            "type": "object", "properties": {"tasks": {"type": "array", "items": {"type": "object"}}}})],
            coordinator=True)[0]
        with pytest.raises(ConvergenceBlocked, match="light_budget_reserved_for_finishing"):
            await assign.ainvoke({"tasks": [{"agent": "a", "gap_id": "g", "prompt": "p"}]})
        ctx = SimpleNamespace(task_id="t", turn=0, tool_calls=[{}], ai_text="", finish_reason="tool_calls", metadata={})
        first = await ConvergenceObserver().on_turn_end(ctx)
        assert "nearly used" in first.inject_messages[-1]
        ctx.turn = 1
        second = await ConvergenceObserver().on_turn_end(ctx)
        assert second is None or "nearly used" not in " ".join(second.inject_messages or [])
        assert not state.escalation_pending
        # A finish is still possible between the two limits.
        receipt = _source(state)
        assert (await state.review_tool().ainvoke(_finalize(receipt["receipt_id"], _ledger(receipt["receipt_id"]))))["ok"]


@pytest.mark.asyncio
async def test_hard_limit_stops_helpers_and_escalates_at_the_next_turn():
    with _run(**TIER, light_soft_calls=1, light_hard_calls=2) as state:
        state.count_light_calls(2)
        with pytest.raises(ConvergenceBlocked, match="light_call_budget_exhausted"):
            await _helper_call(state)
        ctx = SimpleNamespace(task_id="t", turn=0, tool_calls=[{}], ai_text="", finish_reason="tool_calls", metadata={})
        assert (await ConvergenceObserver().on_turn_end(ctx)).stop_reason == ESCALATE_STOP
        assert state.escalation["reason"] == "light_call_budget"
        assert state.escalation["light_calls"] == 2


@pytest.mark.asyncio
async def test_team_stage_and_v1_are_not_held_to_the_light_budget():
    with _run(**TIER, light_soft_calls=1, light_hard_calls=2) as state:
        state.count_light_calls(2)
        state.request_escalation("light_call_budget")
        state.begin_team_stage()
        assert not state.light_hard_reached
        assert await _helper_call(state) == "ok"
        assert state.light_calls == 2
    with _run() as state:
        assert await _helper_call(state) == "ok"
        assert state.light_calls == 0 and "light_calls" not in state.snapshot()


# Real public phrases and overlaps seen in saved runs; the question is synthetic.
_LONG_QUESTION = ("A filmmaker decided to become a director after watching a Brazilian film and later won an "
                  "award for a screenplay about a scientist, then directed a musical and a fantasy series whose "
                  "final part premiered in the same year that a friend published a memoir about growing up in a "
                  "small town in Ohio. What is the title of the memoir written by that friend of the filmmaker?")


def test_one_shared_phrase_is_not_a_copy_of_the_question(tmp_path):
    guard = AnswerLabelGuard(_LONG_QUESTION, tmp_path)
    # 68 question words: the audit's 15% share is a run of 11.
    assert guard.width == 11
    phrase = "[1] **Bill Condon**\nHe decided to become a director after watching a Brazilian film in 1970.\nURL: https://example.org/a\n"
    assert guard.filter_search(phrase) == phrase
    longer = phrase.replace("Brazilian film in", "Brazilian film and later")
    assert guard.filter_search(longer).strip() == MARKER


def test_a_long_copied_run_is_still_withheld(tmp_path):
    guard = AnswerLabelGuard(_LONG_QUESTION, tmp_path)
    copied = ("[1] **Dataset viewer**\ndecided to become a director after watching a Brazilian film and later won an award "
              "for a screenplay\nURL: https://example.org/b\n")
    assert guard.filter_search(copied).strip() == MARKER


def test_short_questions_follow_the_audit_share_without_a_floor(tmp_path):
    guard = AnswerLabelGuard("Which small town did the author of that memoir about Ohio grow up in?", tmp_path)
    assert guard.width == 3


@pytest.mark.parametrize("config", [
    {"tiered_escalation": "yes"}, {"min_question_coverage_pct": 101}, {"escalate_after_main_turns": 0},
    {"light_soft_calls": 90, "light_hard_calls": 90},
])
def test_invalid_tier_config_is_refused(config):
    with pytest.raises(ValueError):
        with _run(**config):
            pass


# Prompt and profile ---------------------------------------------------------

def test_tiered_prompt_states_the_configured_limits_and_v1_prompt_omits_them():
    tools = ["web_search", "web_fetch"]
    tiered = get_main_system_prompt("2026-10-30", planning_mode=False, main_agent_tools=tools,
                                    progressive_research_v1=True,
                                    progressive_config={**TIER, "escalate_after_main_turns": 17})
    assert "17 effective turns" in tiered and "action=\"escalate\"" in tiered
    plain = get_main_system_prompt("2026-10-30", planning_mode=False, main_agent_tools=tools,
                                   progressive_research_v1=True)
    assert "TIERED RESEARCH" not in plain


def test_tiered_profile_adds_only_the_declared_controls():
    import yaml

    directory = Path(__file__).parents[1] / "local_profiles"
    light = yaml.safe_load((directory / "progressive_research_v1.yaml").read_text(encoding="utf-8"))
    tiered = yaml.safe_load((directory / "tiered_research_v1.yaml").read_text(encoding="utf-8"))
    team = yaml.safe_load((directory / "convergence_v1.yaml").read_text(encoding="utf-8"))
    assert tiered["agent"].pop("team_main_agent_tools") == team["agent"]["main_agent_tools"]
    extra = {k: tiered["agent"]["progressive_config"].pop(k)
             for k in ("tiered_escalation", "min_question_coverage_pct", "max_requirement_words",
                       "escalate_after_gate_rejections", "escalate_after_main_turns",
                       "light_soft_calls", "light_hard_calls")}
    assert extra["tiered_escalation"] is True
    assert tiered == light
    assert set(tiered["agent"]["main_agent_tools"]) >= set(team["agent"]["main_agent_tools"])


# End to end through the real main node --------------------------------------

_LIGHT_TOOLS = ["web_search", "web_fetch", "assign_task", "add_task", "update_task"]
_TEAM_TOOLS = ["web_search", "assign_task", "add_task", "update_task"]


class ScriptedTiers:
    model = "offline-tiered-main"

    def __init__(self, light_steps, team_steps, before_team_turn=None):
        self.light_steps, self.team_steps = light_steps, team_steps
        self.before_team_turn = before_team_turn
        self.calls = {"light": 0, "team": 0}
        self.seen = {"light": [], "team": []}

    def bind(self, **kwargs):
        return self

    async def chat(self, messages, **kwargs):
        from frontier_agent.core.llm import LLMResponse

        state = current_convergence()
        stage = "light" if state.progressive is not None else "team"
        self.calls[stage] += 1
        if stage == "team" and self.before_team_turn is not None:
            await self.before_team_turn(self.calls[stage])
        self.seen[stage].append({"system": messages[0]["content"], "user": messages[1]["content"],
                                 "all": json.dumps(messages, ensure_ascii=False, default=str),
                                 "tools": [t.get("function", t).get("name", "") if isinstance(t, dict)
                                           else getattr(t, "name", "") for t in kwargs.get("tools") or []]})
        steps = self.light_steps if stage == "light" else self.team_steps
        name, args = steps[min(self.calls[stage], len(steps)) - 1](state, self.calls[stage])
        return LLMResponse(tool_calls=[{"id": f"{stage}-{self.calls[stage]}", "name": name, "args": args}])

    async def stream(self, messages, **kwargs):
        from frontier_agent.core.llm import StreamDelta

        result = await self.chat(messages, **kwargs)
        call = result.tool_calls[0]
        yield StreamDelta(tool_call_deltas=[{"index": 0, "id": call["id"], "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}],
            finish_reason="tool_calls")


def _search(state, turn):
    return "web_search", {"query": f"synthetic query {turn}"}


def _escalate(state, turn):
    return "review_research", {"action": "escalate", "receipt_ids": [],
                               "disposition": "Needs parallel checks.", "gaps": ["Which college fits?"]}


def _team_finalize(state, turn):
    ids = [item["receipt_id"] for item in state.snapshot()["receipts"]]
    return "review_research", {"action": "finalize", "receipt_ids": ids, "final_answer": "Team answer [1].",
                               "disposition": "The team resolved the open condition."}


class HoldingHelper:
    """A real bus session whose model only records its assignment."""

    model = "offline-tiered-helper"

    def __init__(self):
        self.prompts = []
        self.started = asyncio.Event()

    def bind(self, **kwargs):
        return self

    async def chat(self, messages, **kwargs):
        self.prompts.append(json.dumps(messages, ensure_ascii=False, default=str))
        self.started.set()
        await asyncio.Event().wait()

    async def stream(self, messages, **kwargs):
        await self.chat(messages, **kwargs)
        yield


async def _drive(tmp_path, monkeypatch, model, *, task, progressive_config, helper=None):
    from frontier_agent.components.agent_bus import AgentBus
    from frontier_agent.core.runtime.registries import services
    from frontier_agent.core.tool import tool
    from plugins.tools.assign_task import agent_team_assign_task
    from plugins.tools.task_board import add_task, update_task

    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    bus = AgentBus()
    saved_services = services.snapshot()
    services.register(AgentBus, bus)
    if helper is not None:
        await bus.create_session(task_id=task, name="a1", role_id="researcher",
                                 system_prompt="Offline helper.", tools_override=[], llm_override=helper)

    @tool
    async def web_search(query: str) -> str:
        """Return a synthetic local search result."""
        return "Synthetic source. URL: https://example.invalid/direct"

    @tool
    async def web_fetch(url: str) -> str:
        """Return a synthetic local page body."""
        return "URL: https://example.invalid/direct\nSynthetic page."

    supplied = [web_search, web_fetch, agent_team_assign_task, add_task, update_task]
    profile = {"agent": {"planning_mode": False, "reporter": False, "main_max_turns": 12,
        "finalization_reserve_turns": 1, "progressive_research_v1": True, "team_effort": "max",
        "progressive_config": progressive_config, "team_main_agent_tools": _TEAM_TOOLS,
        "main_agent_tools": _LIGHT_TOOLS}}
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    for part in ("WORKSPACE", "INPUTS", "OUTPUTS"):
        path = tmp_path / part.lower()
        path.mkdir()
        monkeypatch.setenv("FRONTIER_AGENT_" + part + "_DIR", str(path))
    monkeypatch.setattr(node, "_resolve_llm_and_profile", lambda *a, **k: (model, profile, None))
    monkeypatch.setattr(node.registry, "get", lambda cls: bus if cls is AgentBus else object())
    monkeypatch.setattr(node.registry, "get_optional", lambda cls: bus if cls is AgentBus else None)
    monkeypatch.setattr(node, "_resolve_profile_tools", lambda *a, **k:
        (supplied, [item.name for item in supplied]) if k.get("role_id") == node.MAIN_ROLE_ID else ([], []))
    monkeypatch.setattr(node, "_resolve_trajectory_dir", lambda *a: tmp_path / "trajectories")
    monkeypatch.setattr(node, "_log_inputs_dir_contents", lambda *a, **k: None)
    try:
        result = await asyncio.wait_for(node.main_agent_node(
            {"original_question": QUESTION, "metadata": {}}, SimpleNamespace(task_id=task)), timeout=20)
        return result, json.loads((tmp_path / "trajectories/convergence_v1.json").read_text())
    finally:
        await bus.cleanup_task(task)
        services.restore(saved_services)


@pytest.mark.asyncio
async def test_light_path_hands_over_to_the_team_in_one_run(tmp_path, monkeypatch):
    def team_dispatch(state, turn):
        return "assign_task", {"tasks": [{"agent": "a1", "prompt": "Check the college."}]}

    helper = HoldingHelper()

    async def helper_is_running(turn):
        if turn > 1:
            await asyncio.wait_for(helper.started.wait(), timeout=5)

    model = ScriptedTiers([_search, _escalate], [team_dispatch, _team_finalize], helper_is_running)
    result, saved = await _drive(tmp_path, monkeypatch, model, task="tiered-handover",
                                 progressive_config={**TIER}, helper=helper)
    assert result["stopped_by"] == "final_answer"
    assert "Team answer" in result["final_answer"]
    assert model.calls == {"light": 2, "team": 2}
    light, team = model.seen["light"][0], model.seen["team"][0]
    assert "TIERED RESEARCH" in light["system"]
    assert "PROGRESSIVE RESEARCH V1" not in team["system"]
    assert "TIERED RESEARCH" not in team["system"]
    assert "CONVERGENCE V1" in team["system"]
    assert "LIGHT-PATH HANDOFF" in team["user"] and "Synthetic source" in team["user"]
    # The real assign path accepts the team's task with no progressive gap_id
    # and the helper receives the gap handed over by the light path.
    assert "Submitted 1 task" in model.seen["team"][1]["all"]
    assert "Which college fits?" in helper.prompts[0]
    assert saved["escalation"]["reason"] == "coordinator_requested"
    assert saved["escalation"]["light_stage"]["effective_main_turns"] >= 1
    assert saved["phase"] == "finished"
    assert result["research_cleanup"]["cleanup_confirmed"]


@pytest.mark.asyncio
async def test_light_path_alone_finishes_an_easy_question(tmp_path, monkeypatch):
    def fetch(state, turn):
        return "web_fetch", {"url": "https://example.invalid/direct"}

    def finish(state, turn):
        receipts = state.snapshot()["receipts"]
        receipt = receipts[-1]["receipt_id"]
        return "review_research", {"action": "finalize", "receipt_ids": [receipt], "final_answer": "Answer [1].",
            "requirements": [{"question_span": QUESTION.split(". ")[index].rstrip("?."), "receipt_id": receipt,
                              "evidence_quote": "URL: https://example.invalid/direct Synthetic page."}
                             for index in range(3)]}

    model = ScriptedTiers([fetch, finish], [_team_finalize])
    result, saved = await _drive(tmp_path, monkeypatch, model, task="tiered-easy",
                                 progressive_config={**TIER, "max_requirement_words": 20})
    assert result["stopped_by"] == "final_answer"
    assert model.calls == {"light": 2, "team": 0}
    assert "escalation" not in saved


@pytest.mark.parametrize("light_steps,config,reason,light_calls", [
    ([_search, _escalate], {}, "coordinator_requested", 2),
    ([_search], {"escalate_after_main_turns": 3}, "light_turn_budget", 3),
])
@pytest.mark.asyncio
async def test_stop_switch_ends_the_run_without_team_or_extra_model_call(
    tmp_path, monkeypatch, light_steps, config, reason, light_calls,
):
    model = ScriptedTiers(light_steps, [_team_finalize])
    result, saved = await _drive(tmp_path, monkeypatch, model, task="tiered-stop-" + reason,
                                 progressive_config={**TIER, **config, "stop_at_escalation": True})
    assert model.calls == {"light": light_calls, "team": 0}
    assert result["final_answer"] == f"[ESCALATION_REQUIRED] reason={reason}"
    assert result["stopped_by"] == ESCALATE_STOP
    assert saved["escalation"]["reason"] == reason
    assert "stopped_at_escalation" in [event["event"] for event in saved["events"]]


def test_light_stage_profile_differs_only_by_the_stop_switch():
    import yaml

    directory = Path(__file__).parents[1] / "local_profiles"
    tiered = yaml.safe_load((directory / "tiered_research_v1.yaml").read_text(encoding="utf-8"))
    stage = yaml.safe_load((directory / "tiered_light_stage_v1.yaml").read_text(encoding="utf-8"))
    assert stage["agent"]["progressive_config"].pop("stop_at_escalation") is True
    assert stage == tiered


@pytest.mark.asyncio
async def test_rejected_finalize_keeps_the_candidate_answer_for_analysis():
    with _run(**TIER) as state:
        receipt = _source(state)
        await state.review_tool().ainvoke(_finalize(receipt["receipt_id"]))
        rejected = [e for e in state.snapshot()["events"] if e["event"] == "requirement_gate_rejected"]
    assert rejected[0]["candidate_answer"] == "Salt Water [1]."


@pytest.mark.asyncio
async def test_turn_budget_moves_a_stuck_run_to_the_team(tmp_path, monkeypatch):
    model = ScriptedTiers([_search], [_team_finalize])
    result, saved = await _drive(tmp_path, monkeypatch, model, task="tiered-budget",
                                 progressive_config={**TIER, "escalate_after_main_turns": 3})
    assert result["stopped_by"] == "final_answer"
    assert model.calls == {"light": 3, "team": 1}
    assert saved["escalation"]["reason"] == "light_turn_budget"
    assert "Resolve every original-question condition" in model.seen["team"][0]["user"]
