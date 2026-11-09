"""Offline checks for the answer-chain rules of the tiered light stage."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from local_runner.answer_label_guard import MARKER, AnswerLabelGuard
from workflows.agent_team import answer_settlement as settlement
from workflows.agent_team.convergence import (
    ESCALATE_STOP, ConvergenceBlocked, convergence_run, guard_tools,
)
from workflows.agent_team.observers.convergence import ConvergenceObserver
from workflows.agent_team.progressive import ProgressivePolicy, ask_positions
from workflows.agent_team.prompts import get_main_system_prompt

try:
    from tests.test_tiered_research_v1 import (
        QUESTION, SOURCE, TIER, ScriptedTiers, _drive, _ledger, _run, _search, _source,
    )
except ImportError:
    from test_tiered_research_v1 import (
        QUESTION, SOURCE, TIER, ScriptedTiers, _drive, _ledger, _run, _search, _source,
    )

# The package exports the tool under this name, so import the module itself.
search_tool = importlib.import_module("plugins.tools.web_search")
CHAIN = {**TIER, "answer_chain_gate": True, "min_question_coverage_pct": 30, "min_ask_coverage_pct": 80}
STOP = {**TIER, "stop_at_escalation": True, "synthesize_at_stop": True}
ASK_ROW = {"question_span": "What is the title of the episode", "evidence_quote": "They discuss the episode Salt Water"}


# Requirement gate -----------------------------------------------------------

def test_a_question_mark_names_the_asking_sentence():
    words = "Which series starred him? The series was short-lived."
    assert ask_positions(words) == [0, 1, 2, 3]


def test_without_a_question_mark_the_closing_sentence_asks():
    words = "A factory opened in 1902. Please provide the street name."
    assert ask_positions(words) == [5, 6, 7, 8, 9]


@pytest.mark.asyncio
async def test_chain_gate_accepts_the_asked_part_and_records_what_stays_unverified():
    with _run(**CHAIN) as state:
        receipt = _source(state)["receipt_id"]
        rows = [{**ASK_ROW, "receipt_id": receipt}, _ledger(receipt)[1]]
        decided = await state.review_tool().ainvoke({"action": "finalize", "receipt_ids": [receipt],
                                                     "final_answer": "Salt Water.", "requirements": rows})
        record = state.snapshot()["decisions"][-1]
    assert decided["ok"] is True
    assert (
        "a jointly written essay appeared in 2021 about an animated web series"
        in record["unverified_question_parts"][0]
    )


@pytest.mark.asyncio
async def test_chain_gate_refuses_an_answer_that_skips_what_is_asked():
    with _run(**CHAIN) as state:
        receipt = _source(state)["receipt_id"]
        decided = await state.review_tool().ainvoke({"action": "finalize", "receipt_ids": [receipt],
                                                     "final_answer": "Salt Water.",
                                                     "requirements": _ledger(receipt)[:2]})
    assert decided["ok"] is False and decided["ask_coverage_pct"] == 0
    assert decided["required_ask_coverage_pct"] == 80


def test_without_the_switch_the_gate_is_unchanged():
    policy = ProgressivePolicy(TIER)
    receipts = {"r": {"status": "tool_completed", "report": SOURCE}}
    verdict = policy.check_requirements(QUESTION, [{**ASK_ROW, "receipt_id": "r"}], receipts)
    assert verdict["ok"] is False and verdict["required_coverage_pct"] == 95
    assert "ask_coverage_pct" not in verdict


@pytest.mark.parametrize("config", [
    {"max_search_queries": -1}, {"max_search_queries": True}, {"min_ask_coverage_pct": 101},
    {"min_ask_coverage_pct": 0}, {"answer_chain_gate": 1}, {"synthesize_at_stop": "yes"},
    {"independent_check": 1}, {"show_low_value_results": 1},
])
def test_invalid_answer_chain_config_is_refused(config):
    with pytest.raises(ValueError, match="invalid_tier_config"):
        ProgressivePolicy({**TIER, **config})


def test_the_chain_prompt_states_its_limits_and_the_v1_prompt_is_unchanged():
    config = {**CHAIN, "max_search_queries": 40}
    chain = get_main_system_prompt("2026-10-31", progressive_research_v1=True, progressive_config=config)
    assert "40 search queries" in chain and "at least\n80% of the sentence" in chain
    assert "write no named candidate into your plan" in chain and "Background clues" in chain
    plain = get_main_system_prompt("2026-10-31", progressive_research_v1=True, progressive_config=TIER)
    assert "Background clues" not in plain and "Hand a question to the full research team" in plain


def test_the_v2_profile_differs_from_v1_only_by_declared_controls():
    import yaml

    directory = Path(__file__).parents[1] / "local_profiles"
    first = yaml.safe_load((directory / "tiered_light_stage_v1.yaml").read_text(encoding="utf-8"))
    second = yaml.safe_load((directory / "tiered_light_stage_v2.yaml").read_text(encoding="utf-8"))
    controls = second["agent"]["progressive_config"]
    assert controls.pop("min_question_coverage_pct") == 50
    first["agent"]["progressive_config"].pop("min_question_coverage_pct")
    extra = {key: controls.pop(key) for key in (
        "answer_chain_gate", "min_ask_coverage_pct", "max_search_queries", "synthesize_at_stop",
        "independent_check", "show_low_value_results", "unquote_single_words", "archive_reads")}
    assert extra == {"answer_chain_gate": True, "min_ask_coverage_pct": 80, "max_search_queries": 40,
                     "synthesize_at_stop": True, "independent_check": True, "show_low_value_results": True,
                     "unquote_single_words": True, "archive_reads": 10}
    assert second == first


# Search budget --------------------------------------------------------------

def _web_search_tool():
    from frontier_agent.core.tool import tool

    @tool
    async def web_search(q: str | list[str]) -> str:
        """Return a synthetic local search result."""
        return "result"

    return web_search


@pytest.mark.asyncio
async def test_each_query_counts_and_the_budget_refuses_further_searches():
    with _run(**TIER, max_search_queries=3) as state:
        search = guard_tools([_web_search_tool()], coordinator=False)[0]
        assert await search.func(q=["one", "two"]) == "result"
        assert state.search_budget_notice() == ""
        assert await search.func(q="three") == "result"
        assert state.search_budget_spent
        with pytest.raises(ConvergenceBlocked, match="search_budget_spent"):
            await search.func(q="four")
        assert "search budget is used" in state.search_budget_notice()
        assert state.search_budget_notice() == ""
        assert state.snapshot()["progressive"]["search_queries"] == 3


@pytest.mark.asyncio
async def test_no_search_limit_unless_a_profile_sets_one():
    with _run(**TIER) as state:
        search = guard_tools([_web_search_tool()], coordinator=False)[0]
        for index in range(5):
            await search.func(q=f"query {index}")
        assert not state.search_budget_spent


# Search display -------------------------------------------------------------

def test_low_value_results_stay_hidden_by_default():
    with _run(**TIER):
        assert "facebook.com" in search_tool._combined_blacklist()


def test_the_switch_shows_low_value_results_for_the_run_only(monkeypatch):
    monkeypatch.setattr(search_tool, "_extra_blacklist", lambda: frozenset({"blocked.example"}))
    with _run(**TIER, show_low_value_results=True):
        assert search_tool._combined_blacklist() == frozenset({"blocked.example"})
    assert "facebook.com" in search_tool._combined_blacklist()


# Evaluation sources in the guard --------------------------------------------

@pytest.mark.parametrize("entry", [
    "[1] **Ordinary title**\nSnippet.\nURL: https://huggingface.co/datasets/some-org/agent-runs\n",
    "[1] **Ordinary title**\nSnippet.\nURL: https://www.kaggle.com/datasets/some-org/agent-runs\n",
    "[1] **some-org/agent-runs · Datasets at Hugging Face**\nSnippet.\nURL: https://www.google.com/goto?url=abc\n",
    "[1] **Team observation-masking-eval-logs**\nSnippet.\nURL: https://example.org/a\n",
])
def test_evaluation_sources_are_withheld_and_quarantined(tmp_path, entry):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    assert guard.filter_search(entry).strip() == MARKER
    assert guard.filter_search(entry.replace("Snippet.", "Another snippet.")).strip() == MARKER
    stages = [json.loads(line)["stage"] for line in (tmp_path / "ANSWER_LABEL_GUARD.jsonl").read_text().splitlines()]
    assert stages == ["search_eval_source", "search_quarantined_source"]


@pytest.mark.parametrize("entry", [
    "[1] **A paper page**\nSnippet.\nURL: https://huggingface.co/papers/2401.00001\n",
    "[1] **Ordinary page**\nThese evaluation logs were stored on tape.\nURL: https://example.org/a\n",
])
def test_ordinary_entries_survive_the_source_rule(tmp_path, entry):
    assert AnswerLabelGuard(QUESTION, tmp_path).filter_search(entry) == entry


def test_fetching_an_evaluation_source_is_withheld(tmp_path):
    guard = AnswerLabelGuard(QUESTION, tmp_path)
    args = {"url": "https://huggingface.co/datasets/some-org/agent-runs"}
    assert guard.filter_fetch("Page body", args, "tool_result:web_fetch") == MARKER


# Evidence and labels --------------------------------------------------------

def _receipt(index, tool, report, status="tool_completed", arguments=None):
    return {"receipt_id": f"receipt-{index}", "agent": "main", "report": report, "status": status,
            "source": {"tool_name": tool, "arguments": arguments or {}}}


def _entry(number, name):
    return f"[{number}] **{name}**\nA snippet about {name}.\nURL: https://example.org/{name}\n"


def test_evidence_puts_pages_first_and_keeps_one_entry_per_url():
    snapshot = {"receipts": [
        _receipt(1, "web_search", "## Query: a\n\n## Search Results\n" + _entry(1, "old") + "\n" + _entry(2, "both")),
        _receipt(2, "web_search", "## Query: b\n\n## Search Results\n" + _entry(1, "both")),
        _receipt(3, "web_fetch", "Page body.", arguments={"url": "https://example.org/page"}),
        _receipt(4, "web_fetch", "Failed.", status="tool_error"),
        _receipt(5, "web_search", "[1] **Withheld**\n[Evaluation source withheld: x]\nURL: https://example.org/w\n"),
    ], "progressive": {"progress": [{"summary": "Suburb identified."}]}}
    text, aliases = settlement.render_evidence(snapshot)
    assert text.index("Suburb identified.") < text.index("Page body.") < text.index("both") < text.index("old")
    assert text.count("https://example.org/both") == 1 and text.index("[R2] [1] **both**") > 0
    assert "Failed." not in text and "Withheld" not in text
    assert aliases["R3"] == "receipt-3" and "R4" not in aliases


def test_evidence_is_bounded():
    entries = "\n".join(_entry(index, f"name{index}" + "x" * 900) for index in range(1, 200))
    pages = [_receipt(index, "web_fetch", "p" * 20000) for index in range(1, 12)]
    text, _ = settlement.render_evidence({"receipts": [_receipt(20, "web_search", entries), *pages]})
    assert len(text) <= settlement.EVIDENCE_TOTAL_CHARS + 500
    assert "p" * settlement.EVIDENCE_PAGE_CHARS in text and "p" * (settlement.EVIDENCE_PAGE_CHARS + 1) not in text


def test_a_reply_may_arrive_in_a_code_fence():
    assert settlement.parse_reply('```json\n{"answer": "x"}\n```') == {"answer": "x"}
    assert settlement.parse_reply("no json here") is None
    assert settlement.parse_reply('{"answer": ') is None


SNAPSHOT = {"receipts": [_receipt(1, "web_fetch", SOURCE)]}
BACKED = {"answer": "Salt Water", "confidence": 60, "unverified_conditions": ["the college"],
          "key_evidence": [{"source": "R1", "quote": "They discuss the episode Salt Water"}]}


def test_a_forced_answer_with_a_real_quote_is_best_effort():
    outcome = settlement.forced_outcome(BACKED, SNAPSHOT)
    assert outcome == {"level": "best_effort", "answer": "Salt Water", "confidence": 60,
                       "unverified_question_parts": ["the college"]}


@pytest.mark.parametrize("reply,reason", [
    (None, "no_answer_from_evidence"),
    ({"answer": " "}, "no_answer_from_evidence"),
    ({"answer": "X", "key_evidence": []}, "no_verbatim_evidence"),
    ({"answer": "X", "key_evidence": [{"source": "R1", "quote": "a passage nobody saved anywhere"}]},
     "no_verbatim_evidence"),
    ({"answer": "X", "key_evidence": "They discuss the episode"}, "no_verbatim_evidence"),
])
def test_a_forced_answer_without_evidence_counts_as_none(reply, reason):
    outcome = settlement.forced_outcome(reply, SNAPSHOT)
    assert outcome["level"] == "none" and outcome["reason"] == reason


@pytest.mark.parametrize("reply,expected", [
    ({"verdict": "supported"}, ("verified", "supported")),
    ({"verdict": "insufficient"}, ("best_effort", "insufficient")),
    ({"verdict": "contradicted"}, ("best_effort", "contradicted")),
    ({"verdict": "maybe"}, ("best_effort", "no_verdict")),
    (None, ("best_effort", "no_verdict")),
])
def test_only_a_supported_verdict_verifies(reply, expected):
    assert settlement.checked_level(reply) == expected


def test_the_check_sees_quotes_and_not_the_saved_pages():
    prompt = settlement.check_prompt(QUESTION, "Salt Water", [{"question_span": "the span", "receipt_id": "secret-id",
                                                               "evidence_quote": "the quote"}])
    assert "the span" in prompt and "the quote" in prompt and "Salt Water" in prompt
    assert "secret-id" not in prompt


@pytest.mark.asyncio
async def test_the_check_goes_to_the_extraction_model(monkeypatch):
    import httpx

    seen = {}

    async def responder(request):
        seen.update(json.loads(request.content), authorization=request.headers["authorization"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    original = httpx.AsyncClient.__init__

    def init(client, *args, **kwargs):
        original(client, *args, **{**kwargs, "transport": httpx.MockTransport(responder)})

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    monkeypatch.setattr("frontier_agent.infra.summary_llm.summary_llm_candidates", lambda: [
        {"endpoint": "https://api.deepseek.com/chat/completions", "model": "deepseek-flash", "api_key": "offline"}])
    assert await settlement._ask_flash("the prompt", 5) == "{}"
    assert seen["model"] == "deepseek-flash" and seen["max_tokens"] == settlement.CHECK_MAX_TOKENS
    assert seen["authorization"] == "Bearer offline" and seen["messages"][0]["content"] == "the prompt"


# End to end through the real main node --------------------------------------

class ClosingModel(ScriptedTiers):
    def __init__(self, light_steps, closing):
        super().__init__(light_steps, [])
        self.closing, self.prompts = closing, []

    async def chat(self, messages, **kwargs):
        from frontier_agent.core.llm import LLMResponse

        if len(messages) > 1:
            return await super().chat(messages, **kwargs)
        self.prompts.append(messages[0]["content"])
        if isinstance(self.closing, Exception):
            raise self.closing
        return LLMResponse(content=self.closing)


def _fetch(state, turn):
    return "web_fetch", {"url": "https://example.invalid/direct"}


PAGE_QUOTE = "URL: https://example.invalid/direct Synthetic page."
FORCED = json.dumps({"answer": "Salt Water", "confidence": 55, "unverified_conditions": ["the college"],
                     "key_evidence": [{"source": "R1", "quote": PAGE_QUOTE}]})


@pytest.mark.asyncio
async def test_a_stopped_run_reports_its_best_supported_answer_with_a_label(tmp_path, monkeypatch):
    model = ClosingModel([_fetch], FORCED)
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-forced",
                                 progressive_config={**STOP, "escalate_after_main_turns": 2})
    assert model.calls == {"light": 2, "team": 0} and len(model.prompts) == 1
    assert "QUESTION:\n" + QUESTION in model.prompts[0] and "Synthetic page." in model.prompts[0]
    assert result["final_answer"] == ("Salt Water\n\nEvidence level: best_effort\n"
                                      "Self-reported confidence: 55%\nUnverified: the college")
    assert result["stopped_by"] == ESCALATE_STOP
    assert saved["settlement"]["level"] == "best_effort" and saved["settlement"]["kind"] == "forced_synthesis"
    assert saved["phase"] == "finished"


@pytest.mark.parametrize("closing,reason", [
    (json.dumps({"answer": "Salt Water", "key_evidence": [{"source": "R1", "quote": "never saved in any receipt"}]}),
     "no_verbatim_evidence"),
    ('{"answer": ""}', "no_answer_from_evidence"),
    ("not json", "no_answer_from_evidence"),
])
@pytest.mark.asyncio
async def test_a_stopped_run_without_a_backed_answer_keeps_the_marker(tmp_path, monkeypatch, closing, reason):
    model = ClosingModel([_fetch], closing)
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-none-" + reason,
                                 progressive_config={**STOP, "escalate_after_main_turns": 2})
    assert result["final_answer"] == "[ESCALATION_REQUIRED] reason=light_turn_budget"
    assert saved["settlement"]["level"] == "none" and saved["settlement"]["reason"] == reason


@pytest.mark.asyncio
async def test_a_failed_synthesis_still_ends_with_the_marker(tmp_path, monkeypatch):
    model = ClosingModel([_fetch], RuntimeError("provider down"))
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-failed",
                                 progressive_config={**STOP, "escalate_after_main_turns": 2})
    assert result["final_answer"] == "[ESCALATION_REQUIRED] reason=light_turn_budget"
    assert saved["settlement"]["level"] == "none"


def _finish(state, turn):
    receipt = state.snapshot()["receipts"][-1]["receipt_id"]
    return "review_research", {"action": "finalize", "receipt_ids": [receipt], "final_answer": "Answer [1].",
        "requirements": [{"question_span": QUESTION.split(". ")[index].rstrip("?."), "receipt_id": receipt,
                          "evidence_quote": PAGE_QUOTE} for index in range(3)]}


@pytest.mark.parametrize("reply,level,check", [
    ('{"verdict": "supported", "reason": "ok"}', "verified", "supported"),
    ('{"verdict": "contradicted", "reason": "other value"}', "best_effort", "contradicted"),
    ("garbled", "best_effort", "no_verdict"),
])
@pytest.mark.asyncio
async def test_a_gate_passed_answer_is_labelled_by_the_independent_check(tmp_path, monkeypatch, reply, level, check):
    ask = AsyncMock(return_value=reply)
    monkeypatch.setattr(settlement, "_ask_flash", ask)
    model = ClosingModel([_fetch, _finish], "unused")
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-check-" + check,
                                 progressive_config={**TIER, "max_requirement_words": 20, "independent_check": True})
    assert result["final_answer"] == f"Answer [1].\n\nEvidence level: {level}"
    assert saved["settlement"] == {"kind": "independent_check", "level": level, "check": check,
                                   "check_reason": json.loads(reply)["reason"] if reply != "garbled" else "",
                                   "unverified_question_parts": []}
    ask.assert_awaited_once()
    assert model.prompts == []
    assert PAGE_QUOTE in ask.await_args.args[0] and "Answer [1]." in ask.await_args.args[0]


@pytest.mark.asyncio
async def test_a_failed_check_keeps_the_answer_and_verifies_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settlement, "_ask_flash", AsyncMock(side_effect=RuntimeError("down")))
    model = ClosingModel([_fetch, _finish], "unused")
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-check-down",
                                 progressive_config={**TIER, "max_requirement_words": 20, "independent_check": True})
    assert result["final_answer"].startswith("Answer [1].")
    assert saved["settlement"]["level"] == "best_effort" and saved["settlement"]["check"] == "no_verdict"


@pytest.mark.asyncio
async def test_the_closing_step_stays_off_unless_a_profile_enables_it(tmp_path, monkeypatch):
    ask = AsyncMock()
    monkeypatch.setattr(settlement, "_ask_flash", ask)
    model = ClosingModel([_fetch, _finish], "unused")
    result, saved = await _drive(tmp_path, monkeypatch, model, task="v2-off",
                                 progressive_config={**TIER, "max_requirement_words": 20})
    assert result["final_answer"] == "Answer [1]."
    assert "settlement" not in saved
    ask.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_check_request_counts_as_the_single_final_request():
    with convergence_run(enabled=True, progressive_research_v1=True, progressive_config=TIER,
                         bus_task_id="v2-final", question=QUESTION) as state:
        state.request_finalization("test")
        assert state.finalization_request_id() is None
        with state.answer_check_scope():
            assert state.is_finalization_request() and state.checking_answer
        assert state.finalization_request_id() is None and not state.checking_answer


@pytest.mark.asyncio
async def test_the_coordinator_is_told_once_when_searches_run_out():
    from types import SimpleNamespace

    observer = ConvergenceObserver()
    with _run(**TIER, max_search_queries=1) as state:
        await guard_tools([_web_search_tool()], coordinator=True)[0].func(q="one")
        context = SimpleNamespace(task_id="v2", turn=1, tool_calls=[object()], ai_text="", finish_reason="tool_calls",
                                  metadata={})
        first = await observer.on_turn_end(context)
        context.turn = 2
        second = await observer.on_turn_end(context)
    assert any("search budget is used" in item for item in first.inject_messages)
    assert second is None
