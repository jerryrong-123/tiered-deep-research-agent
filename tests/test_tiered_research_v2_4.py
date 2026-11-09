"""Offline checks for the calculator and for answer leads checked apart from the page summary."""
from __future__ import annotations

import asyncio
import importlib
import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from local_runner import trial_helpers
from workflows.agent_team.answer_settlement import check_prompt
from workflows.agent_team.progressive import unquoted_calculation_inputs
from workflows.agent_team.prompts import get_main_system_prompt

try:
    from tests.test_tiered_research_v1 import QUESTION, TIER, _run, _source
except ImportError:
    from test_tiered_research_v1 import QUESTION, TIER, _run, _source

calc = importlib.import_module("plugins.tools.calculate")
leads = importlib.import_module("plugins.tools._answer_leads")
fetch = importlib.import_module("plugins.tools.web_fetch_aligned")

SENTENCE = "The Harbor Lane premiere aired on the Northgate network in March 2019"
PAGE = ("Title: Harbor Lane\n\n" + "Background on the series and its cast members. " * 40
        + SENTENCE + ". " + "Later seasons moved to a new studio in the north. " * 40)
SUMMARY = {"success": True, "extracted_info": f'The cast list is given.\nANSWER LEAD: "{SENTENCE}."', "error": ""}
PAGE_RECEIPT = {"receipt_id": "p", "status": "tool_completed", "report": "page", "source": {"tool_name": "web_fetch"}}


def _calculation(expression, inputs, result):
    report = f"CALCULATION: {expression}\nINPUTS: {inputs}\nRESULT: {result}"
    return {"receipt_id": "c", "status": "tool_completed", "report": report, "source": {"tool_name": "calculate"}}


# Calculator --------------------------------------------------------------------

def test_arithmetic_keeps_its_inputs_in_order():
    assert calc.evaluate("(2024 - 1938) * 12") == (1032, [2024, 1938, 12])


def test_a_signed_coordinate_is_one_input_and_rounding_digits_are_none():
    assert calc.evaluate("-73.5686 + 1")[1] == [-73.5686, 1]
    assert calc.evaluate("round(3.14159, 2)") == (3.14, [3.14159])


def test_distance_follows_the_haversine_formula():
    assert 343 < calc.distance_km(48.8566, 2.3522, 51.5074, -0.1278) < 345
    assert calc.distance_km(10, 20, 10, 20) == 0


@pytest.mark.asyncio
async def test_the_result_names_its_inputs():
    result = await calc.calculate.func(expression="distance_km(45.5081, -73.5686, 45.5106, -73.5564)")
    lines = result.splitlines()
    assert lines[0] == "CALCULATION: distance_km(45.5081, -73.5686, 45.5106, -73.5564)"
    assert lines[1] == "INPUTS: 45.5081, -73.5686, 45.5106, -73.5564"
    assert 0.9 < float(lines[2].split(": ")[1]) < 1.1


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [
    "__import__('os')", "(1).__class__", "eval('1')", "lambda: 1", "[1, 2]", "10 ** 1000",
    "round(x=1)", "1 +", "9" * 401, "distance_km(91, 0, 0, 0)", "1 / 0", "",
])
async def test_anything_but_plain_arithmetic_is_refused(expression):
    assert (await calc.calculate.func(expression=expression)).startswith("[ERROR]")


def test_the_calculator_is_a_local_built_in():
    from plugins.tools import get_builtin_tools
    from plugins.tools.meta import TOOL_META
    from workflows.agent_team.convergence import _LOCAL_TOOLS

    assert "calculate" in get_builtin_tools() and "calculate" in _LOCAL_TOOLS
    assert TOOL_META["calculate"].category == "compute" and TOOL_META["calculate"].is_read_only


def test_a_calculation_is_a_receipt_but_not_a_read():
    with _run(**TIER) as state:
        _source(state, text="results", name="web_search", call="search-1")
        receipt = state.observe_main_tool(tool_call_id="calc-1", tool_name="calculate",
                                          arguments={"expression": "2 + 3"},
                                          result="CALCULATION: 2 + 3\nINPUTS: 2, 3\nRESULT: 5")
        assert receipt["source"]["tool_name"] == "calculate" and receipt["status"] == "tool_completed"
        assert state.progressive.cycles_since_progress == 0
        _source(state, call="read-1")
        assert state.progressive.cycles_since_progress == 1


def test_a_calculation_counts_only_when_its_inputs_are_quoted():
    receipts = {"c": _calculation("distance_km(45.5081, -73.5686, 45.5106, -73.5564)",
                                  "45.5081, -73.5686, 45.5106, -73.5564", "0.99"), "p": PAGE_RECEIPT}
    rows = [{"receipt_id": "c", "evidence_quote": "RESULT: 0.99"},
            {"receipt_id": "p", "evidence_quote": "Station A is at 45.5081° N, 73.5686° W"}]
    found = [(0, "c"), (1, "p")]
    assert unquoted_calculation_inputs("Which stop is closer?", rows, found, receipts) == [
        {"row": 0, "problem": "calculation_input_not_quoted", "inputs": ["45.5106", "73.5564"]}]
    rows.append({"receipt_id": "p", "evidence_quote": "Station B is at 45.5106° N, 73.5564° W"})
    assert unquoted_calculation_inputs("Which stop is closer?", rows, [*found, (2, "p")], receipts) == []


def test_numbers_from_the_question_and_unit_factors_need_no_quote():
    receipts = {"c": _calculation("(2021 - 1987) * 12", "2021, 1987, 12", "408"), "p": PAGE_RECEIPT}
    rows = [{"receipt_id": "c", "evidence_quote": "RESULT: 408"},
            {"receipt_id": "p", "evidence_quote": "The studio opened in 1987 in Leeds"}]
    assert unquoted_calculation_inputs("How many months from its opening to 2021?", rows,
                                       [(0, "c"), (1, "p")], receipts) == []


def test_rows_without_a_calculation_are_untouched():
    rows = [{"receipt_id": "p", "evidence_quote": "A page that gives 4.2 km"}]
    assert unquoted_calculation_inputs("q", rows, [(0, "p")], {"p": PAGE_RECEIPT}) == []


def test_the_check_reads_a_calculation_note_only_when_a_quote_is_one():
    plain = check_prompt("q", "a", [{"question_span": "s", "evidence_quote": "a page quote"}])
    computed = check_prompt("q", "a", [{"question_span": "s", "evidence_quote": "CALCULATION: 2 + 3 INPUTS: 2, 3 RESULT: 5"}])
    assert "calculator" not in plain and "computed by a calculator" in computed
    assert computed.index("computed by a calculator") < computed.index("Return JSON only")


def test_the_prompt_asks_for_the_calculator_only_when_switched_on():
    config = {**TIER, "answer_chain_gate": True}
    on = get_main_system_prompt("2026-11-02", progressive_research_v1=True, progressive_config={**config, "calculator": True})
    off = get_main_system_prompt("2026-11-02", progressive_research_v1=True, progressive_config=config)
    assert "Use `calculate`" in on and "Use `calculate`" not in off


# Answer leads ------------------------------------------------------------------

def test_the_lead_line_is_split_from_the_summary():
    assert leads.split_lead('Facts.\n**ANSWER LEAD:** "It aired in 2019 on Northgate."') == (
        "Facts.", "It aired in 2019 on Northgate.")
    assert leads.split_lead("No lead here.") == ("No lead here.", "")


def test_a_lead_counts_only_when_its_sentence_is_in_the_page():
    assert SENTENCE in leads.window(PAGE, SENTENCE)
    assert leads.window(PAGE, "A sentence the page never says at all") == ""
    assert leads.window(PAGE, "Harbor Lane premiere") == ""


def test_the_check_keeps_only_quotes_found_in_the_page_text():
    around = leads.window(PAGE, SENTENCE)
    reply = json.dumps({"verdict": "partial", "quotes": [SENTENCE, "An invented passage that is nowhere at all"]})
    assert leads.parse_check(reply, around) == ("partial", [SENTENCE])
    assert leads.parse_check("not json", around) == ("unreadable", [])
    assert leads.parse_check('{"verdict": "maybe"}', around) == ("unreadable", [])


def test_the_allowance_stops_at_its_limit_and_checks_a_sentence_once():
    allowance = leads.AnswerLeads("q", 2)
    assert allowance.take("u1", SENTENCE) and not allowance.take("u1", SENTENCE)
    assert allowance.take("u2", SENTENCE) and not allowance.take("u3", SENTENCE)
    assert allowance.checked == 2


def test_the_extraction_prompt_gains_the_rule_only_with_a_question():
    assert fetch._extraction_prompt("dates", "PAGE TEXT") == fetch._EXTRACT_INFO_PROMPT.format("dates", "PAGE TEXT")
    prompt = fetch._extraction_prompt("dates", "PAGE TEXT", "Which year?")
    assert "ANSWER LEAD:" in prompt and "INFORMATION TO EXTRACT:\ndates" in prompt
    assert prompt.index("Which year?") < prompt.index("CONTENT TO ANALYZE:") < prompt.index("PAGE TEXT")


@pytest.fixture
def page(monkeypatch):
    """A rendered page holding the lead sentence; nothing leaves the process."""
    monkeypatch.setattr(fetch, "non_public_url_error", AsyncMock(return_value=""))
    monkeypatch.setattr(fetch, "_scrape_url_with_jina", AsyncMock(return_value={"success": True, "content": PAGE, "error": ""}))
    monkeypatch.setattr(fetch, "_extract_info_with_llm", AsyncMock(return_value=SUMMARY))
    monkeypatch.setattr(fetch, "_ask_lead_check", AsyncMock(return_value=json.dumps({"verdict": "answers", "quotes": [SENTENCE]})))
    return fetch


async def _read(url, **config):
    with _run(**TIER, **{"answer_leads": 3, **config}) as state:
        result = await fetch._fetch_single(url, "the premiere date")
    return result, state.snapshot()


@pytest.mark.asyncio
async def test_a_confirmed_lead_reaches_the_research_model(page):
    result, snapshot = await _read("https://example.org/lead-confirmed")
    assert result.startswith("The cast list is given.\n\n")
    assert f'CHECKED ANSWER LEAD (answers, from this page only): "{SENTENCE}."' in result
    assert f'Supporting text: "{SENTENCE}"' in result
    assert "\nANSWER LEAD:" not in result
    assert page._extract_info_with_llm.await_args.kwargs["question"] == QUESTION
    assert snapshot["answer_leads"] == {"offered": 1, "dropped_not_in_page": 0, "checked": 1,
                                        "verdicts": {"answers": 1}, "limit": 3}


@pytest.mark.asyncio
async def test_a_rejected_lead_leaves_no_trace(page):
    page._ask_lead_check.return_value = json.dumps({"verdict": "no", "quotes": []})
    result, snapshot = await _read("https://example.org/lead-rejected")
    assert result == "The cast list is given."
    assert snapshot["answer_leads"]["verdicts"] == {"no": 1}


@pytest.mark.asyncio
async def test_a_lead_not_in_the_page_is_dropped_unchecked(page):
    page._extract_info_with_llm.return_value = {
        **SUMMARY, "extracted_info": "Facts.\nANSWER LEAD: A sentence this page never contains anywhere."}
    result, snapshot = await _read("https://example.org/lead-invented")
    assert result == "Facts."
    page._ask_lead_check.assert_not_awaited()
    assert snapshot["answer_leads"]["dropped_not_in_page"] == 1


@pytest.mark.asyncio
async def test_lead_checks_stop_at_the_run_limit(page):
    with _run(**TIER, answer_leads=1) as state:
        await page._fetch_single("https://example.org/limit-1", "the premiere date")
        second = await page._fetch_single("https://example.org/limit-2", "the premiere date")
    assert page._ask_lead_check.await_count == 1
    assert "CHECKED ANSWER LEAD" not in second and "\nANSWER LEAD:" not in second
    assert state.snapshot()["answer_leads"]["offered"] == 2 and state.snapshot()["answer_leads"]["checked"] == 1


@pytest.mark.asyncio
async def test_without_the_setting_the_summarizer_sees_no_question(page):
    with _run(**TIER) as state:
        await page._fetch_single("https://example.org/lead-off", "the premiere date")
    assert page._extract_info_with_llm.await_args.kwargs["question"] == ""
    page._ask_lead_check.assert_not_awaited()
    assert "answer_leads" not in state.snapshot()


# A trial wraps the summarizer in its answer-label guard. V2.4 first ran with a guard that
# rejected the new argument, so every page read in a trial failed while these tests passed.

def test_the_trial_guard_passes_the_question_to_the_summarizer(tmp_path):
    from local_runner.answer_label_guard import install_guard

    model = AsyncMock(return_value={"success": True, "extracted_info": "Facts.", "error": ""})
    with patch.object(fetch, "_extract_info_with_llm", model):
        _, restore = install_guard(QUESTION, tmp_path)
        try:
            result = asyncio.run(fetch._extract_info_with_llm("A plain page about trains.", "the dates", question="Which year?"))
        finally:
            restore()
    assert result["extracted_info"] == "Facts."
    assert model.await_args.kwargs["question"] == "Which year?"


@pytest.mark.asyncio
async def test_a_lead_comes_through_the_trial_guard(page, tmp_path):
    from local_runner.answer_label_guard import install_guard

    _, restore = install_guard(QUESTION, tmp_path)
    try:
        result, snapshot = await _read("https://example.org/lead-guarded")
    finally:
        restore()
    assert "CHECKED ANSWER LEAD (answers" in result and snapshot["answer_leads"]["checked"] == 1


@pytest.mark.asyncio
async def test_the_ledger_shows_lead_checks_on_their_own_line(tmp_path, monkeypatch):
    monkeypatch.setattr(trial_helpers, "HERE", tmp_path)
    meter = trial_helpers.Meter()
    check = {"model": "deepseek-flash", "max_tokens": 8192,
             "messages": [{"role": "user", "content": leads.check_prompt("q", "s", "w")}]}
    other = {"model": "deepseek-flash", "max_tokens": 8192, "messages": [{"role": "user", "content": "Extract the dates."}]}
    usage = {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_cache_hit_tokens": 0}
    for payload in (check, other):
        await meter.settle(await meter.reserve(payload, "native_completion"), usage)
    await meter.settle(await meter.reserve(check, "native_completion"), None)
    line = meter.snapshot()["lead_check"]
    assert line["calls"] == 2 and Decimal(line["spent_estimate_cny"]) * 2 == meter.spent
    assert Decimal(line["unknown_held_upper_cny"]) == meter.held > 0
    assert leads.LEAD_CHECK_MARK == trial_helpers.LEAD_CHECK_MARK
