"""Real main loop/dispatcher/observers; only external model and web seams scripted.

These synthetic lifecycle fixtures are not benchmark model success claims.
"""
from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.tool import Tool
from workflows.agent_team.convergence import current_convergence


class SoloModel:
    model = "offline-progressive-native"

    def __init__(self):
        self.calls = 0

    def bind(self, **kwargs):
        return self

    async def chat(self, messages, **kwargs):
        self.calls += 1
        state = current_convergence()
        assert state.progressive is not None
        if self.calls == 1:
            assert not state.snapshot()["receipts"]
            return LLMResponse(tool_calls=[{"id": "actual-search-1", "name": "web_search",
                                            "args": {"query": "synthetic fixture event year"}}])
        if self.calls == 2:
            assert state.phase == "research"  # No obligatory per-page review round.
            return LLMResponse(tool_calls=[{"id": "actual-fetch-1", "name": "web_fetch",
                                            "args": {"url": "https://example.invalid/fixture"}}])
        if self.calls == 3:
            receipts = state.snapshot()["receipts"]
            rendered = json.dumps(messages, default=str)
            assert len(receipts) == 2
            assert all(r["receipt_id"] in rendered for r in receipts)
            assert state.snapshot()["pending_receipt_ids"] == []
            return LLMResponse(tool_calls=[{"id": "actual-commit-1", "name": "review_research", "args": {
                "action": "finalize", "receipt_ids": [r["receipt_id"] for r in receipts],
                "final_answer": "Fixture Harbour opened in 2019. [1]\n\n[1] https://example.invalid/fixture",
            }}])
        raise AssertionError("Unexpected extra review, summary, or verifier model call")

    async def stream(self, messages, **kwargs):
        response = await self.chat(messages, **kwargs)
        yield StreamDelta(tool_call_deltas=[{
            "index": i, "id": call["id"], "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["args"]),
            },
        } for i, call in enumerate(response.tool_calls)], finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_solo_research_runs_through_native_main_and_stops_without_child_or_extra_review(tmp_path, monkeypatch):
    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    bus, model = AgentBus(), SoloModel()
    calls = []

    async def search(**kwargs):
        calls.append(("search", kwargs))
        return "Synthetic search result: https://example.invalid/fixture"

    async def fetch(**kwargs):
        calls.append(("fetch", kwargs))
        return "Synthetic original source: Fixture Harbour opened in 2019."

    tools = [Tool(name="web_search", description="Offline web boundary", func=search,
                  parameters={"type": "object", "properties": {"query": {"type": "string"}}}),
             Tool(name="web_fetch", description="Offline web boundary", func=fetch,
                  parameters={"type": "object", "properties": {"url": {"type": "string"}}})]
    profile = {"agent": {"planning_mode": False, "reporter": False, "main_max_turns": 8,
                         "progressive_research_v1": True}}
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    for part in ("WORKSPACE", "INPUTS", "OUTPUTS"):
        path = tmp_path / part.lower()
        path.mkdir()
        monkeypatch.setenv("FRONTIER_AGENT_" + part + "_DIR", str(path))
    monkeypatch.setattr(node, "_resolve_llm_and_profile", lambda *a, **k: (model, profile, None))
    monkeypatch.setattr(node.registry, "get", lambda cls: bus if cls is AgentBus else object())
    monkeypatch.setattr(node.registry, "get_optional", lambda cls: bus if cls is AgentBus else None)
    monkeypatch.setattr(node, "_resolve_profile_tools", lambda *a, **k:
        (tools, [t.name for t in tools]) if k.get("role_id") == node.MAIN_ROLE_ID else ([], []))
    monkeypatch.setattr(node, "_resolve_trajectory_dir", lambda *a: tmp_path / "trajectories")
    monkeypatch.setattr(node, "_log_inputs_dir_contents", lambda *a, **k: None)
    try:
        result = await asyncio.wait_for(node.main_agent_node(
            {"original_question": "When did Fixture Harbour open?", "metadata": {}},
            SimpleNamespace(task_id="solo-native"),
        ), timeout=10)
        assert result["stopped_by"] == "final_answer"
        assert result["final_answer"].startswith("Fixture Harbour opened in 2019")
        assert model.calls == 3
        assert [name for name, _ in calls] == ["search", "fetch"]
        assert bus.list_sessions_for_task("solo-native") == []
        snapshot = result["convergence"]
        assert snapshot["phase"] == "finished"
        assert len(snapshot["receipts"]) == 2
        assert snapshot["receipts"][1]["source"]["tool_call_id"] == "actual-fetch-1"
        assert snapshot["receipts"][1]["report"] == "Synthetic original source: Fixture Harbour opened in 2019."
        assert snapshot["progressive"]["effective_main_turns"] == 2
        assert result["research_cleanup"]["cleanup_confirmed"]
        source = snapshot["receipts"][1]["source"]
        saved = json.loads(__import__("pathlib").Path(source["metadata"]["source_path"]).read_text())
        assert saved["report"] == snapshot["receipts"][1]["report"]
    finally:
        await bus.cleanup_task("solo-native")


@pytest.mark.asyncio
async def test_main_observer_preserves_failure_details_and_does_not_treat_error_as_progress():
    from frontier_agent.core.loop_types import ToolResult, TurnContext
    from workflows.agent_team.convergence import convergence_run
    from workflows.agent_team.observers.convergence import ConvergenceObserver

    ctx = TurnContext(turn=0, max_turns=8, task_id="failure", role_id="main", ai_text="",
                      thinking="", tool_calls=[], messages=[], usage=None, metadata={})
    result = ToolResult(name="web_fetch", args={"url": "https://example.invalid/slow"},
                        result="Technical failure: incomplete response body", duration_ms=200,
                        tool_call_id="real-failed-call", is_error=True, error_kind="incomplete_body")
    with convergence_run(progressive_research_v1=True) as state:
        annotated = await ConvergenceObserver().on_tool_result(ctx, result)
        receipt = state.snapshot()["receipts"][0]
        assert receipt["reason"] == "incomplete_body"
        assert receipt["status"] == "tool_error"
        assert receipt["report"] == result.result
        assert annotated.result == result.result
        assert annotated.host_metadata["research_receipt_id"] == receipt["receipt_id"]
        denied = await state.review_tool().ainvoke({"action": "finalize",
                    "receipt_ids": [receipt["receipt_id"]], "final_answer": "Unsupported guess"})
        assert denied["reason"] == "successful_evidence_receipt_required"
        assert state.phase == "research"
        assert state.progressive.cycles_since_progress == 0


@pytest.mark.asyncio
async def test_receipt_metadata_never_changes_json_tool_result_protocol():
    from frontier_agent.core.loop_types import ToolResult, TurnContext
    from workflows.agent_team.convergence import convergence_run
    from workflows.agent_team.observers.convergence import ConvergenceObserver

    body = '{"organic":[{"title":"Fixture","link":"https://example.invalid/fixture"}]}'
    ctx = TurnContext(turn=0, max_turns=8, task_id="json", role_id="main", ai_text="",
                      thinking="", tool_calls=[{"name": "web_search"}], messages=[], usage=None, metadata={})
    result = ToolResult(name="web_search", args={"query": "fixture"}, result=body,
                        duration_ms=1, tool_call_id="json-source", is_error=False)
    with convergence_run(progressive_research_v1=True):
        observer = ConvergenceObserver()
        delivered = await observer.on_tool_result(ctx, result)
        assert delivered.result == body
        assert json.loads(delivered.result)["organic"]
        receipt_id = delivered.host_metadata["research_receipt_id"]
        notice = await observer.on_turn_end(ctx)
        assert receipt_id in notice.inject_messages[0]


@pytest.mark.asyncio
async def test_fresh_loop_turn_numbers_do_not_hide_resource_checkpoint():
    from workflows.agent_team.convergence import convergence_run

    with convergence_run(progressive_research_v1=True) as state:
        policy = state.progressive
        for loop in ("planning", "execution"):
            for turn in range(3):
                policy.on_main_turn("same-task", turn, effective=True, loop_id=loop)
        assert policy.main_turns == 6
        assert state.phase == "research"  # Still no automatic delegation.


@pytest.mark.parametrize("name,body,failed", [
    ("web_search", "Error: search failed (timeout).", True),
    ("web_fetch", "[1] URL: https://example.invalid\n    Info: [ERROR]: Scraping failed: timeout", True),
    ("web_fetch", "[1] URL: https://example.invalid\n    Info: [ERROR]: Scraping failed: timeout\n[2] URL: https://example.invalid/ok\n    Info: Original factual prose", False),
    ("read_file", "[ERROR]: is a literal phrase quoted by the author", False),
])
def test_existing_tool_error_strings_are_not_mistaken_for_model_difficulty(name, body, failed):
    from workflows.agent_team.progressive import source_failure_kind

    assert bool(source_failure_kind(name, body)) is failed
