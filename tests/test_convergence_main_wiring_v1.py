"""Real coordinator node/loop and AgentBus, with only the model seam scripted."""
from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.loop_types import BaseObserver, Intervention
from plugins.tools.collect_reports import collect_reports
from workflows.agent_team.convergence import current_convergence


class SourceModel:
    model = "offline-source"

    def __init__(self, blocked=False):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    async def chat(self, messages, **kwargs):
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return LLMResponse(content="Synthetic source: the fixture requirements are supported.")


class SourceReport(BaseObserver):
    critical = True

    async def on_llm_response(self, ctx):
        ctx.metadata["final_answer"] = ctx.ai_text
        ctx.metadata["evidence_cards"] = [{"url": "https://example.invalid/fixture", "text": ctx.ai_text}]
        return Intervention(stop_reason="submit_report")


class MainModel:
    model = "offline-coordinator"

    def __init__(self, *, enabled):
        self.calls = 0
        self.enabled = enabled
        self.bound = []

    def bind(self, **kwargs):
        self.bound.append(kwargs)
        return self

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if not self.enabled:
            return LLMResponse(content="Baseline plain answer.", finish_reason="stop")
        if self.calls == 1:
            return LLMResponse(tool_calls=[{"id": "collect", "name": "collect_reports", "args": {"timeout": 1}}])
        if self.calls == 2:
            state = current_convergence()
            return LLMResponse(tool_calls=[{
                "id": "finish", "name": "review_research", "args": {
                    "action": "finalize", "receipt_ids": state.snapshot()["pending_receipt_ids"],
                    "final_answer": "Fixture answer supported by the saved source [1].\n\n[1] https://example.invalid/fixture",
                },
            }])
        raise AssertionError("An extra coordinator synthesis/research call was sent")

    async def stream(self, messages, **kwargs):
        response = await self.chat(messages, **kwargs)
        if response.tool_calls:
            calls = [{"index": i, "id": call["id"], "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["args"]),
            }} for i, call in enumerate(response.tool_calls)]
            yield StreamDelta(tool_call_deltas=calls, finish_reason="tool_calls")
        else:
            yield StreamDelta(content=response.content, finish_reason="stop")


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_main_node_delivers_without_waiting_for_unneeded_sibling(tmp_path, monkeypatch, enabled):
    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    bus = AgentBus()
    main = MainModel(enabled=enabled)
    task_id = "coordinator-wiring"
    source = SourceModel()
    busy = SourceModel(blocked=True)
    if enabled:
        sid = await bus.create_session(task_id=task_id, name="source", role_id="researcher",
            system_prompt="Synthetic offline fixture.", tools_override=[], llm_override=source)
        good = await bus.submit_task_to_session(sid, "source", observers=[SourceReport()])
        sid = await bus.create_session(task_id=task_id, name="busy", role_id="researcher",
            system_prompt="Synthetic offline fixture.", tools_override=[], llm_override=busy)
        running = await bus.submit_task_to_session(sid, "unneeded exploration", observers=[SourceReport()])
        queued = await bus.submit_task_to_session(sid, "must not start", observers=[SourceReport()])
        await asyncio.wait_for(busy.started.wait(), timeout=1)
        while bus.get_job_status(good) != "completed":
            await asyncio.sleep(0)
    profile = {"agent": {"planning_mode": False, "reporter": False, "main_max_turns": 6,
                         "convergence_v1": enabled}}
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    for part in ("WORKSPACE", "INPUTS", "OUTPUTS"):
        path = tmp_path / part.lower()
        path.mkdir()
        monkeypatch.setenv("FRONTIER_AGENT_" + part + "_DIR", str(path))
    monkeypatch.setattr(node, "_resolve_llm_and_profile", lambda *a, **k: (main, profile, None))
    monkeypatch.setattr(node.registry, "get", lambda cls: bus if cls is AgentBus else object())
    monkeypatch.setattr(node.registry, "get_optional", lambda cls: bus if cls is AgentBus else None)
    monkeypatch.setattr(node, "_resolve_profile_tools", lambda *a, **k:
        ([collect_reports], ["collect_reports"]) if k.get("role_id") == node.MAIN_ROLE_ID else ([], []))
    monkeypatch.setattr(node, "_resolve_trajectory_dir", lambda *a: tmp_path / "trajectories")
    monkeypatch.setattr(node, "_log_inputs_dir_contents", lambda *a, **k: None)
    try:
        result = await node.main_agent_node(
            {"original_question": "Synthetic question: identify the fixture answer.", "metadata": {}},
            SimpleNamespace(task_id=task_id),
        )
        assert result["stopped_by"] == "final_answer"
        if enabled:
            assert "Fixture answer" in result["final_answer"]
            assert result["final_answer_source"] == "convergence_v1"
            assert main.calls == 2
            assert busy.cancelled.is_set()
            assert bus.get_job_status(running) == "aborted"
            assert bus.get_job_status(queued) == "aborted"
            assert result["research_cleanup"]["cleanup_confirmed"]
            assert any(card.get("url") == "https://example.invalid/fixture" for card in result["evidence_cards"])
            saved = json.loads((tmp_path / "trajectories/convergence_v1.json").read_text())
            assert saved["phase"] == "finished"
            assert len(saved["receipts"]) == 1
        else:
            assert result["final_answer"] == "Baseline plain answer."
            assert main.calls == 1
            assert "convergence" not in result
            assert not (tmp_path / "trajectories/convergence_v1.json").exists()
    finally:
        busy.release.set()
        await bus.cleanup_task(task_id)


@pytest.mark.asyncio
async def test_plain_text_cannot_bypass_an_explicit_unresolved_gap():
    from frontier_agent.components.agent_bus import SubAgentResult
    from frontier_agent.core.loop_types import TurnContext
    from workflows.agent_team.convergence import convergence_run
    from workflows.agent_team.observers.bare_text_finalize import BareTextFinalizeObserver

    with convergence_run(enabled=True, bus_task_id="gaps", question="Synthetic original requirement") as state:
        state.observe_reports([("source", SubAgentResult(
            question="fixture", role_id="researcher", final_content="Only a partial source was found.",
            success=True, job_id="partial", metadata={"stopped_by": "submit_report"},
        ))])
        await state.review_tool().ainvoke({
            "action": "continue", "receipt_ids": state.snapshot()["pending_receipt_ids"],
            "gaps": ["The completion year is not established in the current material."],
            "blocking_evidence": ["The available year may refer to publication, not completion."],
        })
        ctx = TurnContext(turn=2, max_turns=10, task_id="gaps", role_id="coordinator",
            ai_text="An unsupported plain-text guess.", thinking="", tool_calls=[], messages=[], usage=None,
            metadata={})
        intervention = await BareTextFinalizeObserver().on_llm_response(ctx)
        assert intervention.continue_to_next_turn
        assert "final_answer" not in ctx.metadata
        assert "completion year" in state.snapshot()["approved_gaps"][0]
        assert state.snapshot()["blocking_evidence"]


@pytest.mark.asyncio
async def test_pending_new_conflict_must_be_seen_before_committing_an_answer():
    from frontier_agent.components.agent_bus import SubAgentResult
    from workflows.agent_team.convergence import convergence_run

    def report(body, job):
        return SubAgentResult(question="fixture", role_id="researcher", final_content=body,
            success=True, job_id=job, metadata={"stopped_by": "submit_report"})

    with convergence_run(enabled=True, bus_task_id="late", question="Synthetic source requirement") as state:
        first = state.observe_reports([("one", report("Initial support", "one"))])[0]

        async def drain_ready():
            state.observe_reports([("two", report("A substantive conflicting date", "two"))])

        decision = await state.review_tool(before_decision=drain_ready).ainvoke({
            "action": "finalize", "receipt_ids": [first["receipt_id"]], "final_answer": "Premature answer",
        })
        assert not decision["ok"]
        assert state.phase == "review_required"
        assert state.answer == ""
        assert any("conflicting date" in receipt["report"] for receipt in state.snapshot()["receipts"])


@pytest.mark.asyncio
async def test_finalize_cannot_simultaneously_declare_missing_evidence_or_silently_drop_a_conflict():
    from frontier_agent.components.agent_bus import SubAgentResult
    from workflows.agent_team.convergence import convergence_run

    with convergence_run(enabled=True, bus_task_id="consistent-decision", question="Synthetic requirement") as state:
        receipt = state.observe_reports([("source", SubAgentResult(
            question="fixture", role_id="researcher", final_content="Public fixture evidence.",
            success=True, job_id="source", metadata={"stopped_by": "submit_report"},
        ))])[0]
        tool = state.review_tool()
        contradictory = await tool.ainvoke({
            "action": "finalize", "receipt_ids": [receipt["receipt_id"]],
            "gaps": ["The required relationship remains unproved."], "final_answer": "Unsupported answer",
        })
        assert not contradictory["ok"]
        assert state.answer == ""
        await tool.ainvoke({"action": "continue", "receipt_ids": [receipt["receipt_id"]],
            "gaps": ["Two sources disagree about the date."], "blocking_evidence": ["Different event endpoints."]})
        dropped = await tool.ainvoke({"action": "finalize", "receipt_ids": [receipt["receipt_id"]],
            "final_answer": "An answer with no account of that conflict."})
        assert not dropped["ok"]
        assert state.blocking_evidence == ["Different event endpoints."]
