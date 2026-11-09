"""Offline planning paths through the actual main node, loop and native bus."""
from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace

import pytest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.llm import LLMResponse, StreamDelta
from plugins.tools.task_board import add_task, clear_board, finish_planning, start_planning, update_task
from tests.test_convergence_main_wiring_v1 import SourceModel, SourceReport
from workflows.agent_team.convergence import convergence_run, current_convergence
from workflows.agent_team.observers.planning_gate import PlanningGateObserver


class PlanningModel:
    model = "offline-planning-coordinator"

    def __init__(self, *, enabled):
        self.enabled = enabled
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        index = len(self.calls)
        if index == 1:
            name, args = "add_task", {"tasks": [{"description": "Check the fixture source"}]}
        elif index == 2 and self.enabled:
            state = current_convergence()
            name, args = "review_research", {
                "action": "finalize", "receipt_ids": state.snapshot()["pending_receipt_ids"],
                "final_answer": "The planning fixture answer follows the received source [1].\n\n"
                                "[1] https://example.invalid/fixture",
            }
        elif index == 2:
            name, args = "finish_planning", {}
        elif index == 3:
            name, args = "update_task", {"updates": [{"id": "t1", "resolution": "cancelled"}]}
        else:
            return LLMResponse(content="Baseline planning answer.", finish_reason="stop")
        return LLMResponse(tool_calls=[{"id": f"step-{index}", "name": name, "args": args}])

    async def stream(self, messages, **kwargs):
        response = await self.chat(messages, **kwargs)
        if response.tool_calls:
            yield StreamDelta(tool_call_deltas=[{
                "index": i, "id": call["id"], "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(call["args"]),
                },
            } for i, call in enumerate(response.tool_calls)], finish_reason="tool_calls")
        else:
            yield StreamDelta(content=response.content, finish_reason="stop")


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_planning_review_and_delivery_do_not_start_an_extra_execution_loop(
    tmp_path, monkeypatch, fresh, enabled,
):
    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    bus = AgentBus()
    main = PlanningModel(enabled=enabled)
    task_id = f"planning-{fresh}-{enabled}"
    source = SourceModel()
    sid = await bus.create_session(
        task_id=task_id, name="fixture_verifier", role_id="researcher", system_prompt="Offline fixture.",
        tools_override=[], llm_override=source,
    )
    job = await bus.submit_task_to_session(sid, "fixture source", observers=[SourceReport()])
    while bus.get_job_status(job) != "completed":
        await asyncio.sleep(0)
    profile = {"agent": {
        "planning_mode": True, "fresh_execution_context": fresh, "planning_max_turns": 6,
        "main_max_turns": 6, "reporter": False, "convergence_v1": enabled,
    }}
    monkeypatch.setenv("SANDBOX_BACKEND", "native")
    for part in ("WORKSPACE", "INPUTS", "OUTPUTS"):
        path = tmp_path / part.lower()
        path.mkdir()
        monkeypatch.setenv("FRONTIER_AGENT_" + part + "_DIR", str(path))
    monkeypatch.setattr(node, "_resolve_llm_and_profile", lambda *a, **k: (main, profile, None))
    monkeypatch.setattr(node.registry, "get", lambda cls: bus if cls is AgentBus else object())
    monkeypatch.setattr(node.registry, "get_optional", lambda cls: bus if cls is AgentBus else None)
    gate = importlib.import_module("plugins.tools.finalize_answer")
    monkeypatch.setattr(gate.registry, "get_optional", lambda cls: bus if cls is AgentBus else None)
    tools = [add_task, update_task, finish_planning]
    monkeypatch.setattr(node, "_resolve_profile_tools", lambda *a, **k:
        (tools, [t.name for t in tools]) if k.get("role_id") == node.MAIN_ROLE_ID else ([], []))
    monkeypatch.setattr(node, "_resolve_trajectory_dir", lambda *a: tmp_path / "trajectories")
    monkeypatch.setattr(node, "_log_inputs_dir_contents", lambda *a, **k: None)
    try:
        result = await node.main_agent_node(
            {"original_question": "Identify the synthetic fixture answer.", "metadata": {}},
            SimpleNamespace(task_id=task_id),
        )
        assert result["stopped_by"] == "final_answer", main.calls[-1][0][-1]
        schemas = main.calls[0][1]["tools"]
        names = [schema["function"]["name"] for schema in schemas]
        if enabled:
            assert "review_research" in names
            assert len(main.calls) == 2, "A finalized planning result must skip execution and synthesis"
            assert result["final_answer_source"] == "convergence_v1"
            assert "planning fixture answer" in result["final_answer"]
            assert len(result["convergence"]["receipts"]) == 1
            assert result["research_cleanup"]["cleanup_confirmed"]
        else:
            assert names == ["add_task", "update_task", "finish_planning"]
            # The existing planning-only policy uses its no-tool recovery
            # before starting the fresh execution loop; preserve that path.
            assert len(main.calls) == (7 if fresh else 4)
            assert result["final_answer"] == "Baseline planning answer."
            assert "convergence" not in result
            assert "CONVERGENCE V1" not in json.dumps(main.calls[0][0])
    finally:
        await bus.cleanup_task(task_id)
        clear_board(task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_planning_gate_only_exempts_opted_in_control_review(enabled):
    task_id = f"planning-gate-{enabled}"
    start_planning(task_id)
    try:
        with convergence_run(enabled=enabled, bus_task_id=task_id):
            intervention = await PlanningGateObserver().on_tool_call(
                SimpleNamespace(task_id=task_id), {"name": "review_research"},
            )
            if enabled:
                assert intervention is None
            else:
                assert "PLANNING MODE" in intervention.skip_with_result
    finally:
        clear_board(task_id)
