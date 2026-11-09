"""Offline public entry-point regressions for opt-in progressive research."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.agent_team.prompts import get_main_system_prompt, render_team_effort


_SUB_TOOLS = ["web_search", "web_fetch", "read_file", "bash", "download_file"]
_MAIN_TOOLS = ["web_search", "web_fetch", "download_file", "read_file", "create_subagent",
               "assign_task", "collect_reports", "stop_subagent", "add_task", "update_task"]
_CONTROLS = {"review_after_main_turns": 6, "no_progress_cycles": 2,
             "max_helpers_single_gap": 1, "max_helpers_parallel": 2}


@pytest.mark.parametrize("phase,planning,fs_mode,digest", [
    ("combined", False, False, "aecbc14755857eb2b36910593a46f3633bfd0e7fea72509991b4874c8baeb241"),
    ("combined", True, True, "e0b118e0af34c291ab6f70ae27d0f6bb526e4c0cda9ed6b5ef71e6ab920e82b5"),
    ("planning", True, False, "e9d4a0194d8bf34f2a90e9cd64a844d4716b67a2725c5b08b5dc0abdc4497e04"),
    ("execution", True, True, "e9408617880b31441b637a85f5d87e0718819aa91ac29f4671f9ed7ee21ba7aa"),
])
def test_legacy_prompt_bytes_remain_unchanged(phase, planning, fs_mode, digest):
    # These hashes were captured from the frozen source before implementation.
    prompt = get_main_system_prompt("2026-10-30", fs_mode=fs_mode,
        planning_mode=planning, phase=phase, sub_agent_tools=_SUB_TOOLS)
    assert hashlib.sha256(prompt.encode()).hexdigest() == digest
    assert hashlib.sha256(render_team_effort("max").encode()).hexdigest() == (
        "519d224805505c93a6cf45da41b364be9fd275dbb8a75b03f5d9e2ff23d51157")


@pytest.mark.parametrize("phase,planning", [
    ("combined", False), ("combined", True), ("planning", True), ("execution", True),
])
def test_progressive_prompt_replaces_delegation_policy_in_every_phase(phase, planning):
    prompt = get_main_system_prompt("2026-10-30", fs_mode=True,
        planning_mode=planning, phase=phase, sub_agent_tools=_SUB_TOOLS,
        main_agent_tools=_MAIN_TOOLS, progressive_research_v1=True,
        progressive_config=_CONTROLS)
    assert "PROGRESSIVE RESEARCH V1" in prompt
    for tool in ("web_search", "web_fetch", "download_file"):
        assert f"`{tool}`" in prompt
    for legacy_instruction in (
        "you do not solve the sub-questions yourself", "Delegate everything",
        "Delegate ALL", "must be earned by the team", "RUN THE TEAM, not to solve",
        "spawn a final_verifier", "spawn a `final_verifier`", "require >=2 independent agents",
        "Assign literature agents", "must-verify finish gates still apply",
    ):
        assert legacy_instruction not in prompt
    if phase != "planning":
        assert "gap_tasks" in prompt
        assert "source_receipt_ids" in prompt
        assert "technical" in prompt.lower()
        assert "not automatic" in " ".join(prompt.lower().split())


def test_progressive_max_effort_cannot_reintroduce_mandatory_fanout():
    suffix = render_team_effort("max", progressive_research_v1=True)
    assert suffix == "\n\n<team_effort>max</team_effort>"


def test_progressive_prompt_uses_current_controls_and_actual_main_tools():
    prompt = get_main_system_prompt("2026-10-30", planning_mode=False,
        main_agent_tools=["web_search"], progressive_research_v1=True,
        progressive_config={**_CONTROLS, "review_after_main_turns": 9})
    assert "9 effective main decision turns" in prompt
    assert "6 effective main decision turns" not in prompt
    assert "Main-agent tools: `web_search`." in prompt


@pytest.fixture
def finalize_context(monkeypatch):
    from plugins.tools import task_board
    convergence = importlib.import_module("workflows.agent_team.convergence")
    finalize = importlib.import_module("plugins.tools.finalize_answer")
    bus = SimpleNamespace(list_sessions_for_task=lambda task: [])
    monkeypatch.setattr(finalize.registry, "get_optional", lambda cls: bus)
    state = SimpleNamespace(progressive=None)
    monkeypatch.setattr(convergence, "current_convergence", lambda: state)
    task = "progressive-finalize-fixture"
    task_board.start_planning(task)
    task_board.force_finish_planning(task)
    try:
        yield finalize, task_board, state, task
    finally:
        task_board._PHASE.pop(task, None)
        task_board._BOARDS.pop(task, None)


def test_progressive_finish_allows_sourced_solo_answer(finalize_context):
    finalize, _, state, task = finalize_context
    assert "solo" in finalize.finalize_gate(task, "Saved source supports the answer [1].")
    state.progressive = SimpleNamespace()
    assert finalize.finalize_gate(task, "Saved source supports the answer [1].") is None


def test_progressive_finish_preserves_planning_board_and_empty_gates(finalize_context):
    finalize, board, state, task = finalize_context
    state.progressive = SimpleNamespace()
    assert "empty" in finalize.finalize_gate(task, " ")
    board.start_planning(task)
    assert "PLANNING MODE" in finalize.finalize_gate(task, "Candidate answer")
    board.force_finish_planning(task)
    board._BOARDS[task] = {"seq": 1, "tasks": {
        "t1": {"description": "Unanswered original requirement", "resolution": "open", "owners": []},
    }}
    assert "unresolved" in finalize.finalize_gate(task, "Candidate answer")
    board._BOARDS[task]["tasks"]["t1"]["resolution"] = "resolved"
    assert finalize.finalize_gate(task, "Saved source supports the answer [1].") is None


def test_progressive_profile_preserves_runtime_and_model_defaults():
    from copy import deepcopy
    import yaml
    from workflows.agent_team.profile import load_swarm_profile

    directory = Path(__file__).parents[1] / "local_profiles"
    baseline = load_swarm_profile("benchmark", inline=yaml.safe_load(
        (directory / "convergence_v1.yaml").read_text(encoding="utf-8")))
    actual = load_swarm_profile("benchmark", inline=yaml.safe_load(
        (directory / "progressive_research_v1.yaml").read_text(encoding="utf-8")))
    expected = deepcopy(baseline)
    expected["agent"].update({"progressive_research_v1": True,
                             "progressive_config": _CONTROLS})
    expected["agent"]["main_agent_tools"].extend(["web_fetch", "download_file", "read_file"])
    assert actual == expected


class DirectResearchModel:
    model = "offline-progressive-main"

    def __init__(self, *, planning):
        self.planning = planning
        self.calls = 0
        self.prompts = []

    def bind(self, **kwargs):
        return self

    async def chat(self, messages, **kwargs):
        from frontier_agent.core.llm import LLMResponse
        from workflows.agent_team.convergence import current_convergence

        self.calls += 1
        state = current_convergence()
        assert state is not None and state.progressive is not None
        self.prompts.append(messages[0]["content"])
        steps = []
        if self.planning:
            steps += [("add_task", {"tasks": [{"description": "Establish synthetic source relation"}]}),
                      ("finish_planning", {})]
        steps += [("web_search", {"query": "synthetic source"}),
                  ("web_fetch", {"url": "https://example.invalid/direct"})]
        if self.planning:
            steps += [("update_task", {"updates": [{"id": "t1", "resolution": "resolved"}]})]
        if self.calls <= len(steps):
            name, args = steps[self.calls - 1]
        else:
            assert self.calls == len(steps) + 1, "Unexpected extra model synthesis"
            receipts = state.snapshot()["receipts"]
            assert len(receipts) == 2
            assert all(item["agent"] == "main" for item in receipts)
            assert not state.snapshot()["pending_receipt_ids"]
            name, args = "review_research", {
                "action": "finalize", "receipt_ids": [item["receipt_id"] for item in receipts],
                "final_answer": "The synthetic relation is supported [1].\n\n[1] https://example.invalid/direct",
            }
        return LLMResponse(tool_calls=[{"id": f"fixture-{self.calls}", "name": name, "args": args}])

    async def stream(self, messages, **kwargs):
        from frontier_agent.core.llm import StreamDelta
        result = await self.chat(messages, **kwargs)
        yield StreamDelta(tool_call_deltas=[{"index": 0, "id": result.tool_calls[0]["id"],
            "type": "function", "function": {"name": result.tool_calls[0]["name"],
            "arguments": json.dumps(result.tool_calls[0]["args"])}}], finish_reason="tool_calls")


@pytest.mark.asyncio
@pytest.mark.parametrize("planning,fresh", [(False, False), (True, False), (True, True)])
async def test_real_main_direct_sources_finish_without_helper_or_extra_synthesis(
    tmp_path, monkeypatch, planning, fresh,
):
    from frontier_agent.components.agent_bus import AgentBus
    from frontier_agent.core.tool import tool
    from plugins.tools.task_board import add_task, finish_planning, update_task

    node = importlib.import_module("workflows.agent_team.nodes.main_agent")
    bus = AgentBus()
    model = DirectResearchModel(planning=planning)

    @tool
    async def web_search(query: str) -> str:
        """Return a synthetic local search result."""
        return "Synthetic source. URL: https://example.invalid/direct"

    @tool
    async def web_fetch(url: str) -> str:
        """Return a synthetic local page body."""
        return "URL: https://example.invalid/direct\nThe synthetic source establishes the required relation."

    supplied = [web_search, web_fetch, add_task, finish_planning, update_task]
    profile = {"agent": {"planning_mode": planning, "fresh_execution_context": fresh,
        "reporter": False, "main_max_turns": 12, "planning_max_turns": 5,
        "finalization_reserve_turns": 1, "progressive_research_v1": True,
        "progressive_config": _CONTROLS, "team_effort": "max"}}
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
    task = f"progressive-direct-{planning}-{fresh}"
    try:
        result = await asyncio.wait_for(node.main_agent_node(
            {"original_question": "Establish the synthetic relation.", "metadata": {}},
            SimpleNamespace(task_id=task)), timeout=10)
        assert result["stopped_by"] == "final_answer"
        assert result["final_answer_source"] == "convergence_v1"
        assert model.calls == (6 if planning else 3)
        assert result["research_cleanup"]["cleanup_confirmed"]
        saved = json.loads((tmp_path / "trajectories/convergence_v1.json").read_text())
        assert saved["phase"] == "finished"
        assert saved["progressive"]["config"] == _CONTROLS
        assert len(saved["receipts"]) == 2
        assert not bus.list_sessions_for_task(task)
        assert all("PROGRESSIVE RESEARCH V1" in prompt for prompt in model.prompts)
        assert all("Verify, then finish" not in prompt for prompt in model.prompts)
    finally:
        await bus.cleanup_task(task)

