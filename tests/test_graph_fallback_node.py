"""The trial-side helpers of the relation-graph fallback."""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from workflows.agent_team import graph_fallback_node as node


class Client:
    def __init__(self):
        self.calls = 0

    def bind(self, **kwargs):
        bound = Client()
        bound.kwargs = kwargs
        return bound

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(content="ok")

    async def stream(self, *args, **kwargs):
        self.calls += 1
        yield "chunk"


def test_the_budget_gate_refuses_research_calls_once_the_reserve_is_reached():
    state = {"open": True}
    gate = node.BudgetGate(Client(), lambda: state["open"])
    assert asyncio.run(gate.chat()).content == "ok"
    bound = gate.bind(max_tokens=10)
    assert isinstance(bound, node.BudgetGate) and bound.kwargs == {"max_tokens": 10}
    state["open"] = False
    with pytest.raises(node.GraphBudgetClosed):
        asyncio.run(gate.chat())

    async def drain():
        return [chunk async for chunk in gate.stream()]

    with pytest.raises(node.GraphBudgetClosed):
        asyncio.run(drain())


def test_the_search_cap_counts_each_query_and_refuses_past_the_limit():
    cap = node.SearchCap(3)
    assert asyncio.run(cap.on_tool_call(None, {"name": "web_search", "args": {"q": ["a", "b"]}})) is None
    assert asyncio.run(cap.on_tool_call(None, {"name": "web_fetch", "args": {"url": "u"}})) is None
    assert asyncio.run(cap.on_tool_call(None, {"name": "web_search", "args": {"q": "c"}})) is None
    refused = asyncio.run(cap.on_tool_call(None, {"name": "web_search", "args": {"q": "d"}}))
    assert refused.skip_with_result.startswith("[SEARCH LIMIT]") and cap.used == 3


def test_tool_sources_join_what_the_tools_returned_and_nothing_else():
    role = types.SimpleNamespace(value="tool")
    messages = [{"role": "user", "content": "task"}, {"role": "tool", "content": "page one"},
                types.SimpleNamespace(role=role, content=[{"text": "page two"}]),
                {"role": "assistant", "content": "my claim"}]
    assert node.tool_sources(messages) == "page one\npage two"


def test_the_delivered_text_carries_the_level_or_the_non_answer_marker():
    marker = node.render({"answer": "", "settlement": {"level": "none", "reason": "decomposition_failed"}})
    assert marker == "[ESCALATION_REQUIRED] reason=decomposition_failed"
    text = node.render({"answer": "North Isle", "settlement": {"level": "verified"}})
    assert text.startswith("North Isle") and "Evidence level: verified" in text


def test_a_streamed_call_keeps_the_answer_text_and_drops_the_reasoning():
    class Streaming:
        def __init__(self):
            self.messages = None

        async def stream(self, messages, timeout=None):
            self.messages = messages
            for delta in ({"reasoning_content": "thinking"}, {"content": '{"a": '}, {"content": "1}"}, {"usage": {}}):
                yield types.SimpleNamespace(content=delta.get("content"), reasoning_content=delta.get("reasoning_content", ""))

    client = Streaming()
    text = asyncio.run(node.stream_text(client, "prompt", chunk_timeout=5, total_timeout=5))
    assert text == '{"a": 1}' and client.messages == [{"role": "user", "content": "prompt"}]


def test_the_ledger_is_read_beside_the_native_folder(tmp_path):
    assert node.read_ledger(None) == {} and node.read_ledger(tmp_path / "native") == {}
    (tmp_path / "MODEL_LEDGER.json").write_text(json.dumps({"spent_estimate_cny": "3"}), encoding="utf-8")
    assert node.read_ledger(tmp_path / "native") == {"spent_estimate_cny": "3"}


class _Chat:
    """Stands for the provider client: it reads its model on every call."""

    def __init__(self, model):
        self.model, self._client = model, object()

    async def chat(self, *args, **kwargs):
        return self.model

    async def stream(self, *args, **kwargs):
        yield self.model


class _Wrapper:
    def __init__(self, llm):
        self._llm, self.label = llm, "wrapper"

    def __getattr__(self, name):
        return getattr(self._llm, name)

    async def chat(self, *args, **kwargs):
        return await self._llm.chat(*args, **kwargs)

    async def stream(self, *args, **kwargs):
        async for chunk in self._llm.stream(*args, **kwargs):
            yield chunk


def test_research_sub_agents_get_a_copy_of_the_client_that_sends_flash():
    import asyncio

    from workflows.agent_team.graph_fallback_node import with_model

    inner = _Chat("deepseek-v4-pro")
    chain = _Wrapper(_Wrapper(inner))
    flash = with_model(chain, "deepseek-flash")
    assert asyncio.run(flash.chat([])) == "deepseek-flash"
    # The trial's own client is untouched, and the copy shares its connection.
    assert asyncio.run(chain.chat([])) == "deepseek-v4-pro" and flash._llm._llm._client is inner._client
    assert flash.label == "wrapper" and flash is not chain


def test_a_client_chain_the_switch_cannot_see_into_is_refused():
    from workflows.agent_team.graph_fallback_node import with_model

    with pytest.raises(TypeError):
        with_model(object(), "deepseek-flash")


def test_every_leg_of_a_fallback_chain_switches():
    import asyncio
    import dataclasses

    from workflows.agent_team.graph_fallback_node import with_model

    @dataclasses.dataclass
    class Entry:
        model: object
        triggers: tuple = ("any_error",)

    @dataclasses.dataclass
    class Chain:
        entries: list
        model: str = dataclasses.field(init=False, default="")

        def __post_init__(self):
            self.model = self.entries[0].model.model

        async def chat(self, *args, **kwargs):
            return await self.entries[0].model.chat(*args, **kwargs)

        async def stream(self, *args, **kwargs):
            yield self.entries[0].model.model

    chain = Chain([Entry(_Chat("deepseek-v4-pro")), Entry(_Chat("deepseek-v4-pro"))])
    flash = with_model(_Wrapper(chain), "deepseek-flash")
    assert asyncio.run(flash.chat([])) == "deepseek-flash" and flash._llm.model == "deepseek-flash"
    assert [e.model.model for e in flash._llm.entries] == ["deepseek-flash", "deepseek-flash"]
    assert [e.model.model for e in chain.entries] == ["deepseek-v4-pro", "deepseek-v4-pro"]
