"""Run the relation-graph fallback inside a trial, in place of the coordinator loop.

``nodes.main_agent`` hands over when the profile sets ``agent.graph_fallback_v1:
true``, after it has built the sub-agent runtime, sandbox and scope. The calls the
controller needs are made from the trial's own clients, so every model call goes
through the trial's metered transport and ledger, and every sub-agent runs on the
agent-team sub-agent runtime with its observers, sandbox and report contract.
"""
from __future__ import annotations

import json
import os
import types
from decimal import Decimal
from pathlib import Path
from typing import Any

from workflows.agent_team import graph_fallback as gf
from workflows.agent_team.graph_fallback_controller import Calls, GraphFallback

STATE_FILE = "graph_v1.json"
RESEARCHER = "graph_researcher"


class GraphBudgetClosed(RuntimeError):
    """Research has reached the reserve kept for the final checks."""


class BudgetGate:
    """A research sub-agent's client: each call first checks the research budget,
    so a sub-agent already running stops at the reserve instead of eating into it."""

    def __init__(self, llm: Any, is_open: Any) -> None:
        self._llm = llm
        self._open = is_open

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def bind(self, **kwargs: Any) -> Any:
        return BudgetGate(self._llm.bind(**kwargs), self._open)

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        if not self._open():
            raise GraphBudgetClosed("graph_research_budget_reached")
        return await self._llm.chat(*args, **kwargs)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        if not self._open():
            raise GraphBudgetClosed("graph_research_budget_reached")
        async for chunk in self._llm.stream(*args, **kwargs):
            yield chunk


class SearchCap:
    """A sub-agent's search allowance: each query in a call counts, and a call
    past the cap is answered with a refusal instead of running."""

    TOOLS = ("web_search", "scholar_search")

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0

    async def on_tool_call(self, ctx: Any, tool_call: dict[str, Any]) -> Any:
        from agent_core.loop_types import ToolCallIntervention

        if str(tool_call.get("name") or "") not in self.TOOLS:
            return None
        args = tool_call.get("args") or {}
        queries = args.get("q") if "q" in args else args.get("query")
        count = len(queries) if isinstance(queries, list) else 1
        if self.used + count > self.limit:
            return ToolCallIntervention(skip_with_result=(
                f"[SEARCH LIMIT] This research task has used {self.used} of its {self.limit} searches. "
                "Read the pages you already found, then finish and report what you could not establish."))
        self.used += count
        return None


async def stream_text(llm: Any, prompt: str, *, chunk_timeout: float, total_timeout: float) -> str:
    """One streamed call; only the answer text is kept, never the reasoning.

    Streaming keeps the connection busy while the model thinks. On 2026-11-03 three
    non-streamed Pro decompositions got one byte at 60 seconds and were closed by the
    peer at about 92, so the first real graph trial ended with no draft.
    """
    import asyncio

    parts = []
    async with asyncio.timeout(total_timeout):
        async for delta in llm.stream([{"role": "user", "content": prompt}], timeout=chunk_timeout):
            chunk = getattr(delta, "content", None)
            if chunk:
                parts.append(chunk)
    return "".join(parts)


def read_ledger(trial_dir: Path | None) -> dict[str, Any]:
    """The trial's model ledger, written by its meter beside the native folder."""
    if trial_dir is None:
        return {}
    try:
        return json.loads((Path(trial_dir).parent / "MODEL_LEDGER.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _field(message: Any, name: str) -> Any:
    return message.get(name) if isinstance(message, dict) else getattr(message, name, None)


def tool_sources(messages: Any) -> str:
    """Everything the sub-agent's tools returned, for checking its quotes."""
    parts = []
    for message in messages or []:
        role = _field(message, "role")
        if str(getattr(role, "value", role)) != "tool":
            continue
        content = _field(message, "content")
        if isinstance(content, list):
            content = "\n".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
        parts.append(str(content or ""))
    return "\n".join(parts)


def render(result: dict[str, Any]) -> str:
    """The delivered text: the answer and how far it was checked, or the explicit
    non-answer marker a stopped light run delivers, which the router reads as none."""
    from workflows.agent_team.answer_settlement import render_final

    answer = str(result.get("answer") or "").strip()
    if not answer:
        return f"[ESCALATION_REQUIRED] reason={result['settlement'].get('reason') or 'no_answer'}"
    return render_final(answer, {"level": result["settlement"]["level"]})


def with_model(llm: Any, model: str) -> Any:
    """The same client chain, sending another model.

    The innermost chat client reads its ``model`` on every call, so a shallow copy
    with ``model`` changed shares its connection, transport and meter; each wrapper
    around it is copied onto the copy. A chain this cannot see into is refused
    rather than left on the trial's model.
    """
    own = vars(llm) if hasattr(llm, "__dict__") else {}

    def twin(**changed: Any) -> Any:
        # Not copy.copy: a wrapper that forwards unknown attributes to its inner
        # client recurses while copy looks for __setstate__ on the empty copy.
        made = object.__new__(type(llm))
        made.__dict__.update(own)
        made.__dict__.update(changed)
        return made

    if isinstance(own.get("model"), str) and "_client" in own:
        return twin(model=model)
    entries = own.get("entries")
    if isinstance(entries, list) and entries and all(hasattr(entry, "model") for entry in entries):
        # A fallback chain (agent_core's LLMFallbackChain): every leg switches, so a
        # failed call does not fall through to the trial's model.
        import dataclasses

        legs = [dataclasses.replace(entry, model=with_model(entry.model, model)) if dataclasses.is_dataclass(entry)
                else with_model(entry, model) for entry in entries]
        if dataclasses.is_dataclass(llm):
            return dataclasses.replace(llm, entries=legs)
        return twin(entries=legs)
    for name, inner in own.items():
        if inner is not llm and callable(getattr(inner, "chat", None)) and callable(getattr(inner, "stream", None)):
            return twin(**{name: with_model(inner, model)})
    raise TypeError(f"graph_subagent_model_unswitchable:{type(llm).__name__}")


class SubagentRunner:
    """One research sub-agent per card, on the agent-team sub-agent runtime.

    Research cards run on ``subagent_model`` when one is set; the re-verifier is
    part of the final checks and stays on the trial's model.
    """

    def __init__(self, *, runtime: Any, scope_meta: dict[str, Any], task_id: str, is_open: Any,
                 subagent_model: str = "") -> None:
        from plugins.tools.create_subagent import _bind_sub_agent_llm, _resolve_specialist_prompt
        from workflows.agent_team.subagent_runtime import build_swarm_session_runtime_spec

        self.runtime = runtime
        self.scope_meta = scope_meta
        self.task_id = task_id
        self.llm = _bind_sub_agent_llm(runtime)
        self.research_llm = self.llm
        if subagent_model:
            source = types.SimpleNamespace(sub_agent_llm=with_model(runtime.sub_agent_llm, subagent_model),
                                           llm_max_tokens=getattr(runtime, "llm_max_tokens", None))
            self.research_llm = _bind_sub_agent_llm(source)
        self.is_open = is_open
        self.system_prompt = _resolve_specialist_prompt(RESEARCHER, "research", fs_mode=bool(runtime.fs_mode),
                                                        runtime=runtime)
        self.build_spec = build_swarm_session_runtime_spec
        self.count = 0

    async def run(self, card: dict[str, Any]) -> dict[str, Any]:
        from frontier_agent.core.runtime.loop.agent_loop import run_agent_loop
        from workflows.agent_team.identity import SUB_ROLE_ID

        self.count += 1
        name = str(card["agent"])
        job_id = f"{self.task_id}.graph.{self.count:03d}"
        item = types.SimpleNamespace(question=card["prompt"], role_id=SUB_ROLE_ID, metadata={})
        spec = self.build_spec(self.runtime, session_name=name, task_id=self.task_id)
        config = spec.config_builder(job_id, item, int(card["max_turns"]))
        observers = [*spec.observers_builder(job_id, item, 1), SearchCap(int(card["max_queries"]))]
        # The re-verifier is part of the final checks, which the reserve exists for.
        llm = self.llm if card.get("role") == "reverifier" else BudgetGate(self.research_llm, self.is_open)
        with spec.context_setup(job_id, item):
            raw = await run_agent_loop(
                system_prompt=self.system_prompt, user_message=card["prompt"], llm=llm,
                tools=list(self.runtime.sub_agent_tools or []), config=config, observers=observers,
                model_profile=spec.model_profile, history_policy=spec.history_policy,
                scope_metadata={**self.scope_meta, "graph_fallback_agent": name},
            )
        # submit_report's content is latched into the loop metadata by
        # FinalizeAnswerObserver; final_content may hold other closing text.
        report = (getattr(raw, "metadata", None) or {}).get("final_answer") or getattr(raw, "final_content", "")
        return {"report": str(report or ""), "sources": tool_sources(getattr(raw, "messages", [])),
                "stopped_by": str(getattr(raw, "stopped_by", "") or "")}


async def run_graph_fallback(*, question: str, metadata: dict[str, Any], llm: Any, runtime: Any,
                             scope_meta: dict[str, Any], agent_cfg: dict[str, Any], directory: Path,
                             task_id: str) -> dict[str, Any]:
    from workflows.agent_team import answer_settlement

    trial_dir = Path(metadata["_trial_dir"]) if metadata.get("_trial_dir") else None
    config = dict(agent_cfg.get("graph_fallback_config") or {})
    settings = gf.settings(config)
    timeout = float(agent_cfg.get("llm_timeout_s", 600))
    whole_call = float(agent_cfg.get("logical_call_timeout_s", 900))

    def ledger() -> dict[str, Any]:
        return read_ledger(trial_dir)

    hard_cap = Decimal(str(ledger().get("hard_cap_cny") or os.environ.get("TRIAL_HARD_CNY") or "50"))
    reserve = Decimal(str(settings["verification_reserve_cny"]))

    def research_open() -> bool:
        return gf.research_allowed(ledger(), hard_cap, reserve)

    async def ask_pro(prompt: str) -> str:
        return await stream_text(llm, prompt, chunk_timeout=timeout, total_timeout=whole_call)

    async def ask_flash(prompt: str) -> str:
        return str(await answer_settlement._ask_flash(prompt, timeout) or "")

    runner = SubagentRunner(runtime=runtime, scope_meta=scope_meta, task_id=task_id, is_open=research_open,
                            subagent_model=str(settings["subagent_model"] or ""))
    directory.mkdir(parents=True, exist_ok=True)
    controller = GraphFallback(question, Calls(ask_pro, ask_flash, runner.run, ledger), config=config,
                               hard_cap=hard_cap, state_path=directory / STATE_FILE,
                               earlier_answer=str(metadata.get("earlier_answer") or ""))
    result = await controller.run()
    text = render(result)
    found = bool(str(result.get("answer") or "").strip())
    return {
        "final_answer": text,
        "final_content": text,
        "answer_status": "found" if found else "not_found",
        "answer_sentinel": "" if found else "<ANSWER_NOT_FOUND>",
        "final_answer_source": "graph_fallback",
        "stopped_by": "graph_fallback",
        "graph_fallback": result["settlement"],
        "reporter_enabled": False,
        "llm_error": "",
        "llm_error_reason": "",
    }
