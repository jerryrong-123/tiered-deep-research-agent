"""Exercise actual native create/assign/search/report flow against HTTP fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx


class NativeSearchReuseTests(unittest.IsolatedAsyncioTestCase):
    async def run_native(self, *, enabled):
        from benchmarks.public.core.kernel_adapter import BenchmarkSession
        from frontier_agent.infra.config import get_config
        from workflows.agent_team.nodes.main_agent import _llm_cache

        searches = []
        model_requests = []
        worker_steps = {"ONE": 0, "TWO": 0}
        coordinator = {"created": False, "one": False, "two": False}

        def sse(payload, *, tool=None, arguments=None, text=""):
            delta = {"role": "assistant"}
            if tool:
                delta["tool_calls"] = [{
                    "index": 0, "id": f"fixture_{len(model_requests)}",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments)},
                }]
            else:
                delta["content"] = text
            frames = [
                {"id": "offline", "object": "chat.completion.chunk", "created": 0,
                 "model": payload["model"],
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": "offline", "object": "chat.completion.chunk", "created": 0,
                 "model": payload["model"],
                 "choices": [{"index": 0, "delta": {},
                              "finish_reason": "tool_calls" if tool else "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                           "total_tokens": 120}},
            ]
            body = b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=body + b"data: [DONE]\n\n")

        async def respond(request):
            payload = json.loads(request.content)
            if request.url.host == "serper.invalid":
                searches.append(payload)
                return httpx.Response(200, json={"organic": [{
                    "title": "Shared source", "link": "https://example.com/source",
                    "snippet": "Public evidence for the offline interface fixture.",
                }]})
            if request.url.host != "model.invalid":
                raise AssertionError(f"Unexpected external request: {request.url.host}")
            model_requests.append(payload)
            tools = {item["function"]["name"] for item in payload.get("tools", [])}
            if "create_subagent" in tools:
                if not coordinator["created"]:
                    coordinator["created"] = True
                    return sse(payload, tool="create_subagent", arguments={"agents": [
                        {"name": "probe_one", "system_prompt": "CACHE_WORKER_ONE"},
                        {"name": "probe_two", "system_prompt": "CACHE_WORKER_TWO"},
                    ]})
                if not coordinator["one"]:
                    coordinator["one"] = True
                    return sse(payload, tool="assign_task", arguments={"tasks": [{
                        "agent": "probe_one", "prompt": "Search the shared source and report.",
                    }]})
                context = json.dumps(payload["messages"])
                if "Finding: PROBE_ONE" not in context:
                    return sse(payload, tool="collect_reports", arguments={"timeout": 2})
                if not coordinator["two"]:
                    coordinator["two"] = True
                    return sse(payload, tool="assign_task", arguments={"tasks": [{
                        "agent": "probe_two", "prompt": "Search the shared source and report.",
                    }]})
                if "Finding: PROBE_TWO" not in context:
                    return sse(payload, tool="collect_reports", arguments={"timeout": 2})
                return sse(payload, text="Native interface completed: both source reports received.")
            system = json.dumps([m for m in payload["messages"] if m["role"] == "system"])
            worker = "ONE" if "CACHE_WORKER_ONE" in system else "TWO"
            worker_steps[worker] += 1
            if worker_steps[worker] == 1:
                return sse(payload, tool="web_search", arguments={"q": "shared evidence query"})
            return sse(payload, tool="submit_report", arguments={
                "content": f"Scope: source lookup\nFinding: PROBE_{worker}\n"
                           "Evidence:\n- https://example.com/source\nConfidence: fixture only",
                "confidence": 0.7,
            })

        original_init = httpx.AsyncClient.__init__

        def initialize(client, *args, **kwargs):
            kwargs.update(transport=httpx.MockTransport(respond), trust_env=False)
            original_init(client, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            environment = {
                "OPENAI_MODEL": "deepseek-flash", "OPENAI_API_KEY": "offline-unused",
                "OPENAI_BASE_URL": "https://model.invalid/v1", "OPENAI_PROVIDER": "openai",
                "LLM_PROVIDER": "openai", "SERPER_API_KEY": "offline-unused",
                "SERPER_BASE_URL": "https://serper.invalid", "WEB_SEARCH_PROVIDER": "serper",
                "APODEX_RUN_DIR": temp, "APODEX_SPILL_DIR": str(Path(temp) / "spill"),
                "FRONTIER_AGENT_OUTPUTS_DIR": str(Path(temp) / "outputs"),
                "FRONTIER_AGENT_WORKSPACE_DIR": str(Path(temp) / "workspace"),
                "SANDBOX_BACKEND": "e2b", "E2B_API_KEY": "offline-unused",
            }
            with patch.dict(os.environ, environment), patch.object(httpx.AsyncClient, "__init__", initialize):
                get_config(force_reload=True)
                _llm_cache.clear()
                try:
                    async with BenchmarkSession() as session:
                        result = await session.run(
                            "Run two source lookup tasks and report their public findings.",
                            pipeline_id="agent_team",
                            meta={"profile": "benchmark", "_trial_dir": temp,
                                  "profile_overrides": {
                                      "llm": {"extra_body": {"top_k": None}},
                                      "agent": {"reporter": False,
                                                "share_search_results": enabled,
                                                "sub_agent_tools": ["web_search", "submit_report"]},
                                  }},
                        )
                finally:
                    _llm_cache.clear()
            get_config(force_reload=True)
        self.assertIn("both source reports received", result["final_answer"])
        self.assertEqual(worker_steps, {"ONE": 2, "TWO": 2})
        self.assertEqual(result["answer_status"], "complete")
        return searches, model_requests

    async def test_native_sequential_agents_share_only_when_explicitly_enabled(self):
        off_searches, off_models = await self.run_native(enabled=False)
        on_searches, on_models = await self.run_native(enabled=True)
        self.assertEqual(len(off_searches), 2)
        self.assertEqual(len(on_searches), 1)
        next_run_searches, _ = await self.run_native(enabled=True)
        self.assertEqual(len(next_run_searches), 1)
        self.assertEqual(off_searches[0], on_searches[0])
        # The cache adds no model stage; worker results still pass through the
        # native dispatcher and both actual submit_report tool calls.
        self.assertEqual(len(off_models), len(on_models))
        controls = ("model", "max_tokens", "max_completion_tokens", "temperature",
                    "top_p", "reasoning_effort", "thinking", "tools")
        for off, on in zip(off_models, on_models, strict=True):
            self.assertEqual({k: off.get(k) for k in controls},
                             {k: on.get(k) for k in controls})
