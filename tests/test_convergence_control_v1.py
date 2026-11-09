"""Offline behavior at the convergence control, native Tool and LLM seams."""

from __future__ import annotations

import asyncio
import hashlib
import json
import unittest

from frontier_agent.components.agent_bus import SubAgentResult
from frontier_agent.core.tool import Tool
from workflows.agent_team.convergence import (
    ConvergenceBlocked, convergence_run, current_convergence, guard_tools, wrap_llm,
)


def report(text="Exact source report", *, job_id="job-1", stopped_by="submit_report"):
    return SubAgentResult(
        question="Research question", role_id="agent_team_sub", final_content=text,
        success=True, job_id=job_id,
        metadata={"stopped_by": stopped_by, "report_path": "/trial/reports/job-1.md"},
    )


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.bound = None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    async def chat(self, messages, **kwargs):
        state = current_convergence()
        marker = state.finalization_request_id() if state else None
        self.calls.append((messages, kwargs, marker))
        await asyncio.sleep(0)
        self.asserted_marker = state.finalization_request_id() if state else None
        return "response"

    async def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs, current_convergence().finalization_request_id()))
        yield "first"
        yield "second"


def local_tool(name, calls):
    async def run(**kwargs):
        calls.append(kwargs)
        return kwargs
    return Tool(name=name, description="Original description", parameters={"type": "object"},
                func=run, metadata={"original": True})


class ConvergenceControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_report_receipts_are_exact_stable_and_require_review(self):
        original = report(stopped_by="max_turns")
        with convergence_run(enabled=True, bus_task_id="run-a", question="Original") as state:
            receipts = state.observe_reports([("researcher", original)])
            self.assertEqual(state.phase, "review_required")
            self.assertEqual(receipts[0]["report"], "Exact source report")
            self.assertEqual(receipts[0]["status"], "incomplete")
            self.assertEqual(receipts[0]["source"]["stopped_by"], "max_turns")
            receipt_id = receipts[0]["receipt_id"]
            original.metadata["report_path"] = "mutated"
            receipts[0]["report"] = "caller mutation"
            repeated = state.observe_reports([("researcher", original)])
            self.assertEqual(repeated[0]["receipt_id"], receipt_id)
            stored = state.snapshot()["receipts"][0]
            self.assertEqual(state.snapshot()["protocol_version"], "agent_team_convergence_v1")
            self.assertEqual(state.snapshot()["question_sha256"], hashlib.sha256(b"Original").hexdigest())
            self.assertEqual(stored["report"], "Exact source report")
            self.assertEqual(stored["source"]["metadata"]["report_path"], "/trial/reports/job-1.md")
            self.assertIn(receipt_id, state.render_review_message())
            self.assertNotIn("Exact source report", state.render_review_message())
            self.assertEqual(len(stored["source"]["report_sha256"]), 64)

    async def test_scope_inherits_to_children_isolates_nested_runs_and_resets(self):
        self.assertIsNone(current_convergence())
        with convergence_run(enabled=True, bus_task_id="outer", question="A") as outer:
            self.assertIs(await asyncio.create_task(self._current()), outer)
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with convergence_run(enabled=True, bus_task_id="inner", question="B") as inner:
                    self.assertIs(current_convergence(), inner)
                    self.assertIsNot(inner, outer)
                    raise RuntimeError("nested")
            self.assertEqual(inner.phase, "finished")
            self.assertIs(current_convergence(), outer)
            with convergence_run(enabled=False, bus_task_id="disabled", question="C") as disabled:
                self.assertIsNone(disabled)
                self.assertIsNone(current_convergence())
            self.assertIs(current_convergence(), outer)
        self.assertEqual(outer.phase, "finished")
        self.assertIsNone(current_convergence())

    async def _current(self):
        return current_convergence()

    async def test_decisions_require_all_pending_receipts_and_specific_gaps(self):
        with convergence_run(enabled=True) as state:
            ids = [r["receipt_id"] for r in state.observe_reports([
                ("one", report()), ("two", report("Conflicting source", job_id="job-2")),
            ])]
            review = state.review_tool()
            missing = await review.ainvoke({"action": "continue", "receipt_ids": ids[:1],
                                            "gaps": ["Read the dated original source"]})
            self.assertFalse(missing["ok"])
            self.assertEqual(state.phase, "review_required")
            empty = await review.ainvoke({"action": "continue", "receipt_ids": ids, "gaps": [" "]})
            self.assertFalse(empty["ok"])
            accepted = await review.ainvoke({"action": "continue", "receipt_ids": ids,
                                            "gaps": ["Read the dated original source"],
                                            "blocking_evidence": ["Two dates conflict"]})
            self.assertTrue(accepted["ok"])
            self.assertEqual(state.phase, "research")
            receipt = state.observe_reports([("one", report("The source date", job_id="job-3"))])[0]
            final = await review.ainvoke({"action": "finalize", "receipt_ids": [receipt["receipt_id"]],
                                         "final_answer": "Answer with a stated date limitation",
                                         "blocking_evidence": [], "disposition": "Date limitation disclosed"})
            self.assertTrue(final["ok"])
            self.assertEqual(state.phase, "finalizing")
            self.assertEqual(state.answer, "Answer with a stated date limitation")
            self.assertEqual(state.blocking_evidence, ["Two dates conflict"])
            self.assertEqual(state.approved_gaps, ["Read the dated original source"])
            self.assertEqual(len(state.snapshot()["decisions"]), 2)
            again = await review.ainvoke({"action": "finalize", "receipt_ids": [], "final_answer": "Overwrite"})
            self.assertFalse(again["ok"])
            self.assertNotEqual(state.answer, "Overwrite")

    async def test_review_drains_ready_reports_before_atomic_commit(self):
        with convergence_run(enabled=True) as state:
            first = state.observe_reports([("one", report())])[0]
            async def drain():
                state.observe_reports([("two", report("Late counterevidence", job_id="job-2"))])
            result = await state.review_tool(before_decision=drain).ainvoke({
                "action": "finalize", "receipt_ids": [first["receipt_id"]], "final_answer": "Too early",
            })
            self.assertFalse(result["ok"])
            self.assertEqual(state.phase, "review_required")
            self.assertEqual(state.answer, "")
            self.assertEqual(len(result["pending_receipts"]), 2)

    async def test_finalize_needs_a_real_receipt_and_foreign_run_receipts_do_not_count(self):
        with convergence_run(enabled=True) as first:
            foreign_id = first.observe_reports([("one", report())])[0]["receipt_id"]
        with convergence_run(enabled=True) as state:
            empty = await state.review_tool().ainvoke({
                "action": "finalize", "receipt_ids": [], "final_answer": "Confidence alone",
            })
            self.assertFalse(empty["ok"])
            foreign = await state.review_tool().ainvoke({
                "action": "finalize", "receipt_ids": [foreign_id], "final_answer": "Wrong run",
            })
            self.assertFalse(foreign["ok"])
            self.assertEqual(state.answer, "")

    async def test_guards_preserve_schema_and_embed_followup_gap_without_mutating_arguments(self):
        calls = []
        tools = [local_tool(name, calls) for name in ("assign_task", "web_search", "read_file", "collect_reports")]
        with convergence_run(enabled=True) as state:
            guarded = guard_tools(tools, coordinator=True)
            self.assertEqual(guarded[0].to_openai_schema(), tools[0].to_openai_schema())
            initial = {"tasks": [{"agent": "one", "prompt": "Initial investigation"}]}
            await guarded[0].ainvoke(initial)
            self.assertEqual(calls[-1], initial)
            receipt = state.observe_reports([("one", report())])[0]
            for index in (0, 1, 3):
                with self.assertRaises(ConvergenceBlocked):
                    await guarded[index].ainvoke(initial)
            await guarded[2].ainvoke({"path": "/local/evidence.txt"})
            await state.review_tool().ainvoke({"action": "continue", "receipt_ids": [receipt["receipt_id"]],
                                              "gaps": ["Verify the original publication date"]})
            await wrap_llm(FakeLLM(), coordinator=True).chat(["Choose the approved follow-up"])
            followup = {"tasks": json.dumps([{"agent": "one", "prompt": "Inspect source"}])}
            original = json.loads(followup["tasks"])
            await guarded[0].ainvoke(followup)
            self.assertIn("Verify the original publication date", calls[-1]["tasks"][0]["prompt"])
            self.assertEqual(json.loads(followup["tasks"]), original)
            state.request_finalization("budget_limit")
            for coordinator in (True, False):
                guarded_search = guard_tools([tools[1]], coordinator=coordinator)[0]
                with self.assertRaises(ConvergenceBlocked):
                    await guarded_search.ainvoke({})
            await guarded[2].ainvoke({"path": "/local/evidence.txt"})

    async def test_parallel_review_and_research_need_next_existing_coordinator_turn(self):
        for review_first in (True, False):
            with self.subTest(review_first=review_first), convergence_run(enabled=True) as state:
                receipt = state.observe_reports([("one", report())])[0]
                calls = []
                research = guard_tools([local_tool("web_search", calls)], coordinator=True)[0]
                review = state.review_tool()
                async def decide():
                    return await review.ainvoke({
                        "action": "continue", "receipt_ids": [receipt["receipt_id"]],
                        "gaps": ["Read the dated original publication"],
                    })
                operations = [decide(), research.ainvoke({"query": "follow-up"})]
                if not review_first:
                    operations.reverse()
                results = await asyncio.gather(*operations, return_exceptions=True)
                self.assertTrue(any(isinstance(result, ConvergenceBlocked) for result in results))
                self.assertEqual(calls, [])
                await wrap_llm(FakeLLM(), coordinator=True).chat(["Plan approved follow-up"])
                await research.ainvoke({"query": "follow-up"})
                self.assertEqual(len(calls), 1)

    async def test_llm_filter_bind_and_final_scope_only_marks_coordinator_logical_request(self):
        with convergence_run(enabled=True) as state:
            native = FakeLLM()
            main = wrap_llm(native, coordinator=True)
            child = wrap_llm(native, coordinator=False)
            state.observe_reports([("one", report())])
            schemas = [{"type": "function", "function": {"name": name}} for name in
                       ("web_search", "read_file", "review_research", "collect_reports")]
            await main.bind(temperature=0).chat(["review"], tools=schemas)
            self.assertEqual(native.bound, {"temperature": 0})
            self.assertEqual([x["function"]["name"] for x in native.calls[-1][1]["tools"]],
                             ["read_file", "review_research"])
            self.assertEqual(len(schemas), 4)
            state.request_finalization("budget_limit")
            with self.assertRaises(ConvergenceBlocked):
                await main.chat(["unscoped"])
            with state.finalization_scope():
                self.assertIsNone(state.finalization_request_id())
                with self.assertRaises(ConvergenceBlocked):
                    await child.chat(["inherited scope is not a child allowance"])
                await main.chat(["final"])
                marker = native.calls[-1][2]
                self.assertIsNotNone(marker)
                self.assertIs(native.asserted_marker, marker)
                self.assertIsNone(state.finalization_request_id())
                await main.chat(["another logical call"])
                self.assertIsNot(native.calls[-1][2], marker)
            self.assertIsNone(state.finalization_request_id())
        with self.assertRaises(ConvergenceBlocked):
            await main.chat(["escaped finished run"])

    async def test_stream_admission_is_checked_when_iterated_and_marker_resets(self):
        with convergence_run(enabled=True) as state:
            native = FakeLLM()
            wrapped = wrap_llm(native, coordinator=False)
            pending = wrapped.stream(["not started"])
            state.request_finalization("review_finalized")
            with self.assertRaises(ConvergenceBlocked):
                await anext(pending)
            self.assertEqual(native.calls, [])
            main = wrap_llm(native, coordinator=True)
            with state.finalization_scope():
                stream = main.stream(["final"])
                self.assertEqual(await anext(stream), "first")
                self.assertTrue(state.is_finalization_request())
                await stream.aclose()
                self.assertFalse(state.is_finalization_request())

    async def test_disabled_wrappers_are_exact_identity_and_metadata_omits_private_fields(self):
        native = FakeLLM()
        tools = [local_tool("web_search", [])]
        self.assertIs(wrap_llm(native, coordinator=True), native)
        self.assertIs(guard_tools(tools, coordinator=True), tools)
        with convergence_run(enabled=True) as state:
            result = report()
            result.metadata.update({"reasoning_content": "private", "nested": {
                "api_key": "credential", "url": "https://public.example/source"}})
            receipt = state.observe_reports([("one", result)])[0]
            metadata = receipt["source"]["metadata"]
            self.assertNotIn("reasoning_content", metadata)
            self.assertNotIn("api_key", metadata["nested"])
            self.assertEqual(metadata["nested"]["url"], "https://public.example/source")
            state.request_finalization("ready")
            state.observe_reports([("two", report("Late report", job_id="late"))])
            self.assertEqual(state.phase, "finalizing")
            self.assertEqual(len(state.snapshot()["receipts"]), 2)


if __name__ == "__main__":
    unittest.main()
