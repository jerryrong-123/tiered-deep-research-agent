"""Offline lifecycle checks through AgentBus and its real agent loop."""
from __future__ import annotations

import asyncio
import unittest

from frontier_agent.components.agent_bus import AgentBus
from frontier_agent.core.llm import LLMResponse, StreamDelta
from frontier_agent.core.loop_types import AgentLoopResult, BaseObserver, Intervention
from frontier_agent.core.messages import assistant_msg, system_msg, user_msg
from frontier_agent.core.runtime.loop.compact_llm import LLMSummaryCompactor
from workflows.agent_team.convergence import ConvergenceBlocked, convergence_run, wrap_llm
from workflows.agent_team.convergence_cleanup import cleanup_converged_task
from workflows.agent_team.subagent_runtime import force_final_answer


class _ScriptedModel:
    """External model boundary; the native dispatcher/loop remain real."""

    model = "offline-lifecycle-fixture"

    def __init__(self, *, blocked=False, resist_cancel=False):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()
        self.resist_cancel = resist_cancel
        self.calls = 0
        if not blocked:
            self.release.set()

    async def _response(self):
        self.calls += 1
        self.started.set()
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    if not self.resist_cancel:
                        raise
            return "Finding: retained offline evidence."
        finally:
            self.closed.set()

    async def chat(self, messages, **kwargs):
        return LLMResponse(content=await self._response(), finish_reason="stop")

    async def stream(self, messages, **kwargs):
        yield StreamDelta(content=await self._response(), finish_reason="stop")


class _ReportObserver(BaseObserver):
    critical = True

    async def on_llm_response(self, ctx):
        ctx.metadata["final_answer"] = ctx.ai_text
        ctx.metadata["evidence_cards"] = [{"source": "https://example.invalid/evidence", "text": "retained"}]
        ctx.metadata["assertions"] = [{"claim": "offline fixture retained"}]
        return Intervention(stop_reason="final_answer")


class _ResistantReportObserver(_ReportObserver):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def on_llm_response(self, ctx):
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
        return await super().on_llm_response(ctx)


async def _session(bus, task_id, name, model):
    return await bus.create_session(
        task_id=task_id, name=name, role_id="researcher",
        system_prompt="Offline lifecycle fixture.", tools_override=[],
        llm_override=model,
    )


class NativeConvergenceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_returns_even_when_native_hook_ignores_cancellation(self):
        bus = AgentBus()
        model = _ScriptedModel()
        observer = _ResistantReportObserver()
        session_id = await _session(bus, "run-a", "worker", model)
        await bus.submit_task_to_session(session_id, "work", observers=[observer])
        await asyncio.wait_for(observer.started.wait(), timeout=1)
        cleanup = asyncio.create_task(cleanup_converged_task(
            bus, "run-a", timeout_s=0.05, cancel_timeout_s=0.01,
        ))
        try:
            done, _ = await asyncio.wait({cleanup}, timeout=0.2)
            self.assertTrue(observer.cancelled.is_set(), "fixture must encounter cancellation")
            self.assertTrue(done, "cleanup must return while cancellation is still pending")
            outcome = cleanup.result()
            self.assertFalse(outcome.cleanup_confirmed)
            self.assertEqual(len(outcome.pending_job_ids), 1)
        finally:
            observer.release.set()
            await asyncio.wait_for(cleanup, timeout=1)
            await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)

    async def test_cleanup_preserves_ready_reports_purges_queue_and_isolates_other_run(self):
        bus = AgentBus()
        source = _ScriptedModel()
        busy = _ScriptedModel(blocked=True)
        other = _ScriptedModel(blocked=True)
        source_session = await _session(bus, "run-a", "source", source)
        busy_session = await _session(bus, "run-a", "worker", busy)
        other_session = await _session(bus, "run-b", "worker", other)
        source_job = await bus.submit_task_to_session(source_session, "evidence", observers=[_ReportObserver()])
        busy_job = await bus.submit_task_to_session(busy_session, "running", observers=[_ReportObserver()])
        queued_job = await bus.submit_task_to_session(busy_session, "must not start", observers=[_ReportObserver()])
        other_job = await bus.submit_task_to_session(other_session, "other run", observers=[_ReportObserver()])
        await asyncio.wait_for(asyncio.gather(busy.started.wait(), other.started.wait()), timeout=1)
        while bus.get_job_status(source_job) != "completed":
            await asyncio.sleep(0)
        bus.accumulate_task_metadata("run-a", evidence_cards=[{"source": "already-collected"}], assertions=[])
        try:
            outcome = await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)
            self.assertTrue(outcome.cleanup_confirmed)
            self.assertEqual(outcome.cleaned_count, 2)
            self.assertEqual(outcome.pending_job_ids, [])
            self.assertEqual(bus.get_job_status(busy_job), "aborted")
            self.assertEqual(bus.get_job_status(queued_job), "aborted")
            self.assertEqual(busy.calls, 1)
            self.assertTrue(busy.cancelled.is_set())
            self.assertEqual([name for name, _ in outcome.reports], [source_session])
            self.assertIn({"source": "already-collected"}, outcome.evidence_cards)
            self.assertIn({"source": "https://example.invalid/evidence", "text": "retained"}, outcome.evidence_cards)
            self.assertEqual(outcome.assertions, [{"claim": "offline fixture retained"}])
            self.assertEqual(bus.list_sessions_for_task("run-a"), [])
            self.assertEqual(bus.get_job_status(other_job), "running")
            self.assertFalse(other.cancelled.is_set())
        finally:
            other.release.set()
            await cleanup_converged_task(bus, "run-b", timeout_s=1, cancel_timeout_s=0.01)

    async def test_slow_first_worker_does_not_shield_running_sibling_from_cancellation(self):
        bus = AgentBus()
        blocker = _ResistantReportObserver()
        stubborn_session = await _session(bus, "run-a", "stubborn", _ScriptedModel())
        sibling = _ScriptedModel(blocked=True)
        sibling_session = await _session(bus, "run-a", "sibling", sibling)
        await bus.submit_task_to_session(stubborn_session, "blocked hook", observers=[blocker])
        sibling_job = await bus.submit_task_to_session(sibling_session, "running sibling", observers=[_ReportObserver()])
        await asyncio.wait_for(asyncio.gather(blocker.started.wait(), sibling.started.wait()), timeout=1)
        try:
            outcome = await cleanup_converged_task(bus, "run-a", timeout_s=0.05, cancel_timeout_s=0.01)
            self.assertFalse(outcome.cleanup_confirmed)
            self.assertEqual(bus.get_job_status(sibling_job), "aborted")
            # A native task may already be aborted while its separate provider
            # task is only scheduled to receive CancelledError. Check native
            # cancellation now and the external model acknowledgement separately.
            await asyncio.wait_for(sibling.cancelled.wait(), timeout=1)
            self.assertEqual(len(outcome.pending_job_ids), 1)
        finally:
            blocker.release.set()
            await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)

    async def test_native_cleanup_confirmation_does_not_claim_provider_io_settled(self):
        bus = AgentBus()
        model = _ScriptedModel(blocked=True, resist_cancel=True)
        session_id = await _session(bus, "run-a", "worker", model)
        await bus.submit_task_to_session(session_id, "work", observers=[_ReportObserver()])
        await asyncio.wait_for(model.started.wait(), timeout=1)
        try:
            outcome = await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)
            self.assertTrue(outcome.cleanup_confirmed)
            self.assertTrue(model.cancelled.is_set())
            self.assertFalse(model.closed.is_set(), "core may retain abandoned provider I/O after the session is gone")
        finally:
            model.release.set()
            await asyncio.wait_for(model.closed.wait(), timeout=1)

    async def test_native_child_and_summary_cannot_send_after_final_answer_is_latched(self):
        bus = AgentBus()
        provider = _ScriptedModel()
        coordinator_provider = _ScriptedModel()
        answer = "Fixture answer with its already obtained source."
        with convergence_run(enabled=True, bus_task_id="run-a", question="Original fixture question") as state:
            child = wrap_llm(provider, coordinator=False)
            coordinator = wrap_llm(coordinator_provider, coordinator=True)
            session_id = await _session(bus, "run-a", "source", child)
            await bus.submit_task_to_session(session_id, "source", observers=[_ReportObserver()])
            receipt = await bus.wait_any_session("run-a", timeout=1)
            self.assertIsNotNone(receipt)
            state.observe_reports([receipt])
            reviewed = await state.review_tool().ainvoke({
                "action": "finalize",
                "receipt_ids": state.snapshot()["pending_receipt_ids"],
                "final_answer": answer,
            })
            self.assertTrue(reviewed["ok"])
            self.assertEqual(state.phase, "finalizing")
            self.assertEqual(provider.calls, 1, "native child really used the external model seam before finalization")
            with self.assertRaises(ConvergenceBlocked):
                await child.chat([user_msg("late child request")])
            # Exercise the real summary consumer, which rolls back locally on
            # a permanent admission error instead of reaching the provider.
            compactor = LLMSummaryCompactor(summary_llm=child)
            messages = [system_msg("fixture"), user_msg("Original fixture question")]
            for index in range(4):
                messages.extend([assistant_msg(f"Earlier public result {index}"), user_msg(f"Next step {index}")])
            await compactor.compact(messages, keep_recent=1)
            self.assertEqual(provider.calls, 1)
            outcome = await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)
            self.assertTrue(outcome.cleanup_confirmed)
            result = AgentLoopResult(
                messages=[assistant_msg(answer)], final_content=answer,
                stopped_by="final_answer", metadata={"final_answer": answer},
            )
            with state.finalization_scope():
                delivered = await force_final_answer(result, coordinator, 1, task_description="Original fixture question")
            self.assertEqual(delivered.metadata["final_answer"], answer)
            self.assertEqual(state.answer, answer)
            self.assertEqual(coordinator_provider.calls, 0, "an already latched answer needs no extra model call")

    async def test_late_native_rescue_remains_available_after_cleanup_deadline(self):
        bus = AgentBus()
        blocker = _ResistantReportObserver()
        session_id = await _session(bus, "run-a", "late-source", _ScriptedModel())
        await bus.submit_task_to_session(session_id, "late source", observers=[blocker])
        await asyncio.wait_for(blocker.started.wait(), timeout=1)
        first = await cleanup_converged_task(bus, "run-a", timeout_s=0.05, cancel_timeout_s=0.01)
        self.assertFalse(first.cleanup_confirmed)
        blocker.release.set()
        # Observe the actual native lifecycle boundary rather than sleeping.
        async with asyncio.timeout(1):
            while bus.list_sessions_for_task("run-a"):
                await asyncio.sleep(0)
        await asyncio.sleep(0)  # let the completed cleanup's callbacks retire
        later = await cleanup_converged_task(bus, "run-a", timeout_s=1, cancel_timeout_s=0.01)
        self.assertTrue(later.cleanup_confirmed)
        self.assertIn({"source": "https://example.invalid/evidence", "text": "retained"}, later.evidence_cards)
        self.assertEqual(later.assertions, [{"claim": "offline fixture retained"}])


if __name__ == "__main__":
    unittest.main()
