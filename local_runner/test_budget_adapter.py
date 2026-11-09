"""Offline checks using the unchanged real trial ledger and its pricing code."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from decimal import Decimal

from budget_adapter import BudgetAdmissionError, make_meter_class, safe_error_with_budget

HERE = Path(__file__).resolve().parent
DEFAULT_HELPERS = (HERE / "trial_helpers.py") if (HERE / "trial_helpers.py").is_file() else (
    HERE.parent / "frontier_optimization_20261028_01/prepared_0007_shared_v3/trial_helpers.py"
)
PRIOR = Path(os.getenv("TRIAL_HELPERS_SOURCE", str(DEFAULT_HELPERS)))
spec = importlib.util.spec_from_file_location("immutable_trial_helpers_for_budget", PRIOR)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class State:
    phase = "research"
    answer = None
    token = None

    def __init__(self):
        self.reasons = []

    def request_finalization(self, reason):
        self.reasons.append(reason)
        self.phase = "finalizing"

    def finalization_request_id(self):
        return self.token

    def is_finalization_request(self):
        return self.token is not None


class BudgetChecks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        helpers.HERE = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.state = State()
        self.Meter = make_meter_class(helpers, lambda: self.state)

    async def test_children_cannot_starve_final_and_target_stays_soft(self):
        meter = self.Meter()
        self.assertEqual(meter.final_upper, Decimal("2.490368"))
        meter.spent = Decimal("20.3")
        child = await meter.reserve({"max_tokens": 100}, "child")
        await meter.settle(child, {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 0})
        self.assertGreater(meter.spent, Decimal("20"))
        meter.spent = Decimal("22.1")
        with self.assertRaisesRegex(BudgetAdmissionError, "model_budget_research_limit"):
            await meter.reserve({"max_tokens": 32768}, "child")
        self.assertFalse(meter.stop.is_set())
        self.assertEqual(self.state.reasons, ["model_budget_research_limit"])
        self.state.token = object()
        final = await meter.reserve({"max_tokens": 49152}, "coordinator")
        self.assertLessEqual(meter.spent + meter.held + meter.reserved, Decimal("25"))
        await meter.settle(final, {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 0})
        self.assertEqual(meter.reserved, 0)

    async def test_settlement_unblocks_admission_without_adding_calls(self):
        meter = self.Meter()
        meter.spent = Decimal("19")
        first = await meter.reserve({"max_tokens": 32768}, "child")
        waiting = asyncio.create_task(meter.reserve({"max_tokens": 32768}, "child"))
        await asyncio.sleep(0)
        self.assertFalse(waiting.done())
        await meter.settle(first, {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0})
        second = await waiting
        self.assertEqual(meter.calls, 2)
        await meter.cancel(second)
        self.assertEqual((meter.reserved, meter.held), (0, 0))

    async def test_cancel_before_dispatch_releases_but_unknown_remote_cost_stays_held(self):
        meter = self.Meter()
        first = await meter.reserve({"max_tokens": 32768}, "child")
        amount = meter.reserved
        await meter.cancel(first)
        self.assertEqual((meter.reserved, meter.held), (0, 0))
        second = await meter.reserve({"max_tokens": 32768}, "child")
        await meter.mark_dispatched(second)
        await meter.cancel(second)
        bound = min(amount, helpers.held_upper({"max_tokens": 32768}))
        self.assertGreater(bound, 0)
        self.assertEqual((meter.reserved, meter.held), (0, bound))
        held = meter.held
        meter.release_final_allowance()
        self.assertEqual(meter.held, held)

    async def test_final_scope_is_one_logical_response_native_retries_keep_hard_cap(self):
        meter = self.Meter()
        self.state.phase = "finalizing"
        self.state.token = object()
        final = await meter.reserve({"max_tokens": 32768}, "coordinator")
        await meter.mark_dispatched(final)
        await meter.settle(final, None)
        retry = await meter.reserve({"max_tokens": 49152}, "coordinator")
        await meter.cancel(retry)
        unknown_held = meter.held
        # A 49152-token final reserves 2.49, so this leaves it just over the hard cap.
        meter.spent = helpers.HARD - meter.held - Decimal("2")
        with self.assertRaisesRegex(BudgetAdmissionError, "model_budget_admission_exhausted"):
            await meter.reserve({"max_tokens": 49152}, "coordinator")
        self.assertEqual(meter.held, unknown_held)
        self.state.token = object()
        with self.assertRaisesRegex(BudgetAdmissionError, "final_response_already_admitted"):
            await meter.reserve({"max_tokens": 32768}, "coordinator")
        self.assertEqual(meter.calls, 2)

    async def test_children_and_auxiliaries_cannot_claim_final_reserve(self):
        meter = self.Meter()
        self.state.phase = "finalizing"
        for role in ["child", "summary", "coordinator_without_marker"]:
            with self.assertRaisesRegex(BudgetAdmissionError, "model_budget_research_closed"):
                await meter.reserve({"max_tokens": 32768}, role)
        self.assertFalse(meter.final_claimed)
        self.assertEqual(meter.calls, 0)

    async def test_already_latched_answer_uses_no_extra_model_call(self):
        meter = self.Meter()
        self.state.answer = {"answer": "existing final tool response"}
        self.state.token = object()
        with self.assertRaisesRegex(BudgetAdmissionError, "final_answer_already_latched"):
            await meter.reserve({"max_tokens": 32768}, "coordinator")
        self.assertTrue(meter.final_allowance_released)
        self.assertEqual((meter.calls, meter.reserved, meter.held), (0, 0, 0))

    async def test_an_independent_check_is_the_one_call_after_a_latched_answer(self):
        meter = self.Meter()
        self.state.answer = "existing final tool response"
        self.state.phase = "finalizing"
        self.state.token = object()
        self.state.checking_answer = True
        ident = await meter.reserve({"max_tokens": 8192}, "native_completion")
        await meter.mark_dispatched(ident)
        await meter.settle(ident, {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 0})
        self.assertTrue(meter.final_claimed)
        self.assertEqual(meter.calls, 1)
        self.state.token = object()
        with self.assertRaisesRegex(BudgetAdmissionError, "final_response_already_admitted"):
            await meter.reserve({"max_tokens": 8192}, "native_completion")
        self.state.checking_answer = False
        with self.assertRaisesRegex(BudgetAdmissionError, "final_answer_already_latched"):
            await meter.reserve({"max_tokens": 8192}, "native_completion")

    async def test_a_pro_closing_synthesis_fits_the_final_reserve(self):
        payload = {"max_tokens": 32768, "model": "deepseek-v4-pro"}
        self.state.phase = "finalizing"
        self.state.token = object()
        meter = self.Meter()
        meter.configure_final_output_upper(32768, "deepseek-v4-pro")
        ident = await meter.reserve(payload, "native_completion")
        await meter.mark_dispatched(ident)
        self.assertTrue(meter.final_claimed)
        unconfigured = self.Meter()
        unconfigured.configure_final_output_upper(32768)
        with self.assertRaisesRegex(BudgetAdmissionError, "final_response_exceeds_reserved_bound"):
            await unconfigured.reserve(payload, "native_completion")

    async def test_a_cut_call_holds_its_request_size_not_the_full_window(self):
        meter = self.Meter()
        small = {"model": "deepseek-v4-pro", "max_tokens": 32768, "messages": [{"role": "user", "content": "q" * 30000}]}
        ident = await meter.reserve(small, "child")
        reserved = meter.reserved
        await meter.mark_dispatched(ident)
        await meter.settle(ident, None)
        self.assertEqual(meter.reserved, 0)
        self.assertEqual(meter.held, helpers.held_upper(small))
        self.assertLess(meter.held, reserved / 2)
        # A body past the context window holds no more than the full reservation.
        large = {"model": "deepseek-v4-pro", "max_tokens": 32768, "messages": [{"role": "user", "content": "q" * 300000}]}
        ident = await meter.reserve(large, "child")
        reserved = meter.reserved
        await meter.mark_dispatched(ident)
        before = meter.held
        await meter.settle(ident, None)
        self.assertEqual(meter.held - before, reserved)

    async def test_final_hard_cap_cannot_waive_existing_held_amounts(self):
        meter = self.Meter()
        meter.held = Decimal("24")
        self.state.phase = "finalizing"
        self.state.token = object()
        with self.assertRaisesRegex(BudgetAdmissionError, "model_budget_admission_exhausted"):
            await meter.reserve({"max_tokens": 32768}, "coordinator")
        self.assertEqual(meter.held, Decimal("24"))
        self.assertEqual(meter.calls, 0)
        self.assertTrue(meter.stop.is_set())

    async def test_waiting_cancellation_leaves_no_reservation(self):
        meter = self.Meter()
        meter.spent = Decimal("19")
        first = await meter.reserve({"max_tokens": 32768}, "child")
        pending = asyncio.create_task(meter.reserve({"max_tokens": 32768}, "child"))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(meter.calls, 1)
        await meter.cancel(first)
        self.assertEqual((meter.reserved, meter.held), (0, 0))

    async def test_actual_contextvar_and_wrapper_scope_only_reserve_coordinator_final(self):
        from workflows.agent_team.convergence import (
            ConvergenceBlocked, convergence_run, wrap_llm,
        )
        with convergence_run(enabled=True, bus_task_id="offline-budget", question="offline") as state:
            meter = make_meter_class(helpers)()

            class Client:
                async def chat(self):
                    ident = await meter.reserve({"max_tokens": 32768}, "actual_wrapper")
                    await meter.settle(ident, {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0})
                    return "public result"

            coordinator = wrap_llm(Client(), coordinator=True)
            child = wrap_llm(Client(), coordinator=False)
            self.assertEqual(await child.chat(), "public result")
            state.request_finalization("offline_budget_test")
            with state.finalization_scope():
                with self.assertRaises(ConvergenceBlocked):
                    await child.chat()
                self.assertEqual(await coordinator.chat(), "public result")
                with self.assertRaisesRegex(BudgetAdmissionError, "final_response_already_admitted"):
                    await coordinator.chat()
            self.assertEqual(meter.calls, 2)

    async def test_dispatch_rechecks_finalization_and_releases_known_unsent_entry(self):
        from workflows.agent_team.convergence import ConvergenceBlocked, convergence_run
        with convergence_run(enabled=True, bus_task_id="dispatch-race", question="offline") as state:
            meter = make_meter_class(helpers)()
            ident = await meter.reserve({"max_tokens": 32768}, "child")
            state.request_finalization("offline_concurrent_finalize")
            with self.assertRaisesRegex(ConvergenceBlocked, "model_request_dispatch_closed"):
                await meter.mark_dispatched(ident)
            self.assertEqual((meter.reserved, meter.held), (0, 0))
            await meter.cancel(ident)
            self.assertEqual((meter.reserved, meter.held), (0, 0))

    async def test_final_dispatch_requires_same_live_coordinator_logical_marker(self):
        from workflows.agent_team.convergence import ConvergenceBlocked, convergence_run, wrap_llm
        with convergence_run(enabled=True, bus_task_id="final-dispatch-race", question="offline") as state:
            meter = make_meter_class(helpers)()
            state.request_finalization("offline_final")

            class Client:
                async def chat(self):
                    return await meter.reserve({"max_tokens": 32768}, "coordinator")

            with state.finalization_scope():
                ident = await wrap_llm(Client(), coordinator=True).chat()
                with self.assertRaisesRegex(ConvergenceBlocked, "model_request_dispatch_closed"):
                    await meter.mark_dispatched(ident)
            self.assertEqual((meter.reserved, meter.held), (0, 0))

    def test_safe_local_budget_error_survives_remote_sdk_wrapper(self):
        local = BudgetAdmissionError("model_budget_research_limit")
        wrapped = RuntimeError("PRIVATE remote-looking wrapper body")
        wrapped.__cause__ = local
        safe = safe_error_with_budget(helpers.safe_error)(wrapped)
        self.assertEqual(safe["error_domain"], "local_model_budget")
        self.assertEqual(safe["budget_reason"], "model_budget_research_limit")
        self.assertNotIn("PRIVATE", json.dumps(safe))

    def test_unverified_date_keeps_original_conservative_pricing_factor(self):
        from datetime import datetime, timezone
        self.assertEqual(helpers.multiplier(datetime(2026, 10, 29, 0, tzinfo=timezone.utc)), Decimal("2"))
        self.assertIn("factor 2", self.Meter().snapshot()["pricing_date_policy"])


if __name__ == "__main__":
    unittest.main()
