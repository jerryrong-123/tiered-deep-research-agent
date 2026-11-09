"""Execute the coordinator's run-local convergence decision, not a new judge."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext
from workflows.agent_team.convergence import ESCALATE_STOP, current_convergence


class ConvergenceObserver(BaseObserver):
    critical = True

    def __init__(self) -> None:
        self._announced_revision = -1
        self._main_receipts = []
        self._loop_id = uuid4().hex

    async def on_loop_start(self, config):
        self._loop_id = uuid4().hex

    async def on_tool_result(self, ctx: TurnContext, result):
        state = current_convergence()
        if state is None or state.progressive is None:
            return None
        from workflows.agent_team.progressive import source_failure_kind

        failure = result.error_kind or source_failure_kind(result.name, result.result)
        receipt = state.observe_main_tool(
            tool_call_id=result.tool_call_id, tool_name=result.name, arguments=result.args,
            result=result.result, is_error=result.is_error or result.interrupted or bool(failure),
            error_kind=failure,
        )
        if receipt is None:
            return None
        self._main_receipts.append({"receipt_id": receipt["receipt_id"],
                                    "status": receipt["status"], "source": receipt["source"]})
        # Do not append text to the tool body: web_search and downstream
        # evidence observers consume its original JSON protocol. Receipt ids
        # are delivered in a separate ordinary context message at turn end.
        return replace(result, host_metadata={**result.host_metadata,
                                              "research_receipt_id": receipt["receipt_id"]})

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        state = current_convergence()
        if state is None:
            return None
        if state.phase == "finalizing" and state.answer:
            from plugins.tools.task_board import snapshot_tasks, update_task

            before = snapshot_tasks(ctx.task_id)
            unused = [task for task in before
                      if task.get("resolution") not in {"resolved", "cancelled"}]
            if unused:
                # Cancellation withdraws further work; it does not turn a
                # missing check into verified evidence. Preserve that fact and
                # the original task descriptions in the convergence artifact.
                await update_task.func(updates=[
                    {"id": task["id"], "resolution": "cancelled"} for task in unused
                ])
                state.note(
                    "remaining_tasks_retired", original_tasks=unused,
                    reason="coordinator_declared_evidence_sufficient_not_independently_verified",
                )
            # The review tool already contains the final answer. Stop here;
            # no extra model synthesis, fan-in wait or research turn is needed.
            ctx.metadata["final_answer"] = state.answer
            ctx.metadata["final_answer_source"] = "convergence_v1"
            ctx.metadata["convergence"] = state.snapshot()
            return Intervention(stop_reason="final_answer")
        if state.escalation_pending:
            return Intervention(stop_reason=ESCALATE_STOP)
        if state.light_hard_reached and state.request_escalation("light_call_budget"):
            return Intervention(stop_reason=ESCALATE_STOP)
        if state.phase == "review_required":
            snapshot = state.snapshot()
            revision = len(snapshot.get("receipts", []))
            if revision != self._announced_revision:
                self._announced_revision = revision
                self._main_receipts.clear()
                return Intervention(inject_messages=[state.render_review_message()])
        if state.progressive is not None and state.phase == "research":
            import json

            messages = []
            if self._main_receipts:
                messages.append(
                    "Recorded main research receipts (original tool bodies remain unchanged; "
                    "source validity does not establish semantic support):\n"
                    + json.dumps(self._main_receipts, ensure_ascii=False, default=str)
                )
                self._main_receipts.clear()
            notice = state.progressive.on_main_turn(
                ctx.task_id, ctx.turn,
                effective=bool(ctx.tool_calls or ctx.ai_text) and ctx.finish_reason not in {"length", "error"},
                loop_id=self._loop_id,
            )
            if notice:
                state.note("progressive_strategy_checkpoint", main_turns=state.progressive.main_turns,
                           cycles=state.progressive.cycles_since_progress)
                messages.append(notice)
            for budget_notice in (state.light_budget_notice(), state.search_budget_notice()):
                if budget_notice:
                    messages.append(budget_notice)
            if state.progressive.turn_budget_spent() and state.request_escalation("light_turn_budget"):
                return Intervention(stop_reason=ESCALATE_STOP)
            if messages:
                return Intervention(inject_messages=messages)
        return None


__all__ = ["ConvergenceObserver"]
