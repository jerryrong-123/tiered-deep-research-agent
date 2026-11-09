"""Terminate the main-agent loop on a plain-text, no-tool turn."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from frontier_agent.core.execution_context import get_current_execution_scope
from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext
from plugins.tools._bus_scope import resolve_bus_task_id
from plugins.tools.finalize_answer import finalize_gate
from plugins.tools.task_board import unresolved_task_ids

logger = logging.getLogger(__name__)


def _build_bypass_warning(task_id: str) -> str:
    """Build the standalone warning for an answer delivered past a blocked gate.

    Built once while the task board is still live — the board is cleared
    before the workflow output is assembled — and stored in
    ``finalize_gate_warning`` for the delivery nodes to append after any
    finalization that would otherwise strip it (the reporter's References
    cleanup drops everything after that heading).
    """
    pending = unresolved_task_ids(task_id)
    if pending:
        what = f"task-board item(s) still unfinished: {', '.join(pending)}"
    else:
        what = "the finalize gate was still rejecting this submission"
    return (
        "\n\n---\n\n"
        f"> ⚠ Unfinished work at submission: {what}. This answer was "
        "delivered on the final turn despite the gate — conclusions that "
        "depend on that work are unverified."
    )


def append_bypass_warning(text: str, source: Mapping[str, Any] | None) -> str:
    """Re-attach the stored bypass warning at a delivery boundary, once.

    The observer stores the ready-made warning; the delivery nodes append
    it *after* finalization (reporter References cleanup would strip an
    earlier append). Idempotent: text already carrying the warning is
    returned unchanged.
    """
    warning = str((source or {}).get("finalize_gate_warning") or "")
    if not warning or not text or warning in text:
        return text
    return f"{text.rstrip()}{warning}"


class BareTextFinalizeObserver(BaseObserver):
    critical = True

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        if ctx.tool_calls:
            return None
        text = (ctx.ai_text or "").strip()
        if not text:
            # Nothing said and no tool — let the loop's no-tool recovery nudge.
            return None

        from workflows.agent_team.convergence import ESCALATE_STOP, current_convergence

        convergence = current_convergence()
        if (convergence is not None and convergence.phase not in {"finalizing", "finished"}
                and convergence.snapshot()["receipts"]):
            # One canonical control decision is shared by scheduling and final
            # delivery. Plain text must not bypass a review of newly arrived
            # reports, especially when one contains a substantive conflict.
            return Intervention(
                continue_to_next_turn=True,
                inject_messages=[convergence.render_review_message()],
            )
        if (convergence is not None and convergence.tier_active
                and convergence.phase not in {"finalizing", "finished"}):
            # The cheap exit needs evidence too; zero research is not an answer.
            message = convergence.reject_unsourced_answer()
            if convergence.escalation_pending:
                return Intervention(stop_reason=ESCALATE_STOP)
            return Intervention(continue_to_next_turn=True, inject_messages=[message])

        scope = get_current_execution_scope()
        err = finalize_gate(
            ctx.task_id,
            text,
            bus_task_id=resolve_bus_task_id(scope) if scope is not None else None,
        )
        # On the very last turn tools are stripped (LastTurnForcer), so the only
        # way to emit anything is plain text; accept it rather than lose the
        # answer to a max_turns stop even if the gate would otherwise block.
        last_turn = ctx.turn >= ctx.max_turns - 1
        if err and not last_turn:
            return Intervention(continue_to_next_turn=True, inject_messages=[err])

        if isinstance(ctx.metadata, dict):
            if err:
                # Last-turn bypass: the gate says BLOCK, but the answer is
                # delivered anyway rather than lost to max_turns. Keep that
                # fallback — and make it visible: store the gate message and
                # a ready-to-append warning (built while the board is live).
                # The visible text is appended by the delivery nodes AFTER
                # reporter finalization, which would strip it otherwise.
                ctx.metadata["finalize_gate_bypassed"] = err
                ctx.metadata["finalize_gate_warning"] = _build_bypass_warning(
                    ctx.task_id,
                )
            ctx.metadata["final_answer"] = text
            ctx.metadata["final_answer_confidence"] = 1.0
        logger.info(
            "BareTextFinalizeObserver latched: turn=%d, len=%d, gate=%s",
            ctx.turn, len(text), "bypassed(last_turn)" if err else "passed",
        )
        return Intervention(stop_reason="final_answer")
