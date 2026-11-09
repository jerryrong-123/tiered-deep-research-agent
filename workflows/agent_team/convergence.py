"""Opt-in run-local research review; native loops still own model execution.

The root owns ``convergence_run``. Native report consumers call
``observe_reports`` with their actual collected ``(name, SubAgentResult)``
pairs. Tools and every model client (including auxiliary clients/compactors)
must use the wrappers below. A review is an ordinary coordinator tool call;
it never invokes a model. Native cancellation/queue cleanup belongs to the
root integration, after the finalizing latch and before answer delivery.
An explicit continuation may consume another ordinary coordinator turn;
this control contract makes no fixed cost or semantic correctness guarantee.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from frontier_agent.components.agent_bus import SubAgentResult
from frontier_agent.components.agent_bus.fan_in import classify_completion
from frontier_agent.core.tool import Tool
from plugins.tools._coerce import coerce_json_list

_current: ContextVar[ConvergenceState | None] = ContextVar(
    "agent_team_convergence", default=None,
)
_final_scope: ContextVar[ConvergenceState | None] = ContextVar("convergence_final_scope", default=None)
_final_request: ContextVar[tuple[ConvergenceState, object] | None] = ContextVar(
    "convergence_final_request", default=None,
)

# These existing tools only read already-local material or close/report work.
# Unknown tools are not assumed local merely because their name says "read".
_LOCAL_TOOLS = frozenset({
    "read_file", "grep_search", "glob_search", "recover_result", "stop_subagent",
    "submit_report", "add_task", "update_task", "finish_planning",
})
_PRIVATE_KEYS = frozenset({
    "reasoning", "reasoning_content", "thinking", "private", "private_reasoning",
    "chain_of_thought", "credentials", "credential", "authorization", "password",
    "api_key", "apikey", "secret", "access_token", "refresh_token",
})
ESCALATE_STOP = "escalate_to_team"
HANDOFF_RECEIPT_CHARS = 800
HANDOFF_TOTAL_CHARS = 12000
_GENERIC_TEAM_GAP = "Resolve every original-question condition the saved evidence does not yet support."


def _public_metadata(value: Any) -> Any:
    """Keep report provenance, excluding private channels and credential fields."""
    if isinstance(value, dict):
        return {key: _public_metadata(item) for key, item in value.items()
                if not str(key).startswith("_") and str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_public_metadata(item) for item in value]
    return deepcopy(value)


def _strings(value: Any) -> list[str] | None:
    value = coerce_json_list(value)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value if item.strip()]


def _tool_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("function", {}).get("name", value.get("name", "")))
    return str(getattr(value, "name", ""))


class ConvergenceBlocked(RuntimeError):
    """Safe control-flow reason; contains no provider response or exception text."""


class ConvergenceState:
    """One run's exact report receipts and explicit coordinator decisions."""

    def __init__(self, *, bus_task_id: str, question: str,
                 progressive_research_v1: bool = False,
                 progressive_config: dict[str, Any] | None = None) -> None:
        self.bus_task_id = bus_task_id
        self.question = question
        self.phase = "research"
        self.answer = ""
        self.approved_gaps: list[str] = []
        self.blocking_evidence: list[str] = []
        self._run_id = uuid4().hex
        self._receipts: list[dict[str, Any]] = []
        self._identities: dict[str, str] = {}
        self._pending: set[str] = set()
        self._decisions: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._continuation_waiting_for_turn = False
        self.escalation: dict[str, Any] | None = None
        self.light_calls = 0
        self.archive_reads: Any = None
        self._light_notice_sent = False
        self._search_notice_sent = False
        self.settlement: dict[str, Any] | None = None
        self.checking_answer = False
        from workflows.agent_team.progressive import ProgressivePolicy

        self.progressive = ProgressivePolicy(progressive_config) if progressive_research_v1 is True else None

    def note(self, event: str, **safe_values: Any) -> None:
        """Record caller-supplied public diagnostics; never pass private channels."""
        self._events.append({"utc": datetime.now(timezone.utc).isoformat(),
                             "event": event, "phase": self.phase, **deepcopy(safe_values)})

    @property
    def tier_active(self) -> bool:
        return self.progressive is not None and self.progressive.tier is not None

    @property
    def escalation_pending(self) -> bool:
        return self.escalation is not None and self.progressive is not None

    def count_light_calls(self, count: int = 1) -> None:
        """Main, helper and page-extraction work, which turns alone do not bound."""
        if self.tier_active:
            self.light_calls += count

    @property
    def light_soft_reached(self) -> bool:
        return self.tier_active and self.light_calls >= self.progressive.tier["light_soft_calls"]

    @property
    def light_hard_reached(self) -> bool:
        return self.tier_active and self.light_calls >= self.progressive.tier["light_hard_calls"]

    def light_budget_notice(self) -> str:
        """Warn once, while there is still budget left to finish properly."""
        if not self.light_soft_reached or self._light_notice_sent:
            return ""
        self._light_notice_sent = True
        self.note("light_budget_notice", light_calls=self.light_calls)
        return ("The light path's budget is nearly used, so no new helper can start. "
                "Finalize now if the evidence supports the answer; otherwise escalate.")

    @property
    def search_budget_spent(self) -> bool:
        if not self.tier_active:
            return False
        limit = self.progressive.tier["max_search_queries"]
        return limit > 0 and self.progressive.search_queries >= limit

    def search_budget_notice(self) -> str:
        """Tell the coordinator once; helpers meet the refusal directly."""
        if not self.search_budget_spent or self._search_notice_sent:
            return ""
        self._search_notice_sent = True
        self.note("search_budget_notice", search_queries=self.progressive.search_queries)
        return ("The search budget is used, so further searches are refused. Read the "
                "known URLs that matter, then finalize.")

    @property
    def synthesizes_at_stop(self) -> bool:
        return self.tier_active and self.progressive.tier["synthesize_at_stop"] is True

    @property
    def settles_answer(self) -> bool:
        """Whether the closing step after cleanup applies to this run."""
        if not self.tier_active:
            return False
        tier = self.progressive.tier
        return tier["synthesize_at_stop"] is True or tier["independent_check"] is True

    def request_escalation(self, reason: str, gaps: Iterable[str] = (), detail: str = "") -> bool:
        """Latch the handover. The light loop stops at this turn's end."""
        if not self.tier_active or self.escalation is not None or self.phase in {"finalizing", "finished"}:
            return False
        self.escalation = {"reason": reason, "detail": detail, "gaps": list(gaps),
                           "light_main_turns": self.progressive.main_turns,
                           "light_calls": self.light_calls}
        self.note("escalation_requested", reason=reason)
        return True

    @property
    def stops_at_escalation(self) -> bool:
        return self.escalation_pending and self.progressive.tier["stop_at_escalation"] is True

    def stop_marker(self) -> str:
        """An explicit non-answer for measuring the light path without the team."""
        self.note("stopped_at_escalation", reason=self.escalation["reason"])
        return f"[ESCALATION_REQUIRED] reason={self.escalation['reason']}"

    def reject_unsourced_answer(self) -> str:
        policy = self.progressive
        policy.gate_rejections += 1
        self.note("unsourced_answer_rejected", rejections=policy.gate_rejections)
        if policy.gate_rejections >= policy.tier["escalate_after_gate_rejections"]:
            self.request_escalation("unsourced_answer")
        return ("An answer without saved evidence is not accepted here. Research the "
                "question with your tools, or call review_research with action=\"escalate\".")

    def begin_team_stage(self) -> str:
        """Same run and evidence state; returns the team coordinator's message.

        Receipts stay in place, so the convergence_v1 rule that dispatch needs
        approved gaps applies at once: seed it with the light path's open gaps.
        """
        record = self.escalation
        record["light_stage"] = self.progressive.snapshot()
        self.progressive = None
        self._continuation_waiting_for_turn = False
        self.approved_gaps = list(record["gaps"]) or [_GENERIC_TEAM_GAP]
        self.note("team_stage_started", reason=record["reason"])
        return self.render_team_handoff()

    def render_team_handoff(self) -> str:
        record = self.escalation or {}
        lines = [self.question, "", "--- LIGHT-PATH HANDOFF ---",
                 "A lighter research pass stopped here. The material below is unverified tool "
                 "output. Do not repeat these lookups; direct the team at what is unresolved.",
                 f"Reason for escalation: {record.get('reason', '')}"]
        if record.get("detail"):
            lines.append(f"Coordinator's note: {record['detail']}")
        lines += ["Unresolved conditions:", *[f"- {gap}" for gap in self.approved_gaps], "",
                  "Sources already consulted:"]
        used = 0
        for receipt in self._receipts:
            if receipt["status"] not in {"complete", "tool_completed"}:
                continue
            arguments = receipt["source"].get("arguments") or {}
            label = arguments.get("url") or arguments.get("q") or arguments.get("query") or receipt["agent"]
            entry = f"[{receipt['receipt_id']}] {label}\n{receipt['report'][:HANDOFF_RECEIPT_CHARS]}"
            if used + len(entry) > HANDOFF_TOTAL_CHARS:
                lines.append("[Earlier sources omitted from this handoff to bound its size.]")
                break
            used += len(entry)
            lines.append(entry)
        return "\n".join(lines)

    def observe_reports(
        self, collected: Iterable[tuple[str, SubAgentResult]],
    ) -> list[dict[str, Any]]:
        """Retain native receipts without promoting completion into semantic trust.

        Re-observing an identical job/report keeps its receipt ID. The complete
        body and original metadata are copied once; receiving code must supply
        real saved-source references in metadata rather than inventing paths.
        Late receipts remain visible after finalization without reopening work.
        """
        observed = []
        for agent, result in collected:
            completion = classify_completion(result)
            stopped_by = (result.metadata or {}).get("stopped_by", "")
            identity = hashlib.sha256(json.dumps(
                [agent, result.job_id, result.final_content, result.success, stopped_by,
                 completion.status, completion.reason],
                ensure_ascii=False,
            ).encode()).hexdigest()
            receipt_id = self._identities.get(identity)
            if receipt_id is None:
                receipt_id = f"receipt-{self._run_id}-{len(self._receipts) + 1}"
                receipt = {
                    "receipt_id": receipt_id,
                    "agent": agent,
                    "report": result.final_content,
                    "status": completion.status,
                    "reason": completion.reason,
                    "source": {
                        "job_id": result.job_id,
                        "report_sha256": hashlib.sha256(result.final_content.encode()).hexdigest(),
                        "role_id": result.role_id,
                        "stopped_by": stopped_by,
                        "success": result.success,
                        "metadata": _public_metadata(result.metadata),
                    },
                }
                self._identities[identity] = receipt_id
                self._receipts.append(receipt)
                self._pending.add(receipt_id)
                self.note("report_received", receipt_id=receipt_id, status=completion.status)
                if self.phase == "research":
                    self.phase = "review_required"
            observed.append(next(r for r in self._receipts if r["receipt_id"] == receipt_id))
        return deepcopy(observed)

    def observe_main_tool(
        self, *, tool_call_id: str, tool_name: str, arguments: dict[str, Any],
        result: str, is_error: bool = False, error_kind: str = "",
    ) -> dict[str, Any] | None:
        """Record a real public tool result; do not fabricate a child job/report.

        Main sources are record-only. Receiving a new child report still creates
        the existing review boundary, but reading one webpage does not force an
        extra review round. The returned receipt id is attached to the delivered
        tool result by the main observer so the coordinator can cite it.
        """
        if self.progressive is None:
            return None
        from workflows.agent_team.progressive import SOURCE_TOOLS

        if tool_name not in SOURCE_TOOLS or not tool_call_id:
            return None
        self.progressive.observe_tool(tool_name, tool_call_id, is_error=is_error)
        public_args = _public_metadata(arguments)
        identity = hashlib.sha256(json.dumps(
            ["main_tool", tool_call_id, tool_name, public_args, result, is_error],
            ensure_ascii=False, default=str,
        ).encode()).hexdigest()
        known_id = self._identities.get(identity)
        if known_id is not None:
            return deepcopy(next(r for r in self._receipts if r["receipt_id"] == known_id))
        receipt_id = f"receipt-{self._run_id}-{len(self._receipts) + 1}"
        receipt = {
            "receipt_id": receipt_id, "agent": "main", "report": result,
            "status": "tool_error" if is_error else "tool_completed",
            "reason": error_kind if is_error else "public_tool_result_not_semantic_approval",
            "source": {"source_kind": "main_tool", "tool_call_id": tool_call_id,
                       "tool_name": tool_name, "arguments": public_args,
                       "report_sha256": hashlib.sha256(result.encode()).hexdigest(),
                       "metadata": {}},
        }
        # Native/container helpers share this workspace. Store the exact public
        # result once so a dispatched gap can read it without another fetch or
        # another model summary. No workspace is assumed for other runtimes.
        from pathlib import Path

        from frontier_agent.core.execution_context import get_current_execution_scope

        scope = get_current_execution_scope()
        workspace = (scope.metadata or {}).get("workspace_root") if scope else None
        if workspace and Path(workspace).is_dir():
            relative = Path(".research_receipts") / self._run_id / (receipt_id + ".json")
            target = Path(workspace) / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, ensure_ascii=False, indent=2)
                receipt["source"]["metadata"].update(
                    source_path=str(target), source_relative_path=relative.as_posix(),
                )
            except OSError as exc:
                self.note("main_source_store_failed", receipt_id=receipt_id, error_type=type(exc).__name__)
        self._receipts.append(receipt)
        self._identities[identity] = receipt_id
        self.note("main_tool_recorded", receipt_id=receipt_id, tool_name=tool_name, is_error=is_error)
        return deepcopy(receipt)

    def review_tool(self, *, before_decision: Any = None) -> Tool:
        """Create the coordinator's ordinary decision tool, without a model call.

        ``before_decision`` may asynchronously drain *already ready* native bus
        reports and observe them here. It must not wait for running work. IDs
        are checked after this drain, so unseen reports cannot be finalized.
        """
        async def review(
            action: str, receipt_ids: list[str], gaps: list[str] | None = None,
            final_answer: str = "", blocking_evidence: list[str] | None = None,
            disposition: str = "",
            gap_tasks: list[dict[str, Any]] | None = None,
            progress_summary: str = "", progress_receipt_ids: list[str] | None = None,
            requirements: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            if before_decision is not None:
                await before_decision()
            if self.phase in {"finalizing", "finished"}:
                return self._rejected("review_closed")
            cited, specific_gaps = _strings(receipt_ids), _strings(gaps or [])
            blocking = _strings(blocking_evidence or [])
            if cited is None or specific_gaps is None or blocking is None:
                return self._rejected("invalid_review_arguments")
            parsed_gaps = None
            if self.progressive is not None:
                try:
                    parsed_gaps = self.progressive.validate_gaps(
                        gap_tasks or [], {r["receipt_id"]: r for r in self._receipts},
                    )
                except ValueError as exc:
                    return self._rejected(str(exc))
                if parsed_gaps:
                    # A single current gap representation drives both display
                    # and dispatch. Do not require duplicate model labels.
                    specific_gaps = [g["question"] for g in parsed_gaps.values()]
            known = {receipt["receipt_id"] for receipt in self._receipts}
            if not set(cited) <= known:
                return self._rejected("unknown_receipt_ids")
            if action == "escalate" and self.tier_active:
                if not isinstance(disposition, str) or not disposition.strip():
                    return self._rejected("escalation_reason_required")
                self._decisions.append({
                    "action": action, "receipt_ids": list(dict.fromkeys(cited)),
                    "gaps": specific_gaps, "blocking_evidence": blocking,
                    "disposition": disposition, "previous_gaps": list(self.approved_gaps),
                    "final_answer": "",
                })
                self.request_escalation("coordinator_requested", specific_gaps, disposition)
                self.note("review_decision", action=action, receipt_ids=cited)
                return {"ok": True, "phase": self.phase, "escalated": True}
            if not self._pending <= set(cited):
                return self._rejected("pending_receipts_not_reviewed")
            if action not in {"continue", "finalize"}:
                return self._rejected("invalid_review_action")
            # Without this a run with no gap yet has no way to report progress.
            progress_only = (self.tier_active and action == "continue" and self.phase == "research"
                             and not specific_gaps and bool(progress_receipt_ids))
            if action == "continue" and not specific_gaps and not progress_only:
                return self._rejected("specific_gaps_required")
            if action == "finalize" and specific_gaps:
                return self._rejected("finalize_has_current_evidence_gaps")
            if action == "finalize" and not cited:
                return self._rejected("received_report_required")
            if action == "finalize" and self.progressive is not None:
                if not any(r["receipt_id"] in cited and r["status"] in {"complete", "tool_completed"}
                           for r in self._receipts):
                    return self._rejected("successful_evidence_receipt_required")
            if action == "finalize" and (not isinstance(final_answer, str) or not final_answer.strip()):
                return self._rejected("final_answer_required")
            if not isinstance(disposition, str):
                return self._rejected("invalid_review_arguments")
            if action == "finalize" and (self.approved_gaps or self.blocking_evidence or blocking) and not disposition.strip():
                return self._rejected("previous_gap_or_counterevidence_disposition_required")
            ledger, corrections, unverified = None, [], []
            if action == "finalize" and self.tier_active:
                ledger = coerce_json_list(requirements or [])
                verdict = self.progressive.check_requirements(
                    self.question, ledger, {r["receipt_id"]: r for r in self._receipts},
                )
                if not verdict["ok"]:
                    policy = self.progressive
                    policy.gate_rejections += 1
                    self.note("requirement_gate_rejected", rejections=policy.gate_rejections,
                              coverage_pct=verdict["coverage_pct"], problems=verdict["row_problems"],
                              candidate_answer=final_answer)
                    if policy.gate_rejections >= policy.tier["escalate_after_gate_rejections"]:
                        self.request_escalation("requirement_gate_exhausted",
                                                verdict["uncovered_question_parts"])
                    return {**self._rejected("requirements_not_supported"), **verdict,
                            "escalated": self.escalation_pending}
                corrections = verdict["receipt_corrections"]
                unverified = verdict["uncovered_question_parts"]
            if self.progressive is not None:
                ids = _strings(progress_receipt_ids or [])
                if ids is None or not isinstance(progress_summary, str):
                    return self._rejected("invalid_progress_arguments")
                try:
                    self.progressive.acknowledge_progress(
                        progress_summary, ids, {r["receipt_id"]: r for r in self._receipts},
                    )
                except ValueError as exc:
                    return self._rejected(str(exc))
                if progress_only:
                    self.note("progress_acknowledged", receipt_ids=ids)
                    return {"ok": True, "phase": self.phase, "approved_gaps": list(self.approved_gaps),
                            "blocking_evidence": list(self.blocking_evidence)}
                if action == "continue":
                    self.progressive.accept_gaps(parsed_gaps or {})
            # This is a structural decision record, not a semantic judge.
            # Empty optional lists never erase earlier counterevidence/gaps.
            self._decisions.append({
                "action": action, "receipt_ids": list(dict.fromkeys(cited)),
                "gaps": specific_gaps, "blocking_evidence": blocking,
                "disposition": disposition, "previous_gaps": list(self.approved_gaps),
                "final_answer": final_answer if action == "finalize" else "",
                **({"requirements": ledger, "receipt_corrections": corrections,
                    "unverified_question_parts": unverified}
                   if ledger is not None else {}),
            })
            for item in blocking:
                if item not in self.blocking_evidence:
                    self.blocking_evidence.append(item)
            self._pending.difference_update(cited)
            if action == "continue":
                self.approved_gaps = specific_gaps
                self.phase = "research"
                # Parallel tools from this same model response cannot race the
                # review. The next ordinary coordinator request opens research.
                self._continuation_waiting_for_turn = True
            else:
                self.answer = final_answer
                self.request_finalization("review_finalized")
            self.note("review_decision", action=action, receipt_ids=cited)
            return {"ok": True, "phase": self.phase, "approved_gaps": list(self.approved_gaps),
                    "blocking_evidence": list(self.blocking_evidence)}

        review_tool = Tool(
            name="review_research",
            description=(
                "Review received research reports before further research. Cite every pending receipt ID. "
                "Continue only with concrete unresolved evidence gaps; or finalize with the complete final "
                "answer now. Record any blocking evidence and explain its disposition. Report completion "
                "Finalize must have no current gaps; explain the disposition of earlier gaps or counterevidence. "
                "does not establish semantic correctness. Finalize latches the answer without another model call."
            ),
            parameters={
                "type": "object", "properties": {
                    "action": {"type": "string", "enum": ["continue", "finalize"]},
                    "receipt_ids": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                    "final_answer": {"type": "string"},
                    "blocking_evidence": {"type": "array", "items": {"type": "string"}},
                    "disposition": {"type": "string"},
                }, "required": ["action", "receipt_ids"], "additionalProperties": False,
            },
            func=review,
        )
        if self.progressive is not None:
            from workflows.agent_team.progressive import gap_schema

            review_tool.description += (
                " Progressive research: main tool receipts are valid evidence receipts; no child report "
                "or final_verifier is required. Before delegation declare gap_tasks tied to actual original "
                "requirements and current source receipts. Only these gap_ids can be assigned. "
                "progress_summary/progress_receipt_ids acknowledge relevant new findings, not mere searches."
            )
            review_tool.parameters["properties"].update({
                "gap_tasks": {"type": "array", "items": gap_schema()},
                "progress_summary": {"type": "string"},
                "progress_receipt_ids": {"type": "array", "items": {"type": "string"}},
            })
        if self.tier_active:
            from workflows.agent_team.progressive import requirement_schema

            review_tool.description += (
                " Tiered research: finalize needs `requirements` rows covering the original question. "
                "action=\"escalate\", with the reason in `disposition`, hands the run to the full team."
            )
            review_tool.parameters["properties"]["action"]["enum"].append("escalate")
            review_tool.parameters["properties"]["requirements"] = {
                "type": "array", "items": requirement_schema(),
            }
        return review_tool

    def _rejected(self, reason: str) -> dict[str, Any]:
        return {"ok": False, "phase": self.phase, "reason": reason,
                "pending_receipts": deepcopy([receipt for receipt in self._receipts
                                              if receipt["receipt_id"] in self._pending])}

    def request_finalization(self, reason: str) -> None:
        """Freeze research; root performs native cleanup and existing final delivery."""
        if self.phase not in {"finalizing", "finished"}:
            self.phase = "finalizing"
            self.note("finalization_requested", reason=reason)

    @contextmanager
    def finalization_scope(self) -> Iterator[None]:
        """Allow the coordinator's existing final response after native cleanup.

        This scope alone grants no budget privilege: only a wrapped coordinator
        request creates the logical request marker consumed by the budget layer.
        """
        token = _final_scope.set(self)
        try:
            yield
        finally:
            _final_scope.reset(token)

    @contextmanager
    def answer_check_scope(self) -> Iterator[None]:
        """Admit the independent check of a latched answer as the final request."""
        self.checking_answer = True
        token = _final_request.set((self, object()))
        try:
            yield
        finally:
            _final_request.reset(token)
            self.checking_answer = False

    def finalization_request_id(self) -> object | None:
        """Opaque ID stable throughout one coordinator request and its retries."""
        marker = _final_request.get()
        if self.phase == "finalizing" and marker is not None and marker[0] is self:
            return marker[1]
        return None

    def is_finalization_request(self) -> bool:
        return self.finalization_request_id() is not None

    def render_review_message(self) -> str:
        """Local receipt navigation for the next existing coordinator turn."""
        return (
            "Research reports were received. Before new research or delegation, "
            "call review_research in this turn. Cite every pending receipt ID; "
            "continue only for specific evidence gaps, or finalize with an answer. "
            "Compare the evidence with every original question constraint and record the disposition "
            "of unresolved gaps and counterevidence. Use the original received reports or their local "
            "saved sources; completion and source identity do not establish semantic correctness.\n"
            + json.dumps({
                "phase": self.phase,
                "pending_receipts": [{
                    "receipt_id": receipt["receipt_id"], "agent": receipt["agent"],
                    "status": receipt["status"], "reason": receipt["reason"],
                    "job_id": receipt["source"].get("job_id", ""),
                    **({"tool_call_id": receipt["source"]["tool_call_id"]}
                       if "tool_call_id" in receipt["source"] else {}),
                    "report_sha256": receipt["source"]["report_sha256"],
                    "saved_sources": {key: value for key, value in
                                      (receipt["source"]["metadata"] or {}).items()
                                      if key in {"report_path", "report_file", "trajectory_path", "source_path"}},
                } for receipt in self._receipts if receipt["receipt_id"] in self._pending],
                "approved_gaps": self.approved_gaps,
                "blocking_evidence": self.blocking_evidence,
                **({"recorded_main_receipts": [
                    {"receipt_id": r["receipt_id"], "source": r["source"], "status": r["status"]}
                    for r in self._receipts if r["source"].get("source_kind") == "main_tool"
                ], "progressive": self.progressive.snapshot()} if self.progressive is not None else {}),
            }, ensure_ascii=False, default=str)
        )

    def snapshot(self) -> dict[str, Any]:
        """Copy public state for the root's artifact writer and local observer."""
        return deepcopy({
            "protocol_version": "agent_team_convergence_v1",
            "bus_task_id": self.bus_task_id,
            "question": self.question,
            "question_sha256": hashlib.sha256(self.question.encode()).hexdigest(),
            "phase": self.phase,
            "answer": self.answer,
            "receipts": self._receipts,
            "pending_receipt_ids": sorted(self._pending),
            "approved_gaps": self.approved_gaps,
            "blocking_evidence": self.blocking_evidence,
            "continuation_waiting_for_turn": self._continuation_waiting_for_turn,
            "decisions": self._decisions,
            "events": self._events,
            **({"progressive": self.progressive.snapshot()} if self.progressive is not None else {}),
            **({"escalation": self.escalation} if self.escalation is not None else {}),
            **({"light_calls": self.light_calls} if self.light_calls else {}),
            **({"archive_reads": self.archive_reads.used} if self.archive_reads and self.archive_reads.used else {}),
            **({"settlement": self.settlement} if self.settlement is not None else {}),
        })

    def mark_finished(self) -> None:
        """Close admission even in children retaining this run after cancellation."""
        if self.phase != "finished":
            self.phase = "finished"
            self.note("finished")


def current_convergence() -> ConvergenceState | None:
    return _current.get()


def _allowed_tool(state: ConvergenceState, name: str, *, coordinator: bool) -> bool:
    if state.phase == "finished":
        return False
    if state.phase == "finalizing":
        return name in _LOCAL_TOOLS
    if coordinator and state.escalation_pending:
        return name in _LOCAL_TOOLS
    if state.progressive is not None:
        if not coordinator and name in {"create_subagent", "assign_task", "delegate_subtask"}:
            return False
        if name == "finalize_answer":
            return False  # One canonical review/commit boundary in this mode.
    if coordinator and (state.phase == "review_required" or state._continuation_waiting_for_turn):
        return name in _LOCAL_TOOLS or name == "review_research"
    return name != "review_research" or coordinator


def guard_tools(tools: list[Tool], *, coordinator: bool) -> list[Tool]:
    """Preserve native schemas and guard actual execution against run state."""
    state = current_convergence()
    if state is None:
        return tools
    guarded = []
    for original in tools:
        async def invoke(_tool: Tool = original, **kwargs: Any) -> Any:
            if not _allowed_tool(state, _tool.name, coordinator=coordinator):
                raise ConvergenceBlocked("convergence_tool_admission_closed")
            if _tool.name == "web_fetch":
                # Each page read makes its own uncounted extraction call.
                urls = coerce_json_list(kwargs.get("url"))
                state.count_light_calls(len(urls) if isinstance(urls, list) else 1)
            if _tool.name == "web_search" and state.tier_active:
                if state.search_budget_spent:
                    raise ConvergenceBlocked("search_budget_spent_read_known_urls_or_finalize")
                # One call may carry several queries; each costs one search.
                queries = coerce_json_list(kwargs.get("q"))
                state.progressive.search_queries += len(queries) if isinstance(queries, list) else 1
                if state.progressive.tier["unquote_single_words"]:
                    from workflows.agent_team.progressive import unquote_single_words

                    kwargs = {**kwargs, "q": unquote_single_words(queries)}
            if state.progressive is not None and _tool.name in {"assign_task", "create_subagent"}:
                from frontier_agent.components.agent_bus import AgentBus
                from frontier_agent.core.runtime.registries import services as registry
                from workflows.agent_team.progressive import assignment_scope

                if state.light_soft_reached:
                    raise ConvergenceBlocked("light_budget_reserved_for_finishing")
                if not state.progressive.gaps:
                    raise ConvergenceBlocked("progressive_delegation_requires_gap_tasks")
                policy = state.progressive
                bus = registry.get(AgentBus)
                # Serialize admission THROUGH native enqueue: two model tool
                # calls cannot both observe a free slot and over-admit jobs.
                async with policy.lock:
                    if state.phase != "research" or state._continuation_waiting_for_turn:
                        raise ConvergenceBlocked("convergence_tool_admission_closed")
                    kwargs = deepcopy(kwargs)
                    if _tool.name == "assign_task":
                        try:
                            tasks, bindings = policy.prepare_dispatch(
                                coerce_json_list(kwargs.get("tasks")), bus, state.bus_task_id,
                            )
                        except ValueError as exc:
                            raise ConvergenceBlocked(str(exc)) from exc
                        kwargs["tasks"] = tasks
                        for item in tasks:
                            gap = bindings[item["agent"]]["progressive_gap"]
                            sources = [r for r in state._receipts if r["receipt_id"] in gap["source_receipt_ids"]]
                            navigation = []
                            for receipt in sources:
                                metadata = receipt["source"].get("metadata") or {}
                                entry = {"receipt_id": receipt["receipt_id"], "source": receipt["source"]}
                                if not metadata.get("source_path"):
                                    # Only selected gap sources, not the entire
                                    # research history; no lossy re-summary.
                                    entry["public_result"] = receipt["report"]
                                navigation.append(entry)
                            item["prompt"] += "\n\n# Existing source material for this gap\n" + json.dumps(
                                navigation, ensure_ascii=False, default=str,
                            ) + "\nRead the saved source_relative_path/source_path before re-fetching it."
                        with assignment_scope(bindings):
                            return await _tool.func(**kwargs)
                    agents = coerce_json_list(kwargs.get("agents"))
                    if not isinstance(agents, list) or not agents:
                        raise ConvergenceBlocked("progressive_agent_specs_required")
                    idle = sum(1 for s in bus.list_sessions_for_task(state.bus_task_id)
                               if getattr(s, "total_task_count", 0) == 0)
                    if idle + len(agents) > policy.config["max_helpers_parallel"]:
                        raise ConvergenceBlocked("progressive_unused_agent_limit")
                    return await _tool.func(**kwargs)
            if state.progressive is None and _tool.name == "assign_task" and state._receipts:
                if not state.approved_gaps:
                    raise ConvergenceBlocked("convergence_approved_gaps_required")
                kwargs = deepcopy(kwargs)
                tasks = coerce_json_list(kwargs.get("tasks"))
                if isinstance(tasks, list):
                    prefix = ("Work only on these explicitly approved evidence gaps:\n"
                              + "\n".join(f"- {gap}" for gap in state.approved_gaps)
                              + "\nReport evidence resolving or contradicting these gaps.\n\n")
                    tasks = deepcopy(tasks)
                    for item in tasks:
                        if isinstance(item, dict) and isinstance(item.get("prompt"), str):
                            item["prompt"] = prefix + item["prompt"]
                    kwargs["tasks"] = tasks
            return await _tool.func(**kwargs)
        parameters = original.parameters
        if state.progressive is not None and original.name == "assign_task":
            parameters = deepcopy(parameters)
            item_schema = parameters["properties"]["tasks"]["items"]
            item_schema.setdefault("properties", {})["gap_id"] = {
                "type": "string", "description": "One current approved gap_tasks[].gap_id."
            }
            item_schema.setdefault("required", []).append("gap_id")
        guarded.append(replace(original, func=invoke, parameters=parameters))
    return guarded


class _ConvergenceLLM:
    def __init__(self, llm: Any, state: ConvergenceState, coordinator: bool) -> None:
        self._llm = llm
        self._state = state
        self._coordinator = coordinator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def bind(self, **kwargs: Any) -> Any:
        return _ConvergenceLLM(self._llm.bind(**kwargs), self._state, self._coordinator)

    def _admit(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        state = self._state
        if state.phase == "finished" or (state.phase == "finalizing" and
                (not self._coordinator or _final_scope.get() is not state)):
            raise ConvergenceBlocked("convergence_model_admission_closed")
        # The coordinator may be parked in collect_reports with no turn ending,
        # so helpers are stopped here rather than at its next turn.
        if not self._coordinator and state.light_hard_reached:
            raise ConvergenceBlocked("light_call_budget_exhausted")
        state.count_light_calls()
        if self._coordinator and state.phase == "research":
            state._continuation_waiting_for_turn = False
        if self._coordinator and state.phase == "review_required" and kwargs.get("tools") is not None:
            kwargs = dict(kwargs)
            kwargs["tools"] = [tool for tool in kwargs["tools"]
                               if _allowed_tool(state, _tool_name(tool), coordinator=True)]
        # Clear any inherited marker on every non-final call. Merely inheriting
        # a finalization scope must never make child/auxiliary work privileged.
        marker = (state, object()) if state.phase == "finalizing" else None
        return kwargs, _final_request.set(marker)

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        kwargs, token = self._admit(kwargs)
        try:
            return await self._llm.chat(*args, **kwargs)
        finally:
            _final_request.reset(token)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        # Admission happens when iteration begins, not when a lazy generator is
        # created. Cancellation/late finalization therefore closes queued calls.
        kwargs, token = self._admit(kwargs)
        try:
            async for chunk in self._llm.stream(*args, **kwargs):
                yield chunk
        finally:
            _final_request.reset(token)


def wrap_llm(llm: Any, *, coordinator: bool) -> Any:
    """Wrap each root/child/aux client once; disabled runs return exact identity."""
    state = current_convergence()
    return _ConvergenceLLM(llm, state, coordinator) if state is not None else llm


@contextmanager
def convergence_run(
    *, enabled: bool = False, bus_task_id: str = "", question: str = "",
    progressive_research_v1: bool = False, progressive_config: dict[str, Any] | None = None,
) -> Iterator[ConvergenceState | None]:
    """Own one isolated run; explicit false also masks an enclosing enabled run."""
    state = ConvergenceState(
        bus_task_id=bus_task_id, question=question,
        progressive_research_v1=progressive_research_v1, progressive_config=progressive_config,
    ) if enabled is True or progressive_research_v1 is True else None
    token = _current.set(state)
    # Same inheritance as the run itself, so helpers see the same search display.
    visible = None
    if state is not None and state.tier_active and state.progressive.tier["show_low_value_results"]:
        from plugins.tools.web_search import set_low_value_results_visible

        visible = set_low_value_results_visible(True)
    saved_pages = None
    if state is not None and state.tier_active and state.progressive.tier["archive_reads"]:
        from plugins.tools._archive_copy import ArchiveReads, open_archive_reads

        state.archive_reads = ArchiveReads(state.progressive.tier["archive_reads"], state.count_light_calls)
        saved_pages = open_archive_reads(state.archive_reads)
    try:
        yield state
    finally:
        if saved_pages is not None:
            from plugins.tools._archive_copy import close_archive_reads

            close_archive_reads(saved_pages)
        if visible is not None:
            from plugins.tools.web_search import reset_low_value_results_visible

            reset_low_value_results_visible(visible)
        if state is not None:
            state.mark_finished()
        _current.reset(token)
