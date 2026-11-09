"""Bound the native task cleanup without claiming remote I/O has settled."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any

logger = logging.getLogger(__name__)
_TERMINAL = {"completed", "failed", "aborted"}


@dataclass
class CleanupOutcome:
    cleaned_count: int = 0
    cleanup_confirmed: bool = False
    pending_job_ids: list[str] = field(default_factory=list)
    evidence_cards: list[Any] = field(default_factory=list)
    assertions: list[Any] = field(default_factory=list)
    reports: list[tuple[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class _CleanupRun:
    bus: Any
    task_id: str
    jobs: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    cleanup: asyncio.Task[Any] | None = None
    evidence: list[Any] = field(default_factory=list)
    assertions: list[Any] = field(default_factory=list)
    reports: list[tuple[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cleaned: int = 0


# Keep cancelled operations alive and retrieve late exceptions. A repeated
# finally cleanup joins the same run instead of racing another cleanup_task,
# whose first action would discard the native metadata aggregate.
_active: dict[tuple[int, str], _CleanupRun] = {}


def _extend_unique(target: list[Any], values: Any) -> None:
    for value in values or ():
        if value not in target:
            target.append(value)


def _capture_metadata(run: _CleanupRun) -> None:
    metadata = run.bus.drain_task_metadata(run.task_id)
    _extend_unique(run.evidence, metadata.get("evidence_cards"))
    _extend_unique(run.assertions, metadata.get("assertions"))


async def _capture_ready_reports(run: _CleanupRun) -> None:
    while True:
        item = await run.bus.wait_any_session(run.task_id, timeout=0)
        if item is None:
            break
        name, report = item
        if not any(name == prior_name and report is prior for prior_name, prior in run.reports):
            run.reports.append(item)
        metadata = getattr(report, "metadata", None) or {}
        _extend_unique(run.evidence, metadata.get("evidence_cards"))
        _extend_unique(run.assertions, metadata.get("assertions"))


def _settled(run: _CleanupRun, task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        error = task.exception()
        if error is not None:
            category = type(error).__name__
            run.errors.append(category)
            logger.warning("Convergence cleanup operation failed (%s)", category)
        elif task is run.cleanup:
            run.cleaned = int(task.result() or 0)
    if run.cleanup is not None and run.cleanup.done() and all(t.done() for t in run.jobs.values()):
        _active.pop((id(run.bus), run.task_id), None)


def _start_abort(run: _CleanupRun, job_id: str) -> None:
    if job_id in run.jobs or run.bus.get_job_status(job_id) in _TERMINAL:
        return
    operation = asyncio.create_task(run.bus.abort(job_id))
    run.jobs[job_id] = operation
    operation.add_done_callback(lambda task, owner=run: _settled(owner, task))


def _request_remaining_aborts(run: _CleanupRun) -> None:
    """Also reach siblings if preparation stalled before normal fan-out."""
    sessions = list(run.bus.list_sessions_for_task(run.task_id))
    for session in sessions:
        for pending in list(session.pending_tasks):
            _start_abort(run, pending.job_id)
    for session in sessions:
        if session.current_job_id is not None:
            _start_abort(run, session.current_job_id)


async def _perform_cleanup(run: _CleanupRun, cancel_timeout_s: float) -> int:
    _capture_metadata(run)
    await _capture_ready_reports(run)
    # Public session objects expose pending tasks. Purge every queue before
    # any live cancellation can run its finally and dispatch a successor.
    sessions = list(run.bus.list_sessions_for_task(run.task_id))
    for session in sessions:
        for pending in list(session.pending_tasks):
            try:
                await run.bus.abort(pending.job_id)
            except Exception as error:
                run.errors.append(type(error).__name__)
    # Issue every cancellation concurrently. A resistant first worker must
    # not prevent a sibling from receiving its own cancellation request.
    for session in sessions:
        if session.current_job_id is not None:
            _start_abort(run, session.current_job_id)
    cleaned = await run.bus.cleanup_task(run.task_id, cancel_timeout_s=cancel_timeout_s)
    if run.jobs:
        await asyncio.gather(*run.jobs.values(), return_exceptions=True)
    # Native cleanup leaves late rescue in the bus aggregate. The active
    # caller drains it below; if this operation outlives its caller, a later
    # finally call can still retrieve it after our task reference is retired.
    return int(cleaned or 0)


async def cleanup_converged_task(
    bus: Any,
    bus_task_id: str,
    *,
    timeout_s: float = 45.0,
    cancel_timeout_s: float = 10.0,
) -> CleanupOutcome:
    """Cancel this run's queued/live jobs and retain already available evidence.

    The two defaults are the existing workflow/native cleanup bounds. Unlike
    wait_for, asyncio.wait does not await a task's cancellation handler after
    its deadline. Timed-out operations remain referenced until they settle.
    ``cleanup_confirmed`` only describes native task/session cleanup: an
    already-sent provider request, thread or remote process may still run.
    Callers must latch finalizing and guard new admissions before calling.
    """
    key = (id(bus), bus_task_id)
    run = _active.get(key)
    if run is None:
        run = _CleanupRun(bus, bus_task_id)
        _active[key] = run
        run.cleanup = asyncio.create_task(_perform_cleanup(run, cancel_timeout_s))
        run.cleanup.add_done_callback(lambda task, owner=run: _settled(owner, task))
    try:
        _, pending = await asyncio.wait({run.cleanup}, timeout=max(float(timeout_s), 0.0))
    except asyncio.CancelledError:
        old_operations = [*run.jobs.values(), run.cleanup]
        _request_remaining_aborts(run)
        for task in old_operations:
            if task is not None and not task.done():
                task.cancel()
        raise
    if pending:
        old_operations = [*run.jobs.values(), run.cleanup]
        _request_remaining_aborts(run)
        for task in old_operations:
            if task is not None and not task.done():
                task.cancel()
    pending_jobs = [job_id for job_id, task in run.jobs.items()
                    if not task.done() and bus.get_job_status(job_id) not in _TERMINAL]
    if pending:
        logger.warning(
            "Convergence cleanup deadline reached: %d job(s) pending; cleanup unconfirmed",
            len(pending_jobs),
        )
    _capture_metadata(run)
    # Callback scheduling is not relied upon for a completed cleanup's count.
    cleanup_done = run.cleanup is not None and run.cleanup.done() and not run.cleanup.cancelled()
    cleanup_ok = cleanup_done and run.cleanup.exception() is None
    if cleanup_ok:
        run.cleaned = int(run.cleanup.result() or 0)
    elif cleanup_done:
        # A failed metadata/event hook must not prevent best-effort requests
        # to stop the other workers. These remain observed, never awaited here.
        _request_remaining_aborts(run)
        pending_jobs = [job_id for job_id, task in run.jobs.items()
                        if not task.done() and bus.get_job_status(job_id) not in _TERMINAL]
    return CleanupOutcome(
        cleaned_count=run.cleaned,
        cleanup_confirmed=bool(not pending and cleanup_ok and not run.errors),
        pending_job_ids=pending_jobs,
        evidence_cards=list(run.evidence), assertions=list(run.assertions),
        reports=list(run.reports), errors=list(run.errors),
    )
