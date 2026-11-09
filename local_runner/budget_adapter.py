"""Fresh-trial budget admission around the unchanged native model transport.

The protected amount is part of the existing hard cap, not added spend. Pricing
and settlement stay in the supplied immutable trial_helpers.Meter. Only an
explicit coordinator finalization marker can consume the one final admission.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal


class BudgetAdmissionError(RuntimeError):
    """Safe local admission failure; never a provider/network diagnosis."""

    category = "local_model_budget"

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def budget_error_code(error: BaseException) -> str | None:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, BudgetAdmissionError):
            return error.code
        error = error.__cause__ or error.__context__
    return None


def safe_error_with_budget(original):
    def safe_error(error):
        result = original(error)
        code = budget_error_code(error)
        if code:
            result.update(error_domain="local_model_budget", budget_reason=code)
        else:
            cursor, seen = error, set()
            while cursor is not None and id(cursor) not in seen:
                seen.add(id(cursor))
                if type(cursor).__module__ == "workflows.agent_team.convergence" and type(cursor).__name__ == "ConvergenceBlocked":
                    result.update(error_domain="local_convergence", convergence_reason="convergence_admission_closed")
                    break
                cursor = cursor.__cause__ or cursor.__context__
        return result
    return safe_error


def reservation_upper(max_tokens: int) -> Decimal:
    if type(max_tokens) is not int or not 0 < max_tokens <= 1048576:
        raise BudgetAdmissionError("invalid_model_output_bound")
    return (Decimal(1048576) * 2 + Decimal(max_tokens) * 8) / Decimal(1000000)


def existing_final_output_upper(active_cap: int) -> int:
    """Reserve for the pinned core's existing fallback, without changing it."""
    from agent_core.runtime.loop._runaway import _expanded_retry_max_tokens
    reservation_upper(active_cap)
    return _expanded_retry_max_tokens(active_cap, [], None) or active_cap


def _answer_latched(state):
    # An independent check is the one model call allowed after the answer latches.
    return bool(getattr(state, "answer", None)) and not getattr(state, "checking_answer", False)


def _current_convergence():
    from workflows.agent_team.convergence import current_convergence
    return current_convergence()


def make_meter_class(helpers, convergence_getter=None):
    """Build on the actual trial ledger without changing its pricing policy."""
    get_state = convergence_getter or _current_convergence

    class ConvergenceMeter(helpers.Meter):
        def __init__(self, *, final_max_tokens=49152):
            super().__init__()
            self.changed = asyncio.Condition()
            self.stop = asyncio.Event()
            self.reason = ""
            self.final_max_tokens = final_max_tokens
            self.final_upper = reservation_upper(final_max_tokens)
            self.final_claimed = False
            self.final_request_id = None
            self.final_allowance_released = False

        def snapshot(self):
            result = super().snapshot()
            result.update(
                budget_mode="bounded_with_existing_final_response_admission",
                target_policy="soft_target_unchanged",
                research_admission_upper_cny=str(helpers.HARD - self.final_upper),
                final_response_upper_cny=str(self.final_upper),
                final_response_output_upper_tokens=self.final_max_tokens,
                final_response_admitted=self.final_claimed,
                final_response_allowance_released=self.final_allowance_released,
                protected_final_admission_upper_cny=(
                    "0" if self.final_claimed or self.final_allowance_released
                    else str(self.final_upper)
                ),
                final_reserve_is_inside_hard_cap=True,
                pricing_date_policy="unchanged; unverified dates retain conservative factor 2",
                scope="fresh_attempt_only",
            )
            return result

        def release_final_allowance(self):
            # A virtual protected allowance is not spent, held, or inflight.
            # Returning an already-latched answer needs no new model request.
            self.final_allowance_released = True
            self.write()

        def configure_final_output_upper(self, active_cap, model="deepseek-flash"):
            if self.calls or self.reserved:
                raise BudgetAdmissionError("final_reserve_configuration_after_admission")
            if isinstance(active_cap, str):
                try:
                    active_cap = int(active_cap)
                except ValueError:
                    raise BudgetAdmissionError("invalid_model_output_bound") from None
            self.final_max_tokens = existing_final_output_upper(active_cap)
            self.final_upper = reservation_upper(self.final_max_tokens)
            if model != "deepseek-flash":
                # A closing synthesis runs on the main model, at that model's prices.
                self.final_upper = max(self.final_upper, helpers.reservation_amount(model, self.final_max_tokens))
            self.write()

        def _reject(self, code, state, *, hard_stop=False):
            if not self.reason or hard_stop:
                self.reason = code
            if state is not None:
                state.request_finalization(code)
            if hard_stop:
                self.stop.set()
            helpers.emit("model_admission_rejected", reason=code,
                         error_domain="local_model_budget", final_admitted=self.final_claimed)
            raise BudgetAdmissionError(code)

        async def reserve(self, payload, role):
            amount = reservation_upper(payload["max_tokens"])
            model = payload.get("model", "deepseek-flash")
            if model != "deepseek-flash":
                amount = helpers.reservation_amount(model, payload["max_tokens"])
            state = get_state()
            final_token = state.finalization_request_id() if state is not None else None
            final_request = final_token is not None and state.is_finalization_request()
            async with self.changed:
                if state is not None and _answer_latched(state):
                    self.release_final_allowance()
                    self._reject("final_answer_already_latched", state)
                if final_request:
                    if (
                        self.final_allowance_released
                        or self.final_claimed and final_token is not self.final_request_id
                    ):
                        self._reject("final_response_already_admitted", state)
                    if amount > self.final_upper:
                        self._reject("final_response_exceeds_reserved_bound", state, hard_stop=True)
                elif state is not None and state.phase not in {"research", "review_required"}:
                    self._reject("model_budget_research_closed", state)

                limit = helpers.HARD if final_request or state is None else helpers.HARD - self.final_upper
                while self.spent + self.held + self.reserved + amount > limit:
                    if self.reserved == 0:
                        if state is not None and not final_request:
                            self._reject("model_budget_research_limit", state)
                        self._reject("model_budget_admission_exhausted", state, hard_stop=True)
                    await self.changed.wait()
                    if state is not None and not final_request and state.phase not in {"research", "review_required"}:
                        self._reject("model_budget_research_closed", state)
                ident = await super().reserve(payload, role)
                self.entries[ident]["final_response"] = final_request
                self.entries[ident]["final_token"] = final_token
                self.entries[ident]["convergence_state"] = state
                self.entries[ident]["dispatched"] = False
                if final_request:
                    self.final_claimed = True
                    self.final_request_id = final_token
                self.write()
                return ident

        async def mark_dispatched(self, ident):
            async with self.changed:
                entry = self.entries[ident]
                state = get_state()
                closed = state is not entry["convergence_state"]
                if state is not None:
                    if entry["final_response"]:
                        closed = closed or not (
                            state.phase == "finalizing" and not _answer_latched(state)
                            and state.is_finalization_request()
                            and state.finalization_request_id() is entry["final_token"]
                        )
                    else:
                        closed = closed or state.phase not in {"research", "review_required"} or _answer_latched(state)
                if closed:
                    # This request has not crossed the external send boundary.
                    # Release here under the same lock; cancel() would re-enter it.
                    self.entries.pop(ident)
                    self.reserved -= entry["amount"]
                    self.write()
                    self.changed.notify_all()
                    helpers.emit("model_dispatch_rejected", call=ident,
                                 reason="model_request_dispatch_closed", error_domain="local_convergence")
                    from workflows.agent_team.convergence import ConvergenceBlocked
                    raise ConvergenceBlocked("model_request_dispatch_closed")
                entry["dispatched"] = True

        async def cancel(self, ident):
            """Release only a known-unsent reservation; unknown remote cost stays held."""
            async with self.changed:
                entry = self.entries.get(ident)
                if entry is None:
                    return
                if entry.get("dispatched"):
                    await super().settle(ident, None)
                else:
                    self.entries.pop(ident)
                    self.reserved -= entry["amount"]
                    self.write()
                    helpers.emit("model_reservation_cancelled_before_dispatch", call=ident)
                self.changed.notify_all()

        async def settle(self, ident, usage):
            async with self.changed:
                if ident in self.entries:
                    try:
                        await super().settle(ident, usage)
                    finally:
                        self.changed.notify_all()

    return ConvergenceMeter
