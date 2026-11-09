"""Shared, dictionary-compatible search failure metadata and safe formatting."""

from __future__ import annotations

# Reserved metadata for failed raw searches. Successful provider dictionaries
# retain their existing shape; a genuine empty response remains empty. Keeping
# failures as dictionaries lets existing raw consumers continue to use .get().
# Consumers must check this field before treating a response as empty/cacheable.
# Only these local reason codes and an optional HTTP status may cross the tool
# boundary: never provider messages, exception text, request URLs, or credentials.
SEARCH_ERROR_KEY = "_search_error"
_SEARCH_ERROR_REASONS = frozenset({
    "configuration_error", "http_error", "provider_error", "rate_limited", "server_error",
    "timeout", "transport_error", "invalid_response", "unexpected_error",
})


def _search_failure(reason: object, http_status: int | None = None) -> dict:
    safe_reason = (
        reason if isinstance(reason, str) and reason in _SEARCH_ERROR_REASONS
        else "unexpected_error"
    )
    error: dict = {"reason": safe_reason}
    if type(http_status) is int and 100 <= http_status <= 599:
        error["http_status"] = http_status
    return {SEARCH_ERROR_KEY: error}


def _format_search_error(data: dict) -> str:
    value = data.get(SEARCH_ERROR_KEY)
    error = value if isinstance(value, dict) else {}
    safe = _search_failure(error.get("reason"), error.get("http_status"))[SEARCH_ERROR_KEY]
    status = f"; HTTP {safe['http_status']}" if "http_status" in safe else ""
    return f"Error: search failed ({safe['reason']}{status})."
