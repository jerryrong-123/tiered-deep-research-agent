"""Opt-in scheduling controls, not an additional semantic judge or model loop.

The coordinator describes factual gaps using current evidence receipts. The
runtime owns dispatch admission. Resource checkpoints request strategy review;
they never upgrade, approve an answer, or reject it on their own.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Iterator

DEFAULTS = {
    "review_after_main_turns": 6,
    "no_progress_cycles": 2,
    "max_helpers_single_gap": 1,
    "max_helpers_parallel": 2,
}
# Off unless a profile opts in. Trial values, not calibrated thresholds:
# see local_profiles/TIERED_RESEARCH_V1.md.
TIER_DEFAULTS = {
    "tiered_escalation": False,
    "stop_at_escalation": False,
    "min_question_coverage_pct": 95,
    "max_requirement_words": 30,
    "escalate_after_gate_rejections": 3,
    "escalate_after_main_turns": 24,
    "light_soft_calls": 60,
    "light_hard_calls": 90,
    "answer_chain_gate": False,
    "min_ask_coverage_pct": 80,
    "max_search_queries": 0,  # 0 means no limit.
    "synthesize_at_stop": False,
    "independent_check": False,
    "show_low_value_results": False,
    "unquote_single_words": False,
    "archive_reads": 0,  # Saved pages one run may read; 0 means none.
    "calculator": False,  # The prompt asks for `calculate` on comparisons and totals.
    "answer_leads": 0,  # Lead checks one run may make; 0 means no leads.
}
TIER_SWITCHES = ("tiered_escalation", "stop_at_escalation", "answer_chain_gate",
                 "synthesize_at_stop", "independent_check", "show_low_value_results",
                 "unquote_single_words", "calculator")
ZERO_MEANS_OFF = ("max_search_queries", "archive_reads", "answer_leads")
# Quoting one word demands that exact word, so a page using a synonym drops out.
# Phrases keep their quotes, as do names and numbers, which have no synonym to lose.
_QUOTED_WORD = re.compile(r'"([a-z]+(?:-[a-z]+)*)"')
MIN_QUOTE_WORDS = 3
MAX_REPORTED_PROBLEMS = 8
_SUCCESSFUL_RECEIPTS = frozenset({"complete", "tool_completed"})
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his in into "
    "is it its of on or she that the their them they this to was were which who "
    "whom whose with".split()
)
GAP_KINDS = ("missing_relation", "source_conflict", "candidate_discovery")
SOURCE_TOOLS = frozenset({"web_search", "web_fetch", "read_file", "grep_search",
                          "glob_search", "download_file", "recover_result"})
READ_TOOLS = SOURCE_TOOLS - {"web_search", "glob_search"}
# A calculation is kept as a receipt so a row can cite it, but it is not a source:
# it neither counts as a read nor needs a search before it.
CALCULATION_TOOL = "calculate"
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# Conversion factors and small counts need no page behind them.
_UNIT_NUMBERS = frozenset({0, 1, 2, 3, 4, 5, 7, 10, 12, 24, 30, 31, 60, 100, 365, 366, 1000,
                           1.609344, 1609.344, 0.621371, 0.3048, 3.28084})
_assignments: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
    "progressive_assignments", default={},
)


def assignment_metadata(agent: str) -> dict[str, Any]:
    """Called at the real native enqueue site, inside the guarded assignment."""
    return deepcopy(_assignments.get().get(agent, {}))


@contextmanager
def assignment_scope(values: dict[str, dict[str, Any]]) -> Iterator[None]:
    token = _assignments.set(values)
    try:
        yield
    finally:
        _assignments.reset(token)


def gap_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "gap_id": {"type": "string"},
            "kind": {"type": "string", "enum": list(GAP_KINDS)},
            "requirement": {"type": "string", "description": "The actual original-question requirement."},
            "subject": {"type": "string", "description": "Specific candidate, object or event under investigation."},
            "question": {"type": "string", "description": "The unresolved factual relationship or concrete conflict, not a generic final audit."},
            "source_receipt_ids": {"type": "array", "items": {"type": "string"}},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["gap_id", "requirement", "subject", "question", "source_receipt_ids", "depends_on"],
    }


def requirement_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_span": {"type": "string", "description": "Words copied verbatim from the original question: one condition."},
            "receipt_id": {"type": "string", "description": "A saved receipt supporting this condition."},
            "evidence_quote": {"type": "string", "description": "A passage copied verbatim from that receipt; put ... between separated passages."},
        },
        "required": ["question_span", "receipt_id", "evidence_quote"],
    }


def _words(text: Any) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold())


def _phrase(words: list[str]) -> str:
    return " " + " ".join(words) + " "


def _quote_phrases(quote: Any) -> list[str]:
    """Punctuation-insensitive, because models drop markdown and curly quotes."""
    pieces = [_words(piece) for piece in re.split(r"\.{3,}|…", str(quote))]
    pieces = [piece for piece in pieces if piece]
    if sum(len(piece) for piece in pieces) < MIN_QUOTE_WORDS:
        return []
    return [_phrase(piece) for piece in pieces]


def _numbers(text: Any) -> set[float]:
    """Every number in a text, unsigned: a page writes 73.56° W where a calculation has -73.56."""
    return {float(match.replace(",", "")) for match in _NUMBER.findall(unicodedata.normalize("NFKC", str(text)))}


def unquoted_calculation_inputs(question: str, rows: list[Any], found_rows: list[tuple[int, str]],
                                receipts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """A calculation counts only when each input comes from the question or a quoted page."""
    def is_calculation(found: str) -> bool:
        return ((receipts.get(found) or {}).get("source") or {}).get("tool_name") == CALCULATION_TOOL

    calculations = [(index, found) for index, found in found_rows if is_calculation(found)]
    if not calculations:
        return []
    sourced = _numbers(question)
    for index, found in found_rows:
        if not is_calculation(found):
            sourced |= _numbers(rows[index].get("evidence_quote", ""))
    problems = []
    for index, found in calculations:
        report = str(receipts[found].get("report") or "")
        line = next((item for item in report.splitlines() if item.startswith("INPUTS:")), "")
        missing = [number for number in sorted(_numbers(line)) if number not in _UNIT_NUMBERS
                   and not any(math.isclose(number, known, rel_tol=1e-9) for known in sourced)]
        if missing:
            problems.append({"row": index, "problem": "calculation_input_not_quoted",
                             "inputs": [f"{number:g}" for number in missing[:5]]})
    return problems


def ask_positions(question: str) -> list[int]:
    """Token positions of the sentences that say what is being asked.

    A question mark names them. Without one the closing sentence asks, as in
    "Please provide the name of the street."
    """
    sentences = re.split(r"(?<=[.?!])\s+", str(question).strip())
    asking = {i for i, sentence in enumerate(sentences) if "?" in sentence} or {len(sentences) - 1}
    positions: list[int] = []
    start = 0
    for index, sentence in enumerate(sentences):
        count = len(_words(sentence))
        if index in asking:
            positions.extend(range(start, start + count))
        start += count
    return positions


def unquote_single_words(query: Any) -> Any:
    """Drop the quotes around single ordinary words in one query or a list."""
    if isinstance(query, list):
        return [unquote_single_words(item) for item in query]
    return _QUOTED_WORD.sub(r"\1", query) if isinstance(query, str) else query


def source_failure_kind(name: str, text: str) -> str:
    """Recognize existing tool error envelopes, not words inside source prose.

    Native web tools can return an error string with ToolResult.is_error=False.
    Multi-URL fetch is failed only when EVERY numbered result is an error.
    Mixed bundles retain their successful text for ordinary semantic review.
    """
    stripped = text.strip()
    if name == "web_search" and stripped.startswith(("Error: search failed (", "[ERROR]:")):
        return "search_reported_failure"
    if name == "web_fetch":
        if stripped.startswith(("[ERROR]:", "[BLOCKED]", "Error: URL is required.")):
            return "fetch_reported_failure"
        blocks = re.split(r"(?m)^\[\d+\] URL: [^\n]*\n\s+Info: ", stripped)
        if len(blocks) > 1 and all(part.lstrip().startswith("[ERROR]:") for part in blocks[1:]):
            return "fetch_reported_failure"
    return ""


class ProgressivePolicy:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        if not isinstance(config, dict) or set(config) - set(DEFAULTS) - set(TIER_DEFAULTS):
            raise ValueError("invalid_progressive_config")
        self.config = {**DEFAULTS, **{k: v for k, v in config.items() if k in DEFAULTS}}
        if any(type(value) is not int or value < 1 for value in self.config.values()):
            raise ValueError("invalid_progressive_threshold")
        if self.config["max_helpers_single_gap"] > self.config["max_helpers_parallel"]:
            raise ValueError("invalid_progressive_helper_limits")
        tier = {**TIER_DEFAULTS, **{k: v for k, v in config.items() if k in TIER_DEFAULTS}}
        numbers = [value for key, value in tier.items()
                   if key not in TIER_SWITCHES and key not in ZERO_MEANS_OFF]
        if (any(type(tier[key]) is not bool for key in TIER_SWITCHES)
                or any(type(value) is not int or value < 1 for value in numbers)
                or any(type(tier[key]) is not int or tier[key] < 0 for key in ZERO_MEANS_OFF)
                or tier["min_question_coverage_pct"] > 100 or tier["min_ask_coverage_pct"] > 100
                or tier["light_soft_calls"] >= tier["light_hard_calls"]):
            raise ValueError("invalid_tier_config")
        self.tier = tier if tier["tiered_escalation"] else None
        self.gate_rejections = 0
        self.search_queries = 0
        self.gaps: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.lock = asyncio.Lock()
        self.main_turns = 0
        self._seen_turns: set[tuple[str, str, int]] = set()
        self.cycles_since_progress = 0
        self._search_waiting = False
        self._noticed_turn = 0
        self._noticed_cycles = 0
        self._progress_receipts: set[str] = set()
        self._progress_bodies: set[str] = set()
        self._progress_summaries: list[dict[str, Any]] = []
        self._tool_ids: set[str] = set()

    def validate_gaps(self, tasks: Any, receipts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not isinstance(tasks, list):
            raise ValueError("gap_tasks_must_be_array")
        parsed: dict[str, dict[str, Any]] = {}
        allowed = set(gap_schema()["properties"])
        for raw in tasks:
            if not isinstance(raw, dict) or set(raw) - allowed:
                raise ValueError("invalid_gap_task")
            for field in ("gap_id", "requirement", "subject", "question"):
                if not isinstance(raw.get(field), str) or not raw[field].strip():
                    raise ValueError("gap_task_missing_" + field)
            item = deepcopy(raw)
            item.setdefault("kind", "missing_relation")
            if item["kind"] not in GAP_KINDS:
                raise ValueError("gap_kind_is_not_a_factual_research_gap")
            for field in ("source_receipt_ids", "depends_on"):
                values = item.get(field)
                if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
                    raise ValueError("invalid_gap_" + field)
                item[field] = list(dict.fromkeys(values))
            if not item["source_receipt_ids"] or not set(item["source_receipt_ids"]) <= set(receipts):
                raise ValueError("gap_requires_current_source_receipts")
            if any(receipts[x]["status"] not in {"complete", "tool_completed"}
                   for x in item["source_receipt_ids"]):
                raise ValueError("gap_source_is_technical_failure")
            key = item["gap_id"].strip()
            if key in parsed:
                raise ValueError("duplicate_gap_id")
            item["gap_id"] = key
            parsed[key] = item
        for key, item in parsed.items():
            if not set(item["depends_on"]) <= set(parsed) | set(self.gaps):
                raise ValueError("unknown_gap_dependency")
            path: set[str] = set()

            def visit(node: str) -> None:
                if node in path:
                    raise ValueError("cyclic_gap_dependency")
                if node not in parsed:
                    return
                path.add(node)
                for dep in parsed[node]["depends_on"]:
                    visit(dep)
                path.remove(node)

            visit(key)
        return parsed

    def accept_gaps(self, tasks: dict[str, dict[str, Any]]) -> None:
        self.history.append({"previous": deepcopy(self.gaps), "current": deepcopy(tasks)})
        self.gaps = tasks

    def acknowledge_progress(self, summary: str, ids: list[str], receipts: dict[str, dict[str, Any]]) -> None:
        if not ids and not summary:
            return
        if not summary.strip() or not ids or not set(ids) <= set(receipts):
            raise ValueError("progress_requires_summary_and_current_receipts")
        if any(receipts[x]["status"] not in {"complete", "tool_completed"} for x in ids):
            raise ValueError("technical_failure_is_not_research_progress")
        bodies = {receipts[x]["source"]["report_sha256"] for x in ids}
        # A second URL or a new receipt ID with identical body is not new evidence.
        if not bodies - self._progress_bodies:
            # Progress bookkeeping must not veto otherwise valid continuation
            # or finalization. Re-reading the same material may help understand
            # a new relationship, but is not counted as obtaining a new source.
            return
        self._progress_receipts.update(ids)
        self._progress_bodies.update(bodies)
        self._progress_summaries.append({"summary": summary, "receipt_ids": list(ids)})
        self.cycles_since_progress = 0
        self._noticed_cycles = 0

    def observe_tool(self, name: str, ident: str, *, is_error: bool) -> None:
        if ident in self._tool_ids:
            return
        self._tool_ids.add(ident)
        if is_error:
            return  # Transport / tool failures are not semantic difficulty.
        if name == "web_search":
            self._search_waiting = True
        elif name in READ_TOOLS and self._search_waiting:
            self.cycles_since_progress += 1
            self._search_waiting = False

    def on_main_turn(self, task_id: str, turn: int, *, effective: bool, loop_id: str = "") -> str:
        identity = (task_id, loop_id, turn)
        if not effective or identity in self._seen_turns:
            return ""
        self._seen_turns.add(identity)
        self.main_turns += 1
        interval = self.config["review_after_main_turns"]
        turns_due = self.main_turns - self._noticed_turn >= interval
        threshold = self.config["no_progress_cycles"]
        # Tiered runs space the notice out; v1 kept its original cadence.
        spacing = threshold if self.tier is not None else 1
        cycles_due = (self.cycles_since_progress >= threshold
                      and self.cycles_since_progress - self._noticed_cycles >= spacing)
        if not turns_due and not cycles_due:
            return ""
        self._noticed_turn = self.main_turns
        self._noticed_cycles = self.cycles_since_progress
        return (
            "Progressive research checkpoint (not an instruction to upgrade or refuse). "
            "Review current evidence against the original question in your next ordinary decision. "
            "Record actual progress through review_research with progress_summary and progress_receipt_ids. "
            "If there is a specific gap, change the query/source or declare gap_tasks before delegation. "
            "A complete search/read cycle without acknowledged progress requests strategy review only. "
            "Do not add a generic final_verifier, duplicate the whole question, or treat a tool failure as difficulty. "
            f"Effective main turns={self.main_turns}; cycles since progress={self.cycles_since_progress}."
        )

    def turn_budget_spent(self) -> bool:
        return self.tier is not None and self.main_turns >= self.tier["escalate_after_main_turns"]

    def _row_problem(self, row: Any, tokens: list[str],
                     bodies: dict[str, str]) -> tuple[str, list[int], str]:
        fields = ("question_span", "receipt_id", "evidence_quote")
        if not isinstance(row, dict) or any(not isinstance(row.get(f), str) or not row[f].strip() for f in fields):
            return "malformed_requirement", [], ""
        span = _words(row["question_span"])
        if not span or len(span) > self.tier["max_requirement_words"]:
            return "question_span_length", [], ""
        starts = [i for i in range(len(tokens) - len(span) + 1) if tokens[i:i + len(span)] == span]
        if not starts:
            return "question_span_not_verbatim", [], ""
        phrases = _quote_phrases(row["evidence_quote"])
        if not phrases:
            return "evidence_quote_too_short", [], ""
        # A real quote under a mistyped receipt ID is still traceable.
        cited = row["receipt_id"]
        for receipt_id in ([cited] if cited in bodies else []) + [key for key in bodies if key != cited]:
            if all(phrase in bodies[receipt_id] for phrase in phrases):
                return "", [i + offset for i in starts for offset in range(len(span))], receipt_id
        return "evidence_quote_not_in_receipt", [], ""

    def check_requirements(self, question: str, rows: Any,
                           receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Traceability only: a real quote can still fail to support its condition.

        The program owns the question, so the uncovered parts it returns are
        gaps found by the program, not declared by the model.
        """
        tokens = _words(question)
        covered = [False] * len(tokens)
        problems, corrections, found_rows = [], [], []
        bodies = {key: _phrase(_words(receipt.get("report", ""))) for key, receipt in receipts.items()
                  if receipt["status"] in _SUCCESSFUL_RECEIPTS}
        for index, row in enumerate(rows if isinstance(rows, list) else []):
            problem, positions, found = self._row_problem(row, tokens, bodies)
            if problem:
                problems.append({"row": index, "problem": problem})
            else:
                found_rows.append((index, found))
                if found != row["receipt_id"]:
                    corrections.append({"row": index, "cited": row["receipt_id"], "found": found})
            for position in positions:
                covered[position] = True
        problems += unquoted_calculation_inputs(question, rows, found_rows, receipts)
        content = [i for i, token in enumerate(tokens) if token not in _STOPWORDS]
        hits = sum(covered[i] for i in content)
        percent = 100 * hits // len(content) if content else 100
        uncovered, run = [], []
        for token, done in zip([*tokens, ""], [*covered, True]):
            if not done:
                run.append(token)
                continue
            if any(word not in _STOPWORDS for word in run):
                uncovered.append(" ".join(run))
            run = []
        verdict = {
            "ok": not problems and percent >= self.tier["min_question_coverage_pct"],
            "coverage_pct": percent,
            "required_coverage_pct": self.tier["min_question_coverage_pct"],
            "uncovered_question_parts": uncovered,
            "row_problems": problems[:MAX_REPORTED_PROBLEMS],
            "receipt_corrections": corrections,
        }
        if self.tier["answer_chain_gate"]:
            # What is asked must be traced; background clues may stay unverified.
            asked = [i for i in ask_positions(question) if tokens[i] not in _STOPWORDS]
            ask_percent = 100 * sum(covered[i] for i in asked) // len(asked) if asked else 100
            required = self.tier["min_ask_coverage_pct"]
            verdict.update(ok=verdict["ok"] and ask_percent >= required,
                           ask_coverage_pct=ask_percent, required_ask_coverage_pct=required)
        return verdict

    def active_jobs(self, bus: Any, task_id: str) -> list[dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        for session in bus.list_sessions_for_task(task_id):
            job = getattr(session, "current_job_id", None)
            if job and bus.get_job_status(job) not in {"completed", "failed", "aborted"}:
                meta = bus.current_job_metadata(session.session_id)
                active[job] = {"job_id": job, "agent": session.name,
                               "gap_id": meta.get("progressive_gap_id", ""),
                               "gap": meta.get("progressive_gap")}
            for queued in getattr(session, "pending_tasks", ()):
                meta = queued.task_metadata or {}
                active[queued.job_id] = {"job_id": queued.job_id, "agent": session.name,
                                        "gap_id": meta.get("progressive_gap_id", ""),
                                        "gap": meta.get("progressive_gap")}
        return list(active.values())

    def prepare_dispatch(self, tasks: Any, bus: Any, task_id: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("progressive_tasks_required")
        active = self.active_jobs(bus, task_id)
        gap_ids = [item.get("gap_id") for item in tasks if isinstance(item, dict)]
        if len(gap_ids) != len(tasks) or any(x not in self.gaps for x in gap_ids):
            raise ValueError("assignment_requires_approved_gap_id")
        if len(set(gap_ids)) != len(gap_ids) or set(gap_ids) & {x["gap_id"] for x in active}:
            raise ValueError("duplicate_inflight_gap")
        def scope_key(gap: dict[str, Any]) -> tuple[str, ...]:
            return tuple(" ".join(gap[key].split()).casefold() for key in ("requirement", "subject", "question"))

        scopes = [scope_key(self.gaps[key]) for key in gap_ids]
        live_scopes = {scope_key(x["gap"]) for x in active if isinstance(x.get("gap"), dict)}
        if len(set(scopes)) != len(scopes) or set(scopes) & live_scopes:
            raise ValueError("duplicate_inflight_gap_scope")
        # No queued second job, even under a fresh alias, can bypass this limit.
        independent = [key for key, gap in self.gaps.items() if not set(gap["depends_on"]) & set(self.gaps)]
        limit = self.config["max_helpers_parallel"] if len(independent) >= 2 else self.config["max_helpers_single_gap"]
        if len(active) + len(tasks) > limit:
            raise ValueError("progressive_active_and_queued_limit")
        names: set[str] = set()
        result, bindings = [], {}
        for raw in tasks:
            item = deepcopy(raw)
            key = item.pop("gap_id")
            gap = self.gaps[key]
            if set(gap["depends_on"]) & set(self.gaps):
                raise ValueError("gap_dependency_still_unresolved")
            agent = item.get("agent")
            if not isinstance(agent, str) or not agent.strip() or agent != agent.strip() or agent in names:
                raise ValueError("distinct_existing_agents_required")
            if bus.get_session(f"{task_id}::{agent}") is None:
                raise ValueError("progressive_unknown_agent")
            names.add(agent)
            prompt = item.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("progressive_task_prompt_required")
            item["prompt"] = (
                "# Assigned factual gap (scope fixed by the coordinator/runtime)\n"
                + json.dumps(gap, ensure_ascii=False) + "\n"
                "Use the shared findings and sources; research only this gap. The full question is context, "
                "not a request to repeat the whole investigation. Return original sources/quotes, any conflict, "
                "and what remains unproved. A local link-search miss is not proof that external evidence is absent.\n\n"
                + prompt
            )
            bindings[agent] = {"progressive_gap_id": key, "progressive_gap": deepcopy(gap)}
            result.append(item)
        return result, bindings

    def snapshot(self) -> dict[str, Any]:
        return deepcopy({"version": "progressive_research_v1", "config": self.config,
                         "gap_tasks": list(self.gaps.values()), "gap_history": self.history,
                         "effective_main_turns": self.main_turns,
                         "cycles_since_progress": self.cycles_since_progress,
                         "progress": self._progress_summaries,
                         **({"tier": self.tier, "gate_rejections": self.gate_rejections,
                             "search_queries": self.search_queries}
                            if self.tier is not None else {})})
