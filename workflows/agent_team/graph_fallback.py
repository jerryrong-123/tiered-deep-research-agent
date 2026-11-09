"""Relation-graph fallback: the parts of the last layer that need no model.

The layer runs only on a question Flash and Pro left without a verified answer.
The question is decomposed three ways into a relation graph, the drafts are merged
and checked several ways, and each relation is then researched by its own
sub-agents. See outputs/fallback_controller_design_20261103/DESIGN.md.

This module decides what a program can decide: the graph's shape and checks, how
the drafts line up, the branches an ambiguity opens, which relation is ready, how
candidates intersect, the budget reserve kept for the final checks and the final
verdict. ``graph_fallback_controller`` makes the model calls around it. Graphs and
branch states are plain JSON-ready dicts, so a checkpoint is just their dump.
"""
from __future__ import annotations

import itertools
import json
import os
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any

from workflows.agent_team.progressive import _STOPWORDS, _phrase, _quote_phrases, _words

RELATION_KINDS = ("lookup", "filter", "compare", "compute")
NODE_KINDS = ("unknown", "constant", "answer")
# Structural parameters approved with the design on 2026-11-03. They are caps, not
# quotas: a run stops using any of them as soon as its work is done.
DEFAULTS: dict[str, Any] = {
    "decompositions": 3,
    "redo_per_decomposition": 2,
    "check_rounds": 3,
    "rechecks_during_run": 2,
    "max_branches": 4,
    "agents_per_relation": 2,
    "extra_agents_per_conflict": 1,
    "max_subagents": 40,
    "max_parallel": 4,
    "subagent_turns": 20,
    "subagent_queries": 25,
    # Research sub-agents' model; empty runs them on the trial's own model. The
    # re-verifier is part of the final checks and always uses the trial's model.
    "subagent_model": "",
    "replans": 2,
    "max_candidates": 5,
    # V2.6, approved on 2026-11-03: a lookup with more candidates than
    # max_candidates keeps up to this many and tests them in batches.
    "wide_candidates": 30,
    "wide_batch": 10,
    # Steps a branch's controller may choose when research cannot go on as planned.
    "next_steps": 6,
    # Rounds of re-research on the relations a failed final check did not accept.
    "repair_rounds": 3,
    "min_coverage_pct": 95,
    "verification_reserve_cny": "8",
    "format_retries": 2,
}
# Drafts' relations are the same relation when their spans share this much.
SAME_RELATION_OVERLAP = 0.5
# A restated condition matches a question clause when it holds this share of
# the clause's content words.
CLAUSE_RECALL = 0.5


def settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """The defaults with a profile's overrides; unknown keys are refused."""
    merged = dict(DEFAULTS)
    for key, value in (config or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown_graph_fallback_setting:{key}")
        merged[key] = value
    return merged


def content_words(text: Any) -> list[str]:
    return [word for word in _words(text) if word not in _STOPWORDS]


def norm_entity(name: Any) -> str:
    """Case, accents, hyphens and punctuation do not make two names different."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(_words(plain))


def _span_positions(tokens: list[str], span: str) -> list[int]:
    words = _words(span)
    if not words:
        return []
    starts = [i for i in range(len(tokens) - len(words) + 1) if tokens[i:i + len(words)] == words]
    return [start + offset for start in starts for offset in range(len(words))]


def check_graph(question: str, graph: Any, *, min_coverage_pct: int = DEFAULTS["min_coverage_pct"]) -> dict[str, Any]:
    """Program checks on one graph: shape, verbatim spans, coverage and no cycle.

    A graph that passes can still misread the question; that is for the reviews.
    """
    problems: list[str] = []
    if not isinstance(graph, dict):
        return {"ok": False, "problems": ["not_an_object"], "coverage_pct": 0, "uncovered": []}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
    answer = graph.get("answer") if isinstance(graph.get("answer"), dict) else {}
    if not nodes:
        problems.append("no_nodes")
    if not relations:
        problems.append("no_relations")
    ids: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict) or not str(node.get("id", "")).strip():
            problems.append("malformed_node")
            continue
        if node["id"] in ids:
            problems.append(f"duplicate_node:{node['id']}")
        if node.get("kind") not in NODE_KINDS:
            problems.append(f"node_kind:{node['id']}")
        if node.get("kind") == "constant" and not str(node.get("value", "")).strip():
            problems.append(f"constant_without_value:{node['id']}")
        if "count" in node and (type(node["count"]) is not int or node["count"] < 1):
            problems.append(f"node_count:{node['id']}")
        ids[node["id"]] = node.get("kind", "")
    answers = [key for key, kind in ids.items() if kind == "answer"]
    if len(answers) != 1:
        problems.append("answer_nodes:" + str(len(answers)))
    if answers and answer.get("node") != answers[0]:
        problems.append("answer_node_mismatch")
    if not str(answer.get("form", "")).strip():
        problems.append("answer_form_missing")

    tokens = _words(question)
    covered = [False] * len(tokens)
    seen: set[str] = set()
    edges: dict[str, set[str]] = {key: set() for key in ids}
    for relation in relations:
        if not isinstance(relation, dict) or not str(relation.get("id", "")).strip():
            problems.append("malformed_relation")
            continue
        rid = relation["id"]
        if rid in seen:
            problems.append(f"duplicate_relation:{rid}")
        seen.add(rid)
        positions = _span_positions(tokens, str(relation.get("span", "")))
        if not positions:
            problems.append(f"span_not_verbatim:{rid}")
        for position in positions:
            covered[position] = True
        subject = relation.get("subject")
        if subject not in ids:
            problems.append(f"unknown_subject:{rid}")
        if relation.get("object") and relation["object"] not in ids:
            problems.append(f"unknown_object:{rid}")
        if relation.get("kind") not in RELATION_KINDS:
            problems.append(f"relation_kind:{rid}")
        depends = relation.get("depends_on") if isinstance(relation.get("depends_on"), list) else None
        if depends is None:
            problems.append(f"depends_on_missing:{rid}")
            depends = []
        for node in depends:
            if node not in ids:
                problems.append(f"unknown_dependency:{rid}:{node}")
            elif node == subject:
                problems.append(f"depends_on_own_subject:{rid}")
            elif subject in edges:
                edges[node].add(subject)
    for key, kind in ids.items():
        if kind == "constant":
            continue
        own = [r for r in relations if isinstance(r, dict) and r.get("subject") == key]
        if not own:
            problems.append(f"unconstrained_node:{key}")
        elif not any(r.get("anchor") is True for r in own):
            problems.append(f"no_anchor:{key}")
    if _has_cycle(edges):
        problems.append("dependency_cycle")

    content = [i for i, token in enumerate(tokens) if token not in _STOPWORDS]
    percent = 100 * sum(covered[i] for i in content) // len(content) if content else 100
    if percent < min_coverage_pct:
        problems.append(f"coverage:{percent}")
    uncovered, run = [], []
    for token, done in zip([*tokens, ""], [*covered, True]):
        if not done:
            run.append(token)
            continue
        if any(word not in _STOPWORDS for word in run):
            uncovered.append(" ".join(run))
        run = []
    return {"ok": not problems, "problems": problems, "coverage_pct": percent, "uncovered": uncovered}


def _has_cycle(edges: dict[str, set[str]]) -> bool:
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for target in edges.get(node, ()):
            if state.get(target) == 1 or (state.get(target) is None and visit(target)):
                return True
        state[node] = 2
        return False

    return any(state.get(node) is None and visit(node) for node in edges)


def _overlap(a: list[str], b: list[str]) -> float:
    left, right = set(a), set(b)
    return len(left & right) / len(left | right) if left | right else 0.0


def align_drafts(drafts: list[dict[str, Any]]) -> dict[str, Any]:
    """Line up the drafts' relations by span, for the merge and the record.

    Each cluster holds at most one relation per draft. A relation joins the best
    cluster, from the drafts it is not already in, whose span words overlap enough.
    """
    clusters: list[dict[str, Any]] = []
    for index, draft in enumerate(drafts):
        for relation in draft.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            words = content_words(relation.get("span", ""))
            best, score = None, 0.0
            for cluster in clusters:
                if index in cluster["drafts"]:
                    continue
                value = _overlap(words, cluster["words"])
                if value > score:
                    best, score = cluster, value
            if best is not None and score >= SAME_RELATION_OVERLAP:
                best["drafts"].append(index)
                best["members"].append({"draft": index, "id": relation.get("id"), "span": relation.get("span"),
                                        "kind": relation.get("kind"), "relation": relation.get("relation")})
                best["words"] = sorted(set(best["words"]) | set(words))
            else:
                clusters.append({"drafts": [index], "words": words,
                                 "members": [{"draft": index, "id": relation.get("id"), "span": relation.get("span"),
                                              "kind": relation.get("kind"), "relation": relation.get("relation")}]})
    total = len(drafts)
    report = []
    for number, cluster in enumerate(clusters, 1):
        kinds = sorted({str(member["kind"]) for member in cluster["members"]})
        report.append({"cluster": number, "in_drafts": sorted(cluster["drafts"]), "members": cluster["members"],
                       "kind_disagreement": len(kinds) > 1})
    in_all = sum(len(item["in_drafts"]) == total for item in report)
    return {
        "drafts": total,
        "clusters": report,
        "in_all": in_all,
        "in_some": sum(1 < len(item["in_drafts"]) < total for item in report),
        "single": sum(len(item["in_drafts"]) == 1 for item in report) if total > 1 else 0,
        "kind_disagreements": sum(item["kind_disagreement"] for item in report),
        "unknown_nodes": [sum(1 for n in d.get("nodes") or [] if isinstance(n, dict) and n.get("kind") == "unknown")
                          for d in drafts],
        "agreement_pct": 100 * in_all // len(report) if report else 100,
    }


def question_clauses(question: str) -> list[str]:
    """Sentences split at commas and semicolons, kept when they carry two content words."""
    pieces: list[str] = []
    for sentence in _sentences(question):
        for part in sentence.replace(";", ",").split(","):
            if len(content_words(part)) >= 2:
                pieces.append(part.strip())
    return pieces


def _sentences(text: str) -> list[str]:
    out, current = [], []
    for ch in str(text):
        current.append(ch)
        if ch in ".?!":
            out.append("".join(current).strip())
            current = []
    if "".join(current).strip():
        out.append("".join(current).strip())
    return out


def restatement_gaps(question: str, conditions: list[str]) -> dict[str, Any]:
    """Program side of the back-translation check.

    A clause is missing when no restated condition holds half its content words.
    A condition is unmatched when it shares no content word with the question.
    Both only point the semantic comparison at places to look.
    """
    restated = [set(content_words(item)) for item in conditions]
    asked = set(content_words(question))
    missing = []
    for clause in question_clauses(question):
        words = set(content_words(clause))
        best = max((len(words & item) / len(words) for item in restated), default=0.0) if words else 1.0
        if best < CLAUSE_RECALL:
            missing.append(clause)
    unmatched = [condition for condition, words in zip(conditions, restated) if words and not words & asked]
    return {"missing_clauses": missing, "unmatched_conditions": unmatched}


def combine_reviews(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """The reviews' findings, deduplicated. Ambiguities do not fail a graph: they open branches."""
    problems: dict[str, dict[str, Any]] = {}
    ambiguous: dict[str, dict[str, Any]] = {}
    for review in reviews:
        if not isinstance(review, dict):
            problems.setdefault("unreadable_review", {"kind": "unreadable", "text": "a review returned no usable JSON",
                                                      "from": []})
            continue
        source = str(review.get("reviewer", ""))
        for kind in ("missing", "extra", "misread"):
            for item in review.get(kind) or []:
                text = str(item.get("text") if isinstance(item, dict) else item).strip()
                if not text:
                    continue
                entry = problems.setdefault(kind + ":" + " ".join(_words(text)), {"kind": kind, "text": text, "from": []})
                if source and source not in entry["from"]:
                    entry["from"].append(source)
        for item in review.get("ambiguous") or []:
            if not isinstance(item, dict) or not str(item.get("span", "")).strip():
                continue
            key = " ".join(_words(item["span"]))
            entry = ambiguous.setdefault(key, {"span": item["span"], "readings": []})
            for reading in item.get("readings") or []:
                if str(reading).strip() and norm_entity(reading) not in {norm_entity(r) for r in entry["readings"]}:
                    entry["readings"].append(str(reading).strip())
    open_ambiguities = [item for item in ambiguous.values() if len(item["readings"]) >= 2]
    return {"ok": not problems, "problems": list(problems.values()), "ambiguities": open_ambiguities}


def agreed_reading(readings: list[str], picks: list[Any]) -> str | None:
    """The reading every judge chose, or None when any judge was undecided,
    unreadable or chose another reading. Only an agreed reading avoids a branch."""
    offered = {norm_entity(r): r for r in readings}
    chosen = {norm_entity(p) for p in picks if isinstance(p, str)}
    if len(picks) < 2 or len(chosen) != 1 or len([p for p in picks if isinstance(p, str)]) != len(picks):
        return None
    key = next(iter(chosen))
    return offered.get(key)


def relation_signature(branch: dict[str, Any], relation: dict[str, Any], inputs: dict[str, Any]) -> str:
    """What makes two branches' relations the same research task: the question's words,
    the relation, what it is about and what it starts from. Ids may differ between
    branch graphs, so nodes are named by their descriptions."""
    def describe(node: str) -> str:
        found = next((n for n in branch["graph"]["nodes"] if n.get("id") == node), {})
        return " ".join(_words(found.get("description", node)))

    return json.dumps({
        "span": " ".join(_words(relation.get("span", ""))), "relation": " ".join(_words(relation.get("relation", ""))),
        "kind": relation.get("kind"), "anchor": relation.get("anchor") is True,
        "subject": describe(relation["subject"]), "count": node_count(branch, relation["subject"]),
        "bound": {describe(node): norm_entity(value) for node, value in inputs["bound"].items()},
        "candidates": sorted(norm_entity(c) for c in inputs["candidates"]),
    }, sort_keys=True, ensure_ascii=False)


def plan_branches(ambiguities: list[dict[str, Any]], max_branches: int = DEFAULTS["max_branches"]) -> list[dict[str, Any]] | None:
    """One branch per combination of readings; None when that is more than allowed."""
    if not ambiguities:
        return [{"id": "b1", "readings": {}}]
    count = 1
    for item in ambiguities:
        count *= len(item["readings"])
    if count > max_branches:
        return None
    spans = [item["span"] for item in ambiguities]
    combos = itertools.product(*[item["readings"] for item in ambiguities])
    return [{"id": f"b{n}", "readings": dict(zip(spans, combo))} for n, combo in enumerate(combos, 1)]


# --- Branch state -----------------------------------------------------------------

def new_branch(branch_id: str, graph: dict[str, Any], readings: dict[str, str] | None = None) -> dict[str, Any]:
    """Constants start bound; every relation starts open."""
    bound = {node["id"]: {"entity": str(node.get("value", "")), "from": "question"}
             for node in graph["nodes"] if node.get("kind") == "constant"}
    return {"id": branch_id, "graph": graph, "readings": dict(readings or {}), "bound": bound,
            "candidates": {}, "status": {r["id"]: "open" for r in graph["relations"]},
            "findings": {}, "conflicts": [], "stuck": []}


def relations_on(branch: dict[str, Any], node: str) -> list[dict[str, Any]]:
    return [r for r in branch["graph"]["relations"] if r.get("subject") == node]


def acts_as_filter(branch: dict[str, Any], relation: dict[str, Any]) -> bool:
    """A relation tests candidates once its node has some, wherever they came from;
    only a node's first relation, its anchor, finds them."""
    return relation.get("anchor") is not True or relation["subject"] in branch["candidates"]


def ready_relations(branch: dict[str, Any]) -> list[dict[str, Any]]:
    """Open relations whose dependencies are bound. A node's anchor goes first:
    its other relations wait for the candidates the anchor finds."""
    ready = []
    for relation in branch["graph"]["relations"]:
        if branch["status"].get(relation["id"]) != "open":
            continue
        if any(node not in branch["bound"] for node in relation.get("depends_on") or []):
            continue
        subject = relation["subject"]
        if subject in branch["bound"]:
            continue
        if relation.get("anchor") is not True and subject not in branch["candidates"]:
            continue
        ready.append(relation)
    ready.sort(key=lambda r: acts_as_filter(branch, r))
    return ready


def describe_node(branch: dict[str, Any], node: str) -> str:
    found = next((n for n in branch["graph"]["nodes"] if n.get("id") == node), {})
    return str(found.get("description") or node)


def task_inputs(branch: dict[str, Any], relation: dict[str, Any], *, unbound: bool = False) -> dict[str, Any]:
    """What a sub-agent is handed: the bound entities its relation depends on and,
    for a relation that tests candidates, the candidates. With ``unbound`` a
    dependency not yet identified is handed over by its description."""
    deps = relation.get("depends_on") or []
    inputs = {node: branch["bound"][node]["entity"] for node in deps if node in branch["bound"]}
    candidates = [c["entity"] for c in branch["candidates"].get(relation["subject"], [])]
    found = {"bound": inputs, "candidates": candidates if acts_as_filter(branch, relation) else []}
    if unbound:
        found["described"] = {node: describe_node(branch, node) for node in deps if node not in branch["bound"]}
    return found


def _quotes(item: dict[str, Any]) -> list[dict[str, str]]:
    quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
    return [q for q in quotes if isinstance(q, dict) and _quote_phrases(q.get("quote", ""))]


def keep_sourced(outcome: dict[str, Any], sources: str) -> dict[str, Any]:
    """Drop every quote that does not occur in what the sub-agent's tools returned.

    A report's quote is the model's word until the page text holds it, the same
    rule the requirement gate applies to evidence quotes.
    """
    body = _phrase(_words(sources))
    kept = dict(outcome)
    dropped = 0
    for key in ("candidates", "verdicts"):
        items = []
        for item in outcome.get(key) or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            quotes = _quotes(item)
            item["quotes"] = [q for q in quotes if all(p in body for p in _quote_phrases(q.get("quote", "")))]
            dropped += len(quotes) - len(item["quotes"])
            items.append(item)
        kept[key] = items
    kept["unsourced_quotes_dropped"] = dropped
    return kept


def _pool(pooled: dict[str, dict[str, Any]], outcome: dict[str, Any], item: Any,
          support: dict[str, list[dict[str, str]]]) -> None:
    key = norm_entity(item["entity"])
    entry = pooled.setdefault(key, {"entity": str(item["entity"]).strip(), "support": {}, "agents": []})
    for rid, quotes in support.items():
        entry["support"].setdefault(rid, []).extend(quotes)
    if outcome.get("agent") not in entry["agents"]:
        entry["agents"].append(outcome.get("agent"))


def record_anchor(branch: dict[str, Any], relation: dict[str, Any], outcomes: list[dict[str, Any]],
                  max_candidates: int = DEFAULTS["max_candidates"],
                  wide_candidates: int = DEFAULTS["wide_candidates"]) -> str:
    """Pool the candidates the anchor's sub-agents found, each with its quotes.

    Only a candidate with at least one quote counts. Up to ``wide_candidates`` are
    kept, so the node's other relations can test them one by one; a longer list is
    ``too_many``. Returns the relation's status.
    """
    pooled: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        for item in outcome.get("candidates") or []:
            if not isinstance(item, dict) or not norm_entity(item.get("entity", "")):
                continue
            quotes = _quotes(item)
            if quotes:
                _pool(pooled, outcome, item, {relation["id"]: quotes})
    branch["findings"].setdefault(relation["id"], []).extend(outcomes)
    if not pooled:
        status = "unresolved"
    elif len(pooled) > max(max_candidates, wide_candidates):
        status = "too_many"
    else:
        branch["candidates"][relation["subject"]] = list(pooled.values())
        status = "established" if len(pooled) <= max_candidates else "wide"
    branch["status"][relation["id"]] = status
    return status


def record_node(branch: dict[str, Any], node: str, outcomes: list[dict[str, Any]],
                wide_candidates: int = DEFAULTS["wide_candidates"]) -> str:
    """Pool a search on all of a node's relations at once.

    Each quote names the relation it supports; a candidate counts with a quote for
    at least one of the node's relations. The node's relations the search did not
    support open again, to be tested on the new candidates.
    """
    own = {r["id"] for r in relations_on(branch, node)}
    pooled: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        for item in outcome.get("candidates") or []:
            if not isinstance(item, dict) or not norm_entity(item.get("entity", "")):
                continue
            support: dict[str, list[dict[str, str]]] = {}
            for quote in _quotes(item):
                if quote.get("relation") in own:
                    support.setdefault(str(quote["relation"]), []).append(quote)
            if support:
                _pool(pooled, outcome, item, support)
    branch["findings"].setdefault(f"node:{node}", []).extend(outcomes)
    if not pooled:
        return "unresolved"
    if len(pooled) > wide_candidates:
        return "too_many"
    branch["candidates"][node] = list(pooled.values())
    for rid in own:
        supported = any(rid in c["support"] for c in pooled.values())
        branch["status"][rid] = "established" if supported else "open"
    return "established"


def record_reverse(branch: dict[str, Any], relation: dict[str, Any], outcomes: list[dict[str, Any]],
                   wide_candidates: int = DEFAULTS["wide_candidates"]) -> str:
    """From a node's candidates, the entities a relation links them to become the
    other node's candidates. The relation itself stays open: it is checked once the
    other node is known. The other node's relations open again, to test them."""
    other = str(relation.get("object") or "")
    pooled: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        for item in outcome.get("candidates") or []:
            if isinstance(item, dict) and norm_entity(item.get("entity", "")) and _quotes(item):
                _pool(pooled, outcome, item, {})
    branch["findings"].setdefault(f"reverse:{relation['id']}", []).extend(outcomes)
    existing = branch["candidates"].get(other)
    if existing:
        found = [c for c in existing if norm_entity(c["entity"]) in pooled]
    else:
        found = list(pooled.values())
    if not found or len(found) > wide_candidates:
        return "unresolved" if not found else "too_many"
    branch["candidates"][other] = found
    for item in relations_on(branch, other):
        if branch["status"].get(item["id"]) != "established":
            branch["status"][item["id"]] = "open"
    return "established"


def record_filter(branch: dict[str, Any], relation: dict[str, Any], outcomes: list[dict[str, Any]]) -> str:
    """Test each candidate against a further relation.

    A candidate passes on a quoted yes with no quoted no. A quoted yes against a
    quoted no is a conflict, left for arbitration; a candidate without any quoted
    yes or no stays, unproven for this relation.
    """
    node = relation["subject"]
    verdicts: dict[str, dict[str, list]] = {}
    for outcome in outcomes:
        for item in outcome.get("verdicts") or []:
            if not isinstance(item, dict):
                continue
            key = norm_entity(item.get("entity", ""))
            quotes = _quotes(item)
            side = {True: "yes", False: "no"}.get(item.get("satisfies"))
            if key and side and quotes:
                verdicts.setdefault(key, {"yes": [], "no": []})[side].extend(quotes)
    branch["findings"].setdefault(relation["id"], []).extend(outcomes)
    kept, conflicts = [], []
    for candidate in branch["candidates"].get(node, []):
        found = verdicts.get(norm_entity(candidate["entity"]), {"yes": [], "no": []})
        if found["yes"] and found["no"]:
            conflicts.append({"relation": relation["id"], "entity": candidate["entity"], "yes": found["yes"], "no": found["no"]})
            kept.append(candidate)
        elif found["no"]:
            continue
        else:
            if found["yes"]:
                candidate["support"].setdefault(relation["id"], []).extend(found["yes"])
            kept.append(candidate)
    branch["candidates"][node] = kept
    branch["conflicts"].extend(conflicts)
    status = "conflict" if conflicts else ("established" if any(relation["id"] in c["support"] for c in kept) else "unresolved")
    branch["status"][relation["id"]] = status
    return status


def node_count(branch: dict[str, Any], node: str) -> int:
    """How many entities a node stands for: one, or a set such as "the two islands"."""
    found = next((n for n in branch["graph"]["nodes"] if n.get("id") == node), {})
    return int(found.get("count") or 1)


def settle_node(branch: dict[str, Any], node: str) -> str:
    """Bind a node when exactly as many candidates as it stands for hold a quote
    for every relation on it.

    Returns "bound", "empty" (too few left), "open" (work remains) or "several"
    (all relations done and more candidates fully supported than the node holds).
    """
    own = [r["id"] for r in relations_on(branch, node)]
    if any(branch["status"].get(rid) in ("open", "dispatched", "conflict") for rid in own):
        return "open"
    full = [c for c in branch["candidates"].get(node, []) if all(rid in c["support"] for rid in own)]
    count = node_count(branch, node)
    if len(full) == count:
        support: dict[str, list] = {}
        for candidate in full:
            for rid, quotes in candidate["support"].items():
                support.setdefault(rid, []).extend(quotes)
        branch["bound"][node] = {"entity": "; ".join(c["entity"] for c in full), "members": [c["entity"] for c in full],
                                 "from": "candidates", "support": support}
        return "bound"
    return "several" if len(full) > count else "empty"


def stuck_node(branch: dict[str, Any]) -> tuple[str, str] | None:
    """The first node, in graph order, whose relations are done without binding it."""
    for node in branch["graph"]["nodes"]:
        if node.get("kind") == "constant" or node["id"] in branch["bound"]:
            continue
        outcome = settle_node(branch, node["id"])
        if outcome in ("empty", "several"):
            statuses = sorted({branch["status"].get(r["id"], "") for r in relations_on(branch, node["id"])})
            return node["id"], f"{outcome} after relations {', '.join(statuses)}"
    return None


def downstream(graph: dict[str, Any], start: set[str]) -> set[str]:
    """The nodes given, and every node whose relations depend on them, transitively."""
    affected, changed = set(start), True
    while changed:
        changed = False
        for relation in graph["relations"]:
            if relation["subject"] not in affected and any(d in affected for d in relation.get("depends_on") or []):
                affected.add(relation["subject"])
                changed = True
    return affected


def _shape(relation: dict[str, Any]) -> tuple[Any, ...]:
    return (relation.get("subject"), relation.get("object") or "", relation.get("kind"),
            relation.get("anchor") is True, tuple(sorted(relation.get("depends_on") or [])))


def _replay(branch: dict[str, Any], relations: list[dict[str, Any]], max_candidates: int, wide_candidates: int) -> None:
    """Rebuild nodes from the findings of relations whose task has not changed,
    each node's anchor first, without researching them again."""
    for relation in sorted(relations, key=lambda r: r.get("anchor") is not True):
        outcomes = branch["findings"].pop(relation["id"], None)
        if outcomes is None:
            branch["status"][relation["id"]] = "open"
        elif acts_as_filter(branch, relation):
            record_filter(branch, relation, outcomes)
        else:
            record_anchor(branch, relation, outcomes, max_candidates, wide_candidates)


def rebase(branch: dict[str, Any], graph: dict[str, Any], node: str, *,
           max_candidates: int = DEFAULTS["max_candidates"],
           wide_candidates: int = DEFAULTS["wide_candidates"]) -> set[str]:
    """Carry a branch onto a revised graph without losing work the change does not reach.

    The stuck node and every node that depends on it are rebuilt. A relation keeps
    its findings when it still links the same nodes the same way and none of what it
    depends on is rebuilt; on the stuck node itself a reworded relation is a new
    task, since the rewording may be the fix. On 2026-11-03 a recheck reworded three
    relations and threw away a lookup already established, and the redone research
    found nothing new.

    A node identified before keeps its identification only while every relation on
    it is kept. A new or changed relation on it must be tested first, so the node
    comes unbound with its candidates, and the nodes that depend on it are rebuilt.
    On 2026-11-03 a recheck gave an identified node new relations; they stayed open
    and untested for the rest of the run, since an identified node's relations are
    never researched. Returns the nodes that are rebuilt or come unbound.
    """
    affected = downstream(graph, {node})
    old = {r["id"]: r for r in branch["graph"]["relations"]}
    constants = {n["id"] for n in graph["nodes"] if n.get("kind") == "constant"}

    def kept(relation: dict[str, Any], moved: set[str]) -> bool:
        before = old.get(relation["id"])
        if before is None or _shape(before) != _shape(relation):
            return False
        if any(d in moved for d in relation.get("depends_on") or []):
            return False
        if relation["subject"] == node:
            return (before.get("relation"), before.get("span")) == (relation.get("relation"), relation.get("span"))
        return True

    unbound: set[str] = set()
    while True:
        keep_ids = {r["id"] for r in graph["relations"] if kept(r, affected | unbound)}
        more = {r["subject"] for r in graph["relations"] if r["id"] not in keep_ids and r["subject"] in branch["bound"]
                and r["subject"] not in constants | affected | unbound}
        if not more:
            break
        unbound |= more
        affected |= downstream(graph, unbound) - unbound
    keep_nodes = {n["id"] for n in graph["nodes"]}
    branch["graph"] = graph
    branch["status"] = {r["id"]: (branch["status"].get(r["id"], "open")
                                  if r["id"] in keep_ids and r["subject"] not in affected else "open")
                        for r in graph["relations"]}
    branch["findings"] = {k: v for k, v in branch["findings"].items() if k in keep_ids}
    branch["bound"] = {k: v for k, v in branch["bound"].items()
                       if k in keep_nodes and k not in affected and k not in unbound}
    for item in graph["nodes"]:
        if item.get("kind") == "constant":
            branch["bound"][item["id"]] = {"entity": str(item.get("value", "")), "from": "question"}
    branch["candidates"] = {k: v for k, v in branch["candidates"].items() if k in keep_nodes and k not in affected}
    for candidates in branch["candidates"].values():
        # Support for a relation that is not kept was given for another task.
        for candidate in candidates:
            candidate["support"] = {rid: q for rid, q in candidate["support"].items() if rid in keep_ids}
    _replay(branch, [r for r in graph["relations"] if r["id"] in keep_ids and r["subject"] in affected],
            max_candidates, wide_candidates)
    branch["conflicts"] = [c for c in branch["conflicts"] if branch["status"].get(c["relation"]) not in (None, "open")]
    branch.setdefault("rebases", []).append({"node": node, "affected": sorted(affected), "unbound": sorted(unbound),
                                             "kept": sorted(r for r in keep_ids if old.get(r))})
    return affected | unbound


def doubt(branch: dict[str, Any], node: str) -> set[str]:
    """Take back a node's identification, when the controller judges it wrong.

    The node keeps its candidates and their support, so a step on it can narrow them
    again; every node that depends on it is rebuilt. A node from the question cannot
    be taken back. Returns the nodes that come unbound.
    """
    if branch["bound"].get(node, {}).get("from", "question") == "question":
        return set()
    graph = branch["graph"]
    after = downstream(graph, {node}) - {node}
    for relation in graph["relations"]:
        if relation["subject"] in after:
            branch["status"][relation["id"]] = "open"
            branch["findings"].pop(relation["id"], None)
    for item in {node} | after:
        if branch["bound"].get(item, {}).get("from") != "question":
            branch["bound"].pop(item, None)
    for item in after:
        branch["candidates"].pop(item, None)
    branch["conflicts"] = [c for c in branch["conflicts"] if branch["status"].get(c["relation"]) != "open"]
    branch.setdefault("doubted", []).append(node)
    return {node} | after


def reopen(branch: dict[str, Any], relation_ids: set[str]) -> set[str]:
    """Open again the relations a final check did not accept.

    Their nodes come unbound and keep their candidates, less these relations'
    support, so each candidate is tested on them again; every node that depends on
    them is rebuilt. Returns the nodes that come unbound.
    """
    graph = branch["graph"]
    by_id = {r["id"]: r for r in graph["relations"]}
    chosen = {rid for rid in relation_ids if rid in by_id}
    subjects = {by_id[rid]["subject"] for rid in chosen}
    after = downstream(graph, subjects) - subjects
    for rid in chosen:
        branch["status"][rid] = "open"
        branch["findings"].pop(rid, None)
        for candidate in branch["candidates"].get(by_id[rid]["subject"], []):
            candidate["support"].pop(rid, None)
    for relation in graph["relations"]:
        if relation["subject"] in after:
            branch["status"][relation["id"]] = "open"
            branch["findings"].pop(relation["id"], None)
    for item in subjects | after:
        if branch["bound"].get(item, {}).get("from") != "question":
            branch["bound"].pop(item, None)
    for item in after:
        branch["candidates"].pop(item, None)
    branch["conflicts"] = [c for c in branch["conflicts"] if branch["status"].get(c["relation"]) != "open"]
    return subjects | after


def answer_of(branch: dict[str, Any]) -> str:
    node = branch["graph"]["answer"]["node"]
    return str(branch["bound"].get(node, {}).get("entity", ""))


def evidence_rows(branch: dict[str, Any]) -> list[dict[str, Any]]:
    """For the final checks: each relation, the entity it was settled for and its quotes."""
    rows = []
    for relation in branch["graph"]["relations"]:
        node = relation["subject"]
        entity = branch["bound"].get(node, {}).get("entity", "")
        support = branch["bound"].get(node, {}).get("support", {}).get(relation["id"], [])
        rows.append({"relation": relation["id"], "span": relation.get("span", ""), "subject": node,
                     "entity": entity, "quotes": support})
    return rows


def research_rows(branch: dict[str, Any], *, per_node: int = 10, per_relation: int = 2) -> list[dict[str, Any]]:
    """Everything a branch established, bound or not: for each relation, its status
    and the entities its quotes support. What an unverified answer can rest on."""
    rows = []
    for relation in branch["graph"]["relations"]:
        node, rid = relation["subject"], relation["id"]
        bound = branch["bound"].get(node)
        if bound and bound.get("support"):
            entities = [{"entity": bound["entity"], "quotes": bound["support"].get(rid, [])[:per_relation]}]
        else:
            entities = [{"entity": c["entity"], "quotes": c["support"].get(rid, [])[:per_relation]}
                        for c in branch["candidates"].get(node, [])][:per_node]
        rows.append({"relation": rid, "span": relation.get("span", ""), "subject": node,
                     "node": describe_node(branch, node), "status": branch["status"].get(rid, "open"),
                     "entities": [e for e in entities if e["quotes"]]})
    return rows


def best_guess(branch: dict[str, Any]) -> str:
    """The program's own pick when no model call can make one: the answer node's
    binding, else its candidate with quotes for the most of its relations."""
    node = branch["graph"]["answer"]["node"]
    if node in branch["bound"]:
        return str(branch["bound"][node].get("entity", ""))
    ranked = sorted(branch["candidates"].get(node, []),
                    key=lambda c: (-len(c["support"]), -sum(len(q) for q in c["support"].values())))
    return str(ranked[0]["entity"]) if ranked and ranked[0]["support"] else ""


def objected_relations(result: dict[str, Any], relation_ids: set[str]) -> dict[str, list[str]]:
    """The relations a final check did not accept, with the reasons its checkers gave."""
    reasons: dict[str, list[str]] = {}

    def add(rid: Any, why: Any = "") -> None:
        if str(rid) in relation_ids:
            reasons.setdefault(str(rid), [])
            if str(why or "").strip():
                reasons[str(rid)].append(str(why).strip()[:300])

    for check in result.get("checks") or []:
        for item in check.get("relations") or []:
            if isinstance(item, dict) and item.get("established") is not True:
                add(item.get("relation"), f"{check.get('checker', 'checker')}: {item.get('reason', '')}")
    for objection in result.get("objections") or []:
        if objection.get("relation"):
            add(objection["relation"])
    for link in (result.get("reverify") or {}).get("links") or []:
        if isinstance(link, dict) and link.get("holds") is not True:
            add(link.get("relation"), "re-verifier: " + str(link.get("reason") or link.get("note") or ""))
    return reasons


def earlier_refuted(replies: list[dict[str, Any] | None], relation_ids: set[str]) -> bool:
    """Whether the evidence contradicts the earlier answer and not the fallback's.

    Both checks must say the earlier answer ("first") conflicts with a researched
    relation, and neither may say so of the fallback's answer ("second"). Only then
    may an unverified fallback answer replace an earlier one.
    """
    if len(replies) < 2 or any(not isinstance(r, dict) for r in replies):
        return False
    for reply in replies:
        first, second = reply.get("first") or {}, reply.get("second") or {}
        if first.get("contradicted") is not True or str(first.get("relation", "")) not in relation_ids:
            return False
        if second.get("contradicted") is not False:
            return False
    return True


# --- Budget, verdict and checkpoints -------------------------------------------------

def budget_used(ledger: dict[str, Any]) -> Decimal:
    """Spent plus everything held or in flight, as the trial's ledger reports it."""
    return sum((Decimal(str(ledger.get(key) or 0)) for key in
                ("spent_estimate_cny", "unknown_held_upper_cny", "inflight_reserved_upper_cny")), Decimal(0))


def research_allowed(ledger: dict[str, Any], hard_cap: Decimal, reserve: Decimal) -> bool:
    """Research stops where the reserve for the final checks begins."""
    return budget_used(ledger) < hard_cap - reserve


def branch_verdict(checks: list[dict[str, Any]], reverify: dict[str, Any] | None,
                   required: list[str]) -> dict[str, Any]:
    """Verified only when both per-relation checks support every required relation
    and the re-verifier confirms. Anything else names who objected and where."""
    objections = []
    for check in checks:
        name = check.get("checker", "")
        if check.get("verdict") != "supported":
            objections.append({"by": name, "verdict": check.get("verdict") or "no_verdict"})
        marks = {item.get("relation"): item for item in check.get("relations") or [] if isinstance(item, dict)}
        for rid in required:
            if (marks.get(rid) or {}).get("established") is not True:
                objections.append({"by": name, "relation": rid})
    if len(checks) < 2:
        objections.append({"by": "program", "verdict": "missing_check"})
    if not reverify or reverify.get("verdict") != "confirmed":
        objections.append({"by": "re_verifier", "verdict": (reverify or {}).get("verdict") or "missing"})
    return {"verified": not objections, "objections": objections}


def combine_branches(results: list[dict[str, Any]]) -> dict[str, Any]:
    """One answer for the question from its branches' results."""
    verified = [r for r in results if r.get("verified") and str(r.get("answer", "")).strip()]
    names = {norm_entity(r["answer"]) for r in verified}
    if len(names) == 1:
        return {"level": "verified", "answer": verified[0]["answer"], "branch": verified[0]["branch"], "reason": ""}
    if len(names) > 1:
        return {"level": "best_effort", "answer": "", "branch": "", "reason": "branches_disagree",
                "answers": [{"branch": r["branch"], "answer": r["answer"]} for r in verified]}
    answered = [r for r in results if str(r.get("answer", "")).strip()]
    if answered:
        return {"level": "best_effort", "answer": answered[0]["answer"], "branch": answered[0]["branch"],
                "reason": "not_verified"}
    return {"level": "none", "answer": "", "branch": "", "reason": "no_answer"}


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomic, so a cut-off leaves the previous checkpoint whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
