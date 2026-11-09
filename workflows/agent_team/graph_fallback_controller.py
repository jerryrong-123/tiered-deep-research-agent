"""Run the relation-graph fallback: the model calls around ``graph_fallback``.

The caller hands in the calls, so the controller runs the same way inside a trial
and under test fakes:
- ``ask_pro(prompt)`` and ``ask_flash(prompt)``: one chat call each, returning text;
- ``run_subagent(card)``: one research sub-agent on a task card, returning
  ``{"report": text, "sources": everything its tools returned}``;
- ``ledger()``: the trial's model ledger, for the reserve kept for the final checks.

Steps, each checkpointed so a resumed run continues where it stopped: three
decompositions, a merge, up to three rounds of checks, branches, research
relation by relation, and the final checks. See the design in
outputs/fallback_controller_design_20261103/DESIGN.md.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable

from workflows.agent_team import graph_fallback as gf
from workflows.agent_team import graph_fallback_prompts as prompts
from workflows.agent_team.answer_settlement import parse_reply

RESEARCH_ANGLES = ("primary", "independent")
JSON_AGAIN = "\n\nYour previous reply was not usable: {why}. Return JSON only, in the shape asked for."


@dataclass
class Calls:
    ask_pro: Callable[[str], Awaitable[str]]
    ask_flash: Callable[[str], Awaitable[str]]
    run_subagent: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    ledger: Callable[[], dict[str, Any]]


def strip(graph: dict[str, Any]) -> dict[str, Any]:
    """The graph proper, without a reply's notes."""
    return {key: graph[key] for key in ("nodes", "relations", "answer") if key in graph}


def _has(key: str) -> Callable[[dict[str, Any]], str]:
    return lambda reply: "" if isinstance(reply.get(key), list) else f'no "{key}" list'


class GraphFallback:
    def __init__(self, question: str, calls: Calls, *, config: dict[str, Any] | None = None,
                 hard_cap: Any = "50", state_path: Path | None = None, earlier_answer: str = "") -> None:
        self.question = question
        # The answer an earlier layer delivered. Research never sees it; it is only
        # compared with the evidence when this layer cannot verify its own answer.
        self.earlier = str(earlier_answer or "").strip()
        self.calls = calls
        self.cfg = gf.settings(config)
        self.hard_cap = Decimal(str(hard_cap))
        self.reserve = Decimal(str(self.cfg["verification_reserve_cny"]))
        self.path = Path(state_path) if state_path else None
        saved = gf.load_checkpoint(self.path) if self.path else None
        self.state: dict[str, Any] = saved if saved and saved.get("question") == question else {
            "question": question, "done": [], "events": [], "replans": 0, "rechecks": 0,
            "calls": {"pro": 0, "flash": 0, "subagents": 0, "reverifiers": 0},
        }
        self.slots = asyncio.Semaphore(int(self.cfg["max_parallel"]))
        self.redo = int(self.cfg["redo_per_decomposition"])

    # --- bookkeeping ------------------------------------------------------------

    def save(self) -> None:
        if self.path:
            gf.save_checkpoint(self.path, self.state)

    def note(self, event: str, **fields: Any) -> None:
        self.state["events"].append({"event": event, **fields})
        self.save()

    def finished(self, step: str) -> bool:
        return step in self.state["done"]

    def mark(self, step: str) -> None:
        self.state["done"].append(step)
        self.save()

    def budget_ok(self) -> bool:
        try:
            ledger = self.calls.ledger() or {}
        except Exception:  # an unreadable ledger must not let research run on
            return False
        return gf.research_allowed(ledger, self.hard_cap, self.reserve)

    def agents_left(self) -> int:
        return int(self.cfg["max_subagents"]) - self.state["calls"]["subagents"]

    # --- model calls ------------------------------------------------------------

    async def ask(self, which: str, prompt: str, *,
                  check: Callable[[dict[str, Any]], str] | None = None) -> dict[str, Any] | None:
        """One call that must return a JSON object, asked again on a failed call or an
        unusable reply. A dropped connection ended the first real trial's decompositions,
        so a failed call is retried like an unusable reply."""
        call = self.calls.ask_pro if which == "pro" else self.calls.ask_flash
        text = prompt
        for attempt in range(1, int(self.cfg["format_retries"]) + 2):
            self.state["calls"][which] += 1
            try:
                reply = parse_reply(await call(text))
            except Exception as exc:
                self.note("model_call_failed", model=which, error=type(exc).__name__, attempt=attempt)
                continue
            why = "no JSON object" if reply is None else (check(reply) if check else "")
            if not why:
                return reply
            text = prompt + JSON_AGAIN.format(why=why)
        self.note("unusable_reply", model=which)
        return None

    async def graph_reply(self, prompt: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """A graph that passes the program checks, redone with the named defects up to the limit."""
        graph = await self.ask("pro", prompt, check=_has("nodes"))
        verdict: dict[str, Any] = {"ok": False, "problems": ["no_reply"]}
        for attempt in range(self.redo + 1):
            if graph is None:
                return None, {"ok": False, "problems": ["no_reply"]}
            verdict = gf.check_graph(self.question, strip(graph), min_coverage_pct=int(self.cfg["min_coverage_pct"]))
            if verdict["ok"] or attempt == self.redo:
                break
            graph = await self.ask("pro", prompts.redo(self.question, strip(graph), verdict["problems"]), check=_has("nodes"))
        return graph, verdict

    async def subagent(self, card: dict[str, Any]) -> dict[str, Any]:
        """One sub-agent; its quotes count only where its own sources hold them."""
        async with self.slots:
            key = "reverifiers" if card.get("role") == "reverifier" else "subagents"
            self.state["calls"][key] += 1
            try:
                result = await self.calls.run_subagent(card)
            except Exception as exc:
                self.note("subagent_failed", agent=card["agent"], error=type(exc).__name__)
                result = {"report": "", "sources": ""}
        reply = parse_reply(result.get("report"))
        if reply is None:
            self.note("subagent_report_unusable", agent=card["agent"])
            return {"agent": card["agent"], "candidates": [], "verdicts": [], "unusable": True}
        if card.get("role") == "reverifier":
            return {**reply, "agent": card["agent"]}
        outcome = gf.keep_sourced({**reply, "agent": card["agent"]}, str(result.get("sources") or ""))
        if outcome["unsourced_quotes_dropped"]:
            self.note("unsourced_quotes_dropped", agent=card["agent"], count=outcome["unsourced_quotes_dropped"])
        return outcome

    def card(self, agent: str, role: str, prompt: str) -> dict[str, Any]:
        return {"agent": agent, "role": role, "prompt": prompt, "max_turns": int(self.cfg["subagent_turns"]),
                "max_queries": int(self.cfg["subagent_queries"])}

    # --- decomposition ----------------------------------------------------------------

    async def decompose(self) -> list[dict[str, Any]]:
        angles = list(prompts.DECOMPOSE_ANGLES)[: int(self.cfg["decompositions"])]
        replies = await asyncio.gather(*(self.graph_reply(prompts.decompose(self.question, a)) for a in angles))
        drafts = []
        self.state["drafts"] = []
        for angle, (graph, verdict) in zip(angles, replies):
            self.state["drafts"].append({"angle": angle, "graph": graph, "check": verdict})
            if graph is not None and verdict["ok"]:
                drafts.append(strip(graph))
        self.save()
        return drafts

    async def merge(self, drafts: list[dict[str, Any]]) -> dict[str, Any]:
        alignment = gf.align_drafts(drafts)
        self.state["alignment"] = alignment
        if len(drafts) == 1:
            self.note("single_draft_no_merge")
            return drafts[0]
        graph, verdict = await self.graph_reply(prompts.merge(self.question, drafts, alignment))
        self.state["merge"] = {"notes": (graph or {}).get("merge_notes"), "check": verdict}
        if graph is None or not verdict["ok"]:
            self.note("merge_failed_first_draft_kept")
            return drafts[0]
        return strip(graph)

    async def review_round(self, graph: dict[str, Any]) -> dict[str, Any]:
        """The four checks: structure, back-translation, and two independent reviews."""
        structural = gf.check_graph(self.question, graph, min_coverage_pct=int(self.cfg["min_coverage_pct"]))
        restated = await self.ask("pro", prompts.restate(graph), check=_has("conditions"))
        conditions = [str(item) for item in (restated or {}).get("conditions") or []]
        gaps = gf.restatement_gaps(self.question, conditions)
        comparison, pro_review, flash_review = await asyncio.gather(
            self.ask("flash", prompts.compare(self.question, conditions, gaps)) if restated else _none(),
            self.ask("pro", prompts.review(self.question, graph)),
            self.ask("flash", prompts.review(self.question, graph)),
        )
        reviews = [({**reply, "reviewer": name} if isinstance(reply, dict) else None)
                   for name, reply in (("back_translation", comparison), ("pro", pro_review), ("flash", flash_review))]
        combined = gf.combine_reviews(reviews)
        problems = [{"kind": "structure", "text": p, "from": ["program"]} for p in structural["problems"]]
        problems += combined["problems"]
        return {"problems": problems, "ambiguities": combined["ambiguities"], "restated": conditions, "gaps": gaps}

    async def settle_graph(self, graph: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
        rounds = int(self.cfg["check_rounds"])
        self.state["check_rounds"] = []
        result: dict[str, Any] = {"problems": [], "ambiguities": []}
        for number in range(1, rounds + 1):
            result = await self.review_round(graph)
            record = {"round": number, "problems": result["problems"], "ambiguities": result["ambiguities"],
                      "restated": result["restated"], "gaps": result["gaps"]}
            self.state["check_rounds"].append(record)
            self.save()
            if not result["problems"]:
                return graph, result["ambiguities"], []
            if number == rounds:
                break
            revised, verdict = await self.graph_reply(prompts.revise(self.question, graph, result["problems"]))
            if revised is not None and verdict["ok"]:
                record["rejected_findings"] = revised.get("rejected_findings")
                graph = strip(revised)
        # Disputes that outlived the rounds are researched as readings, not decided here.
        reply = await self.ask("pro", prompts.readings(self.question, graph, result["problems"], rounds), check=_has("ambiguous"))
        extra = [item for item in (reply or {}).get("ambiguous") or [] if isinstance(item, dict)]
        merged = gf.combine_reviews([{"reviewer": "readings", "ambiguous": result["ambiguities"] + extra}])
        return graph, merged["ambiguities"], result["problems"]

    async def judge(self, ambiguities: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Pro and Flash each choose a reading; only where they agree is no branch needed."""
        chosen, open_ = {}, []
        picked = lambda reply: "" if isinstance(reply.get("reading"), str) else 'no "reading"'
        self.state["judgements"] = []
        for item in ambiguities:
            prompt = prompts.judge(self.question, item)
            pro, flash = await asyncio.gather(self.ask("pro", prompt, check=picked), self.ask("flash", prompt, check=picked))
            picks = [(pro or {}).get("reading"), (flash or {}).get("reading")]
            reading = gf.agreed_reading(item["readings"], picks)
            self.state["judgements"].append({"span": item["span"], "readings": item["readings"], "pro": picks[0],
                                             "flash": picks[1], "chosen": reading})
            if reading:
                chosen[item["span"]] = reading
            else:
                open_.append(item)
        self.save()
        return chosen, open_

    async def branches(self, graph: dict[str, Any], ambiguities: list[dict[str, Any]],
                       chosen: dict[str, str] | None = None) -> list[dict[str, Any]] | None:
        plans = gf.plan_branches(ambiguities, int(self.cfg["max_branches"]))
        if plans is None:
            return None
        out = []
        for plan in plans:
            readings = {**(chosen or {}), **plan["readings"]}
            if not readings:
                out.append(gf.new_branch(plan["id"], graph))
                continue
            rewritten, verdict = await self.graph_reply(prompts.branch_graph(self.question, graph, readings))
            if rewritten is not None and verdict["ok"]:
                out.append(gf.new_branch(plan["id"], strip(rewritten), readings))
            else:
                self.note("branch_graph_failed", branch=plan["id"], problems=verdict["problems"])
        return out

    # --- research ---------------------------------------------------------------------

    def limits(self) -> dict[str, int]:
        return {"max_candidates": int(self.cfg["max_candidates"]), "wide_candidates": int(self.cfg["wide_candidates"])}

    async def dispatch(self, cards: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Run cards within the sub-agent cap; one more try when every report was unusable."""
        cards = cards[: max(0, self.agents_left())]
        if not cards:
            return None
        outcomes = list(await asyncio.gather(*(self.subagent(card) for card in cards)))
        if all(outcome.get("unusable") for outcome in outcomes) and self.agents_left() > 0:
            outcomes.append(await self.subagent(dict(cards[-1], agent=cards[-1]["agent"] + "_again")))
        return outcomes

    async def research_relation(self, branch: dict[str, Any], relation: dict[str, Any], *, unbound: bool = False) -> str:
        rid = relation["id"]
        inputs = gf.task_inputs(branch, relation, unbound=unbound)
        testing = gf.acts_as_filter(branch, relation)
        # A relation a final check did not accept is researched again, never reused.
        reasons = branch.setdefault("repair_notes", {}).pop(rid, None)
        signature = gf.relation_signature(branch, relation, inputs)
        shared = self.state.setdefault("shared", {})
        if reasons is None and signature in shared:
            # The same task was researched in an earlier branch; its findings carry over.
            outcomes = copy.deepcopy(shared[signature]["outcomes"])
            self.note("relation_shared", branch=branch["id"], relation=rid, source=shared[signature]["from"])
        else:
            limit = int(self.cfg["max_candidates"])
            candidates, batch = inputs["candidates"], int(self.cfg["wide_batch"])
            if testing and len(candidates) > limit:
                # A long list is tested in batches, one sub-agent each.
                cards = [self.card(f"{branch['id']}_{rid}_batch{n}", "research",
                                   prompts.task(relation, {**inputs, "candidates": candidates[start:start + batch]},
                                                RESEARCH_ANGLES[(n - 1) % len(RESEARCH_ANGLES)], limit,
                                                as_filter=True, reasons=reasons))
                         for n, start in enumerate(range(0, len(candidates), batch), 1)]
            else:
                cards = [self.card(f"{branch['id']}_{rid}_{angle}", "research",
                                   prompts.task(relation, inputs, angle, limit, as_filter=testing, reasons=reasons))
                         for angle in RESEARCH_ANGLES[: int(self.cfg["agents_per_relation"])]]
            outcomes = await self.dispatch(cards)
            if outcomes is None:
                return "no_agents_left"
            shared[signature] = {"from": f"{branch['id']}:{rid}", "outcomes": copy.deepcopy(outcomes)}
        if not testing:
            return gf.record_anchor(branch, relation, outcomes, **self.limits())
        status = gf.record_filter(branch, relation, outcomes)
        return await self.arbitrate(branch, relation) if status == "conflict" else status

    async def arbitrate(self, branch: dict[str, Any], relation: dict[str, Any]) -> str:
        rid, subject = relation["id"], relation["subject"]
        for number, conflict in enumerate([c for c in branch["conflicts"] if c["relation"] == rid and not c.get("resolved")], 1):
            if self.agents_left() <= 0 or int(self.cfg["extra_agents_per_conflict"]) <= 0:
                break
            outcome = await self.subagent(self.card(f"{branch['id']}_{rid}_arbiter{number}", "arbiter",
                                                    prompts.arbitrate(relation, conflict)))
            key = gf.norm_entity(conflict["entity"])
            verdict = next((v for v in outcome.get("verdicts") or [] if gf.norm_entity(v.get("entity", "")) == key), None)
            conflict["arbiter"] = verdict
            if verdict and verdict.get("quotes") and verdict.get("satisfies") in (True, False):
                conflict["resolved"] = True
                candidates = branch["candidates"].get(subject, [])
                if verdict["satisfies"] is True:
                    for candidate in candidates:
                        if gf.norm_entity(candidate["entity"]) == key:
                            candidate["support"].setdefault(rid, []).extend(verdict["quotes"])
                else:
                    branch["candidates"][subject] = [c for c in candidates if gf.norm_entity(c["entity"]) != key]
        open_conflict = any(c["relation"] == rid and not c.get("resolved") for c in branch["conflicts"])
        supported = any(rid in c["support"] for c in branch["candidates"].get(subject, []))
        status = "conflict_unresolved" if open_conflict else ("established" if supported else "unresolved")
        branch["status"][rid] = status
        return status

    async def research(self, branch: dict[str, Any]) -> None:
        answer_node = branch["graph"]["answer"]["node"]
        while answer_node not in branch["bound"]:
            if not self.budget_ok():
                self.note("research_budget_reached", branch=branch["id"])
                return
            if self.agents_left() <= 0:
                self.note("subagent_cap_reached", branch=branch["id"])
                return
            batch, subjects = [], set()
            for relation in gf.ready_relations(branch):
                if relation["subject"] not in subjects and len(batch) < int(self.cfg["max_parallel"]):
                    subjects.add(relation["subject"])
                    batch.append(relation)
            if batch:
                for relation in batch:
                    branch["status"][relation["id"]] = "dispatched"
                statuses = await asyncio.gather(*(self.research_relation(branch, r) for r in batch))
                for relation, status in zip(batch, statuses):
                    if status == "no_agents_left":
                        branch["status"][relation["id"]] = "open"
                    settled = gf.settle_node(branch, relation["subject"])
                    self.note("relation_researched", branch=branch["id"], relation=relation["id"], status=status,
                              node=relation["subject"], node_state=settled)
                continue
            # A node whose last relation just finished may bind now and free its dependents.
            if any(gf.settle_node(branch, n["id"]) == "bound" for n in branch["graph"]["nodes"]
                   if n.get("kind") != "constant" and n["id"] not in branch["bound"]):
                continue
            # Nothing can go on as planned: a node waits on one no one can find, or its
            # relations are done without binding it. The controller chooses what next.
            if await self.next_step(branch):
                continue
            stuck = gf.stuck_node(branch)
            if stuck is None or not await self.unstick(branch, *stuck):
                return

    # --- choosing the next step -----------------------------------------------------

    def picture(self, branch: dict[str, Any]) -> str:
        """Where a branch's research stands, for the controller's choice."""
        lines = []
        for node in branch["graph"]["nodes"]:
            nid = node["id"]
            if nid in branch["bound"]:
                state = f'identified: {branch["bound"][nid].get("entity", "")}'
            elif branch["candidates"].get(nid):
                names = [c["entity"] for c in branch["candidates"][nid]]
                state = f"{len(names)} candidates: " + "; ".join(names[:10]) + ("; ..." if len(names) > 10 else "")
            else:
                state = "no candidates"
            lines.append(f'{nid} ({node.get("kind")}: {node.get("description", "")}): {state}')
        for relation in branch["graph"]["relations"]:
            deps = ", ".join(f'{d} ({"known" if d in branch["bound"] else "not known"})'
                             for d in relation.get("depends_on") or []) or "nothing"
            lines.append(f'{relation["id"]} on {relation["subject"]} -> {relation.get("object") or "-"}, '
                         f'"{relation.get("span", "")}", {"anchor, " if relation.get("anchor") is True else ""}'
                         f'depends on {deps}: {branch["status"].get(relation["id"], "open")}')
        return "\n".join(lines)

    def step_problem(self, branch: dict[str, Any], reply: dict[str, Any]) -> str:
        """Why a chosen step cannot run on this branch, or "" when it can."""
        nodes = {n["id"]: n for n in branch["graph"]["nodes"]}
        relations = {r["id"]: r for r in branch["graph"]["relations"]}
        action = reply.get("action")
        if action == "stop":
            return ""
        # A step on an identified node takes the identification back, once per node.
        again = lambda nid: nid in branch["bound"] and nid in (branch.get("doubted") or [])
        if action in ("merge_clues", "revise_graph"):
            node = nodes.get(str(reply.get("node")))
            if node is None or node.get("kind") == "constant":
                return f'"node" must be a node that is not a constant of the question, for {action}'
            if again(node["id"]):
                return f'{node["id"]} was taken back once and identified again; choose a step on another node'
            return ""
        if action in ("test_candidates", "reverse"):
            relation = relations.get(str(reply.get("relation")))
            if relation is None or not branch["candidates"].get(relation["subject"]):
                return f'"relation" must be a relation whose node has candidates, for {action}'
            if again(relation["subject"]):
                return f'{relation["subject"]} was taken back once and identified again; choose a step on another node'
            if action == "test_candidates" and branch["status"].get(relation["id"]) == "established":
                return "that relation is already established"
            other = str(relation.get("object") or "")
            if action == "reverse" and (other not in nodes or other in branch["bound"] or nodes[other].get("kind") == "constant"):
                return "reverse needs a relation whose other node is not yet identified"
            return ""
        return 'unknown "action"'

    async def next_step(self, branch: dict[str, Any]) -> bool:
        """One step the controller chooses; False when it stops or has no choices left."""
        taken = branch.setdefault("steps", [])
        if len(taken) >= int(self.cfg["next_steps"]) or self.agents_left() <= 0 or not self.budget_ok():
            return False
        reply = await self.ask("pro", prompts.next_step(self.question, self.picture(branch), taken),
                               check=lambda r: self.step_problem(branch, r))
        if reply is None:
            return False
        step = {k: reply.get(k) for k in ("action", "node", "relation", "why") if reply.get(k)}
        taken.append(step)
        self.note("next_step", branch=branch["id"], action=step.get("action"),
                  node=step.get("node", ""), relation=step.get("relation", ""))
        action = reply["action"]
        if action == "stop":
            return False
        relation = next((r for r in branch["graph"]["relations"] if r["id"] == str(reply.get("relation"))), None)
        target = str(reply["node"]) if action in ("merge_clues", "revise_graph") else relation["subject"]
        if target in branch["bound"]:
            self.note("doubted", branch=branch["id"], node=target, unbound=sorted(gf.doubt(branch, target)))
        if action == "revise_graph":
            return await self.unstick(branch, target, str(reply.get("why") or "chosen by the controller"))
        if action == "merge_clues":
            status = await self.merge_clues(branch, str(reply["node"]))
        elif action == "reverse":
            status = await self.reverse(branch, relation)
        else:
            branch["status"][relation["id"]] = "dispatched"
            status = await self.research_relation(branch, relation, unbound=True)
            if status == "no_agents_left":
                branch["status"][relation["id"]] = "open"
        self.note("step_done", branch=branch["id"], action=action, status=status)
        for node in branch["graph"]["nodes"]:
            if node.get("kind") != "constant" and node["id"] not in branch["bound"]:
                gf.settle_node(branch, node["id"])
        return status != "no_agents_left"

    async def merge_clues(self, branch: dict[str, Any], node: str) -> str:
        """All of a node's relations in one search, their clues combined."""
        own = gf.relations_on(branch, node)
        deps = {d for r in own for d in r.get("depends_on") or []}
        bound = {d: branch["bound"][d]["entity"] for d in deps if d in branch["bound"]}
        described = {d: gf.describe_node(branch, d) for d in deps if d not in branch["bound"]}
        cards = [self.card(f"{branch['id']}_{node}_merged_{angle}", "research",
                           prompts.node_task(gf.describe_node(branch, node), own, bound, described, angle,
                                             int(self.cfg["max_candidates"])))
                 for angle in RESEARCH_ANGLES[: int(self.cfg["agents_per_relation"])]]
        outcomes = await self.dispatch(cards)
        if outcomes is None:
            return "no_agents_left"
        return gf.record_node(branch, node, outcomes, int(self.cfg["wide_candidates"]))

    async def reverse(self, branch: dict[str, Any], relation: dict[str, Any]) -> str:
        """From a node's candidates, find what a relation links each one to."""
        other = str(relation.get("object") or "")
        names = [c["entity"] for c in branch["candidates"].get(relation["subject"], [])]
        batch = int(self.cfg["wide_batch"])
        cards = [self.card(f"{branch['id']}_{relation['id']}_reverse{n}", "research",
                           prompts.reverse_task(relation, gf.describe_node(branch, other), names[start:start + batch],
                                                RESEARCH_ANGLES[(n - 1) % len(RESEARCH_ANGLES)]))
                 for n, start in enumerate(range(0, len(names), batch), 1)]
        outcomes = await self.dispatch(cards)
        if outcomes is None:
            return "no_agents_left"
        return gf.record_reverse(branch, relation, outcomes, int(self.cfg["wide_candidates"]))

    async def unstick(self, branch: dict[str, Any], node_id: str, why: str) -> bool:
        """Check the decomposition again, then replan, before giving the node up."""
        node = next(n for n in branch["graph"]["nodes"] if n["id"] == node_id)
        branch["stuck"].append({"node": node_id, "why": why})
        if self.state["rechecks"] < int(self.cfg["rechecks_during_run"]):
            self.state["rechecks"] += 1
            result = await self.review_round(branch["graph"])
            problems = result["problems"] + [{"kind": "stuck", "text": f"node {node_id}: {why}", "from": ["research"]}]
            revised, verdict = await self.graph_reply(prompts.revise(self.question, branch["graph"], problems))
            if revised is not None and verdict["ok"] and strip(revised) != branch["graph"]:
                affected = gf.rebase(branch, strip(revised), node_id, **self.limits())
                self.note("rechecked", branch=branch["id"], node=node_id, restarted=sorted(affected))
                return True
        if self.state["replans"] < int(self.cfg["replans"]):
            self.state["replans"] += 1
            findings = {rid: {"status": status, "candidates": [c["entity"] for c in branch["candidates"].get(
                next((r["subject"] for r in branch["graph"]["relations"] if r["id"] == rid), ""), [])]}
                for rid, status in branch["status"].items()}
            replanned, verdict = await self.graph_reply(prompts.replan(self.question, branch["graph"], node, why, findings))
            if replanned is not None and verdict["ok"]:
                affected = gf.rebase(branch, strip(replanned), node_id, **self.limits())
                self.note("replanned", branch=branch["id"], node=node_id, restarted=sorted(affected))
                return True
        self.note("stuck", branch=branch["id"], node=node_id, why=why)
        return False

    # --- final checks -------------------------------------------------------------------

    async def final_check(self, branch: dict[str, Any]) -> dict[str, Any]:
        answer = gf.answer_of(branch)
        if not answer:
            return {"branch": branch["id"], "answer": "", "verified": False,
                    "objections": [{"by": "program", "verdict": "no_answer"}]}
        rows = gf.evidence_rows(branch)
        required = [row["relation"] for row in rows]
        prompt = prompts.check(self.question, answer, rows, branch.get("readings"))
        pro, flash = await asyncio.gather(
            self.ask("pro", prompt, check=_has("relations")),
            self.ask("flash", prompt, check=_has("relations")),
        )
        checks = [{**reply, "checker": name} for name, reply in (("pro", pro), ("flash", flash)) if isinstance(reply, dict)]
        reverify = None
        if not gf.branch_verdict(checks, {"verdict": "confirmed"}, required)["objections"]:
            # Only an answer both checks support is worth re-verifying.
            reverify = await self.subagent(self.card(f"{branch['id']}_reverifier", "reverifier",
                                                     prompts.reverify(self.question, answer, rows)))
        verdict = gf.branch_verdict(checks, reverify, required)
        return {"branch": branch["id"], "answer": answer, "verified": verdict["verified"],
                "objections": verdict["objections"], "checks": checks, "reverify": reverify}

    async def checked_and_repaired(self, branch: dict[str, Any]) -> dict[str, Any]:
        """The final checks, and while they object, research again what they did not
        accept and check again. The standard never changes: only an answer every
        check accepts is verified."""
        result = await self.final_check(branch)
        relation_ids = {r["id"] for r in branch["graph"]["relations"]}
        rounds = 0
        while not result["verified"] and result["answer"] and rounds < int(self.cfg["repair_rounds"]):
            objected = gf.objected_relations(result, relation_ids)
            if not objected:
                self.note("repair_nothing_named", branch=branch["id"])
                break
            if not self.budget_ok() or self.agents_left() <= 0:
                self.note("repair_budget_reached", branch=branch["id"])
                break
            rounds += 1
            self.state["repairs"] = self.state.get("repairs", 0) + 1
            gf.reopen(branch, set(objected))
            branch.setdefault("repair_notes", {}).update(objected)
            self.note("repair", branch=branch["id"], round=rounds, relations=sorted(objected))
            await self.research(branch)
            result = await self.final_check(branch)
        result["repair_rounds"] = rounds
        return result

    # --- an answer when nothing verified one ------------------------------------------

    async def forced_answer(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """The best answer the research supports, from Pro as the team's head; else the
        program's own pick; else none. FAT's coordinator likewise always delivers."""
        branches = [{"branch": b["id"], "readings": b.get("readings") or {},
                     "answer_found": (self.state.get("results") or {}).get(b["id"], {}).get("answer", ""),
                     "relations": gf.research_rows(b)} for b in self.state["branches"]]
        form = str(self.state["branches"][0]["graph"]["answer"].get("form", "")) if self.state["branches"] else ""
        # Asked even when the research names no candidate: the head then reasons to the
        # likeliest answer and says it rests on knowledge. On 2026-11-03 a prompt that
        # allowed an empty answer left a forced answer empty after the research.
        if branches:
            reply = await self.ask("pro", prompts.fallback(self.question, form, branches),
                                   check=lambda r: "" if isinstance(r.get("answer"), str) else 'no "answer"')
            if reply and str(reply.get("answer") or "").strip():
                known = {b["id"] for b in self.state["branches"]}
                return {"answer": str(reply["answer"]).strip(), "source": "pro_synthesis",
                        "basis": "knowledge" if reply.get("basis") == "knowledge" else "evidence",
                        "branch": str(reply.get("branch")) if str(reply.get("branch")) in known else "",
                        "established": [str(x) for x in reply.get("established") or []][:20],
                        "unverified": [str(x) for x in reply.get("unverified") or []][:20]}
        for branch in self.state["branches"]:
            guess = gf.best_guess(branch)
            if guess:
                return {"answer": guess, "source": "program_guess", "basis": "evidence", "branch": branch["id"],
                        "established": [], "unverified": []}
        return {"answer": "", "source": "", "basis": "", "branch": "", "established": [], "unverified": []}

    async def refuted(self, answer: str) -> bool:
        """Whether both checks find the earlier answer contradicted by the research and
        this answer not; rule (a) of V2.6."""
        if not self.earlier or not answer or gf.norm_entity(self.earlier) == gf.norm_entity(answer):
            return False
        rows = [{"branch": b["id"], **row} for b in self.state["branches"] for row in gf.research_rows(b) if row["entities"]]
        if not rows:
            return False
        prompt = prompts.refute(self.question, self.earlier, answer, rows)
        both = lambda r: "" if isinstance(r.get("first"), dict) and isinstance(r.get("second"), dict) else 'no "first" and "second"'
        replies = await asyncio.gather(self.ask("pro", prompt, check=both), self.ask("flash", prompt, check=both))
        self.state["refutation"] = [r for r in replies]
        return gf.earlier_refuted(list(replies), {row["relation"] for row in rows})

    async def close_unverified(self, combined: dict[str, Any]) -> dict[str, Any]:
        forced = self.state.get("forced")
        if forced is None:
            forced = await self.forced_answer(list((self.state.get("results") or {}).values()))
            forced["earlier_refuted"] = await self.refuted(forced["answer"])
            self.state["forced"] = forced
            self.mark("forced")
        level = "best_effort" if forced["answer"] else "none"
        return self.finish(level, combined["reason"] or "not_verified", forced["answer"], branch=forced["branch"],
                           forced=forced["source"], basis=forced.get("basis", ""),
                           established=forced["established"], unverified=forced["unverified"],
                           earlier_refuted=forced["earlier_refuted"], branches=len(self.state["branches"]),
                           unresolved_decomposition=len(self.state.get("unresolved") or []))

    # --- the run ------------------------------------------------------------------------

    def finish(self, level: str, reason: str, answer: str = "", **extra: Any) -> dict[str, Any]:
        settlement = {"kind": "graph_fallback", "level": level, "reason": reason,
                      "check": "supported" if level == "verified" else ("insufficient" if answer else ""), **extra}
        self.state["settlement"] = settlement
        self.state["answer"] = answer
        if not self.finished("finished"):
            self.mark("finished")
        return {"answer": answer, "settlement": settlement, "state": self.state}

    async def run(self) -> dict[str, Any]:
        if self.finished("finished"):
            return {"answer": self.state.get("answer", ""), "settlement": self.state["settlement"], "state": self.state}
        if not self.finished("decomposed"):
            drafts = await self.decompose()
            if not drafts:
                return self.finish("none", "decomposition_failed")
            self.state["merged"] = await self.merge(drafts)
            self.mark("decomposed")
        if not self.finished("checked"):
            graph, ambiguities, unresolved = await self.settle_graph(self.state["merged"])
            self.state.update(graph=graph, ambiguities=ambiguities, unresolved=unresolved)
            self.mark("checked")
        if not self.finished("judged"):
            chosen, open_ = await self.judge(self.state["ambiguities"])
            self.state.update(chosen_readings=chosen, open_ambiguities=open_)
            self.mark("judged")
        if not self.finished("branched"):
            branches = await self.branches(self.state["graph"], self.state["open_ambiguities"],
                                           self.state["chosen_readings"])
            if branches is None:
                return self.finish("none", "too_many_branches")
            if not branches:
                return self.finish("none", "no_branch_graph")
            self.state["branches"] = branches
            self.mark("branched")
        for branch in self.state["branches"]:
            if not self.finished("researched:" + branch["id"]):
                await self.research(branch)
                self.mark("researched:" + branch["id"])
        results = self.state.setdefault("results", {})
        for branch in self.state["branches"]:
            if not self.finished("final_checked:" + branch["id"]):
                results[branch["id"]] = await self.checked_and_repaired(branch)
                self.mark("final_checked:" + branch["id"])
        combined = gf.combine_branches(list(results.values()))
        if combined["level"] != "verified":
            return await self.close_unverified(combined)
        return self.finish(combined["level"], combined["reason"], combined["answer"], branch=combined["branch"],
                           branches=len(self.state["branches"]),
                           unresolved_decomposition=len(self.state.get("unresolved") or []))


async def _none() -> None:
    return None
