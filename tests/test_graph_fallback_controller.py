"""The relation-graph fallback controller end to end, with fake models and sub-agents."""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from workflows.agent_team import graph_fallback as gf
from workflows.agent_team.graph_fallback_controller import Calls, GraphFallback

QUESTION = ("A museum that opened in 1950 stands beside a bridge built in the 1930s. "
            "The bridge links two islands. Which of the two islands is larger?")

GRAPH = {
    "nodes": [
        {"id": "M", "kind": "unknown", "description": "the museum"},
        {"id": "B", "kind": "unknown", "description": "the bridge"},
        {"id": "I", "kind": "unknown", "description": "the two islands", "count": 2},
        {"id": "A", "kind": "answer", "description": "the larger island"},
        {"id": "Y", "kind": "constant", "description": "opening year", "value": "1950"},
    ],
    "relations": [
        {"id": "r1", "span": "A museum that opened in 1950", "subject": "M", "relation": "museum opened in",
         "object": "Y", "kind": "lookup", "depends_on": ["Y"], "anchor": True},
        {"id": "r2", "span": "stands beside a bridge built in the 1930s", "subject": "B", "relation": "bridge beside",
         "object": "M", "kind": "lookup", "depends_on": ["M"], "anchor": True},
        {"id": "r3", "span": "built in the 1930s", "subject": "B", "relation": "bridge built in", "object": "",
         "kind": "filter", "depends_on": [], "anchor": False},
        {"id": "r4", "span": "The bridge links two islands", "subject": "I", "relation": "islands linked by",
         "object": "B", "kind": "lookup", "depends_on": ["B"], "anchor": True},
        {"id": "r5", "span": "Which of the two islands is larger?", "subject": "A", "relation": "larger island of",
         "object": "I", "kind": "compare", "depends_on": ["I"], "anchor": True},
    ],
    "answer": {"node": "A", "form": "the island's name"},
}

SOURCES = {
    "museum opened in": ("Harbour Museum", "The Harbour Museum opened in 1950 by the old port."),
    "bridge beside": ("Old Bridge", "The Old Bridge stands beside the Harbour Museum."),
    "islands linked by": (["North Isle", "South Isle"], "The Old Bridge links North Isle and South Isle across the strait."),
    "larger island of": ("North Isle", "North Isle covers 40 square kilometres, more than South Isle."),
}


class Fake:
    """Answers each prompt by its opening words; counts what was asked."""

    def __init__(self, *, ambiguous=None, answers=None, decompose_graph=None, quote_in_sources=True, judges_agree=False,
                 extra=None, steps=None, rejected=(), reject_rounds=0, refute=None, refute_flash=None,
                 fallback="North Isle", basis="evidence", misled=None):
        self.asked: list[str] = []
        self.ambiguous = ambiguous or []
        self.answers = answers or {}
        self.decompose_graph = decompose_graph
        self.quote_in_sources = quote_in_sources
        self.judges_agree = judges_agree
        # More candidates a lookup finds besides the right one, by relation name.
        self.extra = extra or {}
        self.steps = list(steps or [])
        self.rejected, self.reject_rounds = set(rejected), reject_rounds
        no = {"contradicted": False, "relation": "", "reason": ""}
        self.refute = refute or (no, no)
        self.refute_flash = refute_flash or self.refute
        self.fallback, self.basis = fallback, basis
        # Wrong entities a single-relation search finds, by relation name; a search
        # handed one of them finds nothing, and a merged search finds the right one.
        self.misled = misled or {}
        self.cards: list[dict] = []
        self.checks: list[str] = []

    async def flash(self, prompt: str) -> str:
        return await self.model(prompt, flash=True)

    async def model(self, prompt: str, flash: bool = False) -> str:
        first = prompt.strip().splitlines()[0]
        self.asked.append(first[:40])
        if first.startswith("Research on a relation graph built from"):
            return json.dumps(self.steps.pop(0) if self.steps else {"action": "stop", "why": "nothing left"})
        if first.startswith("The research below did not verify"):
            return json.dumps({"answer": self.fallback, "basis": self.basis, "branch": "b1", "established": ["r1", "r2"],
                               "unverified": ["r5"], "reason": "the size comparison is not quoted"})
        if first.startswith("Two answers to the research question"):
            pair = self.refute_flash if flash else self.refute
            return json.dumps({"first": pair[0], "second": pair[1]})
        if first.startswith("Decompose"):
            return json.dumps(self.decompose_graph if self.decompose_graph is not None else GRAPH)
        if first.startswith("Rewrite the relation graph"):
            # A reading changes only the relation it concerns, as a real rewrite would.
            rewritten = json.loads(json.dumps(GRAPH))
            if "by population" in prompt:
                rewritten["relations"][4]["relation"] = "larger island of by population"
            return json.dumps(rewritten)
        if first.startswith(("Your relation graph failed", "Three independent", "The relation graph below",
                             "Research on this relation graph")):
            return json.dumps(self.decompose_graph if self.decompose_graph is not None else GRAPH)
        if first.startswith("Words in the research question"):
            readings = json.loads(prompt.split("Readings: ", 1)[1].split("\nIf the wording", 1)[0])
            pick = readings[0] if self.judges_agree or not flash else readings[-1]
            return json.dumps({"reading": pick, "why": "wording"})
        if first.startswith("Below is a relation graph"):
            return json.dumps({"conditions": ["a museum opened in 1950", "it stands beside a bridge built in the 1930s",
                                              "the bridge links two islands", "which of the two islands is larger"]})
        if first.startswith("A question and a list"):
            return json.dumps({"missing": [], "extra": [], "misread": []})
        if first.startswith("Review this relation graph"):
            return json.dumps({"missing": [], "extra": [], "misread": [], "ambiguous": self.ambiguous})
        if first.startswith("After"):
            return json.dumps({"ambiguous": []})
        if first.startswith("You are checking an answer"):
            self.checks.append(prompt)
            ids = re.findall(r'"relation": "(r\d)"', prompt)
            # Pro and Flash check each round, so a round is two checks.
            rejected = self.rejected if (len(self.checks) + 1) // 2 <= self.reject_rounds else set()
            return json.dumps({"verdict": "insufficient" if rejected & set(ids) else "supported",
                               "relations": [{"relation": i, "established": i not in rejected,
                                              "reason": "the quote does not compare the islands" if i in rejected else ""}
                                             for i in ids]})
        raise AssertionError("unexpected prompt: " + first)

    def found(self, name: str, branch: str) -> tuple[list[str], str]:
        """The right entities for a relation, and the sentence that states them."""
        if name == "bridge built in":
            return ["Old Bridge"], "The Old Bridge was built in 1934."
        key = "larger island of" if name.startswith("larger island of") else name
        entity, sentence = SOURCES[key]
        if key == "larger island of":
            entity = self.answers.get(branch, entity)
        return (entity if isinstance(entity, list) else [entity]), sentence

    async def subagent(self, card: dict) -> dict:
        self.cards.append(card)
        prompt = card["prompt"]
        branch = card["agent"].split("_")[0]
        if card["role"] == "reverifier":
            return {"report": 'Done. {"verdict": "confirmed", "links": []}', "sources": ""}
        if prompt.startswith("You research ONE unknown"):
            # A merged search: the right entity, each quote naming its relation.
            quotes, sources = [], []
            for rid, name in re.findall(r"^- (r\d): (.+?) \(", prompt, re.M):
                entities, sentence = self.found(name, branch)
                quotes.append({"relation": rid, "quote": sentence, "url": "u"})
                sources.append(sentence)
            report = {"candidates": [{"entity": entities[0], "quotes": quotes}]}
            return {"report": json.dumps(report), "sources": " ".join(sources) if self.quote_in_sources else ""}
        name = re.search(r"Relation: (.+)", prompt).group(1).strip()
        entities, sentence = self.found(name, branch)
        if name not in self.misled and any(e in prompt for wrong in self.misled.values() for e in wrong):
            return {"report": json.dumps({"candidates": []}), "sources": ""}
        entities = self.misled.get(name, entities)
        if "Candidates to test:" in prompt:
            listed = json.loads(prompt.split("Candidates to test: ", 1)[1].split("\nFor each candidate", 1)[0])
            verdicts, sources = [], [sentence]
            for candidate in listed:
                if candidate in entities:
                    verdicts.append({"entity": candidate, "satisfies": True, "quotes": [{"quote": sentence, "url": "u"}]})
                else:
                    denial = f"{candidate} does not fit: it dates from 1990."
                    verdicts.append({"entity": candidate, "satisfies": False, "quotes": [{"quote": denial, "url": "u"}]})
                    sources.append(denial)
            return {"report": json.dumps({"verdicts": verdicts}), "sources": " ".join(sources) if self.quote_in_sources else ""}
        everything = entities + list(self.extra.get(name, []))
        report = {"candidates": [{"entity": e, "quotes": [{"quote": sentence, "url": "u"}]} for e in everything]}
        return {"report": "Findings. " + json.dumps(report), "sources": sentence if self.quote_in_sources else "nothing"}


def calls(fake: Fake, ledger=None) -> Calls:
    return Calls(ask_pro=fake.model, ask_flash=fake.flash, run_subagent=fake.subagent,
                 ledger=lambda: ledger or {"spent_estimate_cny": "1"})


def run(controller: GraphFallback) -> dict:
    return asyncio.run(controller.run())


def test_the_full_flow_verifies_an_answer_every_step_supports(tmp_path):
    fake = Fake()
    result = run(GraphFallback(QUESTION, calls(fake), state_path=tmp_path / "graph_v1.json"))
    assert result["settlement"]["level"] == "verified", result["state"]["events"][-5:]
    assert result["answer"] == "North Isle"
    state = result["state"]
    assert state["done"][:4] == ["decomposed", "checked", "judged", "branched"]
    assert sum(a.startswith("Decompose") for a in fake.asked) == 3
    assert state["alignment"]["in_all"] == 5
    assert state["calls"]["subagents"] == 10 and state["calls"]["reverifiers"] == 1
    assert {c["agent"].rsplit("_", 1)[-1] for c in fake.cards if c["role"] == "research"} == {"primary", "independent"}
    assert json.loads((tmp_path / "graph_v1.json").read_text(encoding="utf-8"))["answer"] == "North Isle"


def test_drafts_that_never_pass_the_checks_end_without_an_answer():
    broken = dict(GRAPH, answer={"node": "A", "form": ""})
    fake = Fake(decompose_graph=broken)
    result = run(GraphFallback(QUESTION, calls(fake)))
    assert result["settlement"] == {"kind": "graph_fallback", "level": "none", "reason": "decomposition_failed", "check": ""}
    assert sum(a.startswith("Decompose") for a in fake.asked) == 3
    assert sum(a.startswith("Your relation graph failed") for a in fake.asked) == 6


TWO_READINGS = [{"span": "is larger", "readings": ["by area", "by population"]}]


def test_branches_that_verify_different_answers_are_closed_by_the_heads_best_answer():
    fake = Fake(ambiguous=TWO_READINGS, answers={"b1": "North Isle", "b2": "South Isle"})
    result = run(GraphFallback(QUESTION, calls(fake)))
    assert len(result["state"]["branches"]) == 2
    settlement = result["settlement"]
    assert (settlement["level"], settlement["reason"], settlement["forced"]) == ("best_effort", "branches_disagree", "pro_synthesis")
    assert result["answer"] == "North Isle" and settlement["earlier_refuted"] is False


def test_branches_that_verify_the_same_answer_agree():
    fake = Fake(ambiguous=TWO_READINGS)
    result = run(GraphFallback(QUESTION, calls(fake)))
    assert [r["verified"] for r in result["state"]["results"].values()] == [True, True]
    assert result["settlement"]["level"] == "verified" and result["answer"] == "North Isle"


def test_a_later_branch_reuses_what_an_earlier_one_found_and_researches_only_what_differs():
    fake = Fake(ambiguous=TWO_READINGS)
    state = run(GraphFallback(QUESTION, calls(fake)))["state"]
    shared = [e for e in state["events"] if e["event"] == "relation_shared"]
    assert [e["relation"] for e in shared] == ["r1", "r2", "r3", "r4"] and {e["branch"] for e in shared} == {"b2"}
    # Ten for the first branch, two for the one relation the reading changed.
    assert state["calls"]["subagents"] == 12
    assert [c["agent"] for c in fake.cards if c["agent"].startswith("b2_r") and c["agent"] != "b2_reverifier"] == ["b2_r5_primary", "b2_r5_independent"]


def test_an_ambiguity_both_judges_read_the_same_way_opens_no_branch():
    fake = Fake(ambiguous=TWO_READINGS, judges_agree=True)
    result = run(GraphFallback(QUESTION, calls(fake)))
    state = result["state"]
    assert state["chosen_readings"] == {"is larger": "by area"} and state["open_ambiguities"] == []
    assert [b["id"] for b in state["branches"]] == ["b1"] and state["branches"][0]["readings"] == {"is larger": "by area"}
    assert all("read one way by judgment" in prompt for prompt in fake.checks)
    assert result["settlement"]["level"] == "verified"


def test_judged_readings_keep_three_ambiguities_inside_the_branch_cap():
    three = [{"span": "is larger", "readings": ["by area", "by population"]},
             {"span": "links two islands", "readings": ["directly", "by ferry too"]},
             {"span": "opened in 1950", "readings": ["first opened", "reopened"]}]
    unjudged = run(GraphFallback(QUESTION, calls(Fake(ambiguous=three))))
    assert unjudged["settlement"]["reason"] == "too_many_branches"
    judged = run(GraphFallback(QUESTION, calls(Fake(ambiguous=three, judges_agree=True))))
    assert judged["settlement"]["level"] == "verified" and len(judged["state"]["branches"]) == 1


def test_the_reserve_stops_research_and_the_final_check_still_runs():
    fake = Fake()
    result = run(GraphFallback(QUESTION, calls(fake, {"spent_estimate_cny": "43"}), hard_cap="50"))
    assert not fake.cards
    assert any(e["event"] == "research_budget_reached" for e in result["state"]["events"])
    assert result["state"]["results"]["b1"]["objections"] == [{"by": "program", "verdict": "no_answer"}]
    # The head still gives an answer; it is never checked as verified.
    assert (result["settlement"]["level"], result["settlement"]["forced"]) == ("best_effort", "pro_synthesis")


class Cut(BaseException):
    """Stands for the machine going down mid-run."""


def test_a_cut_off_run_resumes_from_its_last_checkpoint(tmp_path):
    path = tmp_path / "graph_v1.json"
    first = Fake()

    async def cut(card):
        raise Cut()

    with pytest.raises(Cut):
        run(GraphFallback(QUESTION, Calls(first.model, first.model, cut, lambda: {}), state_path=path))
    assert gf.load_checkpoint(path)["done"] == ["decomposed", "checked", "judged", "branched"]
    second = Fake()
    result = run(GraphFallback(QUESTION, calls(second), state_path=path))
    assert result["settlement"]["level"] == "verified"
    assert not any(a.startswith(("Decompose", "Review")) for a in second.asked[:1])
    assert sum(a.startswith("Decompose") for a in second.asked) == 0


def test_quotes_missing_from_the_sources_bind_nothing_and_the_node_is_given_up():
    fake = Fake(quote_in_sources=False, basis="knowledge")
    result = run(GraphFallback(QUESTION, calls(fake)))
    events = [e["event"] for e in result["state"]["events"]]
    assert "unsourced_quotes_dropped" in events and "stuck" in events
    assert "rechecked" not in events
    assert result["state"]["rechecks"] == 2 and result["state"]["replans"] == 2
    # With nothing found the head answers from what it knows, and says so.
    settlement = result["settlement"]
    assert (settlement["level"], settlement["forced"], settlement["basis"]) == ("best_effort", "pro_synthesis", "knowledge")
    with_nothing = run(GraphFallback(QUESTION, calls(Fake(quote_in_sources=False, fallback=""))))
    assert with_nothing["settlement"]["level"] == "none" and with_nothing["answer"] == ""


def test_a_dropped_call_is_retried_instead_of_ending_the_decomposition():
    fake = Fake()
    dropped: set[str] = set()

    async def flaky(prompt):
        # Each decomposition's first call is dropped, as all three were in the real trial.
        if prompt.startswith("Decompose") and prompt not in dropped:
            dropped.add(prompt)
            raise ConnectionError("peer closed connection")
        return await fake.model(prompt)

    result = run(GraphFallback(QUESTION, Calls(flaky, fake.model, fake.subagent, lambda: {"spent_estimate_cny": "1"})))
    failed = [e for e in result["state"]["events"] if e["event"] == "model_call_failed"]
    assert len(failed) == 3 and {e["attempt"] for e in failed} == {1}
    assert result["settlement"]["level"] == "verified"


def test_an_unknown_setting_is_refused():
    with pytest.raises(ValueError):
        GraphFallback(QUESTION, calls(Fake()), config={"fixed_quota": 3})


# --- V2.6 ------------------------------------------------------------------------

BRIDGES = ["Iron Bridge", "Rope Bridge", "Stone Bridge", "Toll Bridge", "Rail Bridge", "Swing Bridge"]


def test_a_long_list_is_kept_and_tested_in_batches_until_one_candidate_is_left():
    fake = Fake(extra={"bridge beside": BRIDGES})
    result = run(GraphFallback(QUESTION, calls(fake), config={"wide_batch": 3}))
    events = result["state"]["events"]
    assert {"r2": "wide"}.items() <= {e["relation"]: e["status"] for e in events if e["event"] == "relation_researched"
                                      and e["relation"] == "r2"}.items()
    batches = [c["agent"] for c in fake.cards if c["agent"].startswith("b1_r3_")]
    assert batches == ["b1_r3_batch1", "b1_r3_batch2", "b1_r3_batch3"]
    assert result["settlement"]["level"] == "verified" and result["answer"] == "North Isle"
    assert result["state"]["branches"][0]["bound"]["B"]["entity"] == "Old Bridge"


def test_the_head_merges_the_clues_of_a_node_no_single_relation_can_narrow():
    museums = [f"Museum {n}" for n in range(31)]
    steps = [{"action": "merge_clues", "node": "Y", "why": "a constant"},   # refused by the program, asked again
             {"action": "merge_clues", "node": "M", "why": "too many museums"}]
    fake = Fake(extra={"museum opened in": museums}, steps=steps)
    result = run(GraphFallback(QUESTION, calls(fake)))
    state = result["state"]
    assert [e["status"] for e in state["events"] if e["event"] == "relation_researched" and e["relation"] == "r1"] == ["too_many"]
    assert [(e["action"], e["node"]) for e in state["events"] if e["event"] == "next_step"] == [("merge_clues", "M")]
    assert any(c["agent"] == "b1_M_merged_primary" for c in fake.cards)
    assert state["rechecks"] == 0 and result["settlement"]["level"] == "verified" and result["answer"] == "North Isle"


def test_a_failed_check_sends_its_relation_back_to_research_and_the_answer_is_checked_again():
    fake = Fake(rejected={"r5"}, reject_rounds=1)
    result = run(GraphFallback(QUESTION, calls(fake)))
    state = result["state"]
    assert [e["relations"] for e in state["events"] if e["event"] == "repair"] == [["r5"]]
    repaired = [c for c in fake.cards if c["agent"].startswith("b1_r5_")][2:]
    assert repaired and all("A checker did not accept this relation" in c["prompt"] for c in repaired)
    assert all("Candidates to test" in c["prompt"] for c in repaired)
    assert state["repairs"] == 1 and len(fake.checks) == 4
    assert result["settlement"]["level"] == "verified" and result["answer"] == "North Isle"


def test_when_repairs_run_out_the_head_still_delivers_an_answer_and_tests_the_earlier_one():
    yes, no = {"contradicted": True, "relation": "r5", "reason": "smaller"}, {"contradicted": False}
    fake = Fake(rejected={"r5"}, reject_rounds=99, refute=(yes, no))
    result = run(GraphFallback(QUESTION, calls(fake), earlier_answer="South Isle"))
    settlement, state = result["settlement"], result["state"]
    assert state["repairs"] == 3 and state["results"]["b1"]["repair_rounds"] == 3
    assert (settlement["level"], settlement["forced"], result["answer"]) == ("best_effort", "pro_synthesis", "North Isle")
    assert settlement["unverified"] == ["r5"] and settlement["earlier_refuted"] is True
    assert "forced" in state["done"]
    # One check that spares the earlier answer is enough to keep it.
    doubtful = Fake(rejected={"r5"}, reject_rounds=99, refute=(yes, no), refute_flash=(no, no))
    assert run(GraphFallback(QUESTION, calls(doubtful), earlier_answer="South Isle"))["settlement"]["earlier_refuted"] is False
    # The same answer as the earlier one is not tested at all.
    same = Fake(rejected={"r5"}, reject_rounds=99, refute=(yes, no))
    kept = run(GraphFallback(QUESTION, calls(same), earlier_answer="north isle"))
    assert kept["settlement"]["earlier_refuted"] is False and not any(a.startswith("Two answers") for a in same.asked)


def test_the_head_takes_back_a_wrong_identification_and_the_research_goes_on_from_it():
    misled = {"bridge beside": ["Wrong Bridge"], "bridge built in": ["Wrong Bridge"]}
    fake = Fake(misled=misled, steps=[{"action": "merge_clues", "node": "B", "why": "no islands for that bridge"}])
    result = run(GraphFallback(QUESTION, calls(fake)))
    state = result["state"]
    assert [(e["node"], e["unbound"]) for e in state["events"] if e["event"] == "doubted"] == [("B", ["A", "B", "I"])]
    assert [(e["action"], e["node"]) for e in state["events"] if e["event"] == "next_step"] == [("merge_clues", "B")]
    assert state["branches"][0]["bound"]["B"]["entity"] == "Old Bridge" and state["rechecks"] == 0
    assert result["settlement"]["level"] == "verified" and result["answer"] == "North Isle"


def test_a_node_is_taken_back_once_and_a_constant_never():
    controller = GraphFallback(QUESTION, calls(Fake()))
    branch = gf.new_branch("b1", json.loads(json.dumps(GRAPH)))
    branch["bound"]["B"] = {"entity": "Old Bridge", "from": "candidates", "support": {}}
    branch["candidates"]["B"] = [{"entity": "Old Bridge", "support": {}, "agents": []}]
    assert controller.step_problem(branch, {"action": "merge_clues", "node": "B"}) == ""
    assert controller.step_problem(branch, {"action": "test_candidates", "relation": "r3"}) == ""
    branch["doubted"] = ["B"]
    assert "taken back once" in controller.step_problem(branch, {"action": "merge_clues", "node": "B"})
    assert "taken back once" in controller.step_problem(branch, {"action": "test_candidates", "relation": "r3"})
    assert "constant" in controller.step_problem(branch, {"action": "revise_graph", "node": "Y"})


def test_with_no_model_answer_the_program_picks_the_best_supported_candidate():
    fake = Fake(rejected={"r5"}, reject_rounds=99, fallback="")
    result = run(GraphFallback(QUESTION, calls(fake)))
    assert (result["settlement"]["forced"], result["answer"]) == ("program_guess", "North Isle")
