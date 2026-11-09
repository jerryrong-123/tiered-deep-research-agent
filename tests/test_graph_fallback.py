"""Offline tests for the relation-graph fallback's model-free parts."""
from __future__ import annotations

import copy
import json
from decimal import Decimal

import pytest

from workflows.agent_team import graph_fallback as gf

QUESTION = ("A museum that opened in 1950 stands beside a bridge built in the 1930s. "
            "The bridge links two islands. Which of the two islands is larger?")


def graph() -> dict:
    return {
        "nodes": [
            {"id": "M", "kind": "unknown", "description": "the museum"},
            {"id": "B", "kind": "unknown", "description": "the bridge"},
            {"id": "I", "kind": "unknown", "description": "the two islands"},
            {"id": "A", "kind": "answer", "description": "the larger island"},
            {"id": "Y", "kind": "constant", "description": "opening year", "value": "1950"},
        ],
        "relations": [
            {"id": "r1", "span": "A museum that opened in 1950", "subject": "M", "relation": "opened in",
             "object": "Y", "kind": "lookup", "depends_on": ["Y"], "anchor": True},
            {"id": "r2", "span": "stands beside a bridge built in the 1930s", "subject": "B", "relation": "beside",
             "object": "M", "kind": "lookup", "depends_on": ["M"], "anchor": True},
            {"id": "r3", "span": "built in the 1930s", "subject": "B", "relation": "built in", "object": "",
             "kind": "filter", "depends_on": [], "anchor": False},
            {"id": "r4", "span": "The bridge links two islands", "subject": "I", "relation": "linked by",
             "object": "B", "kind": "lookup", "depends_on": ["B"], "anchor": True},
            {"id": "r5", "span": "Which of the two islands is larger?", "subject": "A", "relation": "larger of",
             "object": "I", "kind": "compare", "depends_on": ["I"], "anchor": True},
        ],
        "answer": {"node": "A", "form": "the island's name"},
    }


def quote(text: str = "the museum opened its doors in 1950") -> dict:
    return {"quote": text, "url": "https://example.org/page"}


def test_settings_keep_defaults_and_refuse_unknown_keys():
    assert gf.settings(None)["decompositions"] == 3
    assert gf.settings({"max_branches": 2})["max_branches"] == 2
    with pytest.raises(ValueError):
        gf.settings({"fixed_quota": 3})


def test_a_well_formed_graph_passes_the_program_checks():
    verdict = gf.check_graph(QUESTION, graph())
    assert verdict["ok"], verdict
    assert verdict["coverage_pct"] == 100


@pytest.mark.parametrize("change,problem", [
    (lambda g: g["relations"][0].update(span="A museum which opened in 1950"), "span_not_verbatim:r1"),
    (lambda g: g["relations"][1].update(depends_on=["I"]), "dependency_cycle"),
    (lambda g: g["relations"][0].update(anchor=False), "no_anchor:M"),
    (lambda g: g["nodes"][0].update(kind="answer"), "answer_nodes:2"),
    (lambda g: g["nodes"][4].update(value=""), "constant_without_value:Y"),
    (lambda g: g["relations"][2].update(kind="guess"), "relation_kind:r3"),
    (lambda g: g["relations"][3].update(subject="Z"), "unknown_subject:r4"),
    (lambda g: g["answer"].update(form=""), "answer_form_missing"),
])
def test_program_checks_name_each_defect(change, problem):
    broken = graph()
    change(broken)
    verdict = gf.check_graph(QUESTION, broken)
    assert not verdict["ok"]
    assert problem in verdict["problems"]


def test_uncovered_question_words_fail_coverage_and_are_listed():
    partial = graph()
    partial["relations"] = [r for r in partial["relations"] if r["id"] != "r4"]
    verdict = gf.check_graph(QUESTION, partial)
    assert any(p.startswith("coverage:") for p in verdict["problems"])
    assert any("links" in part for part in verdict["uncovered"])


def test_drafts_line_up_by_span_and_count_agreement():
    first, second, third = graph(), graph(), graph()
    second["relations"][2]["kind"] = "lookup"
    third["relations"] = third["relations"][:3] + [{"id": "x9", "span": "is larger", "subject": "I",
                                                    "relation": "count", "object": "", "kind": "filter",
                                                    "depends_on": [], "anchor": True}]
    report = gf.align_drafts([first, second, third])
    assert report["drafts"] == 3
    assert report["in_all"] >= 3
    assert report["kind_disagreements"] >= 1
    assert report["single"] >= 1
    assert 0 < report["agreement_pct"] < 100


def test_restatement_gaps_point_at_missing_clauses_and_foreign_conditions():
    gaps = gf.restatement_gaps(QUESTION, ["a museum opened in 1950", "the bridge links two islands",
                                          "the larger of the two islands", "a lighthouse painted red"])
    assert any("bridge built" in clause for clause in gaps["missing_clauses"])
    assert gaps["unmatched_conditions"] == ["a lighthouse painted red"]


def test_reviews_merge_into_problems_and_ambiguities():
    combined = gf.combine_reviews([
        {"reviewer": "pro", "missing": ["the opening year"], "ambiguous": [{"span": "two islands", "readings": ["A", "B"]}]},
        {"reviewer": "flash", "missing": ["The opening year"], "misread": [],
         "ambiguous": [{"span": "two islands", "readings": ["b", "C"]}, {"span": "larger", "readings": ["by area"]}]},
    ])
    assert not combined["ok"]
    assert len(combined["problems"]) == 1 and combined["problems"][0]["from"] == ["pro", "flash"]
    assert combined["ambiguities"] == [{"span": "two islands", "readings": ["A", "B", "C"]}]


def test_reviews_that_find_nothing_pass():
    assert gf.combine_reviews([{"reviewer": "pro"}, {"reviewer": "flash", "missing": []}])["ok"]


def test_branches_cover_every_reading_combination_up_to_the_cap():
    assert gf.plan_branches([]) == [{"id": "b1", "readings": {}}]
    two_by_two = gf.plan_branches([{"span": "x", "readings": ["1", "2"]}, {"span": "y", "readings": ["a", "b"]}])
    assert len(two_by_two) == 4 and two_by_two[3]["readings"] == {"x": "2", "y": "b"}
    assert gf.plan_branches([{"span": "x", "readings": ["1", "2", "3"]}, {"span": "y", "readings": ["a", "b"]}]) is None


def test_a_reading_counts_as_agreed_only_when_every_judge_chose_it():
    readings = ["by area", "by population"]
    assert gf.agreed_reading(readings, ["by area", "By Area"]) == "by area"
    assert gf.agreed_reading(readings, ["by area", "by population"]) is None
    assert gf.agreed_reading(readings, ["by area", None]) is None
    assert gf.agreed_reading(readings, ["undecided", "undecided"]) is None
    assert gf.agreed_reading(readings, ["by area"]) is None


def test_relations_are_the_same_task_by_words_and_inputs_not_by_ids():
    first, second = gf.new_branch("b1", graph()), gf.new_branch("b2", graph())
    for node in second["graph"]["nodes"]:
        node["id"] = node["id"] + "x"
    r1, r1x = first["graph"]["relations"][0], dict(second["graph"]["relations"][0], id="z1", subject="Mx", depends_on=["Yx"])
    second["bound"] = {"Yx": {"entity": "1950"}}
    same = gf.relation_signature(first, r1, gf.task_inputs(first, r1)) == gf.relation_signature(second, r1x, gf.task_inputs(second, r1x))
    assert same
    changed = dict(r1, relation="museum reopened in")
    assert gf.relation_signature(first, r1, gf.task_inputs(first, r1)) != gf.relation_signature(first, changed, gf.task_inputs(first, changed))


def test_constants_start_bound_and_anchors_go_first():
    branch = gf.new_branch("b1", graph())
    assert branch["bound"]["Y"]["entity"] == "1950"
    assert [r["id"] for r in gf.ready_relations(branch)] == ["r1"]


def test_a_node_binds_when_one_quoted_candidate_remains():
    branch = gf.new_branch("b1", graph())
    r1, r2, r3 = branch["graph"]["relations"][:3]
    status = gf.record_anchor(branch, r1, [
        {"agent": "a", "candidates": [{"entity": "Harbour Museum", "quotes": [quote()]},
                                      {"entity": "Unquoted Museum", "quotes": []}]},
        {"agent": "b", "candidates": [{"entity": "harbour-museum", "quotes": [quote()]}]},
    ])
    assert status == "established"
    assert [c["entity"] for c in branch["candidates"]["M"]] == ["Harbour Museum"]
    assert gf.settle_node(branch, "M") == "bound"
    assert gf.task_inputs(branch, r2) == {"bound": {"M": "Harbour Museum"}, "candidates": []}
    assert [r["id"] for r in gf.ready_relations(branch)] == ["r2"]

    gf.record_anchor(branch, r2, [{"agent": "a", "candidates": [
        {"entity": "Old Bridge", "quotes": [quote("the Old Bridge beside the museum")]},
        {"entity": "New Bridge", "quotes": [quote("the New Bridge also near the museum")]}]}])
    assert [r["id"] for r in gf.ready_relations(branch)] == ["r3"]
    assert gf.task_inputs(branch, r3)["candidates"] == ["Old Bridge", "New Bridge"]
    status = gf.record_filter(branch, r3, [
        {"agent": "a", "verdicts": [{"entity": "Old Bridge", "satisfies": True, "quotes": [quote("Old Bridge was built in 1934")]},
                                    {"entity": "New Bridge", "satisfies": False, "quotes": [quote("New Bridge was built in 1988")]}]},
    ])
    assert status == "established"
    assert gf.settle_node(branch, "B") == "bound"
    assert branch["bound"]["B"]["entity"] == "Old Bridge"
    assert set(branch["bound"]["B"]["support"]) == {"r2", "r3"}


def test_quoted_yes_against_quoted_no_is_a_conflict_not_a_choice():
    branch = gf.new_branch("b1", graph())
    r1, r2, r3 = branch["graph"]["relations"][:3]
    branch["bound"]["M"] = {"entity": "Harbour Museum"}
    gf.record_anchor(branch, r2, [{"agent": "a", "candidates": [{"entity": "Old Bridge", "quotes": [quote("Old Bridge by the museum")]}]}])
    status = gf.record_filter(branch, r3, [
        {"agent": "a", "verdicts": [{"entity": "Old Bridge", "satisfies": True, "quotes": [quote("Old Bridge was built in 1934")]}]},
        {"agent": "b", "verdicts": [{"entity": "Old Bridge", "satisfies": False, "quotes": [quote("Old Bridge was rebuilt in 1961")]}]},
    ])
    assert status == "conflict"
    assert branch["conflicts"][0]["entity"] == "Old Bridge"
    assert gf.settle_node(branch, "B") == "open"


def test_an_anchor_without_quoted_candidates_or_with_too_many_does_not_settle():
    branch = gf.new_branch("b1", graph())
    r1 = branch["graph"]["relations"][0]
    assert gf.record_anchor(branch, r1, [{"agent": "a", "candidates": [{"entity": "X", "quotes": []}]}]) == "unresolved"
    many = [{"entity": f"Museum {n}", "quotes": [quote()]} for n in range(31)]
    branch["status"]["r1"] = "open"
    assert gf.record_anchor(branch, r1, [{"agent": "a", "candidates": many}], max_candidates=5, wide_candidates=30) == "too_many"
    assert gf.settle_node(branch, "M") == "empty" and "M" not in branch["candidates"]


def test_a_longer_list_up_to_the_wide_limit_is_kept_for_the_other_relations_to_test():
    branch = gf.new_branch("b1", graph())
    r1 = branch["graph"]["relations"][0]
    seven = [{"entity": f"Museum {n}", "quotes": [quote()]} for n in range(7)]
    assert gf.record_anchor(branch, r1, [{"agent": "a", "candidates": seven}], max_candidates=5, wide_candidates=30) == "wide"
    assert len(branch["candidates"]["M"]) == 7
    # The node's anchor now tests candidates instead of finding them.
    assert gf.acts_as_filter(branch, r1) and gf.task_inputs(branch, r1)["candidates"][:2] == ["Museum 0", "Museum 1"]


def test_two_fully_supported_candidates_leave_the_node_unbound():
    branch = gf.new_branch("b1", graph())
    r1 = branch["graph"]["relations"][0]
    gf.record_anchor(branch, r1, [{"agent": "a", "candidates": [{"entity": "One", "quotes": [quote()]},
                                                               {"entity": "Two", "quotes": [quote()]}]}])
    assert gf.settle_node(branch, "M") == "several"
    assert "M" not in branch["bound"]


def test_a_set_node_binds_to_as_many_entities_as_it_stands_for():
    pair = graph()
    pair["nodes"][2]["count"] = 2
    assert gf.check_graph(QUESTION, pair)["ok"]
    branch = gf.new_branch("b1", pair)
    branch["bound"]["B"] = {"entity": "Old Bridge"}
    r4 = branch["graph"]["relations"][3]
    gf.record_anchor(branch, r4, [{"agent": "a", "candidates": [
        {"entity": "North Isle", "quotes": [quote("the bridge joins North Isle")]},
        {"entity": "South Isle", "quotes": [quote("and South Isle across the strait")]}]}])
    assert gf.settle_node(branch, "I") == "bound"
    assert branch["bound"]["I"]["members"] == ["North Isle", "South Isle"]
    assert len(branch["bound"]["I"]["support"]["r4"]) == 2
    pair["nodes"][2]["count"] = 0
    assert "node_count:I" in gf.check_graph(QUESTION, pair)["problems"]


def test_evidence_rows_carry_each_relations_entity_and_quotes():
    branch = gf.new_branch("b1", graph())
    branch["bound"]["M"] = {"entity": "Harbour Museum", "support": {"r1": [quote()]}}
    rows = {row["relation"]: row for row in gf.evidence_rows(branch)}
    assert rows["r1"]["entity"] == "Harbour Museum" and rows["r1"]["quotes"] == [quote()]
    assert rows["r4"]["entity"] == "" and rows["r4"]["quotes"] == []


def test_research_stops_where_the_check_reserve_begins():
    ledger = {"spent_estimate_cny": "30", "unknown_held_upper_cny": "5", "inflight_reserved_upper_cny": "6"}
    assert gf.budget_used(ledger) == Decimal(41)
    assert gf.research_allowed(ledger, Decimal(50), Decimal(8))
    ledger["unknown_held_upper_cny"] = "6"
    assert not gf.research_allowed(ledger, Decimal(50), Decimal(8))


def test_verified_needs_both_checks_every_relation_and_the_re_verifier():
    good = [{"checker": "pro", "verdict": "supported", "relations": [{"relation": "r1", "established": True}]},
            {"checker": "flash", "verdict": "supported", "relations": [{"relation": "r1", "established": True}]}]
    assert gf.branch_verdict(good, {"verdict": "confirmed"}, ["r1"])["verified"]
    one_doubt = copy.deepcopy(good)
    one_doubt[1]["verdict"] = "insufficient"
    assert not gf.branch_verdict(one_doubt, {"verdict": "confirmed"}, ["r1"])["verified"]
    unmarked = copy.deepcopy(good)
    unmarked[0]["relations"] = []
    verdict = gf.branch_verdict(unmarked, {"verdict": "confirmed"}, ["r1"])
    assert {"by": "pro", "relation": "r1"} in verdict["objections"]
    assert not gf.branch_verdict(good, None, ["r1"])["verified"]
    assert not gf.branch_verdict(good[:1], {"verdict": "confirmed"}, ["r1"])["verified"]


def test_branches_combine_only_when_verified_answers_agree():
    assert gf.combine_branches([{"branch": "b1", "verified": True, "answer": "Isle Rouge"},
                                {"branch": "b2", "verified": False, "answer": "Other"}])["answer"] == "Isle Rouge"
    same = gf.combine_branches([{"branch": "b1", "verified": True, "answer": "Île Rouge"},
                                {"branch": "b2", "verified": True, "answer": "ile-rouge"}])
    assert same["level"] == "verified"
    split = gf.combine_branches([{"branch": "b1", "verified": True, "answer": "One"},
                                 {"branch": "b2", "verified": True, "answer": "Two"}])
    assert split["level"] == "best_effort" and split["reason"] == "branches_disagree" and split["answer"] == ""
    assert gf.combine_branches([{"branch": "b1", "verified": False, "answer": "One"}])["reason"] == "not_verified"
    assert gf.combine_branches([{"branch": "b1", "verified": False, "answer": ""}])["level"] == "none"


def test_checkpoints_round_trip_and_leave_no_temporary_file(tmp_path):
    path = tmp_path / "graph" / "checkpoint.json"
    gf.save_checkpoint(path, {"step": "decomposed", "graph": graph()})
    assert gf.load_checkpoint(path)["step"] == "decomposed"
    assert not path.with_suffix(".json.tmp").exists()
    assert gf.load_checkpoint(tmp_path / "missing.json") is None
    path.write_text("{broken", encoding="utf-8")
    assert gf.load_checkpoint(path) is None


def test_a_quote_counts_only_when_the_sub_agents_sources_hold_it():
    outcome = {"agent": "a", "candidates": [
        {"entity": "Harbour Museum", "quotes": [quote("the museum opened its doors in 1950"),
                                                quote("a sentence no page contains at all")]}]}
    kept = gf.keep_sourced(outcome, "Title: Harbour Museum\nThe museum opened its doors in 1950, the guide says.")
    assert [q["quote"] for q in kept["candidates"][0]["quotes"]] == ["the museum opened its doors in 1950"]
    assert kept["unsourced_quotes_dropped"] == 1
    assert outcome["candidates"][0]["quotes"][1]["quote"].startswith("a sentence")


def test_entity_names_compare_without_case_accents_or_hyphens():
    assert gf.norm_entity("Place-des-Arts") == gf.norm_entity("place des arts")
    assert gf.norm_entity("Montréal") == gf.norm_entity("MONTREAL")
    assert json.dumps(gf.DEFAULTS)


# --- V2.6: wide lists, merged clues, reverse links, kept work, repair, fallback ----

def _bound_museum(branch: dict) -> None:
    r1 = branch["graph"]["relations"][0]
    gf.record_anchor(branch, r1, [{"agent": "a", "candidates": [{"entity": "Harbour Museum", "quotes": [quote()]}]}])
    assert gf.settle_node(branch, "M") == "bound"


def test_a_merged_search_keeps_candidates_by_the_relations_their_quotes_name():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    found = [{"agent": "n", "candidates": [
        {"entity": "Old Bridge", "quotes": [dict(quote("The Old Bridge stands beside the museum."), relation="r2"),
                                             dict(quote("The Old Bridge was built in 1934."), relation="r3")]},
        {"entity": "New Bridge", "quotes": [dict(quote("The New Bridge is beside it."), relation="r2")]},
        {"entity": "Stray", "quotes": [dict(quote("Unrelated."), relation="r9")]}]}]
    assert gf.record_node(branch, "B", found) == "established"
    assert [c["entity"] for c in branch["candidates"]["B"]] == ["Old Bridge", "New Bridge"]
    assert branch["status"]["r2"] == "established" and branch["status"]["r3"] == "established"
    assert gf.settle_node(branch, "B") == "bound" and branch["bound"]["B"]["entity"] == "Old Bridge"


def test_a_relation_a_merged_search_did_not_support_opens_to_test_the_new_candidates():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    branch["status"]["r3"] = "unresolved"
    found = [{"agent": "n", "candidates": [{"entity": "Old Bridge", "quotes": [dict(quote(), relation="r2")]}]}]
    gf.record_node(branch, "B", found)
    assert branch["status"]["r3"] == "open" and gf.ready_relations(branch)[0]["id"] == "r3"


def test_reverse_links_become_the_other_nodes_candidates_and_open_its_relations():
    branch = gf.new_branch("b1", graph())
    r2 = branch["graph"]["relations"][1]
    branch["candidates"]["B"] = [{"entity": "Old Bridge", "support": {}, "agents": []}]
    branch["status"]["r1"] = "too_many"
    linked = [{"agent": "r", "candidates": [{"entity": "Harbour Museum", "for": "Old Bridge", "quotes": [quote()]}]}]
    assert gf.record_reverse(branch, r2, linked) == "established"
    assert [c["entity"] for c in branch["candidates"]["M"]] == ["Harbour Museum"]
    # The museum's anchor now tests the linked candidate; the link itself waits for the museum.
    assert branch["status"]["r1"] == "open" and gf.acts_as_filter(branch, branch["graph"]["relations"][0])
    assert branch["status"]["r2"] == "open"


def test_a_revised_graph_keeps_work_whose_task_did_not_change():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    r2 = branch["graph"]["relations"][1]
    gf.record_anchor(branch, r2, [{"agent": "a", "candidates": [{"entity": "Old Bridge", "quotes": [quote()]}]}])
    branch["status"]["r3"] = "unresolved"
    revised = copy.deepcopy(branch["graph"])
    revised["relations"][0]["relation"] = "museum first opened in"   # reworded off the stuck node: kept
    revised["relations"][2]["relation"] = "bridge completed in"       # reworded on the stuck node: redone
    affected = gf.rebase(branch, revised, "B")
    assert affected == {"B", "I", "A"}
    assert branch["bound"]["M"]["entity"] == "Harbour Museum" and branch["status"]["r1"] == "established"
    # The bridge is rebuilt from its anchor's findings instead of being searched again.
    assert branch["status"]["r2"] == "established" and [c["entity"] for c in branch["candidates"]["B"]] == ["Old Bridge"]
    assert branch["status"]["r3"] == "open" and "r3" not in branch["findings"]


def test_an_identified_node_a_revision_gives_a_new_relation_comes_unbound_until_it_is_tested():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    revised = copy.deepcopy(branch["graph"])
    revised["relations"].append({"id": "r6", "span": "a museum by the sea", "subject": "M", "relation": "museum near",
                                 "object": "", "kind": "filter", "depends_on": [], "anchor": False})
    branch["candidates"]["M"][0]["support"]["r6"] = [quote("support given for an older task")]
    rebuilt = gf.rebase(branch, revised, "I")
    assert rebuilt == {"M", "B", "I", "A"} and branch["rebases"][-1]["unbound"] == ["M"]
    assert "M" not in branch["bound"] and [c["entity"] for c in branch["candidates"]["M"]] == ["Harbour Museum"]
    assert set(branch["candidates"]["M"][0]["support"]) == {"r1"}
    assert branch["status"]["r1"] == "established" and branch["status"]["r6"] == "open"
    # The new relation is tested on the kept candidate first; the bridge waits for the museum.
    assert [r["id"] for r in gf.ready_relations(branch)] == ["r6"]


def test_the_head_can_take_back_an_identification_and_what_depends_on_it():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    branch["bound"]["B"] = {"entity": "Old Bridge", "from": "candidates", "support": {}}
    branch["status"].update(r2="established", r3="established", r4="established")
    branch["candidates"]["I"] = [{"entity": "North Isle", "support": {"r4": [quote()]}, "agents": []}]
    assert gf.doubt(branch, "Y") == set() and "Y" in branch["bound"]
    assert gf.doubt(branch, "M") == {"M", "B", "I", "A"} and branch["doubted"] == ["M"]
    assert not {"M", "B"} & set(branch["bound"]) and "I" not in branch["candidates"]
    assert set(branch["candidates"]["M"][0]["support"]) == {"r1"} and branch["status"]["r1"] == "established"
    assert branch["status"]["r2"] == "open" and branch["status"]["r4"] == "open"


def test_reopening_a_relation_unbinds_its_node_and_rebuilds_what_depends_on_it():
    branch = gf.new_branch("b1", graph())
    _bound_museum(branch)
    branch["bound"]["B"] = {"entity": "Old Bridge", "from": "candidates", "support": {}}
    branch["status"].update(r2="established", r3="established", r4="established")
    branch["candidates"]["I"] = [{"entity": "North Isle", "support": {"r4": [quote()]}, "agents": []}]
    reopened = gf.reopen(branch, {"r1", "zz"})
    assert reopened == {"M", "B", "I", "A"}
    assert branch["status"]["r1"] == "open" and branch["status"]["r4"] == "open"
    assert "M" not in branch["bound"] and "Y" in branch["bound"]
    assert [c["entity"] for c in branch["candidates"]["M"]] == ["Harbour Museum"]
    assert branch["candidates"]["M"][0]["support"] == {} and "I" not in branch["candidates"]


def test_the_programs_own_pick_is_the_answer_candidate_with_the_most_support():
    branch = gf.new_branch("b1", graph())
    assert gf.best_guess(branch) == ""
    branch["candidates"]["A"] = [{"entity": "South Isle", "support": {"r5": [quote()]}, "agents": []},
                                 {"entity": "North Isle", "support": {"r5": [quote(), quote("more")]}, "agents": []}]
    assert gf.best_guess(branch) == "North Isle"
    rows = {row["relation"]: row for row in gf.research_rows(branch)}
    assert [e["entity"] for e in rows["r5"]["entities"]] == ["South Isle", "North Isle"] and rows["r1"]["entities"] == []


def test_objections_name_the_relations_to_research_again_with_their_reasons():
    result = {"checks": [{"checker": "pro", "relations": [{"relation": "r5", "established": False, "reason": "area not quoted"},
                                                           {"relation": "r4", "established": True}]}],
              "objections": [{"by": "flash", "relation": "r3"}, {"by": "re_verifier", "verdict": "incomplete"}],
              "reverify": {"links": [{"relation": "r2", "holds": False, "reason": "beside another museum"},
                                     {"relation": "r9", "holds": False}]}}
    found = gf.objected_relations(result, {"r1", "r2", "r3", "r4", "r5"})
    assert found == {"r5": ["pro: area not quoted"], "r3": [], "r2": ["re-verifier: beside another museum"]}


def test_an_earlier_answer_counts_as_refuted_only_when_both_checks_say_so_and_spare_the_new_one():
    no = {"contradicted": False}
    yes = {"contradicted": True, "relation": "r5"}
    assert gf.earlier_refuted([{"first": yes, "second": no}, {"first": yes, "second": no}], {"r5"})
    assert not gf.earlier_refuted([{"first": yes, "second": no}, {"first": no, "second": no}], {"r5"})
    assert not gf.earlier_refuted([{"first": yes, "second": no}, {"first": yes, "second": yes}], {"r5"})
    assert not gf.earlier_refuted([{"first": dict(yes, relation="r8"), "second": no}] * 2, {"r5"})
    assert not gf.earlier_refuted([{"first": yes, "second": no}, None], {"r5"})
