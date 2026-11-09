"""Offline checks for the router: a run's own result decides the next layer."""
from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from local_runner import tiered_router as router
from local_runner.prepare import prepare
from local_runner.tiered_router import LAYERS, TEAM_LAYER, Limits, Router, choose, load_questions, models_used, settled

QUESTION = "Which street is left of the 1902 factory?"
ROW = {"id": "q1", "question": QUESTION}
VERIFIED = {"level": "verified", "kind": "independent_check", "check": "supported"}
DOUBTED = {"level": "best_effort", "kind": "independent_check", "check": "insufficient"}
FORCED = {"level": "best_effort", "kind": "forced_synthesis", "answer": "Forced guess"}
EMPTY = {"level": "none", "kind": "forced_synthesis", "reason": "no_answer_from_evidence"}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Script:
    """Stands in for one container run and writes what a real trial leaves behind."""

    def __init__(self, plan):
        self.plan, self.launched = plan, []

    def make_trial(self, source, trial, *, profile_file, model, hard_cap):
        assert json.loads(Path(source).read_text(encoding="utf-8"))["question"]
        Path(trial).mkdir(parents=True)
        (Path(trial) / "PREPARED_BY_TEST").write_text(json.dumps([Path(profile_file).name, model, hard_cap]))

    def launch(self, trial):
        name = f"{trial.parent.name}/{trial.name}"
        self.launched.append(name)
        spec = dict(self.plan[name])
        _write(trial / "STARTED.json", {})
        settlement, escalation = spec.pop("settlement", None), spec.pop("escalation", None)
        model = "deepseek-v4-pro" if trial.name == "pro" else "deepseek-flash"
        answer = spec.pop("text", "")
        if spec.pop("failed", False):
            _write(trial / "WORKER_RESULT.json", {"technical_status": "technical_incomplete", "model_ledger": {"calls": 1}})
            return 1
        state = {"answer": spec.pop("state_answer", "")}
        if settlement:
            state["settlement"] = settlement
        if escalation:
            state["escalation"] = {"reason": escalation}
        _write(trial / "native/agent/trajectories/convergence_v1.json", state)
        _write(trial / "WORKER_RESULT.json", {
            "technical_status": "pipeline_returned", "answer": answer, "duration_s": 9,
            "model_ledger": {"calls": 5, "spent_estimate_cny": spec.pop("estimate", "1"),
                             "unknown_held_upper_cny": spec.pop("held", "0")}})
        _write(trial / "MODEL_BALANCE_AFTER.json", {"balances": [{"total_balance": spec.pop("balance", "50")}]})
        status, text = spec.pop("search", (200, "ok"))
        _write(trial / "web_responses/a.json", {"host": "google.serper.dev", "status": status, "text": text})
        events = spec.pop("events", None) or [{"kind": "model_request", "model": model, "tools": ["web_search"], "max_tokens": 32768}]
        (trial / "events.jsonl").write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
        return 0


def _route(tmp_path, plan, *, questions=None, layers=LAYERS, limits=None, workers=1, stop_doubted=False):
    script = Script(plan)
    instance = Router(questions or [{"id": "q1", "question": QUESTION, "question_sha256": "x"}], tmp_path / "out",
                      layers, limits or Limits(Decimal(100)), script.launch, script.make_trial, stop_doubted=stop_doubted)
    summary = instance.run(workers)
    rows = [json.loads(line) for line in (tmp_path / "out/ROUTER_RESULTS.jsonl").read_text(encoding="utf-8").splitlines()]
    return script, summary, rows


def _gate(level_settlement, text="Synthetic Answer Street\n\nEvidence level: verified", **extra):
    return {"settlement": level_settlement, "text": text, "state_answer": "Synthetic Answer Street", **extra}


def _forced(**extra):
    return {"settlement": FORCED, "text": "Forced guess\n\nEvidence level: best_effort", **extra}


def _empty(**extra):
    return {"settlement": EMPTY, "text": "[ESCALATION_REQUIRED] reason=light_turn_budget", "escalation": "light_turn_budget", **extra}


# Routing --------------------------------------------------------------------

def test_a_verified_answer_stops_at_the_first_layer(tmp_path):
    script, summary, rows = _route(tmp_path, {"q1/flash": _gate(VERIFIED)})
    assert script.launched == ["q1/flash"]
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["answer"]) == ("flash", "verified", "Synthetic Answer Street")
    assert summary["by_level"] == {"verified": 1} and rows[0]["finished"]


def test_an_answer_the_check_doubted_goes_on_to_the_next_layer(tmp_path):
    plan = {"q1/flash": _gate(DOUBTED), "q1/pro": {**_gate(VERIFIED), "state_answer": "Pro street"}}
    script, _, rows = _route(tmp_path, plan)
    assert script.launched == ["q1/flash", "q1/pro"]
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["answer"]) == ("pro", "verified", "Pro street")


def test_the_switch_stops_on_a_doubted_answer(tmp_path):
    script, summary, rows = _route(tmp_path, {"q1/flash": _gate(DOUBTED)}, stop_doubted=True)
    assert script.launched == ["q1/flash"] and rows[0]["level"] == "best_effort" and summary["stop_doubted"]


def test_a_forced_answer_cannot_replace_a_gate_passed_one(tmp_path):
    script, _, rows = _route(tmp_path, {"q1/flash": _gate(DOUBTED), "q1/pro": _forced()})
    assert script.launched == ["q1/flash", "q1/pro"]
    assert (rows[0]["layer"], rows[0]["answer"]) == ("flash", "Synthetic Answer Street")


def test_an_equally_credible_later_answer_keeps_the_earlier_one(tmp_path):
    plan = {"q1/flash": _gate(DOUBTED), "q1/pro": {**_gate(DOUBTED), "state_answer": "Pro street"}}
    _, _, rows = _route(tmp_path / "gate", plan)
    assert (rows[0]["layer"], rows[0]["answer"]) == ("flash", "Synthetic Answer Street")
    _, _, rows = _route(tmp_path / "forced", {"q1/flash": _forced(), "q1/pro": _forced()})
    assert rows[0]["layer"] == "flash"


def test_a_forced_answer_goes_on_and_a_better_one_replaces_it(tmp_path):
    script, _, rows = _route(tmp_path, {"q1/flash": _forced(), "q1/pro": _gate(VERIFIED)})
    assert script.launched == ["q1/flash", "q1/pro"]
    assert (rows[0]["layer"], rows[0]["level"]) == ("pro", "verified")
    assert [step["level"] for step in rows[0]["path"]] == ["best_effort", "verified"]


def test_an_empty_later_layer_keeps_the_earlier_forced_answer(tmp_path):
    _, _, rows = _route(tmp_path, {"q1/flash": _forced(), "q1/pro": _empty()})
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["answer"]) == ("flash", "best_effort", "Forced guess")
    assert rows[0]["finished"]


def test_with_no_answer_anywhere_the_record_says_none(tmp_path):
    _, summary, rows = _route(tmp_path, {"q1/flash": _empty(), "q1/pro": _empty()})
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["answer"]) == ("", "none", "")
    assert summary["by_level"] == {"none": 1}


def test_a_technical_failure_moves_to_the_next_layer(tmp_path):
    script, _, rows = _route(tmp_path, {"q1/flash": {"failed": True}, "q1/pro": _gate(VERIFIED)})
    assert script.launched == ["q1/flash", "q1/pro"]
    assert rows[0]["path"][0]["kind"] == "technical_failure" and rows[0]["layer"] == "pro"


def test_the_team_runs_only_when_switched_on(tmp_path):
    plan = {"q1/flash": _empty(), "q1/pro": _empty(), "q1/fat": {"text": "Team answer", "state_answer": "Team answer"}}
    script, _, rows = _route(tmp_path / "off", plan)
    assert script.launched == ["q1/flash", "q1/pro"]
    script, _, rows = _route(tmp_path / "on", plan, layers=(*LAYERS, TEAM_LAYER))
    assert script.launched == ["q1/flash", "q1/pro", "q1/fat"]
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["answer"]) == ("fat", "team", "Team answer")


def test_routing_reads_only_the_run_result_not_the_question_id(tmp_path):
    results = []
    for ident in ("openai_bc_0989", "something-else"):
        plan = {f"{ident}/flash": _forced(), f"{ident}/pro": _gate(VERIFIED)}
        script, _, rows = _route(tmp_path / ident, plan, questions=[{"id": ident, "question": QUESTION, "question_sha256": "x"}])
        results.append([name.split("/")[1] for name in script.launched])
    assert results[0] == results[1] == ["flash", "pro"]


@pytest.mark.parametrize("level,kind,expected", [
    ("verified", "independent_check", True), ("team", "team_answer", True),
    ("best_effort", "independent_check", False), ("best_effort", "forced_synthesis", False),
    ("best_effort", "unsettled", False), ("none", "technical_failure", False), ("none", "", False),
])
def test_only_a_verified_or_team_answer_settles_a_question(level, kind, expected):
    assert settled({"level": level, "kind": kind}) is expected


def test_the_switch_adds_only_doubted_answers_to_what_stops():
    assert settled({"level": "best_effort", "kind": "independent_check"}, stop_doubted=True) is True
    assert settled({"level": "best_effort", "kind": "forced_synthesis"}, stop_doubted=True) is False


def test_choosing_prefers_the_most_credible_then_the_earliest_answer():
    verified = {"level": "verified", "kind": "independent_check"}
    doubted = {"level": "best_effort", "kind": "independent_check"}
    forced = {"level": "best_effort", "kind": "forced_synthesis"}
    empty = {"level": "none", "kind": "forced_synthesis"}
    assert choose([doubted, forced]) is doubted and choose([forced, doubted]) is doubted
    assert choose([forced, empty]) is forced and choose([empty]) is None
    assert choose([verified, doubted]) is verified
    later = dict(forced)
    assert choose([forced, later]) is forced


def test_answers_for_judging_hold_the_answer_without_its_label(tmp_path):
    _route(tmp_path, {"q1/flash": _gate(VERIFIED)})
    exported = json.loads((tmp_path / "out/ANSWERS_FOR_JUDGING.jsonl").read_text(encoding="utf-8"))
    assert exported == {"id": "q1", "run": "flash_q1", "model": "deepseek-flash", "level": "verified",
                        "answer": "Synthetic Answer Street"}


def test_layers_use_their_own_profile_model_and_cap(tmp_path):
    _route(tmp_path, {"q1/flash": _forced(), "q1/pro": _gate(VERIFIED)})
    read = lambda name: json.loads((tmp_path / f"out/q1/{name}/PREPARED_BY_TEST").read_text())
    assert read("flash") == ["tiered_light_stage_v2.yaml", "", ""]
    assert read("pro") == ["tiered_light_stage_v2_deep.yaml", "deepseek-v4-pro", ""]


# Brakes and resume ----------------------------------------------------------

def _two():
    return [{"id": name, "question": QUESTION, "question_sha256": "x"} for name in ("q1", "q2")]


@pytest.mark.parametrize("plan,limits,reason", [
    (_gate(VERIFIED, balance="3"), Limits(Decimal(100)), "balance_limit"),
    (_gate(VERIFIED, estimate="5"), Limits(Decimal(5)), "spend_limit"),
    (_gate(VERIFIED, held="21"), Limits(Decimal(100)), "held_limit"),
    (_gate(VERIFIED, search=(200, "Not enough credits")), Limits(Decimal(100)), "search_unavailable"),
    (_gate(VERIFIED, search=(400, "refused")), Limits(Decimal(100)), "search_unavailable"),
])
def test_a_brake_stops_new_questions_and_is_recorded(tmp_path, plan, limits, reason):
    script, summary, rows = _route(tmp_path, {"q1/flash": plan}, questions=_two(), limits=limits)
    assert script.launched == ["q1/flash"] and summary["stop"] == reason
    assert rows[1]["path"] == [] and rows[1]["finished"] is False and summary["finished"] == 1
    assert json.loads((tmp_path / "out/STOPS.jsonl").read_text())["reason"] == reason


def test_a_batch_sets_its_held_limit_on_the_command_line(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    seen = {}

    class Recording:
        def __init__(self, questions, out, layers, limits, launch, stop_doubted=False):
            seen["limits"] = limits

        def run(self, workers):
            return {}

    monkeypatch.setattr(router, "Router", Recording)
    command = ["tiered_router", "--questions", str(questions), "--out", str(tmp_path / "out"), "--max-total-cny", "60"]
    monkeypatch.setattr(sys, "argv", command)
    router.main()
    assert seen["limits"].max_held_cny == Decimal(20)
    monkeypatch.setattr(sys, "argv", [*command, "--max-held-cny", "45"])
    router.main()
    assert seen["limits"].max_held_cny == Decimal(45)


def test_a_doubted_answer_survives_an_empty_later_layer(tmp_path):
    _, _, rows = _route(tmp_path, {"q1/flash": _gate(DOUBTED), "q1/pro": _empty()})
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["finished"]) == ("flash", "best_effort", True)


def test_a_brake_between_layers_keeps_the_answer_and_marks_the_question_unfinished(tmp_path):
    script, _, rows = _route(tmp_path, {"q1/flash": _forced(balance="3")})
    assert script.launched == ["q1/flash"]
    assert (rows[0]["layer"], rows[0]["finished"]) == ("flash", False)


def test_three_technical_failures_in_a_row_stop_the_batch(tmp_path):
    questions = [{"id": f"q{index}", "question": QUESTION, "question_sha256": "x"} for index in range(1, 4)]
    plan = {"q1/flash": {"failed": True}, "q1/pro": {"failed": True}, "q2/flash": {"failed": True}}
    script, summary, _ = _route(tmp_path, plan, questions=questions)
    assert summary["stop"] == "consecutive_technical_failures" and len(script.launched) == 3


def test_a_wrong_main_model_stops_the_batch(tmp_path):
    events = [{"kind": "model_request", "model": "deepseek-flash", "tools": ["web_search"], "max_tokens": 32768}]
    _, summary, _ = _route(tmp_path, {"q1/flash": _forced(), "q1/pro": _gate(VERIFIED, events=events)})
    assert summary["stop"] == "main_model_mismatch"


def test_resuming_never_repeats_a_started_trial(tmp_path):
    plan = {"q1/flash": _forced(balance="3"), "q1/pro": _gate(VERIFIED)}
    first, _, _ = _route(tmp_path, plan)
    assert first.launched == ["q1/flash"]
    second, _, rows = _route(tmp_path, plan)
    assert second.launched == ["q1/pro"]
    assert (rows[0]["layer"], rows[0]["level"], rows[0]["finished"]) == ("pro", "verified", True)
    assert rows[0]["estimate_cny"] == "2"


def test_a_trial_cut_off_mid_run_runs_again_and_still_counts(tmp_path):
    cut = tmp_path / "out/q1/flash"
    _write(cut / "STARTED.json", {})
    _write(cut / "MODEL_LEDGER.json", {"spent_estimate_cny": "2", "unknown_held_upper_cny": "1", "inflight_reserved_upper_cny": "3"})
    script, summary, rows = _route(tmp_path, {"q1/flash": _gate(VERIFIED)})
    assert script.launched == ["q1/flash"] and rows[0]["level"] == "verified"
    kept = tmp_path / "out" / router.INTERRUPTED / "q1/flash"
    assert json.loads((kept / "MODEL_LEDGER.json").read_text())["spent_estimate_cny"] == "2"
    assert json.loads((tmp_path / "out/INTERRUPTED.jsonl").read_text())["kept_as"] == "_interrupted/q1/flash"
    assert (summary["estimate_cny"], summary["held_cny"]) == ("3", "4")


def test_a_trial_with_a_finish_record_is_not_taken_for_a_cut_one(tmp_path):
    done = tmp_path / "out/q1/flash"
    _write(done / "STARTED.json", {})
    _write(done / "FINISHED.json", {})
    script, _, rows = _route(tmp_path, {"q1/flash": _gate(VERIFIED), "q1/pro": _gate(VERIFIED)})
    assert script.launched == ["q1/pro"] and rows[0]["path"][0]["kind"] == "technical_failure"
    assert not (tmp_path / "out/INTERRUPTED.jsonl").exists()


def test_a_second_router_on_the_same_folder_is_refused(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    script = Script({})
    with router.run_lock(out):
        with pytest.raises(RuntimeError, match="router_already_running"):
            Router([{"id": "q1", "question": QUESTION, "question_sha256": "x"}], out, LAYERS, Limits(Decimal(100)),
                   script.launch, script.make_trial).run(1)
    assert script.launched == []
    script, _, _ = _route(tmp_path, {"q1/flash": _gate(VERIFIED)})
    assert script.launched == ["q1/flash"]


def test_extraction_is_told_from_a_closing_synthesis(tmp_path):
    lines = [{"kind": "model_request", "model": "deepseek-v4-pro", "tools": ["web_search"], "max_tokens": 32768},
             {"kind": "model_request", "model": "deepseek-flash", "tools": [], "max_tokens": 8192},
             {"kind": "model_request", "model": "deepseek-v4-pro", "tools": [], "max_tokens": 32768}]
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    assert models_used(tmp_path) == ({"deepseek-v4-pro"}, {"deepseek-flash"})


# Input ----------------------------------------------------------------------

def test_a_jsonl_file_needs_only_an_id_and_a_question(tmp_path):
    path = tmp_path / "new.jsonl"
    path.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    assert load_questions(path) == [{**ROW, "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest()}]


def test_a_directory_of_question_files_is_accepted(tmp_path):
    _write(tmp_path / "q2/QUESTION.json", {**ROW, "id": "q2"})
    _write(tmp_path / "q1/QUESTION.json", ROW)
    assert [row["id"] for row in load_questions(tmp_path)] == ["q1", "q2"]


@pytest.mark.parametrize("rows,code", [
    ([{**ROW, "gold": "x"}], "question_rows_may_hold_only_id_and_question"),
    ([{**ROW, "id": "../up"}], "invalid_question_row"),
    ([{**ROW, "question": " "}], "invalid_question_row"),
    ([ROW, ROW], "duplicate_question_id"),
    ([{**ROW, "question_sha256": "0"}], "question_hash_mismatch"),
    ([], "no_questions"),
])
def test_unusable_input_is_refused_before_any_run(tmp_path, rows, code):
    path = tmp_path / "new.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match=code):
        load_questions(path)


# Real trial preparation -----------------------------------------------------

def _question_file(tmp_path):
    path = tmp_path / "QUESTION.json"
    path.write_text(json.dumps({**ROW, "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest()}), encoding="utf-8")
    return path


def test_preparing_a_layer_passes_its_model_to_the_container(tmp_path):
    layer = LAYERS[1]
    prepare(_question_file(tmp_path), tmp_path / "trial", profile_file=router.PROFILES / layer.profile,
            model=layer.model, hard_cap=layer.hard_cap_cny)
    launch = (tmp_path / "trial/launch.ps1").read_text(encoding="utf-8")
    assert "'--env-file',$envFile,'-e','OPENAI_MODEL=deepseek-v4-pro')" in launch


def test_preparing_a_capped_layer_passes_its_cap_to_the_container(tmp_path):
    prepare(_question_file(tmp_path), tmp_path / "trial", profile_file=router.PROFILES / TEAM_LAYER.profile,
            hard_cap=TEAM_LAYER.hard_cap_cny)
    launch = (tmp_path / "trial/launch.ps1").read_text(encoding="utf-8")
    assert "'--env-file',$envFile,'-e','TRIAL_HARD_CNY=10')" in launch
    assert json.loads((tmp_path / "trial/PREPARED.json").read_text(encoding="utf-8"))["hard_cap_cny"] == "10"


@pytest.mark.parametrize("cap", ["26", "0", "-1", "lots"])
def test_a_trial_cap_can_only_lower_the_sealed_one(tmp_path, cap):
    with pytest.raises(ValueError, match="invalid_hard_cap"):
        prepare(_question_file(tmp_path), tmp_path / "trial", hard_cap=cap)


def test_every_layer_profile_exists():
    assert all((router.PROFILES / layer.profile).is_file() for layer in (*LAYERS, TEAM_LAYER))


def test_the_deep_profile_changes_only_its_budget():
    import yaml

    first = yaml.safe_load((router.PROFILES / "tiered_light_stage_v2.yaml").read_text(encoding="utf-8"))
    deep = yaml.safe_load((router.PROFILES / "tiered_light_stage_v2_deep.yaml").read_text(encoding="utf-8"))
    controls = deep["agent"]["progressive_config"]
    changed = {key: controls[key] for key in ("max_search_queries", "escalate_after_main_turns",
                                              "light_soft_calls", "light_hard_calls")}
    assert changed == {"max_search_queries": 80, "escalate_after_main_turns": 40,
                       "light_soft_calls": 40, "light_hard_calls": 60}
    controls.update({key: first["agent"]["progressive_config"][key] for key in changed})
    assert deep == first
