"""Route each question through the tiers on what its own run produced.

Nobody assigns a question to a layer. Every question starts on the first layer,
and that run's evidence level decides whether it stops or moves on. Question ids,
reference answers and judge results play no part, and input rows may not carry
them. A started trial is never repeated, so rerunning with the same --out resumes.
The one exception is a trial the machine cut off, which left neither a result nor a
finish record: it is kept under _interrupted and runs again, and its spend still counts.

Run from the project root:
    python frontier_research/local_runner/tiered_router.py \
        --questions FILE_OR_DIRECTORY --out outputs/NAME --max-total-cny 60

--max-total-cny is a ledger estimate at the conservative factor, about twice the
real offpeak charge. It stops new runs; runs already in flight finish.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

if __package__:
    from .prepare import HERE, prepare
else:
    from prepare import HERE, prepare

MARKER = "[ESCALATION_REQUIRED]"
PROFILES = HERE.parent / "local_profiles"
EXTRACTION_MAX_TOKENS = 8192
CREDIT_ERROR = "Not enough credits"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
QUESTION_KEYS = {"id", "question", "question_sha256"}
EMPTY = {"estimate_cny": "0", "held_cny": "0"}
INTERRUPTED = "_interrupted"


@dataclass(frozen=True)
class Layer:
    name: str
    profile: str
    model: str
    hard_cap_cny: str


# Empty keeps the sealed cap. Held amounts count against a cap without being
# charged, and a correct Flash run reached 17.62 that way, so tighter caps cut
# off answers. The team is a sample, so its cap is lower.
LAYERS = (Layer("flash", "tiered_light_stage_v2.yaml", "", ""),
          Layer("pro", "tiered_light_stage_v2_deep.yaml", "deepseek-v4-pro", ""))
TEAM_LAYER = Layer("fat", "convergence_v1.yaml", "", "10")
# The relation-graph fallback, approved on 2026-11-03 with a per-question cap of
# ¥50, above the ¥25 Flash and Pro keep. Its answer replaces an earlier one only
# when verified; otherwise it is kept as a side answer.
GRAPH_LAYER = Layer("graph", "graph_fallback_v1.yaml", "deepseek-v4-pro", "50")


@dataclass(frozen=True)
class Limits:
    max_total_cny: Decimal
    min_balance_cny: Decimal = Decimal(4)
    max_held_cny: Decimal = Decimal(20)
    max_technical_failures: int = 3


def load_questions(path: str | Path) -> list[dict[str, str]]:
    """Accept a JSONL file or a directory of <id>/QUESTION.json."""
    path = Path(path)
    if path.is_dir():
        rows = [json.loads(item.read_text(encoding="utf-8-sig")) for item in sorted(path.glob("*/QUESTION.json"))]
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("no_questions")
    questions, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - QUESTION_KEYS:
            raise ValueError("question_rows_may_hold_only_id_and_question")
        ident, text = row.get("id"), row.get("question")
        if not isinstance(ident, str) or not SAFE_ID.fullmatch(ident) or not isinstance(text, str) or not text.strip():
            raise ValueError("invalid_question_row")
        if ident in seen:
            raise ValueError("duplicate_question_id")
        seen.add(ident)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if row.get("question_sha256", digest) != digest:
            raise ValueError("question_hash_mismatch")
        questions.append({"id": ident, "question": text, "question_sha256": digest})
    return questions


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def read_outcome(trial: Path, layer: Layer) -> dict[str, Any]:
    """What one finished trial produced, in the terms routing needs."""
    worker = _load(trial / "WORKER_RESULT.json")
    ledger = worker.get("model_ledger") or _load(trial / "MODEL_LEDGER.json")
    saved = (next(iter(sorted(trial.glob("native/**/convergence_v1.json"))), None)
             or next(iter(sorted(trial.glob("native/**/graph_v1.json"))), None))
    state = _load(saved) if saved else {}
    settlement = state.get("settlement") or {}
    text = str(worker.get("answer") or "")
    returned = worker.get("technical_status") == "pipeline_returned"
    outcome = {"layer": layer.name, "model": layer.model or "deepseek-flash", "level": "none", "kind": "",
               "check": settlement.get("check", ""), "answer": "", "delivered": "", "returned": returned,
               "stop_reason": (state.get("escalation") or {}).get("reason", ""), "calls": ledger.get("calls", 0),
               "estimate_cny": str(ledger.get("spent_estimate_cny") or 0),
               "held_cny": str(ledger.get("unknown_held_upper_cny") or 0), "seconds": worker.get("duration_s"),
               "earlier_refuted": settlement.get("earlier_refuted") is True}
    if not returned or not text or text.startswith(MARKER):
        outcome["kind"] = settlement.get("kind", "") if returned else "technical_failure"
        return outcome
    clean = settlement.get("answer") or state.get("answer") or text
    outcome.update(answer=clean, delivered=text)
    if settlement:
        outcome.update(level=settlement["level"], kind=settlement["kind"])
    elif layer is TEAM_LAYER:
        outcome.update(level="team", kind="team_answer")
    else:
        # A budget stop can leave an answer that nothing has checked.
        outcome.update(level="best_effort", kind="unsettled")
    return outcome


def settled(outcome: dict[str, Any], *, stop_doubted: bool = False) -> bool:
    """The routing rule. Only an answer the independent check supported stops a question.

    The gate decides what may be submitted, so relaxing it must not relax this.
    A doubted answer goes on by default: the 6 of 7 right on DEV30 came from the
    stricter gate. See TIERED_RESEARCH_V2.md.
    """
    return outcome["level"] in {"verified", "team"} or (stop_doubted and outcome["kind"] == "independent_check")


def credibility(outcome: dict[str, Any]) -> int:
    if outcome["level"] == "none":
        return 0
    if outcome["level"] == "verified":
        return 4
    if outcome.get("layer") == GRAPH_LAYER.name:
        # The fallback's unverified answer replaces an earlier one only when both of
        # its checks found the earlier answer contradicted by the research and its own
        # not (rule (a), approved 2026-11-03); otherwise it is delivered only when no
        # earlier layer gave an answer.
        return 3 if outcome.get("earlier_refuted") else 1
    return 2 if outcome["kind"] in {"independent_check", "team_answer"} else 1


def choose(outcomes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A later layer replaces an answer only with a more credible one."""
    chosen = None
    for outcome in outcomes:
        if credibility(outcome) and (chosen is None or credibility(outcome) > credibility(chosen)):
            chosen = outcome
    return chosen


def search_health(trial: Path) -> tuple[int, int]:
    working = refused = 0
    for path in (trial / "web_responses").glob("*"):
        item = _load(path)
        if "serper" in str(item.get("host")):
            working += str(item.get("status")) == "200"
            refused += CREDIT_ERROR in str(item.get("text"))
    return working, refused


def searched(trial: Path) -> bool:
    """Whether the trial sent any search at all."""
    return any("serper" in str(_load(path).get("host")) for path in (trial / "web_responses").glob("*"))


def models_used(trial: Path) -> tuple[set[Any], set[Any]]:
    """Reasoning models and page-extraction models.

    A Pro run's closing synthesis also sends no tools, so extraction is told
    apart by its fixed output cap, not by the missing tools.
    """
    reasoning, extraction = set(), set()
    try:
        lines = (trial / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return reasoning, extraction
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") == "model_request":
            page_read = not event.get("tools") and event.get("max_tokens") == EXTRACTION_MAX_TOKENS
            (extraction if page_read else reasoning).add(event.get("model"))
    return reasoning, extraction


@contextmanager
def run_lock(out: Path):
    """Hold an OS lock on the run folder while this process runs. A crash or a power cut releases it,
    so a resume can tell a cut-off trial from one another router is still running."""
    with (out / "ROUTER.lock").open("a+") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("router_already_running") from None
        yield


def powershell_launch(docker: str) -> Callable[[Path], int]:
    def launch(trial: Path) -> int:
        return subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(trial / "launch.ps1"),
             "-Mode", "execute", "-DockerCommand", docker],
            cwd=HERE.parent.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode
    return launch


class Router:
    def __init__(self, questions: list[dict[str, str]], out: Path, layers: tuple[Layer, ...], limits: Limits,
                 launch: Callable[[Path], int], make_trial: Callable[..., Any] = prepare,
                 stop_doubted: bool = False) -> None:
        self.questions, self.out, self.layers, self.limits = questions, Path(out), layers, limits
        self.launch, self.make_trial, self.stop_doubted = launch, make_trial, stop_doubted
        self.lock = threading.Lock()
        self.spent = self.held = Decimal(0)
        self.failures = 0
        self.stop = ""
        self.balance = None

    def halt(self, reason: str) -> None:
        if self.stop:
            return
        self.stop = reason
        with (self.out / "STOPS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"reason": reason, "utc": datetime.now(timezone.utc).isoformat()}) + "\n")

    def trial_for(self, question: dict[str, str], layer: Layer) -> Path:
        folder = self.out / question["id"]
        trial = folder / layer.name
        if not trial.exists():
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / "QUESTION.json"
            if not source.exists():
                source.write_text(json.dumps(question, ensure_ascii=False), encoding="utf-8")
            self.make_trial(source, trial, profile_file=PROFILES / layer.profile, model=layer.model,
                            hard_cap=layer.hard_cap_cny)
        return trial

    def inspect(self, trial: Path, layer: Layer, outcome: dict[str, Any]) -> None:
        """The batch brakes, applied after each new run. Call with the lock held."""
        if outcome["returned"]:
            self.failures = 0
        else:
            self.failures += 1
        working, refused = search_health(trial)
        reasoning, extraction = models_used(trial)
        after = (_load(trial / "MODEL_BALANCE_AFTER.json").get("balances") or [{}])[0].get("total_balance")
        self.balance = after if after is not None else self.balance
        # A graph trial can end before its research starts, with no search sent; that
        # says nothing about the search service, unlike a light run without a working search.
        if refused or (outcome["returned"] and not working
                       and (layer.name != GRAPH_LAYER.name or searched(trial))):
            self.halt("search_unavailable")
        if "deepseek-v4-pro" in extraction:
            self.halt("extraction_used_pro")
        if outcome["returned"] and outcome["model"] not in reasoning:
            self.halt("main_model_mismatch")
        if self.failures >= self.limits.max_technical_failures:
            self.halt("consecutive_technical_failures")
        if self.held >= self.limits.max_held_cny:
            self.halt("held_limit")
        if self.spent >= self.limits.max_total_cny:
            self.halt("spend_limit")
        if after is not None and Decimal(str(after)) < self.limits.min_balance_cny:
            self.halt("balance_limit")

    def run_layer(self, question: dict[str, str], layer: Layer, earlier: str = "") -> dict[str, Any]:
        trial = self.trial_for(question, layer)
        fresh = not (trial / "STARTED.json").exists()
        if fresh and earlier:
            # The graph trial compares this with its evidence only if it verifies nothing itself.
            (trial / "EARLIER_ANSWER.json").write_text(json.dumps({"answer": earlier}, ensure_ascii=False), encoding="utf-8")
        code = self.launch(trial) if fresh else None
        outcome = read_outcome(trial, layer)
        outcome["launch_exit"] = code
        with self.lock:
            self.spent += Decimal(outcome["estimate_cny"])
            self.held += Decimal(outcome["held_cny"])
            if fresh:
                self.inspect(trial, layer, outcome)
            with (self.out / "PROGRESS.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": question["id"], "fresh": fresh, "utc": datetime.now(timezone.utc).isoformat(),
                                         "balance": self.balance, "cumulative_estimate_cny": str(self.spent),
                                         **{k: v for k, v in outcome.items() if k not in {"answer", "delivered"}}},
                                        ensure_ascii=False) + "\n")
        return outcome

    def route(self, question: dict[str, str]) -> dict[str, Any]:
        outcomes: list[dict[str, Any]] = []
        finished = False
        for layer in self.layers:
            started = (self.out / question["id"] / layer.name / "STARTED.json").exists()
            if self.stop and not started:
                break
            earlier = choose(outcomes) if layer.name == GRAPH_LAYER.name else None
            outcomes.append(self.run_layer(question, layer, earlier=(earlier or {}).get("answer", "")))
            if settled(outcomes[-1], stop_doubted=self.stop_doubted):
                finished = True
                break
        else:
            finished = True
        chosen = choose(outcomes)
        return {"id": question["id"], "finished": finished, "layer": chosen["layer"] if chosen else "",
                "model": chosen["model"] if chosen else "", "level": chosen["level"] if chosen else "none",
                "answer": chosen["answer"] if chosen else "", "delivered": chosen["delivered"] if chosen else "",
                "path": [{k: v for k, v in item.items() if k not in {"answer", "delivered"}} for item in outcomes],
                "estimate_cny": str(sum((Decimal(o["estimate_cny"]) for o in outcomes), Decimal(0))),
                "held_cny": str(sum((Decimal(o["held_cny"]) for o in outcomes), Decimal(0)))}

    def recover_interrupted(self) -> None:
        """Keep each cut-off trial aside so it runs again. What it spent still counts, and its calls in
        flight at the cut count as held, since their cost is unknown."""
        for started in sorted(self.out.glob("*/*/STARTED.json")):
            trial = started.parent
            if (trial / "WORKER_RESULT.json").exists() or (trial / "FINISHED.json").exists():
                continue
            target, copy = self.out / INTERRUPTED / trial.parent.name / trial.name, 1
            while target.exists():
                copy += 1
                target = self.out / INTERRUPTED / trial.parent.name / f"{trial.name}_{copy}"
            target.parent.mkdir(parents=True, exist_ok=True)
            trial.rename(target)
            with (self.out / "INTERRUPTED.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"trial": f"{trial.parent.name}/{trial.name}", "kept_as": target.relative_to(self.out).as_posix(),
                                         "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        for path in sorted((self.out / INTERRUPTED).glob("*/*/MODEL_LEDGER.json")):
            ledger = _load(path)
            self.spent += Decimal(str(ledger.get("spent_estimate_cny") or 0))
            self.held += (Decimal(str(ledger.get("unknown_held_upper_cny") or 0))
                          + Decimal(str(ledger.get("inflight_reserved_upper_cny") or 0)))

    def run(self, workers: int) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        with run_lock(self.out):
            return self._run(workers)

    def _run(self, workers: int) -> dict[str, Any]:
        self.recover_interrupted()
        with ThreadPoolExecutor(workers) as pool:
            records = list(pool.map(self.route, self.questions))
        with (self.out / "ROUTER_RESULTS.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (self.out / "ANSWERS_FOR_JUDGING.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps({"id": record["id"], "run": f"{record['layer'] or 'none'}_{record['id']}",
                                         "model": record["model"], "level": record["level"],
                                         "answer": record["answer"]}, ensure_ascii=False) + "\n")
        count = lambda key: {value: sum(r[key] == value for r in records) for value in sorted({r[key] for r in records})}
        summary = {"questions": len(records), "finished": sum(r["finished"] for r in records),
                   "by_level": count("level"), "by_layer": count("layer"), "estimate_cny": str(self.spent),
                   "held_cny": str(self.held), "balance_after": self.balance, "stop": self.stop,
                   "layers": [layer.name for layer in self.layers], "stop_doubted": self.stop_doubted}
        (self.out / "ROUTER_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="JSONL of id and question, or a directory of <id>/QUESTION.json.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-total-cny", required=True, type=Decimal, help="Ledger estimate that stops new runs.")
    parser.add_argument("--min-balance-cny", default=Decimal(4), type=Decimal)
    parser.add_argument("--max-held-cny", default=Limits.max_held_cny, type=Decimal,
                        help="Held upper bound that stops new runs. Finished runs count on a resume.")
    parser.add_argument("--with-fat", action="store_true", help="Add the team as a last layer.")
    parser.add_argument("--with-graph", action="store_true", help="Add the relation-graph fallback as a last layer.")
    parser.add_argument("--stop-doubted", action="store_true", help="Also stop on a gate-passed answer the check doubted.")
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--dry-run", action="store_true", help="Check the input and print the plan; start nothing.")
    args = parser.parse_args()
    questions = load_questions(args.questions)
    if args.with_fat and args.with_graph:
        parser.error("choose one last layer: --with-fat or --with-graph")
    layers = (*LAYERS, TEAM_LAYER) if args.with_fat else (*LAYERS, GRAPH_LAYER) if args.with_graph else LAYERS
    plan = {"questions": len(questions), "layers": [layer.name for layer in layers], "max_total_cny": str(args.max_total_cny),
            "max_held_cny": str(args.max_held_cny), "out": args.out, "workers": args.workers,
            "stop_doubted": args.stop_doubted}
    if args.dry_run:
        print(json.dumps(plan))
        return
    router = Router(questions, Path(args.out), layers,
                    Limits(args.max_total_cny, min_balance_cny=args.min_balance_cny, max_held_cny=args.max_held_cny),
                    powershell_launch(args.docker),
                    stop_doubted=args.stop_doubted)
    print(json.dumps(router.run(args.workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
