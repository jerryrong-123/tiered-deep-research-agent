# Tiered research v1 (explicit experiment)

Select `local_profiles/tiered_research_v1.yaml`. It extends progressive research
v1: a run starts on the light path and moves to the team only when the light
path cannot support an answer. Difficulty is not predicted before the run.
Profiles without `tiered_escalation: true` keep their previous behavior.

## Light path exit

`review_research(action="finalize")` needs `requirements`: rows of
`question_span`, `receipt_id` and `evidence_quote`.

- Each span must be verbatim question text, at most `max_requirement_words`.
- Each quote must occur in a successful saved receipt. The cited receipt is tried
  first; a match elsewhere passes and is logged in `receipt_corrections`. The
  match ignores case, punctuation and markdown, and `...` separates passages.
- The spans must cover `min_question_coverage_pct` of the question's content
  words.

A rejection returns `uncovered_question_parts`, computed by the program from the
question it holds. These are the gaps the program finds; the coordinator may
research them, delegate them through `gap_tasks`, or escalate.

These checks show traceability, not truth. A real quote can still fail to support
its condition, and a poor ledger can pass. Plain-text answers with no saved
evidence are refused on this path.

## Escalation

The light loop stops and the team loop starts, in the same run, when:

| Trigger | `reason` |
|---|---|
| `review_research(action="escalate")`, reason in `disposition` | `coordinator_requested` |
| `escalate_after_gate_rejections` rejected finalizations | `requirement_gate_exhausted` |
| The same count of unsourced plain-text answers | `unsourced_answer` |
| `escalate_after_main_turns` effective main turns | `light_turn_budget` |
| `light_hard_calls` model calls and page reads, helpers included | `light_call_budget` |

Main turns alone do not bound cost, because helpers have their own turns. At
`light_soft_calls` the coordinator is told to finish or escalate and no new helper
can start; running helpers continue. At `light_hard_calls` helper model calls are
refused and the coordinator stops at its turn end. The team stage is not held to
either limit.

The team stage uses the `convergence_v1` coordinator prompt and
`team_main_agent_tools`. Model, budget meter, hard cap, AgentBus, task board and
receipts are shared. Helpers already running stay on the bus and report to the
team coordinator. The wall-clock deadline passed to the team stage is what
remains of the run's. The team coordinator receives the question plus a bounded
handoff, and its dispatch starts from the light path's open gaps.

`convergence_v1.json` records `escalation` with the reason, the open gaps and a
`light_stage` snapshot. A team stage writes `main_agent_team` trajectories.

## Light-stage measurement

`local_profiles/tiered_light_stage_v1.yaml` differs only by
`stop_at_escalation: true`. When a trigger fires, the run ends without a team
stage or a synthesis call, and its answer is `[ESCALATION_REQUIRED] reason=...`,
with `stopped_by=escalate_to_team`. Count that as "needs escalation", never as a
wrong answer. Use it to learn which questions the light path can finish before
paying for the team. A rejected finalization keeps its `candidate_answer` in the
`requirement_gate_rejected` event for offline analysis; it is not sent to the team.

## Trial values, not calibrated thresholds

| Setting | Value | Basis |
|---|---:|---|
| `min_question_coverage_pct` | 95 | Dropping any one condition left at most 93% on 0810, 0220 and 0049 |
| `max_requirement_words` | 30 | Longest clause in those questions fits; one span cannot cover a whole question |
| `escalate_after_gate_rejections` | 3 | Allows two corrections after a rejection names what is missing |
| `escalate_after_main_turns` | 24 | All 21 correct first attempts of the Pro + Serper baseline used at most 24 model calls; Flash needs its own calibration |
| `light_soft_calls` / `light_hard_calls` | 60 / 90 | The 12 helper-free runs needed 15 to 50 calls, and answered runs at most 34; the one run with helpers reached 217 |
| `MIN_QUOTE_WORDS` | 3 | Blocks a single matching word |
| `HANDOFF_RECEIPT_CHARS` / `HANDOFF_TOTAL_CHARS` | 800 / 12000 | Bounds handoff size to about 3,000 tokens |

Calibrate these on the paired 30-question comparison before relying on them. A
question with long format instructions may need more than the coverage allows;
it then escalates, which costs money but loses no answer.

## Evaluation boundary

`local_runner/answer_label_guard.py` v5 applies to every local trial, tiered or
not. It withholds text that copies a run of at least 20% of the question (minimum
8 words) and labelled search entries, quarantines a withheld source by title and
URL for the rest of the run, withholds requests that ask for a benchmark or
reference answer, and screens receipts before they are stored. One shared phrase
no longer withholds a page, because real sources can share a phrase with the
question. In saved runs the longest shared run was 7-16% of the question for
sources and 29-100% for copies. Detection remains limited, so post-run
contamination review is still required.

## Polluted DNS

On a network whose resolver injects `2001::/32` AAAA answers, a public name looks
non-public and `web_fetch` refuses it before the reader receives the URL. Set
`FRONTIER_AGENT_TOLERATE_DNS_POISON=1` in the local `runtime.env` to ignore that
range when a public IPv4 answer is also present. It is off by default, and every
other non-public address is still refused. It does not make such a page readable
by itself: the reader still has to reach it.

## Prepare, do not execute implicitly

Use `local_runner/prepare.py` with `--profile-file
frontier_research/local_profiles/tiered_research_v1.yaml`. `-Mode test` is
offline. `execute` starts a paid attempt and needs a finite authorized scope.
Preparation grants nothing. No online result for this profile exists yet, so no
accuracy or cost claim can be made from it.
