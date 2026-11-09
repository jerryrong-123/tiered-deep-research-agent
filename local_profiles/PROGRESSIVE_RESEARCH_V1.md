# Progressive research v1 (explicit experiment)

Select `local_profiles/progressive_research_v1.yaml`. The production/default
profiles retain their previous behavior. The new flag uses the native Agent
Team main loop, AgentBus and convergence state; it does not restart another
pipeline or load the old strict question-review architecture.

The main agent searches and reads directly. Original web search/fetch
implementations, model, thinking, output, retries and timeout settings stay as
declared in the existing convergence profile. A successful source read does
not create an extra model/review stage. The coordinator can finish using its
own sources, without a compulsory helper or final verifier.

## Runtime control

- Real main tool results and actual child reports share convergence receipts.
  A receipt means a result was received, not that its claims are true.
- Receipt IDs are announced separately; web tool JSON/text is unchanged.
  The exact public tool-result body is saved under the shared workspace's
  `.research_receipts/` when available. A result may be an extraction or summary;
  do not call it the original webpage. The runner separately retains reader
  responses and provenance.
- Before dispatch, call the existing `review_research(action="continue")`
  with factual `gap_tasks`. Each has `gap_id`, `requirement`, `subject`,
  `question`, `source_receipt_ids`, `depends_on`, and optional factual `kind`.
  The runtime checks current receipt references and task dependencies. Whether
  the proposed gap is semantically relevant remains the coordinator's judgment.
- Assignments use `tasks[].gap_id`; the runtime injects its approved gap and
  selected saved sources, then calls the native assignment implementation.
  Native running and queued jobs carry the gap metadata. An admission lock
  spans enqueue. Changing agent name/gap ID cannot duplicate an identical
  active scope. Paraphrase equivalence is not deterministically guaranteed.
- Helpers cannot recursively delegate. A single-gap investigation admits at
  most one helper; at least two independent unresolved gaps permit at most two
  running-plus-queued jobs across the run. Dependency declarations are checked,
  not invented by code. The main agent may solve any gap itself.
- `review_research(action="finalize")` freezes further model/tool research,
  uses the native cancellation/drain path and delivers its answer without a
  separate synthesis call. Pending child reports must be reviewed. Previously
  declared substantive conflicts require a disposition, not automatic deletion.

## Trial thresholds, not correctness criteria

`progressive_config` defaults are `review_after_main_turns: 6`,
`no_progress_cycles: 2`, `max_helpers_single_gap: 1`,
`max_helpers_parallel: 2`. Main effective turns exclude physical retries and
auxiliary summarizer loops. Fresh planning/execution loops count separately.

The first two numbers request a strategy review only; they do not force a
helper, reject an answer, or conclude that the task is difficult. A cycle is an
observed successful search followed by reading. Relevant new material can be
acknowledged with `progress_summary` and current `progress_receipt_ids` during
an existing review decision. Identical bodies do not count as new material;
this bookkeeping observation does not veto valid continuation. Technical/tool
errors do not count as semantic difficulty. Changing query text alone does not
prove progress.

These values are initial experiment settings. Historical Pro call-count
statistics do not calibrate Flash timing or prove the settings optimal. Use
fresh same-configuration trials and held-out DEV questions before claiming
accuracy or cost improvement. Do not use a confidence percentage as a gate.

## Prepare, do not execute implicitly

From the project root, with an existing Python runtime:

```powershell
python frontier_research/local_runner/prepare.py --question-json <three-key-question.json> --out <fresh-output-directory> --profile-file frontier_research/local_profiles/progressive_research_v1.yaml
```

The result's `launch.ps1 -Mode test` runs adapter checks offline. `execute`
starts a paid attempt and requires an authorized finite scope. Preserve the
shared total model budget, existing finalization allowance, unknown held costs
and STOP history; preparation does not authorize more calls. External services
remain metered separately. Recheck that pricing verification covers the actual
next attempt; historical verification windows are not extended by this profile.
