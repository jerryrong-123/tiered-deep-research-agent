# Tiered research v2 (explicit experiment)

Select `local_profiles/tiered_light_stage_v2.yaml`. It keeps the tiered v1 flow
and changes how a light-stage run ends. Profiles that do not set these controls
behave as before, and the defaults in `TIER_DEFAULTS` leave every rule off.

## Why

In the DEV30 trial no stopped run ever called finalize. Runs that already held
the answer kept researching background clues to reach 95% coverage until the
budget ended, and the stop then wrote no answer. See
`outputs/pro_tier_11/BATCH_REPORT.md`.

## Rules and parameters

These are trial values, not calibrated thresholds.

| Rule | Control | Value | Basis |
|---|---|---:|---|
| 1. Trace what is asked and the links to it | `answer_chain_gate` | `true` | Background clues caused the over-research |
| | `min_ask_coverage_pct` | 80 | Most of the asking sentence must be traced |
| | `min_question_coverage_pct` | 50 (v1: 95) | Enough conditions to tie the answer to the question |
| 2. Label an answer when research stops | `synthesize_at_stop` | `true` | One call on saved evidence, as with an agent's final pass at its limit |
| 3. Count searches per query | `max_search_queries` | 40 | Above the 37 searches of the busiest answered Flash run, so no finishing run is cut |
| 4. No named candidate before evidence | prompt text, with `answer_chain_gate` | | One Pro run stayed on a name it assumed in its first plan |
| 5. Withhold what the leak audit flags | guard `question_source_guard_v8` | | Dataset pages slipped under the 20% question share; v8 adds the audit's A/B/C rules and shell commands |
| 6. Show social results in search display | `show_low_value_results` | `true` | The built-in filter hid 9.3% of raw results, and a Facebook result carried a key lead |
| 7. Take quotes off single ordinary words | `unquote_single_words` | `true` | On 0049 quoting one word hid the page that used a synonym; taking only those quotes off reached it in 1 of 16 failed queries |
| 8. Read a saved copy when the live page gave nothing | `archive_reads` | 10 | On 0330 the live page had lost its data and on 1092 a security check blocked 7 pages; a run met about 3 extractor misses and 1 failed ordinary page |
| Independent check | `independent_check` | `true` | The gate checks that quotes exist, not that the answer is right |

Rule 5 applies to every local trial, because the guard belongs to the runner and
not to a profile. It names kinds of host and never a question's own source:
`huggingface.co/datasets/`, `kaggle.com/datasets/`, `· Datasets at`, `eval logs`,
`evaluation logs`, `simple-evals` and `browsecomp`.

Since v7 (2026-11-04), a page read whose URL is such a source, or one already
quarantined, is refused before the reader fetches it. Under v6 the page went
through the reader and the page summarizer, itself a model, and was withheld
only from the agent afterwards. Hugging Face datasets and spaces were already
refused by the fetch tool itself, but the other kinds were not.
`test_fat_full_path_offline` in `local_runner/test_native.py` found the gap. It
drives the team layer through the real runtime on scripted replies:
- the coordinator creates and assigns a helper;
- the helper searches, reads pages and reports;
- the coordinator finalizes through `review_research`.

It checks that no model request holds an evaluation entry, the result of an
answer-seeking search, or an evaluation page.

Since v8 (2026-11-05), the guard withholds whatever the post-run leak audit
flags (`bla11/audit_abc.py`, `local_abc_v1_20261031`). The 11-04 FAT reruns
were all `exposed`: some views hit the audit's rules, though none held the
reference.
- **A:** any page on a listed host, such as all of `huggingface.co` and `kaggle.com`, or with a listed marker in its URL or title, such as `trajectories`.
- **B:** a run of question words reaching 15% of the question, the audit's gray band. Before, the rule was 20% with an eight-word floor. Some real sources reach this share and are withheld too.
- **C:** the audit's labels as well as the guard's, in page reads and shell output too, not only in search entries.
- **Shell and downloads:** a `bash` command or `download_file` request that names such a host, an answer-seeking phrase or a quarantined URL is refused before it runs. Shell output is still screened for B and C afterwards.
- **Recovered results:** `recover_result` returns the unscreened original, so its search entries are screened like search results.
- **Markers:** both withheld-text markers are worded without an audit label.

The guard then uses the same rules as the audit. A `clean` audit means no
flagged view reached a model, not that the agent found its answer
independently. A source can also slip past both: a shell command that builds
its URL at run time, for example.

`test_fat_full_path_offline` adds a page on an audit host, a snippet at the
15% share, a labelled page body and a shell download of an evaluation file.
The shell download must be refused before a sandbox is asked for.

Rule 6 keeps operator hard
blocks in force. Existing limits stay as in v1: 24 effective turns, 3 rejected
finalizations, 60 and 90 calls.

Rule 7 applies to lowercase words only: `"physiotherapist"` loses its quotes,
while `"Wootton"`, `"2005"` and `"first novel"` keep theirs. The search shows the
query it actually ran.

Rule 8 tries the newest saved copy in the Internet Archive when a page failed,
came back too short to be rendered, or the extractor gave nothing or opened by
saying the asked fact is not there. Lookup and reading need no key, except that a saved PDF goes
through the reader. The extraction costs what any page read costs, and each saved
page counts as one light call. Search results, images and
pages that already are saved copies are skipped. When the live page only lacked
the fact, the copy replaces it only if the copy has the fact, and an empty
extraction of a copy is never delivered. The result ends with an `[ARCHIVE COPY]`
note giving the date and address of the copy.

Rule 8 reads the newest copy only, and that copy can be poorer than an older one.
The guard screens a saved page like any other, and the runner records saved pages
in `web_responses` for the source audit.

## Evidence levels

| Level | Meaning |
|---|---|
| `verified` | Passed the gate, and the independent check returned `supported` |
| `best_effort` | Passed the gate but the check did not support it, or the run stopped and the evidence backs a named answer |
| `none` | The run stopped and the evidence backs no answer; the marker stays |

The level says how far an answer was checked, not that it is right. A forced
answer needs one quote found verbatim in a successful receipt. Otherwise it
counts as `none`. A failed check keeps the answer at `best_effort`.

Score levels separately. The model's self-reported confidence is recorded but
decides nothing.

## Routing

`local_runner/tiered_router.py` takes a file or directory of questions and needs
no per-question setting. Every question starts on the first layer. The run's own
result decides the next step, and input rows may hold only `id`, `question` and
`question_sha256`, so a reference answer cannot enter routing.

| Result of a layer | Next step |
|---|---|
| `verified` | Stop |
| Passed the gate but the check doubted it, `best_effort` | Next layer |
| Forced `best_effort`, an unchecked answer, or `none` | Next layer |
| Technical failure | Next layer |
| Team answer | Stop |

The gate decides what may be submitted and the independent check decides what
ends a question, so relaxing the gate does not relax the stop. A later layer
replaces an answer only with a more credible one: `verified`, then a gate-passed
or team answer, then a forced or unchecked answer. On a tie the earlier answer
stays, because on 0007 a later forced answer that failed its quote check was
graded false while the earlier one was right. An empty last layer keeps the best
earlier answer and its label. No layer sees an earlier layer's candidate, which
keeps the later run an independent second opinion.

DEV30 accuracy under the older 95% gate: `verified` 16 of 16, doubted 6 of 7,
forced 4 of 7, or 6 of 7 with the name-form adjudication in
`outputs/router_stage3/BATCH_REPORT.md`. Doubted answers still escalate because
the relaxed gate changes that group and the 6 of 7 does not carry over.
`--stop-doubted` restores stopping on them, once data shows Pro does not fix more
of them than it breaks. A wrong answer the check supports still ends the question,
as on 1092.

| Layer | Profile | Model | Ledger cap per run |
|---|---|---|---:|
| `flash` | `tiered_light_stage_v2.yaml` | deepseek-flash | sealed ¥25 |
| `pro` | `tiered_light_stage_v2_deep.yaml` | deepseek-v4-pro | sealed ¥25 |
| `fat`, only with `--with-fat` | `convergence_v1.yaml` | deepseek-flash | ¥10 |
| `graph`, only with `--with-graph` (V2.5) | `graph_fallback_v1.yaml` | deepseek-v4-pro | ¥50 |

The deep profile changes only the budget: 80 search queries, 40 effective turns,
40 and 60 calls. Across 20 Pro runs every pass took at most 28 of these calls, and
every other run reached 39 or more without passing. The 23 right baseline answers
took at most 34. The limits leave room above both, because a tiered run took up to
3.1 times the baseline's calls on a question. These are trial values and need
testing on questions that did not set them.

Caps are ledger estimates, about twice the real offpeak charge, and `prepare.py`
refuses a cap above the seal, ¥25 until V2.5 raised it to ¥50 for the graph
layer. Flash and Pro keep ¥25 because held
amounts count against it without being charged: one correct Flash run reached
¥17.62 that way, so a tighter cap would cut off answers.

Since V2.3.2 a call cut before its usage arrives, by the reasoning watchdog or a
dropped stream, holds a bound from the request body it sent, not the full-window
reservation. Byte-level BPE gives every token at least one byte, so the body bounds
the prompt tokens, with 4,096 tokens and 64 per message added for the chat
template. Output stays at its cap, the bound never exceeds the reservation, and
admission still reserves the full window. In the 30-question V2.3.1 run, cut calls
held ¥3.24 to ¥3.69 each on Pro and closed 0341, 0554 and 0604 before their 40
turns, while a balance reading put their real cost near ¥0.18 each. Replayed on
the stored requests, the bound keeps 0554 and 0604 open where they closed. 0341's
context was already near Pro's 262,144-token window, so it would still close. The
stored requests omit the replies' reasoning, so live bounds on later turns are
higher than the replay's.

The run brakes stop new runs and let runs in flight finish: search unavailable,
extraction on Pro, wrong main model, 3 technical failures in a row,
`--max-held-cny` held (¥20 by default), the required `--max-total-cny`, and a
balance under `--min-balance-cny`. A started trial is never repeated, so the same
`--out` resumes after a brake. Finished runs still count on a resume, so a resume
after `held_limit` needs a higher `--max-held-cny`.

The exception, since V2.3.2, is a trial the machine cut off, which left neither
`WORKER_RESULT.json` nor `FINISHED.json`. The router keeps it under
`_interrupted`, records it in `INTERRUPTED.jsonl` and runs it again. What it spent
still counts, and its calls in flight count as held. An OS lock on `ROUTER.lock`
refuses a second router on the same `--out`, so a trial another router is still
running is never taken for a cut-off one.

Results are `ROUTER_RESULTS.jsonl`, `ANSWERS_FOR_JUDGING.jsonl`,
`ROUTER_SUMMARY.json`, `PROGRESS.jsonl` and, after a brake, `STOPS.jsonl`.

## Closing step

Each run makes one closing call after cleanup, because the budget layer admits
one final request per run:

- A stopped run makes one synthesis on the main model from bounded saved evidence:
  80,000 characters in all, 8,000 per page or report and 1,200 per search entry.
  Pages and helper reports come first, then search entries, newest first and one
  per URL.
- A gate-passed answer gets one check on the page-extraction model. It sees only
  the question, the answer and the quoted rows, so a Pro run is still checked by
  a different model and context.

`convergence_v1.json` records the outcome under `settlement`.

## V2.4: a calculator and answer leads

Both are on in the two tier profiles and off by default.

`calculate` is a local tool for the main model and helpers. It parses one
expression against a whitelist: numbers, arithmetic, `abs`, `round`, `min`,
`max`, `sqrt`, trigonometry in degrees and `distance_km` by the haversine
formula. It runs in the process, with no network, sandbox or model call. The
result names its inputs and is kept as a receipt, but it is not a source: it
neither counts as a read nor answers a search. With `calculator: true` the
prompt asks for it on every comparison or total the answer depends on. The gate
accepts a row citing a calculation only when each input number appears in the
question or in another row's quoted page; common conversion factors are exempt.
The independent check sees a note only when a quote is a calculation.

With `answer_leads: N` the page summarizer also gets the question and may end
with one `ANSWER LEAD:` sentence copied from the page. The program keeps it only
if the sentence is in the page, then one Flash call checks it against the page
text around it. Only a lead checked as `answers` or `partial`, with quotes found
in that text, reaches the research model; the raw line never does. A run may make
N checks, one per page and sentence, shared with helpers. A check uses no turn,
light call or search query, but it is a metered call: the ledger counts it in the
totals and inside the per-run cap, and shows it on its own `lead_check` line.
`convergence_v1.json` records leads offered, dropped, checked and their verdicts.

Why: on 0640 the main model, which had no way to calculate, judged one stop
"clearly closer" from coordinates it had read; on 0007 the summarizer, told only
what to extract and never shown the question, dropped a page's answer. Both are
observations from one run, and V2.4 is measured on new questions.

V2.4.1 passes the question through the trial's answer-label guard, which wraps
the summarizer. Under V2.4 the guard rejected the new argument, so every page
read in a trial failed while the offline tests, which ran without the guard,
passed. Two tests now send the summarizer and a lead through the guard.

## V2.5: the relation-graph fallback

An optional last layer, `graph`, switched on with `--with-graph` and never
together with `--with-fat`. It runs only on a question Flash and Pro left
without a verified answer. The design, its five decisions and their dates are in
`outputs/fallback_controller_design_20261103/DESIGN.md`. Flash, Pro and their
rules do not change.

`graph_fallback_v1.yaml` sets `agent.graph_fallback_v1: true`, and the
coordinator loop hands over to a controller (`workflows/agent_team/graph_fallback*.py`):

1. Three independent decompositions into a relation graph, each from the
   question alone and from a different angle.
2. A merge that must account for every disagreement among the drafts.
3. Up to three rounds of checks: structure, back-translation, and Pro and Flash
   reviews. A dispute that outlives the rounds becomes a branch, at most four.
4. Two sub-agents per relation, in dependency order, each told not to search the
   whole question. A quote counts only where the sub-agent's own tool output
   holds it.
5. Candidates intersect per node. A quoted yes against a quoted no goes to an
   arbiter. A stuck node gets one more check round, then a replan.
6. Two per-relation checks, by Pro and by Flash, and a re-verifier with
   different searches. All three must pass.

The router counts the layer's answer only when it is verified. Anything else is
a side answer and never replaces an earlier one.

Every structural cap is written out in the profile, as upper bounds:

| Cap | Value |
|---|---|
| Sub-agents per question | 16, at most 4 at once |
| Each sub-agent | 20 turns and 25 searches; past the cap a search is refused, not run |
| Replans | 2 |
| Rechecks during research | 2 |

All branches share the 16 sub-agents.

The cap per question is ¥50, of which ¥8 is held back for the final checks:
research sub-agents stop at their next call once spent, held and in-flight
reach ¥42. `prepare.py` now seals at ¥50. Trials that pass no cap, Flash and
Pro among them, keep `trial_helpers`' default of ¥25.
