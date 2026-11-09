# Maintained native Agent Team: first optimization batch

Upstream: ApodexAI/FrontierAgent commit
`e4cb2397109c957a2f99c3a5d850b9fc479dbef6`, core `apodex-agent-core==0.14.1`.
The original 780-file checkout, previous runs, raw responses and fee records
remain under `../outputs/`. This directory is an independent maintenance copy.

## Implemented behavior

- The original Serper `web_search` path distinguishes a successful empty result
  from HTTP, timeout, transport, invalid JSON and explicit provider failures.
  A mixed batch retains successful queries. Failure metadata is a safe local
  reason and optional actual HTTP status; no service response body is copied
  into that diagnostic. The public tool schema and successful response remain
  unchanged.
- `agent.share_search_results: true` enables reuse of successful raw Serper
  searches only during one coordinator run and its child sessions. The key
  includes endpoint, credential fingerprint and the exact canonical request.
  Reads/writes copy data; failures, empty results and cancellations do not
  poison the cache. The existing single-flight and retry behavior is retained.
  Root completion/cancellation invalidates the shared state and clears it.
- Source evidence, semantic support and answer correctness are not inferred
  from a cache hit. Existing source retention and raw-page cache stay in place.
  Search hits are navigation, and repeated hits are not independent evidence.
- The shared sandbox adapter uses the installed E2B SDK's `stdin`/background
  handle interface when that backend has no `input` parameter. It sends the
  script, closes stdin with EOF and waits with the existing command timeout.
  Local backends keep their existing direct-input behavior. Send, EOF, wait
  failures and cancellation attempt remote cleanup; the worker disconnects
  its stream, and cleanup failures retain a safe warning. No reader bundle
  is embedded in the command line. This is verified against installed SDK
  signatures offline; a new live E2B execution has not been performed.

The optional cache covers `web_search_impl: original` with Serper, as selected
by the comparison profiles. The alternate aligned/Parallel implementations
were not changed. The cache keeps the first successful response for the run;
it is not a guarantee of fresh search-engine rankings during that run.

## Explicit comparison configurations

The original `benchmark`, `simple` and `tui` profiles stay byte-identical.
New configurations live in `local_profiles/` and use the original inline-profile
loader instead of expanding or replacing the native profile namespace.

| Configuration | Change from benchmark |
|---|---|
| benchmark | Native max orchestration; search reuse off |
| effort_high_v1 | Only `agent.team_effort: high` |
| shared_search_v1 | Only `agent.share_search_results: true` |
| integrated_v1 | Both changes; an interaction configuration, not a single-factor test |

`team_effort` controls team expansion. It does not lower model thinking.
Model, output limits, retries, timeouts, reporter and tools otherwise remain
the existing trial configuration. Existing frozen DeepSeek overrides are
applied after inline loading by the prepared trial adapter.

## Offline validation and preparing a trial

The implementation report and reproducible, network-disabled suite runner are
in `../outputs/frontier_optimization_20261028_01/`. That directory also contains
an offline `prepare_trial.py`. It accepts a question-only JSON, a new output
directory, one comparison profile, and an explicit model-cost mode.

Preparation reuses the previous provider/transport observation, source hashing,
supervisor, fee accounting and sandbox cleanup code in a new directory.
It creates a new source fingerprint and question hash; it does not call services.
The generated `launch.ps1 -Mode test` runs HTTP fixtures with `--network none`.
`-Mode execute` starts one real trial and refuses to overwrite an existing start.

The existing lexical answer-contamination guard is retained by the adapter.
Its known paraphrased-snippet leak is not fixed in this batch. A completed answer
must still be evaluated for correctness, evidence and answer exposure separately.
Offline interface tests do not establish a fresh-model accuracy improvement.

No HTN orchestrator, old question-admission protocol, new semantic refusal gate,
or additional model stage has been imported.

## Opt-in convergence V1

`local_profiles/convergence_v1.yaml` adds only `agent.convergence_v1: true` to
`shared_search_v1`. All original profiles remain unchanged. Each root run owns
its own controller, inherited by its native child tasks and isolated by its
AgentBus run ID. Report receipts retain the native status, exact public body,
body hash and actual provenance metadata. A completed report is not a semantic
verification result.

After a report arrives, the next existing coordinator turn uses one control
tool, `review_research`. It either names concrete evidence gaps or supplies the
final sourced answer. New research/delegation is blocked until that decision;
approved follow-ups receive the explicit gaps. Same-batch continuation and
research cannot race past this admission boundary. A final decision needs a
real receipt, no current declared gaps, and a disposition for earlier gaps or
counterevidence. These are consistency checks on the decision, not proof that
the model has understood the question correctly. All received reports remain
available, including sparse, failed and contradictory reports.

On finalization, new child/auxiliary model calls and research tools are blocked.
The existing AgentBus queues and running jobs are cancelled before final
delivery. Remaining board items are withdrawn as cancelled, not marked as
verified. Already received evidence is preserved. An answer already contained
in the review decision is delivered through the native final-answer path
without another synthesis call. Continuation can use another ordinary
coordinator turn; no fixed reduction in model calls or cost is claimed.

Cleanup has the existing 45-second outer and 10-second native bounds. Its
confirmation describes local runtime cleanup only: already-sent provider I/O,
threads or remote sandbox work may outlive cancellation and may still cost
money. Unknown fees remain held. Late reports are retained without restarting
research; reports already ready before the final decision are drained for
review before committing.

The fresh offline preparation tools are in
`../outputs/frontier_convergence_20261029_01/`. The new bounded adapter protects
one existing final-response admission **inside** the unchanged CNY 25 hard cap;
CNY 20 remains a soft target. The default protected upper amount is CNY
2.490368, covering the existing 32768-token request and native 49152-token
fallback bound. It changes admission scheduling, not model/output/retry/time
settings. Unknown remote costs retain their reservations; known-unsent
requests release theirs. The previous conservative pricing-date policy is
explicitly retained and is not a verified current invoice or tariff.

Tests use the actual coordinator node, AgentBus and production loops with
scripted model/HTTP boundaries. The historical public-report replay labels its
injected sufficiency decision explicitly. None of these offline checks proves
that a fresh model will recognize a sufficient answer or improve accuracy.
