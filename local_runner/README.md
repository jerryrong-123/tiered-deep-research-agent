# Native local runner v1

This maintained adapter is derived from the 12 source/config/test files in
`outputs/frontier_two_case_trial_20261030_01/trial_0810`. `TEMPLATE_ORIGIN.json`
records their original hashes. No prior question, answer, research trace,
ledger or credential is a template input.

Prepare a fresh trial from the repository root:

```powershell
python frontier_research/local_runner/prepare.py --question-json <question-only.json> --out <new-trial-directory> --profile-file <explicit-profile.yaml>
```

The question JSON must contain exactly `id`, `question`, `question_sha256`.
Omit `--profile-file` to retain the original convergence profile. Select the
new progressive profile explicitly when preparing that experiment. Preparation
makes no service calls, starts no container/model and grants no paid-run
authorization. Existing destination directories are refused.

The generated `launch.ps1 -Mode test` runs the existing mocked adapter checks
in the pinned image with `--network none` and without `--env-file`. Execute and
cleanup retain the existing explicit modes and credential configuration; they
require the user's separate run authorization. Model, thinking, output, native
retries, deadline, pricing interval and bounded budget policies are retained.
The historical pricing interval is not silently extended.

The source manifest is generated from the current checkout at preparation.
Existing `check_source` runs before execution and checks each enumerated file;
this does not claim detection of additional unlisted files.

`tool_classification.py` is the one internal control/receipt tool classification
used by the answer source guard. Internal routing does not certify claims or
answers. External and unclassified tool results retain the source guard.

Event appends serialize open/write/close across this worker's threads. The
supervisor writes after its worker terminates; concurrent independent processes
writing the same trial are unsupported. Each actual sandbox creation registers
its returned id in a separate atomically replaced, fsynced registry record
before emitting the observational event or returning to the caller. Failure to
register attempts immediate SDK cleanup and propagates the failure.

Supervisor cleanup uses registry ids plus valid legacy event rows, records bad
line/file positions and hashes, and continues after individual parse/kill
failures. Confirmed registry cleanup prevents repeated fallback kills. Raw bad
log lines are preserved. A provider resource whose id was never successfully
returned, or a sandbox whose registry and event records are both lost, cannot
be reconstructed from local evidence.
