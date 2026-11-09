# Portable HTTP benchmark and compatibility preflight

Implemented 2026-11-09. This milestone prepares the measurement client for other
local engines. It does not change inference, download a model, or establish
compatibility with an external engine. The first llama.cpp comparison remains
pending under the [experiment plan](../research/engine-comparison/EXPERIMENT_PLAN.md).

Follow-up: [the pinned llama.cpp CPU configuration](llama-cpp.md) now passes this
client's required-usage preflight and smoke workload. Comparative experiments
remain pending; the statements below record the adapter milestone's original scope.

## How it works

The client first selects a model, optionally checks a readiness endpoint, and sends
one real streaming request using the benchmark prompt and output limit. That
preflight must finish correctly before warmup or measurement starts. Its request
index is `-1`, and its timings are reported separately.

The default `runtime` profile collects `/health` JSON and discovers models through
`/v1/models`. A non-JSON readiness body can also be recorded. If several models are
advertised, choose one with `--model`; the runtime profile verifies that selection
against discovery. The portable profile skips readiness unless configured and,
with an explicit model ID, skips discovery entirely. The streamed response still
has to identify the requested model consistently.

Check our running runtime without measuring a workload:

```bash
python -m llm_runtime.http_benchmark --preflight-only
```

For another already running local chat server, replace the port and model ID with
its actual configuration. This example is a protocol check, not a verified
llama.cpp launch recipe:

```bash
python -m llm_runtime.http_benchmark --base-url http://127.0.0.1:8080 \
  --profile portable --model YOUR_SERVED_MODEL_ID \
  --readiness-path /health --preflight-only
```

Omit `--readiness-path` if there is no suitable endpoint. Readiness here means an
HTTP success from the chosen endpoint; the client does not interpret arbitrary
body fields or measure liveness. Successful generation preflight is separate
evidence that this one request worked. There are no readiness polling retries.

Remove `--preflight-only` to run excluded warmups followed by the closed-loop
workload, for example `--requests 8 --concurrency 2 --warmup 2`. Model loading,
readiness, discovery, preflight, and warmups are outside the measured interval.
The extra preflight request **may warm a prefix cache**. A cold-cache comparison
needs explicit cache control; this client alone does not establish coldness.

## Token usage policy

| `--usage-mode` | Request behavior | Missing final usage |
| --- | --- | --- |
| `required` (default) | Send `stream_options.include_usage=true` | Fail the stream/preflight |
| `optional` | Send the same usage request | Accept latency results; mark usage `not_reported` |
| `disabled` | Omit `stream_options` for servers rejecting that field | Accept latency results; mark usage `not_reported` |

The client never silently falls back to another policy. A rejected request stays
rejected; choose a different mode explicitly after inspecting its failure.
Any supplied usage must contain nonnegative integer prompt/completion/total counts,
consistent totals, and completion counts within the output limit. Usage is
accepted on the finish chunk or in a separate final empty-choices chunk. Duplicate
usage and nonterminal non-null usage are rejected in every mode.

For a server without usage support:

```bash
python -m llm_runtime.http_benchmark --base-url http://127.0.0.1:8080 \
  --profile portable --model YOUR_SERVED_MODEL_ID --usage-mode disabled \
  --requests 8 --concurrency 2 --warmup 2
```

Reports have `schema_version: 2`. Each successful sample has `usage_status` set to
`verified` or `not_reported`. `usage_verified_requests` and `usage_missing_requests`
count only measured completed streams. If **any** completed stream lacks usage,
`completed_tokens_per_second` is `null`; it is also null when nothing completed.
The client does not estimate tokens from content chunks or publish throughput for
only the subset that happened to report usage. Latency distributions and request
throughput remain available. Server-reported counts are structurally validated,
not independently retokenized; model/tokenizer equivalence is a separate preflight
in the comparison experiment.

## Failures and limits

The preflight JSON records status, failing stage (`readiness`, `model_selection`,
or `stream`), message, and the stream sample when available. Failure returns exit
code 1 with no measured samples. Successful `--preflight-only` returns 0 with an
empty measured workload. Warmup failures still abort with a stderr explanation;
measured failures remain in raw samples and return nonzero. Invalid local options
are rejected before networking.

All profiles require loopback URLs, disable redirect following, use one shared
HTTP connection pool, and have no application retries. Readiness/discovery GETs
and each complete streaming request have bounded deadlines, including HTTP body
draining after `[DONE]`. Completion is timed at `[DONE]`, while draining permits
connection reuse. Failed/truncated streams never contribute successful latency
or token counts. Identity, choice index, assistant role, text content, supported
finish reason (`stop`/`length`), terminal marker, malformed usage, and extra SSE
data after `[DONE]` remain validated even when usage is optional or disabled.

This is a text-only chat streaming subset. Tool calls, multiple choices, arbitrary
API prefixes, server authentication, model alias rewriting, and engines omitting
the required chunk metadata or `[DONE]` are not supported. One preflight does not
verify load capacity, cancellation, memory limits, or answer quality. Connection
draining is tested; actual reuse by another engine remains to be checked.

The later [repeated CPU comparison](cpu-comparison.md) found that the pinned
llama.cpp build recomputes `created` on each chunk. For that explicitly documented
case, `--created-policy per_chunk` permits changing integer timestamps while
requiring stable ID/model. The default remains `stable`. Reports record the chosen
policy and per-sample timestamp changes. This is independent of usage policy and
does not relax truncation, terminal, usage, or response identity validation.

## Verification recorded in this milestone

- `python -m pytest -q`: **354 passed**, including **127 HTTP benchmark tests**.
  Coverage includes alternate metadata, ambiguous/invalid model discovery,
  readiness and discovery deadlines, redirect refusal, excluded preflight timing,
  missing/malformed/duplicate usage, finish-chunk usage, and stream truncation.
- Real HTTP against the existing pinned SmolLM2 CPU backend: runtime preflight;
  portable mode with required usage and two overlapping clients; portable mode
  without requested usage; and an unknown model rejected before measurement.
  Four measured smoke requests completed, and the rejected preflight returned
  HTTP 404. These tiny samples are compatibility checks, not capacity evidence.
- Final health showed no queued/active requests and zero logical KV reservations
  or usage. The temporary server was stopped after verification. Raw evidence,
  module hash, launch command, and options are in
  [http-preflight-smoke.json](http-preflight-smoke.json). Its project base commit
  identifies the pre-change checkout; the module hash identifies tested code.

No external engine was installed or run. Next: check the pinned llama.cpp build
and checkpoint conversion, align tokenization/template and CPU/weight/KV settings,
then exercise this client against it before publishing comparative measurements.
