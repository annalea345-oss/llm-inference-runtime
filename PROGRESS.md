# Progress

## 2026-10-15 — Milestone 1: FCFS baseline

Started from an empty repository and created a runnable Python package.

### Increment 1: Architecture and package setup

- Selected Python 3.10+, FastAPI, asyncio, and Uvicorn.
- Added `pyproject.toml`, development dependencies, `src` layout, and ignore rules.
- Defined module boundaries and an incremental roadmap in README.md.
- Kept execution independent of HTTP through a small executor protocol.

### Increment 2: Runtime and HTTP path

- Added a bounded FIFO waiting queue and one run-to-completion FCFS worker.
- Added deterministic prompt-word cycling with simulated prefill and decode delays.
- Added POST `/generate` with SSE token and terminal events, and GET `/health`.
- Added request validation, overload rejection, request IDs, executor error
  isolation, lifespan startup/shutdown, and cooperative disconnected-stream cleanup.
- Used per-request output queues so slow readers do not block the scheduler.
  Their per-request output is limited by `max_tokens`; total retained stream
  memory is not globally bounded.

### Increment 3: Validation and documentation

- Added tests for generation, ordering, overload, shutdown, abandonment, executor
  failure recovery, HTTP validation, health, SSE format, and actual incremental
  ASGI body delivery before generation completes.
- Documented setup, API behavior, concurrency, head-of-line blocking, architectural
  decisions, limitations, and future milestones.
- Validation: `.venv/bin/python -m pytest -q` — **15 passed in 0.33s**
  on Python 3.12. Dependency installation completed in the local `.venv`.
  The direct ASGI streaming test verifies incremental delivery; no live network
  performance benchmark was run.

### Decisions and limits

The scheduling unit is an entire request. Async HTTP handling keeps the server
responsive, but inference is serialized. There is no continuous batching,
independent prefill/decode scheduling, KV cache, explicit cancellation API,
benchmark harness, or real model in this milestone. The current disconnect flag
is minimal lifecycle cleanup, not the planned cancellation subsystem.

Next: add measurements and a repeatable benchmark before changing scheduling, so
later work can demonstrate its effect against a recorded FCFS baseline.

## 2026-10-15 — Continuous batching milestone

Implemented continuous batching at user request, ahead of the measurement
milestone. This includes the necessary step-based execution and an initial
shared token-budget policy; the benchmark milestone remains outstanding.

### Architecture and behavior

- Added `ContinuousBatchScheduler` in `batching.py` and made it the API default.
  Retained the original FCFS scheduler and its tests as a baseline.
- Added explicit waiting, prefill, decode, finished, cancelled, and failed states,
  with scheduler-owned prompt progress and output indices.
- Defined `ExecutionStep` and an asynchronous step executor interface. The
  simulated executor performs chunked prefill separately from single-token
  decode, without HTTP dependencies or retained model state.
- Bounded active requests separately from the waiting queue. Defaults are eight
  active requests, 64 waiting requests, and 32 token units per iteration.
- Shared the budget between prompt word pieces and generated pieces. Chunking
  allows prompts larger than the budget to progress. Round-robin selection
  persists across iterations; each resident request gets at most one step per
  iteration, so a small budget does not starve later active requests.
- Execute selected simulated steps concurrently with an iteration barrier.
  Admission occurs at boundaries, allowing short requests to finish while long
  requests continue. No GPU batching or throughput improvement is claimed.
- Preserved SSE framing, per-request indices, overload responses, isolated
  executor errors, and cooperative disconnect cancellation. Disconnected
  requests discard in-flight results and release capacity at the boundary.
- Shutdown cancels in-flight steps and sends one terminal error to remaining
  connected requests, including requests queued before the worker's first turn.
  Queue accounting remains balanced and scheduler restart is supported.
- Expanded health output with active count and all active request IDs; retained
  the legacy single-ID field as a representative ID.

### Validation and review

- Full suite: `.venv/bin/python -m pytest -q` — **39 passed in 0.51s**.
- Tests cover budgets of 1/2/5/32, long-prompt chunking, round-robin selection,
  concurrent in-flight steps, late admission, short-before-long completion,
  bounded admission, prefill/decode failures and cancellation, queue accounting,
  shutdown in both phases, immediate shutdown, restart, and invalid settings.
- Retained HTTP validation, overload, SSE payload, direct ASGI incremental
  delivery, and FCFS baseline tests. Added an actual ASGI disconnect test that
  checks no additional token is emitted and the active slot is released.
- Live single-worker Uvicorn checks confirmed a two-token request completed
  while a 100-token request remained active, correct SSE/health responses, and
  active-slot cleanup after closing the long stream.
- `git diff --check` passed. README documents configuration, executor contracts,
  fair scheduling, cancellation behavior, and the iteration barrier.
- Changes remain uncommitted for review; no merge or push was performed.

### Remaining limitations

Cancellation is cooperative stream-disconnect cleanup, without a cancellation
endpoint. Slow steps hold up the current iteration. The shared fair policy has
no decode priority or latency guarantee. The executor remains a deterministic
word-piece simulation; KV cache, real model integration, measurements, and
production backpressure remain future work.

## 2026-10-19 — Simulated KV-cache milestone

### Architecture and behavior

- Added scheduler-owned `SimulatedKVCache` accounting with immutable per-request
  allocation records and detached snapshots. It models token slots, not tensors
  or physical memory; the default capacity is 8192 slots.
- Reserve each admitted request's prompt count plus `max_tokens`. Track actual
  usage after successful prefill chunks and decode outputs, including the final
  output before release. Whitespace-only prompts consume one prompt slot.
- Enforce `used <= reserved <= capacity`. Reservations guarantee space for future
  growth, avoiding memory deadlock among active requests without preemption.
- Preserve FIFO admission under memory pressure. A pending queue head retains its
  waiting slot and unfinished-task ownership, remains visible in health, and
  prevents smaller requests from bypassing it. Active round-robin planning and
  token budgets are unchanged.
- Reject impossible footprints with HTTP 422 before queue insertion. Temporary
  pressure delays admission; waiting overload still returns HTTP 503.
- Release usage and reservations on completion, cooperative disconnect, execution
  failure, invalid decode output, and shutdown. In-flight reservations remain
  owned until steps settle. Finalization is idempotent, and restart begins empty.
- Expose capacity, reserved, used, and available-to-reserve token counters through
  `/health`. Keep the executor interface and retained FCFS baseline unchanged.

### Validation

- `.venv/bin/python -m pytest -q` — **64 passed in 0.24s**.
- Added atomic accounting and invalid-capacity tests; exact-fit whitespace and
  oversized HTTP requests; chunked usage and slow-reader release; FIFO pressure,
  pending-head queue limits and health; abandonment of a blocked head; failure,
  cancellation, and shutdown at partial prefill and decode; restart and invalid
  decode cleanup. Existing multi-request budget/fairness tests assert per-request
  and global memory invariants at every planning boundary.
- Extended actual ASGI disconnect and overlapping lifecycle tests to check cache
  release and ownership during delayed executor cancellation.
- `git diff --check` passed. No live network performance benchmark was run.

### Limits and next work

Reservations are conservative and strict FIFO admission can block smaller work.
The bound excludes Python objects, prompt strings, and buffered output events.
There is no paging, eviction, prefix sharing, real cache allocation, or explicit
cancellation endpoint. The recommended next milestone is request timing and a
repeatable benchmark before further scheduling changes.

## 2026-10-20 — Measurement and benchmark milestone

### Architecture and behavior

- Added request-owned monotonic submission, admission, token-emission, and terminal
  timestamps to both continuous batching and the FCFS baseline. No global request
  history or extra runtime dependency is introduced.
- Derive queue wait, time to first token, inter-token gaps, and end-to-end duration
  in milliseconds. Missing lifecycle boundaries remain null, and terminal timing
  is idempotent. Successful SSE done payloads include additive timing data.
- Added `python -m llm_runtime.benchmark`: fixed-concurrency closed-loop clients,
  deterministic prompt/output lengths, warmup exclusion, per-request deadlines,
  raw samples, nearest-rank percentiles, and completed request/token throughput.
- Reports include runtime configuration and Python/platform metadata; failures are
  counted separately and excluded from successful latency distributions. CLI
  failures return nonzero. Shutdown cleans up timed-out work cooperatively.
- Kept the scope explicit: benchmark clients submit directly to schedulers and
  consume internal events; HTTP delivery and real model performance are excluded.

### Validation and recorded experiment

- `.venv/bin/python -m pytest -q` — **79 passed in 0.37s**.
- Clock-controlled tests verify exact durations in both schedulers; failure,
  waiting cancellation, immediate/active shutdown, SSE serialization, percentile
  calculations, warmup exclusion, rejection accounting, timeout cleanup, and CLI
  failure exit codes.
- Recorded identical FCFS/continuous workloads at client concurrency 1 and 4 in
  `benchmarks/simulated-c1.json` and `benchmarks/simulated-c4.json`; reproduction
  commands and interpretation are in `benchmarks/README.md`.
- Twelve measured requests per run, eight prompt pieces, alternating 2/16 output
  pieces, 10ms prefill and 5ms decode delays: all runs completed successfully.
  Concurrency 1: 144.15 FCFS vs 140.24 continuous simulated tokens/s. Concurrency 4:
  144.39 vs 427.77. These are single-run simulator observations, not GPU claims.

### Limits and next work

Scheduler timestamps measure event emission after barriers, not HTTP client
receipt. The workload is closed loop rather than an arrival-rate load test; long
outputs contribute more samples to pooled inter-token statistics. No confidence
intervals or production latency claims are made. Next: use the recorded benchmark
when evaluating prefill/decode policy changes, or implement explicit cancellation
with request lookup and KV cleanup. External HTTP benchmarking remains future work.

## 2026-10-22 — Prefill/decode scheduling milestone

### Architecture and behavior

- Retained `round_robin` as the default; added optional `decode_first` scheduling
  with `prefill_token_budget=8`. The cap is validated as a positive integer and
  affects only the new policy; values above the shared budget are clamped.
- Decode-first schedules decodes before prefills, protects one prefill token in
  mixed batches with budget >= 2, and alternates phases when the budget is one.
  Selected requests rotate within their phase, preserving resident progress under
  newly arriving work. Each request gets at most one step per iteration.
- Cap total prefill work even in prefill-only batches, intentionally allowing
  unused shared budget. Keep iteration barriers, FIFO memory admission, execution
  interfaces, output content, and KV ownership unchanged.
- Extended the benchmark with repeating prompt-length patterns and a `policies`
  mode that compares both continuous-batching policies. JSON schema 2 includes
  per-run configurations and prompt counts for every request sample.

### Validation and experiments

- `.venv/bin/python -m pytest -q` — **113 passed in 0.55s**.
- Tests verify priorities, total/prefill budgets, default preservation, phase
  fairness under arrivals, one-token alternation across phase disappearance,
  exact output and memory invariants, cancellation/failure/shutdown/restart, mixed
  benchmark workloads, invalid configuration, and comparison CLI metadata.
- Independent read-only review found no blocking issues; an additional ad hoc
  check covered 4,410 planner phase/budget/cap combinations.
- Three repeated mixed-workload comparisons are recorded in
  `benchmarks/policies-mixed-{1,2,3}.json`; reproduction and interpretation are in
  `benchmarks/scheduling-policy.md`. All 16 requests completed for both policies
  in every comparison, without rejection, timeout, or failure.
- On this workload, output-gap p95 changed from 8.48–8.86ms round-robin to
  2.51–2.74ms decode-first; throughput fell from 187.06–189.19 to 166.82–171.82
  simulated tokens/s, and first-token p95 rose from 288.17–290.60ms to
  388.76–393.29ms. This tradeoff supports keeping the new policy optional.
- `git diff --check` passed.

### Limits and next work

Progress is guaranteed in scheduling turns for resident work, assuming executor
steps return; there is no wall-clock latency guarantee. Capped prefill can leave
budget unused. The simulator has fixed total prefill sleep per prompt and an
iteration barrier; these observations do not predict GPU behavior. The closed-loop
benchmark preserves workload content/order, not identical arrival times.

Next: explicit request cancellation with request lookup, API lifecycle handling,
and existing KV cleanup, followed by real-model integration when the simulated
runtime behavior is stable. External HTTP benchmarking remains future work.

## 2026-10-22 — Explicit request cancellation milestone

### Architecture and behavior

- Added a scheduler-owned live request registry populated only after successful
  queue insertion and evicted at finalization. Its size is bounded by waiting
  plus active capacity; completed streams do not retain registry entries.
- Added synchronous `cancel(request_id)` on the owning event loop and
  `POST /requests/{request_id}/cancel`. Accepted live requests return HTTP 202;
  unknown or finalized IDs return 404; stopped/stopping runtimes return 503.
- Separate explicit `cancel_requested` intent from disconnected `abandoned`
  streams. Connected cancelled requests receive exactly one done event with
  `finish_reason="cancelled"`, generated count, and timing data. Disconnected
  streams remain silent.
- Remove queued requests and pending FIFO heads immediately, restoring waiting
  capacity and preserving survivor order. Use only public asyncio queue operations;
  reinsert retained work before balancing task bookkeeping so join never observes
  a false zero. Waiting cancellation costs O(waiting capacity).
- Active cancellation remains cooperative at iteration boundaries: preserve KV
  reservations until in-flight work settles, then discard its result and finalize.
  Requests not selected in the current iteration also clean up at the boundary.
- Accepted cancellation wins over final decode, execution failure, or subsequent
  shutdown; an already finalized request is absent and returns 404. Repeated
  active cancellation returns 202 until finalization. Waiting cancellation
  finalizes immediately, so repeating it returns 404.
- Keep both scheduling policies, executor interfaces, and the FCFS baseline
  unchanged. Earlier buffered token events can still be delivered after the
  cancellation response; execution is not interrupted per request.

### Validation

- `.venv/bin/python -m pytest -q` — **133 passed in 0.51s**.
- Added 20 deterministic gated tests for both policies and prefill/decode phases,
  final-decode and error races, unselected active cancellation, retained KV until
  the barrier, immediate waiting/pending slot recovery, survivor FIFO order,
  balanced queue accounting, live lookup cleanup, repeated cancellation,
  shutdown/disconnect precedence, restart, and actual ASGI streaming cancellation
  using X-Request-ID. HTTP tests cover 202, 404, and 503 behavior.
- Independent read-only review found no blocking issues in queue rotation,
  lifecycle precedence, registry cleanup, or memory ownership.
- `git diff --check` passed. README includes endpoint semantics and a manual curl
  example; no new live network performance experiment was needed.

### Limits and next work

Active cleanup still waits for the iteration barrier; a stuck executor delays
cancellation. Terminal IDs and durable cancellation history are not retained.
Cancellation is event-loop-local and the API still requires one runtime process.

The next milestone is optional real-model execution: first select a small model
and supported hardware/backend, then evolve the executor boundary and verify
blocking-work isolation, output correctness, and actual memory behavior. Real KV
tensors and tokenizer integration will need explicit design; current memory and
performance measurements remain simulations. External HTTP benchmarking is also
still available as a separate follow-up.

## 2026-10-22 — Optional real-model CPU milestone

### Architecture and behavior

- Added optional PyTorch/Transformers dependencies and a pinned download helper
  for HuggingFaceTB/SmolLM2-135M-Instruct. Download only tokenizer/configuration and
  safetensors weights into ignored `.models/`. The default server remains the
  lightweight simulator; real startup uses a separate explicit factory and local
  files only, without remote model code or automatic downloads.
- Added a dedicated serialized worker thread for loading and model forwards.
  Repeated asyncio cancellation waits for queued/running work to settle before
  returning, preserving cache ownership. Worker shutdown rejects new submissions,
  joins off-loop, and is idempotent.
- Tokenize single-turn instructions with the model chat template; scheduler
  prefill budgets, context validation, and logical cache reservations now use
  model token IDs when supplied. Context/capacity rejection remains HTTP 422.
- Run actual chunked prefill and greedy decode with independent cached KV tensors
  per request. Added DecodeResult token IDs and EOS handling without changing
  existing string-returning simulator executors. Empty text pieces support EOS
  and buffered incomplete UTF-8 suffixes.
- Add optional request cleanup/lifespan hooks. Completion, EOS, cancellation,
  executor errors, and shutdown drop model state after work settles. Lifespan
  cleanup also covers scheduler startup failures. Health separates backend,
  context, parameter bytes, and KV tensor snapshots from logical token accounting.
- Keep scheduling continuous but model calls serialized, not fused tensor batches.
  CPU float32 and two PyTorch intra-op threads are the initial supported setup.

### Validation and recorded real execution

- Full suite at this increment: `.venv/bin/python -m pytest -q` — **160 passed**.
- Optional offline model tests use a tiny randomly initialized Llama and compare
  chunk sizes 1/2/32 against reference greedy generation; verify EOS, Unicode
  buffering, independent caches, real tensor bytes, and release. These tests skip
  if model dependencies are absent; normal simulator/protocol tests stay offline.
- Worker/protocol tests verify optional dependency isolation, model token counts,
  EOS streaming, capacity/context limits, release after terminal paths, cleanup
  failure isolation, asynchronous loader lifecycle, and startup failure cleanup.
  An integrated blocked-thread shutdown test confirms reservations remain owned
  until work settles, even after repeated cancellation.
- Downloaded the pinned real checkpoint and ran the reproducible
  `scripts/verify_local_model.py`: two concurrent ASGI requests matched reference
  token IDs and decoded text, with EOS/length handling, cancellation, and zero
  remaining per-request KV state. The France prompt generated
  "The capital of France is Paris.".
- Recorded `benchmarks/local-model-smoke.json`; environment and interpretation in
  `benchmarks/local-model.md`. On this Apple M3/16 GB machine the local model loaded
  in 2.20s; parameter tensors used ~513 MiB, observed KV snapshots peaked at
  ~4.18 MiB, and whole-check process RSS peaked at ~1.04 GiB. These are single-run
  observations including reference generation, not production benchmarks.
- Live single-worker Uvicorn/curl verification confirmed local factory startup,
  health metadata, real SSE token/text output, EOS terminal event, and clean
  shutdown. Test server stopped afterward. `git diff --check` passed.

### Limits and next work

Logical token capacity is not a physical RAM bound. The native allocator can keep
freed memory, and a stuck forward can delay shutdown. The local backend is tested
with SmolLM2 only; it does not implement CUDA/MPS, quantization, tensor batching,
paged KV tensors, sampling, multi-turn chat history, or a chat-compatible API.
Model quality is not established by the smoke check.

Next: add a minimal chat-compatible adapter and external HTTP streaming benchmark
before choosing a chat interface, or profile/extend the model-step interface for
true batching. Keep physical memory and real performance measurements separate
from simulated scheduling results.

## 2026-10-23 — Text-only chat-compatible API milestone

### Architecture and behavior

- Added `/v1/models`, model lookup, and `/v1/chat/completions` with a deliberately
  defined text-only Chat Completions subset. Model aliases identify the selected
  backend: simulated, local-model, or runtime/custom alias. No hosted API call,
  new model download, or OpenAI account is required.
- Accept bounded system/user/assistant histories, with an optional initial system
  message and alternating turns ending in user. Validate before admission: at
  most 64 messages and 8192 combined content characters, one output choice, and
  either max_tokens or max_completion_tokens in 1–512 (default 16).
- Preserve greedy semantics: temperature omitted/null/zero only. Reject unsupported
  fields, tools, multimodal content, sampling, and unsupported roles explicitly.
  Use HTTP 400 error envelopes for chat validation while retaining /generate's
  existing HTTP 422 format. Unsupported aliases return 404.
- Pass the full history once to the real tokenizer's chat template through
  submit_chat/tokenize_messages, avoiding single-user double wrapping. Simulator
  fallback uses readable role-labelled text. Context and logical memory limits
  still apply to the resulting tokens before queue insertion.
- Return complete assistant messages with prompt/completion/total token usage, or
  data-only SSE deltas with stable completion IDs, model, timestamp, final stop or
  length reason, optional final usage chunk, and [DONE]. EOS maps to stop.
- Share existing scheduler ownership, capacities, lookup, and cancellation. Keep
  raw runtime IDs in X-Request-ID; chatcmpl-prefixed completion IDs are transport
  identifiers. Cancellation yields a compatible error rather than a fabricated
  success finish reason: JSON409 or streamed error+[DONE] after HTTP200.
- Watch nonstream disconnects while collecting output; reuse streaming cleanup for
  disconnects and header-send failure. No conversation history database is added.

### Validation and real-model evidence

- Tests at completion: `.venv/bin/python -m pytest -q` — **221 passed**.
- Added chat validation/discovery/format/usage tests, error and capacity mappings,
  shared cancellation and shutdown, full-history fake tokenization, direct-ASGI
  incremental content/disconnect/header failure, and nonstream task cleanup.
  Added an offline local tokenizer test proving full-history single templating.
- Independent read-only review found no blocking issues; its transport and local
  tokenizer coverage suggestions were added.
- Ran `scripts/verify_chat_api.py` with the pinned real checkpoint. Nonstream and
  streamed multi-turn results matched independent reference greedy generation,
  with 48 prompt tokens, six completion tokens (including EOS), and zero remaining
  per-request caches. Recorded benchmarks/chat-api-smoke.json and interpretation
  in benchmarks/chat-api.md.
- Live Uvicorn/curl check verified discovery plus real chat deltas, stop finish,
  final usage, and [DONE]. Test server stopped after verification.
- `git diff --check` passed. CHAT_API.md describes exact options, error semantics,
  client setup, ownership, and compatibility limits. README and LOCAL_MODEL updated.

### Limits and next work

This is a text-only compatibility subset, not a complete OpenAI API. Sampling,
tools, images/audio, persistent conversations, arbitrary message ordering, CORS
for arbitrary browser origins, and a packaged chat interface remain outside this
milestone. Some clients need configuration changes to avoid unsupported fields.

Next: external HTTP streaming measurement and a concrete chat-client integration
configured for this supported subset, using local-model rather than hosted model
names. Continue measuring real-model/network performance separately from the
in-process simulator benchmarks.

## 2026-11-08 — Browser client and external HTTP measurement milestone

### Architecture and behavior

- Added a built-in same-origin browser client at `/`, packaged with the Python
  wheel. Plain HTML/CSS/ES modules use the existing text-only chat API; no frontend
  build, npm dependencies, separate host, public deployment, or model download.
- Added streamed conversations, optional instructions, output limit, Stop, New
  chat, model discovery, backend identification, and an explicit second-client
  link. Each tab keeps independent in-memory history. Failed/cancelled/empty
  replies remain visible but do not enter future conversation context.
- Handle partial UTF-8 and SSE boundaries, stable response identity, final usage,
  errors, and missing completion markers. Render all user/model text literally.
  Stop calls request cancellation when the raw ID is available and falls back
  to closing the stream; model cleanup remains cooperative at execution boundaries.
- Added detached request progress snapshots and runtime configuration to health.
  Show all clients' waiting/active states, prompt/output progress, logical slots,
  actual KV tensor snapshots, and this tab's observed response times. Telemetry
  excludes prompt text and token IDs; offline snapshots are marked stale.
- Added a separate loopback HTTP benchmark with bounded closed-loop clients,
  shared connection pool, warmup exclusion, request deadlines, validated SSE
  identity/finish/usage/[DONE], raw samples, error counts, and latency distributions.
  Measure headers, first nonempty content, completion, and received content gaps;
  do not label network deltas as model tokens. Throughput uses successful usage.

### Validation and recorded evidence

- `.venv/bin/python -m pytest -q` — **301 passed** at completion. This includes
  server asset/telemetry tests, 74 HTTP benchmark cases, and a wrapper running the
  optional Node parser suite when Node 20+ is available.
- `node --test tests/browser_stream.test.mjs` — **18 passed**, covering fragmented
  UTF-8/CRLF/multiline SSE, identity/usage errors, truncation, abort, and literal text.
- Built a wheel from an isolated temporary source copy and confirmed all four
  browser assets are included; no build output was left in the repository.
- Real Chrome/Playwright check against the local CPU model verified actual
  answers, conversation replay, two overlapping clients, Stop and cancelled-context
  exclusion, New chat, and zero remaining reservations/KV. Browser fixtures
  separately verified literal HTML rendering, truncated-response errors, failed
  history exclusion, offline health/recovery, and 390/320px layout. No page errors.
- Recorded benchmarks/browser-client-smoke.json and a representative desktop
  screenshot. Optional browser QA uses the already installed Chrome; Playwright
  is a separate optional extra. No browser binary was downloaded.
- Recorded external HTTP runs at client concurrency 1 and 2, eight measured
  requests plus two excluded warmups each. All completed successfully. Mean first
  content was 68.31ms and 133.73ms; throughput was 53.94 and 54.55 output tokens/s.
  These results are consistent with serialized CPU forwards, not a batching speedup.
  Full samples/methodology are in benchmarks/http-local-c1.json, http-local-c2.json,
  and http-local.md. Browser clients were closed during measurement.
- Independent review found a benchmark identity-validation gap; corrected it and
  added malformed/mixed identity cases before recording measurements. Stale health
  labeling was also improved. Final transport review found that stopping at [DONE]
  prevented HTTP connection reuse; fixed by recording that timestamp then draining
  to HTTP EOF. Added drain/deadline/trailing-data tests, verified one/two reused
  connections in server logs, and rerecorded results without concurrent QA work.
  `git diff --check` passed.
- Browser and HTTP test server stopped after verification. BROWSER_CLIENT.md,
  README, CHAT_API, and LOCAL_MODEL document setup, behavior, and limits.

### Limits and next work

This remains a local educational client: no authentication, durable chat storage,
Markdown rendering, tools, file uploads, GPU backend, or fused model batching.
Health polling can miss brief phases/peaks, logical slots are not a RAM bound,
and the tiny single-workload HTTP samples are not production capacity evidence.
Browser QA covers installed Chrome, not every browser or an accessibility audit.

Next: choose a measured backend improvement such as real tensor batching or
hardware acceleration, or strengthen deployment/observability if sharing the
service becomes the goal. Compare equivalent real workloads and preserve the
simulator as a deterministic regression baseline.

## 2026-11-09 — Engine comparison research milestone

- Reviewed primary sources for vLLM, SGLang, llama.cpp, and MLX LM. Recorded
  four exact upstream commits and SHA-256 hashes of five reviewed source files in
  research/engine-comparison/sources.json. Moving documentation pages are labeled
  separately rather than attributed to those commits.
- Documented the current serial CPU executor, per-request KV ownership, logical
  reservation limits, and existing small HTTP baseline alongside the comparison.
  Established engines already offer batching, cache reuse, and observability;
  no novel feature gap, upstream bug, or performance winner is claimed.
- Identified benchmark portability work: explicit model selection, configurable
  readiness/metadata, and preflight for streamed usage/terminal semantics. The
  current runtime-specific benchmark is not assumed to work unchanged elsewhere.
- Proposed llama.cpp as a conditional first CPU control, verifying checkpoint
  conversion, tokenizer/template, float32 weights/KV, and thread/backend settings.
  MLX acceleration would be a separately labeled deployment profile. Neither
  engine has been installed, converted, launched, or measured in this milestone.
- Added an experiment plan covering mixed workloads, closed-loop versus fixed
  arrivals, cold/warm prefixes, cancellation, bounded memory pressure, repetitions,
  raw run manifests, and evidence gates before selecting the runtime improvement.
- Validation: parsed the source manifest, checked commit/hash formats and pinned
  URLs, verified all five hashes against the pinned upstream files, resolved
  18 local Markdown links across the README and research documents,
  and ran `git diff --check`. Runtime code/dependencies were unchanged; the
  previously recorded 301 Python / 18 JavaScript test results were not rerun.

Next: implement the portable benchmark preflight/adapter and verify the first
external engine's compatibility before recording comparative measurements. Use
those results to choose real tensor batching, prefix reuse, or decision traces
with workload replay. Findings and detailed protocol are linked from README.

## 2026-11-09 — Portable HTTP benchmark preflight

### Implemented behavior

- Added `runtime` and `portable` measurement profiles. Runtime defaults preserve
  health metadata/discovery; explicit model selection handles multiple advertised
  models. Portable mode can use an explicit model without any discovery or health
  dependency. A configurable local readiness path accepts JSON or text bodies;
  HTTP success is recorded separately from the generation check, not as liveness.
- Added one excluded streaming preflight before warmup/measurement, plus
  `--preflight-only`. Structured failure reports identify readiness, model
  selection, or stream failure and prevent measured work. No automatic fallback
  or application retry. Metadata calls and streams have bounded deadlines;
  redirects are disabled even for an injected client configured to follow them.
- Added explicit required/optional/disabled usage policies. Optional mode requests
  usage but permits its absence; disabled mode omits the stream option. Any
  supplied usage is still validated. Accept usage on a finish chunk or in its
  own final chunk; reject duplicates and nonterminal usage.
- Report schema version 2 includes preflight details, usage availability counts,
  and closed-loop arrival labeling. Token throughput is null if any completed
  measured stream lacks verified usage or nothing completes; latency remains
  available. No token estimates from network chunks or partial usage subsets.
- Preserved strict identity/terminal/error/deadline validation and response
  draining for connection reuse. Also reject incomplete trailing SSE data after
  the terminal marker. The preflight uses the workload prompt and may warm a
  server prefix cache; it is not evidence of a cold-cache workload.

### Verification and limits

- `.venv/bin/python -m pytest -q` — **354 passed**, including **127 HTTP benchmark
  cases** (53 added). No scheduler, model executor, or browser behavior changed.
- Real loopback HTTP with the existing pinned SmolLM2 CPU checkpoint verified
  runtime preflight, portable usage-required concurrency 2, portable usage-disabled
  latency measurement, and an unknown model rejected with HTTP 404 before any
  measured workload. Four measured smoke requests completed. Recorded raw samples,
  command/options, tested module hash, and final health in
  benchmarks/http-preflight-smoke.json. Server logs showed reused connections
  across discovery/preflight and across warmup/measurement.
- Final health had zero queued/active requests and zero logical reserved/used KV
  slots. The temporary local server was stopped after verification. No engine,
  dependency, or checkpoint was downloaded.
- `git diff --check` passed; smoke JSON/module hash validated and 27 local
  documentation links resolved. Added benchmarks/http-compatibility.md and updated
  the README, browser measurement guide, benchmark index, and research plan status.

This verifies the portable protocol against this runtime and deterministic alternate
transport fixtures, not an actual external engine. No cross-engine speed or feature
gap is claimed. Full comparison manifests/workloads, fixed-arrival replay, and
external cancellation/memory experiments remain pending.

Next: check/build the pinned llama.cpp revision and convert the existing checkpoint
faithfully; verify tokenizer/template and CPU/weight/KV precision, then exercise
the preflight against that server before recording comparative measurements.

## 2026-11-09 — Pinned llama.cpp CPU compatibility verification

- Downloaded the previously reviewed llama.cpp commit
  `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5` as an archive into ignored
  `.experiments/`. Reviewed its MIT license/build/conversion requirements first.
  Installed CMake 4.1.3 and SentencePiece 0.2.1 in a separate experiment tools
  environment; project dependencies and the existing checkpoint were unchanged.
- Built the CPU server with Metal, Accelerate, BLAS, OpenMP, UI/assets, and HTTPS
  support disabled. No upstream patch was required. Set a Git discovery ceiling
  so the archive build reports unknown provenance instead of inheriting the
  parent runtime repository's commit; captured archive/executable/library hashes.
- Converted existing checkpoint weights to a roughly 538 MB float32 GGUF using
  the existing PyTorch/Transformers/NumPy plus upstream source-tree GGUF tooling.
  Did not install upstream's different pinned conversion requirements wholesale.
- Added scripts/verify_llama_cpp.py for existing artifacts and a running external
  server. Verified all **272 tensors** exactly against source values widened to
  float32, including the HF-to-GGML Q/K RoPE layout permutation; verified the
  embedded template and BOS/EOS/padding IDs.
- Four single/system/multi-turn/Unicode fixtures matched both rendered templates
  and prompt IDs between Transformers and llama.cpp. Required-usage portable
  preflight and eight measured HTTP smoke requests completed at concurrency 1/2,
  with two excluded warmups per setting. Each measured request reported 40 prompt
  tokens and 8 output tokens with EOS finish. A separate JSON completion answered
  the France question with `The capital of France is Paris.`
- Captured launch/configuration/build details in benchmarks/llama-cpp-build.json;
  raw fixtures, usage/timing samples, server properties, final slots/metrics,
  checkpoint hashes, and verifier hash in benchmarks/llama-cpp-smoke.json.
  Documented setup, downloads, commands, limits, and follow-up in
  benchmarks/llama-cpp.md and linked the updated research/roadmap.
- Both slots were idle; final metrics reported zero processing/deferred requests
  and zero cached-prefix tokens. Stopped the temporary server. Physical memory
  release, cancellation, overload recovery, long contexts, and fairness have not
  been established by this smoke check. CPU kernels/libraries and scheduling
  still differ, even with matched precision and model-thread settings.
- `.venv/bin/python -m pytest -q` — **354 passed**. The added verifier was exercised
  against actual converted tensors and a live external server. Documentation
  links, recorded hashes, JSON, and `git diff --check` were also validated.

This completes external compatibility for the recorded configuration, not a
performance comparison or a new runtime backend. Large sources/builds/weights
remain ignored locally. Next: versioned short/long/mixed workloads, repeated
alternating-engine CPU runs, bounded cancellation/recovery and memory observations,
then choose one runtime improvement from the evidence.

## 2026-11-09 — Repeated CPU comparison and lifecycle findings

### Measurement implementation

- Added versioned exact short/longer-prompt/larger-output cases and a mixed
  pattern in benchmarks/cpu-workloads-v1.json. Shared comparison client records
  request class and submission offsets, preserves strict SSE validation, and
  checks server prompt usage against verified tokenizer counts.
- Added scripts/compare_engines.py: five repetitions, concurrency 1/2, alternating
  engine/workload/concurrency order, one restarted loopback server at a time,
  excluded startup/preflight/warmup, checkpointed raw reports, and guaranteed
  child-process cleanup. Timing intervals have no concurrent health/RSS polling.
- The bounded pilot discovered changing `created` timestamps in longer llama.cpp
  streams. Pinned server-task.cpp computes time at chunk serialization. Added an
  explicit `--created-policy per_chunk`; default remains stable. Both policies
  validate integer timestamps and stable ID/model, finish reason, usage, terminal
  marker, deadlines, and HTTP drain. Raw samples record timestamp changes.
- Separate lifecycle sessions probe active/queued disconnect, follow-up recovery,
  our explicit cancel endpoint with its connection kept open, and five bounded
  mixed memory waves. Native admission probe separately tests logical reservation
  pressure. Sources/builds/model artifacts remain unchanged and no dependencies
  or weights were downloaded.

### Recorded results and interpretation

- All **80 timing runs / 640 measured requests completed**, with no measured
  rejection/error/timeout. Prompt counts were 40/562/56 and outputs 8/24/96 for
  short/longer-prompt/larger-output cases across both engines. Homogeneous warmup
  covers live concurrency (1/2); mixed warmup covers all four pattern entries.
- Our runtime had lower median per-run mean first-content latency in each tested
  profile. llama.cpp's larger-output median throughput rose from 65.88 tokens/s
  at concurrency 1 to 109.03 at 2; our serial executor rose from 63.20 to 65.13.
  These are whole-engine results for the pinned native CPU build with
  BLAS/Accelerate/Metal disabled, not a general engine ranking or proof of a
  scheduler-only cause. Full tables and per-run variation are in the report.
- Active disconnect reached observed idle in about 23 ms in one probe per engine.
  llama.cpp removed its queued task after about 926 ms while both blockers were
  still active. Our queued disconnected request remained counted during the
  two-second window, reproducing the pilot finding. Both engines reached idle
  after all streams closed and successfully served follow-ups. Explicit native
  cancellation delivered request_cancelled/[DONE] while its stream stayed open.
- The native token-pressure probe submitted two individually valid requests,
  each reserving 4,242 slots against an 8,192 cap. It observed one active/one
  waiting despite a free active slot. Explicit waiting/active cancellation
  released all tracked KV and follow-up succeeded. Reservations precede actual
  usage; this does not enforce or test host-RAM exhaustion.
- Final runtime state had zero queued/active requests, reserved/used slots, and
  tracked KV bytes. RSS rose across the five observation waves (about 857–873 MiB
  after waves for our runtime; 913–914 MiB for llama.cpp). This establishes neither
  a leak nor a stable plateau. Retained process memory needs separate profiling.
- Recorded 281 timestamp transitions in llama.cpp's measured samples. The prior
  short compatibility smoke therefore did not establish longer-stream coverage.

### Verification and next milestone

- `.venv/bin/python -m pytest -q` — **383 passed**. Tests cover mixed workers,
  workload validation, excluded phases, usage mismatches, explicit timestamp
  policy, process cleanup on error/stall, and invalid/incomplete report rejection.
  A test fixture initially patched sockets through async teardown; scoped that
  patch correctly and reran successfully. The native probe was also rerun after
  adding source-hash/command provenance.
- Added scripts/summarize_engine_comparison.py, which refuses incomplete matrices
  and summarizes distributions across per-run statistics. Saved exact raw JSON,
  manifest, lifecycle/admission records, and summary in benchmarks/cpu-comparison-v1.
  The report and artifact index explain settings, actual work, sample sizes,
  limitations, RSS interpretation, reproduction, and future work. Temporary
  servers were stopped; logs stay in ignored .experiments/.
- Validated all 95 dataset JSON records, the complete 80-run matrix, 640 measured
  completions and 170 excluded warmups, source/workload/probe hashes, exact summary
  reproduction from copied data, 56 local documentation links, and `git diff --check`.
- Recommended next: promptly reclaim disconnected waiting requests when active
  capacity is full, preserving FIFO/task accounting and step-boundary release.
  Add deterministic saturation/disconnect tests and repeat the real queued probe.
  Then test real tensor batching against the larger-output profile, preserving
  mixed-request latency checks, and investigate retained RSS. Fixed-arrival
  fairness replay, cache reuse, acceleration, and production-capacity stress are
  not established by this experiment. No scheduler/executor code changed here.

## 2026-11-10 — Immediate queued-disconnect cleanup

- Reproduced the comparison finding deterministically: with two model steps
  blocked and a full waiting queue, disconnecting the middle waiter left its slot
  occupied. The test failed before implementation without releasing either step.
- Added scheduler `abandon(id)`, which marks transport abandonment and reuses
  explicit cancellation. Waiting requests, including a memory-blocked pending
  head, are removed immediately. Active resources remain protected until their
  selected execution barrier settles. Late terminal/shutdown notifications are
  safe and no event is delivered to an abandoned transport.
- Wired native/chat streaming cleanup and non-streaming chat collection to their
  owning scheduler. Response callbacks bind that owner at construction rather
  than looking up a potentially restarted app state on closure.
- Found a further ASGI 2.4 case: installed Starlette relies on send errors, so a
  waiting stream with no new output did not observe the disconnect. A new test
  timed out before the listener change. Added a concurrent disconnect listener
  for modern transports; keep the existing listener for older transports and
  reap stream/listener tasks on completion, error, or external cancellation.
- `.venv/bin/python -m pytest -q` — **411 passed**, including **28 new cases**.
  New tests cover both policies, full-active/full-waiting saturation, FIFO,
  immediate replacement admission, no abandoned execution, single release,
  balanced join, native/chat/nonstream closures, handler cancellation, header
  failures, modern ASGI paths, pending memory admission, shutdown settlement,
  restart/late notifications, and successful response listener cleanup.
- Added a targeted reusable real-model verifier. Three queued-disconnect probes
  reclaimed the waiter with both blockers still active, observed at 1.19/1.28/
  1.24 ms including client/polling overhead. Three active-disconnect probes reached
  observed idle in about 23 ms. Explicit cancellation still emitted its terminal
  error/[DONE], all seven follow-ups completed, and final request/KV counters were
  zero. The temporary model server was stopped; no dependencies/models downloaded.
- Recorded commands, source/checkpoint hashes, versions, trial states, and final
  cleanup in benchmarks/disconnect-cleanup.json. Added a report and updated
  README, chat API, browser, model setup, benchmark index, and comparison follow-up.
  Previous comparison data is unchanged; no new throughput/RSS claim is made.
- Verified the live record's source hashes, all seven successful follow-ups,
  final zero counters/stopped server, 47 local documentation links, and
  `git diff --check` before committing.

Next: investigate retained RSS separately, then test real tensor batching against
the larger-output workload while preserving mixed-client responsiveness checks.

## 2026-11-10 — Retained CPU memory investigation

- Added isolated profiling tools without changing normal scheduler/executor/API
  code. Two normal real-HTTP sessions and a separate private diagnostic factory
  each ran 30 four-request mixed waves at concurrency 2. All **364 requests**
  completed, including the diagnostic post-intervention recovery wave. Sampled
  idle request/registry/reservation/KV counters were zero throughout.
- Normal sessions used no tracing, weak observers, forced GC, or allocator
  interventions. Idle RSS rose during warmup, then ended below wave 1: roughly
  849→719 MiB and 860→693 MiB. This extends the earlier five-wave observation but
  does not prove a permanent plateau or reconstruct its cancellation/polling and
  host conditions. Last-window fluctuations and exact raw observations are kept.
- Diagnostic-only instance hooks used compacted weak references. Across the
  120-request workload, all 120 released sequence objects and 7,200 released K/V
  Tensor wrappers were dead at idle. One completed request stayed alive at idle
  across every wave, consistent with last-iteration scheduler locals, and reached
  zero after shutdown. This is a bounded observed tail, not cumulative retention.
- All 272 model parameter objects stayed alive while loaded; model/parameter
  references were dead after cleanup. After recovery, 124 sequences / 7,440 K/V
  wrappers had been observed and were dead at shutdown. Wrapper lifetimes do not
  account for every native allocation or prove that all pages returned to the OS.
- Traced allocations made after model load rose about 1.05→1.56 MiB. A one-off
  idle GC collected 502 objects and reduced traced bytes about 1.53→0.97 MiB, but
  the corresponding RSS snapshot was higher. A best-effort macOS allocator relief
  probe reported zero released bytes. Neither probe establishes a useful normal
  request-time cleanup policy. No forced GC/trim was added to production.
- Checked Apple's primary malloc API documentation and installed SDK declarations;
  recorded header hash/API semantics and hardware. Native in-use counters, RSS,
  and post-load Python tracing are reported separately rather than added or
  treated as equivalent. The private controls require idle scheduler state and
  are absent from normal factories. Temporary child servers all stopped.
- Recorded completed manifests, raw normal/diagnostic waves, interventions,
  shutdown lifetimes, native API provenance, and reproducible summary in
  benchmarks/retained-memory-v1. Added an optional standard matplotlib chart,
  visually inspected it, and documented methods, observer effects, limitations,
  decisions, and reproduction in benchmarks/retained-memory.md. Matplotlib was
  already available in the base interpreter; no dependency/model was installed.
- `.venv/bin/python -m pytest -q` — **433 passed**, including weak non-retention,
  observer compaction, busy-state intervention refusal, normal-route isolation,
  archived result reproduction, and malformed/incomplete profile rejection.
- Validated all eight result/provenance JSON files, 90 zero-ownership idle waves,
  364 completions, source/workload/render hashes, exact summary reproduction,
  49 local documentation links, and `git diff --check`. Confirmed normal runtime
  source was unchanged; all three experiment child servers stopped.

Decision: no accumulating released KV objects were observed in this bounded test,
and RSS fell after warmup in both normal runs. This is not a general
leak-free guarantee or a unique explanation for every retained byte. Keep normal
memory cleanup unchanged. Next: prototype real tensor batching and compare
larger-output throughput, mixed-request latency, cancellation, and memory against
these baselines; treat earlier release of idle-loop request locals separately.

## 2026-11-16 — Larger-model compatibility before tensor batching

- Selected Qwen/Qwen2.5-0.5B-Instruct at pinned revision
  `7ae557604adf67be50417f59c2c2f167def9a775` after reviewing its model card,
  license, configuration, generation defaults, size, and installed Transformers
  support. Downloaded approximately 1 GB into ignored .models/qwen2.5-0.5b;
  preserved the original SmolLM2 checkpoint and installed no dependency.
- Added explicit pinned model selection to the download helper, preserving the
  original default. Separate model directories and identity/resume checks prevent
  accidental replacement of another selected model/revision or unknown contents.
- Existing float32 CPU executor worked with Qwen2 architecture without a new
  execution algorithm. Parameter count is 494,032,768, about 3.7 times SmolLM2;
  float32 parameter storage is 1,976,131,072 bytes. Model context is 32,768, while
  default logical KV capacity remains 8,192. No quantization/GPU/tensor batching.
- Neutralized reference-only repetition penalties explicitly: Qwen's generation
  defaults include sampling and repetition_penalty=1.1, while our executor is
  plain greedy argmax. Direct .generate references use do_sample=False and
  repetition_penalty=1.0 to match intended runtime behavior.
- Native output token IDs matched direct Transformers for both prompt cases;
  EOS/output-cap termination and in-flight cancellation passed. Multi-turn
  JSON/streamed chat matched direct reference and remembered Ada. Reran the
  original SmolLM2 native/reference smoke successfully after reference changes.
- Eight measured real-HTTP smoke requests completed at concurrency 1/2. Verified
  Unicode transport, one-token cap, oversized-token KV rejection, queued removal
  with two active blockers, active disconnect, explicit cancel, follow-ups, and
  zero final request/KV state. Temporary HTTP server stopped.
- Qwen correctly answered the France example, but called blåbær a blue whale.
  Added the exact Unicode reference fixture: direct Transformers, JSON, and SSE
  all produced the same incorrect answer. This is a reproduced model-quality
  limitation, not a passing factual test or evidence of a serving corruption.
- Native/reference smoke peak RSS was about 2.12 GiB. Final HTTP readiness/after-
  checks RSS samples were about 0.91/2.01 GiB. These are separate scopes, not a
  physical-memory cap or arbitrary-workload capacity. Four-request per-setting
  timing samples are recorded but not treated as a speed/scaling comparison.
- `.venv/bin/python -m pytest -q` — **444 passed**, including seven model-download
  selection/default/identity/resume cases and four additional tiny Qwen2 greedy
  reference/multi-EOS cases. Real verification uses the downloaded checkpoint;
  routine optional model tests remain offline and use tiny random weights.
- Added reusable HTTP verifier and recorded native, history, Unicode reference,
  HTTP, and original-model recheck JSON under benchmarks/qwen-0.5b. Documented
  exact model/decoding settings, hashes, versions, memory observations, quality
  limits, downloads, and browser switching commands. Model weights stay local.
- Validated the pinned model/source hashes, all original SmolLM2 file hashes,
  eight live HTTP completions, direct/native/chat/Unicode reference agreement,
  zero final KV state, server shutdown, 39 local documentation links, and
  `git diff --check` before committing.

The new branch preserves the completed profiling commit, which was not yet on
origin/main when this phase began. Next: implement optional real tensor batching
and evaluate each model against its own unbatched baseline, retaining memory and
cancellation correctness. Broader factual-quality tests and long-context/larger-
model sustained-memory tests remain separate future work.

## 2026-11-24 — Native Android chat client

- Started from merged `origin/main` (`997fc5d`). Added a standalone native Kotlin
  app under `android/`; the existing Python scheduler, executor, API endpoints,
  browser client, and model weights are unchanged. Android is another client;
  the Mac still owns model execution and KV state.
- Added configurable server address, model discovery, streamed text, multi-turn
  history, output limits, New chat, Stop, offline errors, and reconnect. Network
  operations run off the main thread. A ViewModel retains the transcript and
  active operation across Activity recreation; chat is not persisted to disk.
- Stop sends the raw request ID to the existing cancellation endpoint and closes
  the stream as fallback. Failed/stopped turns remain visible but are excluded
  from later history. Stream completion requires stable identity, valid finish
  and usage events, and `[DONE]`; premature EOF cannot become a successful turn.
- Added bounded SSE/discovery/error parsing, strict UTF-8 decoding, Unicode-aware
  history limits, and no automatic generation retries. Debug alone permits HTTP
  for local development; merged release manifest has cleartext disabled. The
  only app permission is Internet; no credentials or model files are packaged.
- Used the existing Android SDK 35, Build Tools 35.0.0, JBR 21, and cached
  AGP 8.8.0/Kotlin 2.0.0 toolchain. Committed the Gradle 8.10.2 wrapper with its
  official distribution checksum. First sync may download Gradle/library
  dependencies; no new language model was downloaded. MockWebServer/OkHttp and
  the JVM JSON implementation are test dependencies, not Android app dependencies.
- **26 Android JVM tests passed**: 20 protocol/history/input cases and six real
  HTTP cases, including one-byte chunks, Unicode, truncation, server rejection,
  raw-ID cancellation, Stop before start, and Stop before response headers.
- Debug APK and unsigned release APK built successfully. Android debug lint and
  release vital lint passed with no errors; debug lint retains seven warnings
  for newer dependency versions, backup configuration guidance, and English
  strings. This is an English development client, not a published/localized app.
- The existing Python suite remains **444 passed**. No Python runtime changes
  or new model downloads were needed for Android.
- Ran an Android 15/API 35 ARM64 emulator against the existing pinned
  Qwen2.5-0.5B-Instruct CPU/float32 checkpoint. Verified discovery, streamed
  answers, follow-up agreement with a direct JSON API request using exactly the
  same history, portrait/landscape controls and retained history, two admitted
  clients, explicit cancellation, successful peer completion, zero final active/
  queued requests and zero reserved/used KV slots, New chat, offline error, and
  reconnect. Inspected portrait and landscape screenshots. Recorded fixture
  evidence in `android/verification/qwen-api35.json`; screenshots/build outputs
  remain ignored under `.experiments/` and `android/app/build/`.
- The prompt `My name is Ada. Reply with my name.` elicited `My name is Qwen.`,
  and the follow-up `What is my name?` elicited `Your name is Qwen.`. The follow-up
  matched a separate direct API request with the same history. This establishes
  client agreement; it does not establish factual correctness or independently
  compare this case with direct Transformers. Model quality remains a separate
  concern, as documented in the previous Qwen milestone.
- Added [ANDROID_CLIENT.md](ANDROID_CLIENT.md), linked from README, with Android
  Studio/build steps, emulator networking, USB reverse and trusted-Wi-Fi phone
  instructions, privacy/state behavior, tests, and limits. Added the opt-in
  `scripts/verify_android_client.py` smoke for a dedicated idle server/emulator.
- Stopped the temporary loopback Qwen server, no-save emulator, and Gradle daemon
  after verification. Preserved the existing untracked `error.log`; model weights,
  local SDK/Gradle caches, APKs, and experiment screenshots are not committed.

Physical phones, Android versions other than API 35, process-death/background
durability, release signing, HTTPS hosting/authentication, app-store distribution,
and on-device inference were not verified or implemented. The next engine
performance milestone remains optional real tensor batching, compared against
each model's unbatched baseline while preserving cancellation and memory safety.

## 2026-11-24 — Friendlier Android chat interface

- Started from merged Android-client main (`bfc1d4b`). Replaced the technical
  form with labelled message bubbles, a bottom message bar, a single Send/Stop
  control, a welcome screen, and example prompts that fill the draft without
  submitting it. Added rounded controls, consistent colors/spacing, and system
  light/dark resources. User messages and replies have separate, selectable
  views rather than being parsed out of a combined transcript.
- Moved the server address and reply-length controls into Settings. Short=64
  tokens preserves the previous default, Medium=192, and Long=512; Advanced
  retains exact 1–512 limits. Save validates inputs; Cancel discards edits.
  Connection status uses simple wording, with raw server/model/error details
  available on tap. Limit, failed, and stopped replies have plain-language notes.
- Added display-only `ChatTurn` state; accepted conversation history still uses
  the same `Conversation` logic. Rotation/theme changes retain turns, drafts,
  saved settings, and reading position. A reader already at the bottom follows
  new text; scrolling back does not force a jump on each token. Settings/detail
  dialogs are dismissed on Activity destruction. Settings and New chat remain
  disabled during generation; Stop retains existing cancellation semantics.
- No changes to Python runtime/inference, model files, generation settings,
  scheduler, KV ownership, endpoints, `RuntimeClient.kt`, `ChatProtocol.kt`, or
  dependency versions. No downloads or new permissions were added. Ignored the
  user's local Android Studio `.idea/` directory without deleting it.
- Android debug and unsigned release builds passed. All **26 Android JVM tests**
  and **444 Python tests** passed. Android debug lint has **zero errors and three
  warnings** (two newer dependency versions and backup-rule guidance); release
  vital lint passed. Navigation-bar theme flags are qualified for API 27+ while
  retaining API 26 support in the manifest; only API 35 was exercised on a device.
- Extended the opt-in emulator smoke for the new controls. The API 35/Qwen run
  checked hidden technical fields on the welcome screen, model discovery,
  streamed history agreement with a direct JSON request, noncollapsed bubble
  widths, portrait/landscape retained history, dark mode, font scale 1.3, Long
  reply selection, two simultaneous clients, Stop, zero final request/KV state,
  New chat, offline errors, and reconnect. Visually inspected light/dark,
  landscape, larger-text, and Settings screenshots. The smoke restores previous
  rotation, theme, and font-scale settings.
- Final-APK Settings checks confirmed zero is rejected, correcting it to 192
  saves and selects Medium, Cancel discards an unsaved Long choice, and Short
  can be restored. Validation errors clear while editing. The full UI smoke
  preceded that final error-feedback adjustment; targeted Settings checks and
  the Android build/test/lint checks passed afterward. Recorded both scopes.
- Recorded the UI run separately in `android/verification/polish-api35.json` and
  a welcome-screen preview in `android/verification/polish-welcome.png`. Original
  Android verification and engine benchmark records remain unchanged. Updated
  README and [ANDROID_CLIENT.md](ANDROID_CLIENT.md) for the new flow.
- Stopped the temporary Qwen server, no-save emulator, and task Gradle daemon
  after verification. Preserved the user's local IDE files, models, and
  untracked `error.log`.

This phase improves presentation and usability, not model intelligence or
inference performance. Physical phones, other Android releases, very large font
scales, localization, and persistent conversation storage remain outside the
verified scope. Optional tensor batching remains the next engine milestone.
