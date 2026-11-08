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
