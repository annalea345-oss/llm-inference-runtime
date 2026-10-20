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
