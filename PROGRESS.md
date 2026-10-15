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
