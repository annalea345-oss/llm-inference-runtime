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
