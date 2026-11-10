# Simulated scheduler baseline — 2026-10-20

These reports exercise the in-process scheduler client, without an HTTP server.
Python/platform details, full configuration, and request samples are in each JSON
file. The measured run excludes two warmup requests for each scheduler.

Reproduce from the repository root (after activating `.venv`):

```bash
python -m llm_runtime.benchmark --requests 12 --concurrency 1 --warmup 2 --prefill-delay 0.01 --token-delay 0.005 > benchmarks/simulated-c1.json
python -m llm_runtime.benchmark --requests 12 --concurrency 4 --warmup 2 --prefill-delay 0.01 --token-delay 0.005 > benchmarks/simulated-c4.json
```

Every prompt contains eight word pieces. Output lengths alternate 2 and 16,
for 108 measured generated pieces per scheduler/run. Continuous batching uses
8 active slots, a 32-token iteration budget, and 8192 KV slots; both schedulers
use a 64-request waiting capacity. Memory is not constrained in these runs.

| Client concurrency | Scheduler | Completed | Tokens/s | TTFT p95 (ms) | End-to-end p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | FCFS | 12 | 144.15 | 16.93 | 102.55 |
| 1 | Continuous | 12 | 140.24 | 17.35 | 105.67 |
| 4 | FCFS | 12 | 144.39 | 243.88 | 249.97 |
| 4 | Continuous | 12 | 427.77 | 17.61 | 122.80 |

All four measured runs completed without rejection, error, or timeout. The
concurrency-1 runs show similar rates; concurrency 4 demonstrates overlapping
simulated waits in continuous batching and serialized work in FCFS. This is one
small observation per setting, with no confidence intervals. Async sleeps,
iteration barriers, machine load, and event-loop overhead determine these
numbers. They are not GPU throughput or production serving measurements.

The later [prefill/decode policy experiment](scheduling-policy.md) compares two
continuous-batching policies with mixed prompt lengths over three repetitions.

The [optional local-model smoke check](local-model.md) records real CPU model
correctness, lifecycle, and memory snapshots separately from simulator results.

The [chat adapter verification](chat-api.md) checks full-history JSON and streaming
responses against reference real-model generation, separately from latency tests.

The [external HTTP baseline](http-local.md) measures actual client receipt from a
running local model server; the browser smoke and screenshot verify the visual
client separately from latency benchmarks.

The [portable HTTP preflight milestone](http-compatibility.md) adds model selection,
readiness configuration, and explicit token usage policies, with recorded real
CPU model compatibility checks. It does not contain external engine measurements.

The [llama.cpp compatibility check](llama-cpp.md) verifies the pinned external
engine's conversion, prompt tokens, and streaming protocol. Its short smoke runs
are not comparative performance evidence.

The [repeated CPU comparison](cpu-comparison.md) records five repetitions of
short/long/mixed workloads for both engines, with 640 measured completions and
separate lifecycle/RSS/admission probes. It identifies delayed queued-disconnect
cleanup in our runtime and a concrete reason to test real tensor batching later.

The [queued-disconnect fix](disconnect-cleanup.md) follows that finding with
deterministic saturation/FIFO/ownership tests and a targeted real CPU recheck.

The [retained-memory investigation](retained-memory.md) separates idle RSS,
released cache objects, traced Python allocations, and native allocator counters
across longer bounded runs. It records the evidence for keeping normal cleanup
unchanged before the next tensor-batching experiment.
