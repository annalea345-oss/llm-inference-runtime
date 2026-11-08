# External HTTP streaming baseline — 2026-11-08

These runs use the existing pinned SmolLM2 checkpoint on the local CPU server.
They measure client receipt through actual loopback HTTP, unlike earlier
in-process scheduler benchmarks and buffered ASGI correctness checks.

Start the server in one terminal:

```bash
LLM_MODEL_PATH=.models/smollm2 uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
```

Run the client in another (the recorded run used port 8765):

```bash
python -m llm_runtime.http_benchmark --base-url http://127.0.0.1:8000 --requests 8 --concurrency 1 --warmup 2 --max-tokens 24 > benchmarks/http-local-c1.json
python -m llm_runtime.http_benchmark --base-url http://127.0.0.1:8000 --requests 8 --concurrency 2 --warmup 2 --max-tokens 24 > benchmarks/http-local-c2.json
```

Each request asks `What is the capital of France? Answer briefly.`. Both runs
completed all eight measured requests, with no rejection, error, or timeout.
Each completion reported eight output tokens including EOS: 64 measured output
tokens per run. Two warmup requests per run are excluded. No browser clients were
running during measurement. Automated test workloads were also stopped. Server
logs confirmed one connection reused in the single-client run and two reused in
the concurrent run.

| Client concurrency | Mean first content (ms) | First content p95 (ms) | Completion p95 (ms) | Output tokens/s |
| --- | --- | --- | --- | --- |
| 1 | 68.31 | 75.07 | 154.98 | 53.94 |
| 2 | 133.73 | 148.32 | 294.52 | 54.55 |

Header receipt averaged 1.05ms and 1.10ms respectively. Headers/role chunks arrive
before model output and do not count as first content. Received content-delta
gap p95 was 14.02ms and 27.04ms. These gaps are not necessarily model-token gaps;
network buffering and client scheduling can affect them.

The similar aggregate throughput with increased per-request latency is consistent
with this backend's serialized CPU forwards. Continuous scheduling does not yet
combine requests into tensor batches. The sample is small, each configuration
has one recorded run, and these values are not production capacity guarantees.

Environment: Apple M3/16 GB, Python 3.12, float32 CPU model, two intra-op threads,
two active request slots, round-robin policy. Health/configuration, discovered
model alias, workload, raw samples, and connection-pool methodology are included
in the reports. The model weights and server are already loaded; warmup/metadata
connections can be reused. This is closed-loop load, not a fixed arrival trace,
and loopback timings exclude remote network distance. Model downloads, startup,
and reference-generation work are not part of the measured interval.

The [browser smoke](browser-client-smoke.json) separately verifies the real UI,
two concurrent clients, cancellation, history handling, and 390/320px layout.
The [desktop screenshot](browser-client.png) shows an actual model answer.
