# Prefill/decode policy experiment — 2026-10-22

The default round-robin policy is compared with optional decode-first scheduling.
Each policy gets the same deterministic workload and fresh scheduler. The client
runs in process; no HTTP server or model is involved. Reports use schema 2 with
per-run configuration and per-request prompt token counts.

Reproduce three independent comparisons from the repository root:

```bash
for run in 1 2 3; do
  python -m llm_runtime.benchmark --scheduler policies --requests 16 \
    --concurrency 4 --warmup 4 --prompt-words 2 64 --output-tokens 16 2 \
    --token-budget 4 --prefill-token-budget 2 \
    --prefill-delay 0.04 --token-delay 0.002 > "benchmarks/policies-mixed-$run.json"
done
```

Every run measures 16 requests, with four clients and four excluded warmup requests.
Two-word prompts request 16 outputs; 64-word prompts request two. Each measured
policy run generates 144 pieces. The defaults of 8 active requests, 64 waiting
slots, and 8192 cache slots leave memory unconstrained. Round-robin ignores the
prefill cap; decode-first limits total prefill to two tokens per iteration.

| Run | Policy | Tokens/s | Token gap p95 (ms) | TTFT p95 (ms) | End-to-end p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 | Round-robin | 188.84 | 8.56 | 289.61 | 292.16 |
| 1 | Decode-first | 167.14 | 2.74 | 392.41 | 394.79 |
| 2 | Round-robin | 187.06 | 8.86 | 290.60 | 293.16 |
| 2 | Decode-first | 166.82 | 2.57 | 393.29 | 395.62 |
| 3 | Round-robin | 189.19 | 8.48 | 288.17 | 290.69 |
| 3 | Decode-first | 171.82 | 2.51 | 388.76 | 391.03 |

All six measured policy runs completed without rejection, error, or timeout.
Decode-first shortened output gaps, but the capped prefill delayed first tokens
and reduced throughput. This is a configurable tradeoff; round-robin remains the
default. The long prompts in this workload dominate tail first-token latency.
Individual request samples permit more detailed comparisons by prompt length.

The simulator apportions the same total 40ms prefill sleep across each prompt's
chunks, regardless of length. Longer prompts require more iterations, not a
larger total simulated prefill delay. Decode steps sleep 2ms, and the iteration
barrier waits for the slowest selected step. Small sleeps, event-loop overhead,
and machine load affect results. These three repetitions show a consistent
tradeoff for this workload; they do not establish confidence intervals, universal
policy superiority, or GPU performance. A closed-loop client also changes future
submission times as completions change. Request content/order is fixed, not an
identical open-loop arrival trace.
