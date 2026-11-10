# Repeated CPU engine comparison

This phase compares the current educational runtime with the previously pinned
llama.cpp CPU build. It also probes disconnect cleanup and observes process RSS.
Neither engine's inference implementation is changed by the experiment.

Follow-up, 2026-11-10: the [queued-disconnect cleanup fix](disconnect-cleanup.md)
now reclaims the waiting request while both active slots remain occupied. The
original comparison data and findings below are preserved unchanged.

In plain language: both servers use the same small model, and we give them the
same tasks. A longer prompt tests the work of reading the question and building
its KV cache (prefill). A larger answer limit tests repeated next-token work
(decode). Mixing these tasks tests whether a quick question stays responsive
beside expensive work. Concurrency 2 represents two independent clients, not two
turns of one conversation. Cancellation checks whether abandoned work stops and
whether another client can still get an answer.

## Protocol

The workload source is [cpu-workloads-v1.json](cpu-workloads-v1.json). It fixes
three user prompts and their output limits:

| Class | Prompt tokens including template | Maximum output tokens |
| --- | ---: | ---: |
| Short question | 40 | 24 |
| Longer prompt | 562 | 24 |
| Larger output limit | 56 | 96 |

These counts are recorded in the run manifest and checked against tokenizer
output. Each homogeneous workload and the mixed pattern run at concurrency 1 and 2, with eight measured requests per
run. Mixed order is longer prompt, short question, larger output limit, short
question, repeated twice. Five repetitions produce 80 timing runs and 640
measured requests across both engines.

Each run excludes one streaming preflight and enough warmups to cover its live
workers and pattern: homogeneous warmup is 1/2 at concurrency 1/2, and mixed
warmup is 4. Engine order alternates per repetition; workload and
concurrency order reverse on alternating repetitions. Each engine is restarted
between repetitions, with only one server running at a time on loopback port
8020. Model loading is excluded. Connection pools are shared within each server
session, and request bodies/stream semantics are the same across engines.

Clients submit their next request after completion. This is **closed-loop load**,
so service-time differences produce different actual arrival times. Raw samples
include submission offsets and request class. It is not an equal-arrival replay
or a direct measurement of internal scheduler queue wait. First-content latency
includes networking, admission, prefill, and any compute contention before text.

Timing runs have no concurrent process-memory or health polling. Baseline/final
state and RSS are sampled outside their measured intervals. Lifecycle tests and
the memory timeline are separate sessions. The host is an ordinary development
machine; load averages are recorded, not controlled by an isolated lab setup.

## Matched settings and remaining differences

Both use the existing pinned SmolLM2 checkpoint, CPU execution, float32 weights
and K/V tensors, two model threads, two active slots, greedy output, and disabled
cross-request prompt reuse. The context settings are 8192 tokens. The prior
[conversion check](llama-cpp.md) established 272 equivalent weight tensors and
prompt-template/tokenizer fixtures. The runner checks the existing checkpoint,
GGUF, executable, and shared-library hashes and verifies the experiment prompts
against llama.cpp tokenization at each external-server startup. Completed request
usage must match the verified prompt count and stay within the output limit.

Our executor serializes separate per-request forwards with eager PyTorch
attention. llama.cpp uses its native CPU backend and actual tensor batching.
The recorded build has Metal, Accelerate, BLAS, and OpenMP disabled. This is a
specific CPU configuration, **not the fastest available llama.cpp deployment**.
PyTorch and GGML use different kernels and libraries. Equal model-thread settings
do not prove identical helper-thread behavior or isolate scheduling as a cause.
The 32-token scheduling/physical batch settings also have different meanings.

Our 8192 logical reservation slots are admission accounting; llama.cpp's shared
KV allocation is a different physical policy. Matching a numeric token capacity
does not establish matching peak resident memory or overload behavior.

## Findings to verify during the experiment

The bounded pilot exposed two behaviors for follow-up:

1. A queued disconnected request remained counted in our runtime's waiting queue
   while two active requests occupied both slots. The pilot's two-second
   observation timed out; after all streams disconnected the runtime recovered.
   The scheduler currently removes abandoned active requests every iteration but
   examines waiting requests during admission. This suggests a targeted admission
   cleanup improvement, rather than a claim about a general memory leak.
2. Longer llama.cpp streams change the integer `created` field between chunks.
   The pinned source calls `std::time(0)` when serializing partial/final chunks in
   [server-task.cpp](https://github.com/ggml-org/llama.cpp/blob/a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5/tools/server/server-task.cpp).
   Our original stable-timestamp check rejected such streams. The client now has
   an explicit `--created-policy per_chunk` option. The default remains `stable`;
   integer timestamp validation, stable request ID/model, finish/usage/terminal
   checks, deadlines, and HTTP draining remain enforced. Samples record timestamp
   changes. There is no silent fallback and no upstream source patch.

The timestamp adjustment is interoperability work in the measurement client. It
is not evidence that llama.cpp produced a wrong answer. The queued-disconnect
observation is a lifecycle/admission finding, not a measured performance benefit
for a future fix.

## Lifecycle and memory observations

In separate sessions, each engine receives an active stream that is closed after
first content and a third queued stream disconnected while two blockers are
active. The runner records server processing/deferred counters, polls for idle,
and verifies a new follow-up request succeeds. Our explicit cancel endpoint is
also exercised while keeping its stream open; the cancellation terminal data is
recorded separately from disconnect cleanup. This does not claim that other
engines lack cancellation APIs.

The queued probe observes removal for up to two seconds before closing blockers.
Whether blockers are still active at removal is recorded, so natural completion
cannot be mistaken for immediate queued cancellation. Cleanup observations are
client action to server-counter observation, with polling/API delay included;
one probe is not a cancellation latency distribution.

Five bounded four-request mixed waves run at concurrency 2 while sampling server
RSS and counters approximately every 100 ms, followed by a settling interval.
RSS comes from macOS `ps` for the server PID, excluding the separate driver.
Polling and scheduling can miss peaks. RSS includes libraries, resident weights,
allocator retention, and cache pages; it is not a live-KV tensor measurement or
an enforced memory bound. A stable cache/allocator allocation is not a leak.
Twenty sustained-observation requests per engine do not establish long-term
stability. This phase does not intentionally exhaust host memory.

## Reproduce

Requires the previous [llama.cpp setup](llama-cpp.md), existing local/dev extras,
and no listener on port 8020. From the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/compare_engines.py \
  --output-dir .experiments/cpu-comparison-recheck --repetitions 5
.venv/bin/python scripts/summarize_engine_comparison.py \
  .experiments/cpu-comparison-recheck \
  --output .experiments/cpu-comparison-recheck/summary.json
```

The output directory must be new. The driver writes raw reports as each run
finishes and marks its manifest completed only after lifecycle checks finish.
It stops its child server in `finally`, including failed experiments. Failed or
incomplete runs cannot produce a headline summary. Do not run unit tests,
browser clients, or other inference jobs concurrently with the timing experiment.

The summarizer reports distributions across **per-run statistics**. With only
eight requests per run, a nearest-rank p95 is effectively the slowest observed
request; it is not a reliable production tail estimate. Per-class mixed samples
are smaller still. Report sample counts, actual generated lengths, failures,
and variation rather than confidence claims.

## Recorded results — 2026-11-09

All **640 measured requests across 80 timing runs completed**, with no measured
rejections, errors, or timeouts. Both engines reported the same output counts:
short questions stopped at 8 tokens; longer-prompt responses used their 24-token
limit; larger-output responses used their 96-token limit. This establishes equal
reported work counts for these cases, not identical logits or answer quality.

Raw results and configuration are in [cpu-comparison-v1/manifest.json](cpu-comparison-v1/manifest.json).
The [summary](cpu-comparison-v1/summary.json) aggregates all five repetitions;
each `repetition-…json` file contains preflight, warmup, raw measured samples,
submission offsets, and per-class results. Timing and lifecycle sessions were
completed and their child servers stopped before the final test suite.

First nonempty content, milliseconds: **median of the five per-run means**, with
the minimum–maximum per-run mean in brackets. Each row represents 40 measured
requests per engine. These are client-observed timings.

| Workload | Clients | Our runtime, ms | llama.cpp, ms |
| --- | ---: | ---: | ---: |
| Short | 1 | 82.2 [79.8–104.1] | 139.7 [138.3–149.0] |
| Short | 2 | 185.9 [152.4–203.3] | 258.5 [245.6–269.7] |
| Longer prompt | 1 | 780.5 [692.0–899.9] | 2188.2 [2031.3–2937.9] |
| Longer prompt | 2 | 1511.0 [1411.9–1647.8] | 3395.2 [3178.6–3533.5] |
| Larger output | 1 | 104.9 [93.5–108.2] | 275.0 [227.6–300.6] |
| Larger output | 2 | 200.0 [180.4–205.7] | 302.8 [283.8–305.1] |
| Mixed | 1 | 280.0 [262.8–286.2] | 711.5 [690.1–725.0] |
| Mixed | 2 | 425.4 [392.2–444.9] | 1056.1 [1025.9–1084.5] |

Output tokens per second: **median of the five whole-run throughput rates**,
using validated server usage. It is not an average of individual token speeds.

| Workload | Clients | Our runtime, tokens/s | llama.cpp, tokens/s |
| --- | ---: | ---: | ---: |
| Short | 1 | 48.58 | 38.56 |
| Short | 2 | 42.91 | 42.87 |
| Longer prompt | 1 | 20.25 | 9.81 |
| Longer prompt | 2 | 20.62 | 8.05 |
| Larger output | 1 | 63.20 | 65.88 |
| Larger output | 2 | 65.13 | 109.03 |
| Mixed | 1 | 44.18 | 30.34 |
| Mixed | 2 | 45.06 | 34.16 |

Our runtime started text sooner in all these profiles. llama.cpp's larger-output
throughput increased from about 66 to 109 tokens/s with two clients, while our
serial executor stayed around 63–65. That makes real tensor batching a useful
future performance experiment. It does not prove batching alone caused the
difference, or that our runtime is generally faster: this tiny model, CPU build,
disabled BLAS/Accelerate/Metal, and small batch settings define the comparison.

Within the mixed workload at concurrency 2, short-question mean first-content
latency had a per-run median of **178.2 ms** for our runtime and **735.6 ms** for
this llama.cpp configuration. Each engine has only 20 short-question samples in
that group, and arrivals differ under closed-loop load. This is responsiveness
evidence for the workload, not a starvation proof or a latency guarantee.

### Cancellation, admission, and memory

The separate [runtime lifecycle record](cpu-comparison-v1/lifecycle-runtime.json)
and [llama.cpp lifecycle record](cpu-comparison-v1/lifecycle-llama_cpp.json) show:

- Active disconnect reached observed idle in roughly 23 ms for both engines.
  These are single probes including polling; no latency ranking is supported.
- llama.cpp's queued request disappeared from its deferred counter after roughly
  926 ms while both blockers were still active. Our runtime still reported one
  queued request after the two-second observation window. Closing all streams
  brought both engines to idle, and follow-up requests succeeded.
- Our explicit active-cancel endpoint produced the documented `request_cancelled`
  terminal error and `[DONE]` while the connection stayed open, then reached idle.
  This separates successful explicit cancellation from delayed queued-disconnect
  cleanup. The latter was also observed in the earlier pilot.
- At the end, our runtime had **zero active/queued requests, logical reserved/used
  slots, and tracked KV tensor bytes**. llama.cpp reported zero processing/deferred
  requests and zero cached-prefix tokens; its allocated KV storage is not expected
  to disappear when idle.

A separate [native reservation-pressure probe](cpu-comparison-v1/admission-pressure-runtime.json)
used two identical 7,400-character prompts with a 512-token output cap. Each
required 4,242 reservation slots: both fit individually, while their combined
8,484 slots exceeded the 8,192 capacity. Health showed one active request and one
waiting despite a free active slot. Explicit waiting cancellation removed the
queued request, active cancellation released all tracked KV, and a follow-up
succeeded. The observation had zero settled used-token slots at the pressure
snapshot: reservations describe planned maximum usage, not actual tensor bytes.
This is a logical-admission test, not an OS-RAM exhaustion experiment.

During the five four-request memory waves, sampled server RSS was:

| Observation, MiB | Our runtime | llama.cpp |
| --- | ---: | ---: |
| At readiness | 625.52 | 907.34 |
| After first wave | 857.08 | 912.59 |
| After fifth wave | 872.62 | 914.14 |
| Highest sample during waves | 872.67 | 914.19 |

RSS increased across this short observation, particularly in our runtime. Zero
tracked live KV does not explain all retained process memory. Allocator retention,
library workspaces, Python objects, and other effects need separate profiling;
these data establish neither a leak nor a steady-state plateau. Do not describe
the process-memory behavior as proven bounded or infer a physical-memory advantage
from this table. Model loading/allocation policies differ between engines.

The longer streams recorded **281 timestamp transitions** in llama.cpp's measured
samples. The explicit timestamp policy made those streams measurable while stable
ID/model and all terminal/usage checks remained enforced. This also explains why
the previous short compatibility smoke was insufficient to cover longer streams.

### Recommended next milestone

Fix **delayed cleanup of disconnected waiting requests when all active slots are occupied**
first. The evidence is specific, reproducible, and concerns waiting-capacity
correctness. Reuse the existing cancellation ownership rules or prune abandoned
waiting work; preserve FIFO order, queue/task accounting, and active step-boundary
KV release. Add deterministic tests with both active slots occupied, disconnect
queued work, and prove its waiting slot becomes available before active work ends.
Then rerun the real queued-disconnect probe and follow-up recovery check.

After that, prototype real tensor batching against the larger-output workload,
where concurrency currently adds little throughput to our serial executor.
Retain mixed-class latency checks and profile retained RSS before making memory
efficiency claims. Prefix caching, hardware acceleration, fixed-arrival fairness
replay, long-context stress, and production overload capacity remain future work.

Validation for this milestone: **383 pytest cases passed**, including workload
validation, mixed-client concurrency, warmup exclusion, usage mismatch handling,
explicit timestamp-policy validation, managed-process cleanup, and rejection of
incomplete/inconsistent summary inputs. No model/executor/scheduler implementation
was changed, and no new dependency or model was downloaded.

To repeat only the bounded native reservation probe after the main comparison:

```bash
.venv/bin/python -m scripts.probe_runtime_admission \
  --output-dir .experiments/cpu-admission-recheck
```
