# Inference engine comparison: research findings

Original research snapshot: **2026-11-09**. At that point, source review was complete; external engine
installation, compatibility checks, and comparative measurements **not run**.
Exact upstream revisions and source hashes are recorded in [sources.json](sources.json).
Documentation sites marked `moving` can change independently of those revisions.

Current follow-up: [llama.cpp compatibility](../../benchmarks/llama-cpp.md) and the
[first repeated CPU comparison](../../benchmarks/cpu-comparison.md) are complete.
Results depend on the workload; the report recommends fixing delayed queued
disconnect cleanup first. This document retains the original research snapshot
below, including its then-pending experiment recommendations.

## Purpose

An inference engine loads a trained model and manages the work needed to produce
answers. This project makes that process visible: requests wait, receive execution
time, consume cache space, stream output, and release resources. Established
engines solve the same problems at much larger scale.

The question is which useful, measurable improvement to build next. This review
does not establish that another engine has a bug or lacks a feature. It also does
not establish a performance winner. Features below are documented capabilities,
not capabilities verified on this Mac with our checkpoint.

## Current project baseline

- The deterministic simulator supports FCFS and continuous admission, shared
  token budgets, optional decode-first scheduling, bounded queues, cancellation,
  and logical KV capacity reservations.
- The real executor uses Transformers and PyTorch on CPU in float32, with two
  intra-op threads. Each request owns its real KV tensors, while a serial worker
  performs model forwards. Multiple active requests **do not form a shared tensor
  batch**. Logical token reservations are not a physical RAM limit.
- The text-only chat API and browser client support streamed conversations and
  Stop. Browser history is sent again as messages; there is no cross-request
  prefix cache. Health exposes request progress and logical/actual cache snapshots.
- The external HTTP benchmark measures receipt of content, successful output
  throughput, completion, and errors at fixed client concurrency. Its setup and
  stream expectations currently depend on this runtime.

The pinned checkpoint is `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, already stored under ignored
`.models/smollm2`. No new model or engine was downloaded for this review.
See [local execution](../../LOCAL_MODEL.md), [API behavior](../../CHAT_API.md),
and [browser/HTTP baseline](../../benchmarks/http-local.md).

The recorded HTTP runs contain only eight measured requests each. Throughput
was 53.94 output tokens/s at concurrency 1 and 54.55 at concurrency 2. This is a
small serialized-CPU baseline, not evidence of a batching speedup or production
capacity. Those runs were not repeated during this documentation milestone.

## What the other engines already do

| Engine | Relevant documented capabilities | What that means for this project |
| --- | --- | --- |
| vLLM | PagedAttention memory management, continuous batching, chunked prefill, prefix caching, quantization, speculative decoding, compatible serving APIs | A reference for production scheduling/cache concepts. Adding these features here would be catching up, not establishing novelty. [Pinned overview](https://github.com/vllm-project/vllm/blob/0ee57d457f56caeaf1b57d5ec527094772e97151/README.md) |
| SGLang | Session-aware radix cache and eviction; production metrics and tracing documentation; broad hardware support | A reference for reuse across requests and cache behavior under pressure. [Cache documentation](https://docs.sglang.io/docs/advanced_features/session_radix_cache), [observability](https://docs.sglang.io/docs/advanced_features/observability), [pinned overview](https://github.com/sgl-project/sglang/blob/704681229f1cdda291dbd2261dabb2ae80a4a49d/README.md) |
| llama.cpp | CPU/GPU inference, continuous batching with parallel slots, chat API, web UI, quantized formats, metrics, slot/cache controls | A promising first local control experiment because it exposes CPU, thread, and KV precision settings. Format conversion and exact checkpoint support still need checking. [Pinned server guide](https://github.com/ggml-org/llama.cpp/blob/a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5/tools/server/README.md) |
| MLX LM | Apple Silicon inference, quantization, batch generation, reusable prompt caches, rotating KV cache; server source includes a batching path | A promising later accelerated deployment comparison on this Mac. Supported batching depends on the model/cache path; acceleration is a separate experimental variable. [Pinned overview](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/README.md), [server source](https://github.com/ml-explore/mlx-lm/blob/56230fa02b8ff8330daf26dc95c28480423ed5fe/mlx_lm/server.py) |

Do not describe vLLM or SGLang as GPU-only: their reviewed overviews list CPU and
Apple Silicon routes, including plugins or backend-specific support. Whether a
particular route supports our model and desired configuration remains untested.
See the pinned overviews above.

Metrics and tracing are also established features, not a missing category we
discovered. For example, [vLLM's metrics documentation](https://docs.vllm.ai/en/latest/usage/metrics/)
describes serving timings and cache-related telemetry, and SGLang documents
Prometheus metrics and request tracing. Our opportunity is a smaller, understandable
workflow for explaining decisions, subject to usability and experiment evidence.

## Findings that affect the next implementation

1. **Make the benchmark portable before comparing speed.** In
   `src/llm_runtime/http_benchmark.py`, setup expects exactly one advertised model
   and JSON `/health`. Requests require streamed final usage. Another server may
   expose different discovery, readiness, or usage behavior. A successful chat
   request alone does not demonstrate benchmark compatibility.
2. **Match both weight and KV precision.** llama.cpp's reviewed server guide
   defaults K/V cache types to float16, while our current executor uses float32
   tensors. A nominal CPU comparison would still be confounded unless these
   settings are aligned or explicitly reported as different profiles.
3. **Compare actual work.** Greedy generation can stop on EOS before `max_tokens`.
   Record actual prompt/output counts, finish reasons, and the rendered chat
   template. Network content chunks do not correspond one-to-one with tokens.
4. **Treat cache retention according to policy.** An engine may deliberately
   retain prefixes after requests finish. Stable retained memory is not by itself
   a leak. Test bounded growth, eviction/reuse, and recovery; distinguish logical
   reservations, live KV tensors, allocator retention, and process memory.
5. **Separate model quality from serving correctness.** Wrong facts from the small
   model are not automatically scheduler bugs. For divergent outputs, check
   tokenizer/template, special tokens, generation settings, and numerical backend
   differences before drawing conclusions about either engine.

## Recommended next phase

First try **llama.cpp as a CPU control**, conditional on faithful conversion of
the existing checkpoint and matching tokenization/template/precision. Then consider
an **MLX accelerated profile**, clearly separated from the CPU control. These are
experiment recommendations, not completed compatibility tests or benchmark results.

The [experiment plan](EXPERIMENT_PLAN.md) defines the preflight, measurements,
workloads, and implementation gates. If an equivalent configuration is unavailable,
record the mismatch and compare deployment profiles without attributing a speed
difference solely to scheduling.

After measurements, choose between real tensor batching, reusable prefix caching,
or decision traces with deterministic workload replay. The last option fits this
project's educational focus particularly well: explain why a request waited or
ran, and connect that decision to latency and memory. Its value is a **hypothesis**;
this review does not prove that mature engines lack equivalent tools.
