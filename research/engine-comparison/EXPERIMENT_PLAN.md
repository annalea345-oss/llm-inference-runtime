# Local comparison experiment plan

Status: **first repeated CPU comparison complete**. Read [the findings](README.md) first.
This plan does not change runtime behavior or install another engine.

Update, 2026-11-09: the portable measurement/preflight portion of step 2 is now
implemented and verified against this runtime. See [HTTP compatibility](../../benchmarks/http-compatibility.md).
Follow-up: [pinned llama.cpp compatibility](../../benchmarks/llama-cpp.md) is now
verified, including conversion, four prompt fixtures, and the portable HTTP smoke.
Follow-up: [repeated CPU comparison](../../benchmarks/cpu-comparison.md) now records
640 measured completions plus separate cancellation, RSS, and logical-admission
observations. Fixed-arrival fairness, cold/warm cache reuse, long-context stress,
and production overload capacity remain future work. The protocol below is the
original plan; the report identifies the executed subset and limitations.

## 1. Compatibility preflight

Use the current Apple M3 / 16 GB host. Record the exact OS, hardware, available
memory, Python/dependency versions, project commit, upstream build commit, compiler,
and build options at execution time. Stop browser polling and other test workloads
during measured runs. Run one engine at a time on loopback.

Start with the pinned llama.cpp revision in [sources.json](sources.json). Before
installation/conversion, check model architecture support, build requirements,
licenses, and disk/memory needs. Use a separate ignored build/artifact directory;
record exact commands and hashes. Do not replace the existing checkpoint or use
an unpinned community quantization as an equivalent model.

Verify that conversion preserves weights at the chosen precision, vocabulary,
special-token IDs, EOS behavior, and chat template. Save hashes of input weights,
converted artifact, tokenizer, and template. Compare rendered prompt token IDs
for single-turn, system-instruction, multi-turn, and non-ASCII fixtures. A text
match alone does not prove token equivalence. Document any unsupported setting.

Capture the built server's `--help` and explicit launch configuration. For the
CPU control, disable acceleration/offloading, use float32 weights and K/V caches
where supported, and align generation/prefill thread counts with the runtime's
two PyTorch intra-op threads. Record other thread pools rather than claiming
identical CPU scheduling. Set compatible per-request context/output limits,
parallel request limits, and cache policies. Verify actual backend/precision in
startup logs; a default configuration is not an acceptable control.

Use greedy generation, matched EOS handling, no speculative decoding, and no
prefix reuse for the initial cold comparison. Floating-point kernels may still
produce different continuations. Inspect divergences and actual work; do not
require bitwise output identity across engines as a universal correctness test.

## 2. Make external measurement portable

Proposed changes for the **next implementation**, not this documentation phase:

| File/area | Intended change |
| --- | --- |
| `src/llm_runtime/http_benchmark.py` | Explicit model selection; separate runtime-specific metadata from a portable chat streaming transport; capability preflight and precise unsupported-result reporting |
| `tests/` HTTP benchmark coverage | Fixtures for alternate discovery/readiness shapes, absent/unsupported usage, fragmented SSE, deadlines, and connection reuse; retain strict identity and terminal validation |
| `benchmarks/` | Versioned workload files, configuration/run manifests, raw samples, and a report separating matched controls from deployment profiles |
| Research docs, README, PROGRESS | Record actual commands, compatibility failures, results, limitations, and the selected follow-up |

Keep the existing loopback restriction. Configure readiness checks per engine;
record readiness separately from liveness. Preflight model selection, request
fields, final usage support, finish reason, `[DONE]`, error semantics, and HTTP
connection reuse. Do not silently treat missing usage as zero or estimate tokens
from content chunk counts. If usage cannot be verified, report latency separately
and exclude that run from comparable token-throughput results.

Keep current runtime measurements available. Adapt metadata and capabilities
without weakening malformed/truncated stream detection. Test an actual alternate
server before labeling the adapter compatible.

## 3. Workloads and measurement

Store exact prompts, message histories, output limits, and workload IDs before
running. Establish context limits with tokenization, not character counts. Start
small and bound total requests and memory pressure; do not intentionally exhaust
host RAM.

| Workload | Question it answers |
| --- | --- |
| Short prompts/short answers, concurrency 1 then 2 | Basic per-request overhead and overlap; reproduce the existing baseline |
| Long prompts and long output limits, separately | Prefill cost versus decode cost; record actual EOS lengths |
| Mixed short/long requests | Does a short request wait behind expensive work? Report per-class latency and maximum wait |
| Identical prefixes, cold then warm | Does reuse help, at what memory cost? Explicitly reset or restart between cold runs |
| Waiting/active cancellation and client disconnect | Does abandoned work stop, and can a follow-up request complete? |
| Bounded sustained use and admission pressure | Do errors, retained memory, and recovery settle under repeated load? |

The existing client is **closed-loop**: each client submits its next request after
the previous one finishes. Faster engines therefore see different arrival times.
Use that mode for throughput at fixed concurrency and label it accurately. A
claim about identical workload replay or fairness under equal arrival pressure
requires a separately implemented fixed-arrival schedule, bounded client queue,
and recorded actual submission times. Do not call closed-loop runs identical
arrival traces.

For each configuration, run at least five repetitions, with excluded warmups
numbering at least the concurrency. Alternate engine order to reduce temperature
and background-load bias. Fix connection-pool/deadline settings and record errors,
timeouts, offered/completed work, and warmups. Keep startup/model loading outside
steady-state timing; report cold startup separately if measured.

Record raw samples and per-run aggregates for:

- Submit-to-first-nonempty-content latency and submit-to-`[DONE]` completion.
  These are client-observed measurements, not pure model compute timings.
- Successful output tokens per wall-clock second, using verified usage; report
  failures and actual generated lengths alongside throughput.
- Median/p95 latencies and per-class tails, sample counts, and variation across
  repetitions. Small samples do not support precise tail-capacity claims.
- Process memory before load, during load, and after settling; separately record
  engine cache counters where available. Polling can miss peaks and process
  memory is not equivalent to live tensor bytes or unified GPU memory.
- Cancellation timing from client action to observed server cleanup where
  exposed, plus follow-up latency. A closed TCP connection alone proves neither
  stopped computation nor released memory. Mark cleanup unverified when internal
  evidence is unavailable. Document each engine's intended prefix retention.

Each run manifest must identify model/artifact/tokenizer/template hashes, engine
commit and executable hash, full command, backend, weight/KV precision, thread
settings, context/active/cache limits, cache warmness/reset method, workload and
arrival mode, sampling/EOS settings, client version, and run timestamps.

## 4. Evidence gate before choosing the runtime change

Publish a report even if compatibility fails. Distinguish documented features,
observed behavior, and hypotheses. A GPU/Metal or quantized result can demonstrate
a useful deployment improvement, but cannot isolate the benefit of a scheduler.

Choose one improvement using measured impact, implementation size, and usefulness:

- **Real tensor batching** if shared forwards are the relevant bottleneck. Prove
  mixed request lengths, EOS, cancellation, and per-request KV ownership remain
  correct, and measure actual throughput/latency against the serial executor.
- **Reusable prefixes** if repeated context dominates. Prove cache identity,
  invalidation, bounded capacity, eviction, and isolation; compare cold/warm runs.
- **Decision traces and replay** if explaining waits and memory pressure is the
  clearest user need. Define events and reasons first, avoid recording prompt
  text by default, test deterministic simulator replay, and show one actionable
  diagnosis. Inspect existing upstream tools before claiming differentiation.

For any chosen runtime change, preserve the simulator regression suite and add
meaningful lifecycle/invariant tests plus a real-model smoke check. Rerun the
relevant HTTP workloads and report regressions as well as improvements. Keep
model answer quality separate from serving correctness and performance.
