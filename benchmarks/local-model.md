# Local CPU model smoke — 2026-10-22

This milestone verifies actual model execution separately from the simulator
benchmark. The reproducible check is:

```bash
python -m pip install -e '.[dev,local]'
python -m llm_runtime.download_model
python scripts/verify_local_model.py > benchmarks/local-model-smoke.json
```

The model card and pinned revision are documented in [LOCAL_MODEL.md](../LOCAL_MODEL.md).
The downloaded checkpoint stays in ignored `.models/`; only the report is committed.

Recorded environment: Apple M3, 16 GB RAM, macOS 15.7.2, Python 3.12.7, PyTorch
2.14.1, Transformers 4.57.6. All inference ran on CPU with two intra-op threads,
float32 parameters, two active request slots, and eight logical tokens per
scheduler iteration. Simulation delay settings in the JSON configuration are
ignored by this backend.

Two API requests ran concurrently through in-process ASGI. Every emitted token ID
matched the library's greedy `model.generate` reference for that prompt. Concatenated
text matched canonical tokenizer decoding. One response generated eight tokens
including EOS; the other reached the 24-token limit. The France prompt produced
`The capital of France is Paris.`. A separate decode result was gated to verify
cancellation and health responses while its request remained active; cancellation
discarded the result, ended the stream, and released all per-request caches.

| Recorded observation | Value |
| --- | --- |
| Model load time (local files) | 2.20 seconds |
| Model parameter tensor bytes | 538,060,032 (~513 MiB) |
| Maximum observed KV tensor snapshot | 4,377,600 (~4.18 MiB) |
| Peak process RSS during the whole check | 1,121,042,432 (~1.04 GiB) |
| Final live KV tensors | 0 bytes |
| Concurrent request TTFT | 353.94–361.57 ms |
| Concurrent request end-to-end duration | 534.27–727.49 ms |

These are single-run observations, not performance guarantees. The process peak
includes imports, model loading, reference generation, API handling, and allocator
overhead; it is not memory attributable solely to serving. KV snapshots exclude
weights and temporary tensors and are not a physical memory cap. The report's
request timings measure scheduler emission, not client receipt. ASGITransport
buffers responses, and the cancellation gate is artificial.

A separate live single-worker Uvicorn run on loopback verified `/health`, streamed
real token IDs/text through curl, returned `done` with `finish_reason="eos"`, and
shut down cleanly. Tests cover thread-boundary responsiveness and safe repeated
shutdown cancellation without relying on timing thresholds. No GPU, large-model,
arrival-rate, or production throughput claims are made.
