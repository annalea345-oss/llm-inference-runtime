# Queued disconnect cleanup — 2026-11-10

The [CPU comparison](cpu-comparison.md) found that a disconnected queued request
still occupied a waiting place while both active slots were busy. It remained
counted through a two-second observation window. This milestone fixes that
behavior using the existing cancellation owner, rather than waiting for admission.

In everyday terms: when someone leaves the waiting line, their place becomes
available to the next client while the server continues its current work.

## Implementation

- `ContinuousBatchScheduler.abandon(id)` marks live work abandoned and delegates
  to `cancel()`. Waiting work, including the memory-blocked FIFO head, is finalized
  immediately. Surviving queue order and unfinished-task accounting use the
  already tested cancellation path.
- Native and chat streaming responses notify their owning scheduler in cleanup,
  including failed headers and externally cancelled HTTP tasks. The callback is
  bound when constructing the response, so late transport cleanup keeps its
  original scheduler owner across app lifespans.
- Non-streaming chat collection sends the same notification on disconnect or
  handler cancellation. Finished/unknown requests and a stopping scheduler are
  safe to notify; completed responses do not create another terminal event.
- On ASGI 2.4+ a response watches `http.disconnect` concurrently with streaming.
  The installed Starlette implementation otherwise relies on send errors in that
  mode, and a waiting response may have no new output to send. Older transports
  already have a listener. The extra stream/listener tasks are cancelled and
  awaited on exit, including normal completion and transport errors.
- Active requests are only marked for cancellation. Their selected execution
  steps retain ownership until the existing barrier settles; the callback does
  not interrupt a PyTorch operation or release its live tensors early. No terminal
  events are queued for a transport marked abandoned.

Follow-up: the [retained-memory investigation](retained-memory.md) is complete;
it records longer normal runs and separate object/native/Python diagnostics.

The server still has the same queue, active limits, policies, token accounting,
checkpoint, and CPU executor. No dependency, model download, acceleration, or
physical-memory-cap change is part of this fix.

## Deterministic regression evidence

The primary regression test holds two prefill operations open behind an explicit
event, fills a three-place waiting queue, and disconnects its middle waiter.
Before the fix it failed because the count stayed at three. After the fix it
observes two waiters and admits a replacement **before either active step is
released**. It then verifies surviving FIFO order, no execution of the abandoned
request, one release per request, balanced `queue.join()`, and zero final tracked
allocations. This proves behavior without using a speed threshold.

The matrix covers both scheduling policies, native streaming, chat streaming,
non-streaming chat, disconnect notifications, handler cancellation, header-send
errors, and modern ASGI disconnect/task-cancel paths. Additional checks cover
the memory-blocked pending head, shutdown while execution is still settling,
late notifications/restart, and reaping the modern disconnect listener after a
successful response. A separate ASGI 2.4 regression initially timed out without
the listener; it passes with the fix.

`.venv/bin/python -m pytest -q` — **411 passed**, including **28 new cases**.
Existing active cancellation, EOS, real-executor ownership, shutdown, HTTP stream,
browser-parser, and benchmark tests remain in that suite.

## Targeted real-model recheck

The new verifier reuses the comparison's exact queued/active disconnect probes
against only our CPU SmolLM2 server. It checks the checkpoint hashes, runs three
trials of each disconnect type, verifies explicit cancellation with its stream
kept open, and checks follow-up requests and final ownership counters. The
temporary server is stopped in the managed server's `finally` cleanup.

Recorded queued removal after closing the third connection:

| Trial | Observed removal, ms | Active blockers at removal |
| --- | ---: | ---: |
| 1 | 1.19 | 2 |
| 2 | 1.28 | 2 |
| 3 | 1.24 | 2 |

All three active-disconnect probes reached observed idle in approximately 23 ms.
Explicit cancellation still delivered `request_cancelled` and `[DONE]`. All seven
follow-up requests completed. Final state had zero queued/active requests,
reserved/used logical tokens, and tracked KV tensor bytes.

These are **three local observations**, including HTTP and polling delay, not a
queue-cleanup latency guarantee or a production capacity comparison. The decisive
regression evidence is reclamation with both blockers still active. The earlier
comparison's two-second window was a failed observation bound, not a measured
fixed cleanup duration. No RSS or throughput improvement is claimed here.

Raw evidence, commands, package versions, checkpoint hashes, source hashes, and
final state are in [disconnect-cleanup.json](disconnect-cleanup.json). Its project
base commit identifies merged main before this fix; hashes identify the tested
uncommitted implementation. The previous comparison records remain unchanged.

## Reproduce

From the repository root with existing local/dev extras and checkpoint installed,
and port 8020 free:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.verify_disconnect_cleanup \
  --output-dir .experiments/disconnect-cleanup-recheck
```

Use a new output directory. The verifier starts/stops only our local CPU server;
it does not run llama.cpp or repeat the full comparison matrix. Logs remain in
the ignored experiment directory. Browser code and the explicit cancellation
endpoint do not need changes for this behavior.

Next: profile retained process RSS separately, then experiment with real tensor
batching using the recorded larger-output/mixed workloads. This fix establishes
waiting-capacity recovery; it does not explain allocator retention or increase
the real executor's tensor-batch throughput.
