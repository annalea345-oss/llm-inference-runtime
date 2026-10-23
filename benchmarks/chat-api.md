# Chat adapter verification — 2026-10-23

The [chat API smoke](chat-api-smoke.json) uses the existing pinned local SmolLM2
checkpoint. Reproduce after installing the optional local/development dependencies
and downloading the checkpoint as described in [LOCAL_MODEL.md](../LOCAL_MODEL.md):

```bash
python scripts/verify_chat_api.py > benchmarks/chat-api-smoke.json
```

The input is a full conversation with an initial system instruction, user message
`My name is Ada.`, assistant reply `Hello, Ada.`, and final question
`What is my name?`. An independent reference call applies the checkpoint's chat
template directly to this list and runs greedy `model.generate`.

Both nonstream JSON and streaming chat responses matched the reference text:
`My name is Ada.`. This verifies tokenization and output consistency, not semantic
answer quality. The usage was 48 prompt tokens, six generated tokens including EOS,
and 54 total. The stream contained eight JSON chunks including the final usage
chunk, followed by `[DONE]`. Chunk identity/creation fields stayed stable. The
runtime finished with zero live per-request KV tensors and no registered requests.

This was CPU execution on the same Apple M3/Python 3.12 environment used for the
local executor milestone; exact package versions and model revision are in the
JSON report. HTTPX's ASGI transport buffers response bodies, so this check does
not measure network streaming latency. It also does not store conversations.

A separate live Uvicorn/curl check exercised `GET /v1/models` and streamed the
same conversation with real token text, a `stop` finish chunk, final usage, and
`[DONE]`; the test server was stopped afterward. Deterministic direct-ASGI tests
cover incremental text, disconnects before/after content, header-send failure,
and task cancellation while collecting a nonstream response.
