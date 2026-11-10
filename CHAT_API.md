# Text-only chat API

The runtime now exposes a small Chat Completions-compatible adapter:

- `GET /v1/models` lists the single served model alias.
- `GET /v1/models/{id}` retrieves that model entry.
- `POST /v1/chat/completions` returns one assistant message or streams text deltas.

The wire formats follow the official OpenAI documentation for
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[streaming chunks](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events),
and [model discovery](https://developers.openai.com/api/reference/resources/models/methods/list).
This is a defined subset, not the full OpenAI API. Requests run through the
existing local scheduler and model; no hosted API, billing, or OpenAI key is involved.

## Run and call

For real responses, follow [LOCAL_MODEL.md](LOCAL_MODEL.md), then start:

```bash
LLM_MODEL_PATH=.models/smollm2 uvicorn llm_runtime.local:create_local_app \
  --factory --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/v1/models
```

The real backend advertises `local-model`. The default simulator advertises
`simulated` and still only cycles words; its role-labelled prompt includes the
full supplied history. Custom executors default to `runtime` or can expose a
`model_id` attribute. These are server aliases, not hosted model names. A model
entry's `created` timestamp records app availability, not checkpoint training.

Get a complete response:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"What is the capital of France? Answer briefly."}],"max_completion_tokens":24,"temperature":0}'
```

The JSON object has `id`, `object: "chat.completion"`, `created`, `model`, one
choice containing `message.role: "assistant"` and `message.content`, a finish
reason, and prompt/completion/total token usage. EOS maps to `"stop"`; the output
limit maps to `"length"`. Real usage counts model tokens, including prompt
formatting and an emitted EOS token; simulator usage counts whitespace pieces.

Enable streaming and optional final usage:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Explain rain briefly."}],"max_tokens":32,"stream":true,"stream_options":{"include_usage":true}}'
```

Frames use `data: {...}` with no named SSE events. Chunks have
`object: "chat.completion.chunk"` and stable completion ID, timestamp, and model.
The first delta supplies the assistant role, subsequent deltas supply content,
and the last choice delta supplies a finish reason. With `include_usage`, an
additional chunk has `choices: []` and final usage; preceding chunks have null
usage. The stream ends with `data: [DONE]`. Incomplete byte tokens and EOS can have
empty text and are omitted from content deltas but still count in usage. Network
chunks need not match SSE frames; clients must parse incrementally.

## Conversations and supported options

Send the entire history in every request, for example:

```json
{
  "model": "local-model",
  "messages": [
    {"role": "system", "content": "Answer briefly."},
    {"role": "user", "content": "My name is Ada."},
    {"role": "assistant", "content": "Hello, Ada."},
    {"role": "user", "content": "What is my name?"}
  ],
  "max_completion_tokens": 32,
  "temperature": 0
}
```

The real backend applies the model's chat template to the message list once,
preserving roles and order. It does not wrap the serialized history as one user
message. The server stores no conversations between requests; include previous
assistant answers in the next request yourself.

| Input | Supported subset |
| --- | --- |
| `model` | Required exact alias from `/v1/models` |
| `messages` | 1–64 text-only messages, each with nonempty string content |
| Roles | Optional initial system message, then alternating user/assistant turns ending in user |
| Content limit | At most 8192 content characters across the whole history |
| Output limit | `max_tokens` or `max_completion_tokens`, integer 1–512; default 16; mutually exclusive |
| `stream` | Boolean, default false |
| `stream_options` | Optional `include_usage` boolean, only with streaming |
| `temperature` | Omitted, null, or numeric zero; greedy decoding only |
| `n` | Omitted or integer one |

Other fields are rejected. This includes sampling settings, stop sequences, tools,
images/audio, developer/tool roles, response formats, logprobs, and conversation
storage. The adapter does not claim compatibility with every chat application's
default request settings. Configure a client to send only the supported subset.

## Errors, cancellation, and ownership

Validation errors use HTTP 400 with an `error` object containing `message`,
`type`, `param`, and `code`. Unsupported model aliases return 404, full/unavailable
runtime returns 503, and impossible model-context or KV reservations return 422.
A custom tokenizing executor without `tokenize_messages` returns 501. Unexpected
preparation/execution failures return safe 500 errors without internal details.
The original `/generate` validation still returns its existing HTTP 422 format.

After a stream starts, HTTP status remains 200. Execution failure, shutdown, or
explicit cancellation sends an error object as an SSE data frame, followed by
`[DONE]`, without a success finish chunk or usage summary. Nonstream cancellation
returns HTTP 409 (`request_cancelled`); shutdown returns 503. This avoids inventing
a nonstandard cancelled finish reason in the compatible success schema.

The `X-Request-ID` response header contains the raw runtime ID. Use it with the
existing `POST /requests/{id}/cancel` endpoint; the `chatcmpl-...` completion ID
is a separate wire identifier and is not accepted by that endpoint. Stream headers
arrive before completion; nonstream headers arrive with the completed response.
Closing the HTTP request marks it abandoned and notifies the owning scheduler.
Waiting requests are removed immediately on that notification, including while
all active slots are occupied. Active resources remain owned until in-flight
execution settles. A disconnected nonstream transport may receive a diagnostic 499
`client_disconnected` response if it can still accept sends.

Both formats share the same bounded queue, active capacity, scheduling policies,
logical reservations, per-request model caches, and cancellation races as
`/generate`. Nonstream requests watch for disconnect while collecting output;
streaming uses response cleanup, including header-send failure. On ASGI 2.4+
transports it also watches disconnect notifications while awaiting output,
because a queued response may have nothing to send that could fail. Older
transports use Starlette's existing listener. Listener/stream tasks are reaped
on completion, errors, or cancellation.
In-flight model work retains ownership until its thread settles. Completed
responses do not retain live request registry entries.

## Connecting a chat client

Use base URL `http://127.0.0.1:8000/v1`, select the model from discovery, set
sampling temperature to zero, and disable unsupported features. If a client
requires a key field, it can use a placeholder: this local server does not check
an Authorization header. A server-side client can use these endpoints; arbitrary
browser origins are not enabled. The [built-in browser client](BROWSER_CLIENT.md)
is served at `/` on the same origin and has been tested with this subset.
Third-party chat interfaces have not been certified.

## Verification

```bash
python -m pytest -q
python scripts/verify_chat_api.py > benchmarks/chat-api-smoke.json
```

Normal adapter tests use gated fake executors and need no model downloads.
The optional real smoke requires the local checkpoint and development dependencies.
It compares both response formats for a multi-turn history with independent
reference greedy generation, checks usage and stream metadata, and verifies zero
remaining live caches. HTTPX buffers ASGI response bodies, so this smoke is not
an external streaming benchmark. Separate direct-ASGI tests prove incremental
content and disconnect cleanup; a live Uvicorn/curl check verified the new routes.
See [the recorded result](benchmarks/chat-api-smoke.json).
