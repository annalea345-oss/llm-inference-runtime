"""HTTP compatibility contract for the deliberately narrow chat adapter."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.protocol import DecodeResult


CONFIG = RuntimeConfig(prefill_delay=0, token_delay=0)
MESSAGES = [{'role': 'user', 'content': 'hello world'}]


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            yield client


async def body_for(client, **overrides):
    model = (await client.get('/v1/models')).json()['data'][0]['id']
    return {'model': model, 'messages': MESSAGES, **overrides}


def assert_error(response, status):
    assert response.status_code == status
    payload = response.json()
    assert set(payload) == {'error'}
    assert isinstance(payload['error']['message'], str)
    assert payload['error']['type']
    assert 'code' in payload['error']
    return payload['error']


def stream_payloads(response):
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    lines = [line for line in response.text.splitlines() if line]
    assert all(line.startswith('data: ') for line in lines)
    assert lines[-1] == 'data: [DONE]'
    assert lines.count('data: [DONE]') == 1
    return [json.loads(line[6:]) for line in lines[:-1]]


async def test_model_discovery_and_default_completion():
    app = create_app(CONFIG)
    async with client_for(app) as client:
        response = await client.get('/v1/models')
        assert response.status_code == 200
        models = response.json()
        assert models['object'] == 'list'
        assert len(models['data']) == 1
        model = models['data'][0]
        assert model['id'] == 'simulated'
        assert model['object'] == 'model'
        assert model['owned_by'] == 'llm-runtime'
        assert type(model['created']) is int
        response = await client.post('/v1/chat/completions', json=await body_for(client))
        assert response.status_code == 200
        result = response.json()
        assert result['id'] == 'chatcmpl-' + response.headers['x-request-id']
        assert result['object'] == 'chat.completion'
        assert result['model'] == 'simulated'
        assert type(result['created']) is int
        assert len(result['choices']) == 1
        choice = result['choices'][0]
        assert choice['index'] == 0
        assert choice['message']['role'] == 'assistant'
        assert isinstance(choice['message']['content'], str)
        assert choice['message']['content']
        assert choice['finish_reason'] == 'length'
        assert result['usage']['completion_tokens'] == 16
        assert result['usage']['prompt_tokens'] > 0
        assert result['usage']['total_tokens'] == (result['usage']['prompt_tokens'] + 16)
        assert not app.state.scheduler._requests


@pytest.mark.parametrize('alias', ['simulated', 'local-model', None])
async def test_model_alias_is_public_and_does_not_expose_model_paths(alias):
    class Executor:
        @property
        def info(self):
            return {'backend': 'custom', 'model': '/private/path/to/weights'}

        async def execute(self, step):
            return await SimulatedExecutor(CONFIG).execute(step)

    executor = Executor()
    if alias is not None:
        executor.model_id = alias
    async with client_for(create_app(CONFIG, executor)) as client:
        response = await client.get('/v1/models')
        assert response.json()['data'][0]['id'] == (alias or 'runtime')
        assert '/private/path' not in response.text
        body = await body_for(client, max_tokens=1, temperature=0.0)
        completed = await client.post('/v1/chat/completions', json=body)
        assert completed.status_code == 200
        assert completed.json()['model'] == (alias or 'runtime')


@pytest.mark.parametrize('override', [
    {'messages': []}, {'messages': MESSAGES * 65},
    {'messages': [{'role': 'user', 'content': ''}]},
    {'messages': [{'role': 'user', 'content': 'x' * 8193}]},
    {'messages': [{'role': 'system', 'content': 'a'}]},
    {'messages': [{'role': 'assistant', 'content': 'a'}]},
    {'messages': [{'role': 'tool', 'content': 'a'}]},
    {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'}]}]},
    {'messages': [{'role': 'user', 'content': 5}]},
    {'messages': [{'role': 'user', 'content': 'a', 'name': 'extra'}]},
    {'messages': [{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}]},
    {'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}]},
    {'messages': [{'role': 'user', 'content': 'a'}, {'role': 'system', 'content': 'b'}]},
    {'messages': [{'role': 'system', 'content': 'a'}, {'role': 'system', 'content': 'b'},
                  {'role': 'user', 'content': 'c'}]},
    {'messages': [{'role': 'system', 'content': 'x' * 4097},
                  {'role': 'user', 'content': 'y' * 4096}]},
    {'max_tokens': True}, {'max_tokens': '2'}, {'max_tokens': 0}, {'max_tokens': 513},
    {'max_completion_tokens': False}, {'max_completion_tokens': 1.5},
    {'max_completion_tokens': 513}, {'max_tokens': 1, 'max_completion_tokens': 1},
    {'stream': 'true'}, {'stream': 1}, {'n': 2}, {'n': True}, {'n': '1'},
    {'temperature': 0.1}, {'temperature': True}, {'temperature': '0'},
    {'stream_options': {'include_usage': True}},
    {'stream': True, 'stream_options': {'include_usage': 1}},
    {'stream': True, 'stream_options': {'other': True}},
    {'tools': []}, {'top_p': 1}, {'response_format': {'type': 'json_object'}},
])
async def test_unsupported_chat_requests_are_openai_errors_without_submission(override):
    app = create_app(CONFIG)
    async with client_for(app) as client:
        body = await body_for(client, **override)
        assert_error(await client.post('/v1/chat/completions', json=body), 400)
        assert not app.state.scheduler._requests
        assert app.state.scheduler.queued_requests == 0
        assert not app.state.scheduler.kv_cache.allocations


async def test_unknown_model_malformed_json_and_original_validation():
    async with client_for(create_app(CONFIG)) as client:
        assert_error(await client.post('/v1/chat/completions',
                                       json=await body_for(client, model='missing')), 404)
        assert_error(await client.post('/v1/chat/completions', content='{',
                                       headers={'content-type': 'application/json'}), 400)
        assert_error(await client.post('/v1/chat/completions', json={}), 400)
        response = await client.post('/generate', json={'prompt': '', 'max_tokens': 0})
        assert response.status_code == 422
        assert 'detail' in response.json()


class TokenizedChatExecutor(SimulatedExecutor):
    def __init__(self):
        super().__init__(CONFIG)
        self.histories = []
        self.seen_requests = []

    def tokenize_messages(self, messages):
        self.histories.append(messages)
        return (101, 102, 103, 104, 105)

    async def execute(self, step):
        self.seen_requests.append(step.request)
        if step.phase == 'prefill':
            return None
        if step.offset == 0:
            return DecodeResult('answer', 201)
        return DecodeResult('', 202, 'eos')


HISTORY = [{'role': 'system', 'content': 'Be brief.'},
           {'role': 'user', 'content': 'first question'},
           {'role': 'assistant', 'content': 'first answer'},
           {'role': 'user', 'content': 'follow up'}]


@pytest.mark.parametrize('limit_field', ['max_tokens', 'max_completion_tokens'])
async def test_chat_template_receives_full_history_once_and_usage_counts_eos(limit_field):
    executor = TokenizedChatExecutor()
    app = create_app(CONFIG, executor)
    async with client_for(app) as client:
        body = await body_for(client, messages=HISTORY, **{limit_field: 4},
                              temperature=0, n=1)
        response = await client.post('/v1/chat/completions', json=body)
        assert response.status_code == 200
        result = response.json()
        assert executor.histories == [HISTORY]
        assert all(request.prompt_token_ids == (101, 102, 103, 104, 105)
                   for request in executor.seen_requests)
        assert result['choices'] == [{'index': 0, 'message': {
            'role': 'assistant', 'content': 'answer'}, 'finish_reason': 'stop'}]
        assert result['usage'] == {'prompt_tokens': 5, 'completion_tokens': 2,
                                   'total_tokens': 7}
        assert not app.state.scheduler.kv_cache.allocations


@pytest.mark.parametrize('include_usage', [False, True])
async def test_stream_chunks_have_stable_identity_role_text_finish_and_optional_usage(include_usage):
    executor = TokenizedChatExecutor()
    async with client_for(create_app(CONFIG, executor)) as client:
        body = await body_for(client, messages=HISTORY, stream=True, max_tokens=4,
                              stream_options={'include_usage': include_usage})
        response = await client.post('/v1/chat/completions', json=body)
        chunks = stream_payloads(response)
        assert len({chunk['id'] for chunk in chunks}) == 1
        assert {chunk['id'] for chunk in chunks} == {'chatcmpl-' + response.headers['x-request-id']}
        assert len({chunk['created'] for chunk in chunks}) == 1
        assert all(type(chunk['created']) is int for chunk in chunks)
        assert {chunk['model'] for chunk in chunks} == {body['model']}
        assert {chunk['object'] for chunk in chunks} == {'chat.completion.chunk'}
        ordinary = [chunk for chunk in chunks if chunk['choices']]
        assert ordinary[0]['choices'][0]['delta']['role'] == 'assistant'
        assert ''.join(chunk['choices'][0]['delta'].get('content', '')
                       for chunk in ordinary) == 'answer'
        assert all(chunk['choices'][0]['index'] == 0 for chunk in ordinary)
        assert all(chunk['choices'][0]['finish_reason'] is None for chunk in ordinary[:-1])
        assert ordinary[-1]['choices'] == [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]
        usages = [chunk for chunk in chunks if not chunk['choices']]
        assert len(usages) == int(include_usage)
        if include_usage:
            assert chunks[-1] is usages[0]
            assert usages[0]['usage'] == {'prompt_tokens': 5, 'completion_tokens': 2,
                                         'total_tokens': 7}


@pytest.mark.parametrize('stream', [False, True])
async def test_execution_failure_maps_to_json_or_stream_error_without_private_details(stream):
    class Executor(SimulatedExecutor):
        async def execute(self, step):
            raise RuntimeError('private model path and credentials')

    app = create_app(CONFIG, Executor(CONFIG))
    async with client_for(app) as client:
        response = await client.post('/v1/chat/completions',
                                     json=await body_for(client, stream=stream))
        if stream:
            chunks = stream_payloads(response)
            errors = [chunk['error'] for chunk in chunks if 'error' in chunk]
            assert len(errors) == 1
            assert errors[0]['message']
        else:
            assert_error(response, 500)
        assert 'private model' not in response.text
        assert not app.state.scheduler._requests
        assert not app.state.scheduler.kv_cache.allocations


@pytest.mark.parametrize('stream', [False, True])
async def test_shutdown_of_inflight_chat_returns_json_or_stream_error(stream):
    entered = asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await asyncio.Event().wait()

    app = create_app(CONFIG, Executor(CONFIG))
    async with client_for(app) as client:
        pending = asyncio.create_task(client.post('/v1/chat/completions',
                                                  json=await body_for(client, stream=stream)))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            await asyncio.wait_for(app.state.scheduler.stop(), 2)
            response = await asyncio.wait_for(pending, 2)
            if stream:
                errors = [chunk['error'] for chunk in stream_payloads(response) if 'error' in chunk]
                assert len(errors) == 1
                assert errors[0]['code'] == 'runtime_unavailable'
            else:
                assert assert_error(response, 503)['code'] == 'runtime_unavailable'
            assert not app.state.scheduler._requests
            assert not app.state.scheduler.kv_cache.allocations
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_capacity_and_unavailable_errors_preserve_envelope():
    config = RuntimeConfig(kv_cache_capacity_tokens=2, prefill_delay=0, token_delay=0)
    app = create_app(config)
    async with client_for(app) as client:
        assert_error(await client.post('/v1/chat/completions',
                                       json=await body_for(client, max_tokens=3)), 422)
        await app.state.scheduler.stop()
        assert_error(await client.post('/v1/chat/completions',
                                       json=await body_for(client, max_tokens=1)), 503)


@pytest.mark.parametrize('stream', [False, True])
async def test_explicit_cancellation_uses_shared_request_registry(stream):
    entered, release = asyncio.Event(), asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            return await super().execute(step)

    app = create_app(CONFIG, Executor(CONFIG))
    async with client_for(app) as client:
        body = await body_for(client, stream=stream, max_tokens=3)
        pending = asyncio.create_task(client.post('/v1/chat/completions', json=body))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            request = app.state.scheduler.active[0]
            cancelled = await client.post(f'/requests/{request.request_id}/cancel')
            assert cancelled.status_code == 202
            assert request.request_id in app.state.scheduler.kv_cache.allocations
            release.set()
            response = await asyncio.wait_for(pending, 2)
            if stream:
                assert response.headers['x-request-id'] == request.request_id
                errors = [chunk['error'] for chunk in stream_payloads(response) if 'error' in chunk]
                assert len(errors) == 1
                assert errors[0]['code'] == 'request_cancelled'
            else:
                assert assert_error(response, 409)['code'] == 'request_cancelled'
            await asyncio.wait_for(app.state.scheduler.queue.join(), 2)
            assert request.phase == 'cancelled'
            assert request.generated_tokens == 0
            assert not app.state.scheduler._requests
            assert not app.state.scheduler.kv_cache.allocations
        finally:
            release.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_nonstream_disconnect_abandons_work_and_releases_after_execution():
    entered, release, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            return await super().execute(step)

    app = create_app(CONFIG, Executor(CONFIG))
    sent = False
    messages = []
    body = json.dumps({'model': 'simulated', 'messages': MESSAGES, 'max_tokens': 3}).encode()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': body}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)

    async with app.router.lifespan_context(app):
        # Executor inherits simulated metadata, hence the simulated alias.
        task = asyncio.create_task(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
            'query_string': b'', 'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            scheduler = app.state.scheduler
            request = scheduler.active[0]
            disconnect.set()
            await asyncio.wait_for(task, 2)
            assert request.abandoned
            assert request.request_id in scheduler.kv_cache.allocations
            release.set()
            await asyncio.wait_for(scheduler.queue.join(), 2)
            assert request.phase == 'cancelled'
            assert request.events.empty()
            assert not scheduler._requests
            assert not scheduler.kv_cache.allocations
            # The client has gone away, but the handler can still construct a
            # diagnostic response for the disconnected ASGI transport.
            start = next(message for message in messages if message['type'] == 'http.response.start')
            assert start['status'] == 499
            diagnostic = json.loads(b''.join(message.get('body', b'') for message in messages))
            assert diagnostic['error']['code'] == 'client_disconnected'
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_chat_queue_overload_returns_503_and_no_additional_request():
    entered, release = asyncio.Event(), asyncio.Event()
    config = RuntimeConfig(max_active_requests=1, queue_capacity=1,
                           prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            return await super().execute(step)

    app = create_app(config, Executor(config))
    async with client_for(app) as client:
        scheduler = app.state.scheduler
        owner = scheduler.submit('owner', 1)
        try:
            await asyncio.wait_for(entered.wait(), 2)
            queued = scheduler.submit('queued', 1)
            response = await client.post('/v1/chat/completions',
                                         json=await body_for(client, max_tokens=1))
            assert_error(response, 503)
            assert set(scheduler._requests) == {owner.request_id, queued.request_id}
        finally:
            release.set()


@pytest.mark.parametrize('transport', ['body-disconnect', 'early-disconnect', 'header-send-error'])
async def test_stream_disconnect_and_header_failure_abandon_chat_request(transport):
    from starlette.requests import ClientDisconnect

    entered, release = asyncio.Event(), asyncio.Event()
    headers_ready, disconnect = asyncio.Event(), asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if transport != 'body-disconnect' or (step.phase == 'decode' and step.offset == 1):
                entered.set()
                await release.wait()
            return await super().execute(step)

    app = create_app(CONFIG, Executor(CONFIG))
    sent, chunks = False, []
    body = json.dumps({'model': 'simulated', 'messages': MESSAGES,
                       'max_tokens': 3, 'stream': True}).encode()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': body}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        if message['type'] == 'http.response.start':
            headers_ready.set()
            if transport == 'header-send-error':
                await disconnect.wait()
                raise OSError('connection closed while sending chat headers')
            if transport == 'early-disconnect':
                await asyncio.Event().wait()
        elif message['type'] == 'http.response.body':
            chunks.append(message.get('body', b''))

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version':
                                   '2.4' if transport == 'header-send-error' else '2.0'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
            'query_string': b'', 'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send))
        try:
            await asyncio.wait_for(headers_ready.wait(), 2)
            await asyncio.wait_for(entered.wait(), 2)
            scheduler = app.state.scheduler
            request = scheduler.active[0]
            assert request.generated_tokens == int(transport == 'body-disconnect')
            disconnect.set()
            if transport == 'header-send-error':
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(task, 2)
            else:
                await asyncio.wait_for(task, 2)
            assert request.abandoned
            assert request.request_id in scheduler.kv_cache.allocations
            release.set()
            await asyncio.wait_for(scheduler.queue.join(), 2)
            assert request.phase == 'cancelled'
            assert request.events.empty()
            assert not scheduler._requests
            assert not scheduler.kv_cache.allocations
            if transport == 'body-disconnect':
                frames = b''.join(chunks).decode()
                assert 'data: [DONE]' not in frames
                payloads = [json.loads(line[6:]) for line in frames.splitlines() if line]
                assert payloads[0]['choices'][0]['delta']['role'] == 'assistant'
                assert sum(bool(p['choices'][0]['delta'].get('content')) for p in payloads) == 1
            else:
                assert not chunks
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancel_nonstream_http_task_cleans_up_collection_and_disconnect_helpers():
    entered, release = asyncio.Event(), asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            return await super().execute(step)

    app = create_app(CONFIG, Executor(CONFIG))
    async with client_for(app) as client:
        body = await body_for(client, max_tokens=3)
        pending = asyncio.create_task(client.post('/v1/chat/completions', json=body))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            request = app.state.scheduler.active[0]
            helpers = {task for task in asyncio.all_tasks()
                       if getattr(task.get_coro(), '__qualname__', '') in {
                           '_complete.<locals>.consume', '_complete.<locals>.disconnected'}}
            assert len(helpers) == 2
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert request.abandoned
            assert all(task.done() for task in helpers)
            assert request.request_id in app.state.scheduler.kv_cache.allocations
            release.set()
            await asyncio.wait_for(app.state.scheduler.queue.join(), 2)
            assert request.phase == 'cancelled'
            assert request.events.empty()
            assert not app.state.scheduler._requests
            assert not app.state.scheduler.kv_cache.allocations
        finally:
            release.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def test_tokenization_failure_is_safe_error_before_admission():
    class Executor(SimulatedExecutor):
        def tokenize_messages(self, messages):
            raise RuntimeError('private tokenizer detail')

    app = create_app(CONFIG, Executor(CONFIG))
    async with client_for(app) as client:
        response = await client.post('/v1/chat/completions', json=await body_for(client))
        error = assert_error(response, 500)
        assert error['code'] == 'preparation_failed'
        assert 'private' not in response.text
        assert app.state.scheduler.queued_requests == 0
        assert not app.state.scheduler._requests
