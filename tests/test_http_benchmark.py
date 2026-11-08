"""Client receipt measurements against deterministic streaming HTTP transports."""

import asyncio
import builtins
import json

import httpx
import pytest

from llm_runtime import http_benchmark as bench


def metadata():
    return {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk', 'created': 123, 'model': 'model'}


def frame(delta=None, reason=None):
    return {**metadata(), 'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': reason}]}


def encode(value):
    return 'data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\n\n'


USAGE = {**metadata(), 'choices': [], 'usage': {
    'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}}
VALID = ''.join(map(encode, [frame({'role': 'assistant'}), frame({'content': 'a'}),
                           frame({'content': 'b'}), frame(reason='stop'), USAGE, '[DONE]']))


class Clock:
    value = 0

    def __call__(self):
        return self.value


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks, clock=None):
        self.chunks = chunks
        self.clock = clock

    async def __aiter__(self):
        for time, data in self.chunks:
            if self.clock:
                self.clock.value = time
            await asyncio.sleep(0)
            yield data


def response_stream(text):
    raw = text.encode()
    return Chunks([(0, raw[i:i + 7]) for i in range(0, len(raw), 7)])


async def get_sample(text=VALID, *, status=200, content_type='text/event-stream', timeout=1):
    async def handler(request):
        return httpx.Response(status, headers={'content-type': content_type}, stream=response_stream(text))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await bench.sample(client, 'http://localhost:8000', 'model', index=0,
                                  prompt='hi', max_tokens=4, timeout=timeout)


async def test_timings_start_at_nonempty_content_and_end_at_done():
    clock = Clock()
    chunks = [(0.2, encode(frame({'role': 'assistant'})).encode()),
              (0.3, encode(frame({'content': ''})).encode()),
              (0.5, encode(frame({'content': 'a'})).encode()),
              (0.8, encode(frame({'content': 'b'})).encode()),
              (0.9, encode(frame(reason='length')).encode()),
              (1.0, encode(USAGE).encode()), (1.1, encode('[DONE]').encode())]

    async def handler(request):
        assert request.url.path == '/v1/chat/completions'
        body = json.loads(request.content)
        assert body['stream_options'] == {'include_usage': True}
        assert body['temperature'] == 0 and body['stream'] is True
        clock.value = 0.1
        return httpx.Response(200, headers={'content-type': 'text/event-stream; charset=utf-8'},
                              stream=Chunks(chunks, clock))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await bench.sample(client, 'http://localhost', 'model', index=3,
                                 prompt='hi', max_tokens=4, timeout=2, clock=clock)
    assert row['status'] == 'completed'
    assert row['content_deltas'] == 2
    assert row['timings_ms']['headers'] == pytest.approx(100)
    assert row['timings_ms']['first_content'] == pytest.approx(500)
    assert row['timings_ms']['completion'] == pytest.approx(1100)
    assert row['timings_ms']['content_delta_gaps'] == pytest.approx([300])
    assert row['usage'] == USAGE['usage']


async def test_sse_crlf_comments_multiline_and_transport_fragmentation():
    encoded = json.dumps(frame({'content': 'hello'}))
    first, second = encoded.split('"choices":', 1)
    text = ': heartbeat\r\n\r\nevent: message\r\ndata: ' + first + '"choices":\r\ndata: ' + second + '\r\n\r\n'
    text += ''.join(map(encode, [frame(reason='stop'), USAGE, '[DONE]'])).replace('\n', '\r\n')
    row = await get_sample(text)
    assert row['status'] == 'completed'
    assert row['content_deltas'] == 1


@pytest.mark.parametrize('text', [
    'data: {bad json}\n\n',
    encode({'error': {'message': 'failed'}}),
    'event: error\ndata: failure\n\n',
    encode({'choices': 'invalid'}),
    encode(frame(reason='unknown')),
    encode('[DONE]'),
    encode(frame(reason='stop')) + encode('[DONE]'),
    encode(USAGE) + encode('[DONE]'),
    VALID.removesuffix(encode('[DONE]')),
    VALID.removesuffix('\n\n'),  # Undispatched final SSE frame.
    encode(frame(reason='stop')) + encode({**metadata(), 'choices': [], 'usage': {
        'prompt_tokens': 1, 'completion_tokens': True, 'total_tokens': 2}}) + encode('[DONE]'),
    encode(frame(reason='stop')) + encode({**metadata(), 'choices': [], 'usage': {
        'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 9}}) + encode('[DONE]'),
    encode(frame(reason='stop')) + encode(frame({'content': 'late'})) + encode(USAGE) + encode('[DONE]'),
])
async def test_malformed_error_and_truncated_streams_never_succeed(text):
    row = await get_sample(text)
    assert row['status'] == 'error'
    assert row['usage'] is None
    assert row['timings_ms']['completion'] is None


@pytest.mark.parametrize('field,value', [
    ('id', ''), ('id', 42), ('id', 'different-request'),
    ('created', True), ('created', '123'), ('created', 456),
    ('object', 'chat.completion'), ('model', 'another-model'),
])
@pytest.mark.parametrize('position', ['content', 'usage'])
async def test_invalid_or_mixed_chunk_metadata_is_rejected(field, value, position):
    content = frame({'content': 'answer'})
    usage = dict(USAGE)
    changed = content if position == 'content' else usage
    changed[field] = value
    text = ''.join(map(encode, [frame({'role': 'assistant'}), content,
                               frame(reason='stop'), usage, '[DONE]']))
    assert (await get_sample(text))['status'] == 'error'


@pytest.mark.parametrize('field', ['id', 'object', 'created', 'model'])
@pytest.mark.parametrize('position', ['content', 'usage'])
async def test_missing_chunk_metadata_is_rejected(field, position):
    content, usage = frame({'content': 'answer'}), dict(USAGE)
    del (content if position == 'content' else usage)[field]
    text = ''.join(map(encode, [content, frame(reason='stop'), usage, '[DONE]']))
    assert (await get_sample(text))['status'] == 'error'


@pytest.mark.parametrize('choice_update,delta', [
    ({'index': False}, {}), ({'index': 0.0}, {}),
    ({}, {'role': 'user'}), ({}, {'role': None}),
    ({}, {'content': 5}), ({}, {'content': []}),
])
async def test_invalid_choice_index_role_or_content_is_rejected(choice_update, delta):
    invalid = frame(delta)
    invalid['choices'][0].update(choice_update)
    text = ''.join(map(encode, [invalid, frame(reason='stop'), USAGE, '[DONE]']))
    assert (await get_sample(text))['status'] == 'error'


@pytest.mark.parametrize('status,expected', [(400, 'rejected'), (422, 'rejected'),
                                            (503, 'rejected'), (500, 'error'), (502, 'error')])
async def test_nonstream_http_failures(status, expected):
    row = await get_sample('not an SSE body', status=status, content_type='application/json')
    assert row['status'] == expected
    assert row['http_status'] == status
    assert row['usage'] is None


async def test_wrong_content_type_is_error():
    assert (await get_sample(VALID, content_type='application/json'))['status'] == 'error'


async def test_total_deadline_includes_stream_body():
    release = asyncio.Event()

    class Blocked(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield encode(frame({'role': 'assistant'})).encode()
            await release.wait()
    async def handler(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Blocked())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await bench.sample(client, 'http://localhost', 'model', index=0,
                                 prompt='hi', max_tokens=4, timeout=0.01)
    assert row['status'] == 'timeout'
    assert row['timings_ms']['first_content'] is None


async def test_closed_loop_workers_bound_live_streams():
    gate, full = asyncio.Event(), asyncio.Event()
    active, peak, calls = 0, 0, 0

    class Stream(httpx.AsyncByteStream):
        def __init__(self):
            self.opened = False

        async def __aiter__(self):
            nonlocal active, peak
            active += 1
            self.opened = True
            peak = max(peak, active)
            if active == 2:
                full.set()
            await gate.wait()
            yield VALID.encode()

        async def aclose(self):
            nonlocal active
            if self.opened:
                active -= 1
                self.opened = False

    async def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Stream())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(bench.workload(client, 'http://localhost', 'model', requests=5,
                                                  concurrency=2, prompt='hi', max_tokens=4, timeout=2))
        try:
            await asyncio.wait_for(full.wait(), 1)
            assert calls == active == peak == 2
        finally:
            gate.set()
        rows = await asyncio.wait_for(task, 2)
    assert peak == 2
    assert [r['index'] for r in rows] == list(range(5))
    assert all(r['status'] == 'completed' for r in rows)


def completed_row(tokens=2):
    return {'status': 'completed', 'usage': {'completion_tokens': tokens},
            'timings_ms': {'headers': 1, 'first_content': 5, 'completion': 10,
                           'content_delta_gaps': [2, 3]}}


async def test_metadata_warmup_exclusion_and_throughput_arithmetic(monkeypatch):
    clock, calls = Clock(), []

    async def workload(client, base_url, model, **options):
        calls.append(options['requests'])
        assert model == 'test-model'
        clock.value += 100 if len(calls) == 1 else 2
        return [completed_row(6) for _ in range(options['requests'])]
    monkeypatch.setattr(bench, 'workload', workload)

    def handler(request):
        return httpx.Response(200, json={'data': [{'id': 'test-model'}]} if request.url.path == '/v1/models'
                              else {'config': {'token_budget': 8}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await bench.benchmark(requests=4, warmup=2, client=client, clock=clock)
    assert calls == [2, 4]
    assert report['elapsed_seconds'] == 2
    assert report['completed_requests_per_second'] == 2
    assert report['completed_tokens_per_second'] == 12
    assert len(report['samples']) == 4
    assert report['health'] == {'config': {'token_budget': 8}}
    assert report['model']['id'] == 'test-model'


async def test_warmup_failure_aborts_measurement(monkeypatch):
    calls = []

    async def workload(*args, **kwargs):
        calls.append(kwargs['requests'])
        return [{'status': 'rejected'}]
    monkeypatch.setattr(bench, 'workload', workload)

    def handler(request):
        return httpx.Response(200, json={'data': [{'id': 'test-model'}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match='warmup failed'):
            await bench.benchmark(client=client)
    assert calls == [2]


def test_failed_samples_do_not_contribute_tokens_or_latencies():
    rows = [completed_row(4), {'status': 'error', 'usage': {'completion_tokens': 100}},
            {'status': 'timeout'}, {'status': 'rejected'}]
    summary = bench.summarize(rows, 2)
    assert summary['completed_tokens_per_second'] == 2
    assert summary['completed_requests_per_second'] == 0.5
    assert summary['errors'] == summary['timeouts'] == summary['rejected'] == 1
    assert summary['latency_ms']['first_content']['count'] == 1


@pytest.mark.parametrize('options', [
    {'base_url': 'http://example.com'}, {'base_url': 'http://localhost@evil.com'},
    {'base_url': 'http://localhost/foo'}, {'base_url': 'http://127.0.0.1:99999'},
    {'requests': 0}, {'concurrency': 0}, {'warmup': -1}, {'max_tokens': 513},
    {'timeout': float('nan')}, {'timeout': float('inf')}, {'timeout': 0}, {'prompt': ''},
])
async def test_invalid_options_rejected_before_network(options):
    with pytest.raises(ValueError):
        await bench.benchmark(**options)


def test_cli_arguments_and_exit_status(monkeypatch, capsys):
    seen = []

    async def benchmark(**options):
        seen.append(options)
        return {'completed': 7}
    monkeypatch.setattr(bench, 'benchmark', benchmark)
    assert bench.main(['--requests', '8', '--concurrency', '2', '--warmup', '0', '--timeout', '4']) == 1
    assert seen[0]['warmup'] == 0
    assert json.loads(capsys.readouterr().out) == {'completed': 7}
    assert bench.main(['--requests', '7']) == 0


async def test_missing_httpx_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'httpx':
            raise ImportError('missing')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(RuntimeError, match=r"pip install '.\[dev\]'"):
        await bench.benchmark()


async def test_http_ending_is_consumed_for_connection_reuse_but_done_time_is_preserved():
    clock = Clock()

    class Ending(httpx.AsyncByteStream):
        exhausted = False

        async def __aiter__(self):
            clock.value = 1
            yield VALID.encode()
            clock.value = 2
            yield b': final heartbeat\n\n'
            self.exhausted = True

    stream = Ending()
    async def handler(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await bench.sample(client, 'http://localhost', 'model', index=0,
                                 prompt='hi', max_tokens=4, timeout=2, clock=clock)
    assert stream.exhausted
    assert row['status'] == 'completed'
    assert row['timings_ms']['completion'] == 1000
    assert row['elapsed_ms'] == 2000


@pytest.mark.parametrize('trailing', [encode('[DONE]'), encode(frame({'content': 'late'}))])
async def test_sse_data_after_done_is_not_successful(trailing):
    row = await get_sample(VALID + trailing)
    assert row['status'] == 'error'
    assert row['usage'] is None
    assert row['timings_ms']['completion'] is None


async def test_deadline_also_bounds_http_ending_after_done():
    class Unfinished(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield VALID.encode()
            await asyncio.Event().wait()

    async def handler(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Unfinished())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        row = await bench.sample(client, 'http://localhost', 'model', index=0,
                                 prompt='hi', max_tokens=4, timeout=0.01)
    assert row['status'] == 'timeout'
    assert row['usage'] is None
