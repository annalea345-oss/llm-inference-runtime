import asyncio

import pytest

from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.scheduler import QueueFullError, RuntimeUnavailableError


async def collect(request):
    events = []
    while True:
        event = await asyncio.wait_for(request.events.get(), 2)
        events.append(event)
        if event.kind in ("done", "error"):
            return events


class RecordingExecutor(SimulatedExecutor):
    def __init__(self, config):
        super().__init__(config)
        self.steps = []

    async def execute(self, step):
        self.steps.append(step)
        return await super().execute(step)


class RecordingScheduler(ContinuousBatchScheduler):
    def __init__(self, executor, config):
        super().__init__(executor, config)
        self.iterations = []

    def _plan(self):
        steps = super()._plan()
        self.iterations.append(steps)
        return steps


@pytest.mark.parametrize('budget', [1, 2, 5, 32])
async def test_budget_chunking_fairness_and_exact_output(budget):
    config = RuntimeConfig(token_budget=budget, max_active_requests=3,
                           prefill_delay=0, token_delay=0)
    executor = RecordingExecutor(config)
    scheduler = RecordingScheduler(executor, config)
    await scheduler.start()
    try:
        requests = [scheduler.submit(prompt, 3) for prompt in
                    ['one two three four five six seven', 'short', '   ']]
        streams = await asyncio.gather(*(collect(r) for r in requests))
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert all(r.phase == 'finished' for r in requests)
        for r, events in zip(requests, streams):
            assert [e.kind for e in events] == ['token'] * 3 + ['done']
            words = r.prompt.split() or ['<empty>']
            assert [e.data['text'] for e in events[:-1]] == [words[i % len(words)] + ' ' for i in range(3)]
            assert [e.data['index'] for e in events[:-1]] == [0, 1, 2]
            assert events[-1].data['generated_tokens'] == 3
            assert {e.data['request_id'] for e in events} == {r.request_id}
            steps = [s for s in executor.steps if s.request is r]
            prefill = [s for s in steps if s.phase == 'prefill']
            assert sum(s.token_count for s in prefill) == r.prompt_token_count
            assert [s.offset for s in prefill] == list(range(0, r.prompt_token_count, budget))
            assert [s.phase for s in steps] == ['prefill'] * len(prefill) + ['decode'] * 3
        for iteration in scheduler.iterations:
            assert sum(s.token_count for s in iteration) <= budget
            assert len({s.request.request_id for s in iteration}) == len(iteration)
            assert all(s.token_count > 0 for s in iteration)
            assert all(s.token_count == 1 for s in iteration if s.phase == 'decode')
        # With a one-token budget, the three resident requests rotate exactly.
        if budget == 1:
            assert [batch[0].request for batch in scheduler.iterations[:3]] == requests
        # A short request finishes while the long prompt still has work pending.
        finished_order = [s.request for batch in scheduler.iterations for s in batch
                          if s.phase == 'decode' and s.offset == 2]
        assert finished_order.index(requests[1]) < finished_order.index(requests[0]) or budget == 32
    finally:
        await scheduler.stop()


async def test_late_short_request_joins_live_batch_and_finishes_before_long():
    entered = asyncio.Event()
    release = asyncio.Event()
    later = asyncio.Event()
    finish = asyncio.Event()
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.request.prompt == 'long' and step.phase == 'decode':
                if step.offset == 0:
                    entered.set()
                    await release.wait()
                if step.offset == 3:
                    later.set()
                    await finish.wait()
            return await super().execute(step)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        long = scheduler.submit('long', 10)
        await asyncio.wait_for(entered.wait(), 2)
        short = scheduler.submit('short', 1)
        release.set()
        await asyncio.wait_for(later.wait(), 2)
        assert (await collect(short))[-1].kind == 'done'
        assert long.phase == 'decode'
        assert long.generated_tokens == 3
        finish.set()
        assert (await collect(long))[-1].data['generated_tokens'] == 10
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
async def test_abandonment_during_step_discards_result_and_releases_capacity(phase):
    entered = asyncio.Event()
    release = asyncio.Event()
    config = RuntimeConfig(max_active_requests=1, prefill_delay=0, token_delay=0)

    class Executor(RecordingExecutor):
        async def execute(self, step):
            if step.request.prompt == 'cancel' and step.phase == phase:
                entered.set()
                await release.wait()
            return await super().execute(step)

    executor = Executor(config)
    scheduler = ContinuousBatchScheduler(executor, config)
    await scheduler.start()
    try:
        request = scheduler.submit('cancel', 10)
        await asyncio.wait_for(entered.wait(), 2)
        request.abandoned = True
        skipped = scheduler.submit('skip', 1)
        skipped.abandoned = True
        good = scheduler.submit('good', 1)
        release.set()
        await collect(good)
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert request.phase == skipped.phase == 'cancelled'
        assert request.events.empty()
        assert skipped.events.empty()
        assert all(s.request is not skipped for s in executor.steps)
        assert not scheduler.active
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
async def test_failure_isolated_within_batch_and_subsequent_admission(phase):
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.request.prompt == 'fail' and step.phase == phase:
                raise RuntimeError('private details')
            return await super().execute(step)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        bad = scheduler.submit('fail', 2)
        good = scheduler.submit('good', 2)
        assert (await collect(bad))[-1].data['message'] == 'model execution failed'
        assert bad.phase == 'failed'
        assert (await collect(good))[-1].kind == 'done'
        assert (await collect(scheduler.submit('later', 1)))[-1].kind == 'done'
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert scheduler.running
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
async def test_bounded_active_batch_overload_shutdown_and_restart(phase):
    entered = asyncio.Queue()
    release = asyncio.Event()
    config = RuntimeConfig(max_active_requests=2, queue_capacity=2)

    class Executor:
        async def execute(self, step):
            if step.phase != phase:
                return None
            entered.put_nowait(step)
            await release.wait()
            return None

    scheduler = ContinuousBatchScheduler(Executor(), config)
    with pytest.raises(RuntimeUnavailableError):
        scheduler.submit('offline', 1)
    await scheduler.start()
    await scheduler.start()
    requests = [scheduler.submit('first', 1), scheduler.submit('second', 1)]
    await asyncio.wait_for(entered.get(), 2)
    await asyncio.wait_for(entered.get(), 2)  # Both selected steps are in flight.
    assert len(scheduler.active) == 2
    requests += [scheduler.submit('third', 1), scheduler.submit('fourth', 1)]
    with pytest.raises(QueueFullError):
        scheduler.submit('overflow', 1)
    await scheduler.stop()
    for request in requests:
        assert (await collect(request))[-1].data['message'] == 'runtime is shutting down'
        assert request.events.empty()  # Exactly one terminal event.
    await asyncio.wait_for(scheduler.queue.join(), 2)
    assert not scheduler.active
    assert not scheduler.running
    await scheduler.stop()
    # Stopping before the worker's first turn also drains pending requests.
    await scheduler.start()
    request = scheduler.submit('immediate shutdown', 1)
    await scheduler.stop()
    assert (await collect(request))[-1].kind == 'error'
    await asyncio.wait_for(scheduler.queue.join(), 2)
    scheduler.executor = SimulatedExecutor(RuntimeConfig(prefill_delay=0, token_delay=0))
    await scheduler.start()
    try:
        assert (await collect(scheduler.submit('restart', 1)))[-1].kind == 'done'
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('name', ['queue_capacity', 'max_active_requests', 'token_budget'])
@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_scheduling_configuration(name, value):
    with pytest.raises(ValueError, match='positive integer'):
        RuntimeConfig(**{name: value})


async def test_http_disconnect_cancels_active_stream_and_health_reports_batch():
    import json
    import httpx
    from llm_runtime.api import create_app

    release = asyncio.Event()
    blocked = asyncio.Event()
    disconnect = asyncio.Event()
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.phase == 'decode' and step.offset == 1:
                blocked.set()
                await release.wait()
            return await super().execute(step)

    app = create_app(config, Executor(config))
    sent = False
    chunks = []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': b'{"prompt":"cancel","max_tokens":10}'}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        if message['type'] == 'http.response.body' and message.get('body'):
            chunks.append(message['body'])

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': '/generate', 'raw_path': b'/generate', 'query_string': b'',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send))
        try:
            await asyncio.wait_for(blocked.wait(), 2)
            scheduler = app.state.scheduler
            cancelled = scheduler.active[0]
            other = scheduler.submit('other', 1)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                health = (await client.get('/health')).json()
                assert health['active_requests'] == 1
                assert health['active_request_ids'] == [cancelled.request_id]
                assert health['queued_requests'] == 1
            disconnect.set()
            await asyncio.wait_for(task, 2)
            assert cancelled.abandoned
            release.set()
            assert (await collect(other))[-1].kind == 'done'
            await asyncio.wait_for(scheduler.queue.join(), 2)
            assert cancelled.phase == 'cancelled'
            assert cancelled.generated_tokens == 1
            assert cancelled.events.empty()
            frames = b''.join(chunks).decode()
            assert frames.count('event: token') == 1
            payload = json.loads(frames.splitlines()[1][6:])
            assert payload['text'] == 'cancel '
            assert 'event: done' not in frames
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
