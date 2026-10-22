"""Explicit cancellation preserves iteration ownership and terminal accounting."""

import asyncio
import json

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.scheduler import QueueFullError, RuntimeUnavailableError


async def collect(request):
    events = []
    while True:
        event = await asyncio.wait_for(request.events.get(), 2)
        events.append(event)
        if event.kind in ('done', 'error'):
            return events


def assert_cancelled(request, events, generated=0):
    assert request.phase == 'cancelled'
    assert [event.kind for event in events] == ['token'] * generated + ['done']
    terminal = events[-1].data
    assert terminal['request_id'] == request.request_id
    assert terminal['finish_reason'] == 'cancelled'
    assert terminal['generated_tokens'] == generated
    assert terminal['timings']['end_to_end_ms'] >= 0
    assert request.generated_tokens == generated
    assert request.events.empty()


@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
@pytest.mark.parametrize('phase,offset', [('prefill', 0), ('decode', 0), ('decode', 2)])
@pytest.mark.parametrize('failure', [False, True])
async def test_active_cancel_discards_inflight_result_and_releases_at_barrier(
    policy, phase, offset, failure,
):
    entered, release = asyncio.Event(), asyncio.Event()
    config = RuntimeConfig(scheduling_policy=policy, max_active_requests=1,
                           prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.request.prompt == 'cancel' and step.phase == phase and step.offset == offset:
                entered.set()
                await release.wait()
                if failure:
                    raise RuntimeError('discard this cancelled execution failure')
            return await super().execute(step)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        # offset=2 also exercises cancellation racing the final decode result.
        request = scheduler.submit('cancel', 3)
        await asyncio.wait_for(entered.wait(), 2)
        before = (request.prefilled_tokens, request.generated_tokens,
                  scheduler.kv_cache.used_tokens, scheduler.kv_cache.reserved_tokens)
        assert scheduler.cancel(request.request_id)
        assert scheduler.cancel(request.request_id)
        assert request.cancel_requested
        assert not request.abandoned
        assert request.phase == phase
        assert (request.prefilled_tokens, request.generated_tokens,
                scheduler.kv_cache.used_tokens, scheduler.kv_cache.reserved_tokens) == before
        later = scheduler.submit('later', 1)
        release.set()
        assert_cancelled(request, await collect(request), offset if phase == 'decode' else 0)
        assert (await collect(later))[-1].data['finish_reason'] == 'length'
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert not scheduler.kv_cache.allocations
        assert not scheduler._requests
        assert not scheduler.cancel(request.request_id)
        assert not scheduler.cancel('unknown')
    finally:
        release.set()
        await scheduler.stop()


@pytest.mark.parametrize('pending', [False, True])
async def test_cancel_waiting_reclaims_slot_immediately_and_preserves_fifo(pending):
    entered, release = asyncio.Event(), asyncio.Event()
    steps = []
    config = RuntimeConfig(max_active_requests=2 if pending else 1, queue_capacity=3,
                           kv_cache_capacity_tokens=4, prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            steps.append(step)
            if step.request.prompt == 'owner' and step.phase == 'decode' and step.offset == 0:
                entered.set()
                await release.wait()
            return await super().execute(step)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        owner = scheduler.submit('owner', 3)  # Reserves all memory.
        first = scheduler.submit('first', 1)
        second = scheduler.submit('second', 1)
        await asyncio.wait_for(entered.wait(), 2)
        assert (scheduler._pending is first) == pending
        third = scheduler.submit('third', 1)
        with pytest.raises(QueueFullError):
            scheduler.submit('overflow', 1)
        assert scheduler.cancel(first.request_id)
        assert_cancelled(first, await collect(first))
        assert first.timings.snapshot()['queue_wait_ms'] is None
        assert not scheduler.cancel(first.request_id)
        assert scheduler.queued_requests == 2
        replacement = scheduler.submit('replacement', 1)
        # Cancel a non-head queued item as well, without disturbing FIFO.
        assert scheduler.cancel(third.request_id)
        assert_cancelled(third, await collect(third))
        assert scheduler.queued_requests == 2
        release.set()
        await asyncio.gather(*(collect(r) for r in (owner, second, replacement)))
        await asyncio.wait_for(scheduler.queue.join(), 2)
        prefill_order = [s.request for s in steps if s.phase == 'prefill']
        assert prefill_order == [owner, second, replacement]
        assert not scheduler._requests
        assert not scheduler.kv_cache.allocations
    finally:
        release.set()
        await scheduler.stop()


async def test_cancel_before_worker_first_turn_and_after_restart():
    config = RuntimeConfig(prefill_delay=0, token_delay=0)
    scheduler = ContinuousBatchScheduler(SimulatedExecutor(config), config)
    with pytest.raises(RuntimeUnavailableError):
        scheduler.cancel('unknown')
    await scheduler.start()
    request = scheduler.submit('never admitted', 1)
    assert scheduler.cancel(request.request_id)
    assert_cancelled(request, await collect(request))
    await asyncio.wait_for(scheduler.queue.join(), 2)
    await scheduler.stop()
    assert not scheduler._requests
    with pytest.raises(RuntimeUnavailableError):
        scheduler.cancel(request.request_id)
    await scheduler.start()
    try:
        assert not scheduler.cancel(request.request_id)
        finished = scheduler.submit('complete', 1)
        assert (await collect(finished))[-1].data['finish_reason'] == 'length'
        assert not scheduler.cancel(finished.request_id)
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
async def test_cancel_unselected_active_request_at_iteration_boundary(policy):
    entered, release = asyncio.Event(), asyncio.Event()
    steps = []
    config = RuntimeConfig(scheduling_policy=policy, token_budget=1,
                           max_active_requests=2, prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            steps.append(step)
            if step.request.prompt == 'owner' and step.phase == 'prefill':
                entered.set()
                await release.wait()
            return await super().execute(step)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        owner = scheduler.submit('owner', 1)
        request = scheduler.submit('unselected', 1)
        await asyncio.wait_for(entered.wait(), 2)
        assert request in scheduler.active
        assert scheduler.cancel(request.request_id)
        assert request.request_id in scheduler.kv_cache.allocations
        release.set()
        assert_cancelled(request, await collect(request))
        assert (await collect(owner))[-1].kind == 'done'
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert all(s.request is not request for s in steps)
        assert not scheduler._requests
        assert not scheduler.kv_cache.allocations
    finally:
        release.set()
        await scheduler.stop()


@pytest.mark.parametrize('abandoned', [False, True])
async def test_accepted_cancel_wins_shutdown_but_disconnect_suppresses_events(abandoned):
    entered = asyncio.Event()
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await asyncio.Event().wait()

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    request = scheduler.submit('cancel', 1)
    await asyncio.wait_for(entered.wait(), 2)
    assert scheduler.cancel(request.request_id)
    request.abandoned = abandoned
    await asyncio.wait_for(scheduler.stop(), 2)
    assert request.phase == 'cancelled'
    if abandoned:
        assert request.events.empty()
    else:
        assert_cancelled(request, await collect(request))
    await asyncio.wait_for(scheduler.queue.join(), 2)
    assert not scheduler.kv_cache.allocations
    assert not scheduler._requests


async def test_http_cancel_stream_using_response_header_request_id():
    entered, release, headers_ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            return await super().execute(step)

    app = create_app(config, Executor(config))
    chunks, headers = [], {}
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': b'{"prompt":"cancel","max_tokens":3}'}
        await asyncio.Event().wait()

    async def send(message):
        if message['type'] == 'http.response.start':
            headers.update(dict(message['headers']))
            headers_ready.set()
        elif message['type'] == 'http.response.body':
            chunks.append(message.get('body', b''))

    async with app.router.lifespan_context(app):
        task = asyncio.create_task(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': '/generate', 'raw_path': b'/generate', 'query_string': b'',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send))
        try:
            await asyncio.wait_for(headers_ready.wait(), 2)
            await asyncio.wait_for(entered.wait(), 2)
            request_id = headers[b'x-request-id'].decode()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url='http://test') as client:
                path = f'/requests/{request_id}/cancel'
                for _ in range(2):
                    response = await client.post(path)
                    assert response.status_code == 202
                    assert response.json() == {'request_id': request_id,
                                               'status': 'cancellation_requested'}
                assert (await client.post('/requests/unknown/cancel')).status_code == 404
                release.set()
                await asyncio.wait_for(task, 2)
                assert (await client.post(path)).status_code == 404
                await app.state.scheduler.stop()
                assert (await client.post(path)).status_code == 503
            frames = b''.join(chunks).decode()
            assert frames.count('event: done') == 1
            assert 'event: token' not in frames
            assert 'event: error' not in frames
            payload = json.loads(frames.splitlines()[1][6:])
            assert payload['finish_reason'] == 'cancelled'
            assert payload['request_id'] == request_id
            assert payload['generated_tokens'] == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
