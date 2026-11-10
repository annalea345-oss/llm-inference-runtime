"""HTTP abandonment must reclaim waiting slots while model steps are in flight."""

import asyncio
from collections import Counter
import json

import pytest
from starlette.requests import ClientDisconnect

from llm_runtime.api import create_app
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.scheduler import QueueFullError


async def until(predicate):
    async def spin():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(spin(), 2)


async def collect(request):
    async def read():
        while True:
            event = await request.events.get()
            if event.kind in ('done', 'error'):
                return event
    return await asyncio.wait_for(read(), 2)


CASES = [(kind, exit_mode) for kind in ('native', 'chat_stream', 'chat_json')
         for exit_mode in ('disconnect', 'task_cancel')]
CASES += [(kind, 'header_failure') for kind in ('native', 'chat_stream')]
CASES += [(kind, 'modern_disconnect') for kind in ('native', 'chat_stream')]
CASES += [(kind, 'modern_task_cancel') for kind in ('native', 'chat_stream')]


@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
@pytest.mark.parametrize('kind,exit_mode', CASES)
async def test_disconnected_waiter_frees_capacity_during_full_active_barrier(policy, kind, exit_mode):
    entered, release, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()
    steps, released = [], []
    config = RuntimeConfig(max_active_requests=2, queue_capacity=3, token_budget=32,
                           scheduling_policy=policy, prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            steps.append(step)
            if step.request.prompt.startswith('owner') and step.phase == 'prefill':
                if sum(s.request.prompt.startswith('owner') for s in steps) == 2:
                    entered.set()
                await release.wait()
            return await super().execute(step)

        def release_request(self, request_id):
            released.append(request_id)

    app = create_app(config, Executor(config))
    if kind == 'native':
        path, body = '/generate', {'prompt': 'abandoned', 'max_tokens': 1}
    else:
        path = '/v1/chat/completions'
        body = {'model': 'simulated', 'messages': [{'role': 'user', 'content': 'abandoned'}],
                'max_tokens': 1, 'stream': kind == 'chat_stream'}
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': json.dumps(body).encode()}
        await disconnect.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        if message['type'] == 'http.response.start' and exit_mode == 'header_failure':
            await disconnect.wait()
            raise OSError('closed during header send')

    async with app.router.lifespan_context(app):
        scheduler = app.state.scheduler
        owners = [scheduler.submit(f'owner-{i}', 1) for i in range(2)]
        await asyncio.wait_for(entered.wait(), 2)
        first = scheduler.submit('first', 1)
        previous = set(scheduler._requests)
        task = asyncio.create_task(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version':
                                   '2.4' if exit_mode in ('header_failure', 'modern_disconnect', 'modern_task_cancel') else '2.0'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': path, 'raw_path': path.encode(), 'query_string': b'',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send))
        joining = None
        try:
            await until(lambda: len(scheduler._requests) == len(previous) + 1)
            target = next(r for key, r in scheduler._requests.items() if key not in previous)
            last = scheduler.submit('last', 1)
            assert scheduler.queued_requests == 3
            with pytest.raises(QueueFullError):
                scheduler.submit('overflow', 1)
            joining = asyncio.create_task(scheduler.queue.join())
            if exit_mode in ('task_cancel', 'modern_task_cancel'):
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                disconnect.set()
                if exit_mode == 'header_failure':
                    with pytest.raises(ClientDisconnect):
                        await asyncio.wait_for(task, 2)
                else:
                    await asyncio.wait_for(task, 2)
            # The model barrier is deliberately still blocked: no admission turn
            # or model result can be responsible for reclaiming this queue slot.
            assert not release.is_set()
            assert scheduler.queued_requests == 2
            assert target.phase == 'cancelled' and target.abandoned
            assert target.generated_tokens == 0 and target.events.empty()
            assert target.request_id not in scheduler._requests
            assert released == [target.request_id]
            assert set(scheduler.kv_cache.allocations) == {r.request_id for r in owners}
            assert scheduler.kv_cache.reserved_tokens == 4
            assert not joining.done()
            assert not scheduler.cancel(target.request_id)
            replacement = scheduler.submit('replacement', 1)
            assert scheduler.queued_requests == 3
            release.set()
            terminals = await asyncio.gather(*(collect(r) for r in [*owners, first, last, replacement]))
            assert all(event.kind == 'done' for event in terminals)
            await asyncio.wait_for(joining, 2)
            assert [s.request for s in steps if s.phase == 'prefill' and s.request not in owners] == [first, last, replacement]
            assert Counter(released) == Counter(r.request_id for r in [*owners, target, first, last, replacement])
            assert not scheduler._requests and not scheduler.kv_cache.allocations
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if joining is not None:
                await asyncio.wait_for(joining, 2)


async def test_abandon_memory_pending_head_frees_waiting_slot_without_releasing_owner():
    entered, release = asyncio.Event(), asyncio.Event()
    released, prefills = [], []
    config = RuntimeConfig(max_active_requests=2, queue_capacity=2, kv_cache_capacity_tokens=4,
                           prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.phase == 'prefill':
                prefills.append(step.request)
            if step.request.prompt == 'owner' and step.phase == 'prefill':
                entered.set()
                await release.wait()
            return await super().execute(step)

        def release_request(self, request_id):
            released.append(request_id)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    try:
        owner = scheduler.submit('owner', 3)
        pending = scheduler.submit('pending', 1)
        await asyncio.wait_for(entered.wait(), 2)
        assert scheduler._pending is pending
        survivor = scheduler.submit('survivor', 1)
        assert scheduler.abandon(pending.request_id)
        assert not scheduler.abandon(pending.request_id)
        assert scheduler._pending is None and scheduler.queued_requests == 1
        assert pending.phase == 'cancelled' and pending.events.empty()
        assert released == [pending.request_id]
        assert scheduler.kv_cache.reserved_tokens == 4
        replacement = scheduler.submit('replacement', 1)
        release.set()
        await asyncio.gather(*(collect(r) for r in (owner, survivor, replacement)))
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert prefills == [owner, survivor, replacement]
        assert Counter(released) == Counter(r.request_id for r in (owner, pending, survivor, replacement))
        assert not scheduler._requests and not scheduler.kv_cache.allocations
    finally:
        release.set()
        await scheduler.stop()


@pytest.mark.parametrize('kind', ['native', 'chat_stream'])
async def test_modern_normal_completion_reaps_disconnect_listener(monkeypatch, kind):
    from llm_runtime.api import GenerationResponse

    listener_started, listener_stopped = asyncio.Event(), asyncio.Event()
    original = GenerationResponse.listen_for_disconnect

    async def listen(self, receive):
        listener_started.set()
        try:
            await original(self, receive)
        finally:
            listener_stopped.set()

    monkeypatch.setattr(GenerationResponse, 'listen_for_disconnect', listen)
    config = RuntimeConfig(prefill_delay=0, token_delay=0)
    app = create_app(config)
    path = '/generate' if kind == 'native' else '/v1/chat/completions'
    body = {'prompt': 'hello', 'max_tokens': 1} if kind == 'native' else {
        'model': 'simulated', 'messages': [{'role': 'user', 'content': 'hello'}],
        'stream': True, 'max_tokens': 1}
    sent, messages = False, []

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': json.dumps(body).encode()}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app({
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.4'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http', 'path': path,
            'raw_path': path.encode(), 'query_string': b'',
            'headers': [(b'content-type', b'application/json')],
            'client': ('127.0.0.1', 1234), 'server': ('test', 80),
        }, receive, send), 2)
        assert listener_started.is_set() and listener_stopped.is_set()
        content = b''.join(m.get('body', b'') for m in messages)
        assert (b'event: done' if kind == 'native' else b'data: [DONE]') in content
        scheduler = app.state.scheduler
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert not scheduler._requests and not scheduler.kv_cache.allocations


async def test_abandon_during_shutdown_keeps_inflight_ownership_and_is_safe_after_restart():
    entered, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    released = []
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            if step.request.prompt == 'owner':
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelling.set()
                    await release.wait()
                    raise
            return await super().execute(step)

        def release_request(self, request_id):
            released.append(request_id)

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    assert not scheduler.abandon('unknown')
    await scheduler.start()
    owner = scheduler.submit('owner', 1)
    await asyncio.wait_for(entered.wait(), 2)
    stopping = asyncio.create_task(scheduler.stop())
    try:
        await asyncio.wait_for(cancelling.wait(), 2)
        assert not scheduler.abandon(owner.request_id)
        assert owner.abandoned and not released
        assert scheduler.kv_cache.reserved_tokens == 2
        release.set()
        await asyncio.wait_for(stopping, 2)
        assert released == [owner.request_id]
        assert owner.events.empty() and not scheduler.kv_cache.allocations
        assert not scheduler.abandon(owner.request_id)
        await scheduler.start()
        replacement = scheduler.submit('replacement', 1)
        assert not scheduler.abandon(owner.request_id)
        assert (await collect(replacement)).kind == 'done'
        await asyncio.wait_for(scheduler.queue.join(), 2)
    finally:
        release.set()
        await asyncio.gather(stopping, return_exceptions=True)
        await scheduler.stop()
