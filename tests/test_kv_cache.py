import asyncio

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.kv_cache import RequestTooLargeError, SimulatedKVCache
from llm_runtime.scheduler import QueueFullError


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_capacity(value):
    with pytest.raises(ValueError):
        SimulatedKVCache(value)
    with pytest.raises(ValueError):
        RuntimeConfig(kv_cache_capacity_tokens=value)


def test_accounting_is_atomic_and_snapshots_do_not_mutate_ownership():
    cache = SimulatedKVCache(5)
    assert cache.reserve('a', 3)
    cache.consume('a', 2)
    assert not cache.reserve('b', 3)
    assert cache.reserve('b', 2)
    assert (cache.reserved_tokens, cache.used_tokens, cache.available_tokens) == (5, 2, 0)
    snapshot = cache.allocations
    snapshot.clear()
    assert len(cache.allocations) == 2
    with pytest.raises(ValueError):
        cache.reserve('a', 1)
    with pytest.raises(RequestTooLargeError):
        cache.reserve('large', 6)
    with pytest.raises(ValueError):
        cache.consume('a', 2)
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            cache.reserve('invalid', value)
        with pytest.raises(ValueError):
            cache.consume('a', value)
    assert (cache.reserved_tokens, cache.used_tokens) == (5, 2)
    cache.release('a')
    cache.release('a')
    assert (cache.reserved_tokens, cache.used_tokens, cache.available_tokens) == (2, 0, 3)
    cache.release('b')
    assert not cache.allocations


class GatedExecutor(SimulatedExecutor):
    def __init__(self, config):
        super().__init__(config)
        self.entered = asyncio.Queue()
        self.permits = asyncio.Queue()

    async def execute(self, step):
        self.entered.put_nowait(step)
        action = await self.permits.get()
        if action == 'fail':
            raise RuntimeError('private error')
        if action == 'invalid':
            return None
        return await super().execute(step)


async def next_step(executor):
    return await asyncio.wait_for(executor.entered.get(), 2)


async def joined(scheduler):
    await asyncio.wait_for(scheduler.queue.join(), 2)
    assert not scheduler.kv_cache.allocations
    assert scheduler.kv_cache.used_tokens == scheduler.kv_cache.reserved_tokens == 0


async def test_chunked_usage_and_completion_release_before_terminal_consumption():
    config = RuntimeConfig(token_budget=2, kv_cache_capacity_tokens=5,
                           prefill_delay=0, token_delay=0)
    executor = GatedExecutor(config)
    scheduler = ContinuousBatchScheduler(executor, config)
    await scheduler.start()
    try:
        request = scheduler.submit('one two three', 2)
        for phase, offset, usage in [('prefill', 0, 0), ('prefill', 2, 2),
                                     ('decode', 0, 3), ('decode', 1, 4)]:
            step = await next_step(executor)
            assert (step.phase, step.offset) == (phase, offset)
            allocation = scheduler.kv_cache.allocations[request.request_id]
            assert allocation.reserved_tokens == 5
            assert allocation.used_tokens == usage
            executor.permits.put_nowait('ok')
        await joined(scheduler)
        assert request.generated_tokens == 2
        assert request.events.qsize() == 3  # Slow reader retains no cache.
        scheduler._finish(request, 'finished')  # Terminal cleanup is idempotent.
        assert request.events.qsize() == 3
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('abandon_head', [False, True])
async def test_memory_wait_preserves_fifo_queue_bound_and_health(abandon_head):
    config = RuntimeConfig(kv_cache_capacity_tokens=6, queue_capacity=2,
                           max_active_requests=3, prefill_delay=0, token_delay=0)
    executor = GatedExecutor(config)
    app = create_app(config, executor)
    async with app.router.lifespan_context(app):
        scheduler = app.state.scheduler
        first = scheduler.submit('first', 3)  # Reserves 4; leaves 2 available.
        assert (await next_step(executor)).request is first
        head = scheduler.submit('head', 2)  # Needs 3; blocks smaller tail.
        tail = scheduler.submit('tail', 1)
        executor.permits.put_nowait('ok')
        assert (await next_step(executor)).request is first
        assert head.phase == tail.phase == 'waiting'
        assert scheduler.queue.qsize() == 1
        assert scheduler.queued_requests == 2  # Includes dequeued pending head.
        with pytest.raises(QueueFullError):
            scheduler.submit('overflow', 1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            health = (await client.get('/health')).json()
            assert health['queued_requests'] == 2
            assert health['kv_cache'] == dict(capacity_tokens=6, reserved_tokens=4,
                                              used_tokens=1, available_tokens=2)
        if abandon_head:
            head.abandoned = True
        executor.permits.put_nowait('ok')
        step = await next_step(executor)
        assert step.request is first
        if abandon_head:
            other = await next_step(executor)
            assert other.request is tail
            assert head.phase == 'cancelled'
            assert head.events.empty()
        # All remaining work can now run; queued permits do not depend on timing.
        for _ in range(30):
            executor.permits.put_nowait('ok')
        await joined(scheduler)
        assert first.phase == tail.phase == 'finished'
        if not abandon_head:
            assert head.phase == 'finished'


@pytest.mark.parametrize('phase,offset,action', [
    (phase, offset, action)
    for phase, offset in [('prefill', 0), ('prefill', 1), ('decode', 0)]
    for action in ['cancel', 'fail', 'shutdown']
] + [('decode', 0, 'invalid')])
async def test_terminal_paths_release_partial_usage_and_pending_head(phase, offset, action):
    config = RuntimeConfig(token_budget=1, kv_cache_capacity_tokens=4,
                           prefill_delay=0, token_delay=0)
    executor = GatedExecutor(config)
    scheduler = ContinuousBatchScheduler(executor, config)
    await scheduler.start()
    try:
        request = scheduler.submit('two words', 2)
        waiting = None
        while True:
            step = await next_step(executor)
            if waiting is None:
                waiting = scheduler.submit('waiting', 1)
            if (step.phase, step.offset) == (phase, offset):
                break
            executor.permits.put_nowait('ok')
        assert scheduler.kv_cache.reserved_tokens == 4
        assert scheduler.kv_cache.used_tokens == (offset if phase == 'prefill' else 2)
        if action == 'shutdown':
            await scheduler.stop()
            assert waiting.phase == 'failed'
        else:
            if action == 'cancel':
                request.abandoned = True
                assert scheduler.kv_cache.reserved_tokens == 4
            executor.permits.put_nowait(action)
            for _ in range(10):
                executor.permits.put_nowait('ok')
        await joined(scheduler)
        assert request.phase == ('cancelled' if action == 'cancel' else 'failed')
        if action == 'cancel':
            assert request.events.empty()
        if action == 'shutdown':
            await scheduler.start()
            again = scheduler.submit('restart', 1)
            for _ in range(3):
                executor.permits.put_nowait('ok')
            await joined(scheduler)
            assert again.phase == 'finished'
    finally:
        await scheduler.stop()


async def test_oversized_http_request_does_not_enter_queue_and_whitespace_exact_fit():
    config = RuntimeConfig(kv_cache_capacity_tokens=2, prefill_delay=0, token_delay=0)
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://test') as client:
            response = await client.post('/generate', json={'prompt': 'two words', 'max_tokens': 1})
            assert response.status_code == 422
            assert response.json()['detail'] == 'request exceeds KV-cache token capacity'
            scheduler = app.state.scheduler
            assert scheduler.queued_requests == 0
            await joined(scheduler)
            response = await client.post('/generate', json={'prompt': '   ', 'max_tokens': 1})
            assert response.status_code == 200
            assert '<empty> ' in response.text
            await joined(scheduler)
