"""Real-token executor integration, without optional model dependencies."""

import asyncio
import subprocess
import sys
import threading

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.protocol import DecodeResult
from llm_runtime.worker import SerialWorker


class TokenExecutor:
    context_limit = 64
    info = {'backend': 'test-model', 'tokenization': 'model', 'device': 'cpu'}

    def __init__(self, *, gate=False, fail=False, cleanup_failure=False):
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()
        if not gate:
            self.proceed.set()
        self.fail = fail
        self.cleanup_failure = cleanup_failure
        self.steps = []
        self.released = []

    def tokenize(self, prompt):
        # One whitespace word deliberately expands into five model tokens.
        return (101, 102, 103, 104, 105)

    async def execute(self, step):
        self.steps.append(step)
        self.started.set()
        await self.proceed.wait()
        if self.fail:
            raise RuntimeError('forward failed')
        if step.phase == 'decode':
            return DecodeResult('', 2, finish_reason='eos')
        return None

    def release_request(self, request_id):
        self.released.append(request_id)
        if self.cleanup_failure:
            raise RuntimeError('cleanup failed')


async def collect(request):
    async def read():
        events = []
        while True:
            event = await request.events.get()
            events.append(event)
            if event.kind in ('done', 'error'):
                return events
    return await asyncio.wait_for(read(), 2)


async def test_model_ids_drive_prefill_reservation_and_empty_eos_token():
    executor = TokenExecutor(gate=True)
    scheduler = ContinuousBatchScheduler(
        executor, RuntimeConfig(token_budget=2, kv_cache_capacity_tokens=8),
    )
    await scheduler.start()
    try:
        request = scheduler.submit('word', 3)
        await asyncio.wait_for(executor.started.wait(), 2)
        assert request.prompt_token_ids == (101, 102, 103, 104, 105)
        assert request.prompt_token_count == 5
        assert scheduler.kv_cache.reserved_tokens == 8
        executor.proceed.set()
        events = await collect(request)
        assert [(s.offset, s.token_count) for s in executor.steps if s.phase == 'prefill'] == [
            (0, 2), (2, 2), (4, 1),
        ]
        assert [e.kind for e in events] == ['token', 'done']
        assert events[0].data == {
            'request_id': request.request_id, 'index': 0, 'text': '', 'token_id': 2,
        }
        assert events[-1].data['finish_reason'] == 'eos'
        assert events[-1].data['generated_tokens'] == 1
        assert request.prefilled_tokens == 5
        assert request.generated_tokens == 1
        assert executor.released == [request.request_id]
        assert scheduler.kv_cache.reserved_tokens == scheduler.kv_cache.used_tokens == 0
        await asyncio.wait_for(scheduler.queue.join(), 2)
    finally:
        executor.proceed.set()
        await scheduler.stop()


@pytest.mark.parametrize('case', ['failure', 'active_cancel', 'waiting_cancel', 'shutdown'])
async def test_release_hook_runs_once_for_all_terminal_paths(case):
    executor = TokenExecutor(gate=True, fail=case == 'failure')
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=1))
    await scheduler.start()
    try:
        active = scheduler.submit('active', 3)
        await asyncio.wait_for(executor.started.wait(), 2)
        waiting = scheduler.submit('waiting', 3)
        if case == 'waiting_cancel':
            assert scheduler.cancel(waiting.request_id)
            assert (await collect(waiting))[-1].data['finish_reason'] == 'cancelled'
            assert executor.released == [waiting.request_id]
            assert not any(s.request is waiting for s in executor.steps)
        elif case == 'active_cancel':
            assert scheduler.cancel(active.request_id)
            assert executor.released == []  # Execution still owns its state.
        elif case == 'shutdown':
            await scheduler.stop()
            assert (await collect(active))[-1].kind == 'error'
            assert (await collect(waiting))[-1].kind == 'error'
        executor.proceed.set()
        if case != 'shutdown':
            active_events = await collect(active)
            assert active_events[-1].kind == ('error' if case == 'failure' else 'done')
            if case == 'active_cancel':
                assert active_events[-1].data['finish_reason'] == 'cancelled'
            if case != 'waiting_cancel':
                await collect(waiting)
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert sorted(executor.released) == sorted([active.request_id, waiting.request_id])
        assert not scheduler.kv_cache.allocations
        assert not scheduler._requests
    finally:
        executor.proceed.set()
        await scheduler.stop()
    assert len(executor.released) == 2


async def test_cleanup_failure_keeps_scheduler_alive_and_balances_accounting():
    executor = TokenExecutor(cleanup_failure=True)
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig())
    await scheduler.start()
    try:
        first = scheduler.submit('first', 3)
        assert (await collect(first))[-1].kind == 'error'
        second = scheduler.submit('second', 3)
        assert (await collect(second))[-1].kind == 'error'
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert scheduler.running
        assert executor.released == [first.request_id, second.request_id]
        assert not scheduler.kv_cache.allocations
        assert not scheduler._requests
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('limit', ['context', 'memory'])
async def test_http_rejects_model_token_footprint_and_allows_exact_fit(limit):
    executor = TokenExecutor()
    executor.context_limit = 8 if limit == 'context' else 64
    config = RuntimeConfig(kv_cache_capacity_tokens=8 if limit == 'memory' else 64)
    app = create_app(config, executor)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            too_big = await client.post('/generate', json={'prompt': 'word', 'max_tokens': 4})
            assert too_big.status_code == 422
            assert ('context' if limit == 'context' else 'KV-cache') in too_big.json()['detail']
            assert not executor.steps
            assert not executor.released
            exact = await client.post('/generate', json={'prompt': 'word', 'max_tokens': 3})
            assert exact.status_code == 200
            assert '"finish_reason": "eos"' in exact.text
            assert len(executor.released) == 1


async def test_async_loader_lifespan_health_and_close_after_scheduler_drain():
    executor = TokenExecutor(gate=True)
    loaded, closed = [], []
    app = None

    async def load():
        await asyncio.sleep(0)
        loaded.append(True)
        return executor

    async def close():
        scheduler = app.state.scheduler
        assert not scheduler.running
        assert not scheduler._requests
        assert not scheduler.kv_cache.allocations
        assert len(executor.released) == 2
        await scheduler.queue.join()
        await asyncio.sleep(0)
        closed.append(True)

    executor.aclose = close
    app = create_app(RuntimeConfig(max_active_requests=1), executor_loader=load)
    assert not loaded
    async with app.router.lifespan_context(app):
        assert loaded == [True]
        assert app.state.scheduler.executor is executor
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get('/health')
            assert response.status_code == 200
            assert response.json()['executor'] == executor.info
        active = app.state.scheduler.submit('active', 3)
        await asyncio.wait_for(executor.started.wait(), 2)
        waiting = app.state.scheduler.submit('waiting', 3)
        assert not closed
    assert closed == [True]
    assert sorted(executor.released) == sorted([active.request_id, waiting.request_id])


async def test_threaded_execution_retains_memory_until_cancelled_shutdown_settles():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class ThreadedExecutor(TokenExecutor):
        def __init__(self):
            super().__init__()
            self.worker = SerialWorker()
            self.closed = False

        def forward(self):
            started.set()
            release.wait()
            finished.set()

        async def execute(self, step):
            await self.worker.run(self.forward)
            return None

        def release_request(self, request_id):
            assert finished.is_set()
            super().release_request(request_id)

        async def aclose(self):
            await self.worker.aclose()
            self.closed = True

    executor = ThreadedExecutor()
    app = create_app(executor=executor)
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    scheduler = app.state.scheduler
    close_task = None
    try:
        request = scheduler.submit('active', 3)

        async def wait_for_thread():
            while not started.is_set():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait_for_thread(), 2)
        assert scheduler.cancel(request.request_id)
        close_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.sleep(0)
        # Repeated shutdown cancellation must not release state still in use.
        for _ in range(3):
            close_task.cancel()
            await asyncio.sleep(0)
            assert not close_task.done()
            assert executor.released == []
            assert scheduler.kv_cache.reserved_tokens == 8
            assert not executor.closed
        release.set()
        await asyncio.wait_for(close_task, 2)
        events = await collect(request)
        assert [event.kind for event in events] == ['done']
        assert events[0].data['finish_reason'] == 'cancelled'
        assert events[0].data['generated_tokens'] == 0
        assert request.events.empty()
        assert executor.released == [request.request_id]
        assert scheduler.kv_cache.reserved_tokens == 0
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert executor.closed
    finally:
        release.set()
        if close_task is None:
            await lifespan.__aexit__(None, None, None)
        elif not close_task.done():
            await asyncio.wait_for(close_task, 2)
        await executor.aclose()


async def test_loader_executor_is_closed_if_scheduler_start_fails(monkeypatch):
    executor = TokenExecutor()
    closed = []

    async def load():
        return executor

    async def close():
        await asyncio.sleep(0)
        closed.append(True)

    async def fail_start(self):
        raise RuntimeError('scheduler startup failed')

    executor.aclose = close
    monkeypatch.setattr(ContinuousBatchScheduler, 'start', fail_start)
    app = create_app(executor_loader=load)
    with pytest.raises(RuntimeError, match='scheduler startup failed'):
        async with app.router.lifespan_context(app):
            pytest.fail('startup failure should prevent entering the lifespan')
    assert closed == [True]


def test_loader_and_executor_cannot_both_be_supplied():
    async def load():
        return TokenExecutor()
    with pytest.raises(ValueError, match='not both'):
        create_app(executor=TokenExecutor(), executor_loader=load)


def test_default_and_local_module_imports_do_not_load_optional_dependencies():
    script = '''
import importlib.abc
import sys

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers'}:
            raise AssertionError('optional dependency imported: ' + fullname)

sys.meta_path.insert(0, RejectOptional())
from llm_runtime.api import app
from llm_runtime.local_executor import LocalModelExecutor
assert app is not None
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
