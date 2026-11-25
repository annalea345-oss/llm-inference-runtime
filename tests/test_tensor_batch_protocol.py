"""Optional executor hook contract and local factory configuration."""

import asyncio

import pytest

from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.protocol import DecodeResult


class BatchExecutor:
    def __init__(self, failure=None):
        self.failure = failure
        self.batches = []
        self.released = []

    async def execute(self, step):
        raise AssertionError('batch-capable executor should use its batch hook')

    async def execute_batch(self, steps):
        self.batches.append(steps)
        if self.failure == 'raise':
            raise RuntimeError('shared forward failed')
        if self.failure == 'length':
            return []
        if self.failure == 'type':
            return None
        return [None if s.phase == 'prefill' else DecodeResult('x', 7) for s in steps]

    def release_request(self, request_id):
        self.released.append(request_id)


@pytest.mark.parametrize('failure', [None, 'raise', 'length', 'type'])
async def test_hook_results_errors_and_queue_accounting(failure):
    executor = BatchExecutor(failure)
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=2))
    await scheduler.start()
    try:
        requests = [scheduler.submit('a', 1), scheduler.submit('b', 2)]
        await asyncio.wait_for(scheduler.queue.join(), 2)
        for request in requests:
            events = []
            while not request.events.empty():
                events.append(request.events.get_nowait())
            assert events[-1].kind == ('error' if failure else 'done')
        assert len(executor.batches[0]) == 2
        assert sorted(executor.released) == sorted(r.request_id for r in requests)
        assert scheduler.running and not scheduler._requests
        assert not scheduler.kv_cache.allocations
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('value', ['true', 'yes', '', '2'])
def test_factory_rejects_ambiguous_tensor_batching_configuration(monkeypatch, value):
    from llm_runtime.local import create_local_app
    monkeypatch.setenv('LLM_MODEL_PATH', '.models/not-loaded')
    monkeypatch.setenv('LLM_TENSOR_BATCHING', value)
    with pytest.raises(ValueError, match='LLM_TENSOR_BATCHING'):
        create_local_app()


@pytest.mark.parametrize('value,enabled', [(None, False), ('0', False), ('1', True)])
async def test_factory_opt_in_preserves_original_default(monkeypatch, value, enabled):
    pytest.importorskip('torch')
    from llm_runtime.local import create_local_app
    from llm_runtime.local_executor import LocalModelExecutor
    monkeypatch.setenv('LLM_MODEL_PATH', '/model-path')
    if value is None:
        monkeypatch.delenv('LLM_TENSOR_BATCHING', raising=False)
    else:
        monkeypatch.setenv('LLM_TENSOR_BATCHING', value)
    seen = []
    async def load(path, **kwargs):
        seen.append((path, kwargs))
        return BatchExecutor()
    monkeypatch.setattr(LocalModelExecutor, 'load', load)
    app = create_local_app()
    async with app.router.lifespan_context(app):
        assert seen == [('/model-path', {'tensor_batching': enabled})]
