import asyncio
import json

import httpx
import pytest

from llm_runtime.api import create_app
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.benchmark import benchmark, distribution
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.metrics import RequestTimings
from llm_runtime.scheduler import FCFSScheduler


def test_exact_monotonic_calculations_and_idempotent_boundaries():
    now = [10.0]
    timings = RequestTimings(clock=lambda: now[0])
    assert timings.snapshot() == dict(queue_wait_ms=None, time_to_first_token_ms=None,
                                      inter_token_ms=[], end_to_end_ms=None)
    now[0] = 12
    timings.admit()
    now[0] = 13
    timings.admit()
    timings.token()
    now[0] = 15
    timings.token()
    now[0] = 16
    timings.finish()
    now[0] = 20
    timings.finish()
    assert timings.snapshot() == dict(queue_wait_ms=2000, time_to_first_token_ms=3000,
                                      inter_token_ms=[2000], end_to_end_ms=6000)
    snapshot = timings.snapshot()
    snapshot['inter_token_ms'].clear()
    assert timings.snapshot()['inter_token_ms'] == [2000]


@pytest.mark.parametrize('mode', ['fcfs', 'continuous'])
async def test_scheduler_records_exact_admission_emission_and_finish_times(mode):
    now = [0.0]
    entered = asyncio.Queue()
    release = asyncio.Queue()
    config = RuntimeConfig(prefill_delay=0, token_delay=0)

    class Executor(SimulatedExecutor):
        async def piece(self):
            entered.put_nowait(True)
            await release.get()
            return 'x '

        async def execute(self, step):
            return None if step.phase == 'prefill' else await self.piece()

        async def generate(self, request):
            for _ in range(request.max_tokens):
                yield await self.piece()

    executor = Executor(config)
    scheduler = (FCFSScheduler(executor, 4) if mode == 'fcfs'
                 else ContinuousBatchScheduler(executor, config))
    await scheduler.start()
    try:
        request = scheduler.submit('x', 2)
        request.timings = RequestTimings(clock=lambda: now[0])
        now[0] = 2
        await asyncio.wait_for(entered.get(), 2)
        now[0] = 5
        release.put_nowait(True)
        await asyncio.wait_for(entered.get(), 2)
        now[0] = 8
        release.put_nowait(True)
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert request.timings.snapshot() == dict(queue_wait_ms=2000,
            time_to_first_token_ms=5000, inter_token_ms=[3000], end_to_end_ms=8000)
        events = [request.events.get_nowait() for _ in range(3)]
        assert events[-1].data['timings'] == request.timings.snapshot()
    finally:
        await scheduler.stop()


@pytest.mark.parametrize('action', ['cancel-waiting', 'shutdown-waiting', 'fail', 'shutdown-active'])
async def test_incomplete_requests_keep_missing_metrics_null(action):
    config = RuntimeConfig(prefill_delay=0, token_delay=0)
    entered = asyncio.Event()
    release = asyncio.Event()

    class Executor(SimulatedExecutor):
        async def execute(self, step):
            entered.set()
            await release.wait()
            raise RuntimeError('test failure')

    scheduler = ContinuousBatchScheduler(Executor(config), config)
    await scheduler.start()
    request = scheduler.submit('x', 1)
    if action == 'cancel-waiting':
        request.abandoned = True
        await asyncio.wait_for(scheduler.queue.join(), 2)
    elif action == 'shutdown-waiting':
        await scheduler.stop()
    else:
        await asyncio.wait_for(entered.wait(), 2)
        if action == 'fail':
            release.set()
            await asyncio.wait_for(scheduler.queue.join(), 2)
        else:
            await scheduler.stop()
    await scheduler.stop()
    metrics = request.timings.snapshot()
    assert metrics['time_to_first_token_ms'] is None
    assert metrics['inter_token_ms'] == []
    assert metrics['end_to_end_ms'] is not None
    if action.endswith('waiting'):
        assert metrics['queue_wait_ms'] is None
    else:
        assert metrics['queue_wait_ms'] is not None


async def test_done_sse_includes_serializable_timing_metrics():
    app = create_app(RuntimeConfig(prefill_delay=0, token_delay=0))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post('/generate', json={'prompt': 'x', 'max_tokens': 1})
            payload = json.loads(response.text.strip().split('\n\n')[-1].splitlines()[1][6:])
            metrics = payload['timings']
            assert 0 <= metrics['queue_wait_ms'] <= metrics['time_to_first_token_ms'] <= metrics['end_to_end_ms']
            assert metrics['inter_token_ms'] == []


def test_percentiles_use_nearest_rank_and_empty_populations_are_null():
    assert distribution([]) == dict(count=0, mean=None, p50=None, p95=None, p99=None)
    assert distribution([4, 1, 3, 2]) == dict(count=4, mean=2.5, p50=2, p95=4, p99=4)


@pytest.mark.parametrize('mode', ['fcfs', 'continuous'])
async def test_benchmark_compares_identical_workloads_and_excludes_warmup(mode):
    report = await benchmark(mode, RuntimeConfig(prefill_delay=0, token_delay=0),
                             requests=5, concurrency=2, prompt='x y', output_pattern=[1, 3], warmup=2)
    assert report['completed'] == 5
    assert report['rejected'] == report['errors'] == report['timeouts'] == 0
    assert [r['generated_tokens'] for r in report['requests']] == [1, 3, 1, 3, 1]
    assert report['latency_ms']['end_to_end_ms']['count'] == 5
    assert report['inter_token_ms']['count'] == 4
    assert report['completed_tokens_per_second'] == 9 / report['elapsed_seconds']


async def test_benchmark_rejections_are_not_success_latency_samples():
    report = await benchmark('continuous', RuntimeConfig(kv_cache_capacity_tokens=2),
                             requests=3, concurrency=2, prompt='x', output_pattern=[2], warmup=0)
    assert report['rejected'] == 3
    assert report['completed'] == 0
    assert report['latency_ms']['queue_wait_ms']['count'] == 0
    assert report['completed_tokens_per_second'] == 0


async def test_benchmark_timeout_cleans_up_and_is_not_success():
    report = await benchmark('continuous', RuntimeConfig(prefill_delay=1), requests=1,
                             concurrency=1, prompt='x', output_pattern=[1], warmup=0, timeout=0.01)
    assert report['timeouts'] == 1
    assert report['completed'] == 0
    assert report['requests'][0]['timings']['time_to_first_token_ms'] is None


@pytest.mark.parametrize('arguments,exit_code', [
    (['--scheduler', 'continuous', '--requests', '1', '--warmup', '0',
      '--kv-cache-capacity-tokens', '1'], 1),
    (['--requests', '0'], 2),
])
def test_cli_failure_exit_codes(arguments, exit_code):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, '-m', 'llm_runtime.benchmark', *arguments],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == exit_code
    if exit_code == 1:
        report = json.loads(result.stdout)
        assert report['runs'][0]['rejected'] == 1
    else:
        assert 'invalid benchmark size' in result.stderr
