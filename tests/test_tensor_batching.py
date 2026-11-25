"""Offline real tensors: ragged batches must match independent greedy models."""

import asyncio
import threading
import weakref

import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')

from test_local_executor import components, prefill
from test_worker import entered
from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.local_executor import LocalModelExecutor
from llm_runtime.protocol import ExecutionStep, InferenceRequest
from llm_runtime.worker import SerialWorker


async def collect(request):
    events = []
    while True:
        event = await asyncio.wait_for(request.events.get(), 5)
        events.append(event)
        if event.kind in ('done', 'error'):
            return events


def make(architecture='llama', eos=None):
    model, tokenizer, torch = components(architecture=architecture, eos=eos)
    return LocalModelExecutor(model, tokenizer, torch, SerialWorker(), tensor_batching=True)


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
@pytest.mark.parametrize('budget', [1, 5, 16, 32])
@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
async def test_ragged_continuous_batches_match_individual_generate(architecture, budget, policy):
    executor = make(architecture)
    prompts = {'short': (1, 4), 'middle': (1, 3, 4, 5, 8), 'late': (1, 4, 9, 8, 7, 6, 5, 4, 2)}
    executor.tokenizer.apply_chat_template = lambda messages, **kw: prompts[messages[-1]['content']]
    expected = []
    with executor.torch.inference_mode():
        for ids, cap in zip(prompts.values(), [1, 4, 7]):
            expected.append(executor.model.generate(
                executor.torch.tensor([ids]), attention_mask=executor.torch.ones(1, len(ids), dtype=executor.torch.long),
                max_new_tokens=cap, do_sample=False, repetition_penalty=1.0,
            )[0, len(ids):].tolist())
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(
        max_active_requests=2, token_budget=budget, scheduling_policy=policy, prefill_token_budget=3))
    await scheduler.start()
    try:
        requests = [scheduler.submit(prompt, cap) for prompt, cap in zip(prompts, [1, 4, 7])]
        streams = await asyncio.gather(*(collect(r) for r in requests))
        await asyncio.wait_for(scheduler.queue.join(), 5)
        for events, ids in zip(streams, expected):
            assert [e.data['token_id'] for e in events[:-1]] == ids
            assert ''.join(e.data['text'] for e in events[:-1]) == executor.tokenizer.decode(ids)
            assert events[-1].data['finish_reason'] == 'length'
        if budget >= 5:
            assert executor.info['batched_forward_calls'] > 0
            assert executor.info['max_forward_batch_size'] == 2
        assert executor.active_sequences == executor.cache_bytes == 0
        assert scheduler.kv_cache.used_tokens == scheduler.kv_cache.reserved_tokens == 0
        assert not scheduler._requests
    finally:
        await scheduler.stop()
        await executor.aclose()


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
async def test_padding_gaps_removed_and_each_row_owns_only_its_tensors(architecture):
    executor = make(architecture)
    requests = [InferenceRequest('a', 4), InferenceRequest('b', 4)]
    requests[0].prompt_token_ids = (1, 4, 5, 6, 7)
    requests[1].prompt_token_ids = (1, 3, 6, 4)
    try:
        assert await executor.execute_batch(tuple(ExecutionStep(r, 'prefill', 0, n)
                                                  for r, n in zip(requests, [1, 3]))) == [None, None]
        assert await executor.execute_batch(tuple(ExecutionStep(r, 'prefill', off, n)
                                                  for r, off, n in zip(requests, [1, 3], [4, 1]))) == [None, None]
        states = [executor._states[r.request_id] for r in requests]
        caches = [s.cache for s in states]
        for layer in range(2):
            for kind in range(2):
                tensors = [cache[layer][kind] for cache in caches]
                assert tensors[0].untyped_storage().data_ptr() != tensors[1].untyped_storage().data_ptr()
                for t, length in zip(tensors, [5, 4]):
                    assert t.shape[-2] == length
                    assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
        # Removing one request cannot retain or change its peer's cache.
        peer = caches[1][0][0].clone()
        executor.release_request(requests[0].request_id)
        executor.torch.testing.assert_close(caches[1][0][0], peer)
        assert executor.active_sequences == 1
        result = await executor.execute_batch((ExecutionStep(requests[1], 'decode', 0, 1),))
        assert not isinstance(result[0], Exception)
        assert executor.info['max_forward_batch_size'] == 2
    finally:
        await executor.aclose()


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
async def test_ragged_decode_rows_share_a_forward_and_match_reference(architecture):
    executor = make(architecture)
    requests = [InferenceRequest('a', 1), InferenceRequest('b', 5), InferenceRequest('c', 7)]
    for r, ids in zip(requests, [(1, 4), (1, 5, 4, 8, 3), (1, 7, 8, 9, 3, 4, 5, 6)]):
        r.prompt_token_ids = ids
    expected = []
    with executor.torch.inference_mode():
        for r in requests:
            expected.append(executor.model.generate(executor.torch.tensor([r.prompt_token_ids]),
                attention_mask=executor.torch.ones(1, r.prompt_token_count, dtype=executor.torch.long),
                max_new_tokens=r.max_tokens, do_sample=False, repetition_penalty=1.0)[0, r.prompt_token_count:].tolist())
    original = executor.model.forward
    shapes = []
    def observe(**kwargs):
        shapes.append(tuple(kwargs['input_ids'].shape))
        return original(**kwargs)
    executor.model.forward = observe
    actual = {r.request_id: [] for r in requests}
    try:
        await executor.execute_batch(tuple(ExecutionStep(r, 'prefill', 0, r.prompt_token_count) for r in requests))
        active = requests[:]
        for offset in range(7):
            results = await executor.execute_batch(tuple(ExecutionStep(r, 'decode', offset, 1) for r in active))
            for r, result in zip(active, results):
                actual[r.request_id].append(result.token_id)
            finished = [r for r in active if r.max_tokens == offset + 1]
            for r in finished:
                executor.release_request(r.request_id)
                active.remove(r)
        assert [actual[r.request_id] for r in requests] == expected
        assert (2, 1) in shapes  # Genuine multi-request tensor decode, not just prefill.
        assert executor.cache_bytes == executor.active_sequences == 0
    finally:
        await executor.aclose()


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
async def test_new_prefill_row_can_join_a_partially_cached_prefill(architecture):
    executor = make(architecture)
    a, b = InferenceRequest('a', 4), InferenceRequest('b', 4)
    a.prompt_token_ids, b.prompt_token_ids = (1, 4, 5, 6, 7, 8), (1, 3, 8)
    expected = []
    with executor.torch.inference_mode():
        for r in (a, b):
            expected.append(executor.model.generate(executor.torch.tensor([r.prompt_token_ids]),
                attention_mask=executor.torch.ones(1, r.prompt_token_count, dtype=executor.torch.long),
                max_new_tokens=4, do_sample=False, repetition_penalty=1.0)[0, r.prompt_token_count:].tolist())
    try:
        await executor.execute_batch((ExecutionStep(a, 'prefill', 0, 2),))
        results = await executor.execute_batch((ExecutionStep(a, 'prefill', 2, 4), ExecutionStep(b, 'prefill', 0, 3)))
        assert results == [None, None]
        actual = [[], []]
        for offset in range(4):
            rows = await executor.execute_batch(tuple(ExecutionStep(r, 'decode', offset, 1) for r in (a, b)))
            for ids, result in zip(actual, rows):
                ids.append(result.token_id)
        assert actual == expected
    finally:
        await executor.aclose()


@pytest.mark.parametrize('case', ['architecture', 'sliding', 'flag'])
async def test_unsupported_batching_configuration_fails_explicitly(case):
    model, tokenizer, torch = components()
    worker = SerialWorker()
    if case == 'architecture':
        model.config.model_type = 'unsupported'
    if case == 'sliding':
        model.config.layer_types = ['sliding_attention', 'full_attention']
    try:
        with pytest.raises(ValueError):
            LocalModelExecutor(model, tokenizer, torch, worker, tensor_batching=1 if case == 'flag' else True)
    finally:
        await worker.aclose()


async def test_finished_rows_are_excluded_from_decode_forward():
    executor = make(eos=0)
    executor.eos_ids = {0, 2}
    requests = [InferenceRequest('eos', 8), InferenceRequest('cap', 1), InferenceRequest('live', 4)]
    for r in requests:
        r.prompt_token_ids = (1, 4)
    try:
        await executor.execute_batch(tuple(ExecutionStep(r, 'prefill', 0, 2) for r in requests))
        for r, token in zip(requests, [2, 5, 7]):
            executor._states[r.request_id].next_token = token
        calls = executor.info['forward_calls']
        results = await executor.execute_batch(tuple(ExecutionStep(r, 'decode', 0, 1) for r in requests))
        assert [r.token_id for r in results] == [2, 5, 7]
        assert results[0].finish_reason == 'eos'
        assert executor.info['forward_calls'] == calls + 1  # Only the live singleton advances.
        assert [executor._states[r.request_id].cache.get_seq_length() for r in requests] == [2, 2, 3]
    finally:
        await executor.aclose()


async def test_batch_failure_does_not_retain_tensor_frames_and_executor_recovers():
    executor = make()
    original = executor.model.forward
    refs = []
    def fail(**kwargs):
        output = original(**kwargs)
        refs.extend(weakref.ref(k) for k, _ in output.past_key_values)
        raise RuntimeError('injected batched forward failure')
    executor.model.forward = fail
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=2))
    await scheduler.start()
    try:
        requests = [scheduler.submit('a', 3), scheduler.submit('b', 3)]
        results = await asyncio.gather(*(collect(r) for r in requests))
        assert all(events[-1].kind == 'error' for events in results)
        await scheduler.queue.join()
        assert executor.cache_bytes == executor.active_sequences == 0
        assert all(ref() is None for ref in refs)
        assert not scheduler.kv_cache.allocations
        executor.model.forward = original
        assert (await collect(scheduler.submit('recovery', 2)))[-1].kind == 'done'
    finally:
        await scheduler.stop()
        await executor.aclose()


@pytest.mark.parametrize('shutdown', [False, True])
async def test_cancellation_and_repeated_shutdown_wait_for_shared_forward(shutdown):
    executor = make()
    original = executor.model.forward
    started, release = threading.Event(), threading.Event()
    def block(**kwargs):
        if kwargs['input_ids'].shape[0] == 2 and not started.is_set():
            started.set(); release.wait()
        return original(**kwargs)
    executor.model.forward = block
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=2))
    await scheduler.start()
    stop = None
    try:
        first, peer = scheduler.submit('first', 4), scheduler.submit('peer', 4)
        await entered(started)
        waiting = scheduler.submit('waiting', 4)
        scheduler.cancel(waiting.request_id)
        assert (await collect(waiting))[-1].data['finish_reason'] == 'cancelled'
        assert scheduler.cancel(first.request_id)
        if shutdown:
            stop = asyncio.create_task(scheduler.stop())
            for _ in range(3):
                await asyncio.sleep(0); stop.cancel(); await asyncio.sleep(0)
                assert not stop.done()
        assert executor.active_sequences == 2
        assert scheduler.kv_cache.reserved_tokens > 0
        release.set()
        if shutdown:
            # Scheduler.stop deliberately completes cleanup despite caller cancellation.
            await asyncio.wait_for(stop, 5)
        first_events, peer_events = await asyncio.gather(collect(first), collect(peer))
        assert first_events[-1].data['finish_reason'] == 'cancelled'
        assert peer_events[-1].kind == ('error' if shutdown else 'done')
        await asyncio.wait_for(scheduler.queue.join(), 5)
        assert not scheduler._requests
        assert executor.active_sequences == executor.cache_bytes == 0
        assert not scheduler.kv_cache.allocations
    finally:
        release.set()
        await scheduler.stop()
        await executor.aclose()


async def test_invalid_row_is_isolated_and_duplicate_request_rejected_before_execution():
    executor = make()
    a, b = InferenceRequest('a', 3), InferenceRequest('b', 3)
    a.prompt_token_ids = b.prompt_token_ids = (1, 4)
    try:
        results = await executor.execute_batch((ExecutionStep(a, 'prefill', 1, 1), ExecutionStep(b, 'prefill', 0, 2)))
        assert isinstance(results[0], ValueError) and results[1] is None
        assert executor._states[b.request_id].prefilled == 2
        calls = executor.info['forward_calls']
        with pytest.raises(ValueError, match='duplicate'):
            await executor.execute_batch((ExecutionStep(b, 'decode', 0, 1),) * 2)
        assert executor.info['forward_calls'] == calls
    finally:
        await executor.aclose()
