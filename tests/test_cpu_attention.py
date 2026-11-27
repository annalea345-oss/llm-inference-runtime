"""Optional SDPA must preserve greedy fixtures and explicit startup settings."""

import asyncio
import copy

import pytest

from llm_runtime.local import create_local_app
from llm_runtime.local_executor import LocalModelExecutor


@pytest.mark.parametrize('attention', ['flash_attention_2', 'SDPA', '', 'auto'])
def test_factory_rejects_unknown_attention_before_loading(monkeypatch, attention):
    monkeypatch.setenv('LLM_MODEL_PATH', '/not-loaded')
    monkeypatch.setenv('LLM_TENSOR_BATCHING', '0')
    monkeypatch.setenv('LLM_ATTENTION_IMPLEMENTATION', attention)
    with pytest.raises(ValueError, match='LLM_ATTENTION_IMPLEMENTATION'):
        create_local_app()


@pytest.mark.parametrize('attention', [None, 0, '', 'flash_attention_2'])
async def test_loader_rejects_attention_before_worker_creation(monkeypatch, attention):
    monkeypatch.setattr('llm_runtime.local_executor.SerialWorker', lambda: pytest.fail('worker should not be created'))
    with pytest.raises(ValueError, match='attention_implementation'):
        await LocalModelExecutor.load('/not-loaded', attention_implementation=attention)


async def test_loader_rejects_silent_attention_fallback_and_closes_worker(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from types import SimpleNamespace
    from llm_runtime.worker import SerialWorker
    class Worker(SerialWorker):
        closed = False
        async def aclose(self):
            await super().aclose()
            self.closed = True
    worker = Worker()
    monkeypatch.setattr('llm_runtime.local_executor.SerialWorker', lambda: worker)
    class Model:
        config = SimpleNamespace(_attn_implementation='eager')
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: Model())
    with pytest.raises(RuntimeError, match='did not select'):
        await LocalModelExecutor.load(str(tmp_path), attention_implementation='sdpa')
    assert worker.closed


@pytest.mark.parametrize('attention', [None, 'eager', 'sdpa'])
@pytest.mark.parametrize('batched', [False, True])
async def test_factory_passes_selected_attention_and_batching(monkeypatch, attention, batched):
    from test_tensor_batch_protocol import BatchExecutor
    monkeypatch.delenv('LLM_ROWWISE_DECODE', raising=False)
    monkeypatch.setenv('LLM_MODEL_PATH', '/test-model')
    monkeypatch.setenv('LLM_TENSOR_BATCHING', str(int(batched)))
    if attention is None:
        monkeypatch.delenv('LLM_ATTENTION_IMPLEMENTATION', raising=False)
    else:
        monkeypatch.setenv('LLM_ATTENTION_IMPLEMENTATION', attention)
    seen = []
    async def load(path, **kwargs):
        seen.append((path, kwargs))
        return BatchExecutor()
    monkeypatch.setattr(LocalModelExecutor, 'load', load)
    app = create_local_app()
    async with app.router.lifespan_context(app):
        assert seen == [('/test-model', {'tensor_batching': batched,
                                      'attention_implementation': attention or 'eager', 'rowwise_decode': False})]


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('policy', ['round_robin', 'decode_first'])
async def test_sdpa_ragged_cached_tokens_match_independent_eager_reference(architecture, batched, policy):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from test_local_executor import components
    from test_tensor_batching import collect
    from llm_runtime.batching import ContinuousBatchScheduler
    from llm_runtime.config import RuntimeConfig
    from llm_runtime.worker import SerialWorker
    model, tokenizer, torch = components(architecture=architecture)
    reference = copy.deepcopy(model)
    reference.set_attn_implementation('eager')
    model.set_attn_implementation('sdpa')
    encodings = {'a': (1, 4), 'b': (1, 5, 7, 8, 3), 'c': (1, 6, 9, 4, 5, 8, 7, 6, 3)}
    tokenizer.apply_chat_template = lambda messages, **kw: encodings[messages[-1]['content']]
    expected = []
    with torch.inference_mode():
        for ids, cap in zip(encodings.values(), [1, 5, 7]):
            expected.append(reference.generate(torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=cap, do_sample=False, repetition_penalty=1.0)[0, len(ids):].tolist())
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker(), tensor_batching=batched)
    scheduler = ContinuousBatchScheduler(executor, RuntimeConfig(max_active_requests=3,
        token_budget=16, prefill_token_budget=5, scheduling_policy=policy))
    await scheduler.start()
    try:
        requests = [scheduler.submit(prompt, cap) for prompt, cap in zip(encodings, [1, 5, 7])]
        streams = await asyncio.gather(*(collect(r) for r in requests))
        for events, ids in zip(streams, expected):
            assert [e.data['token_id'] for e in events[:-1]] == ids
            assert ''.join(e.data['text'] for e in events[:-1]) == tokenizer.decode(ids)
        await scheduler.queue.join()
        assert executor.info['attention_implementation'] == 'sdpa'
        assert executor.active_sequences == executor.cache_bytes == 0
        assert not scheduler.kv_cache.allocations
    finally:
        await scheduler.stop()
        await executor.aclose()
    assert executor.info['attention_implementation'] == 'sdpa'
