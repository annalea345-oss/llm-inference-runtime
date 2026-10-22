"""Optional, offline tests with a tiny randomly initialized real model."""

import asyncio
import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')

from llm_runtime.local_executor import LocalModelExecutor
from llm_runtime.protocol import ExecutionStep, InferenceRequest
from llm_runtime.worker import SerialWorker


def components(eos=None):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=eos, pad_token_id=0)
    model = LlamaForCausalLM(config).eval()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 4, 5, 6]

        def decode(self, ids, **kwargs):
            return ''.join(chr(65 + i) for i in ids if i != eos)

    return model, Tokenizer(), torch


async def prefill(executor, request, size):
    offset = 0
    while offset < request.prompt_token_count:
        count = min(size, request.prompt_token_count - offset)
        assert await executor.execute(ExecutionStep(request, 'prefill', offset, count)) is None
        offset += count


@pytest.mark.parametrize('chunk', [1, 2, 32])
async def test_cached_steps_match_reference_greedy_generation_and_release_real_tensors(chunk):
    model, tokenizer, torch = components()
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker())
    request = InferenceRequest('test', 4)
    request.prompt_token_ids = executor.tokenize(request.prompt)
    try:
        with torch.inference_mode():
            expected = model.generate(torch.tensor([request.prompt_token_ids]),
                                      attention_mask=torch.ones(1, 4, dtype=torch.long),
                                      max_new_tokens=4, do_sample=False, use_cache=True)[0, 4:].tolist()
        await prefill(executor, request, chunk)
        assert executor.active_sequences == 1
        assert executor.cache_bytes == 4 * 256
        outputs = [await executor.execute(ExecutionStep(request, 'decode', i, 1)) for i in range(4)]
        assert [o.token_id for o in outputs] == expected
        assert ''.join(o.text for o in outputs) == tokenizer.decode(expected)
        assert executor.cache_bytes == (4 + 3) * 256
        executor.release_request(request.request_id)
        assert executor.cache_bytes == executor.active_sequences == 0
    finally:
        await executor.aclose()


async def test_requests_have_independent_caches_and_eos_needs_no_extra_forward():
    model, tokenizer, torch = components(eos=0)
    with torch.no_grad():
        model.lm_head.weight.zero_()  # Greedy argmax deterministically chooses EOS 0.
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker())
    requests = [InferenceRequest('x', 8) for _ in range(2)]
    for request in requests:
        request.prompt_token_ids = executor.tokenize('x')
    try:
        await asyncio.gather(*(prefill(executor, r, 2) for r in requests))
        assert executor.active_sequences == 2
        assert executor.cache_bytes == 8 * 256
        output = await executor.execute(ExecutionStep(requests[0], 'decode', 0, 1))
        assert output.finish_reason == 'eos'
        assert output.text == ''
        executor.release_request(requests[0].request_id)
        assert executor.cache_bytes == 4 * 256
        assert (await executor.execute(ExecutionStep(requests[1], 'decode', 0, 1))).token_id == 0
    finally:
        await executor.aclose()
    assert executor.cache_bytes == executor.active_sequences == 0


async def test_partial_unicode_is_buffered_and_final_decode_flushes():
    model, tokenizer, torch = components()
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker())
    request = InferenceRequest('x', 2)
    request.prompt_token_ids = (1,)
    tokenizer.decode = lambda ids, **kwargs: '\ufffd' if len(ids) == 1 else 'é'
    try:
        await prefill(executor, request, 1)
        first = await executor.execute(ExecutionStep(request, 'decode', 0, 1))
        second = await executor.execute(ExecutionStep(request, 'decode', 1, 1))
        assert first.text == ''
        assert second.text == 'é'
    finally:
        await executor.aclose()


async def test_invalid_step_offsets_fail_and_cleanup_remains_available():
    model, tokenizer, torch = components()
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker())
    request = InferenceRequest('x', 2)
    request.prompt_token_ids = (1, 2)
    try:
        with pytest.raises(ValueError, match='prefill offset'):
            await executor.execute(ExecutionStep(request, 'prefill', 1, 1))
        await prefill(executor, request, 1)
        with pytest.raises(ValueError, match='decode step'):
            await executor.execute(ExecutionStep(request, 'decode', 1, 1))
        executor.release_request(request.request_id)
        assert executor.cache_bytes == 0
    finally:
        await executor.aclose()
