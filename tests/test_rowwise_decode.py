"""CPU projection specialization must keep weights, fallbacks, and ownership."""

import copy
import weakref

import pytest

from llm_runtime.local import create_local_app
from llm_runtime.local_executor import LocalModelExecutor


@pytest.mark.parametrize('value,batching', [('2','1'), ('true','1'), ('','1'), ('1','0')])
def test_factory_rejects_invalid_rowwise_settings(monkeypatch, value, batching):
    monkeypatch.setenv('LLM_MODEL_PATH', '/not-loaded')
    monkeypatch.setenv('LLM_TENSOR_BATCHING', batching)
    monkeypatch.setenv('LLM_ATTENTION_IMPLEMENTATION', 'eager')
    monkeypatch.setenv('LLM_ROWWISE_DECODE', value)
    with pytest.raises(ValueError, match='LLM_ROWWISE_DECODE'):
        create_local_app()


@pytest.mark.parametrize('value,batching', [(1,True), (None,True), ('1',True), (True,False)])
async def test_loader_rejects_invalid_rowwise_before_worker(monkeypatch, value, batching):
    monkeypatch.setattr('llm_runtime.local_executor.SerialWorker', lambda: pytest.fail('worker created'))
    with pytest.raises(ValueError, match='rowwise_decode'):
        await LocalModelExecutor.load('/not-loaded', tensor_batching=batching, rowwise_decode=value)


@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
async def test_factory_passes_rowwise_selection(monkeypatch, attention):
    from test_tensor_batch_protocol import BatchExecutor
    monkeypatch.setenv('LLM_MODEL_PATH', '/test-model')
    monkeypatch.setenv('LLM_TENSOR_BATCHING', '1')
    monkeypatch.setenv('LLM_ATTENTION_IMPLEMENTATION', attention)
    monkeypatch.setenv('LLM_ROWWISE_DECODE', '1')
    seen = []
    async def load(path, **kwargs):
        seen.append(kwargs)
        return BatchExecutor()
    monkeypatch.setattr(LocalModelExecutor, 'load', load)
    app = create_local_app()
    async with app.router.lifespan_context(app):
        assert seen == [dict(tensor_batching=True, attention_implementation=attention, rowwise_decode=True)]


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('shape,enabled,training,grad,dtype,expected', [
    ((2,1,8), True,False,False,'float32',1),
    ((1,1,8), True,False,False,'float32',0),
    ((3,1,8), True,False,False,'float32',0),
    ((2,3,8), True,False,False,'float32',0),
    ((2,8), True,False,False,'float32',0),
    ((2,1,8), False,False,False,'float32',0),
    ((2,1,8), True,True,False,'float32',0),
    ((2,1,8), True,False,True,'float32',0),
    ((2,1,8), True,False,False,'float64',0),
])
def test_linear_specialization_and_fallbacks(shape, enabled, training, grad, dtype, expected, bias):
    torch = pytest.importorskip('torch')
    from llm_runtime.cpu_linear import RowwiseDecodeLinear
    original = torch.nn.Linear(8, 5, bias=bias, dtype=getattr(torch,dtype)).train(training)
    layer = RowwiseDecodeLinear(original)
    layer.enabled = enabled
    assert layer.weight is original.weight and layer.bias is original.bias
    assert list(layer.state_dict()) == list(original.state_dict())
    inputs = torch.randn(*shape, dtype=getattr(torch,dtype))
    with torch.set_grad_enabled(grad):
        torch.testing.assert_close(layer(inputs), original(inputs))
    assert layer.rowwise_calls == expected


@pytest.mark.parametrize('architecture', ['llama','qwen2'])
@pytest.mark.parametrize('attention', ['eager','sdpa'])
async def test_rowwise_model_ragged_tokens_match_independent_reference_and_unload(architecture, attention):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from test_local_executor import components
    from llm_runtime.protocol import ExecutionStep, InferenceRequest
    from llm_runtime.worker import SerialWorker
    model, tokenizer, torch = components(architecture=architecture)
    reference = copy.deepcopy(model)
    reference.set_attn_implementation('eager')
    expected = []
    prompts = [(1,4), (1,3,7,8,4)]
    with torch.inference_mode():
        for ids in prompts:
            expected.append(reference.generate(torch.tensor([ids]),
                attention_mask=torch.ones(1,len(ids),dtype=torch.long), max_new_tokens=5,
                do_sample=False, repetition_penalty=1.0)[0,len(ids):].tolist())
    before = {name: p for name,p in model.named_parameters()}
    model.set_attn_implementation(attention)
    executor = LocalModelExecutor(model, tokenizer, torch, SerialWorker(), tensor_batching=True, rowwise_decode=True)
    assert all(p is before[name] for name,p in model.named_parameters())
    request_pair = [InferenceRequest('a',5), InferenceRequest('b',5)]
    for r,ids in zip(request_pair,prompts): r.prompt_token_ids=ids
    layer_ref = weakref.ref(executor._rowwise_layers[0])
    try:
        await executor.execute_batch(tuple(ExecutionStep(r,'prefill',0,len(ids)) for r,ids in zip(request_pair,prompts)))
        actual = [[],[]]
        for offset in range(5):
            results = await executor.execute_batch(tuple(ExecutionStep(r,'decode',offset,1) for r in request_pair))
            for ids,result in zip(actual,results): ids.append(result.token_id)
        assert actual == expected
        assert executor.info['rowwise_decode_linear_calls'] > 0
        assert executor.info['batched_forward_calls'] > 0
        for r in request_pair: executor.release_request(r.request_id)
        assert executor.cache_bytes == executor.active_sequences == 0
    finally:
        del model
        await executor.aclose()
    assert layer_ref() is None
    assert executor.info['rowwise_decode_linear_calls'] > 0
    await executor.aclose()
