"""Opt-in server: LLM_MODEL_PATH=.models/smollm2 uvicorn llm_runtime.local:create_local_app --factory."""

import os

from llm_runtime.api import create_app
from llm_runtime.config import RuntimeConfig


def create_local_app():
    path = os.environ.get('LLM_MODEL_PATH')
    if not path:
        raise ValueError('set LLM_MODEL_PATH to a downloaded local model directory')
    batching = os.environ.get('LLM_TENSOR_BATCHING', '0')
    if batching not in ('0', '1'):
        raise ValueError('LLM_TENSOR_BATCHING must be 0 or 1')
    attention = os.environ.get('LLM_ATTENTION_IMPLEMENTATION', 'eager')
    if attention not in ('eager', 'sdpa'):
        raise ValueError('LLM_ATTENTION_IMPLEMENTATION must be eager or sdpa')
    rowwise = os.environ.get('LLM_ROWWISE_DECODE', '0')
    if rowwise not in ('0', '1') or (rowwise == '1' and batching != '1'):
        raise ValueError('LLM_ROWWISE_DECODE must be 0 or 1 and requires LLM_TENSOR_BATCHING=1')

    async def load():
        from llm_runtime.local_executor import LocalModelExecutor
        return await LocalModelExecutor.load(path, tensor_batching=batching == '1',
                                             attention_implementation=attention,
                                             rowwise_decode=rowwise == '1')

    return create_app(RuntimeConfig(max_active_requests=2), executor_loader=load)
