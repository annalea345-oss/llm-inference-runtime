import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from llm_runtime.config import RuntimeConfig
from llm_runtime.protocol import InferenceRequest


class ModelExecutor(Protocol):
    def generate(self, request: InferenceRequest) -> AsyncIterator[str]: ...


class SimulatedExecutor:
    """Deterministic word pieces, not a tokenizer or a language model."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    async def generate(self, request: InferenceRequest) -> AsyncIterator[str]:
        await asyncio.sleep(self.config.prefill_delay)
        words = request.prompt.split() or ["<empty>"]
        for index in range(request.max_tokens):
            if request.abandoned:
                return
            await asyncio.sleep(self.config.token_delay)
            yield words[index % len(words)] + " "
