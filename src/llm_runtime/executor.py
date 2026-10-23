import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from llm_runtime.config import RuntimeConfig
from llm_runtime.protocol import DecodeResult, ExecutionStep, InferenceRequest


class ModelExecutor(Protocol):
    async def execute(self, step: ExecutionStep) -> str | DecodeResult | None: ...


class LegacyModelExecutor(Protocol):
    """Run-to-completion interface retained for the FCFS baseline."""

    def generate(self, request: InferenceRequest) -> AsyncIterator[str]: ...


class SimulatedExecutor:
    """Deterministic word pieces, not a tokenizer or a language model."""

    model_id = "simulated"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @property
    def info(self) -> dict:
        return {"backend": "simulated", "tokenization": "whitespace"}

    async def execute(self, step: ExecutionStep) -> str | None:
        if step.phase == "prefill":
            # Chunk delays sum to the configured per-prompt prefill delay.
            await asyncio.sleep(
                self.config.prefill_delay * step.token_count / step.request.prompt_token_count
            )
            return None
        await asyncio.sleep(self.config.token_delay)
        words = step.request.prompt.split() or ["<empty>"]
        return words[step.offset % len(words)] + " "

    async def generate(self, request: InferenceRequest) -> AsyncIterator[str]:
        await asyncio.sleep(self.config.prefill_delay)
        words = request.prompt.split() or ["<empty>"]
        for index in range(request.max_tokens):
            if request.abandoned:
                return
            await asyncio.sleep(self.config.token_delay)
            yield words[index % len(words)] + " "
