import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_runtime.metrics import RequestTimings


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=16, ge=1, le=512, strict=True)


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["token", "done", "error"]
    data: dict[str, Any]


RequestPhase = Literal["waiting", "prefill", "decode", "finished", "cancelled", "failed"]


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int
    request_id: str = field(default_factory=lambda: uuid4().hex)
    # At most max_tokens token events plus one terminal event are produced.
    # A slow reader cannot block the single scheduler worker.
    events: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue)
    abandoned: bool = False
    cancel_requested: bool = False
    phase: RequestPhase = "waiting"
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    timings: RequestTimings = field(default_factory=RequestTimings)
    prompt_token_ids: tuple[int, ...] | None = None
    finish_reason: Literal["length", "eos"] = "length"

    @property
    def prompt_token_count(self) -> int:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        return max(1, len(self.prompt.split()))


@dataclass(frozen=True)
class DecodeResult:
    """A real model token; text may be empty for partial bytes or EOS."""

    text: str
    token_id: int
    finish_reason: Literal["eos"] | None = None


@dataclass(frozen=True)
class ExecutionStep:
    """One request's work in an iteration; prefill counts tokens, decode costs 1."""

    request: InferenceRequest
    phase: Literal["prefill", "decode"]
    offset: int
    token_count: int
