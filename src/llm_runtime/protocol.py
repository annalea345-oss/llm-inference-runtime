import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=16, ge=1, le=512, strict=True)


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["token", "done", "error"]
    data: dict[str, Any]


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int
    request_id: str = field(default_factory=lambda: uuid4().hex)
    # At most max_tokens token events plus one terminal event are produced.
    # A slow reader cannot block the single scheduler worker.
    events: asyncio.Queue[StreamEvent] = field(default_factory=asyncio.Queue)
    abandoned: bool = False
