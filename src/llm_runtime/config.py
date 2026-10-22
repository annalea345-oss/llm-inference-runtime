from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RuntimeConfig:
    """Simulation settings; delays are seconds, capacity counts waiting requests."""

    queue_capacity: int = 64
    max_active_requests: int = 8
    token_budget: int = 32
    prefill_delay: float = 0.05
    token_delay: float = 0.03
    kv_cache_capacity_tokens: int = 8192
    scheduling_policy: Literal["round_robin", "decode_first"] = "round_robin"
    prefill_token_budget: int = 8

    def __post_init__(self) -> None:
        for name in ("queue_capacity", "max_active_requests", "token_budget",
                     "kv_cache_capacity_tokens", "prefill_token_budget"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.scheduling_policy not in ("round_robin", "decode_first"):
            raise ValueError("scheduling_policy must be round_robin or decode_first")
        if self.prefill_delay < 0 or self.token_delay < 0:
            raise ValueError("simulation delays must be nonnegative")
