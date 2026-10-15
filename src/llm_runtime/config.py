from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Simulation settings; delays are seconds, capacity counts waiting requests."""

    queue_capacity: int = 64
    prefill_delay: float = 0.05
    token_delay: float = 0.03

    def __post_init__(self) -> None:
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        if self.prefill_delay < 0 or self.token_delay < 0:
            raise ValueError("simulation delays must be nonnegative")
