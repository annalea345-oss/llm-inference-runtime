"""Monotonic scheduler timings; these do not measure delivery to HTTP clients."""

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class RequestTimings:
    clock: Callable[[], float] = field(default=perf_counter, repr=False)
    submitted_at: float = field(init=False)
    admitted_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None

    def __post_init__(self) -> None:
        self.submitted_at = self.clock()

    def admit(self) -> None:
        if self.admitted_at is None:
            self.admitted_at = self.clock()

    def token(self) -> None:
        self.token_times.append(self.clock())

    def finish(self) -> None:
        if self.finished_at is None:
            self.finished_at = self.clock()

    def snapshot(self) -> dict[str, float | list[float] | None]:
        return {
            'queue_wait_ms': ((self.admitted_at - self.submitted_at) * 1000
                              if self.admitted_at is not None else None),
            'time_to_first_token_ms': ((self.token_times[0] - self.submitted_at) * 1000
                                       if self.token_times else None),
            'inter_token_ms': [(b - a) * 1000
                               for a, b in zip(self.token_times, self.token_times[1:])],
            'end_to_end_ms': ((self.finished_at - self.submitted_at) * 1000
                              if self.finished_at is not None else None),
        }
