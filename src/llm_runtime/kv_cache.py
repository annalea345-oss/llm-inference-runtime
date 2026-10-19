"""Token-slot accounting only: no tensors, paging, or physical allocation."""

from dataclasses import dataclass, replace


class RequestTooLargeError(ValueError):
    """A request cannot finish even with the entire cache available."""


@dataclass(frozen=True)
class CacheAllocation:
    reserved_tokens: int
    used_tokens: int = 0


class SimulatedKVCache:
    """Synchronous accounting owned by the scheduler's event-loop task."""

    def __init__(self, capacity_tokens: int) -> None:
        self._positive_integer(capacity_tokens)
        self._capacity_tokens = capacity_tokens
        self._allocations: dict[str, CacheAllocation] = {}

    @staticmethod
    def _positive_integer(value: int) -> None:
        if type(value) is not int or value < 1:
            raise ValueError("token count must be a positive integer")

    @property
    def capacity_tokens(self) -> int:
        return self._capacity_tokens

    @property
    def reserved_tokens(self) -> int:
        return sum(a.reserved_tokens for a in self._allocations.values())

    @property
    def used_tokens(self) -> int:
        return sum(a.used_tokens for a in self._allocations.values())

    @property
    def available_tokens(self) -> int:
        """Capacity available for new reservations, not capacity minus usage."""
        return self.capacity_tokens - self.reserved_tokens

    @property
    def allocations(self) -> dict[str, CacheAllocation]:
        return dict(self._allocations)

    def reserve(self, request_id: str, token_count: int) -> bool:
        self._positive_integer(token_count)
        if request_id in self._allocations:
            raise ValueError("request already has a cache reservation")
        if token_count > self.capacity_tokens:
            raise RequestTooLargeError("request exceeds KV-cache token capacity")
        if token_count > self.available_tokens:
            return False
        self._allocations[request_id] = CacheAllocation(token_count)
        return True

    def consume(self, request_id: str, token_count: int) -> None:
        self._positive_integer(token_count)
        allocation = self._allocations[request_id]
        used = allocation.used_tokens + token_count
        if used > allocation.reserved_tokens:
            raise ValueError("cache usage exceeds request reservation")
        self._allocations[request_id] = replace(allocation, used_tokens=used)

    def release(self, request_id: str) -> None:
        self._allocations.pop(request_id, None)
