"""Serialize blocking model work without releasing ownership on cancellation."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, TypeVar


T = TypeVar('T')


async def _settle(future: asyncio.Future[T]) -> T:
    """Defer caller cancellation until work settles, including repeated cancels."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
            if not future.done():
                continue
            # Retrieve any worker failure even when cancellation wins the race.
            if not future.cancelled():
                future.exception()
            raise
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        else:
            if cancelled:
                raise asyncio.CancelledError
            return result


class SerialWorker:
    """One dedicated CPU thread; use from a single asyncio event loop.

    Cancelling a caller cannot stop a running Python function. Instead, retain
    its ownership until that function finishes so scheduler cleanup is safe.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='llm-model')
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if self._closed:
            raise RuntimeError('Model worker is closed')
        # No await separates the closed check and submission: aclose cannot
        # interleave with admission on the owning event loop.
        submitted = self._executor.submit(partial(fn, *args, **kwargs))
        future = asyncio.wrap_future(submitted)
        return await _settle(future)

    async def aclose(self) -> None:
        """Reject new work, drain submitted work, and join the thread off-loop."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                asyncio.to_thread(self._executor.shutdown, wait=True),
            )
        await _settle(self._close_task)
