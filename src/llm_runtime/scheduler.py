import asyncio
import logging
from typing import Optional

from llm_runtime.executor import LegacyModelExecutor
from llm_runtime.protocol import InferenceRequest, StreamEvent

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    pass


class RuntimeUnavailableError(Exception):
    pass


class FCFSScheduler:
    """One worker owns execution; every admitted request runs to completion."""

    def __init__(self, executor: LegacyModelExecutor, queue_capacity: int) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self.executor = executor
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(queue_capacity)
        self._worker: Optional[asyncio.Task[None]] = None
        self.active: Optional[InferenceRequest] = None

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        if not self.running:
            self._worker = asyncio.create_task(self._run(), name="fcfs-worker")

    def submit(self, prompt: str, max_tokens: int) -> InferenceRequest:
        if not self.running:
            raise RuntimeUnavailableError("runtime is not running")
        request = InferenceRequest(prompt=prompt, max_tokens=max_tokens)
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            raise QueueFullError("waiting queue is full") from exc
        return request

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
        while not self.queue.empty():
            request = self.queue.get_nowait()
            self._emit_error(request, "runtime is shutting down")
            self.queue.task_done()

    @staticmethod
    def _emit_error(request: InferenceRequest, message: str) -> None:
        request.timings.finish()
        request.events.put_nowait(StreamEvent("error", {
            "request_id": request.request_id, "message": message,
        }))

    async def _run(self) -> None:
        while True:
            request = await self.queue.get()
            self.active = request
            try:
                if request.abandoned:
                    continue
                request.timings.admit()
                count = 0
                async for token in self.executor.generate(request):
                    if request.abandoned:
                        break
                    request.timings.token()
                    request.events.put_nowait(StreamEvent("token", {
                        "request_id": request.request_id,
                        "index": count,
                        "text": token,
                    }))
                    count += 1
                request.timings.finish()
                request.events.put_nowait(StreamEvent("done", {
                    "request_id": request.request_id,
                    "generated_tokens": count,
                    "finish_reason": "length",
                    "timings": request.timings.snapshot(),
                }))
            except asyncio.CancelledError:
                self._emit_error(request, "runtime is shutting down")
                raise
            except Exception:
                logger.exception("Execution failed for %s", request.request_id)
                self._emit_error(request, "model execution failed")
            finally:
                request.timings.finish()
                self.active = None
                self.queue.task_done()
