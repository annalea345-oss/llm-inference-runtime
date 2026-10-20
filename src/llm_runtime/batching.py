"""Bounded continuous batching with a shared prefill/decode token budget."""

import asyncio
import logging
from collections import deque

from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import ModelExecutor
from llm_runtime.kv_cache import RequestTooLargeError, SimulatedKVCache
from llm_runtime.protocol import ExecutionStep, InferenceRequest, RequestPhase, StreamEvent
from llm_runtime.scheduler import QueueFullError, RuntimeUnavailableError

logger = logging.getLogger(__name__)


class ContinuousBatchScheduler:
    """One owner plans iterations; independent simulated steps execute together.

    Active requests rotate after selection, including across budget exhaustion.
    Each gets at most one step per iteration. Admission happens only at iteration
    boundaries, so live work is bounded by max_active_requests + queue_capacity.
    """

    def __init__(self, executor: ModelExecutor, config: RuntimeConfig) -> None:
        self.executor = executor
        self.config = config
        self.queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(config.queue_capacity)
        self.kv_cache = SimulatedKVCache(config.kv_cache_capacity_tokens)
        # A dequeued FIFO head can wait for memory without losing its queue slot.
        self._pending: InferenceRequest | None = None
        self._active: deque[InferenceRequest] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._stopping = False

    @property
    def queued_requests(self) -> int:
        return self.queue.qsize() + (self._pending is not None)

    @property
    def active(self) -> tuple[InferenceRequest, ...]:
        return tuple(self._active)

    @property
    def running(self) -> bool:
        return not self._stopping and self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if not self.running:
                self._worker = asyncio.create_task(self._run(), name="continuous-batch-worker")

    def submit(self, prompt: str, max_tokens: int) -> InferenceRequest:
        if not self.running:
            raise RuntimeUnavailableError("runtime is not running")
        if not prompt or type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("invalid generation request")
        request = InferenceRequest(prompt, max_tokens)
        if request.prompt_token_count + max_tokens > self.kv_cache.capacity_tokens:
            raise RequestTooLargeError("request exceeds KV-cache token capacity")
        if self.queued_requests >= self.config.queue_capacity:
            raise QueueFullError("waiting queue is full")
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as exc:
            raise QueueFullError("waiting queue is full") from exc
        return request

    async def stop(self) -> None:
        # Keep restart behind both worker termination and shared-state cleanup.
        async with self._lifecycle_lock:
            self._stopping = True
            try:
                if self._worker is not None:
                    self._worker.cancel()
                    try:
                        await self._worker
                    except asyncio.CancelledError:
                        pass
            finally:
                self._worker = None
                # A task cancelled before its first execution never enters _run.finally.
                self._drain()
                self._stopping = False

    def _drain(self) -> None:
        while self._active:
            self._finish(self._active.popleft(), "failed", "runtime is shutting down")
        if self._pending is not None:
            self._finish(self._pending, "failed", "runtime is shutting down")
            self._pending = None
        while not self.queue.empty():
            self._finish(self.queue.get_nowait(), "failed", "runtime is shutting down")

    def _finish(self, request: InferenceRequest, phase: RequestPhase, message: str = "") -> None:
        if request.phase in ("finished", "cancelled", "failed"):
            return
        request.timings.finish()
        self.kv_cache.release(request.request_id)
        request.phase = phase
        if not request.abandoned:
            if phase == "finished":
                event = StreamEvent("done", {
                    "request_id": request.request_id,
                    "generated_tokens": request.generated_tokens,
                    "finish_reason": "length",
                    "timings": request.timings.snapshot(),
                })
            else:
                event = StreamEvent("error", {
                    "request_id": request.request_id, "message": message,
                })
            request.events.put_nowait(event)
        self.queue.task_done()

    def _admit(self, request: InferenceRequest) -> bool:
        if request.abandoned:
            self._finish(request, "cancelled")
        else:
            if not self.kv_cache.reserve(
                request.request_id, request.prompt_token_count + request.max_tokens,
            ):
                return False
            request.timings.admit()
            request.phase = "prefill"
            self._active.append(request)
        return True

    def _plan(self) -> list[ExecutionStep]:
        budget = self.config.token_budget
        steps = []
        for _ in range(len(self._active)):
            if budget == 0:
                break
            request = self._active.popleft()
            self._active.append(request)
            if request.phase == "prefill":
                count = min(request.prompt_token_count - request.prefilled_tokens, budget)
                step = ExecutionStep(request, "prefill", request.prefilled_tokens, count)
            else:
                step = ExecutionStep(request, "decode", request.generated_tokens, 1)
            budget -= step.token_count
            steps.append(step)
        return steps

    async def _run(self) -> None:
        try:
            while True:
                for request in tuple(self._active):
                    if request.abandoned:
                        self._active.remove(request)
                        self._finish(request, "cancelled")
                if not self._active and not self.queued_requests:
                    self._pending = await self.queue.get()
                while len(self._active) < self.config.max_active_requests:
                    if self._pending is None:
                        if self.queue.empty():
                            break
                        self._pending = self.queue.get_nowait()
                    if not self._admit(self._pending):
                        break
                    self._pending = None
                steps = self._plan()
                results = await asyncio.gather(
                    *(self.executor.execute(step) for step in steps), return_exceptions=True,
                )
                for step, result in zip(steps, results):
                    request = step.request
                    if request.abandoned:
                        self._active.remove(request)
                        self._finish(request, "cancelled")
                    elif isinstance(result, BaseException):
                        logger.error("Execution failed for %s", request.request_id,
                                     exc_info=(type(result), result, result.__traceback__))
                        self._active.remove(request)
                        self._finish(request, "failed", "model execution failed")
                    elif step.phase == "prefill":
                        self.kv_cache.consume(request.request_id, step.token_count)
                        request.prefilled_tokens += step.token_count
                        if request.prefilled_tokens == request.prompt_token_count:
                            request.phase = "decode"
                    elif isinstance(result, str):
                        request.timings.token()
                        self.kv_cache.consume(request.request_id, 1)
                        request.events.put_nowait(StreamEvent("token", {
                            "request_id": request.request_id,
                            "index": request.generated_tokens, "text": result,
                        }))
                        request.generated_tokens += 1
                        if request.generated_tokens == request.max_tokens:
                            self._active.remove(request)
                            self._finish(request, "finished")
                    else:
                        self._active.remove(request)
                        self._finish(request, "failed", "model execution failed")
                # Even a zero-delay replacement executor must yield for HTTP I/O
                # and admission/cancellation between iterations.
                await asyncio.sleep(0)
        finally:
            self._drain()
