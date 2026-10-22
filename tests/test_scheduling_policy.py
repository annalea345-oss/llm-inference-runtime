"""Policy behavior and lifecycle checks for decode-first continuous batching."""

import asyncio

import pytest

from llm_runtime.batching import ContinuousBatchScheduler
from llm_runtime.config import RuntimeConfig
from llm_runtime.executor import SimulatedExecutor
from llm_runtime.protocol import InferenceRequest


def policy_scheduler(**settings):
    config = RuntimeConfig(scheduling_policy="decode_first", prefill_delay=0,
                           token_delay=0, **settings)
    return ContinuousBatchScheduler(SimulatedExecutor(config), config)


def resident(scheduler, name, phase):
    request = InferenceRequest(f"{name} " * 100, 100)
    request.phase = phase
    if phase == "decode":
        request.prefilled_tokens = request.prompt_token_count
    scheduler._active.append(request)
    return request


def validate_plan(steps, budget, prefill_budget):
    assert steps
    assert sum(step.token_count for step in steps) <= budget
    assert sum(step.token_count for step in steps if step.phase == "prefill") <= prefill_budget
    assert len({step.request.request_id for step in steps}) == len(steps)
    assert all(step.token_count > 0 for step in steps)
    assert all(step.token_count == 1 for step in steps if step.phase == "decode")


def test_default_policy_preserves_round_robin_and_ignores_prefill_cap():
    config = RuntimeConfig(token_budget=5, prefill_token_budget=1)
    assert config.scheduling_policy == "round_robin"
    scheduler = ContinuousBatchScheduler(SimulatedExecutor(config), config)
    prefill = resident(scheduler, "prefill", "prefill")
    resident(scheduler, "decode", "decode")
    steps = scheduler._plan()
    assert [(step.request, step.token_count) for step in steps] == [(prefill, 5)]


def test_decode_first_protects_decode_then_spends_remaining_prefill_allowance():
    scheduler = policy_scheduler(token_budget=8, prefill_token_budget=3)
    prefill = resident(scheduler, "prefill", "prefill")
    decodes = [resident(scheduler, f"decode-{i}", "decode") for i in range(2)]
    steps = scheduler._plan()
    assert [step.request for step in steps] == decodes + [prefill]
    assert [step.token_count for step in steps] == [1, 1, 3]


def test_mixed_batch_reserves_prefill_progress_under_decode_pressure():
    scheduler = policy_scheduler(token_budget=3, prefill_token_budget=8)
    prefill = resident(scheduler, "prefill", "prefill")
    decodes = [resident(scheduler, f"decode-{i}", "decode") for i in range(5)]
    steps = scheduler._plan()
    assert [step.request for step in steps] == decodes[:2] + [prefill]
    assert [step.token_count for step in steps] == [1, 1, 1]


@pytest.mark.parametrize("budget", [1, 2, 3, 8])
def test_both_phases_rotate_without_starvation_with_new_arrivals(budget):
    scheduler = policy_scheduler(token_budget=budget, prefill_token_budget=2)
    prefills = [resident(scheduler, f"prefill-{i}", "prefill") for i in range(3)]
    decodes = [resident(scheduler, f"decode-{i}", "decode") for i in range(3)]
    seen = {"prefill": [], "decode": []}
    # Keep both phases resident so phase changes cannot mask starvation. Add new
    # work after each plan to check it cannot overtake older phase peers.
    for iteration in range(6):
        steps = scheduler._plan()
        validate_plan(steps, budget, 2)
        for step in steps:
            seen[step.phase].append(step.request)
        resident(scheduler, f"new-prefill-{iteration}", "prefill")
        resident(scheduler, f"new-decode-{iteration}", "decode")
    assert seen["prefill"][:3] == prefills
    assert seen["decode"][:3] == decodes


def test_one_token_budget_alternates_phases_starting_with_decode():
    scheduler = policy_scheduler(token_budget=1)
    resident(scheduler, "prefill", "prefill")
    resident(scheduler, "decode", "decode")
    assert [scheduler._plan()[0].phase for _ in range(6)] == ["decode", "prefill"] * 3


def test_one_token_phase_disappearance_keeps_pending_prefill_turn():
    scheduler = policy_scheduler(token_budget=1)
    prefill = resident(scheduler, "prefill", "prefill")
    decode = resident(scheduler, "decode", "decode")
    assert scheduler._plan()[0].request is decode
    scheduler._active.remove(decode)
    assert scheduler._plan()[0].request is prefill
    resident(scheduler, "new-decode", "decode")
    assert scheduler._plan()[0].request is prefill
    assert scheduler._plan()[0].phase == "decode"


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_single_phase_rotates_and_uses_its_available_budget(phase):
    scheduler = policy_scheduler(token_budget=4, prefill_token_budget=2)
    requests = [resident(scheduler, str(i), phase) for i in range(3)]
    first = scheduler._plan()
    second = scheduler._plan()
    validate_plan(first, 4, 2)
    validate_plan(second, 4, 2)
    if phase == "prefill":
        assert [(step.request, step.token_count) for step in first] == [(requests[0], 2)]
        assert [(step.request, step.token_count) for step in second] == [(requests[1], 2)]
    else:
        assert [step.request for step in first] == requests
        assert [step.request for step in second] == requests


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_invalid_prefill_budget(value):
    with pytest.raises(ValueError, match="prefill_token_budget"):
        RuntimeConfig(prefill_token_budget=value)


@pytest.mark.parametrize("value", ["", "priority", None, 1, True])
def test_invalid_scheduling_policy(value):
    with pytest.raises(ValueError, match="scheduling_policy"):
        RuntimeConfig(scheduling_policy=value)


async def collect(request):
    events = []
    while True:
        event = await asyncio.wait_for(request.events.get(), 2)
        events.append(event)
        if event.kind in ("done", "error"):
            return events


class CheckedScheduler(ContinuousBatchScheduler):
    """Observe allocation invariants at real iteration boundaries."""

    def __init__(self, executor, config):
        super().__init__(executor, config)
        self.plans = []

    def _plan(self):
        cache = self.kv_cache
        assert set(cache.allocations) == {request.request_id for request in self.active}
        assert 0 <= cache.used_tokens <= cache.reserved_tokens <= cache.capacity_tokens
        for request in self.active:
            assert cache.allocations[request.request_id].used_tokens == (
                request.prefilled_tokens + request.generated_tokens
            )
        steps = super()._plan()
        if steps:
            validate_plan(steps, self.config.token_budget, self.config.prefill_token_budget)
            self.plans.append(steps)
        return steps


@pytest.mark.parametrize("budget", [1, 2, 7])
async def test_decode_first_completes_mixed_workloads_with_exact_output_and_memory(budget):
    config = RuntimeConfig(scheduling_policy="decode_first", token_budget=budget,
                           prefill_token_budget=2, prefill_delay=0, token_delay=0,
                           kv_cache_capacity_tokens=19, max_active_requests=3)
    scheduler = CheckedScheduler(SimulatedExecutor(config), config)
    await scheduler.start()
    try:
        requests = [scheduler.submit(prompt, 4) for prompt in
                    ["one two three four five six", "short", "   ", "late arrival"]]
        streams = await asyncio.gather(*(collect(request) for request in requests))
        await asyncio.wait_for(scheduler.queue.join(), 2)
        for request, events in zip(requests, streams):
            words = request.prompt.split() or ["<empty>"]
            assert [event.kind for event in events] == ["token"] * 4 + ["done"]
            assert [event.data["text"] for event in events[:-1]] == [
                words[index % len(words)] + " " for index in range(4)
            ]
            assert [event.data["index"] for event in events[:-1]] == list(range(4))
            assert request.phase == "finished"
            assert request.prefilled_tokens == request.prompt_token_count
            assert request.generated_tokens == 4
        assert scheduler.plans
        assert not scheduler.kv_cache.allocations
        assert scheduler.kv_cache.used_tokens == scheduler.kv_cache.reserved_tokens == 0
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("outcome", ["cancel", "failure", "shutdown"])
async def test_decode_first_releases_memory_after_inflight_terminal_paths(phase, outcome):
    entered = asyncio.Event()
    release = asyncio.Event()
    config = RuntimeConfig(scheduling_policy="decode_first", token_budget=2,
                           prefill_token_budget=1, prefill_delay=0, token_delay=0,
                           max_active_requests=1, kv_cache_capacity_tokens=4)

    class GatedExecutor(SimulatedExecutor):
        async def execute(self, step):
            if step.request.prompt == "target" and step.phase == phase:
                entered.set()
                await release.wait()
                if outcome == "failure":
                    raise RuntimeError("private failure")
            return await super().execute(step)

    scheduler = CheckedScheduler(GatedExecutor(config), config)
    await scheduler.start()
    try:
        target = scheduler.submit("target", 3)
        await asyncio.wait_for(entered.wait(), 2)
        assert scheduler.kv_cache.reserved_tokens == 4
        waiting = scheduler.submit("good", 3)
        if outcome == "shutdown":
            await scheduler.stop()
            for request in (target, waiting):
                events = await collect(request)
                assert events[-1].kind == "error"
                assert events[-1].data["message"] == "runtime is shutting down"
        else:
            if outcome == "cancel":
                target.abandoned = True
            release.set()
            assert (await collect(waiting))[-1].kind == "done"
            if outcome == "cancel":
                assert target.phase == "cancelled"
                # Prefill produces no stream events, and the gated first decode
                # result is discarded when cancellation arrives during execution.
                assert target.events.empty()
            else:
                events = await collect(target)
                assert events[-1].data["message"] == "model execution failed"
                assert target.phase == "failed"
        await asyncio.wait_for(scheduler.queue.join(), 2)
        assert not scheduler.kv_cache.allocations
        assert scheduler.kv_cache.used_tokens == scheduler.kv_cache.reserved_tokens == 0
        assert not scheduler.active
        if outcome == "shutdown":
            await scheduler.start()
            restarted = scheduler.submit("good", 3)
            assert (await collect(restarted))[-1].kind == "done"
            await asyncio.wait_for(scheduler.queue.join(), 2)
            assert not scheduler.kv_cache.allocations
    finally:
        release.set()
        await scheduler.stop()
