"""Blocking model work retains ownership until it actually finishes."""

import asyncio
import threading

import pytest

from llm_runtime.worker import SerialWorker


async def entered(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(poll(), 2)


async def test_worker_runs_off_loop_and_serializes_calls():
    worker = SerialWorker()
    started, release, second_started = threading.Event(), threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    threads = []

    def first(value):
        threads.append(threading.get_ident())
        started.set()
        release.wait()
        return value

    def second():
        threads.append(threading.get_ident())
        second_started.set()
        return 2

    task = asyncio.create_task(worker.run(first, 1))
    next_task = None
    try:
        await entered(started)
        next_task = asyncio.create_task(worker.run(second))
        # The event loop continues while the dedicated worker is blocked.
        await asyncio.sleep(0)
        assert not second_started.is_set()
        release.set()
        assert await asyncio.wait_for(task, 2) == 1
        assert await asyncio.wait_for(next_task, 2) == 2
        assert threads[0] == threads[1] != loop_thread
    finally:
        release.set()
        await worker.aclose()


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('queued', [False, True])
async def test_repeated_cancel_waits_for_running_or_queued_work(failure, queued):
    worker = SerialWorker()
    started, release = threading.Event(), threading.Event()
    finished = threading.Event()
    blocker = None
    errors = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    def block():
        started.set()
        release.wait()

    def execute():
        if not queued:
            block()
        finished.set()
        if failure:
            raise ValueError('worker failed after cancellation')
        return 'finished'

    if queued:
        blocker = asyncio.create_task(worker.run(block))
        await entered(started)
    task = asyncio.create_task(worker.run(execute))
    try:
        if queued:
            await asyncio.sleep(0)  # Submit behind the blocked first operation.
        else:
            await entered(started)
        for _ in range(3):
            assert task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set()
        if blocker:
            await blocker
        # The worker remains usable and every exception has been retrieved.
        assert await worker.run(lambda: 'next') == 'next'
        await asyncio.sleep(0)
        assert errors == []
    finally:
        release.set()
        await worker.aclose()
        loop.set_exception_handler(previous_handler)


async def test_exception_propagates_and_worker_remains_usable():
    worker = SerialWorker()

    def fail():
        raise ValueError('bad input')

    try:
        with pytest.raises(ValueError, match='bad input'):
            await worker.run(fail)
        assert await worker.run(pow, 2, exp=3) == 8
    finally:
        await worker.aclose()


async def test_close_drains_work_rejects_submissions_and_survives_cancellation():
    worker = SerialWorker()
    started, release = threading.Event(), threading.Event()

    def block():
        started.set()
        release.wait()
        return 'done'

    task = asyncio.create_task(worker.run(block))
    close = None
    other_close = None
    try:
        await entered(started)
        close = asyncio.create_task(worker.aclose())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match='closed'):
            await worker.run(lambda: None)
        other_close = asyncio.create_task(worker.aclose())
        for _ in range(3):
            close.cancel()
            await asyncio.sleep(0)
            assert not close.done()
            assert not other_close.done()
        release.set()
        assert await asyncio.wait_for(task, 2) == 'done'
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close, 2)
        await asyncio.wait_for(other_close, 2)
        await worker.aclose()
        with pytest.raises(RuntimeError, match='closed'):
            await worker.run(lambda: None)
    finally:
        release.set()
        await worker.aclose()


async def test_close_without_work_is_idempotent():
    worker = SerialWorker()
    await worker.aclose()
    await worker.aclose()
    with pytest.raises(RuntimeError, match='closed'):
        await worker.run(lambda: None)
