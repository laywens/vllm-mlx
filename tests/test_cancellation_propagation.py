# SPDX-License-Identifier: Apache-2.0
"""Cancellation propagation regressions for async engine loops."""

import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.anyio
async def test_engine_loop_preserves_cancelled_state():
    from vllm_mlx.engine_core import EngineConfig, EngineCore

    core = object.__new__(EngineCore)
    core.config = EngineConfig(step_interval=10.0)
    core.scheduler = SimpleNamespace(has_requests=lambda: False)
    core._running = True

    task = asyncio.create_task(core._engine_loop())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_mllm_process_loop_preserves_cancelled_state():
    from vllm_mlx.mllm_scheduler import MLLMScheduler

    scheduler = object.__new__(MLLMScheduler)
    scheduler._running = True
    scheduler.has_requests = lambda: False

    task = asyncio.create_task(scheduler._process_loop())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_async_engine_core_start_retains_task_reference():
    from vllm_mlx.engine_core import AsyncEngineCore

    started = asyncio.Event()

    async def start():
        started.set()
        await asyncio.sleep(10)

    wrapper = object.__new__(AsyncEngineCore)
    wrapper.engine = SimpleNamespace(start=start)

    wrapper.start()
    assert wrapper._start_task is not None

    await asyncio.wait_for(started.wait(), timeout=1.0)
    wrapper._start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wrapper._start_task
