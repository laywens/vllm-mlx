# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX stream binding around synchronous batch generation."""

from types import SimpleNamespace

import pytest

from vllm_mlx.engine_core import EngineCore
from vllm_mlx.request import RequestOutput, SamplingParams


def test_generate_batch_sync_binds_generation_streams_before_scheduler_step(
    monkeypatch,
):
    import vllm_mlx.engine_core as engine_core_module

    events: list[str] = []
    request_ids: list[str] = []

    class FakeScheduler:
        def add_request(self, request):
            request_ids.append(request.request_id)

        def has_requests(self):
            return "step" not in events

        def step(self):
            events.append("step")
            return SimpleNamespace(
                outputs=[
                    RequestOutput(
                        request_id=request_ids[0],
                        new_token_ids=[1],
                        output_token_ids=[1],
                        finished=True,
                    )
                ]
            )

        def remove_finished_request(self, request_id):
            events.append(f"cleanup:{request_id}")

    def fake_bind_generation_streams():
        events.append("bind")

    monkeypatch.setattr(
        engine_core_module,
        "bind_generation_streams",
        fake_bind_generation_streams,
        raising=False,
    )

    engine = object.__new__(EngineCore)
    engine.scheduler = FakeScheduler()

    outputs = engine.generate_batch_sync(["hello"], SamplingParams(max_tokens=1))

    assert [event for event in events if event in {"bind", "step"}] == [
        "bind",
        "step",
    ]
    assert outputs[0].finished is True


@pytest.mark.anyio
async def test_engine_loop_binds_generation_streams_before_executor_step(
    monkeypatch,
):
    import vllm_mlx.engine_core as engine_core_module

    events: list[str] = []
    engine = object.__new__(EngineCore)
    engine.config = SimpleNamespace(
        step_interval=0,
        stream_interval=1,
        use_prefill_executor=True,
    )
    engine._running = True
    engine._steps_executed = 0
    engine._output_collectors = {}
    engine._stream_states = {}
    engine._finished_events = {}
    engine._maybe_clear_finished_output_cache = lambda finished_ids: False
    engine._should_run_scheduler_step_in_executor = lambda: True
    engine._record_scheduler_step_execution = lambda *, used_executor: None

    class FakeScheduler:
        batch_generator = None

        def has_requests(self):
            return engine._running

        def step(self):
            events.append("step")
            engine._running = False
            return SimpleNamespace(outputs=[], finished_request_ids=set())

        def _close_batch_generator(self):
            events.append("close")

    def fake_bind_generation_streams():
        events.append("bind")

    monkeypatch.setattr(
        engine_core_module,
        "bind_generation_streams",
        fake_bind_generation_streams,
        raising=False,
    )
    engine.scheduler = FakeScheduler()

    await engine._engine_loop()

    assert events[:2] == ["bind", "step"]
