# SPDX-License-Identifier: Apache-2.0
"""Regression tests for releasing stale request-owned MLX references."""

from collections import deque
from types import SimpleNamespace

import mlx.core as mx

from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest
from vllm_mlx.request import Request, RequestStatus, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class TestMLLMMemoryReferenceRelease:
    def test_vision_encoding_releases_preprocessed_inputs(self):
        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def __call__(self, input_ids, cache=None, **kwargs):
                self.kwargs = kwargs
                return mx.zeros((1, input_ids.shape[-1], 4))

        model = FakeModel()
        generator = MLLMBatchGenerator(model=model, processor=object())
        request = MLLMBatchRequest(
            uid=1,
            request_id="req-mllm",
            prompt="describe",
            input_ids=mx.array([1, 2, 3]),
            pixel_values=mx.array([1]),
            attention_mask=mx.array([1]),
            image_grid_thw=mx.array([1]),
            extra_kwargs={"some_processor_arg": mx.array([7])},
        )

        generator._run_vision_encoding(request, cache=[])

        assert model.kwargs is not None
        assert "pixel_values" in model.kwargs
        assert request.vision_encoded is True
        assert request.pixel_values is None
        assert request.attention_mask is None
        assert request.image_grid_thw is None
        assert request.extra_kwargs == {}


class TestSchedulerMemoryReferenceRelease:
    def test_prompt_cache_reference_is_released_after_successful_schedule(self):
        scheduler = _make_scheduler()
        request = _make_request("req-schedule")
        request.prompt_cache = [object()]
        request.remaining_tokens = [1, 2]
        scheduler.waiting = deque([request])

        class FakeBatchGenerator:
            def insert(self, prompts, **kwargs):
                assert kwargs["caches"] == [[request.prompt_cache[0]]]
                return [42]

        scheduler.batch_generator = FakeBatchGenerator()
        scheduler._ensure_batch_generator = lambda _params: None

        scheduled = scheduler._schedule_waiting()

        assert scheduled == [request]
        assert request.status == RequestStatus.RUNNING
        assert request.prompt_cache is None

    def test_cleanup_finished_releases_request_cache_references(self):
        scheduler = _make_scheduler()
        request = _make_request("req-finished")
        request.status = RequestStatus.RUNNING
        request.prompt_cache = [object()]
        request._extracted_cache = [object()]
        scheduler.running[request.request_id] = request

        scheduler._cleanup_finished({request.request_id})

        assert request.prompt_cache is None
        assert request._extracted_cache is None

    def test_abort_releases_request_cache_references(self):
        scheduler = _make_scheduler()
        request = _make_request("req-abort")
        request.status = RequestStatus.RUNNING
        request.prompt_cache = [object()]
        request._extracted_cache = [object()]
        scheduler.running[request.request_id] = request
        scheduler.requests[request.request_id] = request

        scheduler._do_abort_request(request.request_id)

        assert request.prompt_cache is None
        assert request._extracted_cache is None


def _make_scheduler() -> Scheduler:
    tokenizer = SimpleNamespace(eos_token_id=2, eos_token_ids=None)
    return Scheduler(
        model=object(),
        tokenizer=tokenizer,
        config=SchedulerConfig(enable_prefix_cache=False),
    )


def _make_request(request_id: str) -> Request:
    request = Request(
        request_id=request_id,
        prompt="hello",
        sampling_params=SamplingParams(max_tokens=8),
    )
    request.prompt_token_ids = [1, 2, 3]
    request.num_prompt_tokens = 3
    return request
