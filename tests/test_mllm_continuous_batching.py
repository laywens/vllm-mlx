# SPDX-License-Identifier: Apache-2.0
"""
Tests for MLLM (Multimodal Language Model) continuous batching.

These tests verify that the MLLM batch generator and scheduler work correctly
for batching multiple multimodal requests together.

Test Cases:
- Single MLLM request works correctly
- 2, 4, 8 concurrent requests with batching
- Vision cache hits/misses
- Streaming with batching
- Mixed text-only and multimodal requests
"""

import asyncio
import base64
import os
import sys
import tempfile
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# Skip all tests if MLX is not available
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


# Test image (small PNG)
TEST_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class TestMLLMPromptCacheEval:
    def test_collects_kv_and_arrays_cache_tensors(self):
        from vllm_mlx.mllm_batch_generator import _cache_eval_tensors

        kv_keys = object()
        kv_values = object()
        state_a = object()
        state_b = object()

        class KVLikeCache:
            keys = kv_keys
            values = kv_values

            @property
            def state(self):
                raise AssertionError("KV cache state should not be read")

        class ArraysLikeCache:
            state = [state_a, None, state_b]

        class EmptyCache:
            keys = None
            values = None
            state = [None]

        assert _cache_eval_tensors(
            [KVLikeCache(), ArraysLikeCache(), EmptyCache()]
        ) == [kv_keys, kv_values, state_a, state_b]

    def test_eval_prompt_cache_flattens_cache_tensors(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import _eval_prompt_cache

        kv_keys = object()
        kv_values = object()
        state = object()
        eval_mock = MagicMock()
        monkeypatch.setattr(mx, "eval", eval_mock)

        class KVLikeCache:
            keys = kv_keys
            values = kv_values

        class ArraysLikeCache:
            pass

        ArraysLikeCache.state = [state]

        _eval_prompt_cache([KVLikeCache(), ArraysLikeCache()])

        eval_mock.assert_called_once_with(kv_keys, kv_values, state)


def create_test_image(path: str, size: tuple = (32, 32)) -> str:
    """Create a test image file."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        img.save(path)
        return path
    except ImportError:
        # Fallback: write a minimal valid PNG
        png_data = base64.b64decode(TEST_IMAGE_B64)
        with open(path, "wb") as f:
            f.write(png_data)
        return path


class TestMLLMBatchRequest:
    """Tests for MLLMBatchRequest dataclass."""

    def test_create_request(self):
        """Test creating a basic request."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=0,
            request_id="test-1",
            prompt="What's in this image?",
            images=["test.jpg"],
            max_tokens=100,
        )

        assert req.uid == 0
        assert req.request_id == "test-1"
        assert req.prompt == "What's in this image?"
        assert req.images == ["test.jpg"]
        assert req.max_tokens == 100
        assert req.num_tokens == 0
        assert req.vision_encoded is False

    def test_request_defaults(self):
        """Test default values."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        req = MLLMBatchRequest(
            uid=1,
            request_id="test-2",
            prompt="Hello",
        )

        assert req.images is None
        assert req.videos is None
        assert req.audio is None
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.video_fps == 2.0
        assert req.video_max_frames == 128
        assert req.output_tokens == []


class TestMLLMBatchResponse:
    """Tests for MLLMBatchResponse dataclass."""

    def test_create_response(self):
        """Test creating a response."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        logprobs = mx.array([0.1, 0.2, 0.3])

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=42,
            logprobs=logprobs,
            finish_reason=None,
        )

        assert resp.uid == 0
        assert resp.request_id == "test-1"
        assert resp.token == 42
        assert resp.finish_reason is None

    def test_finished_response(self):
        """Test response with finish reason."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        resp = MLLMBatchResponse(
            uid=0,
            request_id="test-1",
            token=2,  # EOS
            logprobs=mx.array([0.1]),
            finish_reason="stop",
        )

        assert resp.finish_reason == "stop"


class TestMLLMBatch:
    """Tests for MLLMBatch class."""

    def test_batch_length(self):
        """Test batch length calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(3)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2],
            request_ids=["req-0", "req-1", "req-2"],
            y=mx.array([100, 200, 300]),
            logprobs=[mx.array([0.1]), mx.array([0.2]), mx.array([0.3])],
            max_tokens=[100, 100, 100],
            num_tokens=[0, 0, 0],
            cache=[],
            requests=requests,
        )

        assert len(batch) == 3

    def test_batch_filter(self):
        """Test filtering a batch."""
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        requests = [
            MLLMBatchRequest(uid=i, request_id=f"req-{i}", prompt=f"prompt {i}")
            for i in range(4)
        ]

        batch = MLLMBatch(
            uids=[0, 1, 2, 3],
            request_ids=["req-0", "req-1", "req-2", "req-3"],
            y=mx.array([100, 200, 300, 400]),
            logprobs=[
                mx.array([0.1]),
                mx.array([0.2]),
                mx.array([0.3]),
                mx.array([0.4]),
            ],
            max_tokens=[100, 100, 100, 100],
            num_tokens=[0, 0, 0, 0],
            cache=[],
            requests=requests,
        )

        # Keep only indices 1 and 3
        batch.filter([1, 3])

        assert len(batch) == 2
        assert batch.uids == [1, 3]
        assert batch.request_ids == ["req-1", "req-3"]

    def test_batch_extend_handles_empty_protocol_caches_without_keys(self):
        from vllm_mlx.mllm_batch_generator import MLLMBatch, MLLMBatchRequest

        class OpaqueCache:
            def __init__(self):
                self.extend_calls = 0
                self.extended_with = None

            def empty(self):
                return False

            def extend(self, other):
                self.extend_calls += 1
                self.extended_with = other

        primary_cache = OpaqueCache()
        other_cache = OpaqueCache()
        primary = MLLMBatch(
            uids=[1],
            request_ids=["req-1"],
            y=mx.array([100]),
            logprobs=[mx.array([0.1])],
            max_tokens=[100],
            num_tokens=[0],
            cache=[primary_cache],
            requests=[MLLMBatchRequest(uid=1, request_id="req-1", prompt="one")],
        )
        other = MLLMBatch(
            uids=[2],
            request_ids=["req-2"],
            y=mx.array([200]),
            logprobs=[mx.array([0.2])],
            max_tokens=[100],
            num_tokens=[0],
            cache=[other_cache],
            requests=[MLLMBatchRequest(uid=2, request_id="req-2", prompt="two")],
        )

        primary.extend(other)

        assert primary.y.shape == (2,)
        assert primary_cache.extend_calls == 1
        assert primary_cache.extended_with is other_cache


class TestMLLMPrefillCacheMerge:
    """Regression tests for hybrid MLLM prompt-cache merging."""

    def test_merge_prefill_caches_supports_hybrid_layers(self):
        from mlx_lm.models.cache import ArraysCache
        from vllm_mlx.mllm_batch_generator import _merge_prefill_caches

        class FakeKVLikeCache:
            def __init__(self, value):
                self.value = value

            @classmethod
            def merge(cls, caches):
                merged = cls(None)
                merged.value = mx.concatenate([cache.value for cache in caches], axis=0)
                return merged

        def make_arrays(value: float):
            cache = mx.array([[value, value + 1]])
            arrays = ArraysCache(1)
            arrays[0] = cache
            return arrays

        caches = [
            [make_arrays(1.0), FakeKVLikeCache(mx.array([[10.0]]))],
            [make_arrays(2.0), FakeKVLikeCache(mx.array([[20.0]]))],
        ]

        merged = _merge_prefill_caches(caches)

        assert len(merged) == 2
        assert merged[0].cache[0].shape == (2, 2)
        assert merged[0].cache[0].tolist() == [[1.0, 2.0], [2.0, 3.0]]
        assert merged[1].value.tolist() == [[10.0], [20.0]]

    def test_merge_prefill_caches_rejects_non_mergeable_layers(self):
        from vllm_mlx.mllm_batch_generator import _merge_prefill_caches

        with pytest.raises(ValueError, match="merge-capable prompt caches"):
            _merge_prefill_caches([[object()], [object()]])

    def test_normalize_generation_caches_wraps_vector_offsets(self):
        from vllm_mlx.mllm_batch_generator import _normalize_generation_caches

        class FakeBatchCache:
            def __init__(self):
                self.offset = mx.array([4, 6], dtype=mx.int32)
                self._idx = 6

        wrapped = _normalize_generation_caches([FakeBatchCache()])

        assert len(wrapped) == 1
        assert wrapped[0].offset == 6


class TestMLLMBatchStats:
    """Tests for MLLMBatchStats."""

    def test_stats_initialization(self):
        """Test stats initialization."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tokens == 0
        assert stats.generation_tokens == 0
        assert stats.prompt_time == 0
        assert stats.generation_time == 0
        assert stats.num_images_processed == 0

    def test_tps_calculation(self):
        """Test tokens per second calculation."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()
        stats.prompt_tokens = 100
        stats.prompt_time = 2.0
        stats.generation_tokens = 50
        stats.generation_time = 1.0

        assert stats.prompt_tps == 50.0
        assert stats.generation_tps == 50.0

    def test_tps_zero_time(self):
        """Test TPS with zero time."""
        from vllm_mlx.mllm_batch_generator import MLLMBatchStats

        stats = MLLMBatchStats()

        assert stats.prompt_tps == 0
        assert stats.generation_tps == 0


class TestMLLMSchedulerConfig:
    """Tests for MLLMSchedulerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig()

        assert config.max_num_seqs == 16
        # prefill_batch_size set equal to max_num_seqs to avoid batch extend issues
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 16
        assert config.enable_vision_cache is True
        assert config.vision_cache_size == 100
        assert config.chunked_prefill_tokens == 0

    def test_custom_config(self):
        """Test custom configuration."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

        config = MLLMSchedulerConfig(
            max_num_seqs=8,
            prefill_batch_size=2,
            completion_batch_size=8,
            enable_vision_cache=False,
            chunked_prefill_tokens=512,
        )

        assert config.max_num_seqs == 8
        assert config.prefill_batch_size == 2
        assert config.completion_batch_size == 8
        assert config.enable_vision_cache is False
        assert config.chunked_prefill_tokens == 512


class TestMLLMSchedulerAbortMetrics:
    """Tests for MLLM scheduler abort accounting."""

    def test_abort_running_request_credits_inflight_tokens(self):
        from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vllm_mlx.request import RequestStatus

        scheduler = MLLMScheduler(model=object(), processor=object())
        request = MLLMRequest(request_id="req-1", prompt="Hello")
        request.status = RequestStatus.RUNNING
        request.num_prompt_tokens = 5
        request.output_tokens = [10, 11, 12]
        request.num_output_tokens = 3

        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request

        assert scheduler.abort_request(request.request_id) is True

        assert scheduler.total_prompt_tokens == 5
        assert scheduler.total_completion_tokens == 3

    def test_abort_running_request_cleans_detokenizer(self):
        from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vllm_mlx.request import RequestStatus

        scheduler = MLLMScheduler(model=object(), processor=object())
        request = MLLMRequest(request_id="req-1", prompt="Hello")
        request.status = RequestStatus.RUNNING

        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler._detokenizer_pool[request.request_id] = object()

        assert scheduler.abort_request(request.request_id) is True
        assert request.request_id not in scheduler._detokenizer_pool

    def test_reset_clears_detokenizer_pool(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        scheduler = MLLMScheduler(model=object(), processor=object())
        scheduler._detokenizer_pool["orphan"] = object()

        scheduler.reset()

        assert scheduler._detokenizer_pool == {}


class TestMLLMSchedulerDetokenizer:
    """Tests for MLLM streaming detokenizer behavior."""

    def test_stop_token_is_not_emitted_as_new_text(self):
        import mlx.core as mx

        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse
        from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler
        from vllm_mlx.request import RequestStatus

        class FakeDetokenizer:
            def reset(self):
                self.last_segment = ""
                self.text = ""

            def add_token(self, token):
                self.last_segment = f"<{token}>"
                self.text += self.last_segment

            def finalize(self):
                pass

        class FakeTokenizer:
            eos_token_id = 0
            eos_token_ids = None

            @property
            def detokenizer(self):
                return FakeDetokenizer()

            def decode(self, tokens):
                return "".join(f"<{token}>" for token in tokens)

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        scheduler = MLLMScheduler(model=object(), processor=FakeProcessor())
        request = MLLMRequest(request_id="req-1", prompt="Hello")
        request.status = RequestStatus.RUNNING
        request.num_prompt_tokens = 1

        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request
        scheduler.request_id_to_uid[request.request_id] = 7
        scheduler.uid_to_request_id[7] = request.request_id

        outputs, finished_ids = scheduler._process_batch_responses(
            [
                MLLMBatchResponse(
                    uid=7,
                    request_id=request.request_id,
                    token=0,
                    logprobs=mx.array([0.0]),
                    finish_reason="stop",
                )
            ]
        )

        assert finished_ids == {request.request_id}
        assert outputs[0].new_text == ""

    @pytest.mark.asyncio
    async def test_stream_outputs_final_chunk_close_does_not_abort(self):
        """Closing after the final output must not abort the finished request."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler
        from vllm_mlx.request import RequestOutput

        request_id = "req-final"
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.output_queues = {request_id: asyncio.Queue()}
        scheduler.abort_request = MagicMock()

        final_output = RequestOutput(request_id=request_id, finished=True)
        scheduler.output_queues[request_id].put_nowait(final_output)

        stream = scheduler.stream_outputs(request_id)
        assert await stream.__anext__() is final_output
        await stream.aclose()

        scheduler.abort_request.assert_not_called()


class TestPreprocessIdempotent:
    def test_text_only_request_with_input_ids_is_not_preprocessed_twice(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
        )

        req = MLLMBatchRequest(
            uid=0,
            request_id="req-idem",
            prompt="Hello",
        )
        req.input_ids = mx.array([[1, 2, 3]])

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)

        gen._preprocess_request(req)

        assert req.input_ids.shape == (1, 3)

    def test_vision_request_with_input_ids_is_still_preprocessed(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
        )

        req = MLLMBatchRequest(
            uid=0,
            request_id="req-vision",
            prompt="Describe",
            images=["fake.png"],
        )
        req.input_ids = mx.array([[1, 2, 3]])

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)

        with pytest.raises(Exception):
            gen._preprocess_request(req)

    def test_audio_request_is_preprocessed_and_passed_to_prepare_inputs(
        self, monkeypatch
    ):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
        )

        captured = {}

        fake_mlx_vlm = ModuleType("mlx_vlm")
        fake_utils = ModuleType("mlx_vlm.utils")

        def fake_prepare_inputs(processor, **kwargs):
            captured.update(kwargs)
            return {
                "input_ids": mx.array([1, 2, 3]),
                "attention_mask": mx.array([1, 1, 1]),
            }

        fake_utils.prepare_inputs = fake_prepare_inputs
        fake_mlx_vlm.utils = fake_utils
        monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mlx_vlm)
        monkeypatch.setitem(sys.modules, "mlx_vlm.utils", fake_utils)

        class FailOnPixelCache:
            def get_pixel_cache(self, images, prompt):
                raise AssertionError("pixel cache should be disabled for audio")

            def set_pixel_cache(self, **kwargs):
                raise AssertionError("pixel cache should not store audio requests")

        req = MLLMBatchRequest(
            uid=0,
            request_id="req-audio",
            prompt="Transcribe",
            audio=[f"data:audio/wav;base64,{base64.b64encode(b'audio').decode()}"],
        )
        req.input_ids = mx.array([[9, 9]])

        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.model = MagicMock()
        gen.model.config = None
        gen.processor = MagicMock()
        gen.vision_cache = FailOnPixelCache()
        gen._stats = MagicMock()
        gen._stats.num_images_processed = 0
        gen._stats.vision_encoding_time = 0

        gen._preprocess_request(req)

        assert captured["audio"]
        assert captured["images"] is None
        assert req.input_ids.tolist() == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_process_loop_preprocesses_text_requests_before_step(
        self, monkeypatch
    ):
        import vllm_mlx.mllm_scheduler as scheduler_mod
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        req = MLLMBatchRequest(
            uid=0,
            request_id="req-loop",
            prompt="Hello",
        )

        class FakeBatchGenerator:
            unprocessed_requests = [req]

            def _preprocess_request(self, request):
                request.input_ids = object()

        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler._running = True
        scheduler.batch_generator = FakeBatchGenerator()
        step_saw_preprocessed = []

        scheduler.has_requests = lambda: True

        def step():
            step_saw_preprocessed.append(req.input_ids is not None)
            scheduler._running = False

        scheduler.step = step
        monkeypatch.setattr(
            scheduler_mod, "bind_generation_streams", lambda: None, raising=False
        )

        await scheduler._process_loop()

        assert step_saw_preprocessed == [True]


class TestMLLMSamplingControls:
    def test_process_prompts_uses_request_sampler_for_first_token(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        class FakeCache:
            def merge(self, caches):
                return self

        request_sampler = MagicMock(return_value=mx.array([3], dtype=mx.uint32))
        fallback_sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))
        sampler_calls = []

        def fake_make_sampler(**kwargs):
            sampler_calls.append(kwargs)
            return request_sampler

        monkeypatch.setattr(mx, "stream", lambda stream: nullcontext())
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda model, max_kv_size=None: [FakeCache()],
        )
        monkeypatch.setattr("mlx_lm.sample_utils.make_sampler", fake_make_sampler)

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._stats = MLLMBatchStats()
        generator.prefill_step_size = 512
        generator.language_model = object()
        generator.model = MagicMock()
        generator.sampler = fallback_sampler
        generator.max_kv_size = 0
        generator._preprocess_request = lambda req: None
        generator._run_vision_encoding = lambda req, cache: mx.array(
            [[[0.0, 1.0, 2.0, 3.0]]]
        )

        request = MLLMBatchRequest(
            uid=7,
            request_id="req-7",
            prompt="hello",
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            min_p=0.15,
        )
        request.input_ids = mx.array([[42]], dtype=mx.uint32)

        batch = MLLMBatchGenerator._process_prompts(generator, [request])

        assert batch.y.tolist() == [3]
        assert batch.samplers == [request_sampler]
        request_sampler.assert_called_once()
        fallback_sampler.assert_not_called()
        assert sampler_calls == [
            {"temp": 0.3, "top_p": 0.8, "top_k": 20, "min_p": 0.15}
        ]

    def test_next_uses_request_sampler_for_decode_token(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        request_sampler = MagicMock(return_value=mx.array([3], dtype=mx.uint32))
        fallback_sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))

        request = MLLMBatchRequest(uid=7, request_id="req-7", prompt="hello")
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator.language_model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0, 3.0]]])
        )
        generator.sampler = fallback_sampler
        generator.active_batch = MLLMBatch(
            uids=[7],
            request_ids=["req-7"],
            y=mx.array([5], dtype=mx.uint32),
            logprobs=[mx.array([0.5, 0.5])],
            max_tokens=[4],
            num_tokens=[0],
            cache=[],
            requests=[request],
            samplers=[request_sampler],
        )

        responses = MLLMBatchGenerator._next(generator)

        assert [response.token for response in responses] == [5]
        assert generator.active_batch.y.tolist() == [3]
        request_sampler.assert_called_once()
        fallback_sampler.assert_not_called()

    def test_next_applies_request_logits_processors_for_decode_token(self):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        seen_output_tokens = []

        def presence_processor(output_tokens, logits):
            seen_output_tokens.append(output_tokens.tolist())
            return logits

        request_sampler = MagicMock(return_value=mx.array([3], dtype=mx.uint32))

        request = MLLMBatchRequest(uid=7, request_id="req-7", prompt="hello")
        request.output_tokens.extend([1, 2])
        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator.language_model = MagicMock(
            return_value=mx.array([[[0.0, 1.0, 2.0, 3.0]]])
        )
        generator.sampler = MagicMock(return_value=mx.array([1], dtype=mx.uint32))
        generator.active_batch = MLLMBatch(
            uids=[7],
            request_ids=["req-7"],
            y=mx.array([5], dtype=mx.uint32),
            logprobs=[mx.array([0.5, 0.5])],
            max_tokens=[4],
            num_tokens=[0],
            cache=[],
            requests=[request],
            logits_processors=[[presence_processor]],
            samplers=[request_sampler],
        )

        responses = MLLMBatchGenerator._next(generator)

        assert [response.token for response in responses] == [5]
        assert seen_output_tokens == [[1, 2]]
        request_sampler.assert_called_once()
        generator.sampler.assert_not_called()


class TestMLLMSchedulerSamplingParams:
    """Tests for sampling parameter propagation in scheduler request state."""

    def test_add_request_stores_full_sampling_params(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )

        request_id = scheduler.add_request(
            prompt="Describe",
            max_tokens=32,
            top_k=40,
            min_p=0.2,
            presence_penalty=0.4,
            repetition_penalty=1.2,
        )
        request = scheduler.get_request(request_id)

        assert request is not None
        assert request.sampling_params.top_k == 40
        assert request.sampling_params.min_p == 0.2
        assert request.sampling_params.presence_penalty == 0.4
        assert request.sampling_params.repetition_penalty == 1.2

    def test_schedule_waiting_propagates_sampling_params_to_batch_request(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        class DummyBatchGenerator:
            def __init__(self):
                self.inserted = None

            def insert(self, batch_requests):
                self.inserted = batch_requests
                return list(range(100, 100 + len(batch_requests)))

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )
        dummy_batch = DummyBatchGenerator()
        scheduler.batch_generator = dummy_batch

        scheduler.add_request(
            prompt="Describe",
            max_tokens=16,
            top_k=24,
            min_p=0.1,
            presence_penalty=0.3,
            repetition_penalty=1.15,
        )
        scheduled = scheduler._schedule_waiting()

        assert len(scheduled) == 1
        assert dummy_batch.inserted is not None
        assert len(dummy_batch.inserted) == 1
        assert dummy_batch.inserted[0].top_k == 24
        assert dummy_batch.inserted[0].min_p == 0.1
        assert dummy_batch.inserted[0].presence_penalty == 0.3
        assert dummy_batch.inserted[0].repetition_penalty == 1.15


class TestMLLMChunkedPrefill:
    """Tests for MLLM text prefill interleaving."""

    class MergeCache:
        def __init__(self):
            self.extended = []

        def merge(self, caches):
            return self

        def extend(self, other):
            self.extended.append(other)

    def _make_generator(self, monkeypatch, *, budget=2):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchStats,
            install_chunked_prefill_mllm,
        )

        prompt_cache = self.MergeCache()
        monkeypatch.setattr(
            "mlx_lm.models.cache.make_prompt_cache",
            lambda model, max_kv_size=None: [prompt_cache],
        )
        sampler = MagicMock(return_value=mx.array([3], dtype=mx.uint32))
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_sampler",
            lambda **kwargs: sampler,
        )

        calls = []

        def language_model(tokens, cache=None):
            calls.append(tuple(tokens.shape))
            return mx.array([[[0.0, 1.0, 2.0, 3.0]]] * tokens.shape[0])

        generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        generator._stats = MLLMBatchStats()
        generator.stop_tokens = set()
        generator.unprocessed_requests = []
        generator.active_batch = None
        generator.language_model = language_model
        generator.sampler = sampler
        generator.max_kv_size = 0
        generator._preprocess_request = lambda req: None
        generator._next = MLLMBatchGenerator._next.__get__(
            generator, MLLMBatchGenerator
        )

        install_chunked_prefill_mllm(generator, budget=budget)
        return generator, calls

    def test_long_text_prefill_advances_one_chunk_per_next(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator, calls = self._make_generator(monkeypatch, budget=2)
        request = MLLMBatchRequest(
            uid=11,
            request_id="req-long",
            prompt="long",
            max_tokens=4,
        )
        request.input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
        generator.unprocessed_requests.append(request)

        assert generator._next() == []
        assert calls == [(1, 2)]
        assert generator._partial is not None
        assert generator._partial["processed"] == 2

        assert generator._next() == []
        assert calls == [(1, 2), (1, 2)]
        assert generator._partial["processed"] == 4

        responses = generator._next()

        assert [response.request_id for response in responses] == ["req-long"]
        assert [response.token for response in responses] == [3]
        assert generator._partial is None
        assert calls == [(1, 2), (1, 2), (1, 1), (1, 1)]

    def test_active_decode_runs_while_long_prefill_is_partial(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import (
            MLLMBatch,
            MLLMBatchRequest,
        )

        generator, calls = self._make_generator(monkeypatch, budget=2)
        active_request = MLLMBatchRequest(
            uid=7,
            request_id="req-active",
            prompt="active",
            max_tokens=4,
        )
        generator.active_batch = MLLMBatch(
            uids=[7],
            request_ids=["req-active"],
            y=mx.array([5], dtype=mx.uint32),
            logprobs=[mx.array([0.0])],
            max_tokens=[4],
            num_tokens=[0],
            cache=[self.MergeCache()],
            requests=[active_request],
        )
        long_request = MLLMBatchRequest(
            uid=12,
            request_id="req-long",
            prompt="long",
            max_tokens=4,
        )
        long_request.input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
        generator.unprocessed_requests.append(long_request)

        responses = generator._next()

        assert [response.request_id for response in responses] == ["req-active"]
        assert [response.token for response in responses] == [5]
        assert generator._partial is not None
        assert generator._partial["request"] is long_request
        assert calls == [(1, 2), (1, 1)]

    def test_remove_clears_partial_prefill(self, monkeypatch):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        generator, _ = self._make_generator(monkeypatch, budget=2)
        request = MLLMBatchRequest(uid=11, request_id="req-long", prompt="long")
        generator._partial = {
            "request": request,
            "cache": [self.MergeCache()],
            "remaining_ids": mx.array([[3, 4, 5]], dtype=mx.uint32),
            "processed": 2,
            "total": 5,
            "chunk_count": 1,
        }

        generator.remove([11])

        assert generator._partial is None


class TestMLLMChunkedPrefillScheduler:
    def test_ensure_batch_generator_installs_chunked_prefill(self, monkeypatch):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        installed = {}

        class FakeBatchGenerator:
            def __init__(self, **kwargs):
                installed["batch_kwargs"] = kwargs

        def fake_install(batch_generator, *, budget):
            installed["batch_generator"] = batch_generator
            installed["budget"] = budget

        monkeypatch.setattr(
            "vllm_mlx.mllm_scheduler.MLLMBatchGenerator", FakeBatchGenerator
        )
        monkeypatch.setattr(
            "vllm_mlx.mllm_batch_generator.install_chunked_prefill_mllm",
            fake_install,
            raising=False,
        )
        monkeypatch.setattr(
            "mlx_lm.sample_utils.make_sampler",
            lambda **kwargs: MagicMock(),
        )

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None
        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(
                enable_vision_cache=False,
                chunked_prefill_tokens=384,
            ),
        )

        scheduler._ensure_batch_generator()

        assert installed["batch_generator"] is scheduler.batch_generator
        assert installed["budget"] == 384


class TestMLLMSchedulerVideoParams:
    """Tests for video parameter propagation in scheduler request state."""

    def test_add_request_stores_video_sampling_settings(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )

        request_id = scheduler.add_request(
            prompt="Describe this video",
            videos=["demo.mp4"],
            video_fps=1.5,
            video_max_frames=24,
            max_tokens=32,
        )
        request = scheduler.get_request(request_id)

        assert request is not None
        assert request.videos == ["demo.mp4"]
        assert request.video_fps == 1.5
        assert request.video_max_frames == 24

    def test_schedule_waiting_propagates_video_params_to_batch_request(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        class DummyBatchGenerator:
            def __init__(self):
                self.inserted = None

            def insert(self, batch_requests):
                self.inserted = batch_requests
                return list(range(100, 100 + len(batch_requests)))

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )
        dummy_batch = DummyBatchGenerator()
        scheduler.batch_generator = dummy_batch

        scheduler.add_request(
            prompt="Describe this video",
            videos=["demo.mp4"],
            video_fps=3.0,
            video_max_frames=48,
            max_tokens=16,
        )
        scheduled = scheduler._schedule_waiting()

        assert len(scheduled) == 1
        assert dummy_batch.inserted is not None
        assert len(dummy_batch.inserted) == 1
        assert dummy_batch.inserted[0].video_fps == 3.0
        assert dummy_batch.inserted[0].video_max_frames == 48


class TestMLLMSchedulerAudioParams:
    """Tests for audio propagation in scheduler request state."""

    def test_add_request_stores_audio_inputs(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )

        request_id = scheduler.add_request(
            prompt="Transcribe",
            audio=["clip.wav"],
            max_tokens=32,
        )
        request = scheduler.get_request(request_id)

        assert request is not None
        assert request.audio == ["clip.wav"]

    def test_schedule_waiting_propagates_audio_to_batch_request(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig

        class DummyBatchGenerator:
            def __init__(self):
                self.inserted = None

            def insert(self, batch_requests):
                self.inserted = batch_requests
                return list(range(100, 100 + len(batch_requests)))

        model = MagicMock()
        model.config = None
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.eos_token_id = 2
        processor.tokenizer.eos_token_ids = None

        scheduler = MLLMScheduler(
            model=model,
            processor=processor,
            config=MLLMSchedulerConfig(enable_vision_cache=False),
        )
        dummy_batch = DummyBatchGenerator()
        scheduler.batch_generator = dummy_batch

        scheduler.add_request(
            prompt="Transcribe",
            audio=["clip.wav"],
            max_tokens=16,
        )
        scheduled = scheduler._schedule_waiting()

        assert len(scheduled) == 1
        assert dummy_batch.inserted is not None
        assert dummy_batch.inserted[0].audio == ["clip.wav"]


class TestMLLMRequest:
    """Tests for MLLMRequest dataclass."""

    def test_create_request(self):
        """Test creating an MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMRequest
        from vllm_mlx.request import RequestStatus

        req = MLLMRequest(
            request_id="req-1",
            prompt="Describe this image",
            images=["image.jpg"],
        )

        assert req.request_id == "req-1"
        assert req.prompt == "Describe this image"
        assert req.images == ["image.jpg"]
        assert req.status == RequestStatus.WAITING
        assert req.output_text == ""


class TestMLLMSchedulerOutput:
    """Tests for MLLMSchedulerOutput."""

    def test_empty_output(self):
        """Test empty scheduler output."""
        from vllm_mlx.mllm_scheduler import MLLMSchedulerOutput

        output = MLLMSchedulerOutput()

        assert output.scheduled_request_ids == []
        assert output.num_scheduled_tokens == 0
        assert output.finished_request_ids == set()
        assert output.outputs == []
        assert output.has_work is False


class TestMultimodalProcessorBatch:
    """Tests for MultimodalProcessor batch methods."""

    def test_batch_pixel_values_empty(self):
        """Test batching empty pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        # Create mock processor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        result = processor.batch_pixel_values([None, None])
        assert result is None

    def test_batch_pixel_values_single(self):
        """Test batching single pixel value."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        result = processor.batch_pixel_values([pixels])

        assert result is not None
        assert result.shape == (1, 3, 32, 32)

    def test_batch_pixel_values_multiple(self):
        """Test batching multiple pixel values."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels1 = mx.ones((1, 3, 32, 32))
        pixels2 = mx.ones((1, 3, 32, 32)) * 2

        result = processor.batch_pixel_values([pixels1, pixels2])

        assert result is not None
        assert result.shape == (2, 3, 32, 32)

    def test_batch_image_grid_thw(self):
        """Test batching image grid thw."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        grid1 = mx.array([[1, 4, 4]])
        grid2 = mx.array([[1, 8, 8]])

        result = processor.batch_image_grid_thw([grid1, grid2])

        assert result is not None
        assert result.shape[0] == 2

    def test_prepare_for_batch(self):
        """Test prepare_for_batch method."""
        from vllm_mlx.multimodal_processor import (
            MultimodalProcessor,
            ProcessedMultimodalInput,
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        # Create processed inputs
        inputs = [
            ProcessedMultimodalInput(
                input_ids=mx.array([1, 2, 3]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=3,
            ),
            ProcessedMultimodalInput(
                input_ids=mx.array([4, 5, 6, 7, 8]),
                pixel_values=mx.ones((1, 3, 32, 32)),
                num_images=1,
                num_tokens=5,
            ),
        ]

        input_ids, batch_kwargs, padding = processor.prepare_for_batch(inputs)

        # Check left-padding
        assert input_ids.shape == (2, 5)  # max length is 5
        assert padding == [2, 0]  # first input needs 2 padding

    def test_compute_vision_hash(self):
        """Test vision hash computation."""
        from vllm_mlx.multimodal_processor import MultimodalProcessor

        mock_model = MagicMock()
        mock_processor = MagicMock()

        processor = MultimodalProcessor(mock_model, mock_processor)

        pixels = mx.ones((1, 3, 32, 32))
        hash1 = processor.compute_vision_hash(pixels)
        hash2 = processor.compute_vision_hash(pixels)

        # Same input should give same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


class TestVisionCache:
    """Tests for VLM cache functionality."""

    def test_cache_creation(self):
        """Test VLM cache creation."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=10)

        assert len(cache) == 0
        assert cache.max_size == 10

    def test_cache_miss(self):
        """Test cache miss."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is None
        assert hit is False
        assert cache.stats.misses == 1

    def test_cache_store_and_fetch(self):
        """Test storing and fetching from cache."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager()

        # Store cache
        test_cache = [{"key": "value"}]
        cache.store_cache(["image.jpg"], "prompt", test_cache, num_tokens=100)

        # Fetch cache
        result, hit = cache.fetch_cache(["image.jpg"], "prompt")

        assert result is not None
        assert hit is True
        assert cache.stats.hits == 1
        assert cache.stats.tokens_saved == 100

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        from vllm_mlx.mllm_cache import MLLMCacheManager

        cache = MLLMCacheManager(max_entries=2)

        # Fill cache
        cache.store_cache(["img1.jpg"], "prompt1", [1], num_tokens=10)
        cache.store_cache(["img2.jpg"], "prompt2", [2], num_tokens=20)

        assert len(cache) == 2

        # Add one more (should evict oldest)
        cache.store_cache(["img3.jpg"], "prompt3", [3], num_tokens=30)

        assert len(cache) == 2
        assert cache.stats.evictions == 1

        # img1 should be evicted
        _, hit = cache.fetch_cache(["img1.jpg"], "prompt1")
        assert hit is False


# Integration tests (require model loading)
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"), reason="Slow tests disabled")
class TestMLLMSchedulerIntegration:
    """Integration tests for MLLMScheduler with real models."""

    @pytest.fixture
    def test_image_path(self):
        """Create a test image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = create_test_image(f.name)
            yield path
            os.unlink(path)

    async def test_single_request(self, test_image_path):
        """Test single MLLM request."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        # Load a small model
        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = scheduler.add_request(
                prompt="What's in this image?",
                images=[test_image_path],
                max_tokens=50,
            )

            # Run until complete
            while scheduler.has_requests():
                output = scheduler.step()
                if request_id in output.finished_request_ids:
                    break

            # Check result
            request = scheduler.get_request(request_id)
            assert request is not None
            assert len(request.output_tokens) > 0

        finally:
            await scheduler.stop()

    async def test_concurrent_requests(self, test_image_path):
        """Test multiple concurrent MLLM requests."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig(max_num_seqs=4)
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            # Add multiple requests
            request_ids = []
            for i in range(4):
                req_id = scheduler.add_request(
                    prompt=f"Describe image {i}",
                    images=[test_image_path],
                    max_tokens=30,
                )
                request_ids.append(req_id)

            # Run until all complete
            finished = set()
            while len(finished) < len(request_ids):
                output = scheduler.step()
                finished.update(output.finished_request_ids)

            # Check all completed
            assert len(finished) == 4

            # Check stats show batching
            stats = scheduler.get_stats()
            assert stats["num_requests_processed"] == 4

        finally:
            await scheduler.stop()

    async def test_streaming(self, test_image_path):
        """Test streaming MLLM generation."""
        from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig
        from mlx_vlm import load

        model, processor = load("mlx-community/Qwen3-VL-4B-Instruct-3bit")

        config = MLLMSchedulerConfig()
        scheduler = MLLMScheduler(model, processor, config)

        await scheduler.start()

        try:
            request_id = await scheduler.add_request_async(
                prompt="Describe this image briefly",
                images=[test_image_path],
                max_tokens=30,
            )

            tokens_received = 0
            async for output in scheduler.stream_outputs(request_id):
                tokens_received += len(output.new_token_ids)
                if output.finished:
                    break

            assert tokens_received > 0

        finally:
            await scheduler.stop()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
