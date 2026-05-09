# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching system.

These tests verify the scheduler, engine, and request handling
for the vLLM-style continuous batching implementation.
"""

import asyncio
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock

from vllm_mlx.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
)
from vllm_mlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    SchedulingPolicy,
)


class TestRequest:
    """Tests for Request class."""

    def test_request_creation(self):
        """Test basic request creation."""
        params = SamplingParams(max_tokens=100, temperature=0.8)
        request = Request(
            request_id="test-1",
            prompt="Hello, world!",
            sampling_params=params,
        )

        assert request.request_id == "test-1"
        assert request.prompt == "Hello, world!"
        assert request.sampling_params.max_tokens == 100
        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

    def test_request_status_transitions(self):
        """Test request status transitions."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        assert request.status == RequestStatus.WAITING
        assert not request.is_finished()

        request.status = RequestStatus.RUNNING
        assert not request.is_finished()

        request.set_finished(RequestStatus.FINISHED_STOPPED)
        assert request.is_finished()
        assert request.get_finish_reason() == "stop"

    def test_request_output_tokens(self):
        """Test appending output tokens."""
        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3

        assert request.num_output_tokens == 0
        assert request.num_tokens == 3

        request.append_output_token(100)
        request.append_output_token(101)

        assert request.num_output_tokens == 2
        assert request.num_tokens == 5
        assert request.output_token_ids == [100, 101]

    def test_request_comparison(self):
        """Test request comparison for priority queue."""
        req1 = Request(
            request_id="req-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=1.0,
        )
        req2 = Request(
            request_id="req-2",
            prompt="World",
            sampling_params=SamplingParams(),
            priority=1,
            arrival_time=0.5,
        )
        req3 = Request(
            request_id="req-3",
            prompt="Test",
            sampling_params=SamplingParams(),
            priority=0,
            arrival_time=2.0,
        )

        # Lower priority value = higher priority
        assert req1 < req2
        # Same priority, earlier arrival = higher priority
        assert req1 < req3


class TestSchedulerBatchGeneratorCompatibility:
    """Tests for mlx-lm BatchGenerator compatibility shims."""

    def test_scheduler_attaches_prefill_progress_callback_when_ctor_lacks_kwarg(
        self, monkeypatch
    ):
        import vllm_mlx.scheduler as scheduler_module

        captured_kwargs = {}

        class FakeBatchGenerator:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        class TokenizerStub:
            eos_token_id = 2
            eos_token_ids = None

        monkeypatch.setattr(
            scheduler_module,
            "_BATCH_GENERATOR_SUPPORTS_PROMPT_PROGRESS_CALLBACK",
            False,
        )
        monkeypatch.setattr(scheduler_module, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(scheduler_module, "make_sampler", lambda **kwargs: "sampler")

        scheduler = scheduler_module.Scheduler(
            model=MagicMock(),
            tokenizer=TokenizerStub(),
            config=scheduler_module.SchedulerConfig(enable_prefix_cache=False),
        )

        batch_generator = scheduler._create_batch_generator(
            SamplingParams(max_tokens=32)
        )

        assert "prompt_progress_callback" not in captured_kwargs
        assert callable(batch_generator.prompt_progress_callback)

    def test_extract_generation_responses_handles_tuple_shape(self):
        import vllm_mlx.scheduler as scheduler_module

        generated = [object(), object()]

        assert scheduler_module._extract_generation_responses((["prompt"], generated)) == generated
        assert scheduler_module._extract_generation_responses(generated) == generated

    def test_get_batch_generator_active_batch_falls_back_to_generation_batch(self):
        import types

        import vllm_mlx.scheduler as scheduler_module

        generation_batch = object()
        batch_generator = types.SimpleNamespace(
            active_batch=None,
            _generation_batch=generation_batch,
        )

        assert (
            scheduler_module._get_batch_generator_active_batch(batch_generator)
            is generation_batch
        )

    def test_skips_chunked_prefill_when_batch_generator_lacks_private_internals(
        self, monkeypatch
    ):
        import vllm_mlx.scheduler as scheduler_module

        class FakeBatchGenerator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class TokenizerStub:
            eos_token_id = 2
            eos_token_ids = None

        def fail_install(*_args, **_kwargs):
            raise AssertionError("chunked prefill should be skipped")

        monkeypatch.setattr(scheduler_module, "BatchGenerator", FakeBatchGenerator)
        monkeypatch.setattr(scheduler_module, "make_sampler", lambda **kwargs: "sampler")
        monkeypatch.setattr(scheduler_module, "_install_chunked_prefill", fail_install)

        scheduler = scheduler_module.Scheduler(
            model=MagicMock(),
            tokenizer=TokenizerStub(),
            config=scheduler_module.SchedulerConfig(enable_prefix_cache=False),
        )
        scheduler.memory_aware_cache = object()

        batch_generator = scheduler._create_batch_generator(SamplingParams(max_tokens=32))

        assert isinstance(batch_generator, FakeBatchGenerator)

    def test_chunked_prefill_accepts_prompt_checkpoints_tuple_and_max_kv_size(
        self, monkeypatch
    ):
        import importlib

        import mlx.core as mx

        import vllm_mlx.scheduler as scheduler_module

        generate_module = importlib.import_module("mlx_lm.generate")
        captured_make_cache = {}

        class FakeCacheEntry:
            def empty(self):
                return True

        class FakePromptCache:
            state = mx.array([0])

            def finalize(self):
                pass

        class FakeBatchGenerator:
            active_batch = None
            completion_batch_size = 1
            prefill_batch_size = 1

            def __init__(self):
                self.unprocessed_prompts = [
                    (
                        7,
                        [1, 2, 3],
                        8,
                        [FakeCacheEntry()],
                        "sampler",
                        "logits_processor",
                        "prompt_checkpoint",
                    )
                ]
                self._stats = SimpleNamespace(
                    prompt_tokens=0,
                    prompt_time=0.0,
                    generation_time=0.0,
                )
                self.model = lambda *_args, **_kwargs: None
                self.max_kv_size = 64
                self.prompt_progress_callback = lambda *_args, **_kwargs: None

            def _next(self):
                raise AssertionError("original _next should not be used")

            def remove(self, _uids):
                pass

            def _process_prompts(self, _prompts):
                raise AssertionError("direct prompt processing should not be used")

        def fake_make_cache(model, left_padding, max_kv_size):
            captured_make_cache["args"] = (model, left_padding, max_kv_size)
            return [FakePromptCache()]

        monkeypatch.setattr(generate_module, "_make_cache", fake_make_cache)
        monkeypatch.setattr(
            generate_module,
            "Batch",
            lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
            raising=False,
        )
        monkeypatch.setattr(
            generate_module,
            "_left_pad_prompts",
            lambda prompts, max_length: mx.array([prompts[0]]),
        )

        batch_generator = FakeBatchGenerator()
        scheduler_module._install_chunked_prefill(batch_generator, budget=1)

        assert batch_generator._next() == []
        assert captured_make_cache["args"][2] == 64
        assert batch_generator._partial["uids"] == [7]


class TestEngineExecutorAffinity:
    """Tests for thread-stable scheduler stepping in EngineCore."""

    @staticmethod
    def _make_engine(*, use_prefill_executor=True, waiting=0, running=0, batch_generator=None):
        from vllm_mlx.engine_core import EngineConfig, EngineCore

        engine = EngineCore.__new__(EngineCore)
        engine.config = EngineConfig(use_prefill_executor=use_prefill_executor)
        engine.scheduler = SimpleNamespace(
            batch_generator=batch_generator,
            get_num_waiting=lambda: waiting,
            get_num_running=lambda: running,
        )
        engine._scheduler_step_affinity = None
        engine._scheduler_step_affinity_batch_generator_id = None
        return engine

    def test_scheduler_step_uses_executor_for_waiting_prefill(self):
        engine = self._make_engine(waiting=1)

        assert engine._should_run_scheduler_step_in_executor() is True

    def test_scheduler_step_affinity_sticks_until_batch_drains(self):
        batch_generator = SimpleNamespace(
            _partial=None,
            active_batch=None,
            _generation_batch=SimpleNamespace(uids=[1]),
        )
        engine = self._make_engine(running=1, batch_generator=batch_generator)

        engine._record_scheduler_step_execution(used_executor=True)

        assert engine._should_run_scheduler_step_in_executor() is True
        assert engine._scheduler_step_affinity == "executor"
        assert (
            engine._scheduler_step_affinity_batch_generator_id
            == id(batch_generator)
        )

    def test_scheduler_step_affinity_releases_after_idle_drain(self):
        generation_batch = SimpleNamespace(uids=[1])
        batch_generator = SimpleNamespace(
            _partial=None,
            active_batch=None,
            _generation_batch=generation_batch,
        )
        engine = self._make_engine(running=1, batch_generator=batch_generator)

        engine._record_scheduler_step_execution(used_executor=True)

        engine.scheduler = SimpleNamespace(
            batch_generator=batch_generator,
            get_num_waiting=lambda: 0,
            get_num_running=lambda: 0,
        )
        generation_batch.uids = []

        assert engine._should_run_scheduler_step_in_executor() is False
        assert engine._scheduler_step_affinity is None
        assert engine._scheduler_step_affinity_batch_generator_id is None

    def test_scheduler_step_affinity_resets_when_executor_disabled(self):
        batch_generator = SimpleNamespace(
            _partial=None,
            active_batch=None,
            _generation_batch=SimpleNamespace(uids=[1]),
        )
        engine = self._make_engine(
            use_prefill_executor=False,
            running=1,
            batch_generator=batch_generator,
        )
        engine._scheduler_step_affinity = "executor"
        engine._scheduler_step_affinity_batch_generator_id = id(batch_generator)

        assert engine._should_run_scheduler_step_in_executor() is False
        assert engine._scheduler_step_affinity is None
        assert engine._scheduler_step_affinity_batch_generator_id is None


class TestFinishedOutputCacheClearPolicy:
    """Tests for finished-output MLX cache clear cadence controls."""

    @staticmethod
    def _make_engine(*, interval=None, enabled=True, log_timing=False):
        from vllm_mlx.engine_core import EngineConfig, EngineCore

        engine = EngineCore.__new__(EngineCore)
        config_kwargs = {
            "clear_finished_output_cache": enabled,
            "log_finished_output_cache_clear": log_timing,
        }
        if interval is not None:
            config_kwargs["finished_output_cache_clear_interval"] = interval
        engine.config = EngineConfig(**config_kwargs)
        engine._finished_output_cache_clear_events = 0
        engine._finished_output_cache_clear_skips = 0
        return engine

    def test_finished_output_cache_clear_defaults_to_every_fourth_finished_event(
        self, monkeypatch
    ):
        from vllm_mlx import engine_core

        calls = []
        monkeypatch.setattr(engine_core.mx, "clear_cache", lambda: calls.append("clear"))
        engine = self._make_engine()

        assert engine._maybe_clear_finished_output_cache(["r1"]) is False
        assert engine._maybe_clear_finished_output_cache(["r2"]) is False
        assert engine._maybe_clear_finished_output_cache(["r3"]) is False
        assert engine._maybe_clear_finished_output_cache(["r4"]) is True

        assert calls == ["clear"]

    def test_finished_output_cache_clear_interval_one_clears_every_event(
        self, monkeypatch
    ):
        from vllm_mlx import engine_core

        calls = []
        monkeypatch.setattr(engine_core.mx, "clear_cache", lambda: calls.append("clear"))
        engine = self._make_engine(interval=1)

        assert engine._maybe_clear_finished_output_cache(["r1"]) is True
        assert engine._maybe_clear_finished_output_cache(["r2"]) is True

        assert calls == ["clear", "clear"]

    def test_finished_output_cache_clear_interval_skips_until_due(
        self, monkeypatch, caplog
    ):
        from vllm_mlx import engine_core

        calls = []
        monkeypatch.setattr(engine_core.mx, "clear_cache", lambda: calls.append("clear"))
        engine = self._make_engine(interval=3, log_timing=True)

        assert engine._maybe_clear_finished_output_cache(["r1"]) is False
        assert engine._maybe_clear_finished_output_cache(["r2"]) is False
        with caplog.at_level("INFO"):
            assert engine._maybe_clear_finished_output_cache(["r3"]) is True

        assert calls == ["clear"]
        assert engine._finished_output_cache_clear_events == 3
        assert engine._finished_output_cache_clear_skips == 2
        assert "finished_output_cache_clear" in caplog.text


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()

        assert params.max_tokens == 256
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.stop == []
        assert params.stop_token_ids == []

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            stop=["END"],
            stop_token_ids=[1, 2],
        )

        assert params.max_tokens == 100
        assert params.temperature == 0.5
        assert params.top_p == 0.95
        assert params.top_k == 50
        assert params.stop == ["END"]
        assert params.stop_token_ids == [1, 2]


class TestRequestOutput:
    """Tests for RequestOutput."""

    def test_output_creation(self):
        """Test output creation."""
        output = RequestOutput(
            request_id="test-1",
            new_token_ids=[100, 101],
            new_text="Hello",
            output_token_ids=[100, 101],
            output_text="Hello",
            finished=True,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=2,
        )

        assert output.request_id == "test-1"
        assert output.finished
        assert output.finish_reason == "stop"

        usage = output.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 12


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_default_config(self):
        """Test default scheduler config."""
        config = SchedulerConfig()

        assert config.max_num_seqs == 256
        assert config.policy == SchedulingPolicy.FCFS
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 32

    def test_custom_config(self):
        """Test custom scheduler config."""
        config = SchedulerConfig(
            max_num_seqs=64,
            policy=SchedulingPolicy.PRIORITY,
            prefill_batch_size=4,
            completion_batch_size=16,
        )

        assert config.max_num_seqs == 64
        assert config.policy == SchedulingPolicy.PRIORITY


class TestSchedulerBasic:
    """Basic tests for Scheduler (without real model)."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        return MagicMock()

    def test_scheduler_creation(self, mock_model, mock_tokenizer):
        """Test scheduler creation."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=SchedulerConfig(max_num_seqs=10),
        )

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()

    def test_add_request(self, mock_model, mock_tokenizer):
        """Test adding requests to scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello world",
            sampling_params=SamplingParams(max_tokens=10),
        )

        scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 1
        assert scheduler.has_requests()
        assert scheduler.get_request("test-1") is not None

    def test_add_duplicate_request(self, mock_model, mock_tokenizer):
        """Test adding duplicate request raises error."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)

        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(request)

    def test_abort_waiting_request(self, mock_model, mock_tokenizer):
        """Test aborting a waiting request (deferred abort pattern)."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello",
            sampling_params=SamplingParams(),
        )

        scheduler.add_request(request)
        assert scheduler.get_num_waiting() == 1

        # abort_request() enqueues for deferred processing
        result = scheduler.abort_request("test-1")
        assert result is True

        # Process pending aborts (normally happens inside step())
        scheduler._process_pending_aborts()

        assert scheduler.get_num_waiting() == 0
        assert "test-1" in scheduler.finished_req_ids

    def test_abort_running_request_credits_inflight_tokens(
        self, mock_model, mock_tokenizer
    ):
        """Aborted running requests should still contribute to aggregate stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        request = Request(
            request_id="test-1",
            prompt="Hello world",
            sampling_params=SamplingParams(),
        )
        request.prompt_token_ids = [1, 2, 3]
        request.num_prompt_tokens = 3
        request.status = RequestStatus.RUNNING
        request.append_output_token(100)
        request.append_output_token(101)

        scheduler.requests[request.request_id] = request
        scheduler.running[request.request_id] = request

        scheduler._do_abort_request(request.request_id)

        assert scheduler.total_prompt_tokens == 3
        assert scheduler.total_completion_tokens == 2

    def test_abort_nonexistent_request(self, mock_model, mock_tokenizer):
        """Test aborting non-existent request (deferred abort always enqueues)."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # abort_request() always returns True (enqueue is always successful)
        result = scheduler.abort_request("nonexistent")
        assert result is True

    def test_get_stats(self, mock_model, mock_tokenizer):
        """Test getting scheduler stats."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        stats = scheduler.get_stats()

        assert "num_waiting" in stats
        assert "num_running" in stats
        assert "num_requests_processed" in stats
        assert stats["num_waiting"] == 0
        assert stats["num_running"] == 0

    def test_reset(self, mock_model, mock_tokenizer):
        """Test resetting scheduler."""
        scheduler = Scheduler(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Add some requests
        for i in range(5):
            request = Request(
                request_id=f"test-{i}",
                prompt=f"Hello {i}",
                sampling_params=SamplingParams(),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5

        scheduler.reset()

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0
        assert not scheduler.has_requests()


# Integration tests require actual MLX model
@pytest.mark.integration
class TestSchedulerIntegration:
    """Integration tests that require a real model."""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small test model."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    def test_scheduler_with_real_model(self, model_and_tokenizer):
        """Test scheduler with real model."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=4,
                prefill_batch_size=2,
                completion_batch_size=4,
            ),
        )

        # Add a request
        request = Request(
            request_id="test-1",
            prompt="What is 2+2?",
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(request)

        # Run a few steps
        outputs = []
        for _ in range(20):
            output = scheduler.step()
            if output.outputs:
                outputs.extend(output.outputs)
            if output.finished_request_ids:
                break

        assert len(outputs) > 0
        # Check we got at least one output
        final_output = outputs[-1]
        assert final_output.request_id == "test-1"

    def test_multiple_concurrent_requests(self, model_and_tokenizer):
        """Test handling multiple concurrent requests."""
        model, tokenizer = model_and_tokenizer

        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                max_num_seqs=8,
                prefill_batch_size=4,
                completion_batch_size=8,
            ),
        )

        # Add multiple requests
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
        ]

        for i, prompt in enumerate(prompts):
            request = Request(
                request_id=f"test-{i}",
                prompt=prompt,
                sampling_params=SamplingParams(max_tokens=10),
            )
            scheduler.add_request(request)

        # Run until all complete
        finished = set()
        max_steps = 100
        steps = 0

        while len(finished) < len(prompts) and steps < max_steps:
            output = scheduler.step()
            finished.update(output.finished_request_ids)
            steps += 1

        assert len(finished) == len(prompts), f"Only {len(finished)} requests finished"


@pytest.mark.asyncio
class TestEngineAsync:
    """Async tests for the engine."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        tokenizer.decode = lambda x: " ".join(str(t) for t in x)
        tokenizer.eos_token_id = 0
        tokenizer.eos_token_ids = {0}
        return model, tokenizer

    async def test_engine_lifecycle(self, mock_model_and_tokenizer):
        """Test engine start/stop lifecycle."""
        from vllm_mlx.engine import AsyncEngineCore, EngineConfig

        model, tokenizer = mock_model_and_tokenizer

        engine = AsyncEngineCore(model, tokenizer, EngineConfig())

        assert not engine.engine.is_running()

        # Use async context manager
        async with engine:
            assert engine.engine.is_running()
            await asyncio.sleep(0.05)

        assert not engine.engine.is_running()

    async def test_engine_context_manager(self, mock_model_and_tokenizer):
        """Test engine as async context manager."""
        from vllm_mlx.engine import AsyncEngineCore

        model, tokenizer = mock_model_and_tokenizer

        async with AsyncEngineCore(model, tokenizer) as engine:
            assert engine.engine.is_running()

        assert not engine.engine.is_running()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
