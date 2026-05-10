# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_mlx.engine.batched import _normalize_tool_call_arguments_for_template


class TestBatchedEngineGenerate:
    """Test BatchedEngine.generate() output fields."""

    def _make_engine(self, *, is_mllm: bool = False):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=is_mllm):
            engine = BatchedEngine("test-model")

        engine._loaded = True
        engine._is_mllm = is_mllm
        return engine

    @pytest.mark.anyio
    async def test_llm_tokens_field_is_populated(self):
        engine = self._make_engine()
        token_ids = [3681, 374, 279]

        mock_output = MagicMock()
        mock_output.output_text = "Paris"
        mock_output.output_token_ids = token_ids
        mock_output.prompt_tokens = 7
        mock_output.completion_tokens = 3
        mock_output.finish_reason = "stop"
        mock_output.stop_reason = None
        mock_output.stop_reason_detail = None

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=mock_output)
        engine._engine = mock_engine

        result = await engine.generate(
            prompt="What is the capital of France?",
            max_tokens=10,
        )

        assert result.tokens == token_ids
        assert result.text == "Paris"
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 3
        assert result.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_tokens_field_is_populated(self):
        engine = self._make_engine(is_mllm=True)
        token_ids = [101, 102]

        engine._mllm_scheduler = SimpleNamespace(
            generate=AsyncMock(
                return_value=SimpleNamespace(
                    output_text="Done",
                    output_token_ids=token_ids,
                    prompt_tokens=5,
                    completion_tokens=2,
                    finish_reason="stop",
                )
            )
        )

        result = await engine.generate(prompt="Describe this", max_tokens=10)

        assert result.tokens == token_ids
        assert result.text == "Done"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.finish_reason == "stop"

    @pytest.mark.anyio
    async def test_mllm_chat_routes_audio_from_messages(self):
        engine = self._make_engine(is_mllm=True)
        engine._processor = MagicMock()
        engine._processor.apply_chat_template.return_value = "formatted prompt"
        engine._processor.tokenizer = MagicMock()
        engine._mllm_scheduler = SimpleNamespace(
            generate=AsyncMock(
                return_value=SimpleNamespace(
                    output_text="audio answer",
                    output_token_ids=[7],
                    prompt_tokens=4,
                    completion_tokens=1,
                    finish_reason="stop",
                )
            )
        )

        await engine.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this clip?"},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/wav;base64,AAAA"},
                        },
                    ],
                }
            ],
            max_tokens=10,
        )

        engine._mllm_scheduler.generate.assert_awaited_once()
        assert engine._mllm_scheduler.generate.await_args.kwargs["audio"] == [
            "data:audio/wav;base64,AAAA"
        ]


class TestBatchedEngineAbortRequest:
    @pytest.mark.anyio
    async def test_abort_request_routes_to_mllm_scheduler(self):
        engine = TestBatchedEngineGenerate()._make_engine(is_mllm=True)
        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.abort_request.return_value = True

        assert await engine.abort_request("req-1") is True
        engine._mllm_scheduler.abort_request.assert_called_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_routes_to_text_engine(self):
        engine = TestBatchedEngineGenerate()._make_engine()
        engine._engine = MagicMock()
        engine._engine.abort_request.return_value = True

        assert await engine.abort_request("req-1") is True
        engine._engine.abort_request.assert_called_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_routes_to_async_text_engine(self):
        engine = TestBatchedEngineGenerate()._make_engine()
        engine._engine = MagicMock()
        engine._engine.abort_request = AsyncMock(return_value=True)

        assert await engine.abort_request("req-1") is True
        engine._engine.abort_request.assert_awaited_once_with("req-1")

    @pytest.mark.anyio
    async def test_abort_request_returns_false_without_supported_engine(self):
        engine = TestBatchedEngineGenerate()._make_engine()
        engine._engine = None

        assert await engine.abort_request("req-1") is False


class TestBatchedEngineStats:
    """Tests for status/stat promotion from wrapped scheduler components."""

    def test_mllm_batch_generator_stats_are_promoted(self):
        engine = TestBatchedEngineGenerate()._make_engine(is_mllm=True)
        engine._mllm_scheduler = SimpleNamespace(
            get_stats=lambda: {
                "batch_generator": {
                    "prompt_tps": 41.5,
                    "generation_tps": 19.25,
                },
                "num_running": 2,
            }
        )

        stats = engine.get_stats()

        assert stats["batch_generator"] == {
            "prompt_tps": 41.5,
            "generation_tps": 19.25,
        }


class TestBatchedEngineMllmStartup:
    @pytest.mark.anyio
    async def test_mllm_start_honors_prefix_cache_config(self, monkeypatch):
        from vllm_mlx.engine.batched import BatchedEngine

        captured = {}

        class FakeMLXMultimodalLM:
            def __init__(self, *args, **kwargs):
                captured["model_args"] = args
                captured["model_kwargs"] = kwargs
                self.model = object()
                self.processor = object()

            def load(self):
                captured["model_loaded"] = True

        class FakeMLLMSchedulerConfig:
            def __init__(self, **kwargs):
                captured["scheduler_config"] = kwargs

        class FakeMLLMScheduler:
            def __init__(self, *, model, processor, config):
                captured["scheduler"] = {
                    "model": model,
                    "processor": processor,
                    "config": config,
                }

            async def start(self):
                captured["scheduler_started"] = True

        monkeypatch.setattr("vllm_mlx.models.mllm.MLXMultimodalLM", FakeMLXMultimodalLM)
        monkeypatch.setattr(
            "vllm_mlx.mllm_scheduler.MLLMSchedulerConfig",
            FakeMLLMSchedulerConfig,
        )
        monkeypatch.setattr(
            "vllm_mlx.mllm_scheduler.MLLMScheduler",
            FakeMLLMScheduler,
        )

        engine = BatchedEngine(
            "test-model",
            force_mllm=True,
            scheduler_config=SimpleNamespace(
                max_num_seqs=2,
                prefill_batch_size=1,
                completion_batch_size=3,
                enable_vision_cache=True,
                vision_cache_size=4,
                mllm_prefill_step_size=128,
                max_kv_size=99,
                enable_prefix_cache=False,
                prefix_cache_size=17,
            ),
        )

        await engine._start_mllm()

        assert captured["model_kwargs"].get("enable_cache") is False
        assert captured["model_kwargs"].get("cache_size") == 17
        assert captured["model_kwargs"]["max_kv_size"] == 99
        assert captured["model_loaded"] is True
        assert captured["scheduler_started"] is True


class TestToolCallReplayNormalization:
    """Tests for OpenAI tool-call replay normalization before chat templating."""

    def test_parses_function_arguments_string_to_mapping(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "city": "Tokyo"
        }
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
            '{"city": "Tokyo"}'
        )

    def test_wraps_non_mapping_arguments_for_template_items(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "echo",
                            "arguments": '["not", "object"]',
                        }
                    }
                ],
            }
        ]

        normalized = _normalize_tool_call_arguments_for_template(messages)

        assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {
            "value": ["not", "object"]
        }


class TestBatchedEngineChatTemplate:
    """Tests for batched chat template selection."""

    def test_llm_applies_chat_template_kwargs(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-llm-model")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt"
        engine._tokenizer = tokenizer
        engine._is_mllm = False

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            chat_template_kwargs={"enable_thinking": False, "custom_flag": "x"},
        )

        assert prompt == "prompt"
        _, kwargs = tokenizer.apply_chat_template.call_args
        assert kwargs["enable_thinking"] is False
        assert kwargs["custom_flag"] == "x"

    def test_mllm_falls_back_to_tokenizer_when_processor_has_no_template(self):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            engine = BatchedEngine("test-mllm-model")

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt-from-tokenizer"

        processor = MagicMock()
        processor.tokenizer = tokenizer
        processor.apply_chat_template.side_effect = ValueError(
            "Cannot use apply_chat_template because this processor does not "
            "have a chat template."
        )

        engine._is_mllm = True
        engine._processor = processor

        prompt = engine._apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            enable_thinking=False,
        )

        assert prompt == "prompt-from-tokenizer"
        processor.apply_chat_template.assert_called_once()
        tokenizer.apply_chat_template.assert_called_once()
