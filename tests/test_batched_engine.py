# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchedEngine generate() output."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
