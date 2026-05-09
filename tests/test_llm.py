# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Language Model wrapper."""

import platform
import sys

import pytest

# Skip all tests if not on Apple Silicon or MLX not available
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture
def small_model_name():
    """Return a small model for testing."""
    return "mlx-community/Llama-3.2-1B-Instruct-4bit"


def test_model_init():
    """Test model initialization."""
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel("some-model")
    assert model.model_name == "some-model"
    assert not model._loaded


def test_model_info_not_loaded():
    """Test model info when not loaded."""
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel("some-model")
    info = model.get_model_info()

    assert info["loaded"] is False
    assert info["model_name"] == "some-model"


def test_model_repr():
    """Test model string representation."""
    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel("test-model")
    repr_str = repr(model)

    assert "MLXLanguageModel" in repr_str
    assert "test-model" in repr_str
    assert "not loaded" in repr_str


def test_stream_generate_reports_prompt_tokens(monkeypatch):
    """Streaming output should carry prompt token accounting."""
    from types import SimpleNamespace

    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel("test-model")
    model._loaded = True
    model.model = object()
    model.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
    monkeypatch.setattr(model, "_create_sampler", lambda *_args, **_kwargs: "sampler")
    monkeypatch.setattr(
        model,
        "_create_logits_processors",
        lambda *_args, **_kwargs: None,
    )

    def fake_stream_generate(*_args, **_kwargs):
        yield SimpleNamespace(text="ok", token=42)

    monkeypatch.setattr("mlx_lm.stream_generate", fake_stream_generate)

    [output] = list(model.stream_generate("prompt", max_tokens=1))

    assert output.prompt_tokens == 3


@pytest.mark.slow
def test_model_load(small_model_name):
    """Test loading a model (slow test, downloads model)."""
    pytest.importorskip("mlx_lm")

    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    assert model._loaded
    assert model.model is not None
    assert model.tokenizer is not None


@pytest.mark.slow
def test_model_generate(small_model_name):
    """Test text generation."""
    pytest.importorskip("mlx_lm")

    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    output = model.generate("Hello", max_tokens=10)

    assert output.text is not None
    assert len(output.text) > 0
    assert output.finish_reason is not None


@pytest.mark.slow
def test_model_stream_generate(small_model_name):
    """Test streaming generation."""
    pytest.importorskip("mlx_lm")

    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    chunks = list(model.stream_generate("Hello", max_tokens=10))

    assert len(chunks) > 0
    assert any(chunk.finished for chunk in chunks)


@pytest.mark.slow
def test_model_chat(small_model_name):
    """Test chat interface."""
    pytest.importorskip("mlx_lm")

    from vllm_mlx.models.llm import MLXLanguageModel

    model = MLXLanguageModel(small_model_name)
    model.load()

    messages = [{"role": "user", "content": "Hi"}]
    output = model.chat(messages, max_tokens=10)

    assert output.text is not None
