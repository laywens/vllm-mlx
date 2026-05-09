# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded per-sequence KV cache configuration."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


CLI_PATH = Path(__file__).resolve().parents[1] / "vllm_mlx" / "cli.py"


def _serve_parser_flag(flag: str) -> ast.Call | None:
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "serve_parser":
            continue
        if not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and arg0.value == flag:
            return node
    return None


def test_serve_parser_includes_max_kv_size_flag():
    node = _serve_parser_flag("--max-kv-size")
    assert node is not None

    type_kw = next((kw for kw in node.keywords if kw.arg == "type"), None)
    default_kw = next((kw for kw in node.keywords if kw.arg == "default"), None)
    assert type_kw is not None
    assert isinstance(type_kw.value, ast.Name)
    assert type_kw.value.id == "int"
    assert default_kw is not None
    assert isinstance(default_kw.value, ast.Constant)
    assert default_kw.value.value is None


def test_load_model_receives_max_kv_size_from_cli_args():
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "load_model":
            continue

        max_kv_kw = next((kw for kw in node.keywords if kw.arg == "max_kv_size"), None)
        if max_kv_kw is None:
            continue
        assert isinstance(max_kv_kw.value, ast.Attribute)
        assert isinstance(max_kv_kw.value.value, ast.Name)
        assert max_kv_kw.value.value.id == "args"
        assert max_kv_kw.value.attr == "max_kv_size"
        return

    raise AssertionError("load_model call with max_kv_size=args.max_kv_size not found")


def test_scheduler_config_accepts_max_kv_size():
    from vllm_mlx.scheduler import SchedulerConfig

    assert SchedulerConfig().max_kv_size == 0
    assert SchedulerConfig(max_kv_size=65536).max_kv_size == 65536
    with pytest.raises(ValueError, match="max_kv_size"):
        SchedulerConfig(max_kv_size=-1)


def test_scheduler_passes_max_kv_size_to_batch_generator(monkeypatch):
    import vllm_mlx.scheduler as scheduler_module
    from vllm_mlx.request import SamplingParams
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    captured = {}

    class FakeBatchGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(scheduler_module, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(scheduler_module, "_BATCH_GENERATOR_SUPPORTS_MAX_KV_SIZE", True)

    scheduler = Scheduler(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        config=SchedulerConfig(enable_prefix_cache=False, max_kv_size=8192),
    )
    scheduler._ensure_batch_generator(SamplingParams(max_tokens=4))

    assert captured["max_kv_size"] == 8192


def test_mllm_scheduler_config_accepts_max_kv_size():
    from vllm_mlx.mllm_scheduler import MLLMSchedulerConfig

    assert MLLMSchedulerConfig().max_kv_size == 0
    assert MLLMSchedulerConfig(max_kv_size=32768).max_kv_size == 32768
    with pytest.raises(ValueError, match="max_kv_size"):
        MLLMSchedulerConfig(max_kv_size=-1)


def test_simple_engine_stores_max_kv_size():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine("test-model", max_kv_size=4096)

    assert engine._max_kv_size == 4096


def test_server_load_model_passes_max_kv_size_to_simple_engine(monkeypatch):
    import vllm_mlx.server as server

    captured = {}

    class FakeSimpleEngine:
        is_mllm = False

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            return None

        @property
        def preserve_native_tool_format(self):
            return False

        @preserve_native_tool_format.setter
        def preserve_native_tool_format(self, value):
            self._preserve_native_tool_format = value

    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(
        server,
        "_batch_divergence_state",
        SimpleNamespace(reset=lambda **_: None),
    )
    monkeypatch.setattr(server, "SimpleEngine", FakeSimpleEngine)

    server.load_model("test-model", use_batching=False, max_kv_size=4096)

    assert captured["max_kv_size"] == 4096
