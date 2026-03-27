"""Tests for local-first serve host binding behavior."""

from __future__ import annotations

import ast
from pathlib import Path

from vllm_mlx.cli import _resolve_bind_host

CLI_PATH = Path(__file__).resolve().parents[1] / "vllm_mlx" / "cli.py"


def test_resolve_bind_host_respects_localhost_precedence():
    assert _resolve_bind_host("0.0.0.0", localhost=False) == "0.0.0.0"
    assert _resolve_bind_host("192.168.1.10", localhost=False) == "192.168.1.10"

    assert _resolve_bind_host("0.0.0.0", localhost=True) == "127.0.0.1"
    assert _resolve_bind_host("192.168.1.10", localhost=True) == "127.0.0.1"


def test_serve_parser_includes_localhost_flag():
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
        if isinstance(arg0, ast.Constant) and arg0.value == "--localhost":
            action_kw = next((kw for kw in node.keywords if kw.arg == "action"), None)
            assert action_kw is not None
            assert isinstance(action_kw.value, ast.Constant)
            assert action_kw.value.value == "store_true"
            return

    raise AssertionError("--localhost flag not found in serve_parser")


def test_serve_parser_includes_security_hardening_flags():
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))

    expected_flags = {
        "--trust-remote-code",
        "--allow-local-media-paths",
        "--allow-private-media-hosts",
    }
    found_flags = set()

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
        if isinstance(arg0, ast.Constant) and arg0.value in expected_flags:
            action_kw = next((kw for kw in node.keywords if kw.arg == "action"), None)
            assert action_kw is not None
            assert isinstance(action_kw.value, ast.Constant)
            assert action_kw.value.value == "store_true"
            found_flags.add(arg0.value)

    assert found_flags == expected_flags


def test_load_model_receives_trust_remote_code_from_cli_args():
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "load_model":
            continue

        trust_kw = next((kw for kw in node.keywords if kw.arg == "trust_remote_code"), None)
        if trust_kw is None:
            continue
        assert isinstance(trust_kw.value, ast.Attribute)
        assert isinstance(trust_kw.value.value, ast.Name)
        assert trust_kw.value.value.id == "args"
        assert trust_kw.value.attr == "trust_remote_code"
        return

    raise AssertionError("load_model call with trust_remote_code=args.trust_remote_code not found")


def test_uvicorn_bind_uses_resolved_host():
    tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "uvicorn" or node.func.attr != "run":
            continue

        host_kw = next((kw for kw in node.keywords if kw.arg == "host"), None)
        assert host_kw is not None
        assert isinstance(host_kw.value, ast.Name)
        assert host_kw.value.id == "bind_host"
        return

    raise AssertionError("uvicorn.run call not found")
