# SPDX-License-Identifier: Apache-2.0
"""Tests for audio endpoint resource limits."""

import ast
from pathlib import Path

import pytest
from fastapi import HTTPException

from vllm_mlx.audio_limits import (
    save_upload_with_limit,
    validate_tts_input_length,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "vllm_mlx" / "cli.py"
SERVER_PATH = REPO_ROOT / "vllm_mlx" / "server.py"


class FakeUpload:
    def __init__(self, chunks: list[bytes], filename: str = "audio.wav"):
        self._chunks = list(chunks)
        self.filename = filename

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


class TestAudioUploadLimits:
    @pytest.mark.asyncio
    async def test_save_upload_with_limit_writes_file(self):
        upload = FakeUpload([b"a" * 8, b"b" * 4])

        path = await save_upload_with_limit(upload, max_bytes=32)

        try:
            assert Path(path).read_bytes() == b"a" * 8 + b"b" * 4
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_save_upload_with_limit_rejects_oversize_and_cleans_up(self):
        upload = FakeUpload([b"a" * 16, b"b" * 16, b"c"])

        with pytest.raises(HTTPException) as exc_info:
            await save_upload_with_limit(upload, max_bytes=32)

        assert exc_info.value.status_code == 413
        assert "Audio upload too large" in exc_info.value.detail


class TestTTSInputLimits:
    def test_validate_tts_input_length_accepts_short_text(self):
        validate_tts_input_length("hello", max_chars=16)

    def test_validate_tts_input_length_rejects_oversized_text(self):
        with pytest.raises(HTTPException) as exc_info:
            validate_tts_input_length("x" * 17, max_chars=16)

        assert exc_info.value.status_code == 413
        assert "TTS input too long" in exc_info.value.detail


class TestAudioLimitParsers:
    def test_top_level_cli_exposes_audio_limit_flags(self):
        tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))

        defaults = _find_audio_limit_flag_defaults(tree, parser_name="serve_parser")

        assert defaults["--max-audio-upload-mb"] == "DEFAULT_MAX_AUDIO_UPLOAD_MB"
        assert defaults["--max-tts-input-chars"] == "DEFAULT_MAX_TTS_INPUT_CHARS"

    def test_standalone_server_parser_exposes_audio_limit_flags(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))

        defaults = _find_audio_limit_flag_defaults(tree, parser_name="parser")

        assert defaults["--max-audio-upload-mb"] == "DEFAULT_MAX_AUDIO_UPLOAD_MB"
        assert defaults["--max-tts-input-chars"] == "DEFAULT_MAX_TTS_INPUT_CHARS"


def _find_audio_limit_flag_defaults(
    tree: ast.AST,
    *,
    parser_name: str,
) -> dict[str, str]:
    expected_flags = {"--max-audio-upload-mb", "--max-tts-input-chars"}
    defaults: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != parser_name:
            continue
        if not node.args:
            continue

        arg0 = node.args[0]
        if not isinstance(arg0, ast.Constant):
            continue
        if arg0.value not in expected_flags:
            continue

        default_kw = next((kw for kw in node.keywords if kw.arg == "default"), None)
        assert default_kw is not None
        assert isinstance(default_kw.value, ast.Name)
        defaults[arg0.value] = default_kw.value.id

    assert defaults.keys() == expected_flags
    return defaults
