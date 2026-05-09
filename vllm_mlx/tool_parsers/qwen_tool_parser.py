# SPDX-License-Identifier: Apache-2.0
"""
Qwen tool call parser for vllm-mlx.

Handles Qwen's tool calling formats:
- XML style: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
- Bracket style: [Calling tool: func_name({"arg": "value"})]
- Function style: <function=name><parameter=key>value</parameter></function>
"""

import ast
import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _parse_param_value(value: str) -> Any:
    """Parse JSON/Python literal parameter values, otherwise keep text."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, set):
            parsed = sorted(parsed, key=str)
        if isinstance(parsed, (bytes, complex)):
            return value
        json.dumps(parsed)
        return parsed
    except (SyntaxError, ValueError, TypeError):
        return value


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["qwen", "qwen3"])
class QwenToolParser(ToolParser):
    """
    Tool call parser for Qwen models.

    Supports multiple Qwen tool call formats:
    - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Bracket: [Calling tool: func_name({"arg": "value"})]
    - Function: <function=name><parameter=key>value</parameter></function>

    Used when --enable-auto-tool-choice --tool-call-parser qwen are set.
    """

    # Pattern for XML-style: <tool_call>{"json"}</tool_call>
    XML_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # Pattern for bracket-style: [Calling tool: func_name({...})]
    BRACKET_PATTERN = re.compile(r"\[Calling tool:\s*(\w+)\((\{.*?\})\)\]", re.DOTALL)

    # Pattern for function-style: <function=name>...</function>
    FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)

    # Pattern for parameter extraction: <parameter=key>value</parameter>
    PARAM_PATTERN = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Qwen model response.
        """
        tool_calls = []

        # Strip <think> tags first (fallback when no reasoning parser)
        cleaned_text = self.strip_think_tags(model_output)

        # Try bracket pattern first (Qwen3 style)
        bracket_matches = self.BRACKET_PATTERN.findall(cleaned_text)
        for name, args_str in bracket_matches:
            try:
                arguments = json.loads(args_str)
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": (
                            json.dumps(arguments, ensure_ascii=False)
                            if isinstance(arguments, dict)
                            else str(arguments)
                        ),
                    }
                )
            except json.JSONDecodeError:
                continue

        if bracket_matches:
            cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text).strip()

        # Try XML pattern (traditional Qwen style)
        xml_matches = self.XML_PATTERN.findall(cleaned_text)
        for match in xml_matches:
            try:
                data = json.loads(match)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": name,
                            "arguments": (
                                json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict)
                                else str(arguments)
                            ),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if xml_matches:
            cleaned_text = self.XML_PATTERN.sub("", cleaned_text).strip()

        if not tool_calls:
            func_matches = self.FUNCTION_PATTERN.findall(cleaned_text)
            for name, params_block in func_matches:
                arguments: dict[str, Any]
                params_block = params_block.strip()
                if params_block.startswith("{"):
                    try:
                        arguments = json.loads(params_block)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = {
                        param_name.strip(): _parse_param_value(param_value.strip())
                        for param_name, param_value in self.PARAM_PATTERN.findall(
                            params_block
                        )
                    }

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": name.strip(),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )

            if func_matches:
                cleaned_text = self.FUNCTION_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    _PARTIAL_MARKERS = ("<function", "[Calling tool", "<tool_call")

    def _get_partial_marker_len(self, text: str) -> int:
        tail = text[-20:]
        best = 0
        for marker in self._PARTIAL_MARKERS:
            for length in range(len(marker), 0, -1):
                if tail.endswith(marker[:length]):
                    best = max(best, length)
                    break
        return best

    def _has_partial_marker(self, text: str) -> bool:
        return self._get_partial_marker_len(text) > 0

    def _was_buffering(self, previous_text: str) -> bool:
        return self._has_partial_marker(previous_text)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming Qwen model output.
        """
        # Check for tool call markers
        has_tool_marker = (
            "<tool_call>" in current_text
            or "[Calling tool:" in current_text
            or "<function=" in current_text
        )

        if not has_tool_marker:
            if self._has_partial_marker(current_text):
                marker_len = self._get_partial_marker_len(current_text)
                marker_start = len(current_text) - marker_len
                safe_chars = marker_start - len(previous_text)
                if safe_chars > 0:
                    return {"content": delta_text[:safe_chars]}
                return None

            if self._was_buffering(previous_text):
                for marker in self._PARTIAL_MARKERS:
                    for length in range(len(marker), 0, -1):
                        prefix = marker[:length]
                        if previous_text.endswith(prefix):
                            return {"content": prefix + delta_text}
            return {"content": delta_text}

        if "<function=" in current_text:
            close_count = current_text.count("</function>")
            previous_close_count = previous_text.count("</function>")

            if current_text.count("<function=") > close_count:
                return None

            if close_count > previous_close_count:
                result = self.extract_tool_calls(current_text)
                if result.tools_called:
                    new_calls = result.tool_calls[previous_close_count:]
                    if new_calls:
                        return {
                            "tool_calls": [
                                {
                                    "index": previous_close_count + i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for i, tc in enumerate(new_calls)
                            ]
                        }

            return None

        # If we're in a tool call, accumulate and parse at the end. Check the
        # accumulated text because closing markers often split across chunks.
        if "</tool_call>" in current_text or ")]" in current_text:
            # Tool call complete, parse the whole thing
            result = self.extract_tool_calls(current_text)
            if result.tools_called:
                return {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for i, tc in enumerate(result.tool_calls)
                    ]
                }

        return None
