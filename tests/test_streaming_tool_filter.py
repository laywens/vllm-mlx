# SPDX-License-Identifier: Apache-2.0
"""Tests for suppressing tool-call markup from text streams."""

from vllm_mlx.api.utils import StreamingToolCallFilter


class TestStreamingToolCallFilter:
    def test_normal_text_passes_through(self):
        tool_filter = StreamingToolCallFilter()

        assert tool_filter.process("Hello world") == "Hello world"
        assert tool_filter.flush() == ""

    def test_minimax_tool_call_suppressed(self):
        tool_filter = StreamingToolCallFilter()

        assert tool_filter.process("Before <minimax:tool_call>") == "Before "
        assert tool_filter.process('<invoke name="read">') == ""
        assert tool_filter.process("</invoke></minimax:tool_call>After") == "After"

    def test_qwen_xml_tool_call_suppressed_when_split(self):
        tool_filter = StreamingToolCallFilter()

        emitted = [
            tool_filter.process("Before <tool"),
            tool_filter.process('_call>{"name":"read"}</tool_call> After'),
            tool_filter.flush(),
        ]

        assert "".join(emitted) == "Before  After"

    def test_llama_function_tool_call_suppressed(self):
        tool_filter = StreamingToolCallFilter()

        emitted = tool_filter.process(
            'Before <function=read>{"path":"/tmp/a"}</function> After'
        )

        assert emitted == "Before  After"

    def test_qwen_bracket_tool_call_suppressed_until_newline(self):
        tool_filter = StreamingToolCallFilter()

        emitted = tool_filter.process(
            'Before [Calling tool: read({"path":"/tmp/a"})]\nAfter'
        )

        assert emitted == "Before After"

    def test_tool_call_block_tag_suppressed(self):
        tool_filter = StreamingToolCallFilter()

        emitted = tool_filter.process(
            'Before [TOOL_CALL]{"name":"read"}[/TOOL_CALL] After'
        )

        assert emitted == "Before  After"

    def test_multiple_tool_calls_and_think_tags(self):
        tool_filter = StreamingToolCallFilter()

        emitted = tool_filter.process(
            "<think>plan</think>"
            '<function=read>{"path":"a"}</function>'
            " text "
            '<tool_call>{"name":"write","arguments":{}}</tool_call>'
            "done"
        )

        assert emitted == "<think>plan</think> text done"

    def test_flush_partial_open_tag_emits_non_tool_text(self):
        tool_filter = StreamingToolCallFilter()

        assert tool_filter.process("text <fun") == "text "
        assert tool_filter.flush() == "<fun"

    def test_flush_unterminated_block_discards_tool_content(self):
        tool_filter = StreamingToolCallFilter()

        assert tool_filter.process("text <function=read>partial") == "text "
        assert tool_filter.flush() == ""
