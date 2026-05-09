# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniMax tool call parsing."""

import json

from vllm_mlx.api.tool_calling import parse_tool_calls


class TestMiniMaxToolCallParsing:
    """Test parsing of MiniMax-style tool calls."""

    def test_single_tool_call(self):
        text = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
<parameter name="units">celsius</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        args = json.loads(tool_calls[0].function.arguments)
        assert args["city"] == "Wanaka"
        assert args["units"] == "celsius"
        assert cleaned == ""

    def test_tool_call_with_surrounding_text(self):
        text = """Let me check the weather for you.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert "Let me check" in cleaned

    def test_multiple_tool_calls(self):
        text = """<minimax:tool_call>
<invoke name="search">
<parameter name="query">MiniMax M2.5</parameter>
</invoke>
</minimax:tool_call>
<minimax:tool_call>
<invoke name="read_file">
<parameter name="path">/tmp/test.txt</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "search"
        assert tool_calls[1].function.name == "read_file"
        assert cleaned == ""

    def test_json_parameter_value(self):
        text = """<minimax:tool_call>
<invoke name="create_event">
<parameter name="title">Meeting</parameter>
<parameter name="attendees">["stuart", "frida"]</parameter>
</invoke>
</minimax:tool_call>"""

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        args = json.loads(tool_calls[0].function.arguments)
        assert args["title"] == "Meeting"
        assert args["attendees"] == ["stuart", "frida"]

    def test_numeric_parameter(self):
        text = """<minimax:tool_call>
<invoke name="set_temperature">
<parameter name="value">42</parameter>
</invoke>
</minimax:tool_call>"""

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        args = json.loads(tool_calls[0].function.arguments)
        assert args["value"] == 42

    def test_no_parameters(self):
        text = """<minimax:tool_call>
<invoke name="get_time">
</invoke>
</minimax:tool_call>"""

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].function.name == "get_time"
        args = json.loads(tool_calls[0].function.arguments)
        assert args == {}

    def test_with_think_tags_preserved(self):
        text = """<think>
I should check the weather first.
</think>
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Wanaka</parameter>
</invoke>
</minimax:tool_call>"""

        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert "<think>" in cleaned

    def test_no_minimax_tool_calls(self):
        text = "Just a regular message with no tool calls."

        cleaned, tool_calls = parse_tool_calls(text)

        assert tool_calls is None
        assert cleaned == text

    def test_tool_call_id_format(self):
        text = """<minimax:tool_call>
<invoke name="test">
<parameter name="x">1</parameter>
</invoke>
</minimax:tool_call>"""

        _, tool_calls = parse_tool_calls(text)

        assert tool_calls is not None
        assert tool_calls[0].id.startswith("call_")
        assert tool_calls[0].type == "function"
