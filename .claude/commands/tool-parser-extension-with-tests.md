---
name: tool-parser-extension-with-tests
description: Workflow command scaffold for tool-parser-extension-with-tests in vllm-mlx.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /tool-parser-extension-with-tests

Use this workflow when working on **tool-parser-extension-with-tests** in `vllm-mlx`.

## Goal

Adds or updates a tool parser and ensures coverage with parser-specific tests.

## Common Files

- `vllm_mlx/tool_parsers/*.py`
- `tests/test_tool_parsers.py`
- `vllm_mlx/tool_parsers/__init__.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add files in vllm_mlx/tool_parsers/ (e.g., qwen_tool_parser.py, qwen3_xml_tool_parser.py, etc.).
- Edit or add corresponding tests in tests/test_tool_parsers.py or related files.
- Optionally update __init__.py to register new parsers.
- Commit parser and test changes together.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.