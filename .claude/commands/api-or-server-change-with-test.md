---
name: api-or-server-change-with-test
description: Workflow command scaffold for api-or-server-change-with-test in vllm-mlx.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /api-or-server-change-with-test

Use this workflow when working on **api-or-server-change-with-test** in `vllm-mlx`.

## Goal

Modifies API or server logic, always updating or adding server-related tests.

## Common Files

- `vllm_mlx/server.py`
- `vllm_mlx/api/*.py`
- `tests/test_server.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit vllm_mlx/server.py and/or vllm_mlx/api/*.py.
- Edit or create tests in tests/test_server.py or related test files.
- Commit both server/api and test changes together.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.