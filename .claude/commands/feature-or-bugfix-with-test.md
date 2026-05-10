---
name: feature-or-bugfix-with-test
description: Workflow command scaffold for feature-or-bugfix-with-test in vllm-mlx.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-or-bugfix-with-test

Use this workflow when working on **feature-or-bugfix-with-test** in `vllm-mlx`.

## Goal

Implements a new feature or fixes a bug, always accompanied by or updating relevant tests.

## Common Files

- `vllm_mlx/**/*.py`
- `tests/**/*.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or create implementation files in vllm_mlx/ or subdirectories.
- Edit or create test files in tests/ corresponding to the change.
- Commit both code and test changes together.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.