```markdown
# vllm-mlx Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you the key development practices, coding conventions, and collaborative workflows used in the `vllm-mlx` Python codebase. You'll learn how to structure code, write and organize tests, follow commit conventions, and contribute features or fixes in a way that aligns with the project's standards. This guide is especially useful for contributors aiming to make effective, maintainable, and well-tested changes.

## Coding Conventions

- **Language:** Python
- **Framework:** None detected
- **File Naming:** Use `snake_case` for all file and directory names.
  - Example: `model_workflow.py`, `test_tool_parsers.py`
- **Import Style:** Prefer relative imports within the package.
  - Example:
    ```python
    from .scheduler import Scheduler
    from ..tool_parsers import qwen_tool_parser
    ```
- **Export Style:** Use named exports; avoid wildcard (`*`) exports.
  - Example:
    ```python
    __all__ = ["Scheduler", "Batcher"]
    ```
- **Commit Messages:** Use [Conventional Commits](https://www.conventionalcommits.org/), with prefixes like `fix:` and `feat:`.
  - Example: `feat: add Qwen3 XML tool parser support`

## Workflows

### Feature or Bugfix with Test
**Trigger:** When adding a new feature or fixing a bug, and ensuring it is covered by tests.  
**Command:** `/feature-with-test`

1. Edit or create implementation files in `vllm_mlx/` or its subdirectories.
2. Edit or create corresponding test files in `tests/`.
3. Commit both code and test changes together.

**Example:**
```python
# vllm_mlx/new_feature.py
def new_feature():
    return "Hello, vllm-mlx!"
```
```python
# tests/test_new_feature.py
from vllm_mlx.new_feature import new_feature

def test_new_feature():
    assert new_feature() == "Hello, vllm-mlx!"
```

---

### API or Server Change with Test
**Trigger:** When modifying API or server logic and updating/adding server-related tests.  
**Command:** `/server-api-change`

1. Edit `vllm_mlx/server.py` and/or files in `vllm_mlx/api/`.
2. Edit or create tests in `tests/test_server.py` or related files.
3. Commit both server/api and test changes together.

**Example:**
```python
# vllm_mlx/server.py
def start_server():
    pass  # server logic here
```
```python
# tests/test_server.py
from vllm_mlx.server import start_server

def test_start_server():
    assert start_server() is None
```

---

### Tool Parser Extension with Tests
**Trigger:** When adding or updating a tool parser and ensuring coverage with parser-specific tests.  
**Command:** `/add-tool-parser`

1. Edit or add files in `vllm_mlx/tool_parsers/` (e.g., `qwen_tool_parser.py`).
2. Edit or add corresponding tests in `tests/test_tool_parsers.py` or related files.
3. Optionally update `vllm_mlx/tool_parsers/__init__.py` to register new parsers.
4. Commit parser and test changes together.

**Example:**
```python
# vllm_mlx/tool_parsers/my_tool_parser.py
def parse_tool(data):
    return data.upper()
```
```python
# tests/test_tool_parsers.py
from vllm_mlx.tool_parsers.my_tool_parser import parse_tool

def test_parse_tool():
    assert parse_tool("abc") == "ABC"
```

---

### CLI or Model Workflow Feature
**Trigger:** When implementing or extending CLI/model artifact workflows, updating documentation and tests.  
**Command:** `/cli-model-workflow`

1. Edit `vllm_mlx/cli.py` and/or `vllm_mlx/model_workflow.py`.
2. Edit or add documentation in `docs/reference/cli.md` or `README.md`.
3. Edit or add tests in `tests/test_model_workflow.py`.
4. Commit all related changes together.

**Example:**
```python
# vllm_mlx/cli.py
def main():
    print("vllm-mlx CLI")
```
```markdown
<!-- docs/reference/cli.md -->
# CLI Reference
Usage: `python -m vllm_mlx.cli`
```
```python
# tests/test_model_workflow.py
def test_cli_main(capsys):
    from vllm_mlx.cli import main
    main()
    captured = capsys.readouterr()
    assert "vllm-mlx CLI" in captured.out
```

---

### Scheduler or Batching Change with Test
**Trigger:** When modifying batching or scheduler logic and updating/adding batching-related tests.  
**Command:** `/scheduler-batching-change`

1. Edit `vllm_mlx/scheduler.py` and/or `vllm_mlx/mllm_scheduler.py`.
2. Edit or create tests in `tests/test_batching.py` or `tests/test_mllm_continuous_batching.py`.
3. Commit both code and test changes together.

**Example:**
```python
# vllm_mlx/scheduler.py
class Scheduler:
    def schedule(self, jobs):
        return sorted(jobs)
```
```python
# tests/test_batching.py
from vllm_mlx.scheduler import Scheduler

def test_schedule():
    s = Scheduler()
    assert s.schedule([3, 1, 2]) == [1, 2, 3]
```

## Testing Patterns

- **Test File Naming:** Use `snake_case`, prefix with `test_`, and place in the `tests/` directory.
  - Example: `tests/test_server.py`, `tests/test_tool_parsers.py`
- **Test Framework:** Not explicitly specified; likely uses `pytest` or standard Python `unittest`.
- **Test Structure:** Each new feature, bugfix, or workflow change should be accompanied by or covered with corresponding tests.
- **Test Example:**
  ```python
  def test_functionality():
      assert my_function() == expected_result
  ```

## Commands

| Command                    | Purpose                                                         |
|----------------------------|-----------------------------------------------------------------|
| /feature-with-test         | Add a new feature or bugfix with corresponding tests            |
| /server-api-change         | Modify API/server logic and update/add server-related tests      |
| /add-tool-parser           | Add or update a tool parser and its tests                       |
| /cli-model-workflow        | Implement or extend CLI/model workflows, update docs and tests   |
| /scheduler-batching-change | Change batching/scheduler logic and update/add related tests     |
```
