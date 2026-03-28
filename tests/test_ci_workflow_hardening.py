"""Regression checks for CI dependency pinning policy."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS_PATH = ROOT / "requirements" / "constraints-ci.txt"
PINNED_REQUIREMENTS = [ROOT / "requirements" / "ci-lint.txt"]
CONSTRAINED_REQUIREMENTS = [
    ROOT / "requirements" / "ci-typecheck.txt",
    ROOT / "requirements" / "ci-test-matrix.txt",
]


def _non_comment_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _requirement_name(entry: str) -> str:
    return entry.split("==", 1)[0].strip()


def test_constraints_file_is_fully_pinned():
    for entry in _non_comment_lines(CONSTRAINTS_PATH):
        assert "==" in entry, f"constraints entry must be pinned: {entry}"


def test_pinned_requirement_inputs_are_fully_pinned():
    for path in PINNED_REQUIREMENTS:
        for entry in _non_comment_lines(path):
            assert "==" in entry, f"{path.name} entry must be pinned: {entry}"


def test_constrained_requirement_inputs_are_covered_by_constraints():
    constrained_names = {
        _requirement_name(entry) for entry in _non_comment_lines(CONSTRAINTS_PATH)
    }
    for path in CONSTRAINED_REQUIREMENTS:
        for entry in _non_comment_lines(path):
            if "==" in entry:
                continue
            name = _requirement_name(entry)
            assert (
                name in constrained_names
            ), f"{path.name} entry is missing from constraints-ci.txt: {name}"


def test_ci_workflow_uses_tracked_requirement_files():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'PIP_DISABLE_PIP_VERSION_CHECK: "1"' in workflow
    assert "python -m pip install --upgrade pip" not in workflow
    assert "pip install ruff black" not in workflow
    assert "requirements/ci-lint.txt" in workflow
    assert "requirements/ci-typecheck.txt" in workflow
    assert "requirements/ci-test-matrix.txt" in workflow
    assert "requirements/constraints-ci.txt" in workflow
    assert '-e ".[vision]"' in workflow
