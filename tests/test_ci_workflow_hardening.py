"""Regression checks for CI dependency intake and lockfile policy."""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS_PATH = ROOT / "requirements" / "constraints-ci.txt"
REFRESH_SCRIPT_PATH = ROOT / "scripts" / "refresh_dependency_locks.sh"
LOCK_DIR = ROOT / "requirements" / "locks"
PINNED_REQUIREMENTS = [
    ROOT / "requirements" / "build-backend.txt",
    ROOT / "requirements" / "ci-lint.txt",
]
CONSTRAINED_REQUIREMENTS = [
    ROOT / "requirements" / "ci-typecheck.txt",
    ROOT / "requirements" / "ci-test-matrix.txt",
]
LOCK_REQUIREMENTS = [
    LOCK_DIR / "ci-lint.txt",
    LOCK_DIR / "ci-typecheck.txt",
    LOCK_DIR / "ci-test-matrix.txt",
    LOCK_DIR / "local-verify-vision.txt",
]
FORBIDDEN_REQUIREMENT_TOKENS = (
    "git+",
    "--index-url",
    "--extra-index-url",
    "--find-links",
    "--trusted-host",
)
FORBIDDEN_DIRECT_REFERENCE_MARKERS = (" @ http://", " @ https://")


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


def _lock_package_entries(path: Path) -> list[str]:
    return [
        entry for entry in _non_comment_lines(path) if not entry.startswith("--hash=")
    ]


def _project_dependency_entries() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    entries = list(project.get("dependencies", []))
    for values in project.get("optional-dependencies", {}).values():
        entries.extend(values)
    return entries


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


def test_lockfiles_are_tracked_and_hashed():
    for path in LOCK_REQUIREMENTS:
        assert path.exists(), f"missing lockfile: {path.relative_to(ROOT)}"
        contents = path.read_text(encoding="utf-8")
        assert "--hash=sha256:" in contents, f"lockfile is missing hashes: {path.name}"
        entries = _lock_package_entries(path)
        assert entries, f"lockfile has no package entries: {path.name}"
        for entry in entries:
            assert "==" in entry, f"lockfile entry must be pinned: {entry}"


def test_pyproject_dependencies_avoid_direct_references():
    for entry in _project_dependency_entries():
        lowered = entry.lower()
        for token in FORBIDDEN_REQUIREMENT_TOKENS:
            assert (
                token not in lowered
            ), f"direct dependency source is not allowed: {entry}"
        for marker in FORBIDDEN_DIRECT_REFERENCE_MARKERS:
            assert (
                marker not in lowered
            ), f"direct URL dependency is not allowed: {entry}"


def test_requirement_inputs_avoid_index_overrides_and_direct_sources():
    for path in [CONSTRAINTS_PATH, *PINNED_REQUIREMENTS, *CONSTRAINED_REQUIREMENTS]:
        for entry in _non_comment_lines(path):
            lowered = entry.lower()
            for token in FORBIDDEN_REQUIREMENT_TOKENS:
                assert (
                    token not in lowered
                ), f"{path.name} must not override package intake policy: {entry}"
            for marker in FORBIDDEN_DIRECT_REFERENCE_MARKERS:
                assert (
                    marker not in lowered
                ), f"{path.name} must not use direct URL dependencies: {entry}"


def test_lock_refresh_script_regenerates_all_tracked_locks():
    script = REFRESH_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "uv pip compile" in script
    assert "--generate-hashes" in script
    assert "requirements/locks/ci-lint.txt" in script
    assert "requirements/locks/ci-typecheck.txt" in script
    assert "requirements/locks/ci-test-matrix.txt" in script
    assert "requirements/locks/local-verify-vision.txt" in script
    assert "requirements/build-backend.txt" in script
    assert "--python-version 3.11" in script


def test_ci_workflow_enforces_hash_locked_installs():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'PIP_DISABLE_PIP_VERSION_CHECK: "1"' in workflow
    assert "python -m pip install --upgrade pip" not in workflow
    assert "pip install ruff black" not in workflow
    assert "--require-hashes" in workflow
    assert "--only-binary=:all:" in workflow
    assert "--no-build-isolation -e ." in workflow
    assert "requirements/locks/ci-lint.txt" in workflow
    assert "requirements/locks/ci-typecheck.txt" in workflow
    assert "requirements/locks/ci-test-matrix.txt" in workflow
    assert "requirements/locks/local-verify-vision.txt" in workflow
    assert "requirements/ci-lint.txt" not in workflow
    assert "requirements/ci-typecheck.txt" not in workflow
    assert "requirements/ci-test-matrix.txt" not in workflow
    assert "requirements/constraints-ci.txt" not in workflow
    assert '-e ".[vision]"' not in workflow
