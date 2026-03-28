# Dependency Security

This repository treats dependency changes as security-relevant changes.

## CI Pinning Policy

- `requirements/ci-lint.txt` pins lint-only tools directly.
- `requirements/ci-typecheck.txt` and `requirements/ci-test-matrix.txt` define the package sets used in GitHub Actions.
- `requirements/constraints-ci.txt` is the reviewed version policy for CI and other reproducible verification environments.
- GitHub Actions should install from these tracked files instead of floating `pip install ...` commands.
- `python -m pip install --upgrade pip` is intentionally avoided in CI to reduce unnecessary package-resolution churn.

## Local Reproduction

For a CI-like environment from a trusted checkout:

```bash
uv pip install -c requirements/constraints-ci.txt -e ".[vision]"
uv pip install -c requirements/constraints-ci.txt -r requirements/ci-test-matrix.txt
```

For lint and type-check tooling:

```bash
uv pip install -r requirements/ci-lint.txt
uv pip install -c requirements/constraints-ci.txt -r requirements/ci-typecheck.txt
```

## Upstream Sync And Dependency Review Checklist

- [ ] Review dependency diffs in `pyproject.toml`, `requirements/constraints-ci.txt`, and `.github/workflows/ci.yml` before merging upstream changes.
- [ ] Classify each dependency change as runtime, CI-only, or contributor-only.
- [ ] Confirm the source of the change: upstream Git diff, direct local edit, or dependency refresh.
- [ ] For each version bump, read the upstream release notes or change log before accepting it.
- [ ] Refresh CI pins intentionally after a reviewed install in a clean environment; do not bundle opportunistic dependency upgrades with unrelated feature work.
- [ ] Re-run targeted verification after any dependency change:
  - `python -m pytest tests/test_ci_workflow_hardening.py -q`
  - `python -m pytest tests/test_docs_drift.py -q`
  - `python -m pytest tests/test_cli_localhost.py -q`

## Current Scope

This is a _constraints-based_ hardening pass, not a full hash-locked supply-chain solution.

- It reduces risk from accidental floating installs in CI.
- It gives upstream sync work an explicit dependency review step.
- It does _not_ yet provide fully hashed, cross-platform lockfiles or a private package mirror.

Those should be the next step for production-grade supply-chain control.
