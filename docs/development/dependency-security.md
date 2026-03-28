# Dependency Security

This repository treats dependency changes as security-relevant changes.

## CI Pinning Policy

- `requirements/ci-lint.txt` pins lint-only tools directly.
- `requirements/ci-typecheck.txt` and `requirements/ci-test-matrix.txt` define the package sets used in GitHub Actions.
- `requirements/constraints-ci.txt` is the reviewed version policy used to resolve the type-check and test lockfiles.
- `requirements/build-backend.txt` pins the editable-install build backend used by reviewed local verification environments.
- `requirements/locks/` contains the generated, hash-locked install artifacts used by CI and trusted local verification.
- GitHub Actions should install from `requirements/locks/*.txt` instead of floating `pip install ...` commands.
- `python -m pip install --upgrade pip` is intentionally avoided in CI to reduce unnecessary package-resolution churn.
- CI installs are wheel-only via `--only-binary=:all:` and hash-enforced via `--require-hashes`.

## Local Reproduction

For the reviewed Apple Silicon verification environment mirrored by CI
(`python 3.11` on `arm64`) from a trusted checkout:

```bash
uv pip install --require-hashes --only-binary=:all: -r requirements/locks/local-verify-vision.txt
uv pip install --no-deps --no-build-isolation -e .
```

For lint and type-check tooling:

```bash
uv pip install --require-hashes --only-binary=:all: -r requirements/locks/ci-lint.txt
uv pip install --require-hashes --only-binary=:all: -r requirements/locks/ci-typecheck.txt
```

To refresh the tracked lockfiles from a trusted checkout:

```bash
scripts/refresh_dependency_locks.sh
```

## Trusted Package Intake Policy

- Do not add direct URL or VCS dependencies to `pyproject.toml` or `requirements/`.
- Do not add `--index-url`, `--extra-index-url`, `--find-links`, or `--trusted-host` to tracked requirement files or CI workflow steps.
- CI and reviewed verification environments should install only from the tracked lockfiles under `requirements/locks/`.
- Editable installs in trusted flows must use `--no-deps --no-build-isolation` after the hashed dependency set is already installed.
- Git URL installs are operator convenience paths only; they are not acceptable for CI or release automation.

## Upstream Sync And Dependency Review Checklist

- [ ] Review dependency diffs in `pyproject.toml`, `requirements/*.txt`, `requirements/locks/*.txt`, and `.github/workflows/ci.yml` before merging upstream changes.
- [ ] Classify each dependency change as runtime, CI-only, or contributor-only.
- [ ] Confirm the source of the change: upstream Git diff, direct local edit, or dependency refresh.
- [ ] For each version bump, read the upstream release notes or change log before accepting it.
- [ ] Refresh lockfiles intentionally from a trusted checkout; do not bundle opportunistic dependency upgrades with unrelated feature work.
- [ ] Review any change to `requirements/build-backend.txt` as a trusted-intake change, not a routine dependency bump.
- [ ] Re-run targeted verification after any dependency change:
  - `python -m pytest tests/test_ci_workflow_hardening.py -q`
  - `python -m pytest tests/test_docs_drift.py -q`
  - `python -m pytest tests/test_cli_localhost.py -q`

## Current Scope

This is now a _hash-locked CI and reviewed-verification_ hardening pass.

- It reduces risk from accidental floating installs in CI.
- It makes the CI and local Apple Silicon verification path install from tracked hashed lockfiles.
- It gives upstream sync work an explicit dependency review step.
- It does _not_ yet provide a private package mirror or signed internal artifact promotion.

Those remain the next step for production-grade supply-chain control.
