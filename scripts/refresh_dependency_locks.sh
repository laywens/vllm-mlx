#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

mkdir -p requirements/locks

echo "Refreshing CI lint lock"
uv pip compile \
  requirements/ci-lint.txt \
  --generate-hashes \
  --universal \
  -o requirements/locks/ci-lint.txt

echo "Refreshing CI type-check lock"
uv pip compile \
  requirements/ci-typecheck.txt \
  -c requirements/constraints-ci.txt \
  --generate-hashes \
  --universal \
  -o requirements/locks/ci-typecheck.txt

echo "Refreshing CI test-matrix lock"
uv pip compile \
  requirements/ci-test-matrix.txt \
  -c requirements/constraints-ci.txt \
  --generate-hashes \
  --universal \
  -o requirements/locks/ci-test-matrix.txt

echo "Refreshing local Apple Silicon verification lock"
uv pip compile \
  pyproject.toml \
  requirements/ci-test-matrix.txt \
  requirements/build-backend.txt \
  --extra vision \
  --generate-hashes \
  --python-version 3.11 \
  -o requirements/locks/local-verify-vision.txt
