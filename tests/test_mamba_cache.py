# SPDX-License-Identifier: Apache-2.0
"""Tests for BatchMambaCache compatibility shims."""

import pytest


def test_batch_mamba_cache_passes_size_when_flag_is_stale(monkeypatch):
    from vllm_mlx.utils import mamba_cache

    monkeypatch.setattr(mamba_cache, "HAS_MAMBA_CACHE", True)

    cache = mamba_cache.BatchMambaCache(left_padding=[0], size=3)

    assert len(cache.cache) == 3


def test_batch_mamba_cache_extract_passes_size_when_flag_is_stale(monkeypatch):
    import mlx.core as mx

    from vllm_mlx.utils import mamba_cache

    monkeypatch.setattr(mamba_cache, "HAS_MAMBA_CACHE", True)

    batch_cache = mamba_cache.BatchMambaCache(left_padding=[0], size=2)
    batch_cache.cache = [mx.array([[1]]), mx.array([[2]])]

    extracted = batch_cache.extract(0)

    assert len(extracted.cache) == 2
    assert extracted.cache[0].tolist() == [[1]]


@pytest.mark.parametrize("stale_flag", [False, True])
def test_batch_mamba_cache_extract_keeps_left_padding_none(monkeypatch, stale_flag):
    import mlx.core as mx

    from vllm_mlx.utils import mamba_cache

    monkeypatch.setattr(mamba_cache, "HAS_MAMBA_CACHE", stale_flag)

    batch_cache = mamba_cache.BatchMambaCache(left_padding=[0], size=1)
    batch_cache.cache = [mx.array([[42]])]

    extracted = batch_cache.extract(0)

    assert extracted.left_padding is None
