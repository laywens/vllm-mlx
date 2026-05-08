"""Tests for CLI runtime/cache policy helpers."""

from __future__ import annotations

from types import SimpleNamespace

from vllm_mlx.cli import (
    _resolve_benchmark_profile,
    _build_startup_diagnostics,
    _resolve_cache_profile,
    _resolve_deterministic_profile,
)
from vllm_mlx.defaults import DEFAULT_FINISHED_OUTPUT_CACHE_CLEAR_INTERVAL


def _make_args(**overrides):
    base = {
        "enable_prefix_cache": True,
        "disable_prefix_cache": False,
        "no_memory_aware_cache": False,
        "use_paged_cache": False,
        "cache_strategy": "auto",
        "max_num_seqs": 256,
        "benchmark_disable_request_logging": False,
        "benchmark_disable_local_disconnect_guard": False,
        "benchmark_disable_prefill_executor": False,
        "benchmark_disable_finished_output_cache_clear": False,
        "benchmark_finished_output_cache_clear_interval": (
            DEFAULT_FINISHED_OUTPUT_CACHE_CLEAR_INTERVAL
        ),
        "benchmark_log_finished_output_cache_clear": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_startup_diagnostics_warn_on_exposed_unauthenticated_server():
    diagnostics = _build_startup_diagnostics(
        bind_host="0.0.0.0",
        api_key=None,
        rate_limit=0,
        runtime_mode="simple",
    )
    assert any("non-localhost bind without API key auth" in msg for msg in diagnostics)
    assert any("rate limiting disabled" in msg for msg in diagnostics)


def test_startup_diagnostics_include_local_development_note():
    diagnostics = _build_startup_diagnostics(
        bind_host="127.0.0.1",
        api_key=None,
        rate_limit=0,
        runtime_mode="simple",
    )
    assert any("Localhost-only mode" in msg for msg in diagnostics)


def test_resolve_cache_profile_auto_uses_paged_for_high_concurrency_batching():
    profile = _resolve_cache_profile(_make_args(), use_batching=True)
    assert profile.enable_prefix_cache is True
    assert profile.use_paged_cache is True
    assert profile.use_memory_aware_cache is False
    assert profile.strategy_label.startswith("auto->paged")


def test_resolve_cache_profile_legacy_strategy():
    profile = _resolve_cache_profile(
        _make_args(cache_strategy="legacy"),
        use_batching=True,
    )
    assert profile.enable_prefix_cache is True
    assert profile.use_paged_cache is False
    assert profile.use_memory_aware_cache is False
    assert profile.strategy_label == "legacy"


def test_resolve_deterministic_profile_disabled_keeps_runtime_selection():
    profile = _resolve_deterministic_profile(
        deterministic=False,
        use_batching=True,
        runtime_mode_reason="auto mode selected batched",
    )
    assert profile.enabled is False
    assert profile.use_batching is True
    assert profile.runtime_mode_reason == "auto mode selected batched"
    assert profile.forced_temperature is None
    assert profile.forced_top_p is None
    assert profile.serialize_tracked_routes is False


def test_resolve_deterministic_profile_forces_simple_and_greedy():
    profile = _resolve_deterministic_profile(
        deterministic=True,
        use_batching=True,
        runtime_mode_reason="auto mode selected batched",
    )
    assert profile.enabled is True
    assert profile.use_batching is False
    assert profile.forced_temperature == 0.0
    assert profile.forced_top_p == 1.0
    assert profile.serialize_tracked_routes is True
    assert "forcing simple mode" in profile.runtime_mode_reason


def test_resolve_benchmark_profile_tracks_explicit_toggles():
    profile = _resolve_benchmark_profile(
        _make_args(
            benchmark_disable_request_logging=True,
            benchmark_disable_prefill_executor=True,
            benchmark_finished_output_cache_clear_interval=4,
            benchmark_log_finished_output_cache_clear=True,
        )
    )
    assert profile.enabled is True
    assert profile.disable_request_logging is True
    assert profile.disable_local_disconnect_guard is False
    assert profile.disable_prefill_executor is True
    assert profile.disable_finished_output_cache_clear is False
    assert profile.finished_output_cache_clear_interval == 4
    assert profile.log_finished_output_cache_clear is True


def test_default_finished_output_cache_clear_interval_is_not_benchmark_profile():
    profile = _resolve_benchmark_profile(_make_args())
    assert profile.finished_output_cache_clear_interval == 4
    assert profile.enabled is False


def test_interval_one_finished_output_cache_clear_is_benchmark_override():
    profile = _resolve_benchmark_profile(
        _make_args(benchmark_finished_output_cache_clear_interval=1)
    )
    assert profile.enabled is True
    assert profile.finished_output_cache_clear_interval == 1


def test_startup_diagnostics_include_benchmark_notes():
    diagnostics = _build_startup_diagnostics(
        bind_host="127.0.0.1",
        api_key=None,
        rate_limit=0,
        runtime_mode="batched",
        benchmark_profile=_resolve_benchmark_profile(
            _make_args(
                benchmark_disable_request_logging=True,
                benchmark_disable_local_disconnect_guard=True,
                benchmark_finished_output_cache_clear_interval=8,
                benchmark_log_finished_output_cache_clear=True,
            )
        ),
    )
    assert any("Benchmark profile enabled" in msg for msg in diagnostics)
    assert any("request-level hot-path logging disabled" in msg for msg in diagnostics)
    assert any("localhost disconnect polling bypassed" in msg for msg in diagnostics)
    assert any("finished-request cache clears cadenced" in msg for msg in diagnostics)
    assert any("finished-request cache clear timing logged" in msg for msg in diagnostics)
