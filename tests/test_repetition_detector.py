# SPDX-License-Identifier: Apache-2.0
"""Tests for scheduler repetition detection."""

import vllm_mlx.scheduler as scheduler


def _detect_repetition(tokens: list[int], **kwargs) -> bool:
    detector = getattr(scheduler, "_detect_repetition", None)
    assert callable(detector), "scheduler must expose _detect_repetition"
    return detector(tokens, **kwargs)


def _detect_repetition_trigger(tokens: list[int], **kwargs) -> str | None:
    detector = getattr(scheduler, "_detect_repetition_trigger", None)
    assert callable(detector), "scheduler must expose _detect_repetition_trigger"
    return detector(tokens, **kwargs)


def _resolve_repetition_detection_config(policy: str | None):
    resolver = getattr(scheduler, "_resolve_repetition_detection_config", None)
    assert callable(resolver), (
        "scheduler must expose _resolve_repetition_detection_config"
    )
    return resolver(policy)


class TestDetectRepetition:
    def test_repetition_detector_flags_single_token_tail(self):
        tokens = [101, 202, 303, 0, 0, 0, 0, 0, 0, 0, 0]
        assert _detect_repetition(tokens) is True

    def test_repetition_detector_flags_short_pattern_tail(self):
        tokens = [101, 202, 303] + [7, 8] * 6
        assert _detect_repetition(tokens) is True

    def test_repetition_detector_ignores_varied_tokens(self):
        assert _detect_repetition(list(range(32))) is False

    def test_strict_policy_thresholds_trigger_earlier(self):
        min_repeat, pattern_lengths, pattern_repeats = (
            _resolve_repetition_detection_config("strict")
        )

        assert min_repeat == 4
        assert pattern_lengths == (2, 3, 4, 5, 6)
        assert pattern_repeats == 4
        assert (
            _detect_repetition_trigger(
                [9, 9, 9, 9],
                min_repeat=min_repeat,
                pattern_lengths=pattern_lengths,
                pattern_repeats=pattern_repeats,
            )
            == "single_token_run"
        )

    def test_safe_policy_avoids_early_trigger(self):
        min_repeat, pattern_lengths, pattern_repeats = (
            _resolve_repetition_detection_config("safe")
        )

        assert min_repeat == 8
        assert pattern_lengths == (2, 3, 4)
        assert pattern_repeats == 6
        assert (
            _detect_repetition_trigger(
                [9, 9, 9, 9],
                min_repeat=min_repeat,
                pattern_lengths=pattern_lengths,
                pattern_repeats=pattern_repeats,
            )
            is None
        )
