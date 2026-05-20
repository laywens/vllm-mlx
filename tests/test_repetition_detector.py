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


def _resolve_repetition_detection_runtime(policy: str | None):
    resolver = getattr(scheduler, "_resolve_repetition_detection_runtime", None)
    assert callable(resolver), (
        "scheduler must expose _resolve_repetition_detection_runtime"
    )
    return resolver(policy)


class FakeGenerationResponse:
    def __init__(self, uid: int, token: int, finish_reason: str | None = None):
        self.uid = uid
        self.token = token
        self.logprobs = None
        self.finish_reason = finish_reason
        self.prompt_cache = None
        self.all_tokens = None


class FakeBatchGenerator:
    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.index = 0
        self.remove_calls: list[tuple[list[int], bool]] = []

    def next(self):
        if self.index >= len(self.tokens):
            return [], []
        token = self.tokens[self.index]
        self.index += 1
        return [], [FakeGenerationResponse(uid=7, token=token)]

    def remove(self, uids_to_remove, return_prompt_caches=False):
        self.remove_calls.append((list(uids_to_remove), return_prompt_caches))
        if return_prompt_caches:
            return {7: ("cache-7", [101, *self.tokens[: self.index]])}
        return None


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

    def test_repetition_policy_normalization_is_case_insensitive(self):
        normalizer = getattr(scheduler, "_normalize_repetition_policy", None)
        assert callable(normalizer), "scheduler must expose _normalize_repetition_policy"

        assert normalizer("SAFE") == "safe"
        assert normalizer("Strict") == "strict"

    def test_repetition_runtime_buffer_cap_matches_active_policy(self):
        runtime_config = _resolve_repetition_detection_runtime("strict")

        required_history = max(
            runtime_config.min_repeat,
            max(runtime_config.pattern_lengths) * runtime_config.pattern_repeats,
        )
        assert runtime_config.buffer_token_limit == required_history + 8

    def test_default_batch_generator_wrapper_stops_and_cleans_repetition(self):
        installer = getattr(scheduler, "_install_repetition_detection", None)
        assert callable(installer), "scheduler must expose _install_repetition_detection"

        batch_gen = FakeBatchGenerator([9, 9, 9, 9, 9])
        installer(batch_gen, default_repetition_policy="strict")

        for _ in range(3):
            _, responses = batch_gen.next()
            assert responses[0].finish_reason is None

        _, responses = batch_gen.next()
        stopped_response = responses[0]
        assert stopped_response.finish_reason == "stop"
        assert stopped_response.prompt_cache == "cache-7"
        assert stopped_response.all_tokens == [101, 9, 9, 9, 9]
        assert batch_gen.remove_calls == [([7], True)]

        _, responses = batch_gen.next()
        assert responses[0].finish_reason is None
