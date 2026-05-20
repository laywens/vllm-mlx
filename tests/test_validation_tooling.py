# SPDX-License-Identifier: Apache-2.0
"""Tests for public validation helper scripts."""

import json

import requests

from scripts import prefix_reuse_harness, serve_profile_benchmark


def test_serve_profile_benchmark_records_request_failure(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("deadline exceeded")

    monkeypatch.setattr(serve_profile_benchmark.requests, "post", raise_timeout)

    result = serve_profile_benchmark._call_chat(
        base_url="http://127.0.0.1:8000",
        model="local-model",
        prompt="hello",
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        timeout=0.1,
    )

    assert result["ok"] is False
    assert "deadline exceeded" in result["error"]
    assert result["prompt"] == "hello"
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def test_serve_profile_benchmark_reports_failure_rate():
    summarizer = getattr(serve_profile_benchmark, "_summarize_result_status", None)
    assert callable(summarizer), (
        "serve_profile_benchmark must expose _summarize_result_status"
    )

    status = summarizer([{"ok": True}, {"ok": False}, {"error": "missing ok"}])

    assert status == {
        "successful_requests": 1,
        "failed_requests": 2,
        "failure_rate": 0.6667,
    }


def test_prefix_reuse_harness_loads_all_suffixes(tmp_path):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "shared_prefix": "Shared context",
                "suffixes": [
                    {"label": "one", "prompt": "First"},
                    {"label": "two", "prompt": "Second"},
                    {"label": "three", "prompt": "Third"},
                    {"label": "four", "prompt": "Fourth"},
                ],
            }
        ),
        encoding="utf-8",
    )

    fixture = prefix_reuse_harness._load_fixture(str(fixture_path))

    assert [item["label"] for item in fixture["suffixes"]] == [
        "one",
        "two",
        "three",
        "four",
    ]
