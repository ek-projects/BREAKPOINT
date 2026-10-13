import json

from demo_runtime import run_demo
from nebius_reasoner import NemotronClient


def _ok_response():
    content = json.dumps({
        "hypotheses": [
            {"service": "database", "confidence_hint": 0.34, "reason": "candidate"},
            {"service": "queue", "confidence_hint": 0.33, "reason": "candidate"},
            {"service": "dependency", "confidence_hint": 0.33, "reason": "candidate"},
        ],
        "shortlist": ["db_probe", "queue_probe", "dependency_probe"],
        "rationale": "Use approved probes to discriminate candidates.",
    })
    return json.dumps({
        "model": "nvidia/Nemotron-3_5-Lightning",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_one_architect_attempt_for_multistep_incident():
    calls = []
    def transport(url, body, headers, timeout):
        calls.append(url)
        return 200, _ok_response()

    client = NemotronClient(api_key="test-key", transport=transport)
    result = run_demo("database", client=client, max_architect_calls=1)

    assert result["correct"] is True
    assert len(result["steps"]) == 2
    assert result["architect_attempts"] == 1
    assert result["architect_calls"] == 1
    assert [s["nemotron_invoked"] for s in result["steps"]] == [True, False]
    assert [s["fallback_used"] for s in result["steps"]] == [False, False]
    assert len(calls) == 1


def test_failed_architect_attempt_is_not_retried():
    calls = []
    def transport(url, body, headers, timeout):
        calls.append(url)
        return 503, '{"error":"temporary"}'

    client = NemotronClient(api_key="test-key", transport=transport)
    result = run_demo("dependency", client=client, max_architect_calls=1)

    assert result["correct"] is True
    assert len(result["steps"]) == 2
    assert result["architect_attempts"] == 1
    assert result["architect_calls"] == 0
    assert [s["fallback_used"] for s in result["steps"]] == [True, False]
    assert len(calls) == 1
