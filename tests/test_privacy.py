import json

import pytest

from textvbench.privacy import redact_credentials, redact_debug_value
from textvbench.evaluators.ref2v.review_api.server import public_trace as ref_trace
from textvbench.evaluators.t2v.review_api.server import public_trace as t2v_trace


def test_credentials_are_removed_without_mutating_runtime_configuration():
    config = {"profiles": [{"api_key": "test-credential", "access-token": "test-token", "model": "local"}]}
    redacted = redact_credentials(config)
    assert "test-credential" not in json.dumps(redacted)
    assert "test-token" not in json.dumps(redacted)
    assert redacted["profiles"][0]["model"] == "local"
    assert config["profiles"][0]["api_key"] == "test-credential"


@pytest.mark.parametrize("path", [
    "/mnt/private/input.jsonl",
    "/home/private folder/input.jsonl",
    r"C:\Users\private folder\input.jsonl",
    r"\\private-server\share\input.jsonl",
])
def test_paths_from_other_operating_systems_are_redacted(path):
    assert redact_debug_value({"input_path": path}) == {"input_path": "input.jsonl"}


@pytest.mark.parametrize("public_trace", [ref_trace, t2v_trace])
def test_both_review_apis_redact_credentials_paths_and_error_endpoints(public_trace):
    result = public_trace({
        "payload": {"api_key": "test-credential"},
        "user_payload": {"api_key": "test-credential"},
        "error": r"request to http://10.0.0.1/v1 failed for C:\Users\private\input.jsonl",
        "attempts": [{"error": "failed for /mnt/private/input.jsonl"}],
    })
    serialized = json.dumps(result)
    for private in ("test-credential", "10.0.0.1", "Users", "/mnt/"):
        assert private not in serialized
