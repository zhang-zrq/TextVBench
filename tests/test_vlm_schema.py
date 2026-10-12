from textvbench.evaluators.vlm_schema import (
    carrier_response_format,
    ref_matching_response_format,
    ref_routing_response_format,
    request_kwargs,
    request_with_response_format,
    t2v_matching_response_format,
)


def _schema(response_format):
    return response_format["json_schema"]["schema"]


def test_ref_matching_schema_is_stage_specific_and_batch_bound():
    schema = _schema(ref_matching_response_format(["T1", "T2"], ["P1", "P2"]))
    assert set(schema["properties"]) == {"matches"}
    matches = schema["properties"]["matches"]
    assert matches["minItems"] == matches["maxItems"] == 2
    assert [item["properties"]["slot_id"]["const"] for item in matches["prefixItems"]] == ["T1", "T2"]
    selected = matches["prefixItems"][0]["properties"]["selected_ids"]
    assert selected["items"]["enum"] == ["P1", "P2"]
    assert "uniqueItems" not in selected


def test_all_response_formats_only_allow_their_call_type():
    formats = [
        (carrier_response_format(["C1"], ["P1"]), "carriers"),
        (t2v_matching_response_format([0], ["P1"]), "matches"),
        (ref_routing_response_format(["R1"]), "matched_ref_ids"),
        (ref_matching_response_format(["T1"], ["P1"]), "matches"),
    ]
    for response_format, expected_property in formats:
        schema = _schema(response_format)
        assert schema["additionalProperties"] is False
        assert schema["required"] == [expected_property]
        assert set(schema["properties"]) == {expected_property}


def test_per_call_schema_replaces_only_response_format_in_extra_body():
    config = {
        "llm": {
            "structured_output": True,
            "request": {"max_tokens": 64, "extra_body": {"top_k": 20}},
        },
    }
    base = request_kwargs(config)
    response_format = ref_routing_response_format(["R1", "R2"])
    request = request_with_response_format(base, response_format)
    assert request["max_tokens"] == 64
    assert request["extra_body"]["top_k"] == 20
    assert request["extra_body"]["response_format"] == response_format
    assert "response_format" not in base["extra_body"]
