"""Shared structured-output settings for the OpenAI-compatible VLM clients."""

from __future__ import annotations

from typing import Any, Iterable


def _id_array_schema(allowed_ids: Iterable[str]) -> dict[str, Any]:
    values = list(dict.fromkeys(str(item) for item in allowed_ids))
    items: dict[str, Any] = {"type": "string"}
    if values:
        items["enum"] = values
    schema: dict[str, Any] = {
        "type": "array",
        "items": items,
    }
    if not values:
        schema["maxItems"] = 0
    return schema


def _ordered_array_schema(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "array",
        "prefixItems": items,
        "items": False,
        "minItems": len(items),
        "maxItems": len(items),
    }


def _response_format(name: str, property_name: str, value_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {property_name: value_schema},
                "required": [property_name],
            },
        },
    }


def carrier_response_format(carrier_ids: Iterable[str], candidate_ids: Iterable[str]) -> dict[str, Any]:
    """Constrain a carrier-routing response to this frame's ordered carriers."""
    candidates = list(candidate_ids)
    rows = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "carrier_id": {"type": "string", "const": str(carrier_id)},
                "present": {"type": "boolean"},
                "candidate_ids": _id_array_schema(candidates),
            },
            "required": ["carrier_id", "present", "candidate_ids"],
        }
        for carrier_id in carrier_ids
    ]
    return _response_format("textvbench_t2v_carriers", "carriers", _ordered_array_schema(rows))


def t2v_matching_response_format(target_indices: Iterable[int], candidate_ids: Iterable[str]) -> dict[str, Any]:
    """Constrain T2V matching to ordered target indices and current candidates."""
    candidates = list(candidate_ids)
    rows = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "i": {"type": "integer", "const": int(index)},
                "selected_ids": _id_array_schema(candidates),
            },
            "required": ["i", "selected_ids"],
        }
        for index in target_indices
    ]
    return _response_format("textvbench_t2v_matching", "matches", _ordered_array_schema(rows))


def ref_routing_response_format(reference_ids: Iterable[str]) -> dict[str, Any]:
    """Constrain Ref2V routing to IDs present in the current sample."""
    return _response_format(
        "textvbench_ref2v_routing",
        "matched_ref_ids",
        _id_array_schema(reference_ids),
    )


def ref_matching_response_format(slot_ids: Iterable[str], candidate_ids: Iterable[str]) -> dict[str, Any]:
    """Constrain Ref2V matching to ordered slots and remaining candidates."""
    candidates = list(candidate_ids)
    rows = [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "slot_id": {"type": "string", "const": str(slot_id)},
                "selected_ids": _id_array_schema(candidates),
            },
            "required": ["slot_id", "selected_ids"],
        }
        for slot_id in slot_ids
    ]
    return _response_format("textvbench_ref2v_matching", "matches", _ordered_array_schema(rows))


def request_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare base request kwargs while preserving vLLM extra-body overrides."""
    llm = config.get("llm", {})
    request = dict(llm.get("request", {}))
    extra_body = dict(request.pop("extra_body", {}) or {})
    if llm.get("structured_output"):
        template_kwargs = dict(extra_body.get("chat_template_kwargs", {}) or {})
        template_kwargs.setdefault("enable_thinking", False)
        template_kwargs.setdefault("preserve_thinking", False)
        extra_body["chat_template_kwargs"] = template_kwargs
    if extra_body:
        request["extra_body"] = extra_body
    request.setdefault("response_format", {"type": "json_object"})
    return request


def request_with_response_format(
    request: dict[str, Any], response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a per-call request with a stage-specific vLLM JSON schema."""
    result = dict(request)
    if response_format is None:
        return result
    extra_body = dict(result.get("extra_body", {}) or {})
    extra_body["response_format"] = response_format
    result["extra_body"] = extra_body
    return result
