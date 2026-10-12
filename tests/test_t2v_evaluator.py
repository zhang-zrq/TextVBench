import json
import math
from pathlib import Path

import pytest

from textvbench.cli import main
from textvbench.evaluators.t2v import pipeline
from textvbench.evaluators.t2v.core import (
    Candidate,
    Target,
    compress_duplicate_candidates,
    extra_text_ratio,
    filter_candidates,
    harmonic_textv_score,
    normalized_edit_distance,
    normalized_edit_similarity,
    normalize_text,
    sample_timestamps,
    target_length_weighted_score,
)
from textvbench.evaluators.t2v.pipeline import (
    _aggregate,
    _load_video_manifest,
    _resolve_video,
    _targets,
    _validate_match_response,
    _vlm_candidate,
    _write_call_artifacts,
    run as run_evaluator,
)
from textvbench.evaluators.t2v.ocr import OCRError
from textvbench.evaluators.t2v.vlm import VLMClient
from textvbench.evaluators.vlm_validation import InvalidCandidateSelectionError


def test_failed_video_does_not_leave_credentials_in_debug_configuration(tmp_path, monkeypatch):
    config = {
        "llm": {"profiles": {"local": {"api_key": "test-runtime-credential"}}},
        "vlm": {
            "carrier_prompt_path": "prompts/t2v/TextVBench_T2V_carrier_consistency_system_prompt.md",
            "matching_prompt_path": "prompts/t2v/TextVBench_T2V_target_candidate_matching_system_prompt.md",
        },
    }

    def invalid_video(_path):
        raise ValueError("invalid video")

    monkeypatch.setattr(pipeline, "_video_info", invalid_video)
    with pytest.raises(ValueError, match="invalid video"):
        pipeline.evaluate_record(
            {"sample_id": "case", "prompt_en": "scene"}, tmp_path / "invalid.mp4", config,
            language="en", project_root=Path(__file__).resolve().parents[1], review_dir=tmp_path / "debug",
        )
    saved = (tmp_path / "debug" / "resolved_config.json").read_text(encoding="utf-8")
    assert "test-runtime-credential" not in saved
    assert json.loads(saved)["llm"]["profiles"]["local"]["api_key"] == "[redacted]"
    assert config["llm"]["profiles"]["local"]["api_key"] == "test-runtime-credential"


def test_normalization_and_ned_keep_case_but_remove_unicode_punctuation():
    assert normalize_text(" A，b! ") == "Ab"
    assert normalized_edit_distance("shoe polish", "shoe-polish") == 0
    assert normalized_edit_similarity("ABC", "abc") == 0


def test_temporal_sampling_is_deterministic_and_covers_endpoints():
    first = sample_timestamps(5.0, fps=3, max_frames=8, seed=7, sample_id="A", jitter_sec=0.1)
    second = sample_timestamps(5.0, fps=3, max_frames=8, seed=7, sample_id="A", jitter_sec=0.1)
    assert first == second
    assert first[0] == 0.0
    assert first[-1] == 5.0
    assert len(first) == 8


def test_candidate_filtering_compresses_duplicates_and_honors_repeated_gt():
    candidates = [
        Candidate("P1", "SALE", 0.99),
        Candidate("P2", "SALE", 0.95),
        Candidate("P3", "low", 0.50),
    ]
    active, duplicate, capped = filter_candidates(candidates, confidence_threshold=0.8, max_candidates=20)
    assert [item.id for item in active] == ["P1"]
    assert [item.id for item in duplicate] == ["P2"]
    assert capped == []
    active, duplicate = compress_duplicate_candidates(candidates[:2], required_texts=["SALE", "SALE"])
    assert {item.id for item in active} == {"P1", "P2"}
    assert duplicate == []


def test_weighted_metrics_and_extra_ratio():
    targets = [Target("T1", "AA", "C1"), Target("T2", "BBBB", "C1")]
    assert math.isclose(target_length_weighted_score({"T1": 1, "T2": 0.5}, targets), 4 / 6)
    assert extra_text_ratio(["XX"], ["YYYY"]) == 2 / 6
    assert math.isclose(harmonic_textv_score(0.5, 1.0), 100 * 2 / 3)


def test_carrier_absence_is_masked_but_all_video_absence_scores_zero():
    targets = [Target("T1", "SALE", "C1"), Target("T2", "OPEN", "C2")]
    frames = [{
        "timestamp": 0.0,
        "extra_text_ratio": 0.75,
        "matches": [
            {"slot_id": "T1", "carrier_present": True, "selected_ids": ["P1"], "pred_text": "SALE", "score": 1.0},
            {"slot_id": "T2", "carrier_present": False, "selected_ids": [], "pred_text": "", "score": None},
        ],
    }]
    metrics = _aggregate(frames, targets)
    assert metrics["targets"][0]["spatial"] == 1.0
    assert metrics["targets"][1]["spatial"] == 0.0
    assert metrics["extra_text_ratio"] == 0.75
    # Extra text remains diagnostic and cannot alter TextVScore.
    assert metrics["textv_score"] == 100 * (2 * (2 / 4) * (2 / 4) / ((2 / 4) + (2 / 4)))


def test_matching_validator_allows_candidate_reuse_across_targets():
    targets = [{"i": 0}, {"i": 1}]
    result = {"matches": [{"i": 0, "selected_ids": ["P1"]}, {"i": 1, "selected_ids": ["P1"]}]}
    _validate_match_response(result, targets, {"P1"})


def test_matching_validator_rejects_duplicate_id_within_one_target():
    targets = [{"i": 0}]
    result = {"matches": [{"i": 0, "selected_ids": ["P1", "P1"]}]}
    with pytest.raises(ValueError, match="duplicate candidate"):
        _validate_match_response(result, targets, {"P1"})


def test_matching_validator_provides_response_with_invalid_ids_removed():
    result = {"matches": [
        {"i": 0, "selected_ids": ["P1", "P404", "P1"]},
        {"i": 1, "selected_ids": ["P2"]},
    ]}
    with pytest.raises(InvalidCandidateSelectionError) as caught:
        _validate_match_response(result, [{"i": 0}, {"i": 1}], {"P1", "P2"})
    assert caught.value.removed_ids == ["P404", "P1"]
    assert caught.value.recovered_response == {"matches": [
        {"i": 0, "selected_ids": ["P1"]},
        {"i": 1, "selected_ids": ["P2"]},
    ]}


def test_vlm_candidate_payload_uses_marker_id_without_bbox():
    candidate = Candidate("P001", "亮步", 0.9988, ((122.0, 1082.0), (170.0, 1080.0)))
    assert _vlm_candidate(candidate) == {
        "id": "P001",
        "ocr_text": "亮步",
        "confidence": pytest.approx(0.9988),
    }
    assert "bbox" not in _vlm_candidate(candidate)


def test_language_must_be_explicit_in_cli_and_programmatic_contract():
    with pytest.raises(SystemExit):
        main(["evaluator", "t2v", "--input", "input.jsonl", "--output", "output.jsonl"])
    with pytest.raises(ValueError, match="explicitly set"):
        _targets({}, "")


def test_vlm_trace_and_review_artifact_preserve_final_prompt_and_response(tmp_path):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        calls = 0

        def complete_json(self, prompt, payload, image_data_url):
            self.calls += 1
            return {"ok": self.calls > 1}

    traces = []
    client = FakeVLM.__new__(FakeVLM)

    def validate_ok(value):
        if not value["ok"]:
            raise ValueError("not ok")

    result, attempts = client.complete_with_retries(
        "SYSTEM PROMPT",
        {"sample": "A"},
        "data:image/jpeg;base64,AA==",
        validate_ok,
        trace_callback=traces.append,
    )
    assert attempts == 2
    assert traces[0]["status"] == "validation_error"
    assert "Previous response error: not ok" in traces[-1]["system_prompt"]
    assert result == {"ok": True}

    _write_call_artifacts(
        [tmp_path], "call.json", call_type="test", marker_image="marker.jpg",
        payload={"sample": "A"}, attempts=traces, response=result,
    )
    artifact = json.loads((tmp_path / "call.json").read_text(encoding="utf-8"))
    assert artifact["final_prompt"] == traces[-1]["system_prompt"]
    assert artifact["final_response"] == {"ok": True}


def test_vlm_uses_valid_retry_instead_of_recovering_earlier_invalid_response(monkeypatch):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        structured_output = False
        calls = 0

        def complete_json(self, prompt, payload, image_data_url):
            self.calls += 1
            selected = ["P404"] if self.calls == 1 else ["P1"]
            return {"matches": [{"i": 0, "selected_ids": selected}]}

    monkeypatch.setattr("textvbench.evaluators.t2v.vlm.time.sleep", lambda _seconds: None)
    client = FakeVLM.__new__(FakeVLM)
    traces = []
    result, attempts = client.complete_with_retries(
        "match", {}, "data:image/jpeg;base64,AA==",
        lambda value: _validate_match_response(value, [{"i": 0}], {"P1"}),
        trace_callback=traces.append,
    )
    assert attempts == 2
    assert result == {"matches": [{"i": 0, "selected_ids": ["P1"]}]}
    assert [item["status"] for item in traces] == ["validation_error", "ok"]


def test_vlm_recovers_after_all_retries_return_invalid_candidate_ids(monkeypatch):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 2
        structured_output = False

        def complete_json(self, prompt, payload, image_data_url):
            return {"matches": [{"i": 0, "selected_ids": ["P1", "P404", "P1"]}]}

    monkeypatch.setattr("textvbench.evaluators.t2v.vlm.time.sleep", lambda _seconds: None)
    client = FakeVLM.__new__(FakeVLM)
    traces = []
    result, attempts = client.complete_with_retries(
        "match", {}, "data:image/jpeg;base64,AA==",
        lambda value: _validate_match_response(value, [{"i": 0}], {"P1"}),
        trace_callback=traces.append,
    )
    assert attempts == 3
    assert result == {"matches": [{"i": 0, "selected_ids": ["P1"]}]}
    assert traces[-1]["status"] == "recovered_invalid_candidates"
    assert traces[-1]["removed_ids"] == ["P404", "P1"]
    assert traces[-1]["recovered_response"] == result


def test_vlm_does_not_recover_structurally_invalid_matching_response(monkeypatch):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        structured_output = False

        def complete_json(self, prompt, payload, image_data_url):
            return {"matches": [{"i": 1, "selected_ids": ["P1"]}]}

    monkeypatch.setattr("textvbench.evaluators.t2v.vlm.time.sleep", lambda _seconds: None)
    client = FakeVLM.__new__(FakeVLM)
    with pytest.raises(ValueError, match="target order mismatch"):
        client.complete_with_retries(
            "match", {}, "data:image/jpeg;base64,AA==",
            lambda value: _validate_match_response(value, [{"i": 0}], {"P1"}),
        )


def test_video_dir_is_strict_and_manifest_supports_non_sample_filenames(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    expected = videos / "sample_a.mp4"
    expected.write_bytes(b"video")
    record = {"sample_id": "sample_a", "video_path": str(tmp_path / "stale.mp4")}
    assert _resolve_video(record, videos, None) == expected
    with pytest.raises(FileNotFoundError, match="sample_b"):
        _resolve_video({"sample_id": "sample_b"}, videos, None)

    manifest = tmp_path / "videos.jsonl"
    manifest.write_text('{"sample_id":"sample_b","video_path":"nested/custom.webm"}\n', encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    custom = nested / "custom.webm"
    custom.write_bytes(b"video")
    mapping = _load_video_manifest(manifest)
    assert mapping["sample_b"] == custom
    assert _resolve_video({"sample_id": "sample_b"}, None, None, mapping) == custom


def test_evaluator_records_missing_video_as_sample_error_and_continues(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(
        '{"sample_id":"present","prompt_en":"scene"}\n{"sample_id":"missing","prompt_en":"scene"}\n', encoding="utf-8"
    )
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "present.mp4").write_bytes(b"video")

    class FakeVLM:
        model = "fake"

    monkeypatch.setattr(pipeline, "VLMClient", lambda config: FakeVLM())
    monkeypatch.setattr(pipeline, "PaddleOCREngine", lambda config, project_root: object())
    monkeypatch.setattr(
        pipeline,
        "evaluate_record",
        lambda *args, **kwargs: {"metrics": {"textv_score": 42.0}},
    )
    output = tmp_path / "metrics" / "evaluator.jsonl"
    assert run_evaluator(input_path, output, {}, language="en", video_dir=videos,
                         project_root=tmp_path) == 2
    summary_root = output.parent
    summary = json.loads((summary_root / "summary.json").read_text())
    assert summary["evaluator_t2v"] == {
        "total": 2,
        "ok": 1,
        "error": 1,
        "output": "passed/evaluator.jsonl",
        "errors": "failed/evaluator_errors.jsonl",
        "updated_at": summary["evaluator_t2v"]["updated_at"],
    }
    errors = (summary_root / "failed" / "evaluator_errors.jsonl").read_text()
    assert '"sample_id": "missing"' in errors


def test_metrics_only_output_omits_frame_debug_result(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"sample_id":"sample_a","prompt_en":"scene"}\n', encoding="utf-8")
    video = tmp_path / "sample_a.mp4"
    video.write_bytes(b"video")

    class FakeVLM:
        model = "fake"

    monkeypatch.setattr(pipeline, "VLMClient", lambda config: FakeVLM())
    monkeypatch.setattr(pipeline, "PaddleOCREngine", lambda config, project_root: object())
    monkeypatch.setattr(
        pipeline,
        "evaluate_record",
        lambda *args, **kwargs: {"metrics": {"textv_score": 42.0}, "frames": [{"timestamp": 0.0}]},
    )
    run_evaluator(
        input_path, tmp_path / "metrics" / "evaluator.jsonl", {},
        language="en", video_path=video, sample_id="sample_a", project_root=tmp_path,
    )
    output = json.loads((tmp_path / "metrics" / "passed" / "evaluator.jsonl").read_text())
    evaluator = output["evaluator_t2v"]
    assert evaluator["metrics"] == {"textv_score": 42.0}
    assert "result" not in evaluator
    assert evaluator["audit"]["output_mode"] == "metrics"
    assert not (tmp_path / "metrics" / "cache").exists()

    debug_output = tmp_path / "debug-metrics" / "evaluator.jsonl"
    run_evaluator(
        input_path, debug_output, {},
        language="en", video_path=video, sample_id="sample_a",
        project_root=tmp_path, review_root=tmp_path / "debug-bundle", debug=True,
        override=True,
    )
    debug_record = json.loads(
        (tmp_path / "debug-metrics" / "passed" / "evaluator.jsonl").read_text()
    )
    assert "result" in debug_record["evaluator_t2v"]
    assert debug_record["evaluator_t2v"]["audit"]["output_mode"] == "debug"


def test_evaluator_retries_failed_records_and_skips_successes(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"sample_id":"sample_a","prompt_en":"scene"}\n', encoding="utf-8")
    video = tmp_path / "sample_a.mp4"
    video.write_bytes(b"video")
    calls = {"count": 0, "fail": True}

    class FakeVLM:
        model = "fake"

    def fake_evaluate(*args, **kwargs):
        calls["count"] += 1
        if calls["fail"]:
            raise ValueError("ordinary sample failure")
        return {"metrics": {"textv_score": 42.0}}

    monkeypatch.setattr(pipeline, "VLMClient", lambda config: FakeVLM())
    monkeypatch.setattr(pipeline, "PaddleOCREngine", lambda config, project_root: object())
    monkeypatch.setattr(pipeline, "evaluate_record", fake_evaluate)
    output = tmp_path / "metrics" / "evaluator.jsonl"

    assert run_evaluator(input_path, output, {}, language="en", video_path=video,
                         sample_id="sample_a", project_root=tmp_path) == 1
    calls["fail"] = False
    assert run_evaluator(input_path, output, {}, language="en", video_path=video,
                         sample_id="sample_a", project_root=tmp_path) == 1
    assert run_evaluator(input_path, output, {}, language="en", video_path=video,
                         sample_id="sample_a", project_root=tmp_path) == 0
    assert calls["count"] == 2
    assert run_evaluator(input_path, output, {}, language="en", video_path=video,
                         sample_id="sample_a", project_root=tmp_path, override=True) == 1
    assert calls["count"] == 3


def test_t2v_dependency_errors_abort_batch(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text('{"sample_id":"sample_a","prompt_en":"scene"}\n', encoding="utf-8")
    video = tmp_path / "sample_a.mp4"
    video.write_bytes(b"video")

    class FakeVLM:
        model = "fake"

    monkeypatch.setattr(pipeline, "VLMClient", lambda config: FakeVLM())
    monkeypatch.setattr(pipeline, "PaddleOCREngine", lambda config, project_root: object())
    monkeypatch.setattr(
        pipeline,
        "evaluate_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(OCRError("paddle failed")),
    )
    with pytest.raises(OCRError, match="paddle failed"):
        run_evaluator(input_path, tmp_path / "metrics" / "evaluator.jsonl", {},
                      language="en", video_path=video, sample_id="sample_a", project_root=tmp_path)
