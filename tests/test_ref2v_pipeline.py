import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import textvbench.evaluators.ref2v.pipeline as pipeline
from textvbench.evaluators.ref2v.core import Candidate, Slot
from textvbench.evaluators.ref2v.ocr import OCRError
from textvbench.evaluators.ref2v.pipeline import (
    _aggregate,
    _target_payload,
    _validate_match_response,
    _validate_ref_response,
    evaluate_record,
    run,
)
from textvbench.evaluators.ref2v.vlm import VLMClient
from textvbench.evaluators.vlm_validation import InvalidCandidateSelectionError


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "0001"


def _sample_media(root: Path) -> tuple[Path, Path]:
    import cv2

    image = root / "reference.png"
    Image.new("RGB", (96, 64), "navy").save(image)
    video = root / "sample.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((64, 96, 3), index * 5, dtype=np.uint8))
    writer.release()
    return image, video


def _config():
    return {
        "sampling": {"fps": 1, "min_frames": 1, "max_frames": 3, "seed": 42, "jitter_sec": 0},
        "ocr": {"confidence_threshold": 0.8, "max_candidates": 20},
        "matching": {"target_batch_size": 3},
        "marker": {"font_path": ""},
        "vlm": {
            "ref_id_prompt_path": "prompts/ref2v/TextVBench_Ref2V_RefID_Matching_System_Prompt.md",
            "matching_prompt_path": "prompts/ref2v/TextVBench_Ref2V_Target_Candidate_Matching_System_Prompt.md",
        },
    }


class FakeOCR:
    """One clean reference slot and one clean generated candidate."""

    def predict(self, image):
        if isinstance(image, (str, Path)):
            return [Candidate("R001", "OPEN", 0.99, ((100, 100), (300, 100), (300, 140), (100, 140)))]
        return [Candidate("P001", "OPEN", 0.99, ((100, 100), (300, 100), (300, 140), (100, 140)))]


def test_sample_video_evaluation_with_fake_ocr(tmp_path):
    image, video = _sample_media(tmp_path)
    config = _config()
    config["llm"] = {"api_key": "test-runtime-credential", "enabled": False}
    record = {
        "sample_id": SAMPLE_ID,
        "prompt": "static screen",
        "reference_images": [{"ref_id": "R01", "filename": image.name, "category": "message_chat"}],
    }
    result = evaluate_record(
        record,
        video,
        config,
        image_dir=tmp_path,
        project_root=ROOT,
        ocr_engine=FakeOCR(),
        vlm=None,
        debug_dir=tmp_path / "debug",
        include_frames=True,
    )
    metrics = result["metrics"]
    assert result["video"]["frame_count"] == 20
    assert len(result["frames"]) == 2
    assert metrics["per_target_spatial_fidelity"] == {"T0001": 1.0}
    assert metrics["length_weighted_spatial_fidelity"] == 1.0
    assert metrics["length_weighted_temporal_fidelity"] == 1.0
    assert metrics["textv_score"] == 100.0
    assert metrics["reading_order_score"] == 1.0
    assert (tmp_path / "debug" / "result.json").is_file()
    saved_config = json.loads((tmp_path / "debug" / "resolved_config.json").read_text(encoding="utf-8"))
    assert saved_config["llm"]["api_key"] == "[redacted]"
    assert config["llm"]["api_key"] == "test-runtime-credential"
    assert (tmp_path / "debug" / "reference_R01_ocr.json").is_file()
    reference_debug = Image.open(tmp_path / "debug" / "reference_R01.jpg")
    assert max(reference_debug.size) <= 256
    assert any(
        reference_debug.getpixel((x, y)) != (255, 255, 255)
        for y in range(min(30, reference_debug.height))
        for x in range(reference_debug.width)
    )


def test_missing_reference_is_zero_credit():
    refs = [
        {"ref_id": "R01", "slots": [Slot("T1", "AA", "R01", "C1", 1)]},
        {"ref_id": "R02", "slots": [Slot("T2", "BBBB", "R02", "C2", 1)]},
    ]
    slots = [item for ref in refs for item in ref["slots"]]
    metrics = _aggregate(
        [{
            "timestamp": 0.0,
            "evaluated": True,
            "matched_ref_ids": ["R01"],
            "extra_text_ratio": 0.0,
            "matches": [{"slot_id": "T1", "ref_id": "R01", "carrier_present": True,
                         "selected_ids": ["P1"], "selected_bboxes": [((0, 0), (1, 0), (1, 1), (0, 1))],
                         "pred_text": "AA", "score": 1.0}],
        }],
        refs,
        slots,
    )
    assert metrics["refs"][0]["spatial"] == 1.0
    assert metrics["refs"][1]["spatial"] == 0.0
    assert metrics["length_weighted_spatial_fidelity"] == 0.5
    assert metrics["textv_score"] == 50.0


def test_vlm_response_validation_is_strict():
    _validate_ref_response({"matched_ref_ids": ["R01"]}, {"R01", "R02"})
    _validate_match_response(
        {"matches": [{"slot_id": "T1", "selected_ids": ["P1"]}, {"slot_id": "T2", "selected_ids": []}]},
        [{"slot_id": "T1"}, {"slot_id": "T2"}],
        {"P1", "P2"},
    )
    for invalid in (
        {"matched_ref_ids": ["R03"]},
        {"matched_ref_ids": ["R01", "R01"]},
        {"matched_ref_ids": [1]},
    ):
        try:
            _validate_ref_response(invalid, {"R01", "R02"})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ref-id response accepted")


def test_matching_validator_allows_candidate_reuse_across_targets():
    _validate_match_response(
        {"matches": [
            {"slot_id": "T1", "selected_ids": ["P1"]},
            {"slot_id": "T2", "selected_ids": ["P1"]},
        ]},
        [{"slot_id": "T1"}, {"slot_id": "T2"}],
        {"P1"},
    )


def test_matching_validator_provides_response_with_invalid_ids_removed():
    result = {"matches": [
        {"slot_id": "T1", "selected_ids": ["P1", "P404", "P1"]},
        {"slot_id": "T2", "selected_ids": ["P2"]},
    ]}
    with pytest.raises(InvalidCandidateSelectionError) as caught:
        _validate_match_response(
            result,
            [{"slot_id": "T1"}, {"slot_id": "T2"}],
            {"P1", "P2"},
        )
    assert caught.value.removed_ids == ["P404", "P1"]
    assert caught.value.recovered_response == {"matches": [
        {"slot_id": "T1", "selected_ids": ["P1"]},
        {"slot_id": "T2", "selected_ids": ["P2"]},
    ]}


def test_ref2v_vlm_recovers_after_all_retries_return_invalid_candidate_ids(monkeypatch):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        structured_output = False

        def complete_json(self, prompt, payload, image_urls):
            return {"matches": [{"slot_id": "T1", "selected_ids": ["P404"]}]}

    monkeypatch.setattr("textvbench.evaluators.ref2v.vlm.time.sleep", lambda _seconds: None)
    client = FakeVLM.__new__(FakeVLM)
    traces = []
    result, attempts = client.complete_with_retries(
        "match", {}, [],
        lambda value: _validate_match_response(value, [{"slot_id": "T1"}], {"P1"}),
        trace_callback=traces.append,
    )
    assert attempts == 2
    assert result == {"matches": [{"slot_id": "T1", "selected_ids": []}]}
    assert traces[-1]["status"] == "recovered_invalid_candidates"
    assert traces[-1]["removed_ids"] == ["P404"]
    assert traces[-1]["recovered_response"] == result


def test_ref2v_vlm_does_not_recover_wrong_slot_order(monkeypatch):
    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        structured_output = False

        def complete_json(self, prompt, payload, image_urls):
            return {"matches": [{"slot_id": "T2", "selected_ids": ["P1"]}]}

    monkeypatch.setattr("textvbench.evaluators.ref2v.vlm.time.sleep", lambda _seconds: None)
    client = FakeVLM.__new__(FakeVLM)
    with pytest.raises(ValueError, match="target order mismatch"):
        client.complete_with_retries(
            "match", {}, [],
            lambda value: _validate_match_response(value, [{"slot_id": "T1"}], {"P1"}),
        )


def test_recovered_invalid_selection_keeps_candidates_available_for_next_batch(monkeypatch):
    slots = [
        Slot(f"T000{i}", f"target-{i}", "R01", "C_R01", i)
        for i in range(1, 3)
    ]
    references = [{
        "ref_id": "R01",
        "carrier": "complete reference carrier",
        "category": "",
        "image": Image.new("RGB", (32, 32), "white"),
    }]
    requests = []

    class FakeOCR:
        def predict(self, _frame):
            return [Candidate("P001", "candidate", 0.99, ((4, 4), (20, 4), (20, 10), (4, 10)))]

    class FakeVLM(VLMClient):
        model = "fake-vlm"
        request = {"temperature": 0.0}
        record_retries = 1
        structured_output = False
        calls = 0

        def complete_json(self, prompt, payload, image_urls, *, request=None):
            self.calls += 1
            requests.append(payload)
            target = payload["targets"][0]
            selected_ids = ["P404"] if self.calls <= 2 else ["P001"]
            return {"matches": [{"slot_id": target["slot_id"], "selected_ids": selected_ids}]}

    monkeypatch.setattr("textvbench.evaluators.ref2v.vlm.time.sleep", lambda _seconds: None)
    observation = pipeline._frame_observation(
        frame=np.zeros((32, 32, 3), dtype=np.uint8), timestamp=0.0, frame_index=0,
        references=references, slots=slots, config={
            "ocr": {"confidence_threshold": 0.8, "max_candidates": 20},
            "matching": {"target_batch_size": 1}, "marker": {"font_path": ""},
        }, ocr_engine=FakeOCR(), vlm=FakeVLM.__new__(FakeVLM),
        ref_prompt="ref", matching_prompt="match", font_path=None,
        artifact_dir=None, scene_context="",
    )
    assert len(requests) == 3
    assert requests[-1]["ocr_candidates"][0]["id"] == "P001"
    assert [item["selected_ids"] for item in observation["matches"]] == [[], ["P001"]]


def test_matching_reuses_candidate_within_batch_but_removes_it_from_next_batch():
    slots = [
        Slot(f"T000{i}", f"target-{i}", "R01", "C_R01", i)
        for i in range(1, 5)
    ]
    references = [{
        "ref_id": "R01",
        "carrier": "complete reference carrier",
        "category": "",
        "image": Image.new("RGB", (32, 32), "white"),
    }]
    requests = []

    class FakeOCR:
        def predict(self, _frame):
            return [
                Candidate("P001", "shared", 0.99, ((4, 4), (20, 4), (20, 10), (4, 10))),
                Candidate("P002", "other", 0.99, ((4, 12), (20, 12), (20, 18), (4, 18))),
            ]

    class FakeVLM:
        def complete_with_retries(
            self, _prompt, payload, _images, validator, *, response_format=None, trace_callback=None,
        ):
            requests.append(payload)
            assert response_format["json_schema"]["name"] == "textvbench_ref2v_matching"
            selected = ["P001"] if len(requests) == 1 else ["P002"]
            value = {"matches": [
                {"slot_id": target["slot_id"], "selected_ids": selected}
                for target in payload["targets"]
            ]}
            validator(value)
            return value, 1

    observation = pipeline._frame_observation(
        frame=np.zeros((32, 32, 3), dtype=np.uint8), timestamp=0.0, frame_index=0,
        references=references, slots=slots, config={
            "ocr": {"confidence_threshold": 0.8, "max_candidates": 20},
            "matching": {"target_batch_size": 3}, "marker": {"font_path": ""},
        }, ocr_engine=FakeOCR(), vlm=FakeVLM(), ref_prompt="ref", matching_prompt="match",
        font_path=None, artifact_dir=None, scene_context="",
    )
    assert [item["targets"][0]["slot_id"] for item in requests] == ["T0001", "T0004"]
    assert [item["id"] for item in requests[0]["ocr_candidates"]] == ["P001", "P002"]
    assert [item["id"] for item in requests[1]["ocr_candidates"]] == ["P002"]
    assert [item["selected_ids"] for item in observation["matches"]] == [["P001"], ["P001"], ["P001"], ["P002"]]


def test_target_payload_is_context_compact():
    slot = Slot("T0004", "Xiaoran Zhou 4:05 PM", "R01", "C_R01", 4)
    assert _target_payload(slot) == {
        "slot_id": "T0004",
        "target_text": "Xiaoran Zhou 4:05 PM",
        "ref_id": "R01",
    }


def test_vlm_request_uses_compact_target_and_candidate_payload(tmp_path):
    image, video = _sample_media(tmp_path)
    record = {
        "sample_id": SAMPLE_ID,
        "prompt": "static screen",
        "reference_images": [{"ref_id": "R01", "filename": image.name}],
    }
    requests = []

    class FakeVLM:
        def complete_with_retries(
            self, prompt, payload, image_urls, validator, *, response_format=None, trace_callback=None,
        ):
            requests.append(payload)
            assert response_format["json_schema"]["name"] == "textvbench_ref2v_matching"
            assert all(set(item) == {"slot_id", "target_text", "ref_id"} for item in payload["targets"])
            assert all(set(item) == {"id", "ocr_text", "confidence"} for item in payload["ocr_candidates"])
            value = {"matches": [{"slot_id": payload["targets"][0]["slot_id"], "selected_ids": ["P001"]}]}
            validator(value)
            return value, 1

    evaluate_record(
        record,
        video,
        _config(),
        image_dir=tmp_path,
        project_root=ROOT,
        ocr_engine=FakeOCR(),
        vlm=FakeVLM(),
    )
    assert requests


def test_public_run_writes_metrics_result_and_summary(tmp_path):
    image, video = _sample_media(tmp_path)
    input_path = tmp_path / "video_prompt_dataset.jsonl"
    input_path.write_text(json.dumps({
        "sample_id": SAMPLE_ID,
        "prompt": "static screen",
        "reference_images": [{"ref_id": "R01", "filename": image.name}],
    }) + "\n", encoding="utf-8")
    output_dir = tmp_path / "metrics"
    summary = run(
        input_path.parent,
        output_dir,
        _config(),
        image_dir=tmp_path,
        video_path=video,
        sample_id=SAMPLE_ID,
        project_root=ROOT,
        ocr_engine=FakeOCR(),
        vlm=None,
    )
    assert summary["ok"] == 1 and summary["error"] == 0
    assert summary["results"][0]["metrics"]["textv_score"] == 100.0
    result_path = output_dir / SAMPLE_ID / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert "frames" not in result
    assert (output_dir / "summary.json").is_file()
    run_details = json.loads((output_dir / "run_details.json").read_text(encoding="utf-8"))
    assert run_details["samples"][0]["status"] == "ok"
    assert run_details["samples"][0]["elapsed_sec"] >= 0


def test_public_run_resumes_successes_and_retries_errors(tmp_path, monkeypatch):
    input_path = tmp_path / "video_prompt_dataset.jsonl"
    input_path.write_text(json.dumps({"sample_id": "0001"}) + "\n", encoding="utf-8")
    video = tmp_path / "0001.mp4"
    video.write_bytes(b"video")
    calls = {"count": 0, "fail": True}

    def fake_evaluate(*args, **kwargs):
        calls["count"] += 1
        if calls["fail"]:
            raise ValueError("ordinary sample failure")
        return {"metrics": {"textv_score": 42.0}}

    monkeypatch.setattr(pipeline, "evaluate_record", fake_evaluate)
    output_dir = tmp_path / "metrics"
    first = pipeline.run(input_path, output_dir, {}, video_path=video, project_root=tmp_path)
    assert first["ok"] == 0 and first["error"] == 1

    calls["fail"] = False
    second = pipeline.run(input_path, output_dir, {}, video_path=video, project_root=tmp_path)
    assert second["ok"] == 1 and second["error"] == 0
    assert second["resume"]["processed"] == 1

    third = pipeline.run(input_path, output_dir, {}, video_path=video, project_root=tmp_path)
    assert third["ok"] == 1 and third["resume"]["skipped_success"] == 1
    assert calls["count"] == 2

    pipeline.run(input_path, output_dir, {}, video_path=video, project_root=tmp_path, override=True)
    assert calls["count"] == 3


def test_public_run_parallelizes_samples_and_preserves_input_order(tmp_path, monkeypatch):
    rows = [{"sample_id": f"{index:04d}"} for index in range(4)]
    input_path = tmp_path / "video_prompt_dataset.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "videos.jsonl"
    manifest_lines = []
    for row in rows:
        video = tmp_path / f"{row['sample_id']}.mp4"
        video.write_bytes(b"video")
        manifest_lines.append(json.dumps({"sample_id": row["sample_id"], "video_path": video.name}))
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    lock = threading.Lock()
    active = {"value": 0, "max": 0}

    def fake_evaluate(*args, **kwargs):
        with lock:
            active["value"] += 1
            active["max"] = max(active["max"], active["value"])
        time.sleep(0.05)
        with lock:
            active["value"] -= 1
        return {"metrics": {"textv_score": 42.0}}

    monkeypatch.setattr(pipeline, "evaluate_record", fake_evaluate)
    summary = pipeline.run(
        input_path,
        tmp_path / "metrics",
        {},
        video_manifest=manifest_path,
        project_root=tmp_path,
        sample_workers=2,
    )
    assert active["max"] == 2
    assert [item["sample_id"] for item in summary["results"]] == [row["sample_id"] for row in rows]
    assert summary["execution"]["sample_workers"] == 2


def test_dependency_errors_abort_public_run(tmp_path, monkeypatch):
    input_path = tmp_path / "video_prompt_dataset.jsonl"
    input_path.write_text(json.dumps({"sample_id": "0001"}) + "\n", encoding="utf-8")
    video = tmp_path / "0001.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(pipeline, "evaluate_record", lambda *args, **kwargs: (_ for _ in ()).throw(OCRError("paddle failed")))
    with pytest.raises(OCRError, match="paddle failed"):
        pipeline.run(input_path, tmp_path / "metrics", {}, video_path=video, project_root=tmp_path)
