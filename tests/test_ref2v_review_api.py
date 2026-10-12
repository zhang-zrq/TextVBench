import io
import json
from pathlib import Path

import pytest

from textvbench.evaluators.ref2v.review_api.server import CaseStore, Handler, public_trace, safe_path


ROOT = Path(__file__).resolve().parents[1]
def _write_case(root: Path, sample_id: str, score: float, *, extra: float = 0.0) -> Path:
    case = root / sample_id
    frame = case / "frames/t_0p000000"
    frame.mkdir(parents=True)
    (case / "manifest.json").write_text(
        json.dumps({"sample_id": sample_id, "video_path": f"/mnt/private/{sample_id}.mp4"}),
        encoding="utf-8",
    )
    (case / "result.json").write_text(
        json.dumps({
            "sample_id": sample_id,
            "metrics": {
                "textv_score": score,
                "length_weighted_spatial_fidelity": score / 100,
                "length_weighted_temporal_fidelity": score / 100,
                "extra_text_ratio": extra,
            },
            "frames": [{"timestamp": 0.0, "evaluated": True, "matches": []}],
        }),
        encoding="utf-8",
    )
    (frame / "observation.json").write_text(
        json.dumps({"timestamp": 0.0, "evaluated": True, "matches": []}),
        encoding="utf-8",
    )
    (frame / "matching_vlm_call_batch_000.json").write_text(
        json.dumps({
            "payload": {"targets": [], "video_path": "/mnt/private/video.mp4"},
            "attempts": [
                {"attempt": 1, "status": "error", "error": "request failed at http://10.0.0.1/v1"},
                {"attempt": 2, "status": "ok", "system_prompt": "match", "response": {"matches": []}},
            ],
            "response": {"matches": []},
            "error": None,
        }),
        encoding="utf-8",
    )
    return case


def test_list_filters_sorts_paginates_and_counts_successful_retry(tmp_path):
    root = tmp_path / "review"
    _write_case(root, "0001", 80.0, extra=0.1)
    _write_case(root, "0002", 20.0, extra=0.5)
    _write_case(root, "0003", 40.0, extra=0.2)
    store = CaseStore(root)

    assert [item["sample_id"] for item in store.list_cases({})["items"]] == ["0002", "0003", "0001"]
    assert [item["sample_id"] for item in store.list_cases({"sort": ["score_desc"]})["items"]] == ["0001", "0003", "0002"]
    assert [item["sample_id"] for item in store.list_cases({"sort": ["spatial_asc"]})["items"]] == ["0002", "0003", "0001"]
    assert store.list_cases({"search": ["0002"]})["total"] == 1
    assert store.list_cases({"page": ["2"], "page_size": ["1"]})["items"][0]["sample_id"] == "0003"
    assert store.list_cases({})["counts"]["average_score"] == pytest.approx(140 / 3)
    assert store.list_cases({})["items"][0]["vlm_retries"] == 1
    assert store.list_cases({})["items"][0]["trace_errors"] == 0
    assert all(item["status"] == "complete" for item in store.list_cases({})["items"])
    with pytest.raises(ValueError, match="invalid sort"):
        store.list_cases({"sort": ["unknown"]})
    with pytest.raises(ValueError, match="page"):
        store.list_cases({"page": ["0"]})


def test_status_depends_on_completed_result_not_vlm_trace_errors(tmp_path):
    root = tmp_path / "review"
    case = _write_case(root, "0001", 80.0)
    trace = case / "frames/t_0p000000/matching_vlm_call_batch_000.json"
    trace.write_text(json.dumps({
        "attempts": [
            {"attempt": attempt, "status": "validation_error", "response": {}, "error": "invalid response"}
            for attempt in range(1, 4)
        ],
        "response": {},
        "error": "invalid response",
    }), encoding="utf-8")

    completed = CaseStore(root).list_cases({})
    assert completed["items"][0]["status"] == "complete"
    assert completed["items"][0]["trace_errors"] == 1
    assert completed["counts"]["error"] == 0
    assert "incomplete" not in completed["counts"]

    (case / "result.json").unlink()
    unfinished = CaseStore(root).list_cases({})
    assert unfinished["items"][0]["status"] == "error"
    assert unfinished["counts"]["error"] == 1


def test_index_is_rebuilt_on_start_and_lists_without_rescanning(tmp_path, monkeypatch):
    root = tmp_path / "review"
    _write_case(root, "0001", 80.0)
    index_path = root / ".textvbench_review_index.json"
    index_path.write_text('{"case_count":999,"cases":[]}', encoding="utf-8")

    store = CaseStore(root)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert index["track"] == "ref2v"
    assert index["source_root"] == "."
    assert index["case_count"] == 1

    monkeypatch.setattr(store, "summary", lambda *_args: pytest.fail("list request rescanned cases"))
    assert store.list_cases({})["total"] == 1

    _write_case(root, "0002", 60.0)
    restarted = CaseStore(root)
    assert restarted.list_cases({})["total"] == 2
    assert json.loads(index_path.read_text(encoding="utf-8"))["case_count"] == 2


def test_single_case_root_uses_manifest_sample_id(tmp_path):
    generated = _write_case(tmp_path / "staging", "0042", 75.0)
    case = tmp_path / "renamed-directory"
    generated.rename(case)
    store = CaseStore(case)
    assert store.list_cases({})["items"][0]["sample_id"] == "0042"
    assert store.case("0042") == case.resolve()
    assert store.case(case.name) is None


def test_config_trace_and_detail_redact_paths_credentials_and_endpoints(tmp_path):
    root = tmp_path / "review"
    case = _write_case(root, "0001", 80.0)
    (case / "resolved_config.json").write_text(json.dumps({
        "sampling": {"fps": 3},
        "ocr": {"confidence_threshold": 0.8, "detection_model_dir": "/mnt/models/det"},
        "llm": {"active_profile": "qwen", "profiles": {"qwen": {"base_url": "http://10.0.0.1/v1", "api_key": "secret"}}},
    }), encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    (results / "run_details.json").write_text(json.dumps({
        "input": "/mnt/private/input.jsonl",
        "output_dir": "/mnt/private/results",
        "error": "request to http://10.0.0.1/v1 failed for /mnt/private/input.jsonl",
    }), encoding="utf-8")

    detail = CaseStore(root, results_root=results).detail("0001", case)
    serialized = json.dumps(detail)
    assert "/mnt/" not in serialized
    assert "10.0.0.1" not in serialized
    assert "secret" not in serialized
    assert detail["config"] == {
        "sampling": {"fps": 3},
        "ocr": {"confidence_threshold": 0.8},
        "matching": {},
        "model": "qwen",
    }

    trace = public_trace({
        "payload": {"targets": [{"slot_id": "T1"}]},
        "attempts": [{
            "attempt": 1, "status": "ok", "model": "qwen", "system_prompt": "judge",
            "response": {"matches": []}, "request": {"api_key": "secret"},
        }],
        "response": {"matches": []},
    })
    assert trace["payload"]["targets"][0]["slot_id"] == "T1"
    assert trace["final_prompt"] == "judge"
    assert trace["final_response"] == {"matches": []}
    assert "secret" not in json.dumps(trace)


def test_file_access_is_bounded_and_only_allowlisted_artifacts_are_served(tmp_path):
    root = tmp_path / "review"
    case = _write_case(root, "0001", 80.0)
    (case / "reference_R01_ocr.json").write_text('{"ref_id":"R01","slots":[]}', encoding="utf-8")
    (case / "reference_R01.jpg").write_bytes(b"jpeg")
    (case / "frames/t_0p000000/source_frame.jpg").write_bytes(b"jpeg")
    (case / "private.json").write_text('{"secret":true}', encoding="utf-8")
    store = CaseStore(root)

    assert safe_path(case, "..", "private.json") is None
    assert store.case("../0001") is None

    handler = object.__new__(Handler)
    handler.store = store
    handler.static_root = (ROOT / "frontend/ref2v_review").resolve()
    handler.headers = {}
    responses = []
    handler.send_json = lambda value, status=200: responses.append((status, value))
    handler.send_file = lambda path: responses.append((200, path.name))

    handler.path = "/api/cases?page=1&page_size=20&sort=score_asc"
    handler.do_GET()
    assert responses[-1][0] == 200
    assert responses[-1][1]["total"] == 1

    handler.path = "/api/cases/0001/frames/t_0p000000/matching_vlm_call_batch_000.json"
    handler.do_GET()
    assert responses[-1][1]["final_response"] == {"matches": []}

    handler.path = "/media/cases/0001/references/reference_R01.jpg"
    handler.do_GET()
    assert responses[-1] == (200, "reference_R01.jpg")

    for path in (
        "/api/cases/0001/frames/t_0p000000/../../private.json",
        "/api/cases/0001/references/private.json",
        "/media/cases/0001/frames/t_0p000000/private.json",
    ):
        handler.path = path
        handler.do_GET()
        assert responses[-1][0] == 404

    _write_case(root, "0002", 60.0)
    handler.path = "/api/cases?refresh=1"
    handler.do_GET()
    assert responses[-1][1]["total"] == 2


def test_video_manifest_and_browser_range_requests(tmp_path):
    video = tmp_path / "noncanonical-name.mp4"
    video.write_bytes(b"0123456789abcdef")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(json.dumps({"sample_id": "0001", "video_path": video.name}) + "\n", encoding="utf-8")
    root = tmp_path / "review"
    _write_case(root, "0001", 80.0)
    assert CaseStore(root, video_manifest=manifest).video("0001") == video

    handler = object.__new__(Handler)
    handler.headers = {"Range": "bytes=3-6"}
    handler.wfile = io.BytesIO()
    result = {}
    handler.send_response = lambda status: result.update(status=status)
    handler.send_header = lambda key, value: result.update({key: value})
    handler.end_headers = lambda: None
    handler.send_video(video)
    assert result["status"] == 206
    assert result["Content-Range"] == "bytes 3-6/16"
    assert result["Accept-Ranges"] == "bytes"
    assert handler.wfile.getvalue() == b"3456"
