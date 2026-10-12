import io
import json
from pathlib import Path

import pytest

from textvbench.evaluators.t2v.review_api.server import CaseStore, Handler, public_config, public_trace, safe_path


ROOT = Path(__file__).resolve().parents[1]


def _write_case(root: Path, case_id: str = "sample_zh") -> Path:
    case = root / "t2v" / case_id
    frame = case / "frames/t_0p000000_frame_00000"
    frame.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({
        "sample_id": "sample", "language": "zh", "video_path": "/private/sample.mp4",
    }), encoding="utf-8")
    (case / "input_record.json").write_text('{"pair_id":"pair","qc":"PASS"}', encoding="utf-8")
    (case / "resolved_config.json").write_text(json.dumps({
        "sampling": {"fps": 3}, "llm": {"active_profile": "test", "profiles": {}},
    }), encoding="utf-8")
    (case / "result.json").write_text(json.dumps({
        "video_path": "/private/sample.mp4", "metrics": {"textv_score": 52.5},
        "frames": [{"timestamp": 0.0, "frame_index": 0}],
    }), encoding="utf-8")
    (frame / "source_frame.jpg").write_bytes(b"jpeg")
    (frame / "carrier_vlm_call.json").write_text(json.dumps({
        "final_prompt": "judge", "final_response": {"carriers": []},
    }), encoding="utf-8")
    (frame / "matching_vlm_call_batch_000.json").write_text(json.dumps({
        "final_prompt": "match", "final_response": {"matches": []},
    }), encoding="utf-8")
    return case


def test_bundle_lists_and_shows_frame_evidence_without_internal_paths(tmp_path):
    _write_case(tmp_path)
    store = CaseStore(tmp_path)
    listing = store.list_cases({"search": ["pair"], "language": ["zh"]})
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["benchmark"] == "t2v"
    assert item["status"] == "complete"
    assert item["source_qc"] == "PASS"
    assert item["metrics"]["textv_score"] == 52.5
    assert item["video_available"] is False

    case = store.case("t2v", item["case_id"])
    detail = store.detail("t2v", item["case_id"], case)
    assert len(detail["frames"]) == 1
    assert detail["frames"][0]["folder"] == "t_0p000000_frame_00000"
    assert "source_frame.jpg" in detail["frames"][0]["files"]
    assert "matching_vlm_call_batch_000.json" in detail["frames"][0]["files"]
    assert "video_path" not in detail["manifest"]
    assert "video_path" not in detail["result"]
    assert "/private/" not in json.dumps(detail)


def test_list_filters_sort_and_pagination(tmp_path):
    for case_id, score, language in (("a_en", 80, "en"), ("b_zh", 20, "zh"), ("c_zh", 40, "zh")):
        case = tmp_path / "t2v" / case_id
        case.mkdir(parents=True)
        (case / "manifest.json").write_text(json.dumps({"sample_id": case_id, "language": language}))
        (case / "result.json").write_text(json.dumps({"frames": [], "metrics": {"textv_score": score}}))
    store = CaseStore(tmp_path)
    assert [item["case_id"] for item in store.list_cases({})["items"]] == ["b_zh", "c_zh", "a_en"]
    assert [item["case_id"] for item in store.list_cases({"sort": ["score_desc"]})["items"]] == ["a_en", "c_zh", "b_zh"]
    assert store.list_cases({"language": ["zh"]})["total"] == 2
    assert store.list_cases({})["facets"] == {"benchmarks": ["t2v"], "languages": ["en", "zh"]}
    assert store.list_cases({"page": ["2"], "page_size": ["1"]})["items"][0]["case_id"] == "c_zh"
    assert store.list_cases({})["counts"]["average_score"] == pytest.approx(140 / 3)
    with pytest.raises(ValueError, match="invalid sort"):
        store.list_cases({"sort": ["not-a-sort"]})
    with pytest.raises(ValueError, match="page"):
        store.list_cases({"page": ["0"]})


def test_status_depends_on_completed_result_not_vlm_trace_errors(tmp_path):
    case = _write_case(tmp_path)
    trace = case / "frames/t_0p000000_frame_00000/carrier_vlm_call.json"
    trace.write_text(json.dumps({
        "attempts": [{"attempt": 1, "status": "validation_error", "error": "invalid response"}],
        "error": "invalid response",
    }), encoding="utf-8")

    completed = CaseStore(tmp_path).list_cases({})
    assert completed["items"][0]["status"] == "complete"
    assert completed["counts"]["error"] == 0

    (case / "result.json").unlink()
    unfinished = CaseStore(tmp_path).list_cases({})
    assert unfinished["items"][0]["status"] == "error"
    assert unfinished["counts"]["error"] == 1


def test_index_is_rebuilt_on_start_and_lists_without_rescanning(tmp_path, monkeypatch):
    _write_case(tmp_path)
    index_path = tmp_path / ".textvbench_review_index.json"
    index_path.write_text('{"case_count":999,"cases":[]}', encoding="utf-8")

    store = CaseStore(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 1
    assert index["track"] == "t2v"
    assert index["source_root"] == "."
    assert index["case_count"] == 1

    monkeypatch.setattr(store, "summary", lambda *_args: pytest.fail("list request rescanned cases"))
    assert store.list_cases({})["total"] == 1

    _write_case(tmp_path, "second_zh")
    restarted = CaseStore(tmp_path)
    assert restarted.list_cases({})["total"] == 2
    assert json.loads(index_path.read_text(encoding="utf-8"))["case_count"] == 2


def test_file_access_is_bounded_to_selected_case_and_video_dir(tmp_path):
    root = tmp_path / "review"
    case = root / "t2v" / "valid_zh"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text('{"sample_id":"valid"}')
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "valid.mp4").write_bytes(b"mp4")
    store = CaseStore(root, video_dir)
    assert store.case("t2v", "valid_zh") == case
    assert store.video({"sample_id": "valid"}) == video_dir / "valid.mp4"
    assert store.video({"sample_id": "../outside"}) is None
    assert store.case("..", "valid_zh") is None
    assert store.case("t2v", "../valid_zh") is None
    assert safe_path(case, "..", "manifest.json") is None
    assert safe_path(case, "frames", "../../private") is None


def test_config_and_trace_keep_review_evidence_but_omit_connection_details():
    config = public_config({"llm": {"active_profile": "qwen", "profiles": {"qwen": {"base_url": "http://internal"}}},
                            "ocr": {"device": "cpu", "detection_model_dir": "/private", "confidence_threshold": 0.8},
                            "sampling": {"fps": 3}})
    assert config == {"sampling": {"fps": 3}, "ocr": {"confidence_threshold": 0.8},
                      "matching": {}, "model": "qwen"}
    trace = public_trace({"final_prompt": "judge the image", "user_payload": {"target": "亮步"},
                          "attempts": [{"attempt": 1, "status": "ok", "model": "qwen",
                                        "system_prompt": "judge the image", "response": {"yes": True},
                                        "request": {"api_key": "secret"}}], "final_response": {"yes": True}})
    assert trace["final_prompt"] == "judge the image"
    assert trace["user_payload"]["target"] == "亮步"
    assert trace["attempts"][0]["response"] == {"yes": True}
    assert "secret" not in json.dumps(trace)


def test_http_routes_restrict_artifacts_and_sanitize_calls(tmp_path):
    _write_case(tmp_path)
    handler = object.__new__(Handler)
    handler.store = CaseStore(tmp_path)
    handler.static_root = (ROOT / "frontend/t2v_review").resolve()
    responses = []
    handler.send_json = lambda value, status=200: responses.append((status, value))
    handler.send_file = lambda path: responses.append((200, path.name))

    handler.path = "/api/cases/?page=1&page_size=20&sort=score_asc"
    handler.do_GET()
    assert responses[-1][0] == 200
    assert responses[-1][1]["total"] == 1

    handler.path = "/forwarded-port/api/cases?page=1&page_size=20&sort=score_asc"
    handler.do_GET()
    assert responses[-1][0] == 200
    assert responses[-1][1]["total"] == 1

    handler.path = "/api/health"
    handler.do_GET()
    assert responses[-1] == (200, {"status": "ok", "cases": 1})

    case_url = "/api/cases/t2v/sample_zh"
    handler.path = case_url
    handler.do_GET()
    assert responses[-1][0] == 200
    assert "video_path" not in responses[-1][1]["manifest"]

    handler.path = case_url + "/frames/t_0p000000_frame_00000/matching_vlm_call_batch_000.json"
    handler.do_GET()
    assert responses[-1][1]["final_prompt"]
    assert responses[-1][1]["final_response"] == {"matches": []}

    handler.path = "/media/cases/t2v/sample_zh/frames/t_0p000000_frame_00000/source_frame.jpg"
    handler.do_GET()
    assert responses[-1] == (200, "source_frame.jpg")

    handler.path = case_url + "/frames/t_0p000000_frame_00000/../../resolved_config.json"
    handler.do_GET()
    assert responses[-1][0] == 404

    handler.path = "/media/cases/t2v/sample_zh/frames/t_0p000000_frame_00000/carrier_vlm_call.json"
    handler.do_GET()
    assert responses[-1][0] == 404

    _write_case(tmp_path, "new_zh")
    handler.path = "/api/cases?refresh=1"
    handler.do_GET()
    assert responses[-1][1]["total"] == 2


def test_http_video_supports_browser_range_requests(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"0123456789abcdef")
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
    assert handler.wfile.getvalue() == b"3456"
