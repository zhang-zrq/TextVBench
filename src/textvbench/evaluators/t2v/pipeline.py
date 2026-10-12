"""End-to-end T2V video evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ...benchmark import aggregate_benchmark_metrics
from ...privacy import redact_credentials
from ..vlm_schema import carrier_response_format, t2v_matching_response_format
from ..vlm_validation import InvalidCandidateSelectionError
from .core import (
    Candidate,
    Target,
    extra_text_ratio,
    filter_candidates,
    harmonic_textv_score,
    normalized_edit_similarity,
    sample_timestamps,
    target_length_weighted_score,
)
from .marker import marker_data_url, render_marker
from .ocr import OCRError, PaddleOCREngine
from .vlm import VLMClient, VLMError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _validate_language(language: str) -> str:
    if language not in {"en", "zh"}:
        raise ValueError("T2V evaluator language must be explicitly set to 'en' or 'zh'")
    return language


def _load_prompt(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.read_text(encoding="utf-8")


def _targets(record: dict[str, Any], language: str) -> tuple[list[dict[str, Any]], list[Target]]:
    language = _validate_language(language)
    carriers = record.get("carriers", [])
    texts = {item.get("slot_id"): item for item in record.get("texts", [])}
    targets: list[Target] = []
    flat_carriers: list[dict[str, Any]] = []
    for carrier in carriers:
        carrier_id = str(carrier.get("carrier_id", ""))
        flat_carriers.append({
            "carrier_id": carrier_id,
            "type": carrier.get("type", ""),
            "position": carrier.get("position", ""),
        })
        for slot in carrier.get("slots", []):
            slot_id = str(slot.get("slot_id", ""))
            text = texts.get(slot_id, {})
            target_text = text.get("text_zh") if language == "zh" else text.get("text_en")
            if target_text is None:
                target_text = text.get("text_en") or text.get("text_zh") or ""
            targets.append(Target(slot_id, str(target_text), carrier_id, str(carrier.get("type", "")), str(carrier.get("position", ""))))
    return flat_carriers, targets


def _validate_carrier_response(result: dict[str, Any], carriers: list[dict[str, Any]], allowed: set[str]) -> None:
    if set(result) != {"carriers"} or not isinstance(result["carriers"], list):
        raise ValueError("carrier response must contain only carriers array")
    if len(result["carriers"]) != len(carriers):
        raise ValueError("carrier response must contain one item per carrier")
    seen: set[str] = set()
    for expected, actual in zip(carriers, result["carriers"]):
        if set(actual) != {"carrier_id", "present", "candidate_ids"}:
            raise ValueError("carrier response item has unexpected keys")
        if actual["carrier_id"] != expected["carrier_id"]:
            raise ValueError("carrier order or carrier_id mismatch")
        if not isinstance(actual["present"], bool) or not isinstance(actual["candidate_ids"], list):
            raise ValueError("carrier present/candidate_ids types are invalid")
        if not set(actual["candidate_ids"]).issubset(allowed):
            raise ValueError("carrier response selected an unknown candidate")
        if not actual["present"] and actual["candidate_ids"]:
            raise ValueError("absent carrier cannot own candidates")
        overlap = seen.intersection(actual["candidate_ids"])
        if overlap:
            raise ValueError(f"candidate assigned to multiple carriers: {sorted(overlap)}")
        seen.update(actual["candidate_ids"])


def _validate_match_response(result: dict[str, Any], targets: list[dict[str, Any]], allowed: set[str]) -> None:
    if set(result) != {"matches"} or not isinstance(result["matches"], list):
        raise ValueError("matching response must contain only matches array")
    if len(result["matches"]) != len(targets):
        raise ValueError("matching response must contain one item per target")
    expected_indices = [int(target["i"]) for target in targets]
    cleaned_matches: list[dict[str, Any]] = []
    removed_ids: list[str] = []
    for expected_i, actual in zip(expected_indices, result["matches"]):
        if (not isinstance(actual, dict)
                or set(actual) != {"i", "selected_ids"}
                or actual["i"] != expected_i):
            raise ValueError("matching response target order mismatch")
        selected_ids = actual["selected_ids"]
        if (not isinstance(selected_ids, list)
                or any(not isinstance(item, str) for item in selected_ids)):
            raise ValueError("matching response selected_ids must be an array of strings")
        seen: set[str] = set()
        cleaned_ids: list[str] = []
        for candidate_id in selected_ids:
            if candidate_id not in allowed or candidate_id in seen:
                removed_ids.append(candidate_id)
                continue
            seen.add(candidate_id)
            cleaned_ids.append(candidate_id)
        cleaned_matches.append({"i": actual["i"], "selected_ids": cleaned_ids})
    if removed_ids:
        raise InvalidCandidateSelectionError(
            f"matching response selected an unknown or duplicate candidate: {removed_ids}",
            recovered_response={"matches": cleaned_matches},
            removed_ids=removed_ids,
        )


def _video_info(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("opencv-python is required for video evaluation") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return capture, {"fps": fps, "frame_count": frame_count, "duration_sec": duration}


def _read_frame(capture: Any, timestamp: float, fps: float):
    import cv2
    capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ok, frame = capture.read()
    # OpenCV commonly reports EOF for a seek exactly at duration. Preserve the
    # requested audit timestamp but decode the final available frame instead.
    if not ok and timestamp > 0 and fps > 0:
        # Frame-index seeking is more reliable than another timestamp seek at
        # the duration boundary (e.g. a 121-frame, 24-fps clip reports 5.0417s
        # while its last valid frame index is 120).
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(timestamp * fps)) - 1))
        ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to decode frame at {timestamp:.3f}s")
    return frame


def _candidate_by_id(candidates: Iterable[Candidate]) -> dict[str, Candidate]:
    return {candidate.id: candidate for candidate in candidates}


def _vlm_candidate(candidate: Candidate) -> dict[str, Any]:
    """Serialize the compact candidate contract sent to the VLM.

    The marker image is the source of spatial information: it draws each
    candidate's bbox and its stable ID label. Sending polygon coordinates again
    in JSON duplicates that information and consumes context without helping
    the model associate a candidate with the marked image region.
    """
    return {
        "id": candidate.id,
        "ocr_text": candidate.text,
        "confidence": float(candidate.confidence),
    }


def _frame_artifact_dirs(
    roots: Iterable[Path], timestamp: float, frame_index: int | None
) -> list[Path]:
    timestamp_token = f"{timestamp:.6f}".replace(".", "p")
    frame_token = "unknown" if frame_index is None else f"{frame_index:05d}"
    directories = []
    for root in roots:
        directory = root / "frames" / f"t_{timestamp_token}_frame_{frame_token}"
        directory.mkdir(parents=True, exist_ok=True)
        directories.append(directory)
    return directories


def _write_call_artifacts(
    directories: Iterable[Path],
    filename: str,
    *,
    call_type: str,
    marker_image: str,
    payload: dict[str, Any],
    attempts: list[dict[str, Any]],
    response: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    value = {
        "call_type": call_type,
        "marker_image": marker_image,
        "user_payload": payload,
        "attempts": attempts,
        "final_prompt": attempts[-1]["system_prompt"] if attempts else None,
        "final_response": response,
        "error": error,
    }
    for directory in directories:
        _write_json(directory / filename, value)


def _frame_observation(
    *,
    frame: Any,
    timestamp: float,
    record: dict[str, Any],
    carriers: list[dict[str, Any]],
    targets: list[Target],
    config: dict[str, Any],
    ocr_engine: PaddleOCREngine,
    vlm: VLMClient | None,
    carrier_prompt: str,
    matching_prompt: str,
    project_root: Path,
    language: str = "en",
    artifact_roots: Iterable[Path] = (),
    frame_index: int | None = None,
) -> dict[str, Any]:
    ocr = ocr_engine.predict(frame)
    matching_cfg = config.get("matching", {})
    ocr_cfg = config.get("ocr", {})
    active, duplicate_extra, capped_extra = filter_candidates(
        ocr,
        confidence_threshold=float(ocr_cfg.get("confidence_threshold", 0.80)),
        max_candidates=int(ocr_cfg.get("max_candidates", 20)),
        required_texts=[target.text for target in targets],
    )
    by_id = _candidate_by_id(active)
    frame_dirs = _frame_artifact_dirs(artifact_roots, timestamp, frame_index)
    marker = render_marker(frame, active, config.get("marker", {}).get("font_path") and project_root / config["marker"]["font_path"])
    marker_url = marker_data_url(marker)
    for directory in frame_dirs:
        try:
            import cv2
            cv2.imwrite(str(directory / "source_frame.jpg"), frame)
        except (ImportError, OSError):
            pass
        marker.save(directory / "carrier_marker.jpg", quality=92)
        _write_json(directory / "ocr_and_filtering.json", {
            "raw_candidates": [candidate.as_dict() for candidate in ocr],
            "active_candidates": [candidate.as_dict() for candidate in active],
            "duplicate_extras": [candidate.as_dict() for candidate in duplicate_extra],
            "capped_extras": [candidate.as_dict() for candidate in capped_extra],
            "config": ocr_cfg,
        })
    scene = (
        record.get("prompt_zh") if language == "zh" else record.get("prompt_en")
    ) or record.get("source_prompt", "")
    carrier_result = {"carriers": [{"carrier_id": item["carrier_id"], "present": False, "candidate_ids": []} for item in carriers]}
    carrier_attempts = 0
    if vlm is not None:
        carrier_payload = {"scene_context": scene, "carriers": carriers, "ocr_candidates": [_vlm_candidate(item) for item in active]}
        carrier_trace: list[dict[str, Any]] = []
        carrier_error: str | None = None
        try:
            carrier_result, carrier_attempts = vlm.complete_with_retries(
                carrier_prompt, carrier_payload, marker_url,
                lambda result: _validate_carrier_response(result, carriers, set(by_id)),
                response_format=carrier_response_format(
                    (item["carrier_id"] for item in carriers), by_id,
                ),
                trace_callback=carrier_trace.append,
            )
        except Exception as exc:
            carrier_error = str(exc)
            raise
        finally:
            _write_call_artifacts(
                frame_dirs, "carrier_vlm_call.json", call_type="carrier_consistency",
                marker_image="carrier_marker.jpg", payload=carrier_payload,
                attempts=carrier_trace, response=carrier_result if carrier_error is None else None,
                error=carrier_error,
            )
    else:
        # Explicitly useful for offline unit tests; real evaluation always uses VLM.
        carrier_result = {"carriers": [{"carrier_id": item["carrier_id"], "present": True, "candidate_ids": [candidate.id for candidate in active]} for item in carriers]}
    carrier_map = {item["carrier_id"]: item for item in carrier_result["carriers"]}
    carrier_owner = {candidate_id: item["carrier_id"] for item in carrier_result["carriers"] for candidate_id in item["candidate_ids"]}
    extra: list[Candidate] = list(duplicate_extra) + list(capped_extra)
    available = set(carrier_owner)
    matched_ids: set[str] = set()
    matches: list[dict[str, Any]] = []
    unresolved: list[Target] = []
    threshold = float(matching_cfg.get("fast_match_threshold", 0.95))
    for target in targets:
        carrier_state = carrier_map.get(target.carrier_id, {"present": False, "candidate_ids": []})
        if not carrier_state["present"]:
            matches.append({"slot_id": target.slot_id, "carrier_present": False, "selected_ids": [], "pred_text": "", "score": None})
            continue
        candidates_for_target = [by_id[cid] for cid in carrier_state["candidate_ids"] if cid in available and cid not in matched_ids]
        ranked = sorted(candidates_for_target, key=lambda c: (-normalized_edit_similarity(target.text, c.text), -c.confidence, c.id))
        if ranked and normalized_edit_similarity(target.text, ranked[0].text) >= threshold:
            selected = [ranked[0]]
            matched_ids.add(selected[0].id)
            pred = selected[0].text
            matches.append({"slot_id": target.slot_id, "carrier_present": True, "selected_ids": [selected[0].id], "pred_text": pred, "score": normalized_edit_similarity(target.text, pred)})
        else:
            unresolved.append(target)
    batch_size = max(1, int(matching_cfg.get("target_batch_size", 3)))
    match_attempts = 0
    for start in range(0, len(unresolved), batch_size):
        batch = unresolved[start:start + batch_size]
        batch_carriers = {target.carrier_id for target in batch}
        remaining = [
            candidate for candidate_id, candidate in by_id.items()
            if candidate_id in available
            and candidate_id not in matched_ids
            and carrier_owner.get(candidate_id) in batch_carriers
        ]
        batch_payload_targets = [{"i": index, "slot_id": target.slot_id, "target_text": target.text, "carrier": target.carrier, "position": target.position} for index, target in enumerate(batch)]
        selected_by_index: dict[int, list[str]] = {index: [] for index in range(len(batch))}
        if vlm is not None and remaining:
            batch_marker = render_marker(frame, remaining, config.get("marker", {}).get("font_path") and project_root / config["marker"]["font_path"])
            marker_name = f"matching_marker_batch_{start // batch_size:03d}.jpg"
            for directory in frame_dirs:
                batch_marker.save(directory / marker_name, quality=92)
            payload = {"scene_context": scene, "targets": batch_payload_targets, "ocr_candidates": [_vlm_candidate(candidate) for candidate in remaining]}
            matching_trace: list[dict[str, Any]] = []
            matching_error: str | None = None
            result: dict[str, Any] | None = None
            try:
                result, attempts = vlm.complete_with_retries(
                    matching_prompt, payload, marker_data_url(batch_marker),
                    lambda value: _validate_match_response(value, batch_payload_targets, {candidate.id for candidate in remaining}),
                    response_format=t2v_matching_response_format(
                        (target["i"] for target in batch_payload_targets),
                        (candidate.id for candidate in remaining),
                    ),
                    trace_callback=matching_trace.append,
                )
            except Exception as exc:
                matching_error = str(exc)
                raise
            finally:
                _write_call_artifacts(
                    frame_dirs, f"matching_vlm_call_batch_{start // batch_size:03d}.json",
                    call_type="target_candidate_matching", marker_image=marker_name,
                    payload=payload, attempts=matching_trace, response=result,
                    error=matching_error,
                )
            match_attempts += attempts
            if result is None:
                # A VLM client must return a validated object on success. Keep
                # this as a sample-level error instead of an assertion that can
                # terminate callers that invoke the frame evaluator directly.
                raise ValueError("VLM matching returned no result")
            selected_by_index = {int(item["i"]): list(item["selected_ids"]) for item in result["matches"]}
        batch_matched_ids: set[str] = set()
        for index, target in enumerate(batch):
            ids = [candidate_id for candidate_id in selected_by_index.get(index, []) if candidate_id in by_id]
            selected = [by_id[candidate_id] for candidate_id in ids]
            batch_matched_ids.update(ids)
            pred = "".join(candidate.text for candidate in selected)
            matches.append({"slot_id": target.slot_id, "carrier_present": True, "selected_ids": ids, "pred_text": pred, "score": normalized_edit_similarity(target.text, pred) if pred else 0.0})
        # Candidate reuse is allowed among targets in one request. Remove
        # selected candidates only before constructing the next request batch.
        matched_ids.update(batch_matched_ids)
    # Targets were first appended in carrier order, then unresolved matches are
    # appended at the end. Restore the GT slot order for deterministic output.
    order = {target.slot_id: index for index, target in enumerate(targets)}
    matches.sort(key=lambda item: order[item["slot_id"]])
    for candidate_id, candidate in by_id.items():
        if candidate_id not in matched_ids and candidate_id not in carrier_owner:
            extra.append(candidate)
        elif candidate_id in carrier_owner and candidate_id not in matched_ids:
            extra.append(candidate)
    matched_texts = [item["pred_text"] for item in matches if item["selected_ids"]]
    observation = {
        "timestamp": float(timestamp),
        "frame_index": frame_index,
        "ocr_candidates": [candidate.as_dict() for candidate in ocr],
        "active_candidates": [candidate.as_dict() for candidate in active],
        "carriers": carrier_result["carriers"],
        "matches": matches,
        "extra_ids": [candidate.id for candidate in extra],
        "extra_text_ratio": extra_text_ratio(extra, matched_texts),
        "vlm_attempts": {"carrier": carrier_attempts, "matching": match_attempts},
    }
    for directory in frame_dirs:
        _write_json(directory / "observation.json", observation)
    return observation


def _aggregate(frame_results: list[dict[str, Any]], targets: list[Target]) -> dict[str, Any]:
    by_slot: dict[str, list[dict[str, Any]]] = {target.slot_id: [] for target in targets}
    for frame in sorted(frame_results, key=lambda item: item["timestamp"]):
        for match in frame["matches"]:
            if match["carrier_present"]:
                by_slot.setdefault(match["slot_id"], []).append(match)
    spatial: dict[str, float] = {}
    temporal: dict[str, float] = {}
    target_details: list[dict[str, Any]] = []
    for target in targets:
        observations = by_slot.get(target.slot_id, [])
        if not observations:
            spatial[target.slot_id] = temporal[target.slot_id] = 0.0
            target_details.append({"slot_id": target.slot_id, "observations": 0, "spatial": 0.0, "temporal": 0.0})
            continue
        values = [float(item["score"] or 0.0) for item in observations]
        s_score = sum(values) / len(values)
        if len(values) == 1:
            t_score = s_score
        else:
            pair_scores = []
            for left, right in zip(observations, observations[1:]):
                if left["selected_ids"] and right["selected_ids"]:
                    pair_scores.append(normalized_edit_similarity(left["pred_text"], right["pred_text"]))
                else:
                    pair_scores.append(0.0)
            t_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        spatial[target.slot_id] = s_score
        temporal[target.slot_id] = t_score
        target_details.append({"slot_id": target.slot_id, "observations": len(observations), "spatial": s_score, "temporal": t_score})
    s_video = target_length_weighted_score(spatial, targets)
    t_video = target_length_weighted_score(temporal, targets)
    extra_values = [float(frame.get("extra_text_ratio", 0.0)) for frame in frame_results]
    return {
        "targets": target_details,
        "spatial_fidelity": s_video,
        "temporal_fidelity": t_video,
        "textv_score": harmonic_textv_score(s_video, t_video),
        "extra_text_ratio": sum(extra_values) / len(extra_values) if extra_values else 0.0,
    }


def evaluate_record(
    record: dict[str, Any],
    video_path: str | Path,
    config: dict[str, Any],
    *,
    language: str,
    project_root: Path | None = None,
    vlm: VLMClient | None = None,
    ocr_engine: PaddleOCREngine | None = None,
    cache_dir: str | Path | None = None,
    review_dir: str | Path | None = None,
) -> dict[str, Any]:
    project_root = project_root or Path.cwd()
    video_path = Path(video_path)
    language = _validate_language(language)
    carriers, targets = _targets(record, language)
    carrier_prompt = _load_prompt(config["vlm"]["carrier_prompt_path"], project_root)
    matching_prompt = _load_prompt(config["vlm"]["matching_prompt_path"], project_root)
    cache_file: Path | None = None
    cache_artifact_dir: Path | None = None
    review_path = Path(review_dir) if review_dir is not None else None
    scene_context = (record.get("prompt_zh") if language == "zh" else record.get("prompt_en")) or record.get("source_prompt", "")
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        stat = video_path.stat()
        key = _json_hash({
            "config": config,
            "sample_id": record.get("sample_id"),
            "spec": {"carriers": carriers, "targets": [target.__dict__ for target in targets]},
            "video": {"path": str(video_path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "language": language,
            "scene_context": scene_context,
            "prompts": [_json_hash(carrier_prompt), _json_hash(matching_prompt)],
        })
        cache_artifact_dir = cache_root / f"{record.get('sample_id', 'record')}.{key}"
        cache_file = cache_artifact_dir / "result.json"
        if cache_file.exists() and review_path is None:
            return json.loads(cache_file.read_text(encoding="utf-8"))
    artifact_roots = [path for path in (cache_artifact_dir, review_path) if path is not None]
    for root in artifact_roots:
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "input_record.json", record)
        _write_json(root / "resolved_config.json", redact_credentials(config))
        prompts_dir = root / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / "carrier_consistency_system_prompt.md").write_text(carrier_prompt, encoding="utf-8")
        (prompts_dir / "target_candidate_matching_system_prompt.md").write_text(matching_prompt, encoding="utf-8")
    capture, metadata = _video_info(video_path)
    try:
        sampling = config.get("sampling", {})
        timestamps = sample_timestamps(
            metadata["duration_sec"], fps=float(sampling.get("fps", 3)),
            min_frames=int(sampling.get("min_frames", 1)), max_frames=int(sampling.get("max_frames", 30)),
            seed=int(sampling.get("seed", 42)), sample_id=str(record.get("sample_id", "")),
            jitter_sec=float(sampling.get("jitter_sec", 0.0)),
        )
        manifest = {
            "sample_id": record.get("sample_id"),
            "language": language,
            "language_contract": "explicit CLI/programmatic argument; never inferred from the record",
            "video_path": str(video_path.resolve()),
            "video": metadata,
            "scene_context": scene_context,
            "carriers": carriers,
            "targets": [target.__dict__ for target in targets],
            "initial_timestamps": timestamps,
            "files": {
                "input_record": "input_record.json",
                "resolved_config": "resolved_config.json",
                "carrier_prompt": "prompts/carrier_consistency_system_prompt.md",
                "matching_prompt": "prompts/target_candidate_matching_system_prompt.md",
                "frames": "frames/<timestamp>_<frame_index>/",
                "result": "result.json",
            },
        }
        for root in artifact_roots:
            _write_json(root / "manifest.json", manifest)
        ocr_engine = ocr_engine or PaddleOCREngine(config, project_root)
        frame_results = []
        for timestamp in timestamps:
            frame_index = min(int(round(timestamp * metadata["fps"])), metadata["frame_count"] - 1) if metadata["fps"] and metadata["frame_count"] else None
            frame_results.append(_frame_observation(
                frame=_read_frame(capture, timestamp, metadata["fps"]), timestamp=timestamp,
                record=record, carriers=carriers, targets=targets, config=config,
                ocr_engine=ocr_engine, vlm=vlm, carrier_prompt=carrier_prompt,
                matching_prompt=matching_prompt, project_root=project_root,
                language=language,
                artifact_roots=artifact_roots, frame_index=frame_index,
            ))
        # Adaptive hinting is deliberately bounded and only uses observable
        # match transitions / NED jumps from the first sparse pass.
        if sampling.get("adaptive", True) and len(frame_results) < int(sampling.get("max_frames", 30)):
            jump = float(sampling.get("ned_jump_threshold", 0.35))
            existing = {round(item["timestamp"], 6) for item in frame_results}
            additions: list[float] = []
            ordered = sorted(frame_results, key=lambda item: item["timestamp"])
            for left, right in zip(ordered, ordered[1:]):
                left_by = {item["slot_id"]: item for item in left["matches"]}
                right_by = {item["slot_id"]: item for item in right["matches"]}
                trigger = False
                for slot_id in left_by:
                    lm, rm = left_by[slot_id], right_by.get(slot_id, {})
                    if sampling.get("carrier_presence_hint", False) and bool(lm.get("carrier_present")) != bool(rm.get("carrier_present")):
                        trigger = True
                    elif lm.get("carrier_present") and rm.get("carrier_present") and bool(lm.get("selected_ids")) != bool(rm.get("selected_ids")):
                        trigger = True
                    elif lm.get("carrier_present") and rm.get("carrier_present") and lm.get("selected_ids") and rm.get("selected_ids") and abs(float(lm.get("score", 0)) - float(rm.get("score", 0))) >= jump:
                        trigger = True
                if trigger:
                    midpoint = round((left["timestamp"] + right["timestamp"]) / 2.0, 6)
                    if midpoint not in existing:
                        additions.append(midpoint)
            capacity = int(sampling.get("max_frames", 30)) - len(frame_results)
            for timestamp in sorted(additions)[:max(0, capacity)]:
                frame_index = min(int(round(timestamp * metadata["fps"])), metadata["frame_count"] - 1) if metadata["fps"] and metadata["frame_count"] else None
                frame_results.append(_frame_observation(
                    frame=_read_frame(capture, timestamp, metadata["fps"]), timestamp=timestamp,
                    record=record, carriers=carriers, targets=targets, config=config,
                    ocr_engine=ocr_engine, vlm=vlm, carrier_prompt=carrier_prompt,
                    matching_prompt=matching_prompt, project_root=project_root,
                    language=language,
                    artifact_roots=artifact_roots, frame_index=frame_index,
                ))
        result = {
            "sample_id": record.get("sample_id"),
            "video_path": str(video_path),
            "language": language,
            "sampling": {"timestamps": sorted(item["timestamp"] for item in frame_results), "config": config.get("sampling", {})},
            "video": metadata,
            "frames": sorted(frame_results, key=lambda item: item["timestamp"]),
            "metrics": _aggregate(frame_results, targets),
        }
        for root in artifact_roots:
            _write_json(root / "result.json", result)
            completed_manifest = {**manifest, "final_timestamps": result["sampling"]["timestamps"], "metrics": result["metrics"]}
            _write_json(root / "manifest.json", completed_manifest)
        return result
    finally:
        capture.release()


def _load_video_manifest(path: str | Path) -> dict[str, Path]:
    """Load a JSONL sample_id -> video path mapping.

    Paths in the manifest are resolved relative to the manifest itself, so the
    mapping remains portable when a model run is moved as one directory.
    """
    manifest_path = Path(path)
    mapping: dict[str, Path] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid video manifest JSON at line {line_number}: {manifest_path}") from exc
        if not isinstance(item, dict) or not item.get("sample_id") or not item.get("video_path"):
            raise ValueError(
                f"video manifest line {line_number} must contain sample_id and video_path: {manifest_path}"
            )
        sample_id = str(item["sample_id"])
        if sample_id in mapping:
            raise ValueError(f"duplicate sample_id in video manifest: {sample_id}")
        video = Path(str(item["video_path"]))
        mapping[sample_id] = video if video.is_absolute() else (manifest_path.parent / video)
    return mapping


def _resolve_video(
    record: dict[str, Any],
    video_dir: Path | None,
    explicit_video: Path | None,
    video_manifest: dict[str, Path] | None = None,
) -> Path:
    if explicit_video:
        return explicit_video
    sample_id = str(record.get("sample_id", ""))
    if video_manifest is not None:
        if sample_id not in video_manifest:
            raise FileNotFoundError(f"video manifest has no entry for sample_id={sample_id}")
        return video_manifest[sample_id]
    # An explicit directory is a strict batch input: do not silently fall back
    # to a path embedded in the prompt record when its video is missing.
    if video_dir:
        for suffix in (".mp4", ".mov", ".webm", ".avi"):
            path = video_dir / f"{sample_id}{suffix}"
            if path.exists():
                return path
        raise FileNotFoundError(
            f"no video found for sample_id={sample_id} under video-dir={video_dir}"
        )
    for key in ("video_path", "video", "source_video"):
        if record.get(key):
            return Path(record[key])
    raise FileNotFoundError(f"no video found for sample_id={record.get('sample_id')}")


def run(
    input_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    language: str,
    video_dir: str | Path | None = None,
    video_path: str | Path | None = None,
    video_manifest: str | Path | None = None,
    sample_id: str | None = None,
    override: bool = False,
    project_root: Path | None = None,
    review_root: str | Path | None = None,
    debug: bool = False,
) -> int:
    """Evaluate JSONL records and write metrics, with optional debug bundles."""
    input_path, output_path = Path(input_path), Path(output_path)
    project_root = project_root or Path.cwd()
    # Keep the programmatic review_root API backward compatible while making
    # the CLI's --debug-dir switch the documented interface.
    debug = debug or review_root is not None
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_id:
        rows = [row for row in rows if row.get("sample_id") == sample_id]
    passed_path = output_path if output_path.parent.name == "passed" else output_path.parent / "passed" / output_path.name
    failed_path = passed_path.parent.parent / "failed" / f"{passed_path.stem}_errors.jsonl"
    passed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if passed_path.exists() and not override:
        for line in passed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                    if value.get("evaluator_t2v", {}).get("audit", {}).get("status") == "ok":
                        completed_ids.add(str(value.get("sample_id")))
                except json.JSONDecodeError:
                    continue
    rows = [row for row in rows if str(row.get("sample_id")) not in completed_ids]
    language = _validate_language(language)
    video_dir_path = Path(video_dir) if video_dir else None
    explicit_video = Path(video_path) if video_path else None
    video_manifest_map = _load_video_manifest(video_manifest) if video_manifest else None
    if explicit_video and not sample_id and len(rows) > 1:
        raise ValueError("--video is for one record; also pass --sample-id when input contains multiple records")
    pending_ids = {str(record.get("sample_id")) for record in rows}
    if override:
        passed_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)
    else:
        preserved_errors: list[dict[str, Any]] = []
        if failed_path.exists():
            for line in failed_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(value.get("sample_id"))
                if sid not in pending_ids and sid not in completed_ids:
                    preserved_errors.append(value)
        failed_path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in preserved_errors),
            encoding="utf-8",
        )
    vlm = VLMClient(config)
    ocr_engine = PaddleOCREngine(config, project_root)
    ok_count = error_count = 0
    with passed_path.open("a", encoding="utf-8") as passed, failed_path.open("a", encoding="utf-8") as failed:
        for record in rows:
            try:
                resolved = _resolve_video(record, video_dir_path, explicit_video, video_manifest_map)
                sample_review_dir = (
                    Path(review_root) / f"{record.get('sample_id', 'record')}_{language}"
                    if debug and review_root is not None else None
                )
                result = evaluate_record(
                    record, resolved, config, project_root=project_root, language=language,
                    vlm=vlm, ocr_engine=ocr_engine,
                    cache_dir=passed_path.parent.parent / "cache" / "t2v" if debug else None,
                    review_dir=sample_review_dir,
                )
                audit = {
                    "status": "ok", "model_id": vlm.model, "language": language,
                    "video_path": str(resolved), "output_mode": "debug" if debug else "metrics",
                    "created_at": _now(),
                }
                evaluator = {"audit": audit, "metrics": result["metrics"]}
                if debug:
                    evaluator["result"] = result
                output = {**record, "evaluator_t2v": evaluator}
                passed.write(json.dumps(output, ensure_ascii=False) + "\n")
                passed.flush()
                os.fsync(passed.fileno())
                ok_count += 1
            except (OCRError, VLMError):
                raise
            except Exception as exc:
                error = {**record, "category": "error", "error": str(exc), "evaluator_t2v": {"audit": {"status": "error", "created_at": _now()}}}
                failed.write(json.dumps(error, ensure_ascii=False) + "\n")
                failed.flush()
                os.fsync(failed.fileno())
                error_count += 1
    summary_path = passed_path.parent.parent / "summary.json"
    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    all_ok = all_error = 0
    if passed_path.exists():
        for line in passed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    all_ok += int(json.loads(line).get("evaluator_t2v", {}).get("audit", {}).get("status") == "ok")
                except json.JSONDecodeError:
                    all_error += 1
    if failed_path.exists():
        all_error += sum(1 for line in failed_path.read_text(encoding="utf-8").splitlines() if line.strip())
    successful_metrics: list[dict[str, Any]] = []
    if passed_path.exists():
        for line in passed_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = value.get("evaluator_t2v", {}).get("metrics")
            if isinstance(metrics, dict):
                successful_metrics.append(metrics)
    summary["benchmark_metrics"] = aggregate_benchmark_metrics(successful_metrics)
    summary["evaluator_t2v"] = {"total": all_ok + all_error, "ok": all_ok, "error": all_error, "output": passed_path.relative_to(passed_path.parent.parent).as_posix(), "errors": failed_path.relative_to(passed_path.parent.parent).as_posix(), "updated_at": _now()}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return ok_count + error_count
