"""End-to-end independent Ref2V evaluator."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm

from ...benchmark import aggregate_benchmark_metrics
from ...privacy import redact_credentials
from ..vlm_schema import ref_matching_response_format, ref_routing_response_format
from ..vlm_validation import InvalidCandidateSelectionError
from .core import (
    Candidate,
    Slot,
    bbox_center,
    extra_text_ratio,
    filter_candidates,
    harmonic_textv_score,
    normalized_edit_similarity,
    reading_order_score,
    sample_timestamps,
    target_length_weighted_score,
)
from .marker import image_data_url, render_marker, render_reference_label, resize_longest_edge
from .ocr import OCRError, PaddleOCREngine
from .vlm import VLMClient, VLMError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _load_prompt(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.read_text(encoding="utf-8")


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_") or "record"


def _resolve_image(image_dir: Path, filename: str) -> Path:
    path = image_dir / filename
    if path.is_file():
        return path
    # Test fixtures may retain a source-relative name while images are flattened.
    flat = image_dir / Path(filename).name
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"reference image not found under {image_dir}: {filename}")


def _resolve_video(record: dict[str, Any], *, video: Path | None, video_dir: Path | None,
                   video_manifest: dict[str, Path] | None) -> Path:
    if video is not None:
        return video
    sample_id = str(record.get("sample_id", ""))
    if video_manifest is not None:
        if sample_id not in video_manifest:
            raise FileNotFoundError(f"video manifest has no entry for sample_id={sample_id}")
        return video_manifest[sample_id]
    if video_dir is not None:
        sample_path = Path(sample_id)
        for suffix in (".mp4", ".mov", ".webm", ".avi"):
            candidate = video_dir / sample_path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        for suffix in (".mp4", ".mov", ".webm", ".avi"):
            flat = video_dir / f"{sample_path.stem}{suffix}"
            if flat.is_file():
                return flat
        raise FileNotFoundError(f"no video found for sample_id={sample_id} under {video_dir}")
    for key in ("video_path", "video", "source_video"):
        if record.get(key):
            return Path(str(record[key]))
    raise FileNotFoundError(f"no video mapping for sample_id={sample_id}")


def _load_video_manifest(path: str | Path) -> dict[str, Path]:
    manifest = Path(path)
    mapping: dict[str, Path] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not item.get("sample_id") or not item.get("video_path"):
            raise ValueError(f"video manifest line {line_number} must contain sample_id and video_path")
        sample_id = str(item["sample_id"])
        if sample_id in mapping:
            raise ValueError(f"duplicate sample_id in video manifest: {sample_id}")
        value = Path(str(item["video_path"]))
        mapping[sample_id] = value if value.is_absolute() else manifest.parent / value
    return mapping


def _video_info(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for Ref2V video evaluation") from exc
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
    if not ok and timestamp > 0 and fps > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(timestamp * fps)) - 1))
        ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to decode frame at {timestamp:.3f}s")
    return frame


def _candidate_by_id(candidates: Iterable[Candidate]) -> dict[str, Candidate]:
    return {candidate.id: candidate for candidate in candidates}


def _compact_candidate(candidate: Candidate) -> dict[str, Any]:
    return {"id": candidate.id, "ocr_text": candidate.text, "confidence": float(candidate.confidence)}


def _validate_ref_response(value: dict[str, Any], allowed: set[str]) -> None:
    if set(value) != {"matched_ref_ids"} or not isinstance(value["matched_ref_ids"], list):
        raise ValueError("ref-id response must contain only matched_ref_ids array")
    ids = value["matched_ref_ids"]
    if any(not isinstance(item, str) for item in ids):
        raise ValueError("ref-id response IDs must be strings")
    if len(ids) != len(set(ids)) or not set(ids).issubset(allowed):
        raise ValueError("ref-id response contains duplicate or unknown IDs")


def _validate_match_response(value: dict[str, Any], targets: list[dict[str, Any]], allowed: set[str]) -> None:
    if set(value) != {"matches"} or not isinstance(value["matches"], list):
        raise ValueError("matching response must contain only matches array")
    if len(value["matches"]) != len(targets):
        raise ValueError("matching response must contain one item per target")
    cleaned_matches: list[dict[str, Any]] = []
    removed_ids: list[str] = []
    for expected, actual in zip(targets, value["matches"]):
        if (not isinstance(actual, dict)
                or set(actual) != {"slot_id", "selected_ids"}
                or actual["slot_id"] != expected["slot_id"]):
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
        cleaned_matches.append({"slot_id": actual["slot_id"], "selected_ids": cleaned_ids})
    if removed_ids:
        raise InvalidCandidateSelectionError(
            f"matching response selected invalid or duplicate candidates: {removed_ids}",
            recovered_response={"matches": cleaned_matches},
            removed_ids=removed_ids,
        )


def _extract_reference_slots(
    record: dict[str, Any], image_dir: Path, ocr_engine: Any, *, font_path: Path | None,
    debug_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[Slot]]:
    references: list[dict[str, Any]] = []
    slots: list[Slot] = []
    next_slot = 1
    for ref in record.get("reference_images", []):
        ref_id = str(ref.get("ref_id", ""))
        filename = str(ref.get("filename", ""))
        if not ref_id or not filename:
            raise ValueError("each reference image requires ref_id and filename")
        image_path = _resolve_image(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        # OCR uses the original image. VLM reference inputs and their debug
        # copies are compact, labeled images with a maximum edge of 256 px.
        labeled = render_reference_label(resize_longest_edge(image, 256), ref_id, font_path)
        labeled = resize_longest_edge(labeled, 256)
        detections = list(ocr_engine.predict(image_path))
        detections = [item for item in detections if str(item.text).strip()]
        ordered = sorted(detections, key=lambda item: bbox_center(item.bbox)[::-1])
        carrier_id = f"C_{ref_id}"
        ref_slots: list[Slot] = []
        for order, detection in enumerate(ordered, 1):
            slot = Slot(f"T{next_slot:04d}", detection.text, ref_id, carrier_id, order, detection.bbox)
            slots.append(slot)
            ref_slots.append(slot)
            next_slot += 1
        references.append({
            "ref_id": ref_id,
            "filename": filename,
            "image_path": str(image_path),
            "carrier_id": carrier_id,
            "carrier": str(ref.get("carrier") or "complete visual carrier in this reference image"),
            "category": str(ref.get("category", "")),
            "slots": ref_slots,
            "image": labeled,
        })
        if debug_dir is not None:
            labeled.save(debug_dir / f"reference_{ref_id}.jpg", quality=92)
            _write_json(debug_dir / f"reference_{ref_id}_ocr.json", {
                "ref_id": ref_id,
                "filename": filename,
                "image_path": str(image_path),
                "slots": [{"slot_id": item.slot_id, "target_text": item.target_text,
                            "reading_order": item.reading_order, "bbox": item.bbox} for item in ref_slots],
            })
    if not references:
        raise ValueError("sample has no reference_images")
    return references, slots


def _ref_by_id(references: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["ref_id"]): item for item in references}


def _route_refs(
    frame: Any,
    references: list[dict[str, Any]],
    ref_prompt: str,
    vlm: Any,
    *,
    trace_dir: Path | None,
) -> tuple[list[str], int]:
    if len(references) == 1:
        return [str(references[0]["ref_id"])], 0
    if vlm is None:
        return [], 0
    urls = [image_data_url(item["image"]) for item in references]
    from .marker import as_image
    urls.append(image_data_url(as_image(frame)))
    payload = {
        "reference_images": [{"ref_id": item["ref_id"], "carrier": item["carrier"],
                              "category": item.get("category", "")} for item in references],
        "generated_frame": "last image part",
    }
    traces: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    attempts = 0
    error: str | None = None
    try:
        result, attempts = vlm.complete_with_retries(
            ref_prompt, payload, urls,
            lambda value: _validate_ref_response(value, {str(item["ref_id"]) for item in references}),
            response_format=ref_routing_response_format(
                str(item["ref_id"]) for item in references
            ),
            trace_callback=traces.append,
        )
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if trace_dir is not None:
            _write_json(trace_dir / "ref_id_matching_vlm_call.json", {
                "call_type": "ref_id_matching", "payload": payload, "attempts": traces,
                "response": result, "error": error,
                "reference_ids": [item["ref_id"] for item in references],
            })
    if result is None:
        raise ValueError("VLM ref-id matching returned no result")
    return [str(item) for item in result["matched_ref_ids"]], attempts


def _fallback_match(batch: list[Slot], candidates: list[Candidate]) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for index, slot in enumerate(batch):
        ranked = sorted(
            candidates,
            key=lambda candidate: (-normalized_edit_similarity(slot.target_text, candidate.text), -candidate.confidence, candidate.id),
        )
        if ranked and normalized_edit_similarity(slot.target_text, ranked[0].text) > 0.0:
            result[index] = [ranked[0].id]
        else:
            result[index] = []
    return result


def _target_payload(slot: Slot) -> dict[str, Any]:
    return {
        "slot_id": slot.slot_id,
        "target_text": slot.target_text,
        "ref_id": slot.ref_id,
    }


def _frame_observation(
    *, frame: Any, timestamp: float, frame_index: int | None, references: list[dict[str, Any]], slots: list[Slot],
    config: dict[str, Any], ocr_engine: Any, vlm: Any, ref_prompt: str, matching_prompt: str,
    font_path: Path | None, artifact_dir: Path | None, scene_context: str,
) -> dict[str, Any]:
    raw = list(ocr_engine.predict(frame))
    matched_ref_ids, route_attempts = _route_refs(frame, references, ref_prompt, vlm, trace_dir=artifact_dir)
    if not matched_ref_ids:
        observation = {
            "timestamp": float(timestamp), "frame_index": frame_index, "evaluated": False,
            "matched_ref_ids": [], "ocr_candidates": [item.as_dict() for item in raw],
            "matches": [], "extra_ids": [item.id for item in raw], "extra_text_ratio": 0.0,
            "vlm_attempts": {"ref_id": route_attempts, "matching": 0},
        }
        if artifact_dir is not None:
            _write_json(artifact_dir / "observation.json", observation)
        return observation

    active_slots = [slot for slot in slots if slot.ref_id in set(matched_ref_ids)]
    ocr_cfg = config.get("ocr", {})
    active, duplicate_extra, capped_extra = filter_candidates(
        raw,
        confidence_threshold=float(ocr_cfg.get("confidence_threshold", 0.80)),
        max_candidates=int(ocr_cfg.get("max_candidates", 20)),
        required_texts=[slot.target_text for slot in active_slots],
    )
    by_id = _candidate_by_id(active)
    marker = render_marker(frame, active, font_path)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            cv2.imwrite(str(artifact_dir / "source_frame.jpg"), frame)
        except (ImportError, OSError):
            pass
        marker.save(artifact_dir / "marker.jpg", quality=92)
        _write_json(artifact_dir / "ocr_and_filtering.json", {
            "raw_candidates": [item.as_dict() for item in raw],
            "active_candidates": [item.as_dict() for item in active],
            "duplicate_extras": [item.as_dict() for item in duplicate_extra],
            "capped_extras": [item.as_dict() for item in capped_extra],
        })

    matched_ids: set[str] = set()
    matches_by_slot: dict[str, dict[str, Any]] = {}
    matching_attempts = 0
    batch_size = max(1, int(config.get("matching", {}).get("target_batch_size", 3)))
    references_by_id = _ref_by_id(references)
    for start in range(0, len(active_slots), batch_size):
        batch = active_slots[start:start + batch_size]
        remaining = [candidate for candidate in active if candidate.id not in matched_ids]
        # Keep the visual marker consistent with the structured candidate list.
        # Previously, already-assigned candidates remained visible in later
        # batches and could be selected again by the VLM.
        batch_marker = render_marker(frame, remaining, font_path)
        payload_targets = [_target_payload(slot) for slot in batch]
        selected_by_slot: dict[str, list[str]] = {slot.slot_id: [] for slot in batch}
        marker_url = image_data_url(batch_marker)
        reference_urls = [image_data_url(references_by_id[ref_id]["image"]) for ref_id in dict.fromkeys(slot.ref_id for slot in batch)]
        images = reference_urls + [marker_url]
        payload = {
            "reference_images": [{"ref_id": ref_id, "carrier": references_by_id[ref_id]["carrier"],
                                  "category": references_by_id[ref_id].get("category", "")}
                                 for ref_id in dict.fromkeys(slot.ref_id for slot in batch)],
            "scene_context": scene_context,
            "targets": payload_targets,
            "ocr_candidates": [_compact_candidate(item) for item in remaining],
        }
        traces: list[dict[str, Any]] = []
        response: dict[str, Any] | None = None
        if remaining:
            if vlm is None:
                selected_by_index = _fallback_match(batch, remaining)
                selected_by_slot = {
                    slot.slot_id: selected_by_index.get(index, [])
                    for index, slot in enumerate(batch)
                }
            else:
                matching_error: str | None = None
                try:
                    response, attempts = vlm.complete_with_retries(
                        matching_prompt, payload, images,
                        lambda value: _validate_match_response(value, payload_targets, {item.id for item in remaining}),
                        response_format=ref_matching_response_format(
                            (target["slot_id"] for target in payload_targets),
                            (item.id for item in remaining),
                        ),
                        trace_callback=traces.append,
                    )
                except Exception as exc:
                    matching_error = str(exc)
                    raise
                finally:
                    if artifact_dir is not None:
                        marker_name = f"matching_marker_batch_{start // batch_size:03d}.jpg"
                        batch_marker.save(artifact_dir / marker_name, quality=92)
                        _write_json(artifact_dir / f"matching_vlm_call_batch_{start // batch_size:03d}.json", {
                            "call_type": "reference_grounded_matching", "marker_image": marker_name,
                            "payload": payload, "attempts": traces, "response": response,
                            "error": matching_error,
                        })
                matching_attempts += attempts
                if response is None:
                    raise ValueError("VLM target matching returned no result")
                selected_by_slot = {
                    str(item["slot_id"]): list(item["selected_ids"])
                    for item in response["matches"]
                }
        if artifact_dir is not None and (vlm is None or not remaining):
            marker_name = f"matching_marker_batch_{start // batch_size:03d}.jpg"
            batch_marker.save(artifact_dir / marker_name, quality=92)
            _write_json(artifact_dir / f"matching_vlm_call_batch_{start // batch_size:03d}.json", {
                "call_type": "reference_grounded_matching", "marker_image": marker_name,
                "payload": payload, "attempts": traces, "response": response,
            })
        batch_matched_ids: set[str] = set()
        for slot in batch:
            ids = [item for item in selected_by_slot.get(slot.slot_id, [])
                   if item in by_id]
            selected = [by_id[item] for item in ids]
            batch_matched_ids.update(ids)
            text = "".join(item.text for item in selected)
            matches_by_slot[slot.slot_id] = {
                "slot_id": slot.slot_id, "ref_id": slot.ref_id, "carrier_present": True,
                "selected_ids": ids, "selected_bboxes": [item.bbox for item in selected],
                "pred_text": text, "score": normalized_edit_similarity(slot.target_text, text) if text else 0.0,
            }
        # A candidate may serve multiple targets within this request. Remove
        # it only when constructing the next request batch.
        matched_ids.update(batch_matched_ids)
    extra = list(duplicate_extra) + list(capped_extra)
    extra.extend(item for item in active if item.id not in matched_ids)
    matches = [matches_by_slot[slot.slot_id] for slot in active_slots]
    matched_texts = [item["pred_text"] for item in matches if item["selected_ids"]]
    observation = {
        "timestamp": float(timestamp), "frame_index": frame_index, "evaluated": True,
        "matched_ref_ids": matched_ref_ids, "ocr_candidates": [item.as_dict() for item in raw],
        "active_candidates": [item.as_dict() for item in active], "matches": matches,
        "extra_ids": [item.id for item in extra], "extra_text_ratio": extra_text_ratio(extra, matched_texts),
        "vlm_attempts": {"ref_id": route_attempts, "matching": matching_attempts},
    }
    if artifact_dir is not None:
        _write_json(artifact_dir / "observation.json", observation)
    return observation


def _aggregate(frame_results: list[dict[str, Any]], references: list[dict[str, Any]], slots: list[Slot]) -> dict[str, Any]:
    by_slot: dict[str, list[dict[str, Any]]] = {slot.slot_id: [] for slot in slots}
    reading_by_ref: dict[str, list[float]] = {str(ref["ref_id"]): [] for ref in references}
    extra_values: list[float] = []
    for frame in sorted(frame_results, key=lambda item: item["timestamp"]):
        if not frame.get("evaluated"):
            continue
        extra_values.append(float(frame.get("extra_text_ratio", 0.0)))
        for match in frame["matches"]:
            if match["carrier_present"]:
                by_slot.setdefault(match["slot_id"], []).append(match)
        for ref_id in frame.get("matched_ref_ids", []):
            ref_slots = [slot for slot in slots if slot.ref_id == ref_id]
            predicted: list[tuple[float, float, str]] = []
            for match in frame["matches"]:
                if match["ref_id"] != ref_id or not match["selected_ids"]:
                    continue
                boxes = match.get("selected_bboxes", [])
                points = [point for box in boxes for point in box]
                center = bbox_center(points)
                predicted.append((center[0], center[1], match["slot_id"]))
            predicted_order = [item[2] for item in sorted(predicted, key=lambda item: (item[1], item[0]))]
            reference_order = [slot.slot_id for slot in sorted(ref_slots, key=lambda item: item.reading_order)]
            reading_by_ref[ref_id].append(reading_order_score(reference_order, predicted_order))

    spatial_by_ref: dict[str, float] = {}
    temporal_by_ref: dict[str, float] = {}
    target_details: list[dict[str, Any]] = []
    for ref in references:
        ref_id = str(ref["ref_id"])
        ref_slots = [slot for slot in slots if slot.ref_id == ref_id]
        spatial_scores: dict[str, float] = {}
        temporal_scores: dict[str, float] = {}
        for slot in ref_slots:
            observations = by_slot.get(slot.slot_id, [])
            values = [float(item.get("score") or 0.0) for item in observations]
            spatial = sum(values) / len(values) if values else 0.0
            if len(values) <= 1:
                temporal = spatial
            else:
                pairs = []
                for left, right in zip(observations, observations[1:]):
                    if left["selected_ids"] and right["selected_ids"]:
                        pairs.append(normalized_edit_similarity(left["pred_text"], right["pred_text"]))
                    else:
                        pairs.append(0.0)
                temporal = sum(pairs) / len(pairs) if pairs else 0.0
            spatial_scores[slot.slot_id] = spatial
            temporal_scores[slot.slot_id] = temporal
            target_details.append({"slot_id": slot.slot_id, "ref_id": ref_id,
                                   "target_text": slot.target_text, "observations": len(observations),
                                   "spatial": spatial, "temporal": temporal,
                                   "reading_order": slot.reading_order})
        spatial_by_ref[ref_id] = target_length_weighted_score(spatial_scores, ref_slots)
        temporal_by_ref[ref_id] = target_length_weighted_score(temporal_scores, ref_slots)

    spatial = sum(spatial_by_ref.values()) / len(references) if references else 0.0
    temporal = sum(temporal_by_ref.values()) / len(references) if references else 0.0
    reading_scores = {
        ref_id: (sum(values) / len(values) if values else 0.0)
        for ref_id, values in reading_by_ref.items()
    }
    per_target_spatial = {
        item["slot_id"]: float(item["spatial"]) for item in target_details
    }
    per_target_temporal = {
        item["slot_id"]: float(item["temporal"]) for item in target_details
    }
    return {
        "targets": target_details,
        "refs": [{"ref_id": str(ref["ref_id"]), "target_count": len([slot for slot in slots if slot.ref_id == ref["ref_id"]]),
                  "spatial": spatial_by_ref[str(ref["ref_id"])], "temporal": temporal_by_ref[str(ref["ref_id"])],
                  "reading_order": reading_scores[str(ref["ref_id"])]} for ref in references],
        "per_target_spatial_fidelity": per_target_spatial,
        "per_target_temporal_fidelity": per_target_temporal,
        "length_weighted_spatial_fidelity": spatial,
        "length_weighted_temporal_fidelity": temporal,
        # Short aliases retain the T2V evaluator's existing JSON vocabulary.
        "spatial_fidelity": spatial,
        "temporal_fidelity": temporal,
        "textv_score": harmonic_textv_score(spatial, temporal),
        "extra_text_ratio": sum(extra_values) / len(extra_values) if extra_values else 0.0,
        "reading_order_score": sum(reading_scores.values()) / len(reading_scores) if reading_scores else 0.0,
    }


def evaluate_record(
    record: dict[str, Any], video_path: str | Path, config: dict[str, Any], *, image_dir: str | Path,
    project_root: Path | None = None, vlm: Any | None = None, ocr_engine: Any | None = None,
    debug_dir: str | Path | None = None, include_frames: bool = False, show_progress: bool = True,
) -> dict[str, Any]:
    project_root = project_root or Path.cwd()
    image_root = Path(image_dir)
    video_path = Path(video_path)
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)
    ref_prompt = _load_prompt(config["vlm"]["ref_id_prompt_path"], project_root)
    matching_prompt = _load_prompt(config["vlm"]["matching_prompt_path"], project_root)
    font_value = config.get("marker", {}).get("font_path")
    font_path = project_root / font_value if font_value and not Path(font_value).is_absolute() else Path(font_value) if font_value else None
    if ocr_engine is None:
        ocr_engine = PaddleOCREngine(config, project_root)
    references, slots = _extract_reference_slots(record, image_root, ocr_engine, font_path=font_path, debug_dir=debug_path)
    capture, metadata = _video_info(video_path)
    scene_context = str(record.get("prompt", ""))
    try:
        sampling_cfg = config.get("sampling", {})
        timestamps = sample_timestamps(metadata["duration_sec"], fps=float(sampling_cfg.get("fps", 3)),
                                       min_frames=int(sampling_cfg.get("min_frames", 1)),
                                       max_frames=int(sampling_cfg.get("max_frames", 30)),
                                       seed=int(sampling_cfg.get("seed", 42)),
                                       sample_id=str(record.get("sample_id", "")),
                                       jitter_sec=float(sampling_cfg.get("jitter_sec", 0.0)))
        if debug_path is not None:
            prompts_dir = debug_path / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / "ref_id_matching_system_prompt.md").write_text(ref_prompt, encoding="utf-8")
            (prompts_dir / "target_candidate_matching_system_prompt.md").write_text(matching_prompt, encoding="utf-8")
            _write_json(debug_path / "manifest.json", {
                "sample_id": record.get("sample_id"),
                "video_path": str(video_path.resolve()),
                "video": metadata,
                "scene_context": scene_context,
                "references": [{"ref_id": item["ref_id"], "filename": item["filename"]} for item in references],
                "initial_timestamps": timestamps,
                "files": {"input_record": "input_record.json", "resolved_config": "resolved_config.json",
                          "prompts": "prompts/", "frames": "frames/", "result": "result.json"},
            })
        frame_results: list[dict[str, Any]] = []
        frame_context = (
            tqdm(
                timestamps,
                desc=f"Ref2V {record.get('sample_id', 'record')} frames",
                unit="frame",
                leave=False,
            )
            if show_progress else nullcontext(timestamps)
        )
        with frame_context as frame_progress:
            for timestamp in frame_progress:
                frame_index = min(int(round(timestamp * metadata["fps"])), metadata["frame_count"] - 1) if metadata["fps"] and metadata["frame_count"] else None
                frame_dir = debug_path / "frames" / f"t_{timestamp:.6f}".replace(".", "p") if debug_path is not None else None
                frame_results.append(_frame_observation(
                    frame=_read_frame(capture, timestamp, metadata["fps"]), timestamp=timestamp, frame_index=frame_index,
                    references=references, slots=slots, config=config, ocr_engine=ocr_engine, vlm=vlm,
                    ref_prompt=ref_prompt, matching_prompt=matching_prompt, font_path=font_path,
                    artifact_dir=frame_dir, scene_context=scene_context,
                ))
        metrics = _aggregate(frame_results, references, slots)
        result: dict[str, Any] = {
            "sample_id": record.get("sample_id"), "video_path": str(video_path),
            "sampling": {"timestamps": sorted(item["timestamp"] for item in frame_results), "config": sampling_cfg},
            "video": metadata,
            "references": [{"ref_id": item["ref_id"], "filename": item["filename"],
                            "image_path": item["image_path"], "carrier_id": item["carrier_id"],
                            "target_count": len(item["slots"])} for item in references],
            "metrics": metrics,
        }
        if include_frames:
            result["frames"] = frame_results
        if debug_path is not None:
            _write_json(debug_path / "input_record.json", record)
            _write_json(debug_path / "resolved_config.json", redact_credentials(config))
            _write_json(debug_path / "result.json", {**result, "frames": frame_results})
            _write_json(debug_path / "manifest.json", {
                "sample_id": record.get("sample_id"),
                "video_path": str(video_path.resolve()),
                "video": metadata,
                "scene_context": scene_context,
                "references": [{"ref_id": item["ref_id"], "filename": item["filename"]} for item in references],
                "initial_timestamps": timestamps,
                "final_timestamps": result["sampling"]["timestamps"],
                "metrics": result["metrics"],
                "files": {"input_record": "input_record.json", "resolved_config": "resolved_config.json",
                          "prompts": "prompts/", "frames": "frames/", "result": "result.json"},
            })
        return result
    finally:
        capture.release()


def run(
    input_path: str | Path, output_dir: str | Path, config: dict[str, Any], *,
    image_dir: str | Path | None = None, video_dir: str | Path | None = None,
    video_path: str | Path | None = None, video_manifest: str | Path | None = None,
    sample_id: str | None = None, project_root: Path | None = None,
    debug_dir: str | Path | None = None, vlm: Any | None = None, ocr_engine: Any | None = None,
    override: bool = False, sample_workers: int | None = None,
) -> dict[str, Any]:
    input_file = Path(input_path)
    if input_file.is_dir():
        input_file = input_file / "video_prompt_dataset.jsonl"
    if not input_file.is_file():
        raise FileNotFoundError(f"merged input JSONL not found: {input_file}")
    output_root = Path(output_dir)
    project_root = project_root or Path.cwd()
    runtime_cfg = config.get("runtime", {})
    sample_workers = int(
        sample_workers if sample_workers is not None else runtime_cfg.get("sample_workers", 1)
    )
    if sample_workers < 1:
        raise ValueError("sample_workers must be a positive integer")
    rows = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if sample_id:
        rows = [row for row in rows if str(row.get("sample_id")) == sample_id]
    if not rows:
        raise ValueError("no matching Ref2V samples")
    sample_ids = [str(row.get("sample_id", "record")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id in selected Ref2V input")
    image_root = Path(image_dir) if image_dir else input_file.parent / "imgs"
    video_map = _load_video_manifest(video_manifest) if video_manifest else None
    explicit_video = Path(video_path) if video_path else None
    if explicit_video is not None and len(rows) != 1:
        raise ValueError("--video requires exactly one selected sample")
    run_started_at = _now()
    run_started_clock = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=True)
    previous_status: dict[str, str] = {}
    summary_path = output_root / "summary.json"
    if summary_path.is_file() and not override:
        try:
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_summary = {}
        for item in previous_summary.get("results", []) + previous_summary.get("errors", []):
            if isinstance(item, dict) and item.get("sample_id") is not None:
                previous_status[str(item["sample_id"])] = str(item.get("status", ""))

    def existing_success(row: dict[str, Any]) -> dict[str, Any] | None:
        sid = str(row.get("sample_id", "record"))
        if previous_status.get(sid) == "error":
            return None
        result_path = output_root / _safe_id(sid) / "result.json"
        if not result_path.is_file():
            return None
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        metrics = value.get("metrics") if isinstance(value, dict) else None
        if not isinstance(metrics, dict):
            return None
        return {
            "sample_id": sid,
            "status": "ok",
            "result": str(result_path.relative_to(output_root)),
            "metrics": metrics,
        }

    existing = {} if override else {
        str(row.get("sample_id", "record")): existing_success(row)
        for row in rows
    }
    existing = {sid: value for sid, value in existing.items() if value is not None}
    pending_rows = [row for row in rows if str(row.get("sample_id", "record")) not in existing]
    results_by_id = dict(existing)
    errors_by_id: dict[str, dict[str, Any]] = {}
    run_samples_by_id: dict[str, dict[str, Any]] = {
        sid: {
            "sample_id": sid,
            "status": "skipped",
            "reason": "existing_success",
            "result": value["result"],
            "elapsed_sec": 0.0,
        }
        for sid, value in existing.items()
    }
    debug_root = Path(debug_dir) if debug_dir else None
    for row in pending_rows:
        # A stale result must not make a failed retry look successful.
        sid = str(row.get("sample_id", "record"))
        (output_root / _safe_id(sid) / "result.json").unlink(missing_ok=True)

    shared_ocr_engine = ocr_engine
    if pending_rows and shared_ocr_engine is None:
        # One shared PaddleOCR instance is protected by PaddleOCREngine's
        # internal lock. Constructing one per worker would duplicate models.
        shared_ocr_engine = PaddleOCREngine(config, project_root)

    def evaluate_one(row: dict[str, Any]) -> dict[str, Any]:
        sid = str(row.get("sample_id", "record"))
        sample_started_clock = time.perf_counter()
        resolved_video: Path | None = None
        detail: dict[str, Any] = {"sample_id": sid}
        try:
            resolved_video = _resolve_video(
                row,
                video=explicit_video,
                video_dir=Path(video_dir) if video_dir else None,
                video_manifest=video_map,
            )
            if not resolved_video.is_file():
                raise FileNotFoundError(str(resolved_video))
            sample_debug = debug_root / _safe_id(sid) if debug_root else None
            result = evaluate_record(
                row,
                resolved_video,
                config,
                image_dir=image_root,
                project_root=project_root,
                vlm=vlm,
                ocr_engine=shared_ocr_engine,
                debug_dir=sample_debug,
                include_frames=bool(debug_root),
                show_progress=sample_workers == 1,
            )
            detail.update({"status": "ok", "result_value": result})
        except (OCRError, VLMError):
            raise
        except Exception as exc:
            detail.update({"status": "error", "error": str(exc)})
        finally:
            if resolved_video is not None:
                detail["video_path"] = str(resolved_video)
            detail["elapsed_sec"] = round(time.perf_counter() - sample_started_clock, 6)
        return detail

    def consume(detail: dict[str, Any]) -> None:
        sid = str(detail["sample_id"])
        status = detail["status"]
        if status == "ok":
            result = detail.pop("result_value")
            sample_dir = output_root / _safe_id(sid)
            result_path = str((sample_dir / "result.json").relative_to(output_root))
            _write_json(sample_dir / "result.json", result)
            results_by_id[sid] = {
                "sample_id": sid,
                "status": "ok",
                "result": result_path,
                "metrics": result["metrics"],
            }
            detail["result"] = result_path
        else:
            errors_by_id[sid] = {
                "sample_id": sid,
                "status": "error",
                "error": detail["error"],
            }
        run_samples_by_id[sid] = detail

    if pending_rows:
        executor = ThreadPoolExecutor(max_workers=sample_workers, thread_name_prefix="ref2v-sample")
        futures = {executor.submit(evaluate_one, row): str(row.get("sample_id", "record")) for row in pending_rows}
        fatal_error: BaseException | None = None
        try:
            with tqdm(total=len(pending_rows), desc="Ref2V samples", unit="sample") as sample_progress:
                for future in as_completed(futures):
                    try:
                        consume(future.result())
                    except (OCRError, VLMError) as exc:
                        fatal_error = exc
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        break
                    sample_progress.update(1)
        finally:
            executor.shutdown(wait=True, cancel_futures=fatal_error is not None)
        if fatal_error is not None:
            raise fatal_error

    run_samples = [
        run_samples_by_id[sid]
        for sid in sample_ids
        if sid in run_samples_by_id
    ]
    updated_at = _now()
    ordered_results = [results_by_id[str(row.get("sample_id", "record"))]
                       for row in rows if str(row.get("sample_id", "record")) in results_by_id]
    ordered_errors = [errors_by_id[str(row.get("sample_id", "record"))]
                      for row in rows if str(row.get("sample_id", "record")) in errors_by_id]
    summary_metrics = [item["metrics"] for item in ordered_results if isinstance(item.get("metrics"), dict)]
    summary = {"total": len(rows), "ok": len(ordered_results), "error": len(ordered_errors),
               "results": ordered_results, "errors": ordered_errors, "updated_at": updated_at,
               "resume": {"override": bool(override), "skipped_success": len(existing), "processed": len(pending_rows)},
               "execution": {"sample_workers": sample_workers},
               "benchmark_metrics": aggregate_benchmark_metrics(
                   summary_metrics,
                   spatial_key="length_weighted_spatial_fidelity",
                   temporal_key="length_weighted_temporal_fidelity",
               )}
    run_details = {
        "schema_version": "1.0",
        "started_at": run_started_at,
        "finished_at": updated_at,
        "elapsed_sec": round(time.perf_counter() - run_started_clock, 6),
        "input": str(input_file),
        "output_dir": str(output_root),
        "image_dir": str(image_root),
        "video_dir": str(video_dir) if video_dir else None,
        "video_path": str(explicit_video) if explicit_video else None,
        "video_manifest": str(video_manifest) if video_manifest else None,
        "sample_id_filter": sample_id,
        "debug_dir": str(debug_root) if debug_root else None,
        "override": bool(override),
        "sample_workers": sample_workers,
        "skipped_success": len(existing),
        "samples": run_samples,
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "run_details.json", run_details)
    return summary
