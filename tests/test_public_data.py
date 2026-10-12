import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_t2v_release_contract_and_balance():
    rows = _jsonl(ROOT / "data/T2V/benchmark_300.jsonl")
    assert len(rows) == 300
    assert len({row["sample_id"] for row in rows}) == 300
    assert Counter(row["difficulty"] for row in rows) == {
        "easy": 100,
        "medium": 100,
        "hard": 100,
    }
    for row in rows:
        assert set(row) == {"sample_id", "difficulty", "carriers", "texts", "prompt_en", "prompt_zh"}
        slot_ids = {slot["slot_id"] for carrier in row["carriers"] for slot in carrier["slots"]}
        assert slot_ids == {text["slot_id"] for text in row["texts"]}


def test_ref2v_release_contract_and_images():
    root = ROOT / "data/Ref2V"
    rows = _jsonl(root / "video_prompt_dataset.jsonl")
    assert len(rows) == 200
    assert len({row["sample_id"] for row in rows}) == 200
    referenced = []
    for row in rows:
        assert set(row) == {"sample_id", "prompt", "reference_images"}
        assert row["sample_id"].isdigit() and len(row["sample_id"]) == 4
        assert row["reference_images"]
        assert len({item["ref_id"] for item in row["reference_images"]}) == len(row["reference_images"])
        for item in row["reference_images"]:
            assert set(item) == {"ref_id", "filename"}
            assert (root / "imgs" / item["filename"]).is_file()
            referenced.append(item["filename"])
    assert len(referenced) == 200
    assert set(referenced) == {path.name for path in (root / "imgs").iterdir() if path.is_file()}
