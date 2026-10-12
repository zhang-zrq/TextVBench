import math

from textvbench.evaluators.ref2v.core import (
    Candidate,
    Slot,
    extra_text_ratio,
    filter_candidates,
    harmonic_textv_score,
    normalized_edit_similarity,
    reading_order_score,
    target_length_weighted_score,
)
from textvbench.evaluators.ref2v.marker import render_reference_label, resize_longest_edge


def test_text_normalization_and_similarity_are_script_agnostic():
    assert normalized_edit_similarity(" A，b! ", "Ab") == 1.0
    assert normalized_edit_similarity("ABC", "abc") == 0.0
    assert normalized_edit_similarity("你好", "你好") == 1.0


def test_candidate_filtering_keeps_repeated_required_texts():
    active, duplicate, capped = filter_candidates(
        [Candidate("P1", "SALE", 0.99), Candidate("P2", "SALE", 0.95), Candidate("P3", "noise", 0.4)],
        confidence_threshold=0.8,
        required_texts=["SALE", "SALE"],
    )
    assert [item.id for item in active] == ["P1", "P2"]
    assert duplicate == []
    assert capped == []


def test_missing_slots_are_appended_for_pairwise_reading_order():
    # Reference [1,2,3,4,5], predicted [2,1,4,5], effective prediction
    # is [2,1,4,5,3], yielding 7 correct pairs out of 10.
    assert math.isclose(reading_order_score(["1", "2", "3", "4", "5"], ["2", "1", "4", "5"]), 0.7)


def test_length_weighting_and_textv_score():
    slots = [Slot("T1", "AA", "R01", "C_R01", 1), Slot("T2", "BBBB", "R01", "C_R01", 2)]
    assert math.isclose(target_length_weighted_score({"T1": 1.0, "T2": 0.5}, slots), 4 / 6)
    assert math.isclose(extra_text_ratio(["XX"], ["YYYY"]), 2 / 6)
    assert math.isclose(harmonic_textv_score(0.5, 1.0), 100 * 2 / 3)


def test_reference_label_preserves_source_pixels():
    from PIL import Image

    source = Image.new("RGB", (12, 8), (17, 23, 31))
    labeled = render_reference_label(source, "R01")
    assert labeled.size[0] == source.size[0]
    assert labeled.size[1] > source.size[1]
    source_top = labeled.height - source.height
    assert labeled.crop((0, source_top, source.width, labeled.height)).getpixel((0, 0)) == (17, 23, 31)
    assert any(
        labeled.getpixel((x, y)) != (255, 255, 255)
        for y in range(source_top)
        for x in range(labeled.width)
    )


def test_resize_longest_edge_is_proportional_and_does_not_upscale():
    from PIL import Image

    assert resize_longest_edge(Image.new("RGB", (1000, 500)), 256).size == (256, 128)
    assert resize_longest_edge(Image.new("RGB", (100, 50)), 256).size == (100, 50)
