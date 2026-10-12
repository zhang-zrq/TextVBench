from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from textvbench.evaluators.ref2v.ocr import PaddleOCREngine


class _FakePaddleOCR:
    def predict(self, image):
        assert isinstance(image, str)
        return iter([{
            "rec_texts": ["OPEN"],
            "rec_scores": [0.99],
            "rec_polys": [[(1, 2), (3, 2), (3, 4), (1, 4)]],
        }])


def test_paddle_adapter_converts_path_to_string():
    engine = PaddleOCREngine({}, Path.cwd())
    engine._ocr = _FakePaddleOCR()
    result = engine.predict(Path("reference.png"))
    assert result[0].text == "OPEN"
    assert result[0].bbox == ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))


def test_paddle_adapter_serializes_shared_instance_prediction():
    active = {"value": 0, "max": 0}
    lock = threading.Lock()

    class BlockingOCR:
        def predict(self, image):
            with lock:
                active["value"] += 1
                active["max"] = max(active["max"], active["value"])
            time.sleep(0.02)
            with lock:
                active["value"] -= 1
            return iter([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])

    engine = PaddleOCREngine({}, Path.cwd())
    engine._ocr = BlockingOCR()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(engine.predict, ["a", "b", "c", "d"]))
    assert active["max"] == 1
