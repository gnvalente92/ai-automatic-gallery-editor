import cv2
from PIL import Image

from gallery_editor.vision import analyze_cv


def test_close_up_face_is_not_discarded_only_because_it_is_large(tmp_path, monkeypatch):
    class Detector:
        def detectMultiScale(self, image, **kwargs):
            return [(10, 10, 40, 30)]

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda _path: Detector())
    source = tmp_path / "preview.jpg"
    Image.new("RGB", (100, 100), (80, 90, 100)).save(source)

    result = analyze_cv(source, tmp_path / "cv.json")

    assert result["face_count"] == 1
    assert result["protected_regions"][0]["width"] >= 0.4


def test_monochrome_does_not_trigger_color_matching_outliers():
    from gallery_editor.vision import outliers

    reference = {"saturation": {"median": 0.8, "mad": 0}, "temperature": {"median": 0.3, "mad": 0}}
    stats = {"saturation": 0, "temperature": 0}
    assert outliers(stats, reference)
    assert outliers(stats, reference, look="monochrome") == []
