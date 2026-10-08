import hashlib

import numpy as np
import pytest
from conftest import make_photo
from PIL import Image

from gallery_editor.agents import color_agent
from gallery_editor.config import Settings
from gallery_editor.imaging import grade
from gallery_editor.pipeline import Pipeline
from gallery_editor.scanner import scan
from gallery_editor.schemas import AlbumContract, Analysis, Changes, Color, Crop, Override, Review
from gallery_editor.storage import file_hash, read_json
from gallery_editor.vision import statistics


def snapshot(root):
    return {
        str(p.relative_to(root)): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


def test_startup_preserves_contents(settings):
    for folder in ("input", "output", "cache", "reports"):
        p = settings.root / folder / "keep.txt"
        p.write_text("keep")
    settings.initialize()
    for folder in ("input", "output", "cache", "reports"):
        assert (settings.root / folder / "keep.txt").read_text() == "keep"


def test_custom_input_and_output_full_pipeline(tmp_path):
    settings = Settings(
        root=tmp_path / "workspace",
        input_dir=tmp_path / "my photos",
        output_dir=tmp_path / "deliverables",
        resolution_profile="custom",
        min_long_edge=400,
    )
    settings.initialize()
    make_photo(settings.input_dir / "nested" / "photo.jpg")
    report = Pipeline(settings).run(theme="A natural wedding album with dramatic contrast")
    assert report["total"] == 1
    assert (settings.output_dir / report["output_directory"] / "final" / "nested" / "photo.jpg").is_file()
    assert (settings.root / "reports" / "runs" / report["run_id"] / "gallery_report.json").is_file()
    album = read_json(settings.root / "reports" / "album_analysis.json")
    assert album["user_theme"] == "A natural wedding album with dramatic contrast"
    assert scan(settings)["input_directory"] == str(settings.input_dir)
    assert list(settings.input_dir.iterdir()) == [settings.input_dir / "nested"]


@pytest.mark.parametrize("suffix", ["JPG", "JPEG", "PNG", "WEBP", "TIFF", "tif"])
def test_formats(settings, suffix):
    make_photo(settings.root / "input" / f"photo.{suffix}")
    assert scan(settings)["count"] == 1


def test_scanner_reports_preview_progress(settings):
    make_photo(settings.input_dir / "nested" / "one.jpg")
    events = []
    scan(settings, progress=lambda *event: events.append(event))
    assert events == [
        ("reading", "nested/one.jpg", 1, 1),
        ("ready", "nested/one.jpg", 1, 1),
    ]


def test_scan_ignores_hidden_unsupported_generated_symlinks(gallery):
    root = gallery.root
    make_photo(root / "input" / ".hidden" / "a.jpg")
    make_photo(root / "input" / "output" / "a.jpg")
    (root / "input" / "note.txt").write_text("hi")
    (root / "input" / "alias.jpg").symlink_to(root / "input" / "a.jpg")
    (root / "input" / "broken.jpg").write_text("not a photograph")
    result = scan(gallery)
    assert result["count"] == 3
    assert len(result["errors"]) == 1
    assert result["photos"][0]["width"] == 640


def test_full_pipeline_originals_reports_cache_manual(gallery):
    root = gallery.root
    originals = snapshot(root / "input")
    pipeline = Pipeline(gallery)
    report = pipeline.run()
    assert report["total"] == 3
    assert report["crop"]["exported_below_floor"] == 0
    profile = read_json(root / "reports" / "performance.json")
    assert profile["status"] == "complete"
    assert profile["elapsed_seconds"] > 0
    assert profile["stages"]
    assert profile["photos"] and profile["photos"][0]["seconds"] > 0
    assert profile["model_calls"]
    assert {event["role"] for event in profile["model_calls"]}
    assert report["variants"]["combinations"] == sum(len(photo["variants"]) for photo in report["photos"])
    assert report["status_counts"]["manual_review_required"] == 3
    assert (root / "reports" / "gallery_report.html").exists()
    assert (root / "cache" / "contact-sheet-1.jpg").exists()
    assert snapshot(root / "input") == originals
    for r in report["photos"]:
        assert (root / "output" / r["output"]).exists()
        assert "300 PPI" in r["summary"]
        assert r["review"]["source"] == "deterministic"
        with Image.open(root / "output" / r["output"]) as image:
            assert max(image.size) >= gallery.min_long_edge
        assert r["output"].startswith(f"{report['output_directory']}/final/")
        assert r["variants"]
        for variant in r["variants"]:
            assert variant["output_quality"] == 100
            assert (root / "output" / variant["output"]).is_file()
            assert variant["output"].startswith(f"{report['output_directory']}/options/")
        assert (root / "reports" / "photos" / f"{r['id']}.json").exists()
    photo = report["photos"][0]
    first_run_output = report["photos"][0]["output"]
    edited = pipeline.override(photo["id"], Override(crop=Crop(), color=Color(contrast=15, exposure=0.1)))
    assert edited["manual"] and edited["color"]["contrast"] == 15
    rerun = pipeline.run()
    assert rerun["run_id"] != report["run_id"]
    assert (root / "output" / first_run_output).is_file()
    edited = next(p for p in rerun["photos"] if p["id"] == photo["id"])
    assert edited["manual"] and edited["color"]["contrast"] == 15
    reset = pipeline.override(photo["id"], reset=True)
    assert not reset["manual"]
    assert len(read_json(root / "reports" / "feedback.json")) == 2
    assert snapshot(root / "input") == originals


def test_select_variant_exports_full_resolution_and_persists_choice(gallery):
    root = gallery.root
    originals = snapshot(root / "input")
    pipeline = Pipeline(gallery)
    first = pipeline.run()["photos"][0]
    variant = first["variants"][0]
    selected = pipeline.select_variant(first["id"], variant["id"])
    assert selected["manual"]
    assert selected["selected_variant"] == variant["id"]
    assert selected["crop"] == variant["crop"]
    assert selected["color"] == variant["color"]
    with Image.open(root / "output" / selected["output"]) as image:
        assert list(image.size) == [
            variant["crop_metrics"]["result_width"],
            variant["crop_metrics"]["result_height"],
        ]
        assert max(image.size) >= gallery.min_long_edge
    profile = read_json(root / "reports" / "performance.json")
    assert profile["operation"] == "variant_selection"
    assert any(operation["name"] == "full-resolution render" for operation in profile["operations"])
    assert read_json(root / "reports" / "feedback.json")[-1]["action"] == "select_variant"
    assert snapshot(root / "input") == originals
    rerun = pipeline.run()
    persisted = next(item for item in rerun["photos"] if item["id"] == first["id"])
    assert persisted["selected_variant"] == variant["id"]
    assert persisted["crop"] == variant["crop"]


def test_small_original_blocked(settings):
    make_photo(settings.root / "input" / "small.jpg", size=(200, 150))
    report = Pipeline(settings).run()
    assert report["photos"][0]["status"] == "blocked"
    assert not list((settings.root / "output").rglob("*.jpg"))


def test_analysis_cached(gallery):
    pipeline = Pipeline(gallery)
    pipeline.run()
    paths = list((gallery.root / "cache" / "analysis").glob("cv-v4-[0-9]*.json"))
    times = {p: p.stat().st_mtime_ns for p in paths}
    pipeline.run()
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in times.items())


def test_manual_bad_crop_never_exported(gallery):
    pipeline = Pipeline(gallery)
    record = pipeline.run()["photos"][0]
    before = file_hash(gallery.root / "output" / record["output"])
    with pytest.raises(ValueError, match="Resolution floor"):
        pipeline.override(record["id"], Override(crop=Crop(width=0.1, height=0.1), color=Color()))
    assert file_hash(gallery.root / "output" / record["output"]) == before


class RevisionModel:
    """Scripted test double, never used by the application or demos."""

    def __init__(self):
        self.reviews = 0

    def decide(self, role, schema, context, images=()):
        if role == "photo analyst":
            return Analysis(confidence=0.9)
        if role == "reviewer":
            self.reviews += 1
            return Review(
                score=50,
                crop_score=50,
                color_score=50,
                consistency_score=50,
                approved=False,
                needs_revision=True,
                issues=["Test rejection"],
                confidence=0.9,
                suggested_changes=Changes(crop=Crop(width=0.01, height=0.01)),
            )
        return None


def test_revision_limit_and_unsafe_reviewer(settings):
    make_photo(settings.root / "input" / "photo.jpg")
    model = RevisionModel()
    record = Pipeline(settings, model).run()["photos"][0]
    assert model.reviews == 3
    assert record["revision_cycles"] == 2
    assert record["status"] == "manual_review_required"
    assert record["crop"] == Crop().model_dump()


def test_consistency_correction_reduces_exposure_spread(settings):
    class NoModel:
        def decide(self, *args, **kwargs):
            return None

    image = Image.new("RGB", (640, 480), (60, 60, 60))
    before = statistics(image)
    gallery = {"exposure": {"median": 0.4}, "temperature": {"median": 0}, "saturation": {"median": 0}}
    result = color_agent(
        NoModel(), {"preview": "ignored"}, settings.root, {"statistics": before}, gallery, AlbumContract()
    )
    after = statistics(grade(image, result.color))
    assert abs(after["exposure"] - 0.4) < abs(before["exposure"] - 0.4)


def test_per_photo_artistic_color_looks_are_deterministic():
    source = Image.new("RGB", (8, 8), (150, 90, 50))
    monochrome = grade(source, Color(look="monochrome"))
    warm = grade(source, Color(look="warm_monochrome"))
    cinematic = grade(source, Color(look="cinematic"))
    gray = monochrome.getpixel((0, 0))
    assert gray[0] == gray[1] == gray[2]
    assert len(set(warm.getpixel((0, 0)))) > 1
    assert cinematic.getpixel((0, 0)) != grade(source, Color()).getpixel((0, 0))
    assert "look" not in AlbumContract.model_fields


def test_output_collision_preserved(settings):
    make_photo(settings.root / "input" / "photo.jpg")
    path = make_photo(settings.root / "output" / "edited" / "photo.jpg", brightness=0.5)
    before = file_hash(path)
    report = Pipeline(settings).run()
    assert report["photos"][0]["status"] != "blocked"
    assert file_hash(path) == before
    assert (settings.root / "output" / report["photos"][0]["output"]).exists()


def test_nested_output_symlink_cannot_write_input(settings):
    photo = make_photo(settings.root / "input" / "photo.jpg")
    original = file_hash(photo)
    (settings.root / "output" / "edited").symlink_to(settings.root / "input", target_is_directory=True)
    report = Pipeline(settings).run()
    assert report["photos"][0]["status"] != "blocked"
    assert file_hash(photo) == original
    assert (settings.root / "output" / report["photos"][0]["output"]).exists()


def test_render_is_deterministic():
    image = Image.new("RGB", (100, 100), (90, 130, 150))
    color = Color(contrast=8, grain=3)
    assert (
        hashlib.sha256(grade(image, color).tobytes()).digest()
        == hashlib.sha256(grade(image, color).tobytes()).digest()
    )


def test_empty_gallery_fails_before_creating_run_output(settings):
    from gallery_editor.pipeline import NoReadablePhotosError

    with pytest.raises(NoReadablePhotosError, match="No supported photographs found"):
        Pipeline(settings).run()
    assert not (settings.output_dir / "runs").exists()


def test_raw_only_gallery_without_decoder_fails_before_creating_run_output(settings, monkeypatch):
    import gallery_editor.scanner as scanner
    from gallery_editor.pipeline import NoReadablePhotosError

    (settings.input_dir / "sample.RAF").write_bytes(b"not an actual RAW fixture")
    monkeypatch.setattr(scanner, "raw_available", lambda: False)
    with pytest.raises(NoReadablePhotosError, match="RAW decoder unavailable"):
        Pipeline(settings).run()
    assert not (settings.output_dir / "runs").exists()


def test_color_identity():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(image), np.asarray(grade(image, Color())))


def test_exif_orientation_and_source_metadata_preserved(settings):
    path = settings.root / "input" / "oriented.jpg"
    image = Image.new("RGB", (640, 480), (50, 70, 90))
    exif = Image.Exif()
    exif[274] = 6
    exif[271] = "Test camera"
    image.save(path, exif=exif)
    before = file_hash(path)
    photo = scan(settings)["photos"][0]
    assert (photo["width"], photo["height"]) == (480, 640)
    assert photo["metadata"]["Make"] == "Test camera"
    assert file_hash(path) == before


def test_missing_raw_decoder_reports_skipped_file(settings, monkeypatch):
    import gallery_editor.scanner as scanner

    (settings.root / "input" / "sample.RAF").write_bytes(b"not an actual RAW fixture")
    monkeypatch.setattr(scanner, "raw_available", lambda: False)
    result = scanner.scan(settings)
    assert result["count"] == 0
    assert "RAW decoder unavailable" in result["errors"][0]["error"]


@pytest.mark.parametrize("change", ["color_only", "soften", "crop_only"])
def test_raw_revisions_preserve_independent_decisions_and_render_actual_color(settings, monkeypatch, change):
    from gallery_editor import pipeline as pipeline_module
    from gallery_editor import variants as variants_module
    from gallery_editor.imaging import render, save_image
    from gallery_editor.schemas import CropDecision
    from gallery_editor.vision import gallery_statistics

    source_image = Image.new("RGB", (640, 480), (100, 120, 150))
    source = settings.input_dir / "example.RAF"
    source.write_bytes(b"explicit RAW test fixture; decoder replaced below")
    save_image(source_image, settings.root / "cache" / "previews" / "source.jpg")
    photo = {
        "id": "raw-test",
        "hash": file_hash(source),
        "file": source.name,
        "width": 640,
        "height": 480,
        "preview": "previews/source.jpg",
    }
    monkeypatch.setattr(pipeline_module, "load_cached_source", lambda *args: source_image.copy())
    monkeypatch.setattr(variants_module, "load_cached_source", lambda *args: source_image.copy())
    proposed = Crop(x=0.04, y=0.04, width=0.92, height=0.92)
    initial = Color(contrast=4)
    revised_crop = Crop(x=0.02, y=0.02, width=0.96, height=0.96)
    changes = {
        "color_only": Changes(color=Color(look="monochrome", contrast=4)),
        "soften": Changes(),
        "crop_only": Changes(crop=revised_crop),
    }[change]

    class Model:
        reviews = 0

        def decide(self, role, schema, context, images=()):
            if role == "crop editor":
                return CropDecision(crop=proposed, confidence=0.95, rationale="Test edge crop")
            if role == "reviewer":
                self.reviews += 1
                return Review(
                    score=90,
                    crop_score=90,
                    color_score=90,
                    consistency_score=90,
                    confidence=0.95,
                    approved=self.reviews > 1,
                    needs_revision=self.reviews == 1,
                    issues=["Test revision"] if self.reviews == 1 else [],
                    suggested_changes=changes if self.reviews == 1 else Changes(),
                )
            return None

    cv = {
        "statistics": statistics(source_image),
        "protected_regions": [],
        "edge_centroid": [0.5, 0.5],
        "face_region_rgb": [],
        "limitations": [],
    }
    record = Pipeline(settings, Model()).process_photo(
        photo,
        cv,
        gallery_statistics([cv]),
        AlbumContract(),
        [],
        analysis_override=Analysis(confidence=0.95).model_dump(),
        color_override=initial,
        color_confidence=0.95,
    )
    expected_crop = revised_crop if change == "crop_only" else proposed
    expected_color = changes.color or (Color(contrast=2) if change == "soften" else initial)
    assert record["crop"] == expected_crop.model_dump()
    assert record["color"] == expected_color.model_dump()
    assert record["revision_cycles"] == 1
    expected = np.asarray(render(source_image, expected_crop, expected_color, settings), dtype=float)
    with Image.open(settings.output_dir / record["output"]) as actual:
        assert np.abs(np.asarray(actual, dtype=float) - expected).mean() < 2
    for variant in record["variants"]:
        assert variant["full_resolution_exported"]
        expected = np.asarray(
            render(source_image, Crop(**variant["crop"]), Color(**variant["color"]), settings), dtype=float
        )
        with Image.open(settings.output_dir / variant["output"]) as actual:
            assert np.abs(np.asarray(actual, dtype=float) - expected).mean() < 2


def test_failed_final_edit_keeps_color_exports_in_report(settings, monkeypatch):
    from gallery_editor.pipeline import Pipeline

    make_photo(settings.input_dir / "photo.jpg")
    pipeline = Pipeline(settings)
    monkeypatch.setattr(
        pipeline,
        "process_photo",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Test crop/export failure")),
    )
    report = pipeline.run()
    assert report["status_counts"] == {"blocked": 1}
    assert len(report["photos"][0]["color_option_exports"]) == 3
    assert "3 uncropped grade JPEGs remain" in report["photos"][0]["summary"]
    assert all(
        (settings.output_dir / report["output_directory"] / option["path"]).is_file()
        for option in report["photos"][0]["color_option_exports"]
    )
    assert (settings.root / "reports" / "runs" / report["run_id"] / "performance.json").is_file()


def test_raw_decode_cache_reuses_pixels_and_invalidates_on_source_hash(settings, monkeypatch):
    from gallery_editor import imaging

    original_loader = imaging.load_image
    calls = []

    def decode(path):
        if path.suffix.lower() == ".raf":
            calls.append(path)
            return Image.new("RGB", (640, 480), (80, 100, 130))
        return original_loader(path)

    monkeypatch.setattr(imaging, "load_image", decode)
    source = settings.input_dir / "fixture.RAF"
    first = imaging.load_cached_source(source, settings.root, "source-version-1")
    second = imaging.load_cached_source(source, settings.root, "source-version-1")
    assert first.tobytes() == second.tobytes()
    assert len(calls) == 1
    imaging.load_cached_source(source, settings.root, "source-version-2")
    assert len(calls) == 2


def test_failed_semantic_analysis_is_retried_when_model_is_configured(settings, monkeypatch):
    from gallery_editor import pipeline as module
    from gallery_editor.schemas import ColorDecision

    settings.vision_model = "test-vision"
    make_photo(settings.input_dir / "test.jpg")
    photo = scan(settings)["photos"][0]
    calls = []

    def analyze(*args):
        calls.append(True)
        return Analysis(confidence=0 if len(calls) == 1 else 0.9)

    monkeypatch.setattr(module, "photo_agent", analyze)
    monkeypatch.setattr(
        module, "color_agent", lambda *args: ColorDecision(color=Color(), confidence=0.9, rationale="Test")
    )

    def fake_exports(output_dir, *_args):
        exports = []
        for index in range(3):
            path = settings.output_dir / f"test-option-{index}.jpg"
            path.write_bytes(b"test JPEG placeholder")
            exports.append(
                {
                    "id": str(index),
                    "path": path.relative_to(settings.output_dir).as_posix(),
                    "output_hash": file_hash(path),
                }
            )
        return exports

    monkeypatch.setattr(module, "export_color_options", fake_exports)
    pipeline = Pipeline(settings)
    args = (photo, {}, {}, AlbumContract(), [], {})
    assert pipeline.prepare_color_options(*args)["analysis"]["confidence"] == 0
    assert pipeline.prepare_color_options(*args)["analysis"]["confidence"] == 0.9
    pipeline.prepare_color_options(*args)
    assert len(calls) == 2
    settings.model_context_tokens = 65536
    pipeline.prepare_color_options(*args)
    assert len(calls) == 3
