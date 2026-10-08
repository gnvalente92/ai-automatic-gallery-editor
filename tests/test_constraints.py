import random

import pytest
from PIL import Image
from pydantic import ValidationError

from gallery_editor.config import Settings
from gallery_editor.cropping import choose_crop
from gallery_editor.imaging import crop_metrics, enforce_resolution, render
from gallery_editor.schemas import AlbumContract, Analysis, Color, Crop, CropDecision, ProtectedRegion, Review


@pytest.mark.parametrize(
    "data",
    [
        {"x": -0.1},
        {"width": 0},
        {"x": 0.4, "width": 0.7},
        {"y": 0.9, "height": 0.2},
        {"height": -1},
        {"x": float("nan")},
    ],
)
def test_invalid_crop(data):
    with pytest.raises(ValidationError):
        Crop(**data)


@pytest.mark.parametrize(
    "data",
    [
        {"exposure": 2},
        {"contrast": -26},
        {"temperature": 5000},
        {"grain": -1},
        {"saturation": float("inf")},
        {"unknown": 3},
    ],
)
def test_color_validation(data):
    with pytest.raises(ValidationError):
        Color(**data)


def test_signed_color_adjustments_valid():
    assert Color(exposure=-0.4, temperature=-300, shadows=-10).exposure == -0.4


def test_metrics():
    m = crop_metrics(Crop(), (6000, 4000))
    assert m["megapixels"] == 24
    assert m["print_sizes"]["300"]["width_inches"] == 20
    assert m["print_sizes"]["240"]["width_inches"] == 25
    assert m["print_sizes"]["200"]["width_inches"] == 30
    assert m["crop_percentage"] == 0


def test_resolution_examples():
    s = Settings(min_long_edge=4000)
    for width, accepted in ((5800, True), (5200, True), (4200, True), (3000, False)):
        crop = Crop(width=width / 6240, height=width / 6240)
        if accepted:
            enforce_resolution(crop, (6240, 4160), s)
        else:
            with pytest.raises(ValueError, match="Resolution floor"):
                enforce_resolution(crop, (6240, 4160), s)


def test_exact_pixel_floor_never_bypassed():
    rng = random.Random(7)
    s = Settings(min_long_edge=4000, min_megapixels=8)
    for _ in range(2000):
        x, y = rng.random() * 0.3, rng.random() * 0.3
        c = Crop(x=x, y=y, width=rng.uniform(0.1, 1 - x), height=rng.uniform(0.1, 1 - y))
        m = crop_metrics(c, (6240, 4160))
        legal = max(m["result_width"], m["result_height"]) >= 4000 and m["megapixels"] >= 8
        if legal:
            enforce_resolution(c, (6240, 4160), s)
        else:
            with pytest.raises(ValueError):
                enforce_resolution(c, (6240, 4160), s)


def test_renderer_enforces_floor(settings):
    with pytest.raises(ValueError):
        render(Image.new("RGB", (640, 480)), Crop(width=0.2, height=0.2), Color(), settings)


def test_custom_dimensions_and_megapixels():
    for s in (Settings(min_width=5000), Settings(min_height=5000), Settings(min_megapixels=25)):
        with pytest.raises(ValueError):
            enforce_resolution(Crop(), (4000, 3000), s)


def test_ai_crop_rejected_before_render(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    proposal = CropDecision(crop=Crop(width=0.2, height=0.2), confidence=1, rationale="test only")
    crop, candidates = choose_crop(
        (640, 480), settings, cv, Analysis(confidence=0.9), AlbumContract(), proposal
    )
    assert max(crop_metrics(crop, (640, 480))[k] for k in ("result_width", "result_height")) >= 400
    assert any(not c["accepted"] and "Resolution floor" in c["rejection"] for c in candidates)


def test_confident_semantic_crop_is_not_overruled_by_centroid_proxy(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    proposal = CropDecision(
        crop=Crop(x=0.04, y=0.04, width=0.92, height=0.92),
        confidence=0.9,
        rationale="A modest crop removes an irrelevant edge intrusion while retaining context",
    )

    crop, _ = choose_crop((6000, 6000), settings, cv, Analysis(confidence=0.9), AlbumContract(), proposal)

    assert crop == proposal.crop


def test_protected_subjects(settings):
    cv = {
        "protected_regions": [Crop(x=0, y=0, width=0.2, height=0.2).model_dump()],
        "edge_centroid": [0.8, 0.5],
    }
    proposal = CropDecision(crop=Crop(x=0.1, width=0.9), confidence=1, rationale="test only")
    _, candidates = choose_crop((640, 480), settings, cv, Analysis(confidence=0.9), AlbumContract(), proposal)
    assert any("protected" in (c["rejection"] or "") for c in candidates)


def test_contextual_region_guides_but_does_not_block_crop(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    analysis = Analysis(
        confidence=0.9,
        protected_regions=[ProtectedRegion(x=0, y=0, width=0.27, height=1, role="contextual")],
    )
    _, candidates = choose_crop((640, 480), settings, cv, analysis, AlbumContract())
    assert any(candidate["accepted"] and candidate["crop"]["width"] < 1 for candidate in candidates)


def test_broad_semantic_region_cannot_become_an_accidental_hard_lock(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    analysis = Analysis(
        confidence=0.9,
        protected_regions=[ProtectedRegion(x=0.13, y=0.2, width=0.56, height=0.62, role="primary")],
    )
    _, candidates = choose_crop((640, 480), settings, cv, analysis, AlbumContract())
    assert any(candidate["accepted"] and candidate["crop"]["width"] < 1 for candidate in candidates)


def test_primary_subject_region_remains_a_hard_crop_guard(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    analysis = Analysis(
        confidence=0.9,
        protected_regions=[ProtectedRegion(x=0.01, y=0.2, width=0.12, height=0.3, role="primary")],
    )
    _, candidates = choose_crop((640, 480), settings, cv, analysis, AlbumContract())
    assert any("protected" in (candidate["rejection"] or "") for candidate in candidates)


def test_no_semantic_analysis_retains_frame(settings):
    cv = {"protected_regions": [], "edge_centroid": [0.1, 0.8]}
    crop, _ = choose_crop((640, 480), settings, cv, Analysis(), AlbumContract(preferred_aspect_ratio="1:1"))
    assert crop == Crop()


def test_json_schemas_reject_malformed_and_incoherent():
    with pytest.raises(ValidationError):
        Crop.model_validate_json('{"x":')
    with pytest.raises(ValidationError):
        Review(
            score=90,
            crop_score=90,
            color_score=90,
            consistency_score=90,
            approved=True,
            needs_revision=True,
            issues=[],
            confidence=0.9,
        )
    with pytest.raises(ValidationError):
        AlbumContract(preferred_aspect_ratio="0:0")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.com/v1",
        "http://192.168.1.2/v1",
        "http://127.0.0.1.evil.com/v1",
        "file:///tmp/model",
        "http://user:secret@localhost/v1",
    ],
)
def test_remote_models_rejected(endpoint):
    with pytest.raises(ValidationError):
        Settings(local_model_endpoint=endpoint)


def test_large_primary_subject_is_not_silently_excluded_from_protection(settings):
    analysis = Analysis(
        confidence=0.95,
        protected_regions=[ProtectedRegion(x=0, y=0, width=0.8, height=1, role="primary", confidence=0.95)],
    )
    proposed = CropDecision(
        crop=Crop(x=0.04, y=0.04, width=0.92, height=0.92), confidence=0.95, rationale="Must be rejected"
    )
    crop, candidates = choose_crop(
        (6000, 4000),
        settings,
        {"protected_regions": [], "edge_centroid": [0.5, 0.5]},
        analysis,
        AlbumContract(),
        proposed,
    )
    assert crop == Crop()
    assert any(c["rejection"] and "protected" in c["rejection"] for c in candidates)
    assert sum(c["selected"] for c in candidates) == 1


def test_subpixel_model_crop_is_rejected_without_blocking_valid_candidates(settings):
    proposal = CropDecision(crop=Crop(width=0.000001), confidence=0.99, rationale="Invalid tiny crop")
    crop, candidates = choose_crop(
        (640, 480),
        settings,
        {"protected_regions": [], "edge_centroid": [0.5, 0.5]},
        Analysis(confidence=0.95),
        AlbumContract(),
        proposal,
    )
    enforce_resolution(crop, (640, 480), settings)
    assert any(c["rejection"] == "Crop rounds to an empty image" for c in candidates)


@pytest.mark.parametrize(
    "orientation,order",
    [
        (1, [0, 1, 2, 3]),
        (2, [1, 0, 3, 2]),
        (3, [3, 2, 1, 0]),
        (4, [2, 3, 0, 1]),
        (5, [0, 2, 1, 3]),
        (6, [2, 0, 3, 1]),
        (7, [3, 1, 2, 0]),
        (8, [1, 3, 0, 2]),
    ],
)
def test_all_exif_orientations_use_display_coordinates(tmp_path, orientation, order):
    from gallery_editor.imaging import load_image

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    image = Image.new("RGB", (20, 10))
    for color, box in zip(colors, [(0, 0, 10, 5), (10, 0, 20, 5), (0, 5, 10, 10), (10, 5, 20, 10)]):
        image.paste(color, box)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    displayed = load_image(path)
    assert displayed.size == ((10, 20) if orientation >= 5 else (20, 10))
    corners = [
        (0, 0),
        (displayed.width - 1, 0),
        (0, displayed.height - 1),
        (displayed.width - 1, displayed.height - 1),
    ]
    assert [displayed.getpixel(p) for p in corners] == [colors[i] for i in order]
