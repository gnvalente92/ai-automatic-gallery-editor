from PIL import Image

from gallery_editor.config import Settings
from gallery_editor.schemas import Analysis, Color, Crop
from gallery_editor.variants import (
    color_options,
    crop_options,
    export_color_options,
    export_full_resolution_variants,
    generate_variants,
)


def test_landscape_variant_set_includes_safe_portrait_and_landscape():
    settings = Settings(resolution_profile="custom", min_long_edge=4000)
    face = Crop(x=0.44, y=0.30, width=0.12, height=0.18)
    cv = {"protected_regions": [face.model_dump()], "edge_centroid": [0.5, 0.4]}
    crops, rejected = crop_options((6032, 4032), cv, Analysis(confidence=0.9), settings, Crop())
    kinds = {crop["id"] for crop in crops}
    # The full-frame recommendation duplicates the always-present original option.
    assert {"portrait", "landscape", "tighter", "original"} <= kinds
    assert not rejected
    assert all(crop["metrics"]["result_width"] >= 0 for crop in crops)
    assert all(
        max(crop["metrics"]["result_width"], crop["metrics"]["result_height"]) >= 4000 for crop in crops
    )


def test_landscape_variant_may_be_withheld_to_preserve_large_subject():
    settings = Settings(resolution_profile="custom", min_long_edge=4000)
    person = Crop(x=0.14, y=0.20, width=0.56, height=0.62)
    cv = {"protected_regions": [person.model_dump()], "edge_centroid": [0.42, 0.51]}
    crops, rejected = crop_options((4032, 6032), cv, Analysis(confidence=0.9), settings, Crop())
    assert "portrait" in {crop["id"] for crop in crops}
    landscape = next(crop for crop in rejected if crop["id"] == "landscape")
    assert "protected subject" in landscape["reason"]


def test_color_alternatives_are_bounded_and_distinct():
    options = color_options(Color(exposure=0.1, contrast=2, temperature=80, grain=0))
    assert [key for key, _, _ in options] == ["album", "soft", "crisp"]
    assert len({tuple(color.model_dump().values()) for _, _, color in options}) == 3
    assert all(color.grain == 0 for _, _, color in options)


def test_creative_ai_look_has_natural_color_alternative():
    options = color_options(Color(look="monochrome", contrast=3))
    assert [key for key, _, _ in options] == ["album", "natural", "soft"]
    assert options[0][2].look == "monochrome"
    assert options[1][2].look == "natural"


def test_raw_color_options_export_three_full_resolution_jpegs(tmp_path):
    settings = Settings(root=tmp_path, resolution_profile="custom", min_long_edge=400)
    settings.initialize()
    output_dir = tmp_path / "output" / "runs" / "run-test"
    output_dir.mkdir(parents=True)
    original = Image.new("RGB", (640, 480), (90, 120, 150))

    exports = export_color_options(
        output_dir,
        {"file": "nested/example.RAF"},
        original,
        settings,
        Color(exposure=0.1),
    )

    assert [item["id"] for item in exports] == ["album", "soft", "crisp"]
    assert all(item["width"] == 640 and item["height"] == 480 for item in exports)
    for item in exports:
        path = output_dir / item["path"]
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.size == (640, 480)


def test_variant_previews_pair_crop_and_color_and_keep_pixel_metrics(tmp_path):
    settings = Settings(root=tmp_path, resolution_profile="custom", min_long_edge=400)
    settings.initialize()
    original = Image.new("RGB", (640, 480), (90, 120, 150))
    cv = {"protected_regions": [], "edge_centroid": [0.5, 0.5]}
    photo = {"id": "photo", "hash": "source-hash"}
    variants, rejected = generate_variants(
        tmp_path,
        photo,
        original,
        cv,
        Analysis(confidence=0.8),
        settings,
        Crop(),
        Color(),
    )
    assert not rejected
    assert len(variants) == 12
    assert {item["crop_kind"] for item in variants} == {"original", "tighter", "portrait", "landscape"}
    assert {item["color_kind"] for item in variants} == {"album", "soft", "crisp"}
    for variant in variants:
        path = tmp_path / "cache" / variant["preview"]
        assert path.exists()
        assert variant["crop_metrics"]["result_width"] >= 0
        assert max(variant["crop_metrics"]["result_width"], variant["crop_metrics"]["result_height"]) >= 400


def test_full_resolution_variants_are_written_under_the_run_folder(tmp_path):
    settings = Settings(root=tmp_path, resolution_profile="custom", min_long_edge=400)
    settings.initialize()
    output_dir = tmp_path / "output" / "runs" / "run-test"
    output_dir.mkdir(parents=True)
    original = Image.new("RGB", (640, 480), (90, 120, 150))
    crops = [
        {
            "id": "portrait_album",
            "crop": Crop(x=0.05, y=0, width=0.9, height=1).model_dump(),
            "color": Color(look="monochrome").model_dump(),
            "color_kind": "album",
        }
    ]

    exported = export_full_resolution_variants(
        output_dir, "runs/run-test", {"file": "nested/example.jpg"}, original, crops, settings
    )

    image_path = tmp_path / "output" / exported[0]["output"]
    assert image_path.is_file()
    with Image.open(image_path) as image:
        assert image.format == "JPEG"
        assert image.size == (576, 480)
        assert image.getpixel((10, 10))[0] == image.getpixel((10, 10))[1]
        assert image.getpixel((10, 10))[1] == image.getpixel((10, 10))[2]
    assert exported[0]["output_quality"] == 100


def test_equal_stems_do_not_collide_between_source_formats(tmp_path):
    settings = Settings(root=tmp_path, resolution_profile="custom", min_long_edge=400)
    settings.initialize()
    original = Image.new("RGB", (640, 480), (90, 120, 150))
    results = [
        export_color_options(settings.output_dir, {"file": f"photo.{suffix}"}, original, settings, Color())
        for suffix in ("RAF", "DNG")
    ]
    paths = [item["path"] for group in results for item in group]
    assert len(set(paths)) == 6
    for group in results:
        with Image.open(settings.root / "cache" / group[0]["render_source"]) as image:
            assert image.format == "TIFF"
            assert image.getpixel((0, 0)) == original.getpixel((0, 0))


def test_semantic_alternative_is_exportable_and_still_checks_protection(settings):
    from gallery_editor.schemas import CropAlternative

    option = CropAlternative(
        crop=Crop(x=0.01, width=0.99),
        confidence=0.8,
        rationale="Retain foreground depth while trimming one edge",
    )
    crops, _ = crop_options(
        (6000, 4000),
        {"protected_regions": [], "edge_centroid": [0.5, 0.5]},
        Analysis(confidence=0.9),
        settings,
        Crop(),
        [option],
    )
    assert any(c["crop"] == option.crop for c in crops)
    crops, rejected = crop_options(
        (6000, 4000),
        {"protected_regions": [Crop(x=0, width=0.2).model_dump()], "edge_centroid": [0.5, 0.5]},
        Analysis(confidence=0.9),
        settings,
        Crop(),
        [option],
    )
    assert any(c["id"] == "semantic-1" and "protected" in c["reason"] for c in rejected)
