"""Generate conservative, resolution-checked crop/color review combinations."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .cropping import contains, hard_keep_zones, has_standard_aspect
from .imaging import (
    RAWS,
    crop_box,
    crop_metrics,
    crop_only,
    enforce_resolution,
    grade,
    load_cached_source,
    load_image,
    preview,
    render,
    save_image,
)
from .schemas import Color, Crop
from .storage import digest, file_hash, safe_path

MAX_REVIEW_CROP_CHOICES = 10


def _focus(cv, analysis):
    semantic = [region for region in analysis.protected_regions if region.role == "primary" and region.confidence >= 0.6]
    if not semantic:
        semantic = [region for region in analysis.protected_regions if region.role == "secondary" and region.confidence >= 0.6]
    regions = [Crop.model_validate(region.model_dump(exclude={"role", "confidence"})) for region in semantic]
    if not regions:
        regions = [Crop.model_validate(r) for r in cv.get("protected_regions", [])]
    if not regions:
        return cv.get("edge_centroid", [0.5, 0.5])
    return [
        sum(r.x + r.width / 2 for r in regions) / len(regions),
        sum(r.y + r.height / 2 for r in regions) / len(regions),
    ]


def _format_crop(size, ratio, focus, placement=(0.5, 0.5)):
    """Place a crop of the requested aspect around a semantic focus point."""
    width, height = size
    source_ratio = width / height
    if source_ratio > ratio:
        crop_width, crop_height = ratio / source_ratio, 1.0
    else:
        crop_width, crop_height = 1.0, source_ratio / ratio
    x = min(max(focus[0] - placement[0] * crop_width, 0), 1 - crop_width)
    y = min(max(focus[1] - placement[1] * crop_height, 0), 1 - crop_height)
    return Crop(x=x, y=y, width=crop_width, height=crop_height)


def _native_crop(area_fraction, focus, placement=(0.5, 0.5)):
    """Crop at native aspect and position the subject at a composition target."""
    scale = area_fraction**0.5
    x = min(max(focus[0] - placement[0] * scale, 0), 1 - scale)
    y = min(max(focus[1] - placement[1] * scale, 0), 1 - scale)
    return Crop(x=x, y=y, width=scale, height=scale)


def _composition_candidates(size, focus, scene_class, gallery_role):
    """Generate subject-aware framing hypotheses in addition to semantic proposals."""
    context_profiles = {
        "group", "landscape", "street_documentary", "architecture_interior", "event_wedding", "night_astro"
    }
    close_profiles = {"portrait", "wildlife_nature", "macro_closeup", "sports_action"}
    context_role = gallery_role in {"establishing", "group", "context", "transition"}
    if scene_class in context_profiles or context_role:
        scales = (0.96, 0.90, 0.84)
    elif scene_class in close_profiles or gallery_role in {"portrait", "detail", "hero", "emotional", "action"}:
        scales = (0.96, 0.90, 0.84, 0.78)
    else:
        scales = (0.96, 0.90, 0.84, 0.78)
    candidates = [
        (f"native-{int(area * 100)}", f"Native ratio · {int((1 - area) * 100)}% area removed", _native_crop(area, focus))
        for area in scales
    ]
    if scene_class not in context_profiles and not context_role:
        for anchor, label in ((1 / 3, "left third"), (2 / 3, "right third")):
            candidates.append(
                (f"native-{label.replace(' ', '-')}", f"Native ratio · subject on {label}", _native_crop(0.84, focus, (anchor, 0.5)))
            )
    placements = ((0.5, "high"), (0.5, "low"))
    if scene_class not in context_profiles and not context_role:
        placements = ((1 / 3, "left-weighted"), (2 / 3, "right-weighted"), *placements)
    for anchor, label in placements:
        placement = (anchor, 0.5) if "weighted" in label else (0.5, anchor)
        candidates.append(
            (f"native-{label}", f"Native ratio · {label} balance", _native_crop(0.90, focus, placement))
        )
    width, height = size
    formats = (("wide", "16:9", 16 / 9), ("portrait", "4:5", 4 / 5)) if width >= height else (
        ("portrait", "4:5", 4 / 5), ("landscape", "3:2", 3 / 2)
    )
    for key, label, ratio in formats:
        candidates.append((key, f"{label} · subject aware", _format_crop(size, ratio, focus)))
        candidates.append(
            (f"{key}-thirds", f"{label} · subject on third", _format_crop(size, ratio, focus, (1 / 3, 0.5)))
        )
    return candidates


def crop_options(size, cv, analysis, settings, recommended, semantic_options=()):
    """Return safe composition choices plus reasons an orientation was withheld."""
    focus = _focus(cv, analysis)
    choices = [
        ("original", "Original framing", Crop()),
        ("recommended", "Recommended framing", recommended),
    ]
    choices.extend(_composition_candidates(size, focus, analysis.scene_class, analysis.gallery_role))
    choices.extend(
        (f"semantic-{index + 1}", option.rationale, option.crop, option.rationale)
        for index, option in enumerate(semantic_options)
    )
    if analysis.confidence < 0.65:
        choices = [("original", "Original framing · low scene confidence", Crop())]
    protected = hard_keep_zones(cv.get("protected_regions", []), analysis.protected_regions)
    accepted, rejected, seen = [], [], set()
    for choice in choices:
        key, label, crop = choice[:3]
        rationale = choice[3] if len(choice) > 3 else label
        try:
            metrics = crop_metrics(crop, size)
        except ValueError as exc:
            rejected.append({"id": key, "label": label, "reason": str(exc), "crop": crop.model_dump()})
            continue
        pixel_box = tuple(metrics["pixel_box"])
        if pixel_box in seen:
            continue
        seen.add(pixel_box)
        try:
            enforce_resolution(crop, size, settings)
            if not has_standard_aspect(crop, size):
                raise ValueError("Would create a non-standard aspect ratio")
            if any(not contains(crop, region) for region in protected):
                raise ValueError("Would cut a detected face or protected subject region")
            if metrics["crop_percentage"] > 30:
                raise ValueError("Would remove more than 30% of image area")
        except ValueError as exc:
            rejected.append({"id": key, "label": label, "reason": str(exc), "crop": crop.model_dump()})
            continue
        notes = ["Suggestion only: subject preservation and aesthetic quality require human review"]
        if metrics["crop_percentage"] > 30:
            notes.append("Format crop removes more than 30% of the original area; check framing carefully")
        accepted.append(
            {
                "id": key,
                "label": label,
                "crop": crop,
                "metrics": metrics,
                "rationale": rationale,
                "notes": notes,
            }
        )
    if len(accepted) > MAX_REVIEW_CROP_CHOICES:
        def priority(option):
            identifier = option["id"]
            if identifier == "original":
                return (0, identifier)
            if identifier == "recommended":
                return (1, identifier)
            if identifier.startswith("semantic-"):
                return (2, identifier)
            if identifier in {"wide", "portrait", "landscape"}:
                return (3, identifier)
            if identifier.startswith("native-"):
                return (4, identifier)
            return (5, identifier)

        accepted = sorted(accepted, key=priority)[:MAX_REVIEW_CROP_CHOICES]
    return accepted, rejected


def color_options(base):
    album = Color(**base.model_dump())
    soft_data = album.model_dump()
    soft_data.update(
        contrast=max(-25, album.contrast - 4),
        highlights=max(-40, album.highlights - 3),
        shadows=min(40, album.shadows + 2),
        saturation=max(-25, album.saturation - 2),
    )
    crisp_data = album.model_dump()
    crisp_data.update(
        contrast=min(25, album.contrast + 3),
        shadows=max(-20, album.shadows - 1),
        saturation=min(25, album.saturation + 1),
        vibrance=min(25, album.vibrance + 3),
    )
    if album.look == "natural":
        return [
            ("album", "Album match", album),
            ("soft", "Soft natural", Color(**soft_data)),
            ("crisp", "Crisp natural", Color(**crisp_data)),
        ]
    natural_data = album.model_dump()
    natural_data["look"] = "natural"
    return [
        ("album", f"Per-photo {album.look.replace('_', ' ')}", album),
        ("natural", "Natural color alternative", Color(**natural_data)),
        ("soft", "Softer per-photo look", Color(**soft_data)),
    ]


def export_color_options(output_dir, photo, original, settings, base_color):
    """Write full-resolution uncropped grading choices before crop selection begins."""
    enforce_resolution(Crop(), original.size, settings)
    source = Path(photo["file"])
    folder = Path("options") / source.parent / source.name
    exports = []
    for color_id, label, color in color_options(base_color):
        relative = (folder / f"color__{color_id}.jpg").as_posix()
        destination = safe_path(output_dir, relative)
        if destination.exists():
            raise ValueError(
                f"Color-option output already exists; move it aside before processing: {relative}"
            )
        # Keep a lossless full-resolution intermediate: final crops must not decode
        # and re-encode a JPEG that has already lost image information.
        key = digest(["full-grade-v1", photo.get("hash"), source.as_posix(), color, original.size])
        intermediate = settings.root / "cache" / "graded" / f"{key}.tiff"
        if photo.get("hash") and intermediate.is_file():
            image = load_image(intermediate)
        else:
            image = render(original, Crop(), color, settings)
            save_image(image, intermediate)
        save_image(image, destination, quality=100)
        exports.append(
            {
                "id": color_id,
                "label": label,
                "path": relative,
                "output_hash": file_hash(destination),
                "width": image.width,
                "height": image.height,
                "quality": 100,
                "crop": Crop().model_dump(),
                "color": color.model_dump(),
                "render_source": intermediate.relative_to(settings.root / "cache").as_posix(),
            }
        )
    return exports


def export_full_resolution_variants(
    output_dir, run_prefix, photo, original, variants, settings, color_exports=()
):
    """Write each safe crop/grade combination as a quality-100 JPEG in this run."""
    source = Path(photo["file"])
    raw = source.suffix.lower() in RAWS
    color_sources = {}
    for item in color_exports:
        # IDs are labels, not grade identities: a review may have changed the
        # color parameters after these color-only files were written.
        path = (
            safe_path(settings.root / "cache", item["render_source"])
            if item.get("render_source")
            else safe_path(output_dir, item["path"])
        )
        if path.is_file():
            color_sources[digest(Color.model_validate(item["color"]))] = path

    folder = Path("options") / source.parent / source.name
    raw_source = None
    graded_sources = {}
    for variant in variants:
        crop = Crop.model_validate(variant["crop"])
        color = Color.model_validate(variant["color"])
        filename = f"combo__{variant['id']}.jpg"
        relative = (folder / filename).as_posix()
        destination = safe_path(output_dir, relative)
        key = digest(color)
        source_path = color_sources.get(key)
        if source_path is not None:
            with Image.open(source_path) as graded_source:
                image = crop_only(graded_source.convert("RGB"), crop, settings)
        elif raw:
            if raw_source is None:
                raw_source = load_cached_source(
                    safe_path(settings.input_dir, photo["file"]), settings.root, photo["hash"]
                )
            if key not in graded_sources:
                graded_sources[key] = grade(raw_source, color)
            image = crop_only(graded_sources[key], crop, settings)
        else:
            image = render(original, crop, color, settings)
        save_image(image, destination, quality=100)
        variant["output"] = (Path(run_prefix) / relative).as_posix()
        variant["output_hash"] = file_hash(destination)
        variant["output_width"] = image.width
        variant["output_height"] = image.height
        variant["output_quality"] = 100
        variant["full_resolution_exported"] = True
    return variants


def option_contact_sheet(root, photo, variants, columns=4):
    """Create a labeled, bounded contact sheet for the independent option reviewer."""
    if not variants:
        return None
    tile_width, image_height, label_height = 320, 205, 42
    rows = (len(variants) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, option in enumerate(variants):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        with Image.open(root / "cache" / option["preview"]) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_width - 12, image_height - 12))
            sheet.paste(image, (x + (tile_width - image.width) // 2, y + (image_height - image.height) // 2))
        draw.text((x + 6, y + image_height + 3), option["id"], fill="black", font=font)
        draw.text((x + 6, y + image_height + 19), option["label"][:44], fill="#444444", font=font)
    destination = root / "cache" / "reviews" / f"options-{photo['id']}.jpg"
    save_image(sheet, destination, quality=90)
    return destination


def generate_variants(
    root,
    photo,
    original,
    cv,
    analysis,
    settings,
    recommended,
    base_color,
    source_size=None,
    semantic_options=(),
):
    crops, rejected = crop_options(
        source_size or original.size, cv, analysis, settings, recommended, semantic_options
    )
    colors = color_options(base_color)
    small = preview(original)
    variants = []
    for crop_choice in crops:
        for color_id, color_label, color in colors:
            key = digest(["crop-color-preview-v1", photo["hash"], crop_choice["crop"], color])[:12]
            relative = f"variants/{photo['id']}/{crop_choice['id']}-{color_id}-{key}.jpg"
            path = root / "cache" / relative
            if not path.is_file():
                image = small.crop(crop_box(crop_choice["crop"], small.size))
                save_image(grade(image, color), path)
            variants.append(
                {
                    "id": f"{crop_choice['id']}_{color_id}",
                    "label": f"{crop_choice['label']} · {color_label}",
                    "crop_kind": crop_choice["id"],
                    "crop_label": crop_choice["label"],
                    "color_kind": color_id,
                    "color_label": color_label,
                    "crop": crop_choice["crop"].model_dump(),
                    "color": color.model_dump(),
                    "crop_metrics": crop_choice["metrics"],
                    "crop_rationale": crop_choice.get("rationale", crop_choice["label"]),
                    "preview": relative,
                    "notes": crop_choice["notes"],
                    "full_resolution_exported": False,
                }
            )
    return variants, rejected
