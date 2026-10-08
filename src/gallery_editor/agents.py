from pathlib import Path

import numpy as np

from .schemas import (
    Analysis,
    Color,
    ColorDecision,
    Crop,
    CropDecision,
    OptionSelection,
    PhotoEditPlan,
    Review,
)
from .vision import outliers

COMPOSITION_GUIDE = (
    "COMPOSITION PRINCIPLES: Name the photograph's subject and story before changing framing. Simplify only when an "
    "element competes without purpose; empty space, foreground framing, layers and environmental context may be "
    "intentional. Assess visual weight (size, brightness, sharpness, saturation, warmth, faces and text), balance, "
    "leading/implied lines, diagonals, curves, symmetry, patterns, foreground/middle/background depth, figure-ground "
    "separation, light as a focal cue, color relationships, perspective and orientation. Thirds and phi are soft "
    "placement hypotheses; the rule of thirds is only one option. Do not force thirds. Centered, diagonal, radial, asymmetrical or tense compositions may be stronger. Keep a "
    "horizon deliberate, preserve gaze/movement room, and inspect all four edges for tangents and partial intrusions. "
    "Foreground elements can frame a scene and create depth; compare crops that retain them with crops that remove "
    "them. Do not label an element a distraction based on position alone. Genre profile: portraits prioritize eyes, "
    "expression, safe crop levels, hands, headroom and look room; groups preserve all relationships; landscapes "
    "balance horizon, foreground anchor and focal point; street/documentary and events preserve decisive moments and "
    "story context; architecture protects geometry, symmetry and verticals; wildlife protects eyes, extremities and "
    "habitat; macro keeps the sharp focal plane; still life/product/food protects the complete meaningful object and "
    "labels; sports/action preserves face, ball/equipment, body action and lead room; night/astro preserves the "
    "foreground-sky relationship; minimal/abstract work may depend on negative space or deliberate graphic cropping. "
    "Use only standard output formats: original/native ratio, 3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 1:1, 16:9, or 9:16. "
    "Never invent or recommend a custom/free-form aspect ratio. These are defaults, not automatic rules. Alter aspect ratio only when requested or clearly beneficial. A crop must "
    "make a visible, photograph-specific improvement; a technically safe crop is not automatically a good crop."
)

CROP_COMPOSITION_RULE = (
    "Do no harm. Cropping is semantic composition, never face-centering, empty-space removal, or grid fitting. Work in "
    "the already EXIF-oriented display coordinates and reason only from the provided image, analysis, album/cluster "
    "context, and measured dimensions. First understand the scene and story; identify primary, secondary, contextual, "
    "and distracting elements; consider interaction, expression, gaze/movement, environmental context, image role "
    "(hero, establishing, action, portrait, detail, group, emotional, context, supporting), edges/tangents, useful "
    "negative space, leading lines, symmetry, and horizon. Never claim unavailable detections: this system does not "
    "currently provide pose keypoints, instance masks, OCR, depth, calibrated gaze/motion, learned aesthetics, or "
    "automatic straighten/perspective parameters. Express uncertainty where those signals matter.\n"
    "HARD RULES: keep primary subjects and essential interactions wholly understandable; protect faces and heads; "
    "keep every group member when the image is a group; avoid cuts at neck, elbow, wrist, waist, knee, or ankle, and "
    "retain meaningful hands, feet, sports equipment, balls, wheels, labels, and props. Never propose out-of-bounds "
    "coordinates, an unsupported aspect ratio, an unsafe crop, upscaling, or outpainting. Preserve all supplied tight "
    "subject keep-zones; broad context is not itself a subject keep-zone. The deterministic renderer checks protected "
    "regions and the configured pixel/area floors and may reject unsafe proposals.\n"
    "SOFT COMPOSITION: thirds and golden-ratio placements are candidate scoring heuristics only. Preserve centered or "
    "symmetric framing when intentional. Leave roughly 1.5–2 times more room in front of a clear gaze or movement "
    "direction than behind when the frame permits; retain negative space when it provides balance, anticipation, "
    "scale, calm, or context. A foreground pillar, obstruction, partial person/object, or other edge element may be "
    "intentional framing that adds depth or atmosphere: compare keeping a meaningful portion with removing it; do "
    "not automatically label it a distraction. Remove it only when evidence shows it is irrelevant and the story, "
    "relationships, and balance improve. Prefer clean crop edges without tangents or awkward slivers.\n"
    "CANDIDATES: classify the scene genre and gallery role first. Compare the unchanged full frame against distinct "
    "framing hypotheses: a conservative native-ratio refinement, a story-led placement that improves hierarchy/edge "
    "balance, and a genre-appropriate aspect alternative only when it helps and stays safe. Do not return near-duplicate "
    "alternatives. Use only original/native or standard aspect ratios (3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 1:1, 16:9, 9:16); never propose a custom/free-form format. For each one, state what the frame includes/excludes and the visual reason. Default to the original "
    "aspect ratio; change it only when requested or clearly superior and safe. Consider the image's scene profile: groups/context stay wide; portraits retain a deliberate safe crop level "
    "and headroom; action keeps the ball/equipment and lead room; landscape respects horizon/foreground; architecture "
    "respects symmetry/verticals; documents should follow their boundaries rather than artistic grids. Do not force "
    "variation across a sequence. Prefer the least aggressive crop that materially improves the photograph. No crop "
    "is a valid result: when it is within about 5% of the best plausible result, or the gain is marginal, retain the "
    "original. When confidence is below 0.4, preserve framing; below 0.7, favor conservative framing and state the "
    "uncertainty. Return the best recommendation plus up to three ranked alternatives in the CropDecision schema, "
    "each with normalized coordinates, confidence and a concrete rationale. Compare distinct framing hypotheses, "
    "including the original; include portrait/landscape alternatives only when they preserve the story and resolution. "
    "Consider album/cluster consistency without making photographs identical. Matching user-provided references may "
    "inform framing preference only for their paired source; never copy their pixels or force an unsafe match. A crop "
    "needs a reason tied to this photograph; otherwise keep the original.\n" + COMPOSITION_GUIDE
)


def composition_references(root, photo):
    """Find user-supplied examples paired to this source filename, never gallery inputs."""
    folder = root / "reference"
    if not folder.is_dir():
        return []
    stem = Path(photo["file"]).stem.casefold()
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    paths = []
    for path in sorted(folder.iterdir()):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in suffixes
            and path.stem.casefold().startswith(stem.casefold() + "_")
        ):
            paths.append(path)
    return paths[:4]


def reference_guidance(references):
    if not references:
        return None
    return (
        "The attached images are optional user-provided visual examples, not additional source frames. Infer only "
        "transferable preferences about framing, tonal treatment and ambient light. Compare those preferences with "
        "this photograph's own story and scene; do not copy a composition or treatment automatically. Use only the "
        "validated crop/color controls on the source. Never copy pixels, stretch, synthesize, or outpaint from the "
        "examples. Preserve protected subjects and resolution floors."
    )


def photo_agent(model, photo, root, cv, album_context=None):
    analysis_instructions = (
        "Classify scene_class using the allowed schema genres and gallery_role (hero, establishing, action, portrait, "
        "detail, group, emotional, context, transition, supporting, or unknown). Then describe the photographed scene "
        "and identify meaningful subjects, context, distractions, composition and "
        "uncertainty. For each protected_regions box, assign its role as primary, secondary, contextual, or distraction "
        "and a confidence. Use tight normalized boxes only for primary/secondary subjects a crop must preserve "
        "(especially heads/faces); never make broad background, negative space, foreground framing, or every "
        "visible person/object a hard keep-zone. Mark context separately so it can guide composition without blocking "
        "a crop. Keep contextual and foreground elements in the narrative description so the crop agent can decide "
        "whether they contribute. A foreground pillar/obstruction can be intentional framing; do not classify it as "
        "disposable merely because it is blurred or near an edge. Do not infer pose, "
        "gaze, joints, segmentation, horizon angle, or object boxes unless visibly supported."
    )
    result = model.decide(
        "photo analyst",
        Analysis,
        {
            "file": photo["file"],
            "measured": cv,
            "capture_metadata": photo.get("capture_metadata", {}),
            "album_context": album_context,
            "analysis_instructions": analysis_instructions,
        },
        [root / "cache" / photo["preview"]],
    )
    return result or Analysis(technical_issues=cv["limitations"])


def crop_agent(model, photo, root, analysis, cv, contract, settings, album_context=None, related=()):
    if not analysis.confidence:
        return None
    references = composition_references(root, photo)
    return model.decide(
        "crop editor",
        CropDecision,
        {
            "analysis": analysis.model_dump(),
            "measured": cv,
            "contract": contract.model_dump(),
            "album_context": album_context,
            "user_composition_reference": reference_guidance(references),
            "dimensions": [photo["width"], photo["height"]],
            "resolution": settings.model_dump(mode="json", exclude={"root", "input_dir", "output_dir"}),
            "composition_specification": CROP_COMPOSITION_RULE,
            "foreground_framing_guidance": (
                "A foreground obstruction, pillar or other partial element may be intentional framing that adds "
                "depth, atmosphere, scale or context. Do not automatically classify it as a distraction or crop it "
                "away. Compare candidates that retain a meaningful portion with candidates that remove it; choose "
                "based on the photograph's story and balance. This is a general composition principle, not a rule "
                "to retain every obstruction or to favor any particular photograph."
            ),
            "candidate_review": (
                "Consider the unchanged original, a mild crop, a moderate crop, and only when justified a tighter or "
                "alternate-aspect crop. Prefer the least aggressive candidate that clearly improves the photograph. "
                "Do not invent a crop merely to demonstrate activity."
            ),
            "rule": "No more than 30 percent area removed; preserve all people and important context",
        },
        [root / "cache" / photo["preview"], *related, *references],
    )


def color_baseline(cv, gallery, contract, album_context=None):
    stats = cv["statistics"]
    target = gallery["exposure"]["median"]
    rule = album_context.get("cluster_style", {}) if album_context else {}
    strength = rule.get("exposure_match_strength", 0.15)
    wb_strength = rule.get("white_balance_match_strength", 0.25)
    exposure = float(np.clip(np.log2(max(target, 0.02) / max(stats["exposure"], 0.02)) * strength, -0.3, 0.3))
    temperature = float(
        np.clip((gallery["temperature"]["median"] - stats["temperature"]) * 1800 * wb_strength, -250, 250)
    )
    level = contract.processing_level
    baseline = Color(
        exposure=round(exposure * level / 0.3, 3) if level <= 0.3 else exposure,
        contrast=float(np.clip((contract.contrast - 0.5) * 20, -10, 10)),
        temperature=temperature + (contract.warmth - 0.5) * 600,
        highlights=-min(18, stats["highlights"] * 100) * contract.highlight_protection,
        shadows=contract.shadow_lift * 12,
        saturation=float(
            np.clip(
                (gallery["saturation"]["median"] - stats["saturation"]) * 12
                + (contract.saturation - 0.5) * 20,
                -15,
                15,
            )
        ),
        grain=contract.grain,
    )
    return baseline


def photo_edit_agent(
    model, photo, root, analysis, cv, gallery, contract, settings, album_context=None, related=()
):
    baseline = color_baseline(cv, gallery, contract, album_context)
    references = composition_references(root, photo)
    reference_notes = reference_guidance(references)
    return model.decide(
        "photo crop and color editor",
        PhotoEditPlan,
        {
            "file": photo["file"],
            "analysis": analysis.model_dump(),
            "measured": cv,
            "gallery": gallery,
            "contract": contract.model_dump(),
            "album_context": album_context,
            "capture_metadata": photo.get("capture_metadata", {}),
            "dimensions": [photo["width"], photo["height"]],
            "resolution": settings.model_dump(mode="json", exclude={"root", "input_dir", "output_dir"}),
            "crop_rule": CROP_COMPOSITION_RULE,
            "color_rule": "Use cluster statistics and the album contract for restrained relative corrections. "
            "Public editing descriptions from Curtis Padley support a flexible range from subtle/natural to "
            "moodier/cinematic, using color depth, shaped light, selective shadow depth and refined color. Treat "
            "these as broad optional principles, not a signature look or preset. Choose any creative treatment "
            "independently for this photograph only: natural color by default, monochrome or warm monochrome when "
            "light, gesture, texture or subject matter makes it stronger, or restrained cinematic split tone when "
            "it serves the image. Do not propagate that choice to other photographs or make it an album-wide look. "
            "Keep creative looks as restrained alternatives to a natural treatment. Choose only what fits this album "
            "and scene; adapt exposure, white balance and contrast to the actual lighting. Preserve shadow detail, "
            "highlights, skin, team colors and ambient light; no LUT, film simulation or added grain by default. Treat "
            "film recipes and LUTs as optional descriptions of a look, never as a preset to apply blindly. Follow a "
            "reference's ambient color only when consistent with measured scene context.",
            "color_baseline": baseline.model_dump(),
            "user_composition_reference": reference_notes,
        },
        [root / "cache" / photo["preview"], *related, *references],
    )


def color_agent(model, photo, root, cv, gallery, contract, album_context=None, related=()):
    baseline = color_baseline(cv, gallery, contract, album_context)
    result = model.decide(
        "color editor",
        ColorDecision,
        {
            "measured": cv,
            "gallery": gallery,
            "contract": contract.model_dump(),
            "album_context": album_context,
            "capture_metadata": photo.get("capture_metadata", {}),
            "instruction": "Use the appropriate cluster, not the whole-gallery median. Preserve legitimate ambient "
            "lighting and tonal differences; coherent visual language does not mean identical values. Consider "
            "subtle/natural or moodier/cinematic tonal depth, shaping light and selectively deepening shadows only "
            "when the photograph and album contract support it. Adapt exposure, white balance and contrast per image; "
            "retain shadow detail, highlights, natural skin and real colors. Curtis Padley's public preset descriptions "
            "inform this broad range of choices, not a fixed look to copy. Choose natural, monochrome, warm monochrome "
            "or restrained cinematic only for this individual photograph when the image benefits. Never propagate an "
            "artistic look to the album or neighboring images; keep natural color as an alternative. Do not infer a "
            "crop rule or add grain. Use capture_metadata only as factual context: camera body, lens, aperture, "
            "shutter, ISO and focal length can help interpret depth of field, motion, and capture conditions, but "
            "must not dictate a camera-specific look or recipe. Base grading on visible pixels, actual illumination, "
            "the image's intent and its cluster contract. Do not assume high ISO means the image needs brightening, "
            "or that a lens/body implies a particular color response. Treat absent metadata as unknown.",
            "baseline": baseline.model_dump(),
        },
        [root / "cache" / photo["preview"], *related],
    )
    if result:
        result.color.grain = contract.grain
        return result
    return ColorDecision(
        color=baseline,
        confidence=0.45,
        rationale="Restrained relative correction within a measured visual group; preserve ambient lighting",
    )


def reviewer_agent(
    model, original, edited, neighbors, context, before, after, gallery, analysis, references=()
):
    # A fresh independent request: no editor conversation/history is supplied.
    review_context = {
        **context,
        "composition_guide": COMPOSITION_GUIDE,
        "review_instruction": (
            "Compare the original and edited frame for story, hierarchy, balance, subject/interaction preservation, "
            "genre-appropriate framing, lead room, edges and context. A crop may be safe yet compositionally weak. "
            "Do not approve merely because it preserves resolution; do not demand a crop when the original is stronger."
        ),
    }
    result = model.decide("reviewer", Review, review_context, [original, edited, *neighbors, *references])
    issues = []
    if after["highlights"] > before["highlights"] + 0.025:
        issues.append("Edit introduced highlight clipping")
    if after["shadows"] > before["shadows"] + 0.04:
        issues.append("Edit crushed shadow detail")
    issues.extend(outliers(after, gallery, look=context.get("color", {}).get("look", "natural")))
    if result:
        result.source = "local_model"
        result.issues.extend(issues)
        if issues or result.confidence < 0.65 or result.score < 75:
            result.approved = False
            result.needs_revision = True
        return result
    issues.append("Semantic reviewer unavailable; a human must check subjects, body cuts and skin tones")
    score = max(0, 90 - 10 * (len(issues) - 1))
    return Review(
        score=score,
        crop_score=90 if context["crop"] == Crop().model_dump() else 65,
        color_score=score,
        consistency_score=90 if len(issues) == 1 else 65,
        approved=False,
        needs_revision=len(issues) > 1,
        issues=issues,
        confidence=0.4,
    )


def option_reviewer(model, photo, root, options, sheet, contract, album_context, analysis, related=()):
    """Choose among actual rendered candidates using per-photo and album context."""
    return model.decide(
        "option reviewer",
        OptionSelection,
        {
            "instruction": "Independently compare every numbered candidate in the contact sheet and select the "
            "single crop-and-grade combination that best communicates this photograph while fitting the album's "
            "visual language. Judge the photo's mood and story together with the global mood, its scene cluster, "
            "cluster color tendencies and related images. Consistent means a shared visual language, not identical "
            "settings: preserve legitimate lighting and framing differences. Respect photographer intent, context, "
            "subject relationships, negative space, lead room and the supplied crop-safety metrics. Only choose a "
            "listed crop using the native image ratio or a standard preset (3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 1:1, "
            "16:9 or 9:16); custom formats are never allowed. Compare the full "
            "range of framing choices for real compositional benefit; do not default to the original simply because it "
            "is safest, and do not choose a crop simply because it is tighter. Prefer the less aggressive candidate "
            "when the visual result is otherwise equally strong. Capture metadata is factual context for "
            "interpreting lens depth/motion and capture conditions, never a camera-look preset; pixels and scene "
            "intent decide the grade. Select only an exact listed option ID; do not invent new crop/color values.",
            "photo_id": photo["id"],
            "scene_analysis": analysis.model_dump(),
            "capture_metadata": photo.get("capture_metadata", {}),
            "global_album_contract": contract.model_dump(),
            "photo_cluster_context": album_context,
            "composition_guide": COMPOSITION_GUIDE,
            "candidate_options": [
                {
                    "id": option["id"],
                    "label": option["label"],
                    "crop": option["crop"],
                    "color": option["color"],
                    "crop_metrics": option["crop_metrics"],
                    "crop_rationale": option.get("crop_rationale", ""),
                }
                for option in options
            ],
        },
        # The source framing is already present in the labeled contact sheet; do not
        # send it twice. Related album frames remain useful for consistency review.
        [sheet, *related],
    )


def soften(color):
    return Color(
        **{
            key: value * 0.5 if key not in {"grain", "look"} else value
            for key, value in color.model_dump().items()
        }
    )
