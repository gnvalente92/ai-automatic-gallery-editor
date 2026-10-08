import math

from .imaging import crop_box, crop_metrics, enforce_resolution
from .schemas import Crop

STANDARD_ASPECTS = ("3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "1:1", "16:9", "9:16")


def has_standard_aspect(crop, size, tolerance=0.003):
    """Allow the camera's native ratio or one of the product's named presets."""
    width, height = size
    ratio = crop.width * width / (crop.height * height)
    source_ratio = width / height
    expected = [source_ratio, *(int(a) / int(b) for a, b in (item.split(":") for item in STANDARD_ASPECTS))]
    return any(abs(ratio - value) <= tolerance for value in expected)


def aspect_crop(crop, size, ratio):
    if ratio == "original":
        return crop
    a, b = map(int, ratio.split(":"))
    target = a / b
    w, h = crop.width, crop.height
    if w * size[0] / (h * size[1]) > target:
        w = h * size[1] * target / size[0]
    else:
        h = w * size[0] / target / size[1]
    return Crop(x=crop.x + (crop.width - w) / 2, y=crop.y + (crop.height - h) / 2, width=w, height=h)


def contains(crop, region):
    return (
        crop.x <= region.x
        and crop.y <= region.y
        and crop.x + crop.width >= region.x + region.width
        and crop.y + crop.height >= region.y + region.height
    )


def hard_keep_zones(cv_regions, semantic_regions):
    """Only detected faces and explicitly classified important subjects are hard crop guards."""
    return [Crop.model_validate(region) for region in cv_regions] + [
        Crop.model_validate(region.model_dump(exclude={"role", "confidence"}))
        for region in semantic_regions
        if region.role in {"primary", "secondary"} and region.confidence >= 0.6
    ]


def choose_crop(size, settings, cv, analysis, contract, proposal=None):
    protected = hard_keep_zones(cv["protected_regions"], analysis.protected_regions)
    candidates = [Crop()]
    # Without semantic analysis, preserve full context. Face/edge proxies do not justify removing objects.
    if analysis.confidence >= 0.65:
        for retain in (0.96, 0.92, 0.88):
            for dx, dy in ((0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)):
                candidates.append(Crop(x=(1 - retain) * dx, y=(1 - retain) * dy, width=retain, height=retain))
    if contract.preferred_aspect_ratio != "original":
        candidates.extend(aspect_crop(c, size, contract.preferred_aspect_ratio) for c in list(candidates))
    if proposal:
        candidates.append(proposal.crop)
        candidates.extend(option.crop for option in proposal.alternatives)
    results = []
    center = cv["edge_centroid"]
    for crop in candidates:
        reason = None
        try:
            crop_metrics(crop, size)
        except ValueError as exc:
            results.append(
                {
                    "crop": crop.model_dump(),
                    "metrics": None,
                    "score": 0,
                    "accepted": False,
                    "rejection": str(exc),
                }
            )
            continue
        try:
            metrics = enforce_resolution(crop, size, settings)
            if not has_standard_aspect(crop, size):
                raise ValueError("Crop must use the original ratio or a supported standard aspect ratio")
            if metrics["crop_percentage"] > 30:
                raise ValueError("Automatic crop removes more than 30% of original area")
            if any(not contains(crop, region) for region in protected):
                raise ValueError("Crop removes a protected face, body context or important subject")
            if analysis.confidence < 0.65 and crop != Crop():
                raise ValueError("Insufficient semantic confidence to remove context")
        except ValueError as exc:
            metrics = crop_metrics(crop, size)
            reason = str(exc)
        cx, cy = (center[0] - crop.x) / crop.width, (center[1] - crop.y) / crop.height
        distance = min(
            math.hypot(cx - x, cy - y)
            for x, y in ((0.5, 0.5), (1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))
        )
        composition = max(0, 1 - distance)
        prominence = min(1, 0.7 / (crop.width * crop.height))
        face_score = 1 if all(contains(crop, r) for r in protected) else 0
        resolution = 1 - metrics["crop_percentage"] / 100
        score = 0.4 * composition + 0.25 * prominence + 0.15 * face_score + 0.2 * resolution
        score -= metrics["crop_percentage"] / 500  # Explicit preference for retaining context.
        results.append(
            {
                "crop": crop.model_dump(),
                "metrics": metrics,
                "score": score,
                "accepted": reason is None,
                "rejection": reason,
            }
        )
    valid = [r for r in results if r["accepted"]]
    if not valid:
        raise ValueError("Original and all crop candidates fail the resolution floor")
    best = max(valid, key=lambda r: r["score"])
    similar = [r for r in valid if best["score"] - r["score"] < 0.025]
    chosen = min(similar, key=lambda r: r["metrics"]["crop_percentage"])
    if proposal and proposal.confidence >= 0.7:
        # The semantic crop agent can reason about story/context that the deterministic
        # edge-centroid proxy cannot. Honor its confident proposal after hard checks;
        # the proxy should not silently turn every such decision back into no-crop.
        proposed = next(
            (r for r in valid if r["crop"] == proposal.crop.model_dump()),
            None,
        )
        if proposed:
            chosen = proposed
    for result in results:
        result["selected"] = result is chosen
        result["source"] = (
            "semantic_proposal" if proposal and result["crop"] == proposal.crop.model_dump() else "candidate"
        )
    return Crop(**chosen["crop"]), results


def validate_manual(crop, size, settings, aspect):
    metrics = enforce_resolution(crop, size, settings)
    if not has_standard_aspect(crop, size):
        raise ValueError("Crop must use the original ratio or a supported standard aspect ratio")
    if aspect != "original":
        a, b = map(int, aspect.split(":"))
        _, _, _, _ = crop_box(crop, size)
        if abs(metrics["result_width"] / metrics["result_height"] - a / b) > 0.003:
            raise ValueError("Crop does not match the selected aspect ratio")
    return metrics
