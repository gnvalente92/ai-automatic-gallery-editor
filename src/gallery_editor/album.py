"""Full-library understanding, deterministic grouping and hierarchical local vision analysis.

Batch sizes bound a request's memory/context; they never cap library coverage.
"""

from collections import defaultdict

import numpy as np

from .schemas import (
    AlbumContract,
    ClusterInsight,
    ConditionalStyle,
    ConditionalStyleBatch,
    Scene,
    SceneBatch,
)
from .storage import digest, write_json
from .vision import gallery_statistics


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def features(photo, cv):
    s = cv["statistics"]
    rgb = s["white_balance"]
    return [
        s["exposure"] * 3,
        s["temperature"] * 3,
        s["contrast"] * 3,
        s["saturation"] * 2,
        s["shadows"] * 2,
        s["highlights"] * 2,
        *rgb,
        min(cv["face_count"], 5) * 0.3,
        photo["width"] / max(photo["height"], 1) * 0.3,
        *cv["edge_centroid"],
        *s["luminance_histogram"],
    ]


def group_library(photos, analyses, scenes):
    """Separate known scene/lighting categories, then split distant measured visual groups.

    Deterministic farthest-first partitioning keeps dissimilar lighting apart. Large groups
    split until every group has <=24 members, so deep-analysis coverage scales with the library.
    Labels remain measured/unknown when semantic evidence is absent.
    """
    if not photos:
        return [], {}
    matrix = np.asarray([features(p, c) for p, c in zip(photos, analyses)], dtype=np.float64)
    buckets = defaultdict(list)
    for index, photo in enumerate(photos):
        scene = scenes[photo["id"]]
        key = (
            (scene.environment, scene.lighting, scene.category)
            if scene.confidence >= 0.6
            else ("unknown",) * 3
        )
        buckets[key].append(index)
    partitions = []

    def partition(indices, key):
        points = matrix[indices]
        center = np.median(points, axis=0)
        radii = np.linalg.norm(points - center, axis=1)
        if len(indices) <= 24 and (len(indices) <= 2 or float(radii.max()) <= 0.7):
            partitions.append((indices, key))
            return
        a = int(np.argmax(radii))
        b = int(np.argmax(np.linalg.norm(points - points[a], axis=1)))
        if np.allclose(points[a], points[b]):
            middle = len(indices) // 2
            partition(indices[:middle], key)
            partition(indices[middle:], key)
            return
        left_mask = np.linalg.norm(points - points[a], axis=1) <= np.linalg.norm(points - points[b], axis=1)
        left = [index for index, keep in zip(indices, left_mask) if keep]
        right = [index for index, keep in zip(indices, left_mask) if not keep]
        partition(left, key)
        partition(right, key)

    for key, indices in sorted(buckets.items()):
        partition(indices, key)
    groups, membership = [], {}
    for indices, key in partitions:
        ids = [photos[i]["id"] for i in indices]
        group_id = "cluster-" + digest(sorted(ids))[:12]
        points = matrix[indices]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        medoid = int(np.argmin(distances.sum(axis=1)))
        chosen = {medoid}
        # Include exposure/color extremes plus all remaining visually distinct members.
        for axis in (0, 1, 2, 3):
            chosen.update((int(np.argmin(points[:, axis])), int(np.argmax(points[:, axis]))))
        while True:
            nearest = distances[:, sorted(chosen)].min(axis=1)
            farthest = int(np.argmax(nearest))
            if nearest[farthest] <= 0.3:
                break
            chosen.add(farthest)
        representatives = [photos[indices[i]]["id"] for i in sorted(chosen)]
        stats = gallery_statistics([analyses[i] for i in indices])
        known = [part for part in key if part != "unknown"]
        label = (
            " / ".join(known)
            if known
            else (
                f"Measured group · luminance {stats['exposure']['median']:.2f} · "
                f"warm/cool proxy {stats['temperature']['median']:+.2f}; scene unknown"
            )
        )
        groups.append(
            {
                "id": group_id,
                "label": label,
                "members": ids,
                "representatives": representatives,
                "statistics": stats,
                "semantic_key": list(key),
                "insights": [],
                "representative_method": "Medoid, exposure/color extremes and distance coverage",
            }
        )
        membership.update({photo_id: group_id for photo_id in ids})
    return groups, membership


def effective_contract(global_contract, rule):
    """Conditional offsets preserve user global changes without flattening scene differences."""
    data = global_contract.model_dump()
    for field in ("warmth", "contrast", "saturation", "shadow_lift", "highlight_protection"):
        data[field] = float(np.clip(data[field] + getattr(rule, field + "_offset"), 0, 1))
    data["crop_style"] = rule.crop_style
    if rule.preferred_aspect_ratio != "original":
        data["preferred_aspect_ratio"] = rule.preferred_aspect_ratio
    return AlbumContract.model_validate(data)


class AlbumAnalyzer:
    def __init__(self, root, model, progress):
        self.root, self.model, self.progress = root, model, progress

    def analyze(self, photos, analyses, stats, requested=None, theme=None):
        theme = " ".join(theme.split()) if theme else None
        photo_map = {p["id"]: p for p in photos}
        cv_map = {p["id"]: a for p, a in zip(photos, analyses)}
        scenes = {p["id"]: Scene(photo_id=p["id"]) for p in photos}
        self.progress("Analyzing scene information for all photographs", 0, len(photos))
        completed = 0
        for batch in chunks(photos, 8):
            result = self.model.decide(
                "library scene inventory",
                SceneBatch,
                {
                    "task": "Classify EVERY supplied image, in order, using its exact photo_id. Unknown is valid. "
                    "Observe environment, lighting and framing; do not infer identities.",
                    "user_theme": theme,
                    "photos": [
                        {
                            "photo_id": p["id"],
                            "file": p["file"],
                            "metadata": p["metadata"],
                            "capture_metadata": p.get("capture_metadata", {}),
                            "measured": cv_map[p["id"]],
                        }
                        for p in batch
                    ],
                },
                [self.root / "cache" / p["preview"] for p in batch],
            )
            expected = {p["id"] for p in batch}
            if (
                result
                and len(result.photos) == len(expected)
                and {p.photo_id for p in result.photos} == expected
            ):
                scenes.update({s.photo_id: s for s in result.photos})
            completed += len(batch)
            self.progress("Analyzing scene information for all photographs", completed, len(photos))
        self.progress("Clustering photographs", 0, len(photos))
        groups, membership = group_library(photos, analyses, scenes)
        self.progress("Clustering photographs", len(photos), len(photos))
        # One structured call per representative batch covers all lenses; the full image/group coverage is unchanged.
        lenses = {
            "lighting": "natural/artificial light, indoor/outdoor differences, exposure intent, shadow/highlight "
            "handling; distinguish lighting differences to preserve",
            "color": "recurring palettes, skin-tone observations without identity inference, white-balance tendencies, "
            "saturation/contrast and shadow/highlight intent; warm must not mean orange. Consider both subtle/natural "
            "and moody/cinematic directions only where the gallery supports them",
            "composition": "documentary/posed/action relationships, crop/aspect tendencies, subject framing and "
            "importance of environmental or motion context",
        }
        total = sum(len(g["representatives"]) for g in groups)
        stage = "Analyzing cluster lighting, color and composition"
        self.progress(stage, 0, total)
        seen = 0
        for group in groups:
            for ids in chunks(group["representatives"], 6):
                insight = self.model.decide(
                    "album cluster vision analyst",
                    ClusterInsight,
                    {
                        "analysis_lenses": lenses,
                        "cluster": group["label"],
                        "cluster_statistics": group["statistics"],
                        "gallery_statistics": stats,
                        "member_count": len(group["members"]),
                        "question": "What connects these photographs, and what should stay different? Address every "
                        "analysis lens separately while inspecting the full representative batch.",
                        "user_theme": theme,
                        "representatives": [scenes[i].model_dump() for i in ids],
                    },
                    [self.root / "cache" / photo_map[i]["preview"] for i in ids],
                )
                group["insights"].append(
                    {
                        "stage": stage,
                        "lenses": list(lenses),
                        "photo_ids": ids,
                        "result": insight.model_dump() if insight else None,
                    }
                )
                seen += len(ids)
                self.progress(stage, seen, total)
        self.progress("Understanding album mood", 0, len(groups))
        # Hierarchical synthesis covers all cluster reports, never truncating to a random global sample.
        summaries = [
            {
                "cluster_id": g["id"],
                "label": g["label"],
                "size": len(g["members"]),
                "insights": g["insights"],
                "statistics": g["statistics"],
            }
            for g in groups
        ]
        level = 0
        while len(summaries) > 6:
            reduced = []
            for batch in chunks(summaries, 6):
                insight = self.model.decide(
                    "album relationship synthesizer",
                    ClusterInsight,
                    {
                        "question": "Connect these groups without erasing meaningful lighting, subject or scene differences",
                        "clusters": batch,
                        "level": level,
                    },
                )
                reduced.append(
                    insight.model_dump()
                    if insight
                    else {
                        "measured_groups": [
                            {
                                "label": g.get("label", "subgroups"),
                                "summary": "Semantic relationships unknown",
                            }
                            for g in batch
                        ]
                    }
                )
            summaries = reduced
            level += 1
        self.progress("Understanding album mood", len(groups), len(groups))
        self.progress("Building visual contract", 0, len(groups) + 1)
        proposed = (
            self.model.decide(
                "album stylist",
                AlbumContract,
                {
                    "question": "Establish ONE photographic visual language spanning all analyzed groups. "
                    "Create global principles. Do not force identical exposure or white balance. "
                    "Curtis Padley's published editing descriptions range from natural/subtle to moodier/cinematic, "
                    "using color depth, shaped light, selective shadow depth and refined color, then adjusting "
                    "exposure, white balance and contrast to the individual image. Treat these as broad optional "
                    "editing principles, not a look to copy or a preset to apply. Choose the direction from this "
                    "gallery and the user's theme; preserve scene-specific light, shadow detail and natural color. "
                    "Conditional rules are built separately for all clusters; leave conditional_rules empty.",
                    "full_library_count": len(photos),
                    "cluster_summaries": summaries,
                    "gallery_statistics": stats,
                    "user_theme": theme,
                    "theme_instruction": "Treat this as photographer/editor guidance. Translate it into a coherent "
                    "visual language while preserving legitimate scene and lighting differences.",
                },
            )
            if photos
            else None
        )
        contract = requested or proposed or AlbumContract()
        if requested is None and proposed:
            contract.source = "local_model"
        if requested is None and not proposed and stats:
            rgb = stats["white_balance"]["median"]
            contract.dominant_palette = ["#" + "".join(f"{int(np.clip(c, 0, 1) * 255):02x}" for c in rgb)]
        user_rules = {r.cluster_id: r for r in requested.conditional_rules} if requested else {}
        rules = []
        generated_rules = {}
        for batch in chunks([g for g in groups if g["id"] not in user_rules], 8):
            result = self.model.decide(
                "cluster style author",
                ConditionalStyleBatch,
                {
                    "global_contract": contract.model_dump(),
                    "user_theme": theme,
                    "clusters": [
                        {
                            "cluster_id": group["id"],
                            "label": group["label"],
                            "insights": group["insights"],
                            "statistics": group["statistics"],
                        }
                        for group in batch
                    ],
                    "instruction": "Return exactly one conditional style for every supplied cluster_id. Use modest "
                    "conditional offsets. Preserve original ambient lighting and context. A cluster may take a more "
                    "natural/subtle or moodier, deeper-toned treatment only when its images support it; protect shadow "
                    "detail and highlights. Adjust exposure, white balance and contrast per lighting condition. "
                    "Do not equalize portraits, action and environmental photographs. Never invent cluster IDs.",
                },
            )
            allowed = {g["id"] for g in batch}
            if result:
                for rule in result.styles:
                    if rule.cluster_id in allowed and rule.cluster_id not in generated_rules:
                        generated_rules[rule.cluster_id] = rule
        for index, group in enumerate(groups):
            rule = user_rules.get(group["id"]) or generated_rules.get(group["id"])
            if rule is None:
                rule = ConditionalStyle(
                    cluster_id=group["id"],
                    label=group["label"],
                    legitimate_differences=[
                        "Measured lighting differences retained; semantic intent unknown"
                    ],
                )
            rules.append(rule)
            group["contract"] = rule.model_dump()
            self.progress("Building visual contract", index + 2, len(groups) + 1)
        contract.conditional_rules = rules
        bundle = {
            "version": "deep-album-v1",
            "library_count": len(photos),
            "clusters": groups,
            "membership": membership,
            "scenes": {key: value.model_dump() for key, value in scenes.items()},
            "gallery_statistics": stats,
            "user_theme": theme,
            "contract": contract.model_dump(),
            "coverage": {
                "cheap_analysis": len(analyses),
                "scene_requests": len(photos),
                "semantic_scene_results": sum(s.confidence > 0 for s in scenes.values()),
                "clusters_analyzed": len(groups),
                "representatives": total,
                "vision_lenses_per_cluster": len(lenses),
            },
        }
        write_json(self.root / "cache" / "album_analysis.json", bundle)
        write_json(self.root / "reports" / "album_analysis.json", bundle)
        return contract, bundle
