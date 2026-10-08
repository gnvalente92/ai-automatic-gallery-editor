import pytest
from conftest import make_photo

from gallery_editor.album import AlbumAnalyzer, effective_contract, group_library
from gallery_editor.pipeline import Pipeline
from gallery_editor.schemas import (
    AlbumContract,
    Analysis,
    Changes,
    ClusterInsight,
    Color,
    ColorDecision,
    ConditionalStyle,
    ConditionalStyleBatch,
    Crop,
    CropDecision,
    GalleryIssue,
    GalleryReview,
    PhotoEditPlan,
    Review,
    Scene,
    SceneBatch,
)
from gallery_editor.vision import gallery_statistics


def library(count=60):
    photos, analyses, scenes = [], [], {}
    for i in range(count):
        indoor = i < count // 2
        photo = {
            "id": f"photo-{i}",
            "file": f"{i}.jpg",
            "metadata": {},
            "width": 640,
            "height": 480,
            "preview": f"previews/{i}.jpg",
            "thumbnail": f"thumbnails/{i}.jpg",
        }
        stats = {
            "exposure": 0.2 if indoor else 0.7,
            "contrast": 0.15,
            "saturation": 0.2,
            "temperature": 0.15 if indoor else -0.1,
            "shadows": 0.1 if indoor else 0.01,
            "highlights": 0.01,
            "white_balance": [0.4, 0.35, 0.3],
            "luminance_histogram": [1 / 16] * 16,
        }
        analyses.append({"statistics": stats, "face_count": 1, "edge_centroid": [0.5, 0.5]})
        photos.append(photo)
        scenes[photo["id"]] = Scene(
            photo_id=photo["id"],
            environment="indoor" if indoor else "outdoor",
            lighting="tungsten" if indoor else "daylight",
            category="portrait",
            confidence=0.9,
        )
    return photos, analyses, scenes


class AlbumTestModel:
    """Contract fixture; synthetic responses are restricted to tests."""

    def __init__(self, scenes):
        self.scenes, self.calls = scenes, []

    def decide(self, role, schema, context, images=()):
        self.calls.append((role, context, list(images)))
        if schema is SceneBatch:
            return SceneBatch(photos=[self.scenes[p["photo_id"]] for p in context["photos"]])
        if schema is ClusterInsight:
            return ClusterInsight(observations=["Test observation"], confidence=0.9)
        if schema is AlbumContract:
            return AlbumContract(confidence=0.9)
        if schema is ConditionalStyleBatch:
            return ConditionalStyleBatch(
                styles=[
                    ConditionalStyle(cluster_id=g["cluster_id"], label=g["label"], confidence=0.9)
                    for g in context["clusters"]
                ]
            )
        return None


def test_all_photos_covered_and_every_cluster_deeply_analyzed(settings):
    photos, analyses, scenes = library(200)
    model = AlbumTestModel(scenes)
    stages = []
    contract, bundle = AlbumAnalyzer(settings.root, model, lambda *args: stages.append(args)).analyze(
        photos, analyses, gallery_statistics(analyses)
    )
    inventory = [call for call in model.calls if call[0] == "library scene inventory"]
    covered = [p["photo_id"] for _, context, _ in inventory for p in context["photos"]]
    assert set(covered) == {p["id"] for p in photos} and len(covered) == 200
    assert bundle["coverage"]["cheap_analysis"] == 200
    assert bundle["coverage"]["semantic_scene_results"] == 200
    assert set(bundle["membership"]) == set(covered)
    assert len(contract.conditional_rules) == len(bundle["clusters"])
    assert bundle["coverage"]["vision_lenses_per_cluster"] == 3
    cluster_calls = [role for role, _, _ in model.calls if role == "album cluster vision analyst"]
    assert len(cluster_calls) == sum((len(g["representatives"]) + 5) // 6 for g in bundle["clusters"])
    style_calls = [role for role, _, _ in model.calls if role == "cluster style author"]
    assert len(style_calls) == (len(bundle["clusters"]) + 7) // 8
    cluster_calls = [role for role, _, _ in model.calls if role == "album cluster vision analyst"]
    assert len(cluster_calls) == sum((len(g["representatives"]) + 5) // 6 for g in bundle["clusters"])
    style_calls = [role for role, _, _ in model.calls if role == "cluster style author"]
    assert len(style_calls) == (len(bundle["clusters"]) + 7) // 8
    for cluster in bundle["clusters"]:
        assert cluster["insights"]
        assert all(i["lenses"] == ["lighting", "color", "composition"] for i in cluster["insights"])
        observed = {p for insight in cluster["insights"] for p in insight["photo_ids"]}
        assert observed == set(cluster["representatives"])
    assert stages[-1][0] == "Building visual contract"
    assert (settings.root / "reports" / "album_analysis.json").exists()


def test_clustering_preserves_lighting_and_is_deterministic():
    photos, analyses, scenes = library()
    groups, membership = group_library(photos, analyses, scenes)
    assert (groups, membership) == group_library(photos, analyses, scenes)
    for group in groups:
        environments = {scenes[p].environment for p in group["members"]}
        assert len(environments) == 1
        assert group["statistics"]["exposure"]["median"] in (0.2, 0.7)


def test_conditional_contract_preserves_global_and_local_rules():
    contract = AlbumContract(warmth=0.6, contrast=0.5)
    indoor = ConditionalStyle(cluster_id="indoor", warmth_offset=0.1, contrast_offset=-0.1)
    outdoor = ConditionalStyle(cluster_id="outdoor", warmth_offset=-0.05, contrast_offset=0.1)
    assert effective_contract(contract, indoor).warmth == pytest.approx(0.7)
    assert effective_contract(contract, outdoor).warmth == pytest.approx(0.55)
    assert effective_contract(contract, indoor).contrast < effective_contract(contract, outdoor).contrast


def test_malformed_scene_batch_cannot_claim_coverage(settings):
    photos, analyses, scenes = library(3)

    class Incomplete(AlbumTestModel):
        def decide(self, role, schema, context, images=()):
            if schema is SceneBatch:
                return SceneBatch(photos=[Scene(photo_id="invented", confidence=1)])
            return None

    _, bundle = AlbumAnalyzer(settings.root, Incomplete(scenes), lambda *args: None).analyze(
        photos, analyses, gallery_statistics(analyses)
    )
    assert bundle["coverage"]["semantic_scene_results"] == 0
    assert all(s["environment"] == "unknown" for s in bundle["scenes"].values())


class FinalRevisionModel:
    def __init__(self, initial_rejections=0):
        self.initial_rejections = initial_rejections
        self.reviews = 0
        self.calls = []

    def decide(self, role, schema, context, images=()):
        self.calls.append(role)
        if schema is Analysis:
            return Analysis(confidence=0.9)
        if schema is PhotoEditPlan:
            return PhotoEditPlan(
                crop=CropDecision(crop=Crop(), confidence=0.9, rationale="Test fixture"),
                color=ColorDecision(color=Color(), confidence=0.9, rationale="Test fixture"),
            )
        if schema is CropDecision:
            return CropDecision(crop=Crop(), confidence=0.9, rationale="Test fixture")
        if schema is ColorDecision:
            return ColorDecision(color=Color(), confidence=0.9, rationale="Test fixture")
        if role == "reviewer":
            self.reviews += 1
            failed = self.reviews <= self.initial_rejections
            return Review(
                score=50 if failed else 95,
                crop_score=95,
                color_score=95,
                consistency_score=95,
                approved=not failed,
                needs_revision=failed,
                issues=["Test failure"] if failed else [],
                confidence=0.9,
            )
        if role == "final gallery reviewer":
            return GalleryReview(
                confidence=0.9,
                issues=[
                    GalleryIssue(
                        photo_id=context["photos"][0]["photo_id"],
                        issues=["Test cross-image mismatch"],
                        needs_revision=True,
                        suggested_changes=Changes(color=Color(contrast=2)),
                    )
                ],
            )
        if schema is GalleryReview:
            return GalleryReview(confidence=0.9)
        return None


def test_final_gallery_can_revise_after_individual_approval(settings):
    make_photo(settings.root / "input" / "a.jpg")
    model = FinalRevisionModel()
    result = Pipeline(settings, model).run()
    record = result["photos"][0]
    assert record["revision_cycles"] == 1
    assert record["color"]["contrast"] == 2
    assert record["final_gallery_revision"]
    assert model.reviews == 2
    assert model.calls.index("album stylist") < model.calls.index("photo analyst")
    assert model.calls.index("color editor") < model.calls.index("crop editor")
    assert "photo crop and color editor" not in model.calls
    assert model.calls.index("color editor") < model.calls.index("crop editor")
    assert "photo crop and color editor" not in model.calls
    assert model.calls.index("final gallery reviewer") > model.calls.index("reviewer")
    assert "final gallery revision verification" in model.calls


def test_initial_and_final_reviews_share_two_revision_budget(settings):
    make_photo(settings.root / "input" / "a.jpg")
    model = FinalRevisionModel(initial_rejections=2)
    record = Pipeline(settings, model).run()["photos"][0]
    assert record["revision_cycles"] == 2
    assert model.reviews == 3
    assert record["status"] == "manual_review_required"
    assert "two-revision budget" in " ".join(record["final_review_issues"])


def test_album_analysis_reaches_small_outlying_groups(settings):
    photos, analyses, scenes = library(25)
    scenes[photos[-1]["id"]] = Scene(
        photo_id=photos[-1]["id"],
        environment="indoor",
        lighting="low_light",
        category="action",
        confidence=0.9,
    )
    model = AlbumTestModel(scenes)
    _, bundle = AlbumAnalyzer(settings.root, model, lambda *args: None).analyze(
        photos, analyses, gallery_statistics(analyses)
    )
    outlier = next(g for g in bundle["clusters"] if photos[-1]["id"] in g["members"])
    assert outlier["members"] == [photos[-1]["id"]]
    assert len(outlier["insights"]) == 1
    assert outlier["insights"][0]["lenses"] == ["lighting", "color", "composition"]


def test_final_gallery_never_overwrites_manual_edit(settings):
    from gallery_editor.schemas import Override

    make_photo(settings.root / "input" / "a.jpg")
    pipeline = Pipeline(settings, FinalRevisionModel())
    first = pipeline.run()["photos"][0]
    pipeline.override(first["id"], Override(crop=Crop(), color=Color(contrast=14)))
    result = pipeline.run()["photos"][0]
    assert result["manual"] and result["color"]["contrast"] == 14
    assert result["revision_cycles"] == 0


def test_color_uses_cluster_not_global_luminance(settings):
    from PIL import Image

    from gallery_editor.agents import color_agent
    from gallery_editor.vision import statistics

    class NoModel:
        def decide(self, *args, **kwargs):
            return None

    cv = {"statistics": statistics(Image.new("RGB", (640, 480), (50, 50, 50)))}
    stats = {
        "exposure": {"median": cv["statistics"]["exposure"]},
        "temperature": {"median": 0},
        "saturation": {"median": 0},
    }
    context = {
        "gallery_statistics": {"exposure": {"median": 0.75}},
        "cluster_style": ConditionalStyle(cluster_id="dark-indoor").model_dump(),
    }
    decision = color_agent(
        NoModel(), {"preview": "ignored"}, settings.root, cv, stats, AlbumContract(), context
    )
    assert decision.color.exposure == 0
