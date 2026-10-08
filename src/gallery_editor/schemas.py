from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Crop(Schema):
    x: float = Field(default=0, ge=0, lt=1)
    y: float = Field(default=0, ge=0, lt=1)
    width: float = Field(default=1, gt=0, le=1)
    height: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def bounds(self):
        if self.x + self.width > 1.000000001 or self.y + self.height > 1.000000001:
            raise ValueError("Crop extends beyond the original image")
        return self


class ProtectedRegion(Crop):
    """Semantically classified subject region; context is advisory, people/subjects are hard keep-zones."""

    role: Literal["primary", "secondary", "contextual", "distraction"] = "primary"
    confidence: float = Field(default=0.7, ge=0, le=1)


class Color(Schema):
    look: Literal["natural", "monochrome", "warm_monochrome", "cinematic"] = "natural"
    exposure: float = Field(default=0, ge=-1, le=1)
    contrast: float = Field(default=0, ge=-25, le=25)
    highlights: float = Field(default=0, ge=-40, le=20)
    shadows: float = Field(default=0, ge=-20, le=40)
    whites: float = Field(default=0, ge=-20, le=20)
    blacks: float = Field(default=0, ge=-20, le=20)
    temperature: float = Field(default=0, ge=-1200, le=1200)
    tint: float = Field(default=0, ge=-15, le=15)
    saturation: float = Field(default=0, ge=-25, le=25)
    vibrance: float = Field(default=0, ge=-25, le=25)
    grain: float = Field(default=0, ge=0, le=10)


Aspect = Literal["original", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "1:1", "16:9", "9:16"]


class ConditionalStyle(Schema):
    cluster_id: str
    label: str = "Measured visual group"
    rationale: str = "Preserve the group's original lighting and context"
    warmth_offset: float = Field(default=0, ge=-0.25, le=0.25)
    contrast_offset: float = Field(default=0, ge=-0.25, le=0.25)
    saturation_offset: float = Field(default=0, ge=-0.25, le=0.25)
    shadow_lift_offset: float = Field(default=0, ge=-0.25, le=0.25)
    highlight_protection_offset: float = Field(default=0, ge=-0.25, le=0.25)
    exposure_match_strength: float = Field(default=0.15, ge=0, le=0.5)
    white_balance_match_strength: float = Field(default=0.25, ge=0, le=0.5)
    preserve_context: bool = True
    crop_style: Literal["conservative", "subject_priority"] = "conservative"
    preferred_aspect_ratio: Aspect = "original"
    legitimate_differences: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)


class AlbumContract(Schema):
    album_type: str = "unknown"
    mood: list[str] = Field(default_factory=list, max_length=8)
    style: str = "Natural, restrained corrections"
    dominant_palette: list[str] = Field(default_factory=list, max_length=8)
    contrast: float = Field(default=0.5, ge=0, le=1)
    saturation: float = Field(default=0.5, ge=0, le=1)
    warmth: float = Field(default=0.5, ge=0, le=1)
    highlight_protection: float = Field(default=0.75, ge=0, le=1)
    shadow_lift: float = Field(default=0.1, ge=0, le=1)
    grain: float = Field(default=0, ge=0, le=10)
    crop_style: Literal["conservative", "subject_priority"] = "conservative"
    preferred_aspect_ratio: Aspect = "original"
    skin_tone_priority: Literal["high", "normal"] = "high"
    processing_level: float = Field(default=0.3, ge=0, le=1)
    consistency_target: str = "Preserve scene lighting while reducing technical outliers"
    confidence: float = Field(default=0, ge=0, le=1)
    source: Literal["local_model", "deterministic", "user"] = "deterministic"
    global_rules: list[str] = Field(
        default_factory=lambda: [
            "Preserve differences in scene lighting",
            "Natural skin tones",
            "Restrained saturation",
            "No added grain",
        ]
    )
    conditional_rules: list[ConditionalStyle] = Field(default_factory=list)
    album_relationships: list[str] = Field(default_factory=list)


class Scene(Schema):
    photo_id: str
    environment: Literal["indoor", "outdoor", "mixed", "unknown"] = "unknown"
    lighting: Literal["daylight", "tungsten", "mixed", "artificial", "low_light", "unknown"] = "unknown"
    category: Literal["portrait", "group", "action", "close_up", "environmental", "other", "unknown"] = (
        "unknown"
    )
    scene_description: str = "Unknown"
    recurring_subjects: list[str] = Field(default_factory=list)
    context_importance: Literal["high", "normal", "unknown"] = "unknown"
    confidence: float = Field(default=0, ge=0, le=1)


class SceneBatch(Schema):
    photos: list[Scene]


class ClusterInsight(Schema):
    observations: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    legitimate_differences: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)


class GalleryIssue(Schema):
    photo_id: str
    issues: list[str]
    needs_revision: bool = False
    suggested_changes: "Changes" = Field(default_factory=lambda: Changes())


class GalleryReview(Schema):
    observations: list[str] = Field(default_factory=list)
    issues: list[GalleryIssue] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)


class Analysis(Schema):
    subjects: list[str] = Field(default_factory=list, max_length=30)
    expressions: list[str] = Field(default_factory=list, max_length=30)
    composition: str = "Unknown"
    horizon: str = "Unknown"
    distractions: list[str] = Field(default_factory=list, max_length=30)
    technical_issues: list[str] = Field(default_factory=list, max_length=30)
    protected_regions: list[ProtectedRegion] = Field(default_factory=list, max_length=40)
    confidence: float = Field(default=0, ge=0, le=1)


class CropAlternative(Schema):
    crop: Crop = Field(default_factory=Crop)
    confidence: float = Field(ge=0, le=1)
    rationale: str


class CropDecision(CropAlternative):
    alternatives: list[CropAlternative] = Field(default_factory=list, max_length=3)


class ColorDecision(Schema):
    color: Color
    confidence: float = Field(ge=0, le=1)
    rationale: str


class PhotoEditPlan(Schema):
    """Joint crop and color proposal; measurements and hard bounds remain deterministic."""

    crop: CropDecision
    color: ColorDecision


class ConditionalStyleBatch(Schema):
    styles: list[ConditionalStyle] = Field(default_factory=list, max_length=24)


class Changes(Schema):
    crop: Crop | None = None
    color: Color | None = None


class Review(Schema):
    score: float = Field(ge=0, le=100)
    crop_score: float = Field(ge=0, le=100)
    color_score: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=100)
    approved: bool
    needs_revision: bool
    issues: list[str]
    suggested_changes: Changes = Field(default_factory=Changes)
    confidence: float = Field(ge=0, le=1)
    source: Literal["local_model", "deterministic"] = "deterministic"

    @model_validator(mode="after")
    def coherent(self):
        if self.approved and self.needs_revision:
            raise ValueError("An approved edit cannot also require revision")
        return self


class OptionSelection(Schema):
    """Independent reviewer chooses one actual rendered crop/grade option."""

    selected_option_id: str
    ranked_option_ids: list[str] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1)
    rationale: str


class Override(Schema):
    crop: Crop
    color: Color
    aspect_ratio: Aspect = "original"


class ProcessRequest(Schema):
    style: AlbumContract | None = None
    theme: str | None = Field(default=None, max_length=2000)

    @field_validator("theme")
    @classmethod
    def normalize_theme(cls, value):
        value = " ".join(value.split()) if value else None
        return value or None
