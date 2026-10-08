import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator

from .schemas import Schema

PROFILES = {"digital": 3000, "digital_print": 4000, "high_quality_print": 5000, "custom": 0}


class Settings(Schema):
    root: Path = Field(default_factory=Path.cwd)
    input_dir: Path | None = None
    output_dir: Path | None = None
    resolution_profile: str = "digital_print"
    min_long_edge: int = Field(default=4000, ge=0)
    min_width: int = Field(default=0, ge=0)
    min_height: int = Field(default=0, ge=0)
    min_megapixels: float = Field(default=0, ge=0)
    local_model_endpoint: str = "http://127.0.0.1:11434/v1"
    vision_model: str = ""
    text_model: str = ""
    review_model: str = ""
    model_timeout: float = Field(default=180, gt=0, le=3600)
    option_review_timeout: float = Field(default=300, gt=0, le=3600)
    model_max_tokens: int = Field(default=8192, ge=512, le=32768)
    model_context_tokens: int = Field(default=32768, ge=8192, le=131072)
    analysis_workers: int = Field(default=4, ge=1, le=16)

    @field_validator("resolution_profile")
    @classmethod
    def profile(cls, value):
        if value not in PROFILES:
            raise ValueError("Unknown resolution profile")
        return value

    @field_validator("local_model_endpoint")
    @classmethod
    def local_only(cls, value):
        url = urlparse(value)
        host = url.hostname or ""
        if host == "localhost":
            value = value.replace("localhost", "127.0.0.1", 1)
            host = "127.0.0.1"
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
        if not allowed or url.scheme not in ("http", "https") or url.username or url.password:
            raise ValueError("Model endpoint must use a loopback IP; remote photo transfer is forbidden")
        if url.query or url.fragment:
            raise ValueError("Endpoint must not have a query or fragment")
        return value.rstrip("/")

    @classmethod
    def from_env(cls, root=None):
        data = {k: os.environ[k.upper()] for k in cls.model_fields if k.upper() in os.environ}
        profile = data.get("resolution_profile", "digital_print")
        data.setdefault("min_long_edge", PROFILES.get(profile, 4000))
        if root is not None:
            data["root"] = root
        return cls(**data)

    def initialize(self):
        self.root = self.root.expanduser().resolve()
        input_dir = self.input_dir or self.root / "input"
        output_dir = self.output_dir or self.root / "output"
        if Path(input_dir).expanduser().is_symlink() or Path(output_dir).expanduser().is_symlink():
            raise ValueError("Input and output directories cannot be symlinks")
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        managed = {
            "input": self.input_dir,
            "output": self.output_dir,
            "cache": self.root / "cache",
            "reports": self.root / "reports",
        }
        names = list(managed)
        for index, name in enumerate(names):
            path = managed[name]
            if path.is_symlink():
                raise ValueError(f"Managed directory cannot be a symlink: {path}")
            for other_name in names[index + 1 :]:
                other = managed[other_name]
                if path == other or path.is_relative_to(other) or other.is_relative_to(path):
                    raise ValueError(f"Managed directories must not overlap: {name} and {other_name}")
            path.mkdir(parents=True, exist_ok=True)
        for name in (
            "thumbnails",
            "previews",
            "analysis",
            "crop_candidates",
            "reviews",
            "overrides",
            "variants",
            "model_previews",
        ):
            path = self.root / "cache" / name
            if path.is_symlink():
                raise ValueError(f"Cache directory cannot be a symlink: {name}")
            path.mkdir(exist_ok=True)
