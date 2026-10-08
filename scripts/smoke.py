"""Run a reproducible full-resolution smoke gallery in an isolated workspace.

These generated test patterns are not photographs and are never presented as AI evidence.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gallery_editor.config import Settings
from gallery_editor.pipeline import Pipeline
from gallery_editor.schemas import Color, Crop, Override


def hashes(folder):
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    settings = Settings(root=args.root)
    pipeline = Pipeline(settings)
    assert all((args.root / name).is_dir() for name in ("input", "output", "cache", "reports"))
    for i, level in enumerate((0.85, 1, 1.15)):
        w, h = 4200, 2800
        x = np.linspace(0.12, 0.75, w, dtype=np.float32)
        lum = np.tile(x, (h, 1))
        array = np.stack([lum * 0.96, lum, lum * 0.87], axis=2)
        image = Image.fromarray((np.clip(array * level, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.rectangle((1000, 700, 1750, 2200), fill=(90, 115, 135))
        draw.ellipse((2450, 1000, 3150, 1700), fill=(160, 125, 100))
        path = args.root / "input" / f"synthetic-{i + 1}.jpg"
        if path.exists():
            raise RuntimeError("Smoke workspace must be fresh; refusing to overwrite samples")
        image.save(path, quality=95)
    originals = hashes(args.root / "input")
    report = pipeline.run()
    assert report["total"] == 3 and report["crop"]["exported_below_floor"] == 0
    assert all(r["status"] == "manual_review_required" for r in report["photos"])
    for record in report["photos"]:
        with Image.open(args.root / "output" / record["output"]) as output:
            assert max(output.size) >= 4000
        assert record["summary"] and record["revision_cycles"] <= 2
    first = report["photos"][0]
    manual = pipeline.override(first["id"], Override(crop=Crop(), color=Color(contrast=12)))
    assert manual["color"]["contrast"] == 12
    rerun = pipeline.run()
    assert rerun["photos"][0]["manual"] and rerun["photos"][0]["color"]["contrast"] == 12
    assert hashes(args.root / "input") == originals
    assert (args.root / "reports" / "gallery_report.json").exists()
    assert (args.root / "cache" / "contact-sheet-1.jpg").exists()
    print(
        f"PASS: {args.root}: 3 full-resolution exports, reports, cache, preserved originals and manual edits"
    )


if __name__ == "__main__":
    main()
