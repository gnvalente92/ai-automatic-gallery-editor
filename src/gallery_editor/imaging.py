"""The only module that changes pixels. No model executes image operations."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .schemas import Color, Crop
from .storage import atomic_bytes

RAWS = {".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng"}
RASTERS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def raw_available():
    try:
        import rawpy  # noqa: F401

        return True
    except ImportError:
        return False


def load_image(path):
    if Path(path).suffix.lower() in RAWS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            return Image.fromarray(raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8))
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        profile = image.info.get("icc_profile")
        if profile:
            # Convert to the renderer's explicit sRGB working space, rather than silently discarding ICC.
            image = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        return image.convert("RGB")


def preview(image, edge=1280):
    result = image.copy()
    result.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return result


def load_cached_source(path, root, source_hash):
    """Reuse the oriented sRGB RAW decode across scan, grading and revisions."""
    if Path(path).suffix.lower() not in RAWS:
        return load_image(path)
    cached = Path(root) / "cache" / "decoded" / f"srgb-v1-{source_hash}.tiff"
    if cached.is_file():
        return load_image(cached)
    image = load_image(path)
    save_image(image, cached)
    return image


def save_image(image, path, quality=95):
    ext = Path(path).suffix.lower()
    fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}.get(ext, ext[1:].upper())
    stream = io.BytesIO()
    options = {"quality": quality, "subsampling": 0} if fmt == "JPEG" else {}
    if fmt == "WEBP":
        options = {"quality": quality}
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    image.save(stream, format=fmt, icc_profile=profile, **options)
    atomic_bytes(path, stream.getvalue())


def crop_box(crop: Crop, size):
    crop = Crop.model_validate(crop.model_dump())
    w, h = size
    x, y = round(crop.x * w), round(crop.y * h)
    right, bottom = min(w, round((crop.x + crop.width) * w)), min(h, round((crop.y + crop.height) * h))
    if x >= right or y >= bottom:
        raise ValueError("Crop rounds to an empty image")
    return x, y, right, bottom


def crop_metrics(crop, size):
    x, y, right, bottom = crop_box(crop, size)
    w, h = right - x, bottom - y
    prints = {
        str(ppi): {
            "width_inches": round(w / ppi, 2),
            "height_inches": round(h / ppi, 2),
            "width_cm": round(w / ppi * 2.54, 2),
            "height_cm": round(h / ppi * 2.54, 2),
        }
        for ppi in (300, 240, 200)
    }
    short, long = sorted((w, h))
    paper = {}
    for name, (a, b) in {"A4": (8.27, 11.69), "A3": (11.69, 16.54), "A2": (16.54, 23.39)}.items():
        ppi = min(short / a, long / b)
        paper[name] = "yes" if ppi >= 300 else "approximate" if ppi >= 200 else "no"
    return {
        "original_width": size[0],
        "original_height": size[1],
        "result_width": w,
        "result_height": h,
        "crop_percentage": round(100 * (1 - w * h / (size[0] * size[1])), 2),
        "megapixels": w * h / 1e6,
        "print_sizes": prints,
        "printability": paper,
        "digital": long >= 3000,
        "pixel_box": [x, y, right, bottom],
    }


def enforce_resolution(crop, size, settings):
    m = crop_metrics(crop, size)
    w, h = m["result_width"], m["result_height"]
    if (
        max(w, h) < settings.min_long_edge
        or w < settings.min_width
        or h < settings.min_height
        or w * h / 1e6 < settings.min_megapixels
    ):
        raise ValueError(
            f"Resolution floor violated: {w} × {h}; required long edge {settings.min_long_edge}, "
            f"width {settings.min_width}, height {settings.min_height}, MP {settings.min_megapixels}"
        )
    return m


def grade(image, color):
    c = Color.model_validate(color.model_dump())
    a = np.asarray(image, dtype=np.float32) / 255
    a *= 2**c.exposure
    a *= np.array([1 + c.temperature / 12000, 1 - c.tint / 300, 1 - c.temperature / 12000], np.float32)
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    shadows, highlights = (1 - np.clip(lum, 0, 1)) ** 2, np.clip(lum, 0, 1) ** 2
    a += (shadows * c.shadows / 300 + highlights * c.highlights / 300)[..., None]
    a += ((1 - np.clip(lum, 0, 1)) ** 4 * c.blacks / 400 + np.clip(lum, 0, 1) ** 4 * c.whites / 400)[
        ..., None
    ]
    a = (a - 0.5) * (1 + c.contrast / 100) + 0.5
    gray = (a @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    chroma = np.max(a, axis=2) - np.min(a, axis=2)
    factor = 1 + c.saturation / 100 + c.vibrance / 100 * (1 - np.clip(chroma, 0, 1))[..., None]
    a = gray + (a - gray) * factor
    if c.look == "monochrome":
        a = np.repeat(gray, 3, axis=2)
    elif c.look == "warm_monochrome":
        mono = np.repeat(gray, 3, axis=2)
        a = mono * np.array([1.035, 1.0, 0.94], dtype=np.float32)
    elif c.look == "cinematic":
        # Restrained cool shadows and warm highlights; preserve the image's
        # luminance structure so this remains a grade, not a synthetic filter.
        luminance = np.clip(gray[..., 0], 0, 1)
        shadows = (1 - luminance) ** 2
        highlights = luminance**2
        a[..., 0] += 0.035 * highlights - 0.012 * shadows
        a[..., 1] += 0.006 * highlights
        a[..., 2] += 0.028 * shadows - 0.012 * highlights
    if c.grain:
        a += np.random.default_rng(0).normal(0, c.grain / 1000, (*a.shape[:2], 1)).astype(np.float32)
    return Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8))


def render(image, crop, color, settings):
    enforce_resolution(crop, image.size, settings)
    return grade(image.crop(crop_box(crop, image.size)), color)


def crop_only(image, crop, settings):
    """Apply an already-rendered grade's crop while enforcing the same hard pixel floor."""
    enforce_resolution(crop, image.size, settings)
    return image.crop(crop_box(crop, image.size))
