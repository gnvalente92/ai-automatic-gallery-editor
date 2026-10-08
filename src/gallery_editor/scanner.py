from collections import Counter
from contextlib import nullcontext

from PIL import ExifTags, Image

from .capture_metadata import extract_capture_metadata
from .imaging import RASTERS, RAWS, load_cached_source, preview, raw_available, save_image
from .storage import digest, file_hash, read_json, safe_path, write_json


def scan(settings, measure=None, progress=None):
    settings.initialize()
    root = settings.root
    photos, errors = [], []
    input_dir = settings.input_dir
    paths = sorted(input_dir.rglob("*"))
    candidates = [
        path
        for path in paths
        if not any(part.startswith(".") for part in path.relative_to(input_dir).parts)
        and path.is_file()
        and not path.is_symlink()
        and not any(part in {"output", "cache", "reports"} for part in path.relative_to(input_dir).parts[:-1])
        and path.suffix.lower() in RASTERS | RAWS
    ]
    for index, path in enumerate(candidates, 1):
        rel = path.relative_to(input_dir)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
            continue
        suffix = path.suffix.lower()
        if progress:
            progress("reading", rel.as_posix(), index, len(candidates))
        if suffix in RAWS and not raw_available():
            errors.append({"file": str(rel), "error": "RAW decoder unavailable; install the raw extra"})
            if progress:
                progress("error", rel.as_posix(), index, len(candidates))
            continue
        try:
            with measure("source hashing") if measure else nullcontext():
                content_hash = file_hash(path)
            photo_id = digest(str(rel))[:24]
            key = digest([content_hash, "decoder-srgb-v1"])
            metadata_path = safe_path(root / "cache", f"analysis/metadata-{key}.json")
            info = read_json(metadata_path)
            thumb = f"thumbnails/{key}.jpg"
            medium = f"previews/{key}.jpg"
            if (
                not isinstance(info, dict)
                or not (root / "cache" / medium).exists()
                or not (root / "cache" / thumb).exists()
            ):
                with measure("source decode and preview generation") if measure else nullcontext():
                    image = load_cached_source(path, root, content_hash)
                    metadata = {}
                    if suffix in RASTERS:
                        with Image.open(path) as original:
                            for k, v in original.getexif().items():
                                if isinstance(v, (str, int, float)):
                                    metadata[ExifTags.TAGS.get(k, str(k))] = str(v)[:500]
                    info = {"width": image.width, "height": image.height, "metadata": metadata}
                    save_image(preview(image, 360), root / "cache" / thumb)
                    save_image(preview(image), root / "cache" / medium)
                    write_json(metadata_path, info)
            # Upgrade existing metadata cache entries without decoding the image again.
            # RAW capture facts are read from the embedded preview/LibRaw, never from
            # or written back to the source file.
            if not isinstance(info, dict):
                info = {}
            if info.get("capture_metadata_version") != 1:
                info["capture_metadata"] = extract_capture_metadata(path)
                info["capture_metadata_version"] = 1
                write_json(metadata_path, info)
            photos.append(
                {
                    "id": photo_id,
                    "file": rel.as_posix(),
                    "hash": content_hash,
                    "type": suffix[1:].upper(),
                    **info,
                    "thumbnail": thumb,
                    "preview": medium,
                }
            )
            if progress:
                progress("ready", rel.as_posix(), index, len(candidates))
        except Exception as exc:
            errors.append({"file": str(rel), "error": str(exc)})
            if progress:
                progress("error", rel.as_posix(), index, len(candidates))
    result = {
        "photos": photos,
        "count": len(photos),
        "types": dict(Counter(p["type"] for p in photos)),
        "errors": errors,
        "input_directory": str(input_dir),
        "output_directory": str(settings.output_dir),
        "reports_directory": str(root / "reports"),
        "raw_available": raw_available(),
    }
    write_json(root / "cache" / "scan.json", result)
    return result
