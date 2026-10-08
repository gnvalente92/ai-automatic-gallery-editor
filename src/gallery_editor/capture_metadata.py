"""Read factual capture metadata without deriving photographic intent from it."""

from io import BytesIO
from pathlib import Path

from PIL import ExifTags, Image

from .imaging import RAWS

_TAG_IDS = {
    "make": 271,
    "model": 272,
    "orientation": 274,
    "datetime": 306,
    "datetime_original": 36867,
    "exposure_time": 33434,
    "f_number": 33437,
    "iso": 34855,
    "exposure_bias_ev": 37380,
    "focal_length_mm": 37386,
    "white_balance": 41987,
    "lens_make": 42035,
    "lens_model": 42036,
    "focal_length_35mm": 41989,
}


def _text(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        value = ", ".join(map(str, value))
    result = str(value).replace("\x00", " ").strip()
    return result if result else None


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _ifd_values(exif):
    values = dict(exif.items())
    try:
        values.update(exif.get_ifd(34665))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return values


def _rawpy_fields(raw):
    lens = raw.lens
    other = raw.other
    result = {}
    for field, value in (
        ("lens_model", getattr(lens, "model", None)),
        ("lens_make", getattr(lens, "make", None)),
        ("focal_length_mm", getattr(other, "focal_length", None)),
        ("f_number", getattr(other, "aperture", None)),
        ("exposure_time", getattr(other, "shutter_speed", None)),
        ("iso", getattr(other, "iso_speed", None)),
        ("datetime_original", getattr(other, "timestamp", None)),
    ):
        if value is not None:
            result[field] = value
    return result


def _read_embedded_exif(path):
    """Fuji RAF stores standard EXIF in its embedded JPEG preview."""
    fallback_values = {}
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            fallback_values = _rawpy_fields(raw)
            thumbnail = raw.extract_thumb()
        if thumbnail.format == rawpy.ThumbFormat.JPEG:
            with Image.open(BytesIO(thumbnail.data)) as image:
                exif = image.getexif()
                return _ifd_values(exif), "embedded_jpeg_exif", fallback_values
    except (ImportError, OSError, ValueError, RuntimeError, AttributeError):
        pass
    return {}, "none", fallback_values


def extract_capture_metadata(path):
    """Return normalized, sourced camera settings; unavailable values stay absent."""
    path = Path(path)
    fallback_values = {}
    if path.suffix.lower() in RAWS:
        raw_values, source, fallback_values = _read_embedded_exif(path)
        # Embedded RAF EXIF is preferred, while LibRaw fills any fields that the
        # camera omitted from its preview metadata.
        if not raw_values and fallback_values:
            source = "libraw"
        elif raw_values and fallback_values:
            source = "embedded_jpeg_exif+libraw"
    else:
        try:
            with Image.open(path) as image:
                raw_values = _ifd_values(image.getexif())
            source = "file_exif" if raw_values else "none"
        except (OSError, ValueError):
            raw_values, source = {}, "none"

    by_name = {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in raw_values.items()}
    aliases = {name.casefold(): value for name, value in by_name.items()}
    fields = {}
    field_sources = {}
    for field, tag_id in _TAG_IDS.items():
        value = raw_values.get(tag_id)
        if value is None:
            value = aliases.get(field.casefold())
        value_source = source
        if value is None and field in fallback_values:
            value = fallback_values[field]
            value_source = "libraw"
        if value is not None:
            fields[field] = value
            field_sources[field] = value_source

    def text_field(name):
        value = _text(fields.get(name))
        if value:
            fields[name] = value
        else:
            fields.pop(name, None)

    for name in ("make", "model", "datetime", "datetime_original", "white_balance", "lens_make", "lens_model"):
        text_field(name)
    for name in ("exposure_time", "f_number", "exposure_bias_ev", "focal_length_mm", "focal_length_35mm"):
        if name in fields:
            number = _number(fields[name])
            if number is None:
                fields.pop(name)
            else:
                fields[name] = number
    if "iso" in fields:
        iso = _number(fields["iso"])
        if iso is None:
            fields.pop("iso")
        else:
            fields["iso"] = int(iso) if iso.is_integer() else iso

    return {
        "source": source,
        "fields": fields,
        "field_sources": field_sources,
        "missing_fields": sorted(set(_TAG_IDS) - set(fields)),
    }
