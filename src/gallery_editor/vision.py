import cv2
import numpy as np
from PIL import Image

from .storage import read_json, write_json


def statistics(image):
    image = image.copy()
    image.thumbnail((512, 512))
    a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    lum = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    means = np.mean(a, axis=(0, 1))
    histogram = np.histogram(lum, bins=16, range=(0, 1))[0].astype(float)
    histogram /= max(1, histogram.sum())
    quantized = (a * 5).astype(np.int32).clip(0, 4)
    bins = quantized[:, :, 0] * 25 + quantized[:, :, 1] * 5 + quantized[:, :, 2]
    counts = np.bincount(bins.ravel(), minlength=125)
    palette = []
    for index in np.argsort(counts)[-5:][::-1]:
        if counts[index]:
            rgb = [((index // 25) + 0.5) / 5, ((index // 5 % 5) + 0.5) / 5, ((index % 5) + 0.5) / 5]
            palette.append({"rgb": rgb, "fraction": float(counts[index] / bins.size)})
    return {
        "exposure": float(np.median(lum)),
        "contrast": float(np.std(lum)),
        "saturation": float(np.median(hsv[:, :, 1])),
        "temperature": float(means[0] - means[2]),
        "white_balance": [float(x) for x in means],
        "shadows": float(np.mean(lum < 0.08)),
        "highlights": float(np.mean(lum > 0.97)),
        "sharpness": float(cv2.Laplacian((lum * 255).astype(np.uint8), cv2.CV_64F).var()),
        "luminance_histogram": histogram.tolist(),
        "dominant_colors": palette,
        "temperature_units": "mean red minus blue in sRGB; a relative proxy, not Kelvin",
    }


def analyze_cv(path, cache_path):
    cached = read_json(cache_path)
    if cached:
        return cached
    with Image.open(path) as image:
        image = image.convert("RGB")
        a = np.asarray(image)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        regions, skin = [], []
        for x, y, w, h in faces:
            # Size alone cannot distinguish a false positive from a close-up face.
            # Keep detections conservative until a stronger detector is available.
            # Protect the detected face plus modest head/chin margin; body/context is handled semantically.
            x0, y0 = max(0, x - w * 0.12), max(0, y - h * 0.18)
            x1, y1 = min(image.width, x + w * 1.12), min(image.height, y + h * 1.4)
            regions.append(
                {
                    "x": x0 / image.width,
                    "y": y0 / image.height,
                    "width": (x1 - x0) / image.width,
                    "height": (y1 - y0) / image.height,
                }
            )
            skin.append(np.mean(a[y : y + h, x : x + w], axis=(0, 1)).tolist())
        edges = cv2.Canny(gray, 60, 140)
        ys, xs = np.nonzero(edges)
        centroid = (
            [float(xs.mean() / image.width), float(ys.mean() / image.height)] if len(xs) else [0.5, 0.5]
        )
        result = {
            "statistics": statistics(image),
            "protected_regions": regions,
            "face_count": len(regions),
            "face_region_rgb": skin,
            "skin_tone_note": "Face-region RGB proxy, not semantic skin segmentation",
            "edge_centroid": centroid,
            "limitations": [
                "Frontal-face detector can miss faces",
                "No body, pose or object detector bundled",
                "Edge centroid is a composition proxy, not subject recognition",
            ],
        }
    write_json(cache_path, result)
    return result


METRICS = ("exposure", "contrast", "saturation", "temperature", "shadows", "highlights")


def gallery_statistics(analyses):
    if not analyses:
        return {}
    result = {}
    for key in METRICS:
        values = [a["statistics"][key] for a in analyses]
        median = float(np.median(values))
        result[key] = {
            "median": median,
            "mad": float(np.median(np.abs(np.array(values) - median))),
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    result["white_balance"] = {
        "median": np.median([a["statistics"]["white_balance"] for a in analyses], axis=0).tolist(),
        "values": [a["statistics"]["white_balance"] for a in analyses],
    }
    result["skin_tones"] = {
        "face_region_rgb": [rgb for a in analyses for rgb in a.get("face_region_rgb", [])],
        "method": "Face-region RGB proxy; unknown when no faces detected",
    }
    return result


def outliers(stats, gallery, look="natural"):
    issues = []
    for key, floor in (("exposure", 0.14), ("temperature", 0.12), ("saturation", 0.22), ("contrast", 0.12)):
        if look in {"monochrome", "warm_monochrome"} and key in {"temperature", "saturation"}:
            continue
        target = gallery.get(key)
        if target and abs(stats[key] - target["median"]) > max(floor, 3 * target["mad"]):
            direction = "higher" if stats[key] > target["median"] else "lower"
            issues.append(
                f"{key}: {stats[key]:.3f}, {direction} than reference median {target['median']:.3f}"
            )
    return issues
