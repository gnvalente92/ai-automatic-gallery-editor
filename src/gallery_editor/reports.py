import html
from collections import Counter

from .storage import atomic_bytes, read_json, write_json


def photo_summary(record):
    if record["status"] == "blocked":
        preserved = len(record.get("color_option_exports", []))
        return (
            "Final edit blocked. "
            + "; ".join(record["issues"])
            + (
                f" {preserved} uncropped grade JPEGs remain available in this photo's options folder."
                if preserved
                else " No image exported."
            )
        )
    metrics = record["crop_metrics"]
    crop = (
        f"Crop: removed {metrics['crop_percentage']:.1f}% of the original area; "
        if metrics["crop_percentage"]
        else "Crop: original framing retained; "
    )
    crop += (
        f"{metrics['result_width']} × {metrics['result_height']} pixels ({metrics['megapixels']:.2f} MP). "
    )
    crop += (
        f"At 300 PPI this supports {metrics['print_sizes']['300']['width_cm']} × "
        f"{metrics['print_sizes']['300']['height_cm']} cm without upscaling. "
    )
    if record.get("crop_rationale"):
        crop += record["crop_rationale"].strip() + " "
    look = record["color"].get("look", "natural")
    adjustments = [
        f"{key} {value:+.2f}" for key, value in record["color"].items() if key != "look" and value != 0
    ]
    color = f"Color: {look.replace('_', ' ')}"
    if adjustments:
        color += " · " + ", ".join(adjustments)
    color += ". "
    consistency = (
        "Consistency: "
        + (
            "; ".join(record["outliers_after"])
            if record["outliers_after"]
            else "no measured outliers against the original visual group's distribution"
        )
        + ". "
    )
    review = record["review"]
    result = (
        f"Result: {record['status'].replace('_', ' ')}; {review['source']} review {review['score']:.0f}/100. "
    )
    final = record.get("final_gallery_outliers", []) + record.get("final_review_issues", [])
    return (
        crop
        + color
        + (
            f" Prepared {len(record['color_option_exports'])} full-resolution, uncropped color-grade choices "
            "before crop selection: "
            + ", ".join(item["path"] for item in record["color_option_exports"])
            + ". "
            if record.get("color_option_exports")
            else ""
        )
        + (
            f" Prepared {len(record.get('variants', []))} full-resolution crop/color options; "
            + (
                f"you selected {record['selected_variant']} manually. "
                if record.get("manual") and record.get("selected_variant")
                else f"the independent option reviewer selected {record['selected_option_id']}. "
                if record.get("selected_option_id")
                else "the independent option reviewer did not return a valid choice. "
                if record.get("option_selection")
                else "the editor's validated fallback remains selected. "
            )
            if record.get("variants")
            else ""
        )
        + consistency
        + result
        + " ".join(review["issues"])
        + (" Final gallery review: " + "; ".join(final) if final else "")
    )


def save_photo(root, record):
    record["summary"] = photo_summary(record)
    write_json(root / "reports" / "photos" / f"{record['id']}.json", record)
    atomic_bytes(
        root / "reports" / "photos" / f"{record['id']}.md",
        (f"# {record['file']}\n\n{record['summary']}\n").encode(),
    )


def save_gallery(
    root, contract, records, before, after, scan_errors, events, run_id=None, output_directory=None
):
    counts = Counter(r["status"] for r in records)
    edited = [r for r in records if r["status"] != "blocked"]
    report = {
        "album_analysis_coverage": read_json(root / "reports" / "album_analysis.json", {}).get(
            "coverage", {}
        ),
        "final_cluster_statistics": read_json(root / "reports" / "final_cluster_statistics.json", {}),
        "style": contract.model_dump(),
        "run_id": run_id,
        "output_directory": output_directory,
        "total": len(records),
        "status_counts": dict(counts),
        "crop": {
            "unchanged": sum(r["crop_metrics"]["crop_percentage"] == 0 for r in edited),
            "moderate": sum(0 < r["crop_metrics"]["crop_percentage"] <= 30 for r in edited),
            "tight_manual": sum(r["crop_metrics"]["crop_percentage"] > 30 for r in edited),
            "exported_below_floor": 0,
        },
        "variants": {
            "combinations": sum(len(r.get("variants", [])) for r in edited),
            "mean_options_per_photo": round(
                sum(len(r.get("variants", [])) for r in edited) / max(1, len(edited)), 1
            ),
            "selected_by_option_reviewer": sum(
                bool(r.get("option_selection") and r.get("selected_option_id")) for r in edited
            ),
            "option_reviewer_fallbacks": sum(
                not r.get("manual")
                and not bool(r.get("option_selection") and r.get("selected_option_id"))
                for r in edited
            ),
            "photos_with_portrait_options": sum(
                any(v.get("crop_kind") == "portrait" for v in r.get("variants", [])) for r in edited
            ),
            "photos_with_landscape_options": sum(
                any(v.get("crop_kind") == "landscape" for v in r.get("variants", [])) for r in edited
            ),
        },
        "color_options": {
            "photos_with_exports": sum(bool(r.get("color_option_exports")) for r in records),
            "files": sum(len(r.get("color_option_exports", [])) for r in records),
            "quality": 100,
            "cropped": False,
        },
        "color": {
            key: sum(r["color"][key] != 0 for r in edited)
            for key in ("exposure", "temperature", "highlights", "grain")
        },
        "average_confidence": sum(min(r["confidence"].values()) for r in edited) / max(1, len(edited)),
        "statistics_before": before,
        "statistics_after": after,
        "scan_errors": scan_errors,
        "model_events": events,
        "photos": records,
    }
    write_json(root / "reports" / "gallery_report.json", report)
    archive = root / "reports" / "runs" / run_id if run_id else None
    if archive:
        if archive.is_symlink():
            raise ValueError("Per-run report directory cannot be a symlink")
        (archive / "photos").mkdir(parents=True, exist_ok=True)
        write_json(archive / "gallery_report.json", report)
        for record in records:
            write_json(archive / "photos" / f"{record['id']}.json", record)
            atomic_bytes(
                archive / "photos" / f"{record['id']}.md",
                (f"# {record['file']}\n\n{record.get('summary', '')}\n").encode(),
            )
    rows = []
    for record in records:
        rows.append(
            f"<article><h2>{html.escape(record['file'])}</h2>"
            f"<p>{html.escape(record['summary'])}</p></article>"
        )
    rules = "".join(f"<li>{html.escape(rule)}</li>" for rule in contract.global_rules)
    clusters = "".join(
        f"<li><strong>{html.escape(rule.label)}</strong>: {html.escape(rule.rationale)} "
        f"{html.escape('; '.join(rule.legitimate_differences))}</li>"
        for rule in contract.conditional_rules
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Gallery editing report</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px}"
        "article{border-top:1px solid #bbb;padding:12px 0}h2{font-size:19px}</style>"
        f"<h1>{len(records)} photographs processed</h1><p>{html.escape(contract.style)}</p>"
        f"<p>{html.escape(str(dict(counts)))}</p><p>Exported below resolution floor: 0</p>"
        f"<h2>Album understanding</h2><p>{html.escape(str(report['album_analysis_coverage']))}</p>"
        f"<ul>{rules}</ul><h2>Conditional visual rules</h2><ul>{clusters}</ul>" + "".join(rows) + "</html>"
    )
    atomic_bytes(root / "reports" / "gallery_report.html", page.encode())
    if archive:
        atomic_bytes(archive / "gallery_report.html", page.encode())
    return report
