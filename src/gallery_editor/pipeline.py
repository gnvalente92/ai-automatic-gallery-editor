import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw

from .agents import (
    color_agent,
    color_baseline,
    composition_references,
    crop_agent,
    option_reviewer,
    photo_agent,
    reference_guidance,
    reviewer_agent,
    soften,
)
from .album import AlbumAnalyzer, chunks, effective_contract
from .cropping import choose_crop, contains, hard_keep_zones, validate_manual
from .imaging import RAWS, crop_only, enforce_resolution, load_cached_source, preview, render, save_image
from .models import PROMPT_VERSION, LocalModel
from .performance import PerformanceProfile
from .reports import save_gallery, save_photo
from .scanner import scan
from .schemas import (
    AlbumContract,
    Analysis,
    Changes,
    Color,
    ConditionalStyle,
    Crop,
    CropDecision,
    GalleryReview,
    Override,
)
from .storage import digest, file_hash, read_json, safe_path, write_json
from .variants import (
    export_color_options,
    export_full_resolution_variants,
    generate_variants,
    option_contact_sheet,
)
from .vision import analyze_cv, gallery_statistics, outliers, statistics


class NoReadablePhotosError(RuntimeError):
    """Raised when the scan found no image the configured decoders can read."""


class Pipeline:
    def __init__(self, settings, model=None):
        self.settings = settings
        settings.initialize()
        self.root = settings.root
        self.output_root = Path(settings.output_dir).resolve()
        self.run_id = None
        self.run_prefix = None
        self.performance = None
        self.model = model or LocalModel(settings)
        if hasattr(self.model, "event_callback"):
            self.model.event_callback = self._model_event
        self.lock = threading.Lock()
        self.progress = {"state": "idle", "completed": 0, "total": 0, "error": None}
        self.scan_progress_callback = None
        self.album = None

    def stage(self, name, completed, total):
        previous = self.progress.get("stage")
        if previous and previous != name:
            self.progress.setdefault("finished_stages", []).append(previous)
        self.progress.update(stage=name, completed=completed, total=total)
        if self.performance:
            self.performance.stage(name)

    def _model_event(self, event):
        if self.performance:
            self.performance.model_event(event)

    def public_progress(self):
        result = dict(self.progress)
        if self.performance and result.get("state") in {"analyzing", "processing", "reviewing"}:
            result["elapsed_seconds"] = round(perf_counter() - self.performance.started, 1)
            if self.performance._stage_started is not None:
                result["stage_elapsed_seconds"] = round(perf_counter() - self.performance._stage_started, 1)
        return result

    def inventory(self):
        measure = self.performance.measure if self.performance else None
        return scan(self.settings, measure, self.scan_progress_callback)

    def measure(self, name):
        return self.performance.measure(name) if self.performance else nullcontext()

    def prepare(self):
        self.stage("Scanning photographs and generating previews", 0, 0)
        inventory = self.inventory()
        photos = inventory["photos"]
        self.stage("Analyzing all photographs", 0, len(photos))

        def analyze(p):
            return analyze_cv(
                self.root / "cache" / p["preview"],
                self.root / "cache" / "analysis" / f"cv-v4-{p['hash']}.json",
            )

        analyses = []
        with ThreadPoolExecutor(max_workers=self.settings.analysis_workers) as pool:
            for result in pool.map(analyze, photos):
                analyses.append(result)
                self.stage("Analyzing all photographs", len(analyses), len(photos))
        stats = gallery_statistics(analyses)
        return inventory, analyses, stats

    def propose_style(self):
        with self.lock:
            if hasattr(self.model, "begin_run"):
                self.model.begin_run()
            self.progress = {
                "state": "analyzing",
                "operation": "style",
                "completed": 0,
                "total": 0,
                "error": None,
            }
            self.performance = PerformanceProfile(self.root, "style")
            try:
                inventory, analyses, stats = self.prepare()
                style, self.album = AlbumAnalyzer(self.root, self.model, self.stage).analyze(
                    inventory["photos"], analyses, stats
                )
                write_json(self.root / "cache" / "album_style.json", style.model_dump())
                self.progress.update(state="complete")
                self.performance.finish()
                return style
            except Exception:
                self.progress.update(state="failed")
                self.performance.finish("failed")
                raise

    def run(self, style=None, theme=None):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Processing is already running")
        try:
            return self._run(style, theme)
        except Exception as exc:
            self.progress.update(state="failed", error=str(exc))
            if self.performance:
                self.performance.finish("failed")
            raise
        finally:
            self.settings.output_dir = self.output_root
            self.lock.release()

    def _activate_report_output(self, report):
        relative = report.get("output_directory")
        if not relative:
            # Legacy reports are readable, but new edits stay in a new run folder.
            return
        run_dir = safe_path(self.output_root, relative)
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError("The report's run output folder is missing or unsafe")
        self.settings.output_dir = run_dir
        self.run_prefix = Path(relative).as_posix()
        self.run_id = Path(relative).name

    def build_variants(self):
        """Create alternative previews from existing decisions, without rerunning album inference."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Processing is already running")
        try:
            report = read_json(self.root / "reports" / "gallery_report.json")
            if not report:
                raise ValueError("Process the gallery before generating crop and color alternatives")
            self._activate_report_output(report)
            self.performance = PerformanceProfile(self.root, "variants")
            self.progress = {
                "state": "processing",
                "operation": "variants",
                "completed": 0,
                "total": len(report["photos"]),
                "error": None,
            }
            records = report["photos"]
            self.stage("Generating crop and color alternatives", 0, len(records))
            for index, record in enumerate(records):
                self.progress["file"] = record["file"]
                self.performance.photo_start(record["file"])
                if record["status"] != "blocked":
                    source = safe_path(self.settings.input_dir, record["file"])
                    with self.measure("source integrity hash"):
                        current_hash = file_hash(source)
                    if current_hash != record["hash"]:
                        raise ValueError(f"Source changed since processing: {record['file']}")
                    preview_path = safe_path(self.root / "cache", record["original"]["preview"])
                    with self.measure("cached preview decode"):
                        with Image.open(preview_path) as cached_preview:
                            original_preview = cached_preview.convert("RGB").copy()
                    with self.measure("crop and color alternative previews"):
                        exported = {v["id"]: v for v in record.get("variants", [])}
                        record["variants"], record["unavailable_variants"] = generate_variants(
                            self.root,
                            record["original"],
                            original_preview,
                            record["cv"],
                            Analysis(**record["analysis"]),
                            self.settings,
                            Crop(**record["crop"]),
                            Color(**record["color"]),
                            source_size=(record["original"]["width"], record["original"]["height"]),
                            semantic_options=CropDecision(**record["crop_proposal"]).alternatives
                            if record.get("crop_proposal")
                            else (),
                        )
                        for variant in record["variants"]:
                            old = exported.get(variant["id"], {})
                            if old.get("crop") == variant["crop"] and old.get("color") == variant["color"]:
                                variant.update(
                                    {
                                        key: value
                                        for key, value in old.items()
                                        if key.startswith("output") or key == "full_resolution_exported"
                                    }
                                )
                    record["selected_variant"] = record.get("selected_variant")
                    save_photo(self.root, record)
                self.performance.photo_end()
                self.stage("Generating crop and color alternatives", index + 1, len(records))
            result = save_gallery(
                self.root,
                AlbumContract(**report["style"]),
                records,
                report["statistics_before"],
                report["statistics_after"],
                report["scan_errors"],
                report["model_events"],
                run_id=report.get("run_id"),
                output_directory=report.get("output_directory"),
            )
            self.progress.update(state="complete")
            self.performance.finish()
            return result
        except Exception as exc:
            self.progress.update(state="failed", error=str(exc))
            if self.performance:
                self.performance.finish("failed")
            raise
        finally:
            self.settings.output_dir = self.output_root
            self.lock.release()

    def _run(self, style=None, theme=None):
        self.progress = {
            "state": "analyzing",
            "operation": "process",
            "completed": 0,
            "total": 0,
            "error": None,
        }
        self.performance = PerformanceProfile(self.root, "process")
        self.run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")
        self.run_prefix = f"runs/{self.run_id}"

        # Scan and decode before creating a run folder or invoking any album/model
        # agents. A gallery with no readable images must fail visibly, rather than
        # producing a successful-looking empty report.
        inventory, analyses, stats = self.prepare()
        photos = inventory["photos"]
        if not photos:
            errors = inventory.get("errors", [])
            if errors:
                details = "; ".join(
                    f"{item.get('file', 'unknown file')}: {item.get('error', 'could not read image')}"
                    for item in errors[:5]
                )
                if len(errors) > 5:
                    details += f"; and {len(errors) - 5} more scan errors"
                raise NoReadablePhotosError(
                    f"No readable photographs found in {self.settings.input_dir}. Scan errors: {details}"
                )
            raise NoReadablePhotosError(
                f"No supported photographs found in {self.settings.input_dir}. "
                "Add JPG, JPEG, PNG, WEBP, TIFF, or readable RAW files and run again."
            )

        runs_dir = self.output_root / "runs"
        if runs_dir.is_symlink():
            raise ValueError("Output runs directory cannot be a symlink")
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = safe_path(self.output_root, self.run_prefix)
        if run_dir.is_symlink():
            raise ValueError("Run output directory cannot be a symlink")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "options").mkdir()
        (run_dir / "final").mkdir()
        self.settings.output_dir = run_dir
        if hasattr(self.model, "begin_run"):
            self.model.begin_run()
        elif hasattr(self.model, "events"):
            self.model.events.clear()
        contract, self.album = AlbumAnalyzer(self.root, self.model, self.stage).analyze(
            photos, analyses, stats, style, theme
        )
        write_json(self.root / "cache" / "album_style.json", contract.model_dump())
        color_pass = {}
        self.progress.update(state="processing", total=len(photos))
        color_stage = "Grading photographs and exporting per-photo color options"
        self.stage(color_stage, 0, len(photos))
        for index, (photo, cv) in enumerate(zip(photos, analyses)):
            self.progress["file"] = photo["file"]
            self.stage(color_stage, index, len(photos))
            context, cluster_stats, cluster_contract, neighbors = self.photo_context(photo, photos, contract)
            try:
                color_pass[photo["id"]] = self.prepare_color_options(
                    photo, cv, cluster_stats, cluster_contract, neighbors, context
                )
            except Exception as exc:
                color_pass[photo["id"]] = {"error": str(exc)}
            self.stage(color_stage, index + 1, len(photos))

        self.progress.update(state="processing", completed=0, total=len(photos))
        crop_stage = "Choosing crops, rendering and reviewing photographs"
        self.stage(crop_stage, 0, len(photos))
        records = []
        for index, (photo, cv) in enumerate(zip(photos, analyses)):
            self.progress["file"] = photo["file"]
            self.stage(crop_stage, index, len(photos))
            self.performance.photo_start(photo["file"])
            context, cluster_stats, cluster_contract, neighbors = self.photo_context(photo, photos, contract)
            try:
                prepared = color_pass[photo["id"]]
                if prepared.get("error"):
                    raise ValueError("Color grading phase failed: " + prepared["error"])
                record = self.process_photo(
                    photo,
                    cv,
                    cluster_stats,
                    cluster_contract,
                    neighbors,
                    album_context=context,
                    analysis_override=prepared["analysis"],
                    color_override=Color(**prepared["color"]),
                    color_confidence=prepared["color_confidence"],
                    color_rationale=prepared["color_rationale"],
                    pregraded_image_path=prepared.get("pregraded_image_path"),
                    color_option_exports=prepared["color_option_exports"],
                )
            except Exception as exc:
                previous = read_json(self.root / "reports" / "photos" / f"{photo['id']}.json", {})
                record = {
                    "id": photo["id"],
                    "file": photo["file"],
                    "hash": photo["hash"],
                    "status": "blocked",
                    "issues": [str(exc)],
                    "original": photo,
                    "output_hash": previous.get("output_hash"),
                    "previous_output": previous.get("output"),
                    "color_option_exports": color_pass.get(photo["id"], {}).get("color_option_exports", []),
                }
            save_photo(self.root, record)
            records.append(record)
            self.performance.photo_end()
            self.progress["completed"] += 1
            self.stage(crop_stage, index + 1, len(photos))
        self.progress.update(state="reviewing")
        self.deep_final_review(records, contract)
        after = gallery_statistics(
            [
                {"statistics": r["statistics_after"], "face_region_rgb": r.get("face_region_rgb_after", [])}
                for r in records
                if "statistics_after" in r
            ]
        )
        self.final_consistency(records, after)
        report = save_gallery(
            self.root,
            contract,
            records,
            stats,
            after,
            inventory["errors"],
            getattr(self.model, "events", []),
            run_id=self.run_id,
            output_directory=self.run_prefix,
        )
        self.contact_sheet(records)
        self.progress.update(state="complete")
        self.performance.finish(
            "complete_with_errors" if report["status_counts"].get("blocked") else "complete"
        )
        archive = self.root / "reports" / "runs" / self.run_id
        for name in ("performance", "album_analysis", "final_gallery_review", "final_cluster_statistics"):
            data = read_json(self.root / "reports" / f"{name}.json")
            if data is not None:
                write_json(archive / f"{name}.json", data)
        return report

    def photo_context(self, photo, photos, contract):
        group_id = self.album["membership"][photo["id"]]
        group = next(g for g in self.album["clusters"] if g["id"] == group_id)
        rule = next(
            (r for r in contract.conditional_rules if r.cluster_id == group_id),
            ConditionalStyle(cluster_id=group_id, label=group["label"]),
        )
        related = [p for p in photos if p["id"] in group["representatives"] and p["id"] != photo["id"]][:3]
        context = {
            "global_contract": contract.model_dump(),
            "user_theme": self.album.get("user_theme"),
            "cluster_id": group_id,
            "cluster_label": group["label"],
            "cluster_style": rule.model_dump(),
            "cluster_statistics": group["statistics"],
            "gallery_statistics": self.album["gallery_statistics"],
            "original_scene_analysis": self.album["scenes"][photo["id"]],
            "related_photos": [p["id"] for p in related],
        }
        return (
            context,
            group["statistics"],
            effective_contract(contract, rule),
            [self.root / "cache" / p["preview"] for p in related],
        )

    def prepare_color_options(self, photo, cv, gallery, contract, neighbors, album_context):
        """Analyze and persist uncropped grade choices before any crop call starts."""
        override_path = self.root / "cache" / "overrides" / f"{photo['id']}.json"
        saved = read_json(override_path)
        if saved and saved["hash"] != photo["hash"]:
            raise ValueError(
                "Source changed since manual edit; reset explicitly before processing this photograph"
            )
        applied = Override.model_validate(saved["override"]) if saved else None
        cache_path = self.root / "cache" / "analysis" / f"color-pass-{photo['id']}.json"
        model_settings = getattr(self.model, "settings", None)
        cache_key = digest(
            [
                # Includes semantic photo analysis as well as the grade. Bump this
                # whenever analysis/crop guidance changes so old untyped boxes cannot
                # silently bypass the current subject-vs-context prompt.
                "color-options-pass-v7-reviewer-selection",
                PROMPT_VERSION,
                photo["hash"],
                photo.get("capture_metadata", {}),
                cv,
                gallery,
                contract.model_dump(),
                album_context,
                applied.model_dump() if applied else None,
                getattr(model_settings, "vision_model", ""),
                getattr(model_settings, "local_model_endpoint", ""),
                getattr(model_settings, "model_context_tokens", 0),
                getattr(model_settings, "model_max_tokens", 0),
                self.settings.min_long_edge,
                self.settings.min_width,
                self.settings.min_height,
                self.settings.min_megapixels,
            ]
        )
        cached = read_json(cache_path)
        reusable = cached and (
            (
                cached.get("analysis", {}).get("confidence", 0) > 0
                and cached.get("color_confidence", 0) >= 0.65
            )
            or (isinstance(self.model, LocalModel) and not self.settings.vision_model)
        )
        if reusable and cached.get("cache_key") == cache_key:
            files_exist = all(
                safe_path(self.settings.output_dir, option["path"]).is_file()
                and file_hash(safe_path(self.settings.output_dir, option["path"])) == option["output_hash"]
                for option in cached.get("color_option_exports", [])
            )
            if files_exist and len(cached.get("color_option_exports", [])) == 3:
                return cached
        analysis = photo_agent(self.model, photo, self.root, cv, album_context)
        decision = (
            None
            if applied
            else color_agent(self.model, photo, self.root, cv, gallery, contract, album_context, neighbors)
        )
        color = (
            applied.color
            if applied
            else decision.color
            if decision
            else color_baseline(cv, gallery, contract, album_context)
        )
        if not applied:
            color.grain = contract.grain
        color_option_exports = []
        pregraded_image_path = None
        source = safe_path(self.settings.input_dir, photo["file"])
        with self.measure("source integrity hash"):
            if file_hash(source) != photo["hash"]:
                raise ValueError("Source changed during processing; rescan required")
        with self.measure("full-resolution source decode for per-photo options"):
            original = load_cached_source(source, self.root, photo["hash"])
        with self.measure("full-resolution uncropped color-option exports"):
            color_option_exports = export_color_options(
                self.settings.output_dir, photo, original, self.settings, color
            )
        pregraded = next((item for item in color_option_exports if item["id"] == "album"), None)
        if pregraded:
            pregraded_image_path = str(self.root / "cache" / pregraded["render_source"])
        data = {
            "cache_key": cache_key,
            "source_hash": photo["hash"],
            "analysis": analysis.model_dump(),
            "color": color.model_dump(),
            "color_confidence": decision.confidence if decision else 0.9 if applied else 0.45,
            "color_rationale": decision.rationale
            if decision
            else "Preserved manual color edit"
            if applied
            else ("Measured conservative color fallback within the album contract"),
            "color_option_exports": color_option_exports,
            "pregraded_image_path": pregraded_image_path,
        }
        write_json(cache_path, data)
        return data

    def process_photo(
        self,
        photo,
        cv,
        gallery,
        contract,
        neighbors,
        manual=None,
        reset=False,
        selected_variant=None,
        analysis_override=None,
        color_override=None,
        color_confidence=None,
        color_rationale=None,
        pregraded_image_path=None,
        color_option_exports=None,
        album_context=None,
        existing=None,
        final_changes=None,
    ):
        size = (photo["width"], photo["height"])
        enforce_resolution(Crop(), size, self.settings)
        override_path = self.root / "cache" / "overrides" / f"{photo['id']}.json"
        saved = read_json(override_path)
        if saved and saved["hash"] != photo["hash"] and not reset:
            raise ValueError(
                "Source changed since manual edit; reset explicitly before processing this photograph"
            )
        applied = manual or (Override.model_validate(saved["override"]) if saved and not reset else None)
        analysis = (
            Analysis(**existing["analysis"])
            if existing
            else (
                Analysis(**analysis_override)
                if analysis_override
                else photo_agent(self.model, photo, self.root, cv, album_context)
            )
        )
        color_decision = None
        if existing:
            color = final_changes.color or Color(**existing["color"])
            color_confidence = existing["confidence"].get("color_confidence", 0.45)
        elif color_override is not None:
            color = Color.model_validate(color_override)
        elif applied:
            color = applied.color
        else:
            color_decision = color_agent(
                self.model, photo, self.root, cv, gallery, contract, album_context, neighbors
            )
            color = (
                color_decision.color
                if color_decision
                else color_baseline(cv, gallery, contract, album_context)
            )
            color_confidence = color_decision.confidence if color_decision else 0.45
        if not applied:
            color.grain = contract.grain

        source = safe_path(self.settings.input_dir, photo["file"])
        with self.measure("source integrity hash"):
            if file_hash(source) != photo["hash"]:
                raise ValueError("Source changed during processing; rescan required")
        original = None
        if pregraded_image_path:
            with Image.open(pregraded_image_path) as image:
                original = image.convert("RGB").copy()
            if original.size != size:
                raise ValueError("Full-resolution color option does not match the source dimensions")
        else:
            with self.measure("full-resolution image decode"):
                original = load_cached_source(source, self.root, photo["hash"])

        if color_option_exports is None:
            color_option_exports = existing.get("color_option_exports", []) if existing else []
            if not existing:
                with self.measure("full-resolution uncropped color-option exports"):
                    color_option_exports = export_color_options(
                        self.settings.output_dir, photo, original, self.settings, color
                    )
                base_export = next((item for item in color_option_exports if item["id"] == "album"), None)
                if base_export:
                    pregraded_image_path = str(self.root / "cache" / base_export["render_source"])
                    with Image.open(pregraded_image_path) as image:
                        original = image.convert("RGB").copy()

        proposal = None
        if not existing and not applied and analysis.confidence >= 0.65:
            proposal = crop_agent(
                self.model,
                photo,
                self.root,
                analysis,
                cv,
                contract,
                self.settings,
                album_context,
                neighbors,
            )
        crop, candidates = choose_crop(size, self.settings, cv, analysis, contract, proposal)
        write_json(self.root / "cache" / "crop_candidates" / f"{photo['id']}.json", candidates)
        crop_rationale = (
            proposal.rationale
            if proposal and crop == proposal.crop
            else (
                existing.get("crop_rationale")
                if existing
                else "No confident semantic crop proposal; original framing retained conservatively"
            )
        )
        if proposal and crop != proposal.crop:
            rejected = next((c for c in candidates if c["crop"] == proposal.crop.model_dump()), None)
            crop_rationale = (
                "AI proposal was not applied: "
                + ((rejected or {}).get("rejection") or "confidence or conservative candidate selection")
                + ". Selected validated framing instead."
            )
        elif not proposal and crop != Crop() and not existing:
            crop_rationale = "Selected a measured composition candidate; semantic crop proposal unavailable, review required"
        initial_ai = (
            existing.get("initial_ai_decision")
            if existing and existing.get("initial_ai_decision")
            else {
                "crop": crop.model_dump(),
                "crop_rationale": crop_rationale,
                "color": color.model_dump(),
            }
        )
        if applied:
            crop, color = applied.crop, applied.color
            validate_manual(crop, size, self.settings, applied.aspect_ratio)
            crop_rationale = "User-selected framing"
        if existing:
            crop = final_changes.crop or Crop(**existing["crop"])
            try:
                m = enforce_resolution(crop, size, self.settings)
                protected = hard_keep_zones(cv["protected_regions"], analysis.protected_regions)
                if m["crop_percentage"] > 30 or any(not contains(crop, r) for r in protected):
                    raise ValueError("Unsafe final review crop")
                if analysis.confidence < 0.65 and crop != Crop():
                    raise ValueError("Semantic confidence too low")
            except ValueError:
                crop = Crop(**existing["crop"])
                crop_rationale = "Final reviewer crop rejected; retained the last validated framing"
            else:
                if crop != Crop(**existing["crop"]):
                    crop_rationale = (
                        "Crop revised after final gallery review; see the recorded gallery-review issues"
                    )
        option_selection = (existing or {}).get("option_selection")
        selected_option_id = selected_variant or (existing or {}).get("selected_option_id")
        preview_path = safe_path(self.root / "cache", photo["preview"])
        with self.measure("generate validated per-photo options"):
            with Image.open(preview_path) as source_preview:
                variants, unavailable_variants = generate_variants(
                    self.root,
                    photo,
                    source_preview.convert("RGB"),
                    cv,
                    analysis,
                    self.settings,
                    crop,
                    color,
                    source_size=size,
                    semantic_options=proposal.alternatives
                    if proposal
                    else (
                        CropDecision(**existing["crop_proposal"]).alternatives
                        if existing and existing.get("crop_proposal")
                        else ()
                    ),
                )
        if not existing and not applied and variants:
            sheet = option_contact_sheet(self.root, photo, variants)
            if sheet:
                option_selection = option_reviewer(
                    self.model,
                    photo,
                    self.root,
                    variants,
                    sheet,
                    contract,
                    album_context,
                    analysis,
                    neighbors,
                )
            selected = next(
                (
                    option
                    for option in variants
                    if option_selection and option["id"] == option_selection.selected_option_id
                ),
                None,
            )
            if selected:
                crop = Crop.model_validate(selected["crop"])
                color = Color.model_validate(selected["color"])
                selected_option_id = selected["id"]
                crop_rationale = "Selected by the independent option reviewer: " + option_selection.rationale
                selected_export = next(
                    (
                        item
                        for item in color_option_exports
                        if digest(Color.model_validate(item["color"])) == digest(color)
                    ),
                    None,
                )
                if selected_export and selected_export.get("render_source"):
                    pregraded_image_path = str(self.root / "cache" / selected_export["render_source"])
                    with Image.open(pregraded_image_path) as image:
                        original = image.convert("RGB").copy()
                else:
                    pregraded_image_path = None
                    with self.measure("decode source for selected option"):
                        original = load_cached_source(source, self.root, photo["hash"])
                initial_ai = {
                    "crop": crop.model_dump(),
                    "crop_rationale": crop_rationale,
                    "color": color.model_dump(),
                }
        edited_path = self.root / "cache" / "previews" / f"edited-{photo['id']}.jpg"
        history = list(existing["review_history"]) if existing else []
        references = composition_references(self.root, photo)
        protected = hard_keep_zones(cv["protected_regions"], analysis.protected_regions)
        start_cycle = existing["revision_cycles"] + 1 if existing else 0
        render_uses_pregraded = bool(pregraded_image_path)
        for cycle in range(start_cycle, 3):  # one shared budget, including final gallery revisions
            with self.measure("full-resolution render"):
                edited = (
                    crop_only(original, crop, self.settings)
                    if render_uses_pregraded
                    else render(original, crop, color, self.settings)
                )
            with self.measure("edited preview and statistics"):
                save_image(preview(edited), edited_path)
                after = statistics(edited)
                final_cv = analyze_cv(
                    edited_path,
                    self.root / "cache" / "analysis" / f"cv-v4-edited-{file_hash(edited_path)}.json",
                )
            context = {
                "file": photo["file"],
                "image_order": [
                    "source preview",
                    "edited preview",
                    *["related gallery image"] * len(neighbors),
                    *["optional user visual example"] * len(references),
                ],
                "crop": crop.model_dump(),
                "color": color.model_dump(),
                "contract": contract.model_dump(),
                "crop_metrics": enforce_resolution(crop, size, self.settings),
                "analysis": analysis.model_dump(),
                "statistics_before": cv["statistics"],
                "statistics_after": after,
                "manual": applied is not None,
                "album_context": album_context,
                "user_composition_reference": reference_guidance(references),
                "reference_review_rule": (
                    "Use supplied examples only to infer broad user preferences; judge this image on its own "
                    "story and scene, and never demand an unsafe or exact match."
                    if references
                    else None
                ),
                "final_gallery_revision": existing is not None,
            }
            review = reviewer_agent(
                self.model,
                self.root / "cache" / photo["preview"],
                edited_path,
                neighbors,
                context,
                cv["statistics"],
                after,
                gallery,
                analysis,
                references,
            )
            history.append(
                {
                    "cycle": cycle,
                    "crop": crop.model_dump(),
                    "color": color.model_dump(),
                    "review": review.model_dump(),
                }
            )
            if applied or not review.needs_revision or cycle == 2:
                break
            suggested = review.suggested_changes
            new_crop = suggested.crop or crop
            try:
                m = enforce_resolution(new_crop, size, self.settings)
                if m["crop_percentage"] > 30 or any(not contains(new_crop, r) for r in protected):
                    raise ValueError("Reviewer suggested an unsafe crop")
                if analysis.confidence < 0.65 and new_crop != Crop():
                    raise ValueError("Insufficient semantic confidence")
            except ValueError:
                new_crop = crop
            new_color = suggested.color or (color if suggested.crop is not None else soften(color))
            if new_color != color and render_uses_pregraded:
                with self.measure("full-resolution image decode for color revision"):
                    original = load_cached_source(source, self.root, photo["hash"])
                render_uses_pregraded = False
            if new_crop != crop:
                crop_rationale = "Crop adjusted after reviewer feedback: " + "; ".join(review.issues)
            crop, color = new_crop, new_color
            color.grain = contract.grain
        confidence = (
            dict(existing["confidence"])
            if existing
            else {
                "crop_confidence": min(proposal.confidence, analysis.confidence)
                if proposal and initial_ai["crop"] == proposal.crop.model_dump()
                else 0,
                "color_confidence": color_confidence if color_confidence is not None else 0.45,
                "review_confidence": review.confidence,
            }
        )
        confidence["review_confidence"] = review.confidence
        if review.approved and min(confidence.values()) < 0.65:
            review.approved = False
            review.issues.append("At least one editing decision has low confidence")
        status = "approved" if review.approved else "manual_review_required"
        # Preserve directory hierarchy to prevent duplicate-basename collisions; RAW exports get a suffix.
        relative = photo["file"] + ".jpg" if Path(photo["file"]).suffix.lower() in RAWS else photo["file"]
        dest = safe_path(self.settings.output_dir / "final", relative)
        if not dest.resolve().is_relative_to(self.settings.output_dir):
            raise ValueError("Output directory escapes output/")
        # Never overwrite an unrelated pre-existing output. Our own prior export is identified by checksum.
        old = read_json(self.root / "reports" / "photos" / f"{photo['id']}.json", {})
        if dest.exists() and file_hash(dest) != old.get("output_hash"):
            raise ValueError(
                "Output file is not the previous generated export; move it aside before processing"
            )
        enforce_resolution(crop, original.size, self.settings)  # final export boundary
        with self.measure("export encode and write"):
            save_image(edited, dest, quality=100)
        with self.measure("full-resolution crop and grade option exports"):
            variants = export_full_resolution_variants(
                self.settings.output_dir,
                self.run_prefix or "",
                photo,
                original,
                variants,
                self.settings,
                color_option_exports,
            )
        record = {
            "id": photo["id"],
            "file": photo["file"],
            "hash": photo["hash"],
            "original": photo,
            "output": dest.relative_to(self.output_root).as_posix(),
            "output_hash": file_hash(dest),
            "edited_preview": edited_path.relative_to(self.root / "cache").as_posix(),
            "status": status,
            "manual": applied is not None,
            "crop": crop.model_dump(),
            "crop_rationale": crop_rationale,
            "crop_proposal": proposal.model_dump() if proposal else (existing or {}).get("crop_proposal"),
            "crop_candidates": (existing or {}).get("crop_candidates", candidates),
            "crop_metrics": enforce_resolution(crop, size, self.settings),
            "color": color.model_dump(),
            "color_rationale": color_rationale
            or (color_decision.rationale if color_decision else "Color selected before crop processing"),
            "color_option_exports": color_option_exports,
            "variants": variants,
            "unavailable_variants": unavailable_variants,
            "selected_variant": selected_variant
            or (saved.get("selected_variant") if saved and applied and not reset else None),
            "selected_option_id": selected_option_id,
            "option_selection": (
                option_selection.model_dump()
                if hasattr(option_selection, "model_dump")
                else option_selection
            ),
            "review": review.model_dump(),
            "confidence": confidence,
            "analysis": analysis.model_dump(),
            "cv": cv,
            "album_style": contract.model_dump(),
            "album_context": album_context,
            "cluster_id": album_context["cluster_id"] if album_context else None,
            "ai_version": (saved.get("ai_version", initial_ai) if saved else initial_ai)
            if applied
            else {"crop": crop.model_dump(), "color": color.model_dump()},
            "initial_ai_decision": initial_ai,
            "statistics_before": cv["statistics"],
            "statistics_after": after,
            "face_region_rgb_after": final_cv["face_region_rgb"],
            "outliers_before": outliers(cv["statistics"], gallery),
            "outliers_after": outliers(after, gallery, look=color.look),
            "revision_cycles": len(history) - 1,
            "review_history": history,
        }
        write_json(self.root / "cache" / "reviews" / f"{photo['id']}.json", history)
        return record

    def select_variant(self, photo_id, variant_id):
        report = read_json(self.root / "reports" / "gallery_report.json")
        if not report:
            raise ValueError("Process the gallery before selecting an edit variant")
        record = next((item for item in report["photos"] if item["id"] == photo_id), None)
        if not record or record["status"] == "blocked":
            raise ValueError("Unknown or blocked photograph")
        variant = next((item for item in record.get("variants", []) if item["id"] == variant_id), None)
        if not variant:
            raise ValueError("Unknown edit variant")
        return self.override(
            photo_id,
            Override(crop=Crop(**variant["crop"]), color=Color(**variant["color"])),
            source_variant=variant_id,
        )

    def override(self, photo_id, override=None, reset=False, source_variant=None):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Wait for processing to finish before editing")
        try:
            self.performance = PerformanceProfile(
                self.root, "variant_selection" if source_variant else "manual_edit"
            )
            self.progress = {
                "state": "processing",
                "operation": self.performance.operation,
                "completed": 0,
                "total": 1,
                "error": None,
            }
            self.stage("Rendering and reviewing selected edit", 0, 1)
            self.performance.photo_start(photo_id)
            report = read_json(self.root / "reports" / "gallery_report.json")
            if not report:
                raise ValueError("Process the gallery before making manual edits")
            self._activate_report_output(report)
            old = next((r for r in report["photos"] if r["id"] == photo_id), None)
            if old is None:
                raise ValueError("Unknown photograph")
            if old["status"] == "blocked":
                raise ValueError("Resolve the blocked photograph before editing")
            if reset:
                override = Override(**old["ai_version"])
            photo = old["original"]
            validate_manual(
                override.crop, (photo["width"], photo["height"]), self.settings, override.aspect_ratio
            )
            related = [
                self.root / "cache" / r["edited_preview"]
                for r in report["photos"]
                if r.get("cluster_id") == old.get("cluster_id")
                and r["id"] != photo_id
                and r.get("edited_preview")
            ][:3]
            record = self.process_photo(
                photo,
                old["cv"],
                old.get("album_context", {}).get("cluster_statistics", report["statistics_before"]),
                AlbumContract(**old["album_style"]),
                related,
                manual=override,
                reset=reset,
                selected_variant=source_variant,
                analysis_override=old["analysis"],
                album_context=old.get("album_context"),
                color_option_exports=old.get("color_option_exports", []),
            )
            record["ai_version"] = old["ai_version"]
            record["manual"] = not reset
            record["selected_variant"] = source_variant
            record["option_selection"] = old.get("option_selection")
            record["selected_option_id"] = old.get("selected_option_id")
            save_photo(self.root, record)
            override_path = self.root / "cache" / "overrides" / f"{photo_id}.json"
            write_json(
                override_path,
                {
                    "hash": photo["hash"],
                    "override": override.model_dump(),
                    "ai_version": old["ai_version"],
                    "selected_variant": source_variant,
                },
            )
            if reset:
                override_path.unlink(missing_ok=True)
            events_path = self.root / "reports" / "feedback.json"
            events = read_json(events_path, [])
            events.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "photo": photo_id,
                    "source_hash": photo["hash"],
                    "action": "reset" if reset else ("select_variant" if source_variant else "override"),
                    "selected_variant": source_variant,
                    "ai_decision": old["ai_version"],
                    "previous": {"crop": old["crop"], "color": old["color"]},
                    "user_modification": override.model_dump(),
                    "final": {"crop": record["crop"], "color": record["color"]},
                    "photo_context": old["analysis"],
                    "album_style": old["album_style"],
                }
            )
            write_json(events_path, events)
            records = [record if r["id"] == photo_id else r for r in report["photos"]]
            after = gallery_statistics(
                [
                    {
                        "statistics": r["statistics_after"],
                        "face_region_rgb": r.get("face_region_rgb_after", []),
                    }
                    for r in records
                    if "statistics_after" in r
                ]
            )
            self.final_consistency(records, after)
            final_review = read_json(self.root / "reports" / "final_gallery_review.json", {})
            final_review.setdefault("manual_edits_since_album_review", []).append(photo_id)
            write_json(self.root / "reports" / "final_gallery_review.json", final_review)
            save_gallery(
                self.root,
                AlbumContract(**report["style"]),
                records,
                report["statistics_before"],
                after,
                report["scan_errors"],
                report["model_events"],
                run_id=report.get("run_id"),
                output_directory=report.get("output_directory"),
            )
            self.contact_sheet(records)
            self.performance.photo_end()
            self.progress.update(state="complete", completed=1)
            self.performance.finish()
            return record
        except Exception as exc:
            self.progress.update(state="failed", error=str(exc))
            if self.performance:
                self.performance.finish("failed")
            raise
        finally:
            self.settings.output_dir = self.output_root
            self.lock.release()

    def final_consistency(self, records, gallery):
        """Audit the delivered gallery distribution, including manually edited photographs."""
        clusters = {}
        for record in records:
            if record["status"] != "blocked":
                clusters.setdefault(record.get("cluster_id"), []).append(
                    {
                        "statistics": record["statistics_after"],
                        "face_region_rgb": record.get("face_region_rgb_after", []),
                    }
                )
        cluster_stats = {key: gallery_statistics(values) for key, values in clusters.items()}
        self.stage("Checking final outliers and resolution", 0, len(records))
        for index, record in enumerate(records):
            if record["status"] == "blocked":
                continue
            issues = outliers(
                record["statistics_after"],
                cluster_stats[record.get("cluster_id")],
                look=record["color"].get("look", "natural"),
            )
            enforce_resolution(
                Crop(**record["crop"]),
                (record["original"]["width"], record["original"]["height"]),
                self.settings,
            )
            record["final_gallery_outliers"] = issues
            if issues:
                record["review"]["approved"] = False
                record["status"] = "manual_review_required"
            save_photo(self.root, record)
            self.stage("Checking final outliers and resolution", index + 1, len(records))
        write_json(self.root / "reports" / "final_cluster_statistics.json", cluster_stats)

    def deep_final_review(self, records, contract):
        """Inspect all final images within groups, then compare groups across the album.

        Numeric outliers alone are review flags, not proof that intentional lighting must be changed.
        Only actionable vision requests or introduced clipping can trigger a bounded revision.
        """
        usable = {r["id"]: r for r in records if r["status"] != "blocked"}
        requests, observations = {}, []
        completed = 0
        self.stage("Reviewing crop, color and skin-tone consistency within clusters", 0, len(usable))

        def accept_review(result, allowed, scope):
            observations.append({"scope": scope, "review": result.model_dump() if result else None})
            if result:
                for issue in result.issues:
                    if issue.photo_id not in allowed:
                        continue
                    record = usable[issue.photo_id]
                    record.setdefault("final_review_issues", []).extend(issue.issues)
                    if issue.issues:
                        record["status"] = "manual_review_required"
                        record["review"]["approved"] = False
                    if result.confidence >= 0.65 and issue.needs_revision:
                        requests[issue.photo_id] = issue.suggested_changes

        for group in self.album["clusters"]:
            members = [usable[i] for i in group["members"] if i in usable]
            stats = gallery_statistics(
                [
                    {
                        "statistics": r["statistics_after"],
                        "face_region_rgb": r.get("face_region_rgb_after", []),
                    }
                    for r in members
                ]
            )
            for batch in chunks(members, 6):
                result = self.model.decide(
                    "final gallery reviewer",
                    GalleryReview,
                    {
                        "instruction": "Independently review these FINAL edits as an album group. Check color, exposure, "
                        "contrast, skin-tone consistency, crop context and excessive editing. Respect legitimate "
                        "differences and flag stylistically disconnected images using exact photo_ids. "
                        "Request revisions only when justified; never harmonize different skin identities.",
                        "global_contract": contract.model_dump(),
                        "cluster": group["label"],
                        "conditional_contract": group["contract"],
                        "original_statistics": group["statistics"],
                        "final_statistics": stats,
                        "photos": [
                            {
                                "photo_id": r["id"],
                                "file": r["file"],
                                "crop": r["crop_metrics"],
                                "color": r["color"],
                                "original_scene": r["album_context"]["original_scene_analysis"],
                                "review": r["review"],
                                "remaining_revisions": 2 - r["revision_cycles"],
                            }
                            for r in batch
                        ],
                    },
                    [self.root / "cache" / r["edited_preview"] for r in batch],
                )
                accept_review(result, {r["id"] for r in batch}, group["id"])
                completed += len(batch)
                self.stage(
                    "Reviewing crop, color and skin-tone consistency within clusters", completed, len(usable)
                )
        # Interleave groups so each contact-sheet page compares scenes, rather than filling
        # whole pages with one group's representatives before the next group appears.
        group_representatives = [
            [usable[i] for i in g["representatives"] if i in usable] for g in self.album["clusters"]
        ]
        representatives = [
            record for row in zip_longest(*group_representatives) for record in row if record is not None
        ]
        self.stage("Reviewing visual relationships across the final album", 0, len(representatives))
        completed = 0
        for page, batch in enumerate(chunks(representatives, 12)):
            sheet = Image.new("RGB", (960, ((len(batch) + 3) // 4) * 190), "#20231f")
            draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(batch):
                with Image.open(self.root / "cache" / r["edited_preview"]) as image:
                    small = preview(image, 220)
                    small.thumbnail((220, 145))
                    x, y = (i % 4) * 240, (i // 4) * 190
                    sheet.paste(small, (x, y))
                    draw.text((x, y + 150), r["id"], fill="white")
            path = self.root / "cache" / f"final-album-{page + 1}.jpg"
            save_image(sheet, path)
            result = self.model.decide(
                "final album relationship reviewer",
                GalleryReview,
                {
                    "instruction": "Check that these groups share one photographic visual language, not identical settings. "
                    "Look for disconnected grading, overprocessing or inconsistent framing across groups. "
                    "Preserve intentional scene, lighting and skin-tone differences. Exact IDs label the sheet.",
                    "global_contract": contract.model_dump(),
                    "photos": [
                        {
                            "photo_id": r["id"],
                            "cluster_id": r["cluster_id"],
                            "conditional_rules": r["album_context"]["cluster_style"],
                        }
                        for r in batch
                    ],
                },
                [path],
            )
            accept_review(result, {r["id"] for r in batch}, "cross-cluster-page")
            completed += len(batch)
            self.stage(
                "Reviewing visual relationships across the final album", completed, len(representatives)
            )
        for record in usable.values():
            before, after = record["statistics_before"], record["statistics_after"]
            if (
                after["highlights"] > before["highlights"] + 0.025
                or after["shadows"] > before["shadows"] + 0.04
            ):
                requests.setdefault(record["id"], Changes())
        self.stage("Applying final gallery revisions within remaining budgets", 0, len(requests))
        for index, (photo_id, changes) in enumerate(requests.items()):
            old = usable[photo_id]
            if old["manual"] or old["revision_cycles"] >= 2:
                old.setdefault("final_review_issues", []).append(
                    "Manual edit protected or two-revision budget exhausted"
                )
                old["review"]["approved"] = False
                old["status"] = "manual_review_required"
            else:
                neighbors = [
                    self.root / "cache" / r["edited_preview"]
                    for r in usable.values()
                    if r["cluster_id"] == old["cluster_id"] and r["id"] != photo_id
                ][:3]
                try:
                    revised = self.process_photo(
                        old["original"],
                        old["cv"],
                        old["album_context"]["cluster_statistics"],
                        AlbumContract(**old["album_style"]),
                        neighbors,
                        album_context=old["album_context"],
                        existing=old,
                        final_changes=changes,
                    )
                    revised["final_review_issues"] = old.get("final_review_issues", [])
                    revised["final_gallery_revision"] = True
                    records[records.index(old)] = revised
                    usable[photo_id] = revised
                    save_photo(self.root, revised)
                except Exception as exc:
                    old.setdefault("final_review_issues", []).append(f"Final revision failed: {exc}")
                    old["review"]["approved"] = False
                    old["status"] = "manual_review_required"
            self.stage("Applying final gallery revisions within remaining budgets", index + 1, len(requests))
        # Recheck revised groups visually once, without opening another revision loop.
        for batch in chunks([r for r in records if r.get("final_gallery_revision")], 6):
            result = self.model.decide(
                "final gallery revision verification",
                GalleryReview,
                {
                    "instruction": "Verify final revised photographs against their album/cluster contracts. "
                    "Remaining issues require manual review; no additional revision loop.",
                    "photos": [
                        {"photo_id": r["id"], "context": r["album_context"], "review": r["review"]}
                        for r in batch
                    ],
                },
                [self.root / "cache" / r["edited_preview"] for r in batch],
            )
            accept_review(result, {r["id"] for r in batch}, "post-revision-verification")
        write_json(
            self.root / "reports" / "final_gallery_review.json",
            {
                "observations": observations,
                "revision_requests": list(requests),
                "manual_and_budget_protection": True,
            },
        )

    def contact_sheet(self, records):
        # Paginated to bound memory for large galleries.
        for page in range(0, len(records), 40):
            group = records[page : page + 40]
            sheet = Image.new("RGB", (1000, ((len(group) + 3) // 4) * 190), "#20231f")
            draw = ImageDraw.Draw(sheet)
            for n, r in enumerate(group):
                path = r.get("edited_preview", r["original"].get("thumbnail"))
                if path:
                    with Image.open(self.root / "cache" / path) as im:
                        small = preview(im, 230)
                        small.thumbnail((230, 150))
                        x, y = n % 4 * 250, n // 4 * 190
                        sheet.paste(small, (x, y))
                        draw.text((x, y + 154), r["file"][:32], fill="white")
                        draw.text((x, y + 170), r["status"], fill="#c8d59c")
            save_image(sheet, self.root / "cache" / f"contact-sheet-{page // 40 + 1}.jpg")
