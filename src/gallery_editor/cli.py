import argparse
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import Field

from .config import Settings
from .ollama_runtime import DEFAULT_VISION_MODEL, prepare_ollama
from .pipeline import NoReadablePhotosError, Pipeline
from .schemas import ProcessRequest, Schema
from .storage import write_json


class VisionReadiness(Schema):
    description: str = Field(min_length=1, max_length=500)


def verify_local_model(pipeline):
    """Exercise actual image input and structured output before an expensive album run."""
    inventory = pipeline.inventory()
    if not inventory["photos"]:
        return True
    photo = inventory["photos"][0]
    print("Checking local model: image input and a structured final answer...", flush=True)
    result = pipeline.model.decide(
        "vision readiness check",
        VisionReadiness,
        {
            "instruction": "Describe the visible photograph in one short sentence in the description field.",
            "request_id": uuid4().hex,
        },
        [pipeline.root / "cache" / photo["preview"]],
    )
    write_json(
        pipeline.root / "reports" / "model_preflight.json",
        {
            "model": pipeline.settings.vision_model,
            "passed": result is not None,
            "events": pipeline.model.events,
        },
    )
    return result is not None


def edit():
    parser = argparse.ArgumentParser(
        prog="ai-automatic-gallery-editor",
        description="Understand, edit and review a complete local photo gallery.",
    )
    parser.add_argument("--input", type=Path, default=Path("./input"), help="Source gallery (read-only)")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Edited exports destination")
    parser.add_argument("--theme", help="Optional plain-language album guidance (maximum 2,000 characters)")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Continue with measured exports if the local AI readiness check fails",
    )
    parser.add_argument(
        "--model",
        help=f"Local vision model override; defaults to {DEFAULT_VISION_MODEL}",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace for cache/ and reports/ (defaults to current directory)",
    )
    args = parser.parse_args()
    theme = ProcessRequest(theme=args.theme).theme
    settings = Settings.from_env(args.workspace)
    settings.input_dir = args.input
    settings.output_dir = args.output
    settings.initialize()
    print("AI Automatic Gallery Editor", flush=True)
    print(f"Input directory: {settings.input_dir} (read-only)", flush=True)
    print(f"Output base directory: {settings.output_dir} (each run gets its own subfolder)", flush=True)
    print(f"Reports directory: {settings.root / 'reports'}", flush=True)
    print(f"Cache directory: {settings.root / 'cache'}", flush=True)
    print("Scanning the complete input tree; preparing local AI...", flush=True)
    model = prepare_ollama(settings, args.model)
    print(f"Album theme: {theme or 'none; infer from the gallery'}", flush=True)
    print(f"Vision, text and review model: {model}", flush=True)
    pipeline = Pipeline(settings)

    active_stage = None
    stage_started = None
    last_progress = None
    original_stage = pipeline.stage

    def report_stage(name, completed, total):
        nonlocal active_stage, stage_started, last_progress
        now = perf_counter()
        if name != active_stage:
            if active_stage is not None:
                print(f"  Finished in {now - stage_started:.1f}s", flush=True)
            print(f"\n[{name}]", flush=True)
            active_stage, stage_started, last_progress = name, now, None
        file_name = pipeline.progress.get("file")
        progress_key = (name, completed, total, file_name)
        if total and progress_key != last_progress:
            suffix = (
                f" — {file_name}"
                if file_name
                and name
                in {
                    "Grading photographs and exporting RAW color options",
                    "Choosing crops, rendering and reviewing photographs",
                    "Editing photographs",
                }
                else ""
            )
            if name == "Analyzing all photographs" and completed == 0:
                print(f"Found {total} photographs. Beginning whole-library analysis.", flush=True)
            print(f"  {completed}/{total}{suffix}", flush=True)
            last_progress = progress_key
        elif not total and progress_key != last_progress:
            print("  In progress...", flush=True)
            last_progress = progress_key
        original_stage(name, completed, total)

    def report_model(event):
        role = event.get("role", "model")
        if event.get("status") == "running":
            print(f"  Local AI: {role} started ({event.get('image_count', 0)} previews)", flush=True)
        else:
            status = event.get("status", "unknown")
            reason = f" — {event['reason']}" if event.get("reason") else ""
            seconds = event.get("seconds", 0)
            print(f"  Local AI: {role} {status} in {seconds:.1f}s{reason}", flush=True)

    pipeline.stage = report_stage
    pipeline.model.progress_callback = report_model

    def report_scan(phase, file_name, completed, total):
        if phase == "reading":
            print(f"  Preview {completed}/{total}: decoding {file_name}", flush=True)
        elif phase == "ready":
            print(f"  Preview {completed}/{total}: ready {file_name}", flush=True)
        else:
            print(f"  Preview {completed}/{total}: could not read {file_name}; see scan report", flush=True)

    pipeline.scan_progress_callback = report_scan
    if not verify_local_model(pipeline):
        print("Local AI failed its structured vision check. See reports/model_preflight.json.", flush=True)
        if not args.allow_fallback:
            print(
                "Stopped before album editing. Select a working vision model with --model; "
                "use --allow-fallback only if measured fallback exports are wanted.",
                flush=True,
            )
            raise SystemExit(3)
    try:
        result = pipeline.run(theme=theme)
    except NoReadablePhotosError as exc:
        print(f"Processing stopped: {exc}", flush=True)
        raise SystemExit(2) from exc
    run_output = Path(settings.output_dir) / result["output_directory"]
    print(f"Found: {result['total']} photographs")
    print(f"Run: {result['run_id']}")
    print(f"Output directory: {run_output}")
    print(f"Per-photo options: {run_output / 'options'}")
    print(f"Reviewer-selected final images: {run_output / 'final'}")
    print(f"Run reports: {settings.root / 'reports' / 'runs' / result['run_id']}")
    print(f"Performance profile: {settings.root / 'reports' / 'performance.json'}")
    print(json.dumps(result["status_counts"], indent=2))
    for photo in result.get("photos", []):
        if photo["status"] == "blocked":
            print(f"BLOCKED {photo['file']}: {'; '.join(photo['issues'])}")
        else:
            cropped = sum(v["crop_metrics"]["crop_percentage"] > 0 for v in photo.get("variants", []))
            print(
                f"{photo['file']}: reviewer selected {photo.get('selected_option_id') or 'validated fallback'} · "
                f"{cropped} cropped options · {photo['color'].get('look', 'natural')}"
            )
            print(f"  {photo.get('crop_rationale', '')}")
            for withheld in photo.get("unavailable_variants", []):
                print(f"  Withheld {withheld['label']}: {withheld['reason']}")
    events = result.get("model_events", [])
    if events and not any(e.get("status") in {"local_model", "cached"} for e in events):
        print(
            "No valid local AI decisions were available. Exports use measured fallbacks and require review."
        )
    if result["status_counts"].get("blocked"):
        raise SystemExit(2)
    if (
        events
        and not args.allow_fallback
        and not any(e.get("status") in {"local_model", "cached"} for e in events)
    ):
        raise SystemExit(3)


def main():
    parser = argparse.ArgumentParser(description="Place photographs in input/ and run the application.")
    parser.add_argument(
        "command", choices=["serve", "process", "scan", "variants"], nargs="?", default="serve"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Workspace root; defaults to current directory"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    settings = Settings.from_env(args.root)
    settings.initialize()
    if args.command == "serve":
        import uvicorn

        from .app import create_app

        print(f"Input directory: {settings.root / 'input'}\nOpen http://127.0.0.1:{args.port}")
        uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)
    else:
        pipeline = Pipeline(settings)
        if args.command == "scan":
            result = pipeline.inventory()
            print(json.dumps(result, indent=2))
        elif args.command == "variants":
            result = pipeline.build_variants()
            run_output = Path(pipeline.output_root) / result["output_directory"]
            print(
                f"Refreshed crop/color previews for {result['total']} photographs.\n"
                f"Reviewer-selected full-resolution exports remain in {run_output / 'final'}; all options are in "
                f"{run_output / 'options'}.\n"
                "Performance profile: reports/performance.json"
            )
        else:
            result = pipeline.run()
            run_output = Path(settings.output_dir) / result["output_directory"]
            print(
                f"Input directory: input/\nFound: {result['total']} photographs\n"
                f"Output directory: {run_output}\n"
                f"Per-photo options: {run_output / 'options'}\n"
                f"Reviewer-selected final images: {run_output / 'final'}\n"
                f"Reports directory: reports/runs/{result['run_id']}/\n"
                "Performance profile: reports/performance.json"
            )
            print(json.dumps(result["status_counts"]))


if __name__ == "__main__":
    main()
