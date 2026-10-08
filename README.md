# Stillroom — local gallery editor

## Quick start (macOS)

```sh
git clone https://github.com/gnvalente92/ai-automatic-gallery-editor.git
cd ai-automatic-gallery-editor
# Copy your photographs into input/, then run:
./ai-automatic-gallery-editor
```

The launcher creates `input/` if needed, installs missing local tools and dependencies, starts Ollama, and downloads the default vision model. First run needs internet access and substantial free disk space for the model (about 21 GB); later runs use the installed setup. This bootstrap currently supports macOS. You do not need to install Homebrew, set environment variables, or start `ollama serve` yourself. A fresh clone with no photos will stop with a clear “no supported photographs” message until you add images to `input/`.

Place your photographs in `input/` and run the application.

Stillroom understands the library before making individual edits, then reviews the edited album. It preserves originals and uses deterministic crop/color rendering. All model requests go to a loopback endpoint on your computer. The headless command starts/configures local Ollama automatically; the optional browser UI uses explicitly configured model settings and otherwise falls back to measured corrections marked for human review.

Status: working experimental MVP. Owner/release maintainer: Unknown. Project type: local CLI, service and browser UI. Python 3.11+; validated with Python 3.12 on macOS. No cloud deployment or model training.

## First run details

On first launch, the script checks for `uv` and Ollama, installs missing tools using their official installers, provisions Python 3.12, installs this project's Python and RAW dependencies, starts local Ollama, and downloads the default vision model if it is not already present. It creates `input/`, `output/`, `cache/`, and `reports/` while preserving their existing contents. The theme is optional; add `--theme "A wedding gallery with clean aesthetics and dramatic, natural contrast"` for album guidance, or `--model MODEL` to choose another Ollama vision model.

First-time setup requires an internet connection. It downloads the Python dependencies and RAW decoder, Ollama, and the default local vision model (Qwen3-VL 32B Instruct Q4_K_M, about 21 GB). Allow enough free disk space and expect the model pull to take time. Ollama may request macOS administrator approval during its installation. The launcher currently targets macOS; no Homebrew installation is required. The selected CLI model is used for vision, text, and review roles. The app starts `ollama serve` itself when needed; the server remains running after editing. Photos and inference requests stay on the machine.

To run the optional browser interface instead, install `uv` and Ollama first, then:

```sh
uv run --python 3.12 --extra raw gallery-editor
```

Open **http://127.0.0.1:8765**. Startup creates the workspace directories and scans `input/`. Use **Analyze album style** to inspect/edit global and conditional rules, then **Process gallery**; processing also performs deep album analysis if you skip the style button. Stages remain visible while the background job runs.

The command prints each processing stage, discovered photo count, current photo, local AI request/result and model-download output to stdout. Logs for an automatically started Ollama server go to `cache/ollama-serve.log`. Each run writes to a new `output/runs/<run-id>/` folder, leaving earlier deliverables intact. It scans the complete source recursively, analyzes the whole album, generates safe crop/grade options for each photo, asks an independent option reviewer to choose one using the image and album mood, then renders and reviews that selection. The run's image output has exactly two folders: `options/` contains one folder per source photo with full-resolution uncropped grade choices and safe crop/grade combinations; `final/` contains the reviewer-selected exports. Reports are archived under `reports/runs/<run-id>/`; `reports/gallery_report.*` remains the latest-run view. Input/output paths may be outside the repository; `cache/` and `reports/` live under the current workspace (or `--workspace`). The input and output trees must not overlap. Originals are read-only.

Camera body, lens, aperture, shutter speed, ISO, focal length and related EXIF fields are extracted into the scan cache and per-photo records, including from embedded Fuji RAF previews with LibRaw fallback. The color agent receives these as factual context alongside the measured image and album/cluster contract. They can help interpret depth of field or motion, but do not select a camera recipe or dictate the grade: visible light, the photograph's intent and its cluster remain authoritative. Missing fields stay unknown.

The automatic edit may preserve the original framing when context or model uncertainty argues against cropping. It still exports every validated grade and crop/grade option under that photo's `options/` folder. If semantic scene analysis fails, crop choices are restricted to the original framing while the reviewer may still select a grade; the final export is flagged for review.

Artistic looks are chosen per photograph. The color schema supports natural color, monochrome, warm monochrome and restrained cinematic split tone; the model may select a creative look when it suits that individual image. Natural color remains available as an alternative, and artistic choices are not added to the gallery-wide style contract or propagated to other photos.

This uses a native process rather than Docker so it can reach Ollama at `127.0.0.1` directly and use the host's RAW decoder. The project also exposes the `ai-automatic-gallery-editor` console entry point when installed in a Python environment.

Every full run automatically archives its performance profile, album analysis, gallery review and per-photo reports under `reports/runs/<run-id>/`. No environment variables or separate Ollama startup are needed for the normal launcher command. The headless command uses `qwen3-vl:32b-instruct-q4_K_M` by default, even if an old `VISION_MODEL` shell export is present, and downloads it if needed. This 21 GB quantization leaves more memory for macOS, Ollama context and image processing than the 36 GB Q8 variant. Use `--model MODEL` only when you want to override the project default. The launcher prints the selected model and whether it came from `--model` or the project default. It rescans `input/` on every invocation; unchanged measurements and valid model decisions can be reused. Prompt/model/context changes invalidate the relevant decision caches. Failed semantic analysis is retried on the next run. Preserve `cache/overrides/`, which contains manual edits.

At completion, stdout lists each photo's applied crop percentage, cropped JPEG alternatives and reasons for withheld formats. A blocked image makes the CLI exit with status 2; other images and any color-only JPEGs remain available. Before album processing, the launcher scans the source and checks the selected model with a real preview and a required structured final answer. Failure is saved to `reports/model_preflight.json` and exits with status 3, avoiding a full run of unusable AI requests. Pass `--allow-fallback` explicitly to continue with measured exports. A model fallback cannot count as AI approval; if all album requests fail after a successful preflight, the command also exits 3 unless fallback was requested.

The current directory is the workspace root. To use another workspace without choosing a source folder in the UI:

```sh
uv run gallery-editor serve --root /path/to/workspace --port 8765
```

Each workspace has its own `input/` and generated directories. Run one application process per workspace. Stop with Ctrl+C. A stopped job can be rerun; cached model decisions are reused.

## Configure local models

For Ollama, the editor uses the native `/api/chat` endpoint with schema-constrained JSON and `think: false` ([Ollama thinking controls](https://docs.ollama.com/capabilities/thinking), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs)). Some thinking-model responses may still return empty `content`; the editor detects this, reports Ollama's prompt/completion counts, and safely marks affected edits for fallback/manual review. The headless command checks its default or explicit model against Ollama's local model capability metadata via `/api/show` and [vision support](https://docs.ollama.com/capabilities/vision). Other local OpenAI-compatible servers use `/v1/chat/completions` and can be selected by changing `LOCAL_MODEL_ENDPOINT`; those need an explicit `--model` (or `VISION_MODEL`). The browser UI continues to use explicitly configured model names; auto-start and auto-pull apply to the headless command.

The headless default, `qwen3-vl:32b-instruct-q4_K_M`, is listed at about 21 GB. Its Q8 variant is about 36 GB and the BF16 variant about 67 GB; the latter exceeds this Mac's unified memory, while Q8 leaves little room for macOS, context and image processing. Download sizes are not peak memory use. Qwen3-VL requires Ollama 0.12.7 or later. [Ollama model variants and sizes](https://ollama.com/library/qwen3-vl/tags).

Ollama requests default to a 32,768-token context and up to 8,192 output tokens. In the latest two-photo run, completed requests used 4,974–9,954 prompt tokens and 99–518 completion tokens, then ended with `stop` but returned empty content. This was not an output-token-limit failure. The adapter now reports native token counts in fallback diagnostics. Override with `MODEL_CONTEXT_TOKENS` or `MODEL_MAX_TOKENS` if a later profile shows context/output truncation.

The normal headless launcher starts Ollama when needed. The optional browser UI uses the manual Ollama setup below.

For the browser UI, start the local service manually and leave it running:

```sh
ollama serve
```

In another Terminal window, download one vision model and check that it appears locally:

```sh
ollama pull qwen3-vl:32b-instruct-q4_K_M
ollama list
```

Then return to this project directory, select the model, and start Stillroom:

```sh
export LOCAL_MODEL_ENDPOINT=http://127.0.0.1:11434/v1
export VISION_MODEL=qwen3-vl:32b-instruct-q4_K_M
export TEXT_MODEL="$VISION_MODEL"
export REVIEW_MODEL="$VISION_MODEL"
uv run gallery-editor
```

Put two or three JPG/JPEG photos in `input/`. Include different scenes if you want to see conditional groups. In the browser, select **Analyze album style** and wait for the stages to finish, inspect the global and group rules, then select **Process gallery**. Check `reports/album_analysis.json` for coverage, scene classifications and per-group contracts. Check `reports/gallery_report.json` for edit statuses and model events. Successful calls show `local_model` or `cached`; `fallback` entries include their cause. The first inference can take longer while Ollama loads the model. The UI keeps the progress stage visible.

Keep the test photos in `input/`; Stillroom never writes edits there. Review final exports and crop/grade alternatives under the latest `output/runs/<run-id>/`, and the archived summaries under `reports/runs/<run-id>/`. Inspect the model's group assignments and crop suggestions on your own photographs before processing a large album.

```sh
export LOCAL_MODEL_ENDPOINT=http://127.0.0.1:11434/v1 # or your other local OpenAI-compatible endpoint
export VISION_MODEL='vision-model-name-from-ollama-list' # replace with an installed vision model
export TEXT_MODEL="$VISION_MODEL"
export REVIEW_MODEL="$VISION_MODEL"
export MODEL_TIMEOUT=180
export MODEL_MAX_TOKENS=8192
uv run gallery-editor
```

`TEXT_MODEL` defaults to `VISION_MODEL`. `REVIEW_MODEL` defaults to `VISION_MODEL`; choose a different local vision model when available. Review uses independent requests with no editor conversation history. Only numerical loopback addresses and `localhost` are accepted. HTTP redirects and proxy environment variables are disabled. No external fonts, analytics, hosted scripts, or image APIs are used.

The endpoint must support multiple images per request and enough context for group statistics and visual contracts. Increase the local server context window and `MODEL_MAX_TOKENS` if responses truncate. `MODEL_TIMEOUT` is a configurable per-request timeout, not a per-photo or album time budget. A library can take several minutes or longer to understand. No total-time cap shortcuts analysis.

The browser application reads exported environment variables; it does **not** automatically load `.env`. The headless command needs no model variables with the default Ollama setup. [`.env.example`](.env.example) lists supported overrides.

If the server is absent, rejects a request, or returns malformed JSON, the pipeline records the failure and falls back safely. Connection failures stop further inference attempts for that run. Restarting analysis retries the server. Fallback never invents subjects, moods, expressions, model approvals or AI confidence. It leaves framing unchanged when semantic confidence is insufficient.

## Album-first editing

1. Recursively scan **every** supported image and cache dimensions, EXIF fields, thumbnails and sRGB previews.
2. Analyze **every** preview for luminance histogram, exposure, contrast, saturation, color balance, dominant colors, highlight/shadow fractions, sharpness, frontal faces and composition proxies. CPU analysis uses a bounded worker pool.
3. Ask the local VLM for scene/lighting/category information for **all** photographs in batches. Unknown information remains explicitly unknown.
4. Group by known scene/lighting/category, then by measured visual distance. Large groups subdivide for coverage. Without a VLM, group labels describe measurements, not invented indoor/outdoor classifications.
5. Select medoids, color/exposure extremes and additional visually distinct representatives **from every group**, including singletons. Analyze every selected representative for lighting, color relationships and composition in separate passes. Batch size limits one request, never total album coverage.
6. Synthesize relationships across all groups hierarchically, establish global principles, and derive a conditional visual contract for each group. You can change both global controls and group offsets before editing.
7. Give individual agents the full contract, group contract/statistics, gallery statistics, original scene analysis and related images. Correct relative to the appropriate group, with limited convergence strength. A dark indoor scene is not forced toward a bright outdoor median.
8. Render, independently review and revise each image if justified.
9. Review **all final edited images** within groups, compare groups through representative contact sheets, and check final statistics/resolution. Actionable final-review findings can trigger revisions within the **same maximum of two revisions per photo**, followed by verification. Manual edits stay protected.

The objective is a shared photographic language, not identical exposure or white balance. Full analysis coverage, group membership, observations and contracts are saved in `reports/album_analysis.json`. Numerical outliers alone are flags for review, not proof that a legitimate lighting difference must be removed.

## Resolution and crops

The default is **Digital + Print: 4000 px minimum long edge**. Profiles:

| `RESOLUTION_PROFILE` | Minimum long edge |
|---|---:|
| `digital` | 3000 px |
| `digital_print` | 4000 px |
| `high_quality_print` | 5000 px |
| `custom` | Set your own floors |

```sh
export RESOLUTION_PROFILE=custom
export MIN_LONG_EDGE=4200
export MIN_WIDTH=2800
export MIN_HEIGHT=2800
export MIN_MEGAPIXELS=12
uv run gallery-editor process
```

`MIN_WIDTH` and `MIN_HEIGHT` refer to oriented image axes. Long edge alone does not guarantee a minimum short edge; use dimensions or megapixels when needed. Custom profile with all floors zero disables print-resolution requirements intentionally. Environment values override profile defaults.

All crop proposals, reviewer suggestions, manual edits, and final exports use the same pixel-rounded checks. An original already below the floor is **blocked from export**, never upscaled. Automatic crops remove at most 30% of area and preserve detected/proposed protected regions. Missing semantic confidence retains original framing in the main edit; alternative JPEGs remain suggestions for human review. The crop agent treats the task as story-aware composition: it considers subject relationships, gallery role, useful context, negative space, movement/gaze room, frame edges and photographer intent. Thirds is only a soft candidate. When the original works or evidence is uncertain, it should remain unchanged. Deterministic checks enforce bounds, protected regions and pixel floors; CV edge-centroid/composition scores are only proxies, not semantic understanding. Each crop record includes the original model proposal, its ranked alternatives, candidate rejection reasons and the rationale for the actual selection. Primary and secondary keep-zones are protected regardless of their size; contextual regions remain advisory. Color-only reviewer changes preserve the chosen crop, and crop-only changes preserve the grade.

Composition prompts consider rule of thirds, leading lines, gaze/action space, negative space, horizon and visual balance as options, not a checklist. They avoid cramped framing and cuts through a face, ball, hand, foot or limb at a joint. Color prompts treat Fujifilm recipe collections and LUT catalogs as inspiration for describing a requested look, not defaults: no film simulation, LUT or grain is applied unless the album contract or user asks. Scene intent and skin/object colors take priority over a fashionable cast.

This guidance draws on [Digital Photography School's cropping guide](https://digital-photography-school.com/cropping-your-photos/), [Walsworth's cropping rules](https://www.walsworthyearbooks.com/top-10-essential-rules-for-good-photo-cropping/), and [Cambridge in Colour's composition tutorials](https://www.cambridgeincolour.com/tutorials.htm). Fujifilm recipe catalogs such as [Fuji X Weekly](https://fujixweekly.com/) and [Film Recipes](https://film.recipes/) inform look vocabulary only; their camera-specific recipes are not copied as RAW presets. Platform-specific social sizes are not automatically imposed on an album crop.

The UI reports remaining pixels, megapixels, area removed and physical sizes at 300/240/200 PPI. A4/A3/A2 indicators describe a fit within that paper size: ✓ at 300+ PPI, ~ at 200–299 PPI, otherwise ✕. They are not a guarantee of sharpness or borderless paper coverage.

## Files and privacy

```text
input/                         Original photographs; read-only
reference/<source-stem>_*.jpg  Optional local visual examples for a matching photo
output/runs/<run-id>/options/<source>/ Full-resolution grade and crop/grade options for one photo
output/runs/<run-id>/final/             Reviewer-selected final exports; original relative paths retained
cache/decoded/                 Lossless oriented RAW decodes, keyed by source hash
cache/graded/                  Lossless full-resolution grades for final JPEG exports
cache/thumbnails/              360 px thumbnails
cache/previews/                Up to 1280 px original/edited previews
cache/analysis/                Metadata, CV measurements, validated model responses
cache/crop_candidates/         Accepted and rejected candidate details
cache/reviews/                 Review responses and histories
cache/variants/<id>/           Cached crop × color comparison previews
cache/overrides/               Persistent manual decisions — back these up
cache/contact-sheet-*.jpg      Paginated edited gallery contact sheets
reports/gallery_report.html    Human-readable gallery report
reports/gallery_report.json    Complete machine-readable report
reports/album_analysis.json    Full-library coverage, groups and visual contracts
reports/runs/<run-id>/         Archived reports, album analysis, final review and performance
reports/performance.json       Latest run's stage, RAW/render, and per-model-call timings
reports/final_gallery_review.json
reports/final_cluster_statistics.json
reports/photos/<id>.json        Per-photo edit record
reports/photos/<id>.md          Per-photo human-readable summary
reports/feedback.json           AI decision, user change, final values and context
```

IDs derive from relative source paths; JSON records include the original filename. This prevents collisions between identically named photos in different folders. Generated exports can be replaced on reprocessing only if their checksum matches the prior application record. Unrelated or externally modified output files block export instead of being overwritten. No source image is written, renamed, removed, or given a sidecar.

Optional examples in `reference/` may be paired by the generic `<source-stem>_...` filename convention. Their filenames are not sent to the model; example images are attached only to their paired source and are never scanned as gallery photos. The model is told to infer transferable preferences rather than copy a framing. Model cache keys include reference bytes and the current prompt version.

Supported raster formats: JPG/JPEG, PNG, WEBP, TIFF/TIF. Hidden files/directories, unsupported files, symlinks and nested `output/`, `cache/`, `reports/` directories are skipped. Corrupt images are reported without stopping the remaining gallery. Multi-frame files use their first frame. ICC profiles are converted to sRGB. EXIF orientation is applied to the decoded image; original EXIF remains untouched. Exports intentionally do not copy source EXIF/GPS; extracted metadata stays in local reports/cache.

Optional RAW decoder, prioritizing RAF through LibRaw/rawpy:

```sh
uv sync --frozen --extra raw
uv run --extra raw gallery-editor
```

RAF/CR2/CR3/NEF/ARW/DNG decoding depends on the camera support of the installed LibRaw. RAW exports are named `original.RAF.jpg`, retaining the RAW filename and avoiding JPEG collisions. Per-photo records and edit reports are stored under `reports/`, outside the image-delivery folders. This MVP renders 8-bit sRGB; it is not a camera-profiled, 16-bit RAW development suite. Two real Fuji RAFs were decoded in this workspace; sensor-specific color fidelity has not been validated.

## Review and manual editing

Open an edited gallery tile for before/after comparison, zoom and pan. The crop overlay uses normalized coordinates. Change X/Y/width/height, choose an aspect ratio, or disable the crop. Change exposure, contrast, highlight/shadow levels, whites/blacks, temperature, tint, saturation or vibrance, then save. The saved output is independently reviewed; reviewer suggestions never overwrite a manual edit. Reset restores the recorded automatic version. Feedback records decisions and edits without training any model.

Each photo has an `options/<source filename>/` folder containing its uncropped grade options and all safe crop/grade combinations. Combinations include the unchanged framing, recommended and tighter framings, semantic alternatives, and portrait/landscape framing where they pass pixel floors and preserve protected subjects. Every option is a quality-100 JPEG; review previews also show dimensions, megapixels and printability. The independent option reviewer compares the labeled contact sheet and candidate data against that photo's mood, cluster contract and album style, then chooses the export written under `final/`. Its selected option is marked in the UI. If the selection call fails, the validated editor recommendation remains the fallback and is reported. The standard reviewer then checks the chosen full-resolution render. You can choose a different option manually; that choice is saved as an override. To refresh previews from existing analysis without repeating album inference, run `uv run gallery-editor variants`.

The displayed edited preview changes after saving; sliders do not provide an unsaved full-resolution color preview. Original/edited viewing previews are limited to 1280 px, so zoom is for composition comparison, not a full-resolution sharpness inspection. Inspect the export for pixel-level decisions.

Manual saves re-review the photograph with related edits and recompute group statistics. The final gallery review record marks that manual edits occurred after the last full album review; process the gallery again for a fresh album-wide visual review without losing those edits.

Do not remove `cache/overrides/` if you want to keep manual edits. Analysis/thumbnail/preview caches are regenerable; feedback, edit records and overrides should be backed up. If a source is replaced after a manual edit, processing blocks it so stale coordinates cannot silently apply to a different image. Back up the override, remove that photograph's override JSON, and reprocess to accept the new source.

## Build, test and package

```sh
uv sync --frozen --extra dev
uv run --extra dev ruff check src tests scripts
uv run --extra dev pytest
uv build
uv run --extra dev python scripts/smoke.py /tmp/stillroom-sample
```

The smoke directory must be new. The script creates three explicitly synthetic 4200 × 2800 test patterns, runs the complete pipeline, verifies exports/reports/cache, checks original hashes, and verifies manual persistence. These patterns test mechanics, not aesthetic quality. Install the wheel from `dist/` to deploy locally; no Node build is needed. CI runs the same core checks on Linux.

The checks cover malformed JSON, invalid color/crop parameters, 2,000 crop-boundary cases, supported formats, input preservation, metadata/cache behavior, model transport, full-library/group coverage, independent reviewer revisions, final-review budgets, reports, UI API and manual edits. See [verification](docs/verification.md) for the actual validation results and limits.

## Operations and limitations

Health: `http://127.0.0.1:8765/api/health`. Gallery progress and model events are visible in the UI and JSON reports. Model failures are recorded with role and reason. UI mutation requests require a local session token; Host checks and a restrictive content policy help protect the local service. It is not designed for network hosting or multiple users.

| Symptom | Check / action |
|---|---|
| No photographs found | Put supported files under this workspace's `input/`; press Scan |
| RAF/other RAW files skipped with “RAW decoder unavailable” | The running Python environment does not have `rawpy`. On macOS use `./ai-automatic-gallery-editor` so the launcher installs the `raw` extra, or run `uv sync --frozen --extra raw` before using `uv run gallery-editor`. The processor now stops before creating an empty run when no readable photographs remain. |
| All photos flagged for manual review | Check `reports/performance.json` for model failures; the headless command reports server/model setup errors before editing |
| Original below floor | Choose a suitable resolution profile or keep it blocked; the app never upscales |
| Model JSON rejected | Check server JSON/image support and token/context limits; inspect reported failure |
| RAW skipped | Install `--extra raw`; verify that LibRaw supports the camera |
| Export collision | Move the unrelated/modified export elsewhere, retain its backup, then rerun |
| Slow stage or model call | Open `reports/performance.json`; check elapsed stages, model roles, model/request durations, timeout fallbacks and full-resolution RAW/render costs |
| Interrupted processing | Rerun; completed decisions are cached; retain reports to prove output ownership |

Known limits: no bundled semantic model, segmentation, person/pose/object detector, identity tracking, or learned composition embedding. Frontal-face detection can miss faces; edge centroids are composition proxies. Face-region RGB is only a skin-color proxy and never a reason to equalize different people's skin tones. Warm/cool statistics are relative sRGB red-minus-blue values, **not estimated Kelvin**. Luminance/histogram statistics are sRGB proxies, not calibrated scene exposure. The renderer uses conservative photographic adjustments rather than physical camera/illuminant simulation. Model confidence is self-reported, not calibrated probability. Large full-resolution images require substantial RAM; images are rendered one at a time.

No time-per-photo performance claim is made. CPU analysis is parallel; local model calls remain sequential and cached to avoid competing for unified memory. Cluster representatives now receive lighting, color and composition analysis together in each request; per-photo crop and color planning share one request after photo analysis; conditional styles are returned in bounded batches. Every photo is still scanned, and every representative in every cluster is still analyzed. Each image sent to a model is converted once to a cached, orientation-correct 1280 px preview; full-resolution originals remain reserved for deterministic rendering/export. The performance profile is checkpointed during runs so a timeout or interruption leaves timing evidence. It separates source hashing/preview generation, full-resolution decode, render/export, pipeline stages and individual model requests, including model-input byte totals.

## Architecture and reused repository skills

See [architecture](docs/architecture.md), [repository inventory and skill reuse](docs/skill-reuse.md), and [verification](docs/verification.md). The repository initially contained methodology skills only. They guide this implementation; no image/model utility existed to call or duplicate. Existing skill files remain unchanged.

## Crop and export audit

See [the crop/export audit](docs/crop-audit.md) for the repaired failure paths and remaining limits. Model-proposed crops and geometric format suggestions are distinct: a suggestion is not proof of aesthetic quality. The application does not yet implement pose/segmentation/OCR models, automatic straightening, perspective correction, or a learned aesthetic scorer. The 90% human-acceptance target has not been measured.

RAW decoding and full-resolution grades are cached as lossless TIFFs. This uses more disk space but avoids repeated RAW decoding and JPEG-to-JPEG intermediate loss. Final JPEGs use quality 100 and no chroma subsampling. The renderer still works in 8-bit sRGB; these exports are not a high-bit-depth RAW development workflow.
