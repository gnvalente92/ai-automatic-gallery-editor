# Verification record

Initial validation: 2026-10-06. Follow-up: 2026-10-07. Platform: macOS ARM64, Python 3.12.14. Tests use synthetic image patterns and explicit model test doubles; neither is presented as photographic or live-AI evidence.

Headless runtime follow-up: The 2026-10-07 live profile confirms that the requested `qwen3-vl:32b` was used for vision while stale configured role models selected `qwen3-vl:4b` for text and review. Completed calls ended with `stop` but returned empty `content`; completion counts were only 99–518, below the 8,192-token cap. Requests took 6.8–146 seconds before the 180-second crop/color timeout. One edit request contained 19 MB because full-size reference images were included. The CLI applies its selected model to every role and caches orientation-correct 1280 px model inputs. Native token diagnostics appear in fallback reasons. On 2026-10-07 the headless default was changed to `qwen3-vl:32b-instruct-q4_K_M` (about 21 GB) and stale `VISION_MODEL` exports no longer override it. Live instruct-model speed and photographic quality are not yet verified.

## Completed checks

- 70 pytest checks passed before the crop/color alternatives change; 73 passed after reference-aware crop/color prompt updates. The suite covers safe portrait/landscape options, withheld protected-subject crops, grade combinations, reference scoping, prompt guidance, preview generation and full-resolution manual selection.
- Ruff validation passed; source and tests were formatted.
- Wheel and source distribution built successfully.
- Three 4200 × 2800 synthetic images completed the album-first pipeline with default 4000 px long-edge enforcement. Output images, contact sheets, edit records and gallery reports were generated outside `input/`. Original hashes were unchanged. A manual contrast edit survived another automatic run.
- A 200-item test library proved that every photograph reached the scene inventory and every meaningful group reached all three deep-analysis lenses. Rare/singleton groups were included. This is a coverage/architecture test, not a 200-photo model-quality benchmark.
- Deep lens coverage tests also prove the three dimensions remain represented after being consolidated into one request per representative batch. Photo crop/color planning and cluster-style batching are schema-validated; no live inference speed or quality comparison has been run.
- Final gallery review can revise a previously approved image. Tests prove that individual and final reviews share a maximum of two revisions, and that final review cannot overwrite a manual edit.
- Real Google Chrome browser automation passed desktop and 390 px mobile checks: full-library style analysis, stage progress, conditional controls, complete processing, before/after toggle, zoom, aspect-ratio crop rejection, manual save/reset, responsive width and no JavaScript errors. Desktop/mobile screenshots were visually inspected.
- Optional rawpy installed and imports successfully. Two supplied RAFs decoded and rendered; missing-decoder behavior is tested. Camera-specific RAW color fidelity remains unverified.
- 82 pytest checks pass after adding the one-command Ollama orchestration, scanner progress, stage/photo/model stdout reporting, native Ollama structured requests, and custom-path workflow. Ruff passes. CLI help displays the optional `--model` override without contacting Ollama.
- 86 pytest checks pass after profiling follow-up changes. Tests prove an explicit CLI model replaces stale role-specific environment values, large local model images are cached as max-1280 px inputs, and empty/malformed Ollama responses expose token diagnostics without logging reasoning content. Ruff and formatting checks pass; CLI help names the 30B instruct recommendation. The test suite uses a local mock endpoint, not a live Ollama model.
- Current crop/export and per-photo look changes pass 96 pytest checks. Pipeline tests verify that each run gets a distinct output folder, full-resolution quality-100 crop/grade alternatives are exported there, reports are archived per run, and originals remain unchanged. Renderer tests cover natural, monochrome, warm-monochrome and cinematic looks; prompts/schema keep those choices per image with a natural alternative. Ruff, JS syntax and launcher shell syntax pass. This does not verify what a live model will choose for a particular image.

The dependency test client emits one upstream Starlette warning that its HTTPX integration is deprecated. Tests pass; runtime requests use HTTPX normally.

## Reproduce

```sh
uv sync --frozen --extra dev
uv run --extra dev ruff check src tests scripts
uv run --extra dev pytest
uv build
uv run --extra dev python scripts/smoke.py /tmp/stillroom-new-sample
uv run --extra dev python scripts/check_ui.py /tmp/stillroom-new-sample --chrome
```

The browser command uses installed Google Chrome. Omit `--chrome` if a Playwright-managed Chromium has been installed. The smoke path must be unused; the script refuses to overwrite its sample sources. Browser checks save screenshots under that sample workspace's `reports/` and stop their server/browser on completion.

## Fresh-source validation

The supplied directory had no Git metadata, so a literal clean Git checkout was unavailable. Validation uses an isolated source copy with its own newly created virtual environment, no generated directories, no source photographs and no pre-existing model/cache data. Only repository source/configuration, tests, documentation and the unchanged skills are copied. Startup, full-resolution smoke processing, original preservation, the full test suite and packaging are checked there.

Result: the isolated copy at `/tmp/stillroom-final-checkout` installed from `uv.lock`, passed all 65 tests and Ruff, built both package formats, and passed the complete smoke workflow. Before startup, none of the four runtime directories existed; the application created them. A later contact-sheet ordering refinement was revalidated with the full suite and package build in the primary workspace.

## Evidence limits

No live local VLM/LLM was configured during validation. The later read-only browser check of Ollama's default `http://127.0.0.1:11434/api/tags` returned `ERR_CONNECTION_REFUSED`, so this session could not enumerate installed models or run live inference. The adapter contracts, caching, schema rejection, redirects and reviewer behavior are exercised with test doubles; real inference performance, prompt quality and photographic aesthetics are not claimed as validated. Fallback exports are labeled for manual review, with semantic observations unknown.

The supplied RAFs provide a real Fuji decoding and rendering path sample, but camera-specific RAW color fidelity has not been assessed. Human assessment with a live local model on a representative photographic album remains the next quality check. Configure Ollama, run **Analyze album style**, and inspect its groups and conditional contracts before processing a larger album.

Reference support uses a generic source-stem naming convention. The model receives attached examples without their filenames and is instructed to infer transferable preferences rather than copy a specific composition. The `reference/` directory may be empty. Tests verify pairing behavior and that unrelated examples are not attached.

## Performance and alternatives check

On 2026-10-06, the two RAFs already in this workspace were used for a read-only alternatives-generation profile; input hashes remained unchanged. The initial profile took 14.645 seconds, of which 13.663 seconds were full-resolution RAW decoding solely to build low-resolution comparison previews. Generating previews from the already cached 1280 px source previews removed that redundant RAW decode. The repeat took 0.059 seconds total (0.038 s source-integrity hashing, 0.010 s cached-preview decoding, and 0.001 s crop/color preview generation; timings are rounded and machine-specific).

The run produced 21 crop/color combinations for the two images. Both have portrait options; one has landscape options, while the other’s landscape crop was withheld because it would cut a detected protected subject region. Each choice reports resulting full-source dimensions and resolution/print metrics. This measures variant preparation only, not the earlier album/model processing that had already run; Ollama was not reachable from the validation environment, so no new live-inference timing or quality claim is made. `reports/performance.json` contains the latest run profile and will be replaced by later operations.

A full two-RAF pipeline run completed in 16.167 seconds with no local model configured. Editing took 16.080 seconds: full-resolution RAW decode took 13.455 seconds and render took 2.148 seconds. All model roles were fallback calls, so this timing profile measures the deterministic pipeline and RAW cost, not local VLM inference. The two exports were correctly marked for manual review. The archived profile, gallery report and album report are under `reports/profiles/` with run ID `20261006-162003`.

## Comprehensive crop/export revision — 2026-10-07

117 tests pass after the follow-up audit, including real pixel-versus-parameter comparisons through RAW color/crop revisions, all eight EXIF orientations, failed-analysis recovery, cached RAW decode invalidation, large-subject protection and blocked CLI exit status. Ruff checks/formatting pass. The existing Starlette test-client deprecation warning remains.

Two complete deterministic runs on the supplied RAFs succeeded in the isolated workspace `/var/folders/s_/f0fmcfc938z5r0n0yb7n6w6r0000gn/T/stillroom-final-raw-audit-kozhqvgi`. Each run produced three uncropped color JPEGs and nine cropped JPEG alternatives per photo (three cropped framings × three grades). Checked file dimensions matched metadata and met the 4000px long-edge floor; source hashes and modification times were unchanged. Lossless intermediate caching reduced fresh-run time from 40.011s to 27.283s; repeat processing took 9.516s. These measurements exclude live model inference.

Verification summaries, per-run reports and profiles are retained under `reports/audits/revision-20261007-144849/`. The user's latest reports were preserved. Ollama connections from this session were denied (`operation not permitted`), so the stronger local model and photographic acceptance remain unverified. Detailed findings and remaining gaps are in [crop-audit.md](crop-audit.md).
