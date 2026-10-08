# Crop and export audit — 2026-10-07

Scope: experimental local CLI/photo-editing service, with optional UI. Reused the existing scanner, model adapter, agents, renderer, variant generator, reviewer and reporting modules. Applied project-framing, project-quality-audit, service-project, architecture-method, quality-method, operations-method, delivery-method, documentation-method and human-writing-style. No separate image-processing implementation was introduced. Owner remains unknown; meeting notes are absent.

Assessment: 3/5 for the experimental project. The deterministic workflow is tested, but live-model photographic acceptance and unattended fresh-Mac installation are not verified.

## Findings and repairs

| Severity | Problem | Change |
|---|---|---|
| High | RAW color export referenced an image that had never been rendered because its assignment was inside an error branch. | Render happens on the normal path; a direct RAW exporter regression covers it. |
| High | A color-only reviewer revision reset a successful crop to full frame. | Crop and color revisions are independent; absent crop changes retain the current crop. Invalid reviewer crops retain the last validated crop. |
| High | Softened color parameters could be recorded while an old pregraded RAW image was rendered unchanged. | Any change to the actual color parameters switches rendering back to the source. Pixel comparisons test explicit color, implicit softening and crop-only revisions. |
| High | RAW alternatives matched color files by label, even after revisions changed the settings. | Matching uses the complete validated color parameters; unmatched grades are rendered from the source. Export metadata now truthfully marks full-resolution files. |
| High | Large semantic subjects and large Haar faces were discarded from crop protection solely because of size. | Primary/secondary regions and detected faces remain protected regardless of size. Context remains advisory. False positives remain a detector limitation, not grounds for silently removing protection. |
| Medium | Rejected AI proposals could still supply the final rationale and confidence; later passes obscured initial crop evidence. | Records retain the model proposal, semantic alternatives, initial candidates, selection/rejection reasons and rationale for the actual crop. |
| Medium | The model returned only one crop and other choices were geometric formats. | The schema now accepts up to three ranked semantic alternatives, each with confidence and rationale. They pass the same bounds, subject and resolution checks before being offered. |
| Medium | Subpixel crops could block the photo instead of being rejected individually. | Empty pixel-rounded proposals are rejected while valid candidates continue. |
| Medium | Failed semantic analysis could persist across repeated runs; cache keys omitted relevant inference settings. | Failed analysis is retried when a model is configured. Prompt version, endpoint, model, token/context settings and resolution settings participate in cache invalidation. |
| Medium | Crops reused a JPEG intermediate, and source RAWs were decoded twice in a fresh run. | Lossless TIFF decode/grade caches are shared between scanning, rendering and variants. Final JPEGs remain quality 100 with no chroma subsampling. |
| Medium | Same-stem files with different extensions collided in variant folders. | Per-source folders include the original extension. |
| Medium | A crop failure could hide successfully exported color-only JPEGs in the report. | Blocked records retain those exports; reports count and explain them. |
| Medium | Terminal output did not explain crop choices, and blocked runs exited successfully. | The CLI prints crop percentages, cropped alternative counts, rejection reasons and blocked errors; blocked runs exit 2. |
| Medium | Performance/album analysis files were only latest-run views. | Full runs archive performance, album analysis and final review alongside per-run gallery/photo reports. |
| Medium | Deliberate monochrome could be flagged for lacking gallery saturation or warmth. | Numerical saturation/temperature outlier checks do not treat intentional monochrome as a color mismatch. Visual review still applies. |

Per-image monochrome, warm monochrome and restrained cinematic choices remain available to the color agent. Natural alternatives are retained. Foreground framing guidance is generic: it can justify retaining a substantial foreground element for depth or story, without mentioning a specific photograph or always preserving obstructions.

## Verification

The latest user-run report available at audit time was `run-20261007-142032-065586`: two blocked photos, eight model fallbacks, no valid AI decisions, model `qwen3-vl:4b`. The environment denied connections to localhost Ollama (`operation not permitted`), so installation or successful use of a stronger local model could not be confirmed.

Two later complete runs used the supplied RAFs in a fresh isolated workspace, with model inference explicitly unconfigured. Each produced three uncropped color JPEGs and nine cropped JPEG alternatives per photograph. All checked JPEG dimensions matched their records and met the 4000 px long-edge floor. Both original hashes and modification times stayed unchanged. Each run had its own output directory and archived reports/profile. Main edits were correctly marked for manual review because semantic AI was unavailable.

Before decode sharing, the isolated fresh run took 40.011 seconds. After it, a fresh run took 27.283 seconds and a repeat run took 9.516 seconds. This is a two-photo deterministic rendering measurement, not a VLM benchmark. Lossless caches consume additional disk space.

Regression coverage includes actual rendered color versus metadata, independent revisions, bounded revision loops, full-size RAW choices, protected large subjects, failed analysis recovery, semantic alternatives, source-format collisions, malformed responses, all eight EXIF orientations, resolution floors and CLI failure status. See `verification.md` for the final suite result and retained evidence paths.

## Remaining limits

- The stronger model's live output, crop judgement and latency remain unverified. A successful rerun must show `local_model` or `cached` results rather than repeated empty-content fallback.
- A safe pixel rectangle does not prove a good composition. Geometric alternatives require human review; automatic main crops still require semantic confidence. Subject boxes can be wrong or incomplete.
- No pose/keypoint, instance-segmentation, OCR, depth, learned-aesthetic, horizon/straightening or perspective-correction module is installed. Prompt guidance does not implement these capabilities. Joint-cut and subject-identity guarantees cannot be claimed.
- There is no 500-image expert crop dataset, and the requested 90% human acceptance target has not been measured. Candidate search is a small conservative set plus semantic alternatives, not the proposed thousands-of-crops learned scoring engine.
- Rendering remains 8-bit sRGB with approximate temperature controls, not calibrated camera-profile/high-bit-depth RAW development. Capture metadata is a contextual hint for editing, not a camera-profile calibration or recipe system; available fields depend on the source format and decoder.
- Bootstrap syntax and orchestration are tested, but a fresh macOS machine, installer prompts and model downloads were not available for end-to-end installation verification.

## Run and inspect

```sh
./ai-automatic-gallery-editor --input ./input --output ./output
```

An explicit `--model MODEL` overrides the selected model for all roles. Every invocation scans the library. No cache deletion is needed after these prompt changes; manual overrides remain protected. Inspect the printed `output/runs/<run-id>/options/` and `final/` directories, plus `reports/runs/<run-id>/gallery_report.json` and `performance.json`.
