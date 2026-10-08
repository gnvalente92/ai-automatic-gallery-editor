import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings
from .pipeline import Pipeline
from .schemas import Override, ProcessRequest
from .storage import read_json, safe_path


def create_app(settings=None):
    pipeline = Pipeline(settings or Settings.from_env())
    app = FastAPI(title="Local Gallery Editor")
    app.state.pipeline = pipeline
    token = secrets.token_urlsafe(32)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            if request.headers.get("x-gallery-token") != token:
                return JSONResponse({"detail": "Missing local session token"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(RuntimeError)
    async def busy(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "model_configured": bool(pipeline.settings.vision_model),
            "progress": pipeline.public_progress(),
        }

    @app.get("/api/session")
    def session():
        return {
            "token": token,
            "settings": pipeline.settings.model_dump(
                mode="json", exclude={"root", "input_dir", "output_dir"}
            ),
        }

    @app.get("/api/gallery")
    def gallery():
        report = read_json(pipeline.root / "reports" / "gallery_report.json")
        inventory = read_json(pipeline.root / "cache" / "scan.json", {"photos": [], "count": 0})
        return {
            "scan": inventory,
            "report": report,
            "progress": pipeline.public_progress(),
            "performance": read_json(pipeline.root / "reports" / "performance.json"),
            "style": read_json(pipeline.root / "cache" / "album_style.json"),
        }

    @app.post("/api/scan")
    def scan_gallery():
        if not pipeline.lock.acquire(blocking=False):
            raise RuntimeError("Processing is running")
        try:
            return pipeline.inventory()
        finally:
            pipeline.lock.release()

    @app.post("/api/style", status_code=202)
    def style():
        if pipeline.lock.locked() or pipeline.progress["state"] == "queued":
            raise RuntimeError("Processing is running")
        pipeline.progress.update(state="queued", operation="style", error=None)

        def analyze():
            try:
                pipeline.propose_style()
            except Exception as exc:
                pipeline.progress.update(state="failed", error=str(exc))

        threading.Thread(target=analyze, daemon=True).start()
        return {"status": "started"}

    @app.post("/api/process", status_code=202)
    def process(body: ProcessRequest):
        if pipeline.lock.locked() or pipeline.progress["state"] == "queued":
            raise RuntimeError("Processing is running")
        pipeline.progress.update(state="queued", error=None)

        def worker():
            try:
                pipeline.run(body.style, body.theme)
            except Exception as exc:
                pipeline.progress.update(state="failed", error=str(exc))

        threading.Thread(target=worker, daemon=True).start()
        return {"status": "started"}

    @app.post("/api/photos/{photo_id}/override")
    def override(photo_id: str, body: Override):
        return pipeline.override(photo_id, body)

    @app.post("/api/photos/{photo_id}/reset")
    def reset(photo_id: str):
        return pipeline.override(photo_id, reset=True)

    @app.post("/api/photos/{photo_id}/variants/{variant_id}/select")
    def select_variant(photo_id: str, variant_id: str):
        return pipeline.select_variant(photo_id, variant_id)

    @app.get("/media/{area}/{relative:path}")
    def media(area: str, relative: str):
        if area not in {"cache", "output", "reports"}:
            raise HTTPException(404)
        bases = {
            "output": pipeline.output_root,
            "cache": pipeline.root / "cache",
            "reports": pipeline.root / "reports",
        }
        path = safe_path(bases[area], relative)
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    # Initial scan makes files visible without a directory picker or an extra UI action.
    pipeline.inventory()
    return app
