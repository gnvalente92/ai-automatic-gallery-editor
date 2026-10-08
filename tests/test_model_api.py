import base64
import hashlib
import io
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gallery_editor.app import create_app
from gallery_editor.models import LocalModel
from gallery_editor.schemas import Analysis, Crop


@pytest.mark.parametrize(
    "payload",
    [None, "not JSON", '{"x":9}', '{"x":0,"width":-1}', '{"x":0,"width":1,"extra":true}'],
)
def test_malformed_model_response_falls_back(settings, monkeypatch, payload):
    settings.vision_model = "test-only"

    def handler(request):
        return httpx.Response(200, json={"message": {"content": payload}, "done_reason": "stop"})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    adapter = LocalModel(settings)
    assert adapter.decide("test", Crop, {}) is None
    assert adapter.events[-1]["status"] == "fallback"


def test_model_cache_schema_and_payload(settings, monkeypatch):
    settings.vision_model = "test-only"
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "message": {"content": Analysis(confidence=0.8).model_dump_json()},
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 15,
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    adapter = LocalModel(settings)
    assert adapter.decide("photo analyst", Analysis, {}).confidence == 0.8
    assert calls[0]["think"] is False
    assert calls[0]["stream"] is False
    assert calls[0]["format"] == Analysis.model_json_schema()
    assert calls[0]["options"]["num_predict"] == settings.model_max_tokens
    assert calls[0]["options"]["num_ctx"] == settings.model_context_tokens
    assert adapter.events[-1]["api"] == "ollama_native"
    assert adapter.events[-1]["request_seconds"] >= 0
    assert adapter.events[-1]["seconds"] >= adapter.events[-1]["request_seconds"]
    assert adapter.decide("photo analyst", Analysis, {}).confidence == 0.8
    assert len(calls) == 1
    assert adapter.events[-1]["status"] == "cached"
    assert adapter.events[-1]["seconds"] >= 0


def test_large_model_images_are_cached_as_1280px_previews(settings, monkeypatch, tmp_path):
    settings.vision_model = "test-only"
    settings.initialize()
    source = tmp_path / "large-reference.png"
    pixels = np.random.default_rng(7).integers(0, 256, (1600, 2400, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "message": {"content": Analysis(confidence=0.8).model_dump_json()},
                "done_reason": "stop",
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    adapter = LocalModel(settings)
    assert adapter.decide("photo analyst", Analysis, {"pass": 1}, [source]) is not None
    assert adapter.decide("photo analyst", Analysis, {"pass": 2}, [source]) is not None

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cached = settings.root / "cache" / "model_previews" / f"{source_hash}.jpg"
    with Image.open(cached) as model_preview:
        assert max(model_preview.size) == 1280
    assert cached.exists()
    assert adapter.events[-1]["request_image_bytes"] < source.stat().st_size
    image_value = calls[0]["messages"][1]["images"][0]
    decoded = Image.open(io.BytesIO(base64.b64decode(image_value)))
    assert max(decoded.size) == 1280


def test_empty_ollama_content_reports_native_token_diagnostics(settings, monkeypatch):
    settings.vision_model = "test-only"

    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": "", "thinking": "private internal reasoning"},
                "done_reason": "stop",
                "prompt_eval_count": 5000,
                "eval_count": 210,
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    adapter = LocalModel(settings)

    assert adapter.decide("photo analyst", Analysis, {}) is None
    event = adapter.events[-1]
    assert "prompt_tokens=5000" in event["reason"]
    assert "completion_tokens=210" in event["reason"]
    assert "private internal reasoning" not in event["reason"]


def test_openai_compatible_non_ollama_endpoint_keeps_standard_request(settings, monkeypatch):
    settings.vision_model = "test-only"
    settings.local_model_endpoint = "http://127.0.0.1:1234/v1"
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": Analysis(confidence=0.8).model_dump_json()}}]}
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    assert LocalModel(settings).decide("photo analyst", Analysis, {}).confidence == 0.8
    assert "think" not in calls[0]
    assert "response_format" in calls[0]


def test_prompt_treats_composition_rules_and_film_recipes_as_guidance(settings, monkeypatch):
    settings.vision_model = "test-only"
    messages = []

    def handler(request):
        messages.extend(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"message": {"content": Analysis(confidence=0.8).model_dump_json()}})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    assert LocalModel(settings).decide("photo analyst", Analysis, {}).confidence == 0.8
    system = messages[0]["content"]
    assert "Rule of thirds is only a candidate heuristic" in system
    assert "preserve the original when it already works" in system
    assert "Avoid cuts at joints" in system
    assert "do not apply film simulation, LUTs, grain" in system


def test_redirect_not_followed(settings, monkeypatch):
    settings.vision_model = "test-only"
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://example.com/collect"})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    assert LocalModel(settings).decide("photo analyst", Analysis, {}) is None
    assert len(calls) == 1 and calls[0].startswith("http://127.0.0.1")


def test_ui_api_and_manual_override(gallery):
    app = create_app(gallery)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/gallery").json()["scan"]["count"] == 3
        assert client.post("/api/scan").status_code == 403
        token = client.get("/api/session").json()["token"]
        headers = {"x-gallery-token": token}
        assert client.post("/api/scan", headers=headers).status_code == 200
        report = app.state.pipeline.run()
        state = client.get("/api/gallery").json()
        assert state["performance"]["status"] == "complete"
        assert client.get("/media/reports/performance.json").status_code == 200
        record = report["photos"][0]
        response = client.post(
            f"/api/photos/{record['id']}/override",
            headers=headers,
            json={"crop": {}, "color": {"contrast": 15}},
        )
        assert response.status_code == 200
        assert response.json()["color"]["contrast"] == 15
        invalid = client.post(
            f"/api/photos/{record['id']}/override",
            headers=headers,
            json={"crop": {"width": 0.01, "height": 0.01}, "color": {}},
        )
        assert invalid.status_code == 422
        assert client.get("/media/input/a.jpg").status_code == 404
        assert client.get("/media/cache/%2e%2e/input/a.jpg").status_code in (404, 422)
        assert client.get("/api/health", headers={"host": "evil.test"}).status_code == 400


def test_api_busy(gallery):
    app = create_app(gallery)
    with TestClient(app) as client:
        token = client.get("/api/session").json()["token"]
        app.state.pipeline.lock.acquire()
        try:
            assert client.post("/api/process", headers={"x-gallery-token": token}, json={}).status_code == 409
        finally:
            app.state.pipeline.lock.release()
