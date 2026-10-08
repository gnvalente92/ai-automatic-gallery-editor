import httpx
import pytest

from gallery_editor.config import Settings
from gallery_editor.ollama_runtime import (
    DEFAULT_VISION_MODEL,
    _is_ollama_endpoint,
    choose_vision_model,
    prepare_ollama,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Client:
    def __init__(self, capabilities):
        self.capabilities = capabilities

    def post(self, url, json):
        return Response({"capabilities": self.capabilities.get(json["model"], [])})


def test_ollama_endpoint_detection():
    assert _is_ollama_endpoint("http://127.0.0.1:11434/v1")
    assert _is_ollama_endpoint("http://localhost:11434/v1")
    assert not _is_ollama_endpoint("http://127.0.0.1:1234/v1")


def test_selects_installed_vision_model_and_prefers_project_default():
    client = Client({"llama3.2:3b": ["completion"], DEFAULT_VISION_MODEL: ["completion", "vision"]})
    assert choose_vision_model(client, "http://127.0.0.1:11434", ["llama3.2:3b", DEFAULT_VISION_MODEL]) == (
        DEFAULT_VISION_MODEL
    )


def test_download_candidate_is_default_when_no_vision_model_is_installed():
    client = Client({"llama3.2:3b": ["completion"]})
    assert choose_vision_model(client, "http://127.0.0.1:11434", ["llama3.2:3b"]) == DEFAULT_VISION_MODEL


def test_explicit_text_only_model_is_rejected():
    client = Client({"llama3.2:3b": ["completion"]})
    with pytest.raises(ValueError, match="does not advertise image/vision"):
        choose_vision_model(client, "http://127.0.0.1:11434", ["llama3.2:3b"], "llama3.2:3b")


def test_prepares_server_and_pulls_default_model_when_needed(settings, monkeypatch, tmp_path):
    import gallery_editor.ollama_runtime as runtime

    state = {"server_started": False, "first_request": True, "pulled": []}

    class RuntimeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            if state["first_request"]:
                state["first_request"] = False
                raise httpx.ConnectError("offline")
            assert state["server_started"]
            return Response({"models": [] if not state["pulled"] else [{"name": DEFAULT_VISION_MODEL}]})

        def post(self, url, json):
            return Response({"capabilities": ["completion", "vision"]})

    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/ollama")
    monkeypatch.setattr(runtime.httpx, "Client", RuntimeClient)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)

    def start(command, **kwargs):
        state["server_started"] = True

    def pull(command, **kwargs):
        state["pulled"].append(command[-1])

    monkeypatch.setattr(runtime.subprocess, "Popen", start)
    monkeypatch.setattr(runtime.subprocess, "run", pull)

    custom_settings = Settings(root=tmp_path / "workspace")
    custom_settings.initialize()
    messages = []
    model = prepare_ollama(custom_settings, emit=messages.append)

    assert model == DEFAULT_VISION_MODEL
    assert state["server_started"]
    assert state["pulled"] == [DEFAULT_VISION_MODEL]
    assert custom_settings.vision_model == custom_settings.text_model == custom_settings.review_model == model
    assert any("Starting Ollama" in message for message in messages)
    assert any("Downloading local Ollama model" in message for message in messages)


def test_project_default_overrides_stale_vision_model_environment(settings, monkeypatch):
    import gallery_editor.ollama_runtime as runtime

    state = {"pulled": []}

    class RuntimeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            models = ["qwen3-vl:4b"] + state["pulled"]
            return Response({"models": [{"name": name} for name in models]})

        def post(self, url, json):
            return Response({"capabilities": ["completion", "vision"]})

    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/ollama")
    monkeypatch.setattr(runtime.httpx, "Client", RuntimeClient)
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda command, **kwargs: state["pulled"].append(command[-1])
    )
    settings.vision_model = "qwen3-vl:4b"

    messages = []
    selected = prepare_ollama(settings, emit=messages.append)

    assert selected == DEFAULT_VISION_MODEL
    assert state["pulled"] == [DEFAULT_VISION_MODEL]
    assert settings.vision_model == settings.text_model == settings.review_model == DEFAULT_VISION_MODEL
    assert any("project quality default" in message for message in messages)


def test_explicit_cli_model_replaces_stale_role_models(settings, monkeypatch):
    import gallery_editor.ollama_runtime as runtime

    class RuntimeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url):
            return Response({"models": [{"name": "chosen-vl"}]})

        def post(self, url, json):
            return Response({"capabilities": ["completion", "vision"]})

    monkeypatch.setattr(runtime.shutil, "which", lambda command: "/usr/bin/ollama")
    monkeypatch.setattr(runtime.httpx, "Client", RuntimeClient)
    settings.vision_model = "old-vision"
    settings.text_model = "old-text"
    settings.review_model = "old-review"

    selected = prepare_ollama(settings, requested_model="chosen-vl", emit=lambda _: None)

    assert selected == "chosen-vl"
    assert settings.vision_model == settings.text_model == settings.review_model == selected
