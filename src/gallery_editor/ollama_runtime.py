"""Prepare a local Ollama runtime for the one-command editor workflow."""

import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

import httpx

DEFAULT_VISION_MODEL = "qwen3-vl:32b-instruct-q4_K_M"
PREFERRED_INSTALLED_VISION_MODELS = (
    DEFAULT_VISION_MODEL,
    "qwen3-vl:32b-instruct",
    "qwen3-vl:32b",
    "qwen3-vl:30b-a3b-instruct",
    "qwen3-vl:30b",
    "gemma4:31b",
    "gemma4:26b",
    "qwen3.6:27b",
)


def _base_url(endpoint):
    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_ollama_endpoint(endpoint):
    parsed = urlparse(endpoint)
    return parsed.scheme == "http" and parsed.port == 11434 and parsed.path.rstrip("/") in {"", "/v1"}


def _request_json(client, method, url, **kwargs):
    response = getattr(client, method)(url, **kwargs)
    response.raise_for_status()
    return response.json()


def _installed_models(client, base_url):
    payload = _request_json(client, "get", base_url + "/api/tags")
    return [item.get("name") or item.get("model") for item in payload.get("models", [])]


def _has_vision(client, base_url, model):
    try:
        payload = _request_json(client, "post", base_url + "/api/show", json={"model": model})
    except (httpx.HTTPError, ValueError):
        return any(term in model.lower() for term in ("vision", "-vl", ":vl", "llava", "moondream"))
    capabilities = payload.get("capabilities", [])
    if capabilities:
        return "vision" in capabilities
    return any(term in model.lower() for term in ("vision", "-vl", ":vl", "llava", "moondream"))


def choose_vision_model(client, base_url, installed, requested=None):
    if requested:
        if requested not in installed:
            return requested
        if not _has_vision(client, base_url, requested):
            raise ValueError(f"Ollama model {requested!r} does not advertise image/vision support")
        return requested
    vision_models = [model for model in installed if model and _has_vision(client, base_url, model)]
    for preferred in PREFERRED_INSTALLED_VISION_MODELS:
        if preferred in vision_models:
            return preferred
    # Smaller installed models do not silently override the quality-focused default.
    return DEFAULT_VISION_MODEL


def _ensure_server(executable, settings, base_url, emit, startup_timeout=90):
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            _installed_models(client, base_url)
            return
    except (httpx.HTTPError, ValueError):
        pass

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    bind_host = f"[{host}]" if ":" in host else host
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{bind_host}:{port}"
    log_path = settings.root / "cache" / "ollama-serve.log"
    log_stream = log_path.open("a", encoding="utf-8")
    try:
        subprocess.Popen(
            [executable, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        log_stream.close()
        raise RuntimeError(f"Could not start Ollama: {exc}") from exc
    log_stream.close()
    emit(f"Starting Ollama at {base_url} (log: {log_path})")
    deadline = time.monotonic() + startup_timeout
    last_message = 0.0
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                _installed_models(client, base_url)
            emit("Ollama is ready.")
            return
        except (httpx.HTTPError, ValueError):
            now = time.monotonic()
            if now - last_message >= 5:
                emit("Waiting for Ollama to start...")
                last_message = now
            time.sleep(0.5)
    raise RuntimeError(
        f"Ollama did not become ready at {base_url} within {startup_timeout}s. "
        f"Check {log_path} for its startup log."
    )


def prepare_ollama(settings, requested_model=None, emit=print):
    """Start Ollama if needed, select/pull a vision model, and configure all agent roles."""
    if not _is_ollama_endpoint(settings.local_model_endpoint):
        model = requested_model or settings.vision_model
        if not model:
            raise RuntimeError(
                "A non-default local endpoint needs a model name. Pass --model or set VISION_MODEL."
            )
        settings.vision_model = model
        settings.text_model = model
        settings.review_model = model
        return model

    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("Ollama is not installed or not on PATH. Install Ollama, then rerun this command.")
    base_url = _base_url(settings.local_model_endpoint)
    _ensure_server(executable, settings, base_url, emit)

    with httpx.Client(timeout=10, trust_env=False) as client:
        installed = _installed_models(client, base_url)
        if requested_model:
            emit(f"Model selection: explicit --model {requested_model}")
        else:
            emit(
                f"Model selection: project quality default {DEFAULT_VISION_MODEL}; "
                "use --model to choose a different model."
            )
        chosen = choose_vision_model(client, base_url, installed, requested_model or DEFAULT_VISION_MODEL)
        # The single CLI --model option intentionally selects one model for every
        # role. Environment role overrides previously caused confusing mixed runs.
        requested = [chosen]
        for model in dict.fromkeys(requested):
            if model not in installed:
                emit(f"Downloading local Ollama model {model!r}; photo data stays on this machine.")
                try:
                    subprocess.run([executable, "pull", model], check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"Failed to download Ollama model {model!r} (exit {exc.returncode})."
                    ) from exc
                installed.append(model)
        for model in requested:
            if model in {chosen, settings.review_model or chosen} and not _has_vision(
                client, base_url, model
            ):
                raise RuntimeError(f"Ollama model {model!r} is installed but does not support vision.")

    settings.vision_model = chosen
    settings.text_model = chosen
    settings.review_model = chosen
    emit(f"Using local Ollama vision model: {chosen}")
    return chosen
