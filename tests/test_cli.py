import sys

from gallery_editor import cli


def test_headless_cli_prints_stages_and_model_progress(tmp_path, monkeypatch, capsys):
    class Model:
        progress_callback = None

    class FakePipeline:
        def __init__(self, settings):
            self.settings = settings
            self.model = Model()
            self.progress = {"file": None}

        def stage(self, name, completed, total):
            return None

        def run(self, theme=None):
            self.scan_progress_callback("reading", "sample.jpg", 1, 1)
            self.scan_progress_callback("ready", "sample.jpg", 1, 1)
            self.stage("Scanning photographs and generating previews", 0, 0)
            self.stage("Analyzing all photographs", 0, 1)
            self.progress["file"] = "sample.jpg"
            self.stage("Editing photographs", 1, 1)
            self.model.progress_callback({"role": "reviewer", "status": "running", "image_count": 1})
            self.model.progress_callback({"role": "reviewer", "status": "local_model", "seconds": 2.3})
            assert theme == "A wedding gallery"
            return {
                "total": 1,
                "status_counts": {"approved": 1},
                "run_id": "run-test",
                "output_directory": "runs/run-test",
            }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-automatic-gallery-editor", "--theme", "A wedding gallery"])
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)
    monkeypatch.setattr(cli, "prepare_ollama", lambda settings, requested: "test-vision-model")

    monkeypatch.setattr(cli, "verify_local_model", lambda pipeline: True)
    cli.edit()

    output = capsys.readouterr().out
    assert "Scanning the complete input tree" in output
    assert "Preview 1/1: decoding sample.jpg" in output
    assert "Preview 1/1: ready sample.jpg" in output
    assert "[Analyzing all photographs]" in output
    assert "[Editing photographs]" in output
    assert "1/1 — sample.jpg" in output
    assert "Local AI: reviewer started" in output
    assert "Local AI: reviewer local_model in 2.3s" in output
    assert "output/runs/run-test" in output
    assert '"approved": 1' in output


def test_blocked_cli_run_returns_failure_status(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import pytest

    pipeline = SimpleNamespace(
        stage=lambda *args: None,
        progress={},
        model=SimpleNamespace(),
        run=lambda **kwargs: {
            "total": 1,
            "status_counts": {"blocked": 1},
            "run_id": "test",
            "output_directory": "runs/test",
            "photos": [{"file": "image.RAF", "status": "blocked", "issues": ["Test export failure"]}],
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-automatic-gallery-editor"])
    monkeypatch.setattr(cli, "Pipeline", lambda settings: pipeline)
    monkeypatch.setattr(cli, "prepare_ollama", lambda *args: "test-vision")
    monkeypatch.setattr(cli, "verify_local_model", lambda pipeline: True)
    with pytest.raises(SystemExit) as exc:
        cli.edit()
    assert exc.value.code == 2
    assert "BLOCKED image.RAF: Test export failure" in capsys.readouterr().out


def test_failed_preflight_stops_before_album_processing(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    import pytest

    def unexpected_run(**kwargs):
        raise AssertionError("Album processing must not start")

    pipeline = SimpleNamespace(
        stage=lambda *args: None, progress={}, model=SimpleNamespace(), run=unexpected_run
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-automatic-gallery-editor"])
    monkeypatch.setattr(cli, "Pipeline", lambda settings: pipeline)
    monkeypatch.setattr(cli, "prepare_ollama", lambda *args: "test-vision")
    monkeypatch.setattr(cli, "verify_local_model", lambda pipeline: False)
    with pytest.raises(SystemExit) as exc:
        cli.edit()
    assert exc.value.code == 3
    assert "Stopped before album editing" in capsys.readouterr().out


def test_preflight_uses_actual_preview_and_records_failure(settings):
    from conftest import make_photo

    from gallery_editor.pipeline import Pipeline
    from gallery_editor.storage import read_json

    make_photo(settings.input_dir / "source.jpg")

    class Model:
        events = []

        def decide(self, role, schema, context, images):
            assert role == "vision readiness check"
            assert images[0].is_file()
            assert images[0].is_relative_to(settings.root / "cache")
            self.events.append({"status": "fallback", "reason": "Test empty final answer"})
            return None

    pipeline = Pipeline(settings, Model())
    assert cli.verify_local_model(pipeline) is False
    assert read_json(settings.root / "reports/model_preflight.json")["passed"] is False
    assert not list(settings.output_dir.rglob("*.jpg"))
