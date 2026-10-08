from gallery_editor.agents import composition_references, crop_agent, photo_edit_agent
from gallery_editor.config import Settings
from gallery_editor.schemas import AlbumContract, Analysis


class CaptureModel:
    def __init__(self):
        self.call = None

    def decide(self, role, schema, context, images=()):
        self.call = (role, context, list(images))
        return None


def test_reference_files_match_only_the_corresponding_photo(tmp_path):
    folder = tmp_path / "reference"
    folder.mkdir()
    left = folder / "source_one_left.jpg"
    right = folder / "source_one_right.jpg"
    unrelated = folder / "source_two_example.jpg"
    unsupported = folder / "source_one_notes.txt"
    for path in (left, right, unrelated, unsupported):
        path.write_bytes(b"reference")

    refs = composition_references(tmp_path, {"file": "sports/source_one.RAF"})

    assert refs == [left, right]
    assert composition_references(tmp_path, {"file": "source_two.RAF"}) == [unrelated]


def test_photo_edit_prompt_receives_matching_references(tmp_path):
    folder = tmp_path / "reference"
    folder.mkdir()
    first = folder / "source_one_left.jpg"
    second = folder / "source_one_right.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    model = CaptureModel()
    photo = {
        "file": "source_one.RAF",
        "width": 4032,
        "height": 6032,
        "preview": "previews/source.jpg",
    }
    cv = {"statistics": {"exposure": 0.4, "temperature": 0, "saturation": 0.5, "highlights": 0.1}}
    gallery = {
        "exposure": {"median": 0.4},
        "temperature": {"median": 0},
        "saturation": {"median": 0.5},
    }

    photo_edit_agent(
        model,
        photo,
        tmp_path,
        Analysis(confidence=0.9),
        cv,
        gallery,
        AlbumContract(),
        Settings(root=tmp_path),
    )

    role, context, images = model.call
    assert role == "photo crop and color editor"
    assert "reference_files" not in context
    assert "transferable preferences" in context["user_composition_reference"]
    assert "Never copy pixels" in context["user_composition_reference"]
    assert "rule of thirds" in context["crop_rule"]
    assert "Do not force thirds" in context["crop_rule"]
    assert "no LUT, film simulation" in context["color_rule"]
    assert "for this photograph only" in context["color_rule"]
    assert "Do not propagate that choice" in context["color_rule"]
    assert images[-2:] == [first, second]


def test_crop_prompt_treats_foreground_elements_as_possible_framing(tmp_path):
    model = CaptureModel()
    photo = {
        "file": "source_one.RAF",
        "width": 6032,
        "height": 4032,
        "preview": "previews/source.jpg",
    }
    crop_agent(
        model,
        photo,
        tmp_path,
        Analysis(confidence=0.9),
        {"statistics": {}, "edge_centroid": [0.5, 0.5], "protected_regions": []},
        AlbumContract(),
        Settings(root=tmp_path),
    )

    role, context, _ = model.call
    assert role == "crop editor"
    assert "may be intentional framing" in context["foreground_framing_guidance"]
    assert "not a rule to retain every obstruction" in context["foreground_framing_guidance"]

    other = dict(photo, file="source_two.RAF")
    crop_agent(
        model,
        other,
        tmp_path,
        Analysis(confidence=0.9),
        {"statistics": {}, "edge_centroid": [0.5, 0.5], "protected_regions": []},
        AlbumContract(),
        Settings(root=tmp_path),
    )
    assert model.call[1]["foreground_framing_guidance"] == context["foreground_framing_guidance"]


def test_crop_prompt_receives_only_matching_composition_references(tmp_path):
    folder = tmp_path / "reference"
    folder.mkdir()
    matching = folder / "source_one_preferred.jpg"
    unrelated = folder / "source_two_example.jpg"
    matching.write_bytes(b"matched example")
    unrelated.write_bytes(b"other photo")
    model = CaptureModel()
    photo = {
        "file": "source_one.RAF",
        "width": 4032,
        "height": 6032,
        "preview": "previews/source.jpg",
    }

    crop_agent(
        model,
        photo,
        tmp_path,
        Analysis(confidence=0.9),
        {"statistics": {}, "edge_centroid": [0.5, 0.5], "protected_regions": []},
        AlbumContract(),
        Settings(root=tmp_path),
    )

    role, context, images = model.call
    assert role == "crop editor"
    assert "reference_files" not in context
    assert "transferable preferences" in context["user_composition_reference"]
    assert images[-1] == matching
    assert unrelated not in images
