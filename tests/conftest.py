import numpy as np
import pytest
from PIL import Image

from gallery_editor.config import Settings


@pytest.fixture
def settings(tmp_path):
    result = Settings(root=tmp_path, resolution_profile="custom", min_long_edge=400)
    result.initialize()
    return result


def make_photo(path, brightness=1, size=(640, 480)):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.1, 0.8, size[0], dtype=np.float32)
    a = np.tile(x, (size[1], 1))
    rgb = np.stack([a * 0.95, a, a * 0.9], axis=2)
    image = Image.fromarray((np.clip(rgb * brightness, 0, 1) * 255).astype(np.uint8))
    image.save(path)
    return path


@pytest.fixture
def gallery(settings):
    for name, brightness in (("a.jpg", 1), ("nested/b.png", 0.7), ("c.webp", 1.15)):
        make_photo(settings.root / "input" / name, brightness)
    return settings
