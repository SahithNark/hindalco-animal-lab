from pathlib import Path
from PIL import Image


ROOT = Path(__file__).parents[1] / "data" / "animals"


def test_data_layout():
    assert ROOT.is_dir()
    animals = [path for path in ROOT.iterdir() if path.is_dir()]
    assert len(animals) >= 2
    assert all(path.name == path.name.lower() and " " not in path.name for path in animals)
    assert not any(path.is_file() for path in ROOT.iterdir())
    for animal in animals:
        images = [path for path in animal.iterdir() if path.is_file()]
        assert len(images) >= 10
        for image in images:
            with Image.open(image) as opened:
                opened.verify()