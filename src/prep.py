"""Validate and deterministically split the raw image folders."""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = Path(args.raw_data)
    folders = sorted(p for p in raw.iterdir() if p.is_dir())
    if len(folders) == 1:
        folders = sorted(p for p in folders[0].iterdir() if p.is_dir())
    if len(folders) < 2:
        raise ValueError("Need at least two animal folders")
    rng = random.Random(args.seed)
    rows = []
    for folder in folders:
        images = sorted(p for p in folder.iterdir() if p.is_file())
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
        if len(images) < args.min_images:
            raise ValueError(f"{folder.name} has only {len(images)} images")
        rng.shuffle(images)
        test_count = max(1, round(len(images) * args.test_ratio))
        test, train = images[:test_count], images[test_count:]
        for destination, selected in [(Path(args.train_out), train), (Path(args.test_out), test)]:
            target = destination / folder.name
            target.mkdir(parents=True, exist_ok=True)
            for image in selected:
                shutil.copy2(image, target / image.name)
        rows.append((folder.name, len(train), len(test)))
    print("Animal             train  test")
    print("------------------  -----  ----")
    for name, train, test in rows:
        print(f"{name:<18} {train:>5}  {test:>4}")


if __name__ == "__main__":
    main()