"""Write test images with the model prediction and confidence at the top."""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model, predict_image


def get_font(size):
    """Use a readable system font when available, otherwise Pillow's default."""
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def annotate_image(image, result):
    """Add a black banner containing the prediction and confidence."""
    image = image.convert("RGB")
    banner_height = 54
    annotated = Image.new("RGB", (image.width, image.height + banner_height), "white")
    annotated.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(annotated)
    draw.rectangle((0, 0, annotated.width, banner_height), fill=(25, 25, 25))
    text = f"Prediction: {result['animal']}    Confidence: {result['confidence']:.1%}"
    draw.text((12, 13), text, fill="white", font=get_font(25))
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--test_data", default="split/test")
    parser.add_argument("--out_dir", default="annotated_test_images")
    args = parser.parse_args()

    model, classes = load_model(args.model_dir)
    test_root = Path(args.test_data)
    output_root = Path(args.out_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path for path in test_root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise ValueError(f"No image files found below {test_root}")

    for image_path in image_paths:
        with Image.open(image_path) as image:
            result = predict_image(model, classes, image)
            annotated = annotate_image(image, result)
        destination = output_root / image_path.relative_to(test_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(destination, quality=95)
        print(f"{image_path}: {result['animal']} ({result['confidence']:.1%}) -> {destination}")

    print(f"Wrote {len(image_paths)} annotated images to {output_root}")


if __name__ == "__main__":
    main()