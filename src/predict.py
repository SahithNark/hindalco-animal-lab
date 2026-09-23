import argparse
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model_dir", required=True); parser.add_argument("--image", required=True)
    args = parser.parse_args(); model, classes = load_model(args.model_dir)
    with Image.open(args.image) as image: result = predict_image(model, classes, image)
    print(f"Prediction: {result['animal']}  (confidence {result['confidence']:.0%})")


if __name__ == "__main__": main()