"""Evaluate a saved model and enforce the accuracy quality gate."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
sys.path.insert(0, str(Path(__file__).parent))
from model_utils import get_transform, load_model, log_metric


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model_dir", required=True); parser.add_argument("--test_data", required=True); parser.add_argument("--metrics_out", required=True); parser.add_argument("--min_accuracy", type=float, default=.70)
    args = parser.parse_args(); model, classes = load_model(args.model_dir)
    data = datasets.ImageFolder(args.test_data, transform=get_transform()); loader = DataLoader(data, batch_size=32)
    matrix = [[0 for _ in classes] for _ in classes]; correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(1)
            for real, predicted in zip(labels.tolist(), predictions.tolist()): matrix[real][predicted] += 1
            correct += (predictions == labels).sum().item(); total += labels.size(0)
    accuracy = correct / total if total else 0
    per_animal = {name: (matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else 0) for i, name in enumerate(classes)}
    print(f"Overall accuracy: {accuracy:.4f}"); print("Per-animal accuracy:")
    for name, value in per_animal.items(): print(f"  {name}: {value:.4f}")
    print("Confusion matrix (rows=real, columns=predicted):"); print("       " + " ".join(classes))
    for name, row in zip(classes, matrix): print(f"{name:>7} " + " ".join(map(str, row)))
    output = {"accuracy": accuracy, "per_animal_accuracy": per_animal, "classes": classes, "confusion_matrix": matrix}
    out = Path(args.metrics_out); out.mkdir(parents=True, exist_ok=True); (out / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    log_metric("test_accuracy", accuracy)
    if accuracy < args.min_accuracy: raise SystemExit(1)


if __name__ == "__main__": main()