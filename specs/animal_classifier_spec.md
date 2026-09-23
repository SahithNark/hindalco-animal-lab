# Animal Classifier Specification

## Goal

Predict which animal is shown in a photo and return the predicted animal together
with a confidence score and scores for every known animal.

## Model

Use transfer learning with torchvision MobileNetV2 and ImageNet weights. Freeze
`model.features`, replace `model.classifier[1]` with a new `Linear` layer whose
output size is the number of folders/classes, and keep `model.features` in eval
mode while training. A `--from_scratch` experiment may disable weights and make
all layers trainable.

## Images and labels

The folder name is the label. Animal names and the number of animals must always
be discovered from folder names; no animal name or fixed class count may be
hard-coded. A new folder therefore works without a code change. Images are
resized to 224x224 and ImageNet-normalised. Exactly the same transform is used
for training and prediction.

## Data split

For each animal independently, split 80% of its images into train and 20% into
test, using fixed seed 42. Preparation validates that every image opens, there
are at least two animals, and every animal has at least 10 images.

## Artifacts and quality gate

The saved model is a directory containing `model.pt` (a PyTorch `state_dict`)
and `classes.json` (the ordered list of animal names). Evaluation writes
`metrics.json` and exits with code 1 when test accuracy is below `min_accuracy`
(default 0.70).

## Acceptance criteria

- **AC1**: Pillow/pytest data checks pass.
- **AC2**: Preparation discovers five animals and creates train/test folders.
- **AC3**: Training finishes and saves `model.pt` and `classes.json`.
- **AC4**: Evaluation prints accuracy and a confusion matrix and writes metrics.
- **AC5**: Test accuracy is at least 0.70 and the quality gate passes.
- **AC6**: Prediction prints an animal and confidence for a demo image.
- **AC7**: The scoring script returns animal and confidence for a sample request.
- **AC8**: No script hard-codes animal names or the number of animals.
- **AC9**: README explains every step.