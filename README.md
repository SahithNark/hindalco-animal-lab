# Animal Classifier

This beginner-friendly project classifies animal photos with a torchvision
MobileNetV2 transfer-learning model.

## Install

From the project root on Linux:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -v
```

The tests use only Pillow and pytest and validate the folder labels and readable
images in `data/animals`. Add another animal by adding another lowercase folder;
the scripts discover it automatically.

## Prepare the data

```bash
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
```

This checks each image and creates an 80/20 per-animal split using seed 42.

## Train and evaluate

```bash
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
```

Training prints loss and training accuracy each epoch. Evaluation prints overall
and per-animal accuracy plus a confusion matrix, writes `metrics/metrics.json`,
and exits 1 if accuracy is below 0.70. Use `--from_scratch` for an optional
all-layer, no-ImageNet-weights experiment.

## Predict locally

```bash
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
```

## Create annotated test images

To create copies of every test image with the predicted animal and confidence
displayed in a banner at the top:

```bash
python src/annotate_test_images.py \
  --model_dir models \
  --test_data split/test \
  --out_dir annotated_test_images
```

The originals in `split/test` are not changed. The annotated copies are written
under `annotated_test_images/<animal>/`.

## Test the Azure ML scoring script

```bash
python src/make_request.py --image demo_images/cat_demo.jpg
AZUREML_MODEL_DIR=models python -c 'import sys; sys.path.insert(0, "src"); import score; score.init(); print(score.run(open("sample-request.json").read()))'
```

`score.py` is the Azure ML online endpoint entry point. It expects JSON with a
base64 JPEG/PNG under `image` and returns the animal, confidence, and all scores.

## Streamlit app

Set the endpoint values and start the UI:

```bash
# Use the scoring URL shown by `az ml online-endpoint show`.
export ENDPOINT_URL="https://your-endpoint.region.inference.ml.azure.com/score"
export ENDPOINT_KEY="your-key"
python -m pip install -r app/requirements.txt
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8502
```

The app requires both variables because it sends the image to the deployed Azure
ML online endpoint. If either variable is missing, it now displays a setup warning
instead of attempting a request to an empty URL. The endpoint must be deployed and
its key must be copied into `ENDPOINT_KEY`; local `models/` inference is tested by
`predict.py` and `score.py`, not by the endpoint-connected UI.

The app shrinks uploads to a maximum 512 pixels, sends the authenticated request,
and displays the prediction, confidence, score chart, and a low-confidence warning.

The supplied `azureml/` files and `azure-pipelines.yml` document the Azure ML
pipeline/deployment and are intentionally unchanged.