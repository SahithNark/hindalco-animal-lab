# Validation Report

Validation was run on Linux from `/root/Downloads/Day_3/animal_lab_starter`.

| Criterion | Result | Evidence |
|---|---|---|
| AC1 | PASS | `tests/test_data.py::test_data_layout PASSED`; `1 passed` |
| AC2 | PASS | Prep table listed `cat`, `chicken`, `cow`, `dog`, and `horse`, each with `16` train and `4` test images. |
| AC3 | PASS | Training output: `Saved model to models`; files `/root/Downloads/Day_3/animal_lab_starter/models/model.pt` and `classes.json` exist. |
| AC4 | PASS | Evaluation printed `Overall accuracy: 0.7500` and `Confusion matrix (rows=real, columns=predicted)`; `/root/Downloads/Day_3/animal_lab_starter/metrics/metrics.json` exists. |
| AC5 | PASS | Evaluation accuracy was `0.7500`, above the `0.70` quality gate, and evaluation exited successfully. |
| AC6 | PASS | `Prediction: cat  (confidence 61%)` |
| AC7 | PASS | Scoring output included `{"animal": "cat", "confidence": 0.5946123003959656, "all_scores": ...}` for `sample-request.json`. |
| AC8 | PASS | Static grep over `src`, `app`, and `tests` found no animal-name literals or fixed `num_classes = 5`; classes are read from folders/`classes.json`. |
| AC9 | PASS | `/root/Downloads/Day_3/animal_lab_starter/README.md` documents install, tests, prep, training, evaluation, prediction, scoring, and Streamlit. |

Additional checks:

- `python -m pip install -r requirements.txt` completed successfully.
- `make_request.py` wrote a 17,734-byte request, below the Azure endpoint size limit.
- The supplied `azureml/` directory and `azure-pipelines.yml` were read and not modified.