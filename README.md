# ML Network Intrusion Detector

A Scikit-learn pipeline that classifies network traffic from the **NSL-KDD** dataset as normal
or one of four attack categories (DoS, Probe, R2L, U2R), using Random Forest, Decision Tree,
and Logistic Regression — with hyperparameter tuning, full evaluation, and saved models.

## Project Structure

```
ML Network Intrusion Detector/
├── data/
│   ├── KDDTrain+.txt   ← included
│   └── KDDTest+.txt    ← included
├── models/             ← saved .pkl pipelines (written by the script)
├── plots/              ← saved charts (written by the script)
├── intrusion_detector.py
├── requirements.txt
└── README.md
```

## What It Does

- Loads and cleans the NSL-KDD dataset (41 features, binary + 5-class labels)
- Preprocesses with a leak-free `ColumnTransformer` (one-hot encoding + scaling) inside
  each model's `Pipeline`
- Trains Random Forest, Decision Tree, and Logistic Regression on both binary
  (normal vs. attack) and multi-class (normal/dos/probe/r2l/u2r) tasks
- Tunes Random Forest with `GridSearchCV` (3-fold CV, optimizing F1)
- Evaluates every model: classification reports, confusion matrices, ROC/AUC, feature importance
- Saves all 7 trained pipelines (`models/*.pkl`) and all charts (`plots/*.png`)
- Exports a metrics comparison table (`model_comparison.csv`)
- Includes a `predict_anomaly(sample_dict)` function to classify a single new traffic record

## Dataset

**NSL-KDD** (`KDDTrain+.txt`, `KDDTest+.txt`) is included in `data/` — no download needed.
Original source: https://www.unb.ca/cic/datasets/nsl.html (mirrored via
https://www.kaggle.com/datasets/hassan06/nslkdd as the official page is currently unavailable).

## Step-by-Step Setup

### 1. Open the folder in VS Code

File → Open Folder → select `ML Network Intrusion Detector`.

### 2. Open a terminal

Terminal → New Terminal (make sure it opens inside the project folder).

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

Your terminal prompt should now show `(.venv)` at the start of the line.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs numpy, pandas, scikit-learn, matplotlib, seaborn, and joblib.

### 5. Run the script

```bash
python intrusion_detector.py
```

What happens:
- Checks the dataset files exist (and tells you clearly if they're missing)
- Loads and labels the data (binary + multi-class)
- Trains 6 baseline models (3 classifiers × 2 tasks)
- Runs `GridSearchCV` to tune Random Forest — **this step takes a few minutes**, don't
  interrupt it
- Prints classification reports for every model straight to the terminal
- Saves 5 charts into `plots/`
- Saves 7 trained models into `models/`
- Saves `model_comparison.csv` in the project root

### 6. Check your results

Open the `plots/` folder — you should see:
- `01_class_distribution.png`
- `02_confusion_matrices_binary.png`
- `03_confusion_matrices_multi.png`
- `04_roc_curves.png`
- `05_feature_importance.png`

Open `model_comparison.csv` in Excel or a text editor to see the accuracy/precision/recall/F1
table across all models.

## Troubleshooting

- **`ModuleNotFoundError`** → your virtual environment isn't active, or step 4 wasn't run.
  Re-activate the venv and reinstall requirements.
- **`FileNotFoundError: Missing dataset file(s)`** → confirm `KDDTrain+.txt` / `KDDTest+.txt`
  are in `data/` with exact, case-sensitive filenames.
- **Script runs but `plots/` is empty** → check the terminal output for errors partway through;
  the exact error message points to the failing step.

## Design Notes

- The official `KDDTest+` set is reserved for final evaluation only — never used during
  training or cross-validation.
- `GridSearchCV` runs entirely within `KDDTrain+`, so test-set statistics never leak into
  model selection.
- Preprocessing lives inside each `Pipeline`, so there is zero leakage between CV folds.