# ML Network Intrusion Detector

A Scikit-learn pipeline that classifies network traffic from the **NSL-KDD** dataset as normal
or one of four attack categories (DoS, Probe, R2L, U2R), using Random Forest, Decision Tree,
and Logistic Regression — with hyperparameter tuning, full evaluation, and saved models.

## Project Structure

```
ML Network Intrusion Detector/
├── data/
│   ├── KDDTrain+.txt   ← place dataset files here
│   └── KDDTest+.txt
├── models/             ← saved .pkl pipelines (written by the script)
├── plots/              ← saved charts (written by the script)
├── intrusion_detector.py
├── requirements.txt
├── README.md
└── GUIDE.md            ← step-by-step walkthrough
```

## Quick Start

See `GUIDE.md` for the full step-by-step walkthrough. Short version:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place KDDTrain+.txt and KDDTest+.txt into data/
python intrusion_detector.py
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

## Dataset Source

NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html — download `NSL-KDD.zip`, extract
`KDDTrain+.txt` and `KDDTest+.txt` into `data/` (filenames must match exactly).

## Design Notes

- The official `KDDTest+` set is reserved for final evaluation only — never used during
  training or cross-validation.
- `GridSearchCV` runs entirely within `KDDTrain+`, so test-set statistics never leak into
  model selection.
- Preprocessing lives inside each `Pipeline`, so there is zero leakage between CV folds.
