# Step-by-Step Guide

## 1. Unzip the project

Extract the zip anywhere on your computer. You should see:

```
ML Network Intrusion Detector/
├── data/
├── models/
├── plots/
├── intrusion_detector.py
├── requirements.txt
├── README.md
└── GUIDE.md
```

## 2. Open the folder in VS Code

File → Open Folder → select `ML Network Intrusion Detector`.

## 3. Open a terminal

Terminal → New Terminal (make sure it opens inside the project folder).

## 4. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **macOS / Linux:** `source .venv/bin/activate`

Your terminal prompt should now show `(.venv)` at the start of the line.

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

This installs numpy, pandas, scikit-learn, matplotlib, seaborn, and joblib.

## 6. Download the NSL-KDD dataset

1. Go to https://www.unb.ca/cic/datasets/nsl.html
2. Download **NSL-KDD.zip**
3. Extract it, and find `KDDTrain+.txt` and `KDDTest+.txt`
4. Copy both files into the `data/` folder in this project
5. Double-check the filenames are exactly `KDDTrain+.txt` and `KDDTest+.txt` (case-sensitive)

## 7. Run the script

```bash
python intrusion_detector.py
```

What happens:
- It checks your dataset files exist (and tells you clearly if they're missing)
- Loads and labels the data (binary + multi-class)
- Trains 6 baseline models (3 classifiers × 2 tasks)
- Runs `GridSearchCV` to tune Random Forest — **this step takes a few minutes**, don't
  interrupt it
- Prints classification reports for every model straight to your terminal
- Saves 5 charts into `plots/`
- Saves 7 trained models into `models/`
- Saves `model_comparison.csv` in the project root

## 8. Check your results

Open the `plots/` folder — you should see:
- `01_class_distribution.png`
- `02_confusion_matrices_binary.png`
- `03_confusion_matrices_multi.png`
- `04_roc_curves.png`
- `05_feature_importance.png`

Open `model_comparison.csv` in Excel or a text editor to see the accuracy/precision/recall/F1
table across all models.

## 9. If something errors

- **`ModuleNotFoundError`** → your virtual environment isn't active, or step 5 wasn't run.
  Re-activate the venv and reinstall requirements.
- **`FileNotFoundError: Missing dataset file(s)`** → your `KDDTrain+.txt` / `KDDTest+.txt`
  aren't in `data/`, or are misnamed. Re-check step 6.
- **Script runs but `plots/` is empty** → check the terminal output for errors partway through;
  copy the exact error message to debug it.

## 10. Write up your submission

For your BharatCares project submission, include:
- A short problem statement (why ML for intrusion detection)
- Dataset source (NSL-KDD, with the link)
- Your preprocessing approach (one-hot encoding + scaling, no leakage)
- Models compared and why (Random Forest, Decision Tree, Logistic Regression)
- Your results — paste in 2-3 of the saved charts (confusion matrix, ROC curve, feature
  importance) plus the metrics table from `model_comparison.csv`
- A short "limitations & future work" paragraph (e.g. doesn't catch zero-day attacks,
  could add deep learning or SMOTE for class imbalance later)


