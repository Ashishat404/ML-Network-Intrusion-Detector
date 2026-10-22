"""
ML Network Intrusion Detector
==============================
Trains and evaluates Random Forest, Decision Tree, and Logistic Regression
classifiers on the NSL-KDD dataset for binary (normal/attack) and multi-class
(normal/dos/probe/r2l/u2r) network intrusion detection.

Usage:
    python intrusion_detector.py

Expects:
    data/KDDTrain+.txt
    data/KDDTest+.txt
(download from https://www.unb.ca/cic/datasets/nsl.html)

Outputs:
    plots/*.png          - EDA, confusion matrices, ROC curve, feature importance
    models/*.pkl          - 7 trained pipelines (joblib)
    Console output         - classification reports, metrics comparison table
"""

import os
import pathlib
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # safe for headless/script execution; plots are saved, not shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

RANDOM_STATE = 42

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",       # original multi-class attack type string
    "difficulty",  # NSL-KDD difficulty level — dropped before modeling
]
assert len(COLUMN_NAMES) == 43, "Column count mismatch"

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [
    c for c in COLUMN_NAMES if c not in CATEGORICAL_FEATURES + ["label", "difficulty"]
]

# Verified NSL-KDD taxonomy (Tavallaee et al. 2009)
ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos", "worm": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "httptunnel": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def check_files():
    missing = [p for p in (TRAIN_PATH, TEST_PATH) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing dataset file(s): {names}. "
            f"Download NSL-KDD from https://www.unb.ca/cic/datasets/nsl.html "
            f"and place them in {DATA_DIR}/"
        )
    print(f"Found dataset files in {DATA_DIR}/")


def load_data():
    df_train = pd.read_csv(TRAIN_PATH, header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(TEST_PATH, header=None, names=COLUMN_NAMES)
    print(f"Train shape: {df_train.shape} | Test shape: {df_test.shape}")
    return df_train, df_test


def map_attack_category(series: pd.Series) -> pd.Series:
    unmapped = set(series.unique()) - set(ATTACK_MAP.keys())
    if unmapped:
        raise ValueError(f"Unmapped NSL-KDD labels — update ATTACK_MAP: {sorted(unmapped)}")
    return series.map(ATTACK_MAP)


def add_labels(df_train, df_test):
    df_train["attack_category"] = map_attack_category(df_train["label"])
    df_test["attack_category"] = map_attack_category(df_test["label"])
    df_train["binary_label"] = (df_train["attack_category"] != "normal").astype(int)
    df_test["binary_label"] = (df_test["attack_category"] != "normal").astype(int)
    print("\nMulti-class distribution (train):")
    print(df_train["attack_category"].value_counts())
    print("\nBinary distribution (train):")
    print(df_train["binary_label"].value_counts().rename({0: "normal", 1: "attack"}))
    return df_train, df_test


def plot_eda(df_train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cat_counts = df_train["attack_category"].value_counts()
    sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=axes[0], palette="Set2")
    axes[0].set_title("Multi-Class Label Distribution (Train)")
    axes[0].set_xlabel("Attack Category")
    axes[0].set_ylabel("Count")

    bin_counts = df_train["binary_label"].value_counts().rename({0: "Normal", 1: "Attack"})
    sns.barplot(x=bin_counts.index, y=bin_counts.values, ax=axes[1],
                palette=["steelblue", "tomato"])
    axes[1].set_title("Binary Label Distribution (Train)")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved plots/01_class_distribution.png")


def split_features_labels(df_train, df_test):
    drop_cols = ["label", "difficulty", "attack_category", "binary_label"]
    X_train = df_train.drop(columns=drop_cols)
    X_test = df_test.drop(columns=drop_cols)
    y_train_bin = df_train["binary_label"].values
    y_test_bin = df_test["binary_label"].values
    y_train_multi = df_train["attack_category"].values
    y_test_multi = df_test["attack_category"].values
    print(f"\nX_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    return X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi


def make_preprocessor():
    """Fresh ColumnTransformer each call — safe to embed in multiple Pipelines."""
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ], remainder="drop")


def make_pipeline(classifier):
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])


def train_baselines(X_train, y_train_bin, y_train_multi):
    estimators = {
        "rf_binary": make_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        "dt_binary": make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "lr_binary": make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)),
        "rf_multi": make_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        "dt_multi": make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "lr_multi": make_pipeline(LogisticRegression(max_iter=1000, solver="saga", random_state=RANDOM_STATE, n_jobs=-1)),
    }
    print("\nTraining baseline models...")
    for name, pipe in estimators.items():
        y_tr = y_train_bin if "binary" in name else y_train_multi
        pipe.fit(X_train, y_tr)
        print(f"  done: {name}")
    return estimators


def predict_all(estimators, X_test):
    return {name: pipe.predict(X_test) for name, pipe in estimators.items()}


def tune_random_forest(X_train, y_train_bin):
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    gs_pipeline = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    grid_search = GridSearchCV(
        estimator=gs_pipeline, param_grid=param_grid, cv=3,
        scoring="f1", n_jobs=-1, refit=True, verbose=1,
    )
    print("\nRunning GridSearchCV (3-fold, 12 candidates)... this can take a few minutes.")
    grid_search.fit(X_train, y_train_bin)
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def print_classification_reports(baseline_preds, y_test_bin, y_test_multi):
    print("\n=== BINARY CLASSIFICATION REPORTS ===")
    for name in ["rf_binary", "dt_binary", "lr_binary"]:
        print(f"\n--- {name.upper()} ---")
        print(classification_report(y_test_bin, baseline_preds[name], target_names=["Normal", "Attack"]))

    print("\n=== MULTI-CLASS CLASSIFICATION REPORTS ===")
    for name in ["rf_multi", "dt_multi", "lr_multi"]:
        print(f"\n--- {name.upper()} ---")
        print(classification_report(y_test_multi, baseline_preds[name]))


def plot_confusion_matrices(model_names, y_trues, y_preds, titles, filename, ncols=2):
    nrows = -(-len(model_names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    for ax, name, y_true, y_pred, title in zip(axes, model_names, y_trues, y_preds, titles):
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(set(y_true))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[len(model_names):]:
        ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved plots/{filename}")


def plot_roc(estimators, tuned_pipeline, X_test, y_test_bin):
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_models = {
        "RF Baseline": estimators["rf_binary"],
        "DT Baseline": estimators["dt_binary"],
        "LR Baseline": estimators["lr_binary"],
        "RF Tuned": tuned_pipeline,
    }
    colors = ["steelblue", "forestgreen", "darkorange", "crimson"]
    for (label, pipe), color in zip(roc_models.items(), colors):
        scores = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Binary Intrusion Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "04_roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved plots/04_roc_curves.png")


def plot_feature_importance(tuned_pipeline):
    clf = tuned_pipeline.named_steps["classifier"]
    preprocessor = tuned_pipeline.named_steps["preprocessor"]
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = cat_names + NUMERIC_FEATURES

    fi_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
    fi_df = fi_df.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", ax=ax, palette="viridis_r", orient="h")
    ax.set_title("Top 15 Feature Importances — Tuned Random Forest (Binary)")
    ax.set_xlabel("Mean Decrease in Gini Impurity")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved plots/05_feature_importance.png")
    print("\nTop 15 features:")
    print(fi_df.to_string(index=False))


def compute_metrics_binary(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="binary", zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="binary", zero_division=0), 4),
    }


def compute_metrics_multi(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def build_comparison_table(baseline_preds, y_pred_tuned_bin, y_test_bin, y_test_multi):
    rows = []
    for name in ["rf_binary", "dt_binary", "lr_binary"]:
        m = compute_metrics_binary(y_test_bin, baseline_preds[name])
        rows.append({"Model": name.replace("_binary", "").upper(), "Task": "Binary", "Variant": "Baseline", **m})
    m = compute_metrics_binary(y_test_bin, y_pred_tuned_bin)
    rows.append({"Model": "RF", "Task": "Binary", "Variant": "Tuned (GridSearchCV)", **m})
    for name in ["rf_multi", "dt_multi", "lr_multi"]:
        m = compute_metrics_multi(y_test_multi, baseline_preds[name])
        rows.append({"Model": name.replace("_multi", "").upper(), "Task": "Multi-class (5)", "Variant": "Baseline", **m})

    comparison_df = pd.DataFrame(rows).set_index(["Task", "Model", "Variant"])
    print("\n=== Model Comparison Table ===")
    print("(Binary metrics: average=binary | Multi-class metrics: average=weighted)")
    print(comparison_df.to_string())
    comparison_df.to_csv(PROJECT_ROOT / "model_comparison.csv")
    print("\nSaved model_comparison.csv")
    return comparison_df


def save_models(estimators, tuned_pipeline):
    save_map = {
        "rf_binary_baseline.pkl": estimators["rf_binary"],
        "dt_binary_baseline.pkl": estimators["dt_binary"],
        "lr_binary_baseline.pkl": estimators["lr_binary"],
        "rf_binary_tuned.pkl": tuned_pipeline,
        "rf_multi_baseline.pkl": estimators["rf_multi"],
        "dt_multi_baseline.pkl": estimators["dt_multi"],
        "lr_multi_baseline.pkl": estimators["lr_multi"],
    }
    print("\nSaving models...")
    for filename, pipeline in save_map.items():
        out_path = MODELS_DIR / filename
        joblib.dump(pipeline, out_path)
        print(f"  saved: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")


def predict_anomaly(sample: dict, model_path: pathlib.Path = None) -> str:
    """
    Classify a single new traffic record as 'Normal' or 'Anomaly'.
    `sample` must be a dict with all 41 NSL-KDD feature keys (no label/difficulty).
    """
    model_path = model_path or (MODELS_DIR / "rf_binary_tuned.pkl")
    pipeline = joblib.load(model_path)
    row = pd.DataFrame([sample])
    pred = pipeline.predict(row)[0]
    return "Anomaly" if pred == 1 else "Normal"


def main():
    check_files()
    df_train, df_test = load_data()
    df_train, df_test = add_labels(df_train, df_test)
    plot_eda(df_train)

    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = \
        split_features_labels(df_train, df_test)

    estimators = train_baselines(X_train, y_train_bin, y_train_multi)
    baseline_preds = predict_all(estimators, X_test)

    tuned_pipeline = tune_random_forest(X_train, y_train_bin)
    y_pred_tuned_bin = tuned_pipeline.predict(X_test)
    print("\nTuned RF (binary) — Test Set Classification Report")
    print(classification_report(y_test_bin, y_pred_tuned_bin, target_names=["Normal", "Attack"]))

    print_classification_reports(baseline_preds, y_test_bin, y_test_multi)

    plot_confusion_matrices(
        ["rf_binary", "dt_binary", "lr_binary", "rf_binary_tuned"],
        [y_test_bin] * 4,
        [baseline_preds["rf_binary"], baseline_preds["dt_binary"], baseline_preds["lr_binary"], y_pred_tuned_bin],
        ["RF Baseline (Binary)", "DT Baseline (Binary)", "LR Baseline (Binary)", "RF Tuned (Binary)"],
        "02_confusion_matrices_binary.png", ncols=2,
    )
    plot_confusion_matrices(
        ["rf_multi", "dt_multi", "lr_multi"],
        [y_test_multi] * 3,
        [baseline_preds["rf_multi"], baseline_preds["dt_multi"], baseline_preds["lr_multi"]],
        ["RF Baseline (Multi)", "DT Baseline (Multi)", "LR Baseline (Multi)"],
        "03_confusion_matrices_multi.png", ncols=3,
    )

    plot_roc(estimators, tuned_pipeline, X_test, y_test_bin)
    plot_feature_importance(tuned_pipeline)
    build_comparison_table(baseline_preds, y_pred_tuned_bin, y_test_bin, y_test_multi)
    save_models(estimators, tuned_pipeline)

    # Sanity check: reload one model and confirm predictions match
    reloaded = joblib.load(MODELS_DIR / "rf_binary_tuned.pkl")
    assert np.array_equal(reloaded.predict(X_test[:10]), tuned_pipeline.predict(X_test[:10])), "Reload mismatch!"
    print("\nPipeline reload verification passed.")

    print("\nAll done. Check plots/, models/, and model_comparison.csv.")


if __name__ == "__main__":
    main()
