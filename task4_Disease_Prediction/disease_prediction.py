"""
TASK : Disease Prediction from Medical Data
--------------------------------------------
Predicts heart disease, diabetes and breast cancer from structured patient data
using SVM, Logistic Regression, Random Forest and XGBoost.

Usage:
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    python disease_prediction.py

Datasets
    - Breast Cancer (Wisconsin Diagnostic, UCI) : bundled with scikit-learn
    - Diabetes (Pima Indians)                   : downloaded from OpenML
    - Heart Disease (Cleveland, UCI)            : downloaded from UCI
  If you're offline, put 'diabetes.csv' (target column 'Outcome') and
  'heart.csv' (target column 'target') in a ./data folder.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warning] xgboost not installed -> skipping XGBoost (pip install xgboost)")

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_breast_cancer_data():
    d = load_breast_cancer(as_frame=True)
    X, y = d.data, d.target  # 0 = malignant, 1 = benign
    y = 1 - y                # make 1 = disease (malignant)
    return X, y


def load_diabetes_data():
    """Pima Indians Diabetes. Zeros in some columns actually mean 'missing'."""
    if os.path.exists("data/diabetes.csv"):
        df = pd.read_csv("data/diabetes.csv")
        y = df["Outcome"]
        X = df.drop(columns="Outcome")
    else:
        d = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
        X = d.data
        y = (d.target == "tested_positive").astype(int)
        X.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = X.copy()
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        X[col] = X[col].replace(0, np.nan)   # imputed later inside the pipeline
    return X, y


def load_heart_data():
    """Cleveland Heart Disease (UCI). target>0 means disease present."""
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "target"]
    if os.path.exists("data/heart.csv"):
        df = pd.read_csv("data/heart.csv")
    else:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        df = pd.read_csv(url, header=None, names=cols, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df.drop(columns="target"), df["target"]


DATASETS = {
    "Breast Cancer": load_breast_cancer_data,
    "Diabetes": load_diabetes_data,
    "Heart Disease": load_heart_data,
}


# ----------------------------------------------------------------------------
# 2. MODELS (each wrapped in a pipeline: impute -> scale -> classifier)
# ----------------------------------------------------------------------------
def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def get_models():
    models = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "SVM": (
            make_pipeline(SVC(probability=True, random_state=RANDOM_STATE)),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"],
             "clf__gamma": ["scale", 0.01]},
        ),
        "Random Forest": (
            make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 3]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            make_pipeline(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1]},
        )
    return models


# ----------------------------------------------------------------------------
# 3. TRAINNING &  EVALUATION
# ----------------------------------------------------------------------------
def evaluate_dataset(name, X, y):
    print(f"\n{'=' * 70}\n{name.upper()}  |  samples={len(X)}  features={X.shape[1]}  "
          f"positive rate={y.mean():.2%}\n{'=' * 70}")

    # Keep the class distribution balanced between training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted, roc_data = [], {}, {}
    for model_name, (pipe, grid) in get_models().items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[model_name] = best

        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        cv_acc = cross_val_score(best, X_train, y_train, cv=cv, scoring="accuracy").mean()

        rows.append({
            "Model": model_name,
            "CV Accuracy": cv_acc,
            "Test Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
        roc_data[model_name] = roc_curve(y_test, proba)
        print(f"[{model_name}] best params: {search.best_params_}")

    results = pd.DataFrame(rows).set_index("Model").round(4)
    print("\n", results.sort_values("ROC-AUC", ascending=False).to_string())

    # Select the model with the highest ROC-AUC on the test set
    best_name = results["ROC-AUC"].idxmax()
    print(f"\nBest model for {name}: {best_name}")
    print(classification_report(y_test, fitted[best_name].predict(X_test),
                                target_names=["Healthy", "Disease"]))

    tag = name.lower().replace(" ", "_")
    results.to_csv(f"{OUT_DIR}/{tag}_metrics.csv")
    plot_roc(name, tag, roc_data, results)
    plot_confusion(name, tag, best_name, fitted[best_name], X_test, y_test)
    plot_importance(name, tag, fitted, X.columns)
    return results


# ----------------------------------------------------------------------------
# 4. PLOTS
# ----------------------------------------------------------------------------
def plot_roc(name, tag, roc_data, results):
    plt.figure(figsize=(6, 5))
    for m, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{m} (AUC={results.loc[m, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{tag}_roc.png", dpi=150)
    plt.close()


def plot_confusion(name, tag, model_name, model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["Healthy", "Disease"],
        cmap="Blues", ax=ax)
    ax.set_title(f"{name} - {model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{tag}_confusion.png", dpi=150)
    plt.close()


def plot_importance(name, tag, fitted, feature_names):
    """Feature importance from Random Forest (tree-based, easy to explain)."""
    rf = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values().tail(10)
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color="teal")
    plt.title(f"Top 10 Feature Importances - {name} (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{tag}_feature_importance.png", dpi=150)
    plt.close()


def plot_summary(all_results):
    summary = pd.DataFrame({k: v["ROC-AUC"] for k, v in all_results.items()})
    summary.plot(kind="bar", figsize=(8, 5))
    plt.ylabel("ROC-AUC")
    plt.title("Model comparison across diseases")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/summary_comparison.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = {}
    for ds_name, loader in DATASETS.items():
        try:
            X, y = loader()
        except Exception as e:  # e.g. no internet
            print(f"\n[skipped] {ds_name}: could not load data ({e})")
            continue
        all_results[ds_name] = evaluate_dataset(ds_name, X, y)

    if all_results:
        plot_summary(all_results)
        print(f"\nAll metrics and plots saved in ./{OUT_DIR}/")
