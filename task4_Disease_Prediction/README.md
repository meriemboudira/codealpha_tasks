# Disease Prediction from Medical Data

Machine learning project that predicts the likelihood of **breast cancer**, **diabetes** and **heart disease** from structured patient data, comparing four classification algorithms.

## Objective
Apply classification techniques to structured medical datasets and compare model performance across diseases.

## Datasets
| Disease | Source | Samples | Features |
|---|---|---|---|
| Breast Cancer | UCI Wisconsin Diagnostic (via scikit-learn) | 569 | 30 |
| Diabetes | Pima Indians Diabetes (OpenML) | 768 | 8 |
| Heart Disease | UCI Cleveland | 303 | 13 |

Features include age, blood pressure, cholesterol, glucose, BMI, insulin, tumour measurements, etc.

## Algorithms
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

## Methodology
1. **Load data** and convert the target to binary (1 = disease, 0 = healthy).
2. **Preprocessing** inside a scikit-learn `Pipeline`: median imputation of missing values, then standard scaling. In the Diabetes data, zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as missing.
3. **Split** 80% train / 20% test (stratified).
4. **Tune** hyperparameters with 5-fold `GridSearchCV` (scored by ROC-AUC).
5. **Evaluate** on the test set: accuracy, precision, recall, F1, ROC-AUC.
6. **Visualise**: ROC curves, confusion matrix, feature importance, model comparison.

Using a pipeline means imputation and scaling are learned only from training data, which prevents data leakage.

## Results (test set)

**Breast Cancer**
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.929 | 0.951 | 0.996 |
| SVM | 0.983 | 0.952 | 0.976 | 0.996 |
| Random Forest | 0.974 | 0.929 | 0.963 | 0.993 |
| XGBoost | 0.965 | 0.905 | 0.950 | 0.993 |

**Diabetes**
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| XGBoost | 0.740 | 0.556 | 0.600 | 0.820 |
| Random Forest | 0.727 | 0.519 | 0.571 | 0.813 |
| Logistic Regression | 0.688 | 0.481 | 0.520 | 0.810 |
| SVM | 0.708 | 0.463 | 0.526 | 0.805 |

**Heart Disease**
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.853 | 0.893 | 0.848 | 0.958 |
| Random Forest | 0.902 | 0.929 | 0.897 | 0.958 |
| SVM | 0.885 | 0.893 | 0.877 | 0.947 |
| XGBoost | 0.869 | 0.893 | 0.862 | 0.945 |

## Key Findings
- Breast Cancer is the easiest to predict (about 97% accuracy) because the tumour measurements are highly informative.
- Heart Disease reaches about 85-90% accuracy with strong recall (about 0.9).
- Diabetes is the hardest (about 70-74% accuracy). Recall is low (about 0.5), so many diabetic patients would be missed.
- No single algorithm wins everywhere; differences between models are small, especially on the small Heart Disease test set (61 samples).
- In medical screening, **recall** matters more than accuracy, because missing a sick patient is costlier than a false alarm.

## Limitations
- Small datasets, so results vary with the train/test split.
- Models were not calibrated for clinical use. This is a learning project, **not a medical diagnostic tool**.

## How to Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```
If the datasets fail to download, place `diabetes.csv` (target column `Outcome`) and `heart.csv` (target column `target`) in a `data/` folder.

Outputs (metrics CSVs and plots) are saved in the `results/` folder.

## Project Structure
```
├── disease_prediction.py   # full pipeline
├── requirements.txt
├── README.md
├── .gitgnore
└── results/                # metrics and plots (generated)
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn
