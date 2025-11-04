# 🩺 Hypertension Risk Classification Using SVM

This project implements a **Support Vector Machine (SVM)** to classify hypertension (heart disease) risk based on patient data.

The primary focus of this analysis is an experiment to compare two model-finding approaches:

1.  **Human-Selected Model:** A specific SVM configuration (`kernel='poly'`, `C=10`, `gamma=0.7`, `degree=3`) based on parameters from a journal reference.
2.  **Machine-Automated Models:** The best SVM models found using a hyperparameter search (GridSearch) for the `rbf` and `linear` kernels.

The goal is to validate whether the journal-referenced model is superior to models found through an automated tuning process on this specific dataset.

---

## 🎯 Project Objectives
- To build a machine learning pipeline to predict hypertension risk (`target`).
- To implement a specific SVM `poly` model as a baseline (based on a journal reference).
- To find the best-performing SVM `rbf` and `linear` models through hyperparameter tuning.
- To compare the performance of the "Human-Selected" vs. "Machine-Automated" models to determine which approach is most accurate.

---

## 📊 Dataset
- **Source:** `hypertension_data.csv` (Based on the UCI Heart Disease dataset).
- **Target Variable:** `target` (0 = Low Risk, 1 = High Risk).
- **Continuous Features:** `age`, `trestbps` (Resting Blood Pressure), `chol` (Cholesterol), `thalach` (Max Heart Rate), `oldpeak`.
- **Categorical Features:** `sex`, `cp` (Chest Pain Type), `fbs` (Fasting Blood Sugar), `restecg`, `exang`, `slope`, `ca`, `thal`.

---

## ⚙️ Data Preprocessing Pipeline
Before model training, the data undergoes the following preprocessing steps:

1.  **Encoding:** Categorical features (`sex`, `cp`, `thal`, etc.) are converted into numerical representations using **One-Hot Encoding** (`pd.get_dummies` with `drop_first=True`).
2.  **Standardization:** Continuous features (`age`, `trestbps`, `chol`, etc.) are scaled using **`StandardScaler`** to normalize their value ranges.
3.  **Data Splitting:** The dataset is split into 80% training data and 20% testing data using `train_test_split`. This process uses `stratify=y` to ensure the target class proportions are maintained in both sets.

---

## 🧩 Experiment Design: Human vs. Machine

This experiment compares three different model scenarios, all trained on the same 80% training set and evaluated on the same 20% test set.

### Scenario 1: The Baseline (Human-Selected Model)
This model acts as the benchmark, using parameters derived from external research/journal.

-   **Algorithm:** `SVC` (Support Vector Classification)
-   **Kernel:** `poly`
-   **Parameters (Fixed):**
    -   `C = 10`
    -   `gamma = 0.7`
    -   `degree = 3`

### Scenario 2: The Challenger (Machine-Automated - RBF)
This seeks the best parameters for the popular RBF kernel.

-   **Algorithm:** `SVC`
-   **Kernel:** `rbf`
-   **Parameter Grid (Searched):**
    -   `C`: [1, 10, 100]
    -   `gamma`: [0.01, 0.1, 1]

### Scenario 3: The Challenger (Machine-Automated - Linear)
This seeks the best parameters for the Linear kernel, which is often effective on high-dimensional data.

-   **Algorithm:** `SVC`
-   **Kernel:** `linear`
-   **Parameter Grid (Searched):**
    -   `C`: [1, 10, 100]

---

## 📈 Evaluation Metrics
The performance of all three scenarios is compared head-to-head using the 20% test set. The "Champion" model is the one with the highest **Accuracy**.

The primary metrics used for comparison are:
-   **Accuracy**
-   **Precision**
-   **Recall**
-   **F1-Score**

A `classification_report` and `confusion_matrix` are also generated for an in-depth analysis of each model's performance. The "Champion" (best-performing) model is then saved to a `best_hypertension_artifacts.pkl` file for future use.

---
