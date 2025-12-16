# Mercor Cheating Detection

This project implements a robust machine learning pipeline to detect cheating users based on social graph connections and user features.

## Methodology

### 1. Social Risk Scoring (The Core Innovation)
The key driver of this model is the **Social Risk Score**. We derived this by treating cheating as a "contagious" property in the social graph.
- **Propagation**: We propagated risk scores iteratively (10 iterations) through the social graph.
- **Clamping**: Known "Cheaters" (1) and "Clean" users (0) were clamped to their labels to act as sources/sinks of risk.
- **Leakage Prevention**: To ensure valid evaluation, risk scores were calculated *separately* for each fold/split using only that training set's labels.

### 2. Model Selection & Benchmarking
We conducted a rigorous benchmark of multiple models, ensuring no data leakage occurred during validation.

| Model | MSE (Lower is Better) | AUC | Decision |
| :--- | :--- | :--- | :--- |
| **XGBoost (5-Fold)** | **0.1001** | **0.931** | **Selected** |
| LightGBM | 0.20+ | 0.888 | Rejected |
| Random Forest | High | 0.729 | Rejected |
| Logistic Regression | High | 0.664 | Rejected |

**Result**: XGBoost significantly outperformed others, proving that while `risk_factor` is dominant, complex non-linear interactions captured by Gradient Boosting are essential.

### 3. Final Architecture: K-Fold Ensemble
- **Model**: XGBoost Classifier (`max_depth=12`, `lr=0.01`).
- **Ensemble**: 5-Fold Cross Validation Ensemble.
- **Process**:
    1. Split data into 5 folds.
    2. For each fold, recalculate social risk scores from scratch (using only that fold's training data).
    3. Train an independent XGBoost model.
    4. Average the predictions of all 5 models on the Test set.

## Project Structure

### `main/` (Production Code)
- `code.py`: **Main production script**. Runs the end-to-end 5-Fold Ensemble and generates `submission_with_risk.csv`.
- `fine_tune_model.py`: Script for fine-tuning.
- Data Files: `social_graph.csv`, `train.csv`, `test.csv`.

### `benchmark/` (Analysis & Verification)
- `model_comparison.py`: Benchmark script comparing XGBoost, LightGBM, Random Forest, and Logistic Regression.
- `xgboost_kfold.py`: Validation script confirming the robust MSE of 0.1001.

### `other_models/` (Experiments)
- `run_lgbm.py`: Independent run script for LightGBM.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy networkx scikit-learn xgboost lightgbm
   ```
2. Run the main script:
   ```bash
   cd main
   python code.py
   ```
   Output will be saved to `main/submission_with_risk.csv`.
