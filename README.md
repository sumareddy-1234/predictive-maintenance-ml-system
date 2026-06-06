# Predictive Maintenance ML System

A production-grade end-to-end predictive maintenance machine learning system built from raw sensor data ingestion to a deployed, monitored model serving real-time failure risk predictions.

## Overview

This system uses the **AI4I 2020 Predictive Maintenance Dataset** (10,000 samples, 14 features, ~3.4% failure rate) to predict machine failures before they occur. It implements a full ML pipeline including data ingestion, physics-based feature engineering, ensemble model training with class imbalance handling, cost-sensitive threshold optimization, SHAP explainability, and production drift monitoring.

### Key Results

| Metric | Value |
|---|---|
| Best Model | Soft Voting Classifier (LR + RF + GB) |
| Validation F1 | 0.8710 |
| Optimal Threshold | 0.45 (cost-minimized) |
| Test Accuracy | 0.982 |
| Test MCC | 0.763 |
| Test ROC AUC | 0.967 |

## Project Structure

```
predictive_maintenance/
    data/
        raw/                          # Raw AI4I 2020 dataset
        processed/                    # Cleaned and encoded data
        features/                     # Engineered feature matrices
    notebooks/
        01_eda.ipynb                  # Exploratory Data Analysis
        02_feature_engineering.ipynb  # Feature engineering walkthrough
        03_model_training.ipynb       # Model training and tuning
        04_model_evaluation.ipynb     # Evaluation and explainability
    src/
        data_pipeline/
            __init__.py
            ingestion.py              # Data loading and validation
            preprocessing.py          # Encoding, splitting, scaling
            feature_store.py          # Physics-based feature engineering
        models/
            __init__.py
            baseline.py               # Logistic Regression, Decision Tree
            advanced.py               # Random Forest, Gradient Boosting, multi-label
            ensemble.py               # Voting, Stacking classifiers
        evaluation/
            __init__.py
            metrics.py                # Classification, cost-sensitive, multi-label metrics
            explainability.py         # SHAP, permutation importance
        serving/
            __init__.py
            predictor.py              # PredictiveMaintenancePredictor class
            drift_monitor.py          # PSI-based drift detection
    tests/
        test_preprocessing.py
        test_features.py
        test_models.py
        test_metrics.py
    configs/
        model_config.yaml             # Model and evaluation configuration
        pipeline_config.yaml          # Data paths and drift thresholds
    requirements.txt
    README.md
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Quick Start

### Run the Pipeline

```bash
python pipeline.py
```

### Run Tests

```bash
python -m pytest tests/
```

## System Architecture

### Phase 1: Data Ingestion & Preprocessing
- **Ingestion**: Loads the CSV, validates all 14 expected columns
- **Encoding**: Ordinal encoding of product types (L=0, M=1, H=2)
- **Splitting**: Stratified 3-way split (70% train, 10% val, 20% test)
- **Scaling**: StandardScaler fit on training data only to prevent leakage

### Phase 2: Feature Engineering
Five domain-specific physics features are engineered:
- **power_watts**: Torque × angular velocity (2πRPM/60)
- **temp_delta_K**: Process temperature − Air temperature
- **tool_wear_torque_interaction**: Tool wear × Torque
- **tool_wear_rpm_interaction**: Tool wear × RPM
- **strain_index**: (Torque × Tool wear) / (RPM + 1)

### Phase 3: Model Development
- **Baseline**: Logistic Regression (balanced), Decision Tree
- **Advanced**: Random Forest (balanced), Gradient Boosting (sample-weighted)
- **Ensemble**: Soft Voting Classifier, Stacking Classifier with LR meta-learner
- **Multi-label**: MultiOutputClassifier for predicting individual failure modes (TWF, HDF, PWF, OSF, RNF)
- **Tuning**: RandomizedSearchCV with F1 scoring, 20 iterations, 5-fold CV

### Phase 4: Evaluation & Explainability
- **Metrics**: Accuracy, macro precision/recall, F1, ROC AUC, MCC, confusion matrix
- **Threshold optimization**: Cost-sensitive analysis (FN=$5,000 vs FP=$200)
- **SHAP**: TreeExplainer for feature attribution on failure predictions
- **Permutation importance**: Model-agnostic feature ranking

### Phase 5: Serving & Drift Monitoring
- **PredictiveMaintenancePredictor**: Production inference class with single and batch prediction
- **Risk levels**: LOW (<0.3), MEDIUM (0.3–0.6), HIGH (≥0.6)
- **PSI monitoring**: Population Stability Index for detecting feature and prediction drift
- **KL divergence**: Additional distribution shift metric

## Class Imbalance Strategy

The dataset has a ~3.4% failure rate (339 failures out of 10,000 samples). This severe imbalance is handled through:
- `class_weight="balanced"` for all sklearn models that support it
- `compute_sample_weight("balanced")` for Gradient Boosting
- Cost-sensitive threshold tuning (optimal threshold = 0.45 vs default 0.5)
- SMOTE available as a strategy option (applied only after splitting to avoid data leakage)

## Cost Matrix

| Outcome | Cost | Rationale |
|---|---|---|
| True Positive | -$500 | Savings from preventing unplanned failure |
| True Negative | $0 | Normal operation, no cost |
| False Positive | $200 | Unnecessary inspection cost |
| False Negative | $5,000 | Catastrophic unplanned downtime |

## Technical Stack

- **pandas** ≥ 1.5.0 — Data manipulation
- **numpy** ≥ 1.23.0 — Numerical computation
- **scikit-learn** ≥ 1.2.0 — ML models and evaluation
- **imbalanced-learn** ≥ 0.10.0 — SMOTE for class imbalance
- **shap** ≥ 0.41.0 — Model explainability
- **xgboost** ≥ 1.7.0 — Gradient boosting alternative
- **lightgbm** ≥ 3.3.0 — Gradient boosting alternative
- **matplotlib** ≥ 3.6.0 — Visualization
- **seaborn** ≥ 0.12.0 — Statistical visualization
- **scipy** ≥ 1.10.0 — Scientific computing
- **pyyaml** ≥ 6.0 — Configuration parsing
- **joblib** ≥ 1.2.0 — Model serialization

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 data points, 14 features, CC BY 4.0 license.

Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

### Features
- **UDI**: Unique identifier (1–10,000)
- **Product ID**: Quality variant letter + serial number
- **Type**: Product quality variant (L/M/H)
- **Air temperature [K]**: Ambient temperature
- **Process temperature [K]**: Process temperature
- **Rotational speed [rpm]**: Spindle speed
- **Torque [Nm]**: Torque applied
- **Tool wear [min]**: Cumulative tool usage time
- **Machine failure**: Binary target variable
- **TWF/HDF/PWF/OSF/RNF**: Individual failure mode indicators
