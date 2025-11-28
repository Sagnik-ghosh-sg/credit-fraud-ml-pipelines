# 🧠 AI Finance Suite  
### Credit Scoring (GMSC) + Fraud Detection (IEEE-CIS)

A unified machine learning system for **financial risk modeling**, covering:

- **Credit Scoring (GMSC dataset)**
- **Fraud Detection (IEEE-CIS dataset)**  
- **SHAP explainability**
- **Batch + Individual simulation tools**
- **End-to-end training with `master_ai_finance.py`**

This repository is designed as a clean, modular, production-ready pipeline suitable for internships, research projects, or real-world ML engineering tasks.

---

## 📂 Project Structure

.
├── master_ai_finance.py # Main training runner (GMSC + IEEE-CIS)
├── /Credit_Scoring_GMSC
│ ├── src/
│ │ ├── data_loader.py # Load raw + preprocessed datasets
│ │ ├── preprocess.py # Feature cleaning, encoding, scaling
│ │ ├── train.py # Model training & saving
│ │ ├── eval.py # Metrics, ROC-AUC, confusion matrix
│ │ └── predict.py # Simulation tools
│ ├── models/ # Saved CatBoost/LGBM models
│ └── outputs/ # Scores, graphs, reports
│
├── /Fraud_Detection_IEEE
│ ├── src/
│ │ ├── preprocess.py # Feature engineering for IEEE-CIS
│ │ ├── model_zoo.py # LGBM/XGB/CatBoost base models
│ │ ├── stacker.py # Meta-model stacking
│ │ ├── eval.py # Fraud metrics
│ │ └── shap_explain.py # SHAP visualizations
│ ├── models/
│ └── outputs/
│
└── README.md

yaml
Copy code

---

## 🚀 Features

### 🔹 **1. End-to-End Training with 1 Command**
Run all credit + fraud models:

```bash
python master_ai_finance.py
Automatically performs:

Data loading

Preprocessing

Training

Evaluation

Saving models

Generating SHAP explainability

🔹 2. Credit Scoring (GMSC)
Includes:

CatBoost + LightGBM ensemble

Score generation (0–1000)

Feature importance

SHAP summary + dependence plots

Batch prediction on any CSV

🔹 3. Fraud Detection (IEEE-CIS)
Implemented pipeline:

Feature engineering (amount, time, device, email domain)

Base models (CatBoost, LightGBM, XGBoost)

Level-2 meta-model (stacker)

Probabilistic fraud risk scoring

SHAP interpretability

🧪 Simulation Tools
Single Customer Simulation
python
Copy code
from Credit_Scoring_GMSC.src.predict import make_single_prediction
make_single_prediction()
Batch Simulation
python
Copy code
python Credit_Scoring_GMSC/src/predict.py --batch some_file.csv
📊 Explainability (SHAP)
Both pipelines automatically generate:

Summary plots

Bar feature impact

Force plots (optional)

CSV explanation tables

These appear in:

bash
Copy code
Credit_Scoring_GMSC/outputs/shap/
Fraud_Detection_IEEE/outputs/shap/
📦 Installation
bash
Copy code
pip install -r requirements.txt
Recommended Versions:

Python 3.10+

LightGBM ≥ 4.0

CatBoost ≥ 1.2

XGBoost ≥ 2.0

Shap ≥ 0.45

🏁 Quick Start
Train everything:

bash
Copy code
python master_ai_finance.py
