import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from Fraud_Detection_IEEE_CIS.src.utils import load
from Fraud_Detection_IEEE_CIS.src.preprocess import preprocess_test
from Fraud_Detection_IEEE_CIS.src.config import MODEL_DIR as MODEL_DIR_IEEE

def predict_individual(data):
    """
    Accepts a dict for a single transaction
    Returns predicted fraud probability and SHAP explanation (bar chart)
    """
    df = pd.DataFrame([data])
    art = load(MODEL_DIR_IEEE / "stacking_artifacts.joblib")
    feats = art["feats"]
    X = preprocess_test(df, feats)

    # Load base models
    base_models = []
    for name in ["lgb", "xgb", "cat"]:
        try:
            m = load(MODEL_DIR_IEEE / f"{name}_fold1.joblib")
            base_models.append(m)
        except Exception:
            continue
    if not base_models:
        raise RuntimeError("No base models found. Train first.")

    # Ensemble predictions using stacking meta model
    base_preds = np.vstack([m.predict_proba(X)[:, 1] for m in base_models]).T
    meta = load(MODEL_DIR_IEEE / "stacking_meta.joblib")
    final_prob = meta.predict_proba(base_preds)[:, 1]

    print(f"Predicted Fraud Probability: {final_prob[0]:.4f}")

    # SHAP explanation using first base model (LGB)
    explainer = shap.Explainer(base_models[0], X)
    shap_values = explainer(X)

    # Convert SHAP values to array for bar chart
    shap_vals = shap_values.values[0] if hasattr(shap_values, "values") else shap_values[0]
    importance = np.abs(shap_vals)
    sorted_idx = np.argsort(importance)

    # Plot horizontal bar chart of feature importance
    # --- SHAP explanation ---
    model = base_models[0]
    if "lightgbm" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    elif "xgb" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    elif "catboost" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X)

    shap_values = explainer(X)

    # Get SHAP array safely
    if hasattr(shap_values, "values"):
        shap_vals = shap_values.values[0]
    else:
        shap_vals = shap_values[0]

    importance = np.abs(shap_vals)
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(8, 5))
    plt.barh(X.columns[sorted_idx], importance[sorted_idx])
    plt.xlabel("Feature importance (|SHAP value|)")
    plt.title("Feature impact for this transaction")
    plt.tight_layout()
    plt.show()



def predict_from_csv(csv_file):
    """
    Simulate predictions on a CSV file and optionally evaluate metrics
    if ground truth labels are provided.
    """
    df = pd.read_csv(csv_file)
    art = load(MODEL_DIR_IEEE / "stacking_artifacts.joblib")
    feats = art["feats"]
    X = preprocess_test(df, feats)

    base_models = []
    for name in ["lgb", "xgb", "cat"]:
        try:
            m = load(MODEL_DIR_IEEE / f"{name}_fold1.joblib")
            base_models.append(m)
        except Exception:
            continue
    if not base_models:
        raise RuntimeError("No base models found. Train first.")

    base_preds = np.vstack([m.predict_proba(X)[:, 1] for m in base_models]).T
    meta = load(MODEL_DIR_IEEE / "stacking_meta.joblib")
    df["Fraud_Probability"] = meta.predict_proba(base_preds)[:, 1]

    # Evaluate metrics if ground truth column exists\
    target_col = None
    for col in ["isFraud", "target", "label"]:
        if col in df.columns:
            target_col = col
            break

    if target_col:
        y_true = df[target_col]
        y_pred_prob = df["Fraud_Probability"]
        y_pred = (y_pred_prob > 0.5).astype(int)

        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        try:
            print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_prob):.4f}")
        except ValueError:
            print("ROC-AUC: N/A (only one class present)")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Plot confusion matrx
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("No ground truth column found (isFraud/target/label). Skipping metrics.")

    out_file = "fraud_simulation_output.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved batch fraud simulation output to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_individual", action="store_true", help="Single transaction simulation")
    parser.add_argument("--simulate_csv", type=str, help="CSV file with multiple transactions")
    args = parser.parse_args()

    if args.simulate_individual:
        print("Enter feature values for transaction (numeric only). Press Enter for missing values:")
        data = {}
        art = load(MODEL_DIR_IEEE / "stacking_artifacts.joblib")
        feats = art["feats"]
        for c in feats:
            val = input(f"{c}: ")
            try:
                data[c] = float(val) if val.strip() else -999
            except:
                data[c] = -999
        predict_individual(data)
    elif args.simulate_csv:
        predict_from_csv(args.simulate_csv)
    else:
        print("Provide --simulate_individual or --simulate_csv <file>")