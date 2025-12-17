import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report
)
from Credit_Scoring_GMSC.src.utils import load
from Credit_Scoring_GMSC.src.preprocess import prepare_test
from Credit_Scoring_GMSC.src.config import MODEL_DIR, NUMERIC_COLS


def predict_individual(data):
    """
    Accepts a dict of feature values for a single individual
    Returns predicted probability, credit score (300-850), SHAP plot
    """
    df = pd.DataFrame([data])
    pre = load(MODEL_DIR / "preproc.joblib")
    X = prepare_test(df, pre["imputer"], pre["scaler"])

    cat = load(MODEL_DIR / "cat_final.joblib")
    lgb = load(MODEL_DIR / "lgb_final.joblib")

    p_cat = cat.predict_proba(X)[:, 1]
    p_lgb = lgb.predict_proba(X)[:, 1]
    prob = 0.5 * p_cat + 0.5 * p_lgb

    # Map probability to credit score (simple linear scaling 300-850)
    score = int(850 - (550 * prob))
    print(f"Predicted Default Probability: {prob[0]:.4f}")
    print(f"Estimated Credit Score: {score}")

    # SHAP explanation
    explainer = shap.TreeExplainer(cat)
    shap_values = explainer.shap_values(X)

    shap_vals = shap_values[0]
    importance = np.abs(shap_vals)
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(8, 5))
    plt.barh(X.columns[sorted_idx], importance[sorted_idx])
    plt.xlabel("Feature importance (|SHAP value|)")
    plt.title("Feature impact for this individual")
    plt.tight_layout()
    plt.show()


def predict_from_csv(csv_file):
    """
    Reads CSV, predicts probabilities and credit scores, and 
    evaluates model performance if 'SeriousDlqin2years' column exists.
    """
    df = pd.read_csv(csv_file)
    pre = load(MODEL_DIR / "preproc.joblib")
    X = prepare_test(df, pre["imputer"], pre["scaler"])

    cat = load(MODEL_DIR / "cat_final.joblib")
    lgb = load(MODEL_DIR / "lgb_final.joblib")

    p_cat = cat.predict_proba(X)[:, 1]
    p_lgb = lgb.predict_proba(X)[:, 1]
    prob = 0.5 * p_cat + 0.5 * p_lgb

    df["Default_Probability"] = prob
    df["Credit_Score"] = (850 - (550 * prob)).astype(int)

    # If actual labels are available, calculate metrics
    if "SeriousDlqin2yrs" in df.columns:
        y_true = df["SeriousDlqin2yrs"]
        y_pred = (prob >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, prob)

        print("\nModel Evaluation Metrics:")
        print("---------------------------")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Save metrics to file for record
        metrics_out = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC AUC": auc,
            "Confusion_Matrix": cm.tolist()
        }
        metrics_df = pd.DataFrame([metrics_out])
        metrics_df.to_csv("credit_simulation_metrics.csv", index=False)
        print("\nSaved metrics summary to credit_simulation_metrics.csv")

    out_file = "credit_simulation_output.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved batch simulation output to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_individual", action="store_true", help="Interactive single person simulation")
    parser.add_argument("--simulate_csv", type=str, help="CSV file with multiple individuals (optionally with 'SeriousDlqin2years' column)")
    args = parser.parse_args()

    if args.simulate_individual:
        print("Enter feature values (numeric only). Press Enter for missing values:")
        data = {}
        for c in NUMERIC_COLS:
            val = input(f"{c}: ")
            data[c] = float(val) if val.strip() else np.nan
        predict_individual(data)

    elif args.simulate_csv:
        predict_from_csv(args.simulate_csv)

    else:
        print("Provide --simulate_individual or --simulate_csv <file>")
