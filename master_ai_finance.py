import argparse
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

# GMSC imports
from Credit_Scoring_GMSC.src.data_loader import load_train, load_test
from Credit_Scoring_GMSC.src.preprocess import prepare_train, prepare_test
from Credit_Scoring_GMSC.src.train import train_and_save
from Credit_Scoring_GMSC.src.eval import evaluate_test
from Credit_Scoring_GMSC.src.predict import make_submission
from Credit_Scoring_GMSC.src.utils import load

# IEEE imports
from Fraud_Detection_IEEE_CIS.src.data_loader import load_train as load_train_ieee
from Fraud_Detection_IEEE_CIS.src.preprocess import preprocess_train, preprocess_test
from Fraud_Detection_IEEE_CIS.src.train_stack import train_stack
from Fraud_Detection_IEEE_CIS.src.predict import predict_submission
from Fraud_Detection_IEEE_CIS.src.utils import load as load_ieee

from Credit_Scoring_GMSC.src.config import MODEL_DIR as MODEL_DIR_GMSC
from Fraud_Detection_IEEE_CIS.src.config import MODEL_DIR as MODEL_DIR_IEEE

def shap_plot(model, X, dataset_name, top_n=10):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=False)
    plt.title(f"{dataset_name} Top {top_n} Feature Importance (SHAP)")
    plt.tight_layout()
    out_file = f"{dataset_name}_shap_summary.png"
    plt.savefig(out_file)
    print(f"Saved SHAP plot to {out_file}")
    plt.close()

def run_gmsc():
    print("=== GMSC Credit Scoring Pipeline ===")
    # train models
    train_and_save()
    # evaluate
    evaluate_test()
    # predict submission
    make_submission()
    # SHAP plots
    print("Generating SHAP summary for CatBoost...")
    cat = load(MODEL_DIR_GMSC / "cat_final.joblib")
    df_train = load_train()
    X_train, _, _, _ = prepare_train(df_train)
    shap_plot(cat, X_train, "GMSC")

def run_ieee():
    print("=== IEEE-CIS Fraud Detection Pipeline ===")
    # train stacking model
    train_stack()
    # predict submission
    predict_submission()
    # SHAP plots for LGB as example
    print("Generating SHAP summary for LGB...")
    lgb = load_ieee(MODEL_DIR_IEEE / "lgb_fold1.joblib")
    df_train = load_train_ieee()
    X_train, y_train, feats = preprocess_train(df_train)
    shap_plot(lgb, X_train, "IEEE_Fraud")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["gmsc","ieee"], help="Dataset to run full pipeline")
    args = parser.parse_args()

    if args.dataset == "gmsc":
        run_gmsc()
    elif args.dataset == "ieee":
        run_ieee()
    else:
        print("Specify --dataset gmsc or ieee")
