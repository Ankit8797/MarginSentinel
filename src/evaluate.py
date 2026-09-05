import os
import json
import pandas as pd
import numpy as np
import shap
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, THRESHOLD_PATH,
    FP_COST, FN_COST, CATEGORICAL_FEATURES, REPORTS_DIR, MODELS_DIR
)
from src.preprocessing import load_and_preprocess
from src.graph_features import GraphFeatureExtractor
from src.eda import get_population

def calculate_financial_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (fp * FP_COST) + (fn * FN_COST)

def evaluate():
    print("Loading test dataset...")
    df_train, df_val, df_test = load_and_preprocess(force_resplit=False)
    
    print("Extracting graph features for test set...")
    extractor = GraphFeatureExtractor()
    extractor.fit(df_train) # Fit ONLY on train
    test_feat = extractor.transform(df_test) # Transform test
    
    features = [
        'address_quality_score', 'cart_value_inr', 'checkout_velocity_seconds',
        'is_discount_applied', 'customer_order_history_count',
        'connected_component_size', 'neighborhood_rto_rate', 'max_customers_per_device'
    ] + CATEGORICAL_FEATURES
    
    X_test = test_feat[features]
    y_test = test_feat['is_fraud_ring']
    
    # Identify population for cost breakdown
    test_feat['population'] = test_feat.apply(get_population, axis=1)
    
    print("Loading model and threshold...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Get latest run
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found.")
        
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
    if runs.empty:
        raise ValueError("No MLflow runs found.")
        
    latest_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{latest_run_id}/catboost_model"
    model = mlflow.catboost.load_model(model_uri)
    
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = json.load(f)['optimal_threshold']
        
    print("Predicting...")
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    test_feat['risk_score'] = y_test_proba
    test_feat['defense_action'] = y_test_pred
    
    # 1. Metrics
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    pr_auc = average_precision_score(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    
    # 3. Financial Cost by Segment
    cost_breakdown = {}
    total_cost = 0
    for pop in test_feat['population'].unique():
        pop_mask = test_feat['population'] == pop
        pop_cost = calculate_financial_cost(y_test[pop_mask], y_test_pred[pop_mask])
        cost_breakdown[pop] = pop_cost
        total_cost += pop_cost
        
    # 4. Calibration Check
    test_feat['score_bucket'] = pd.qcut(test_feat['risk_score'], q=10, duplicates='drop')
    calibration = test_feat.groupby('score_bucket')['is_rto'].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=calibration, x='score_bucket', y='is_rto')
    plt.xticks(rotation=45)
    plt.title('Calibration Check: Empirical RTO vs Predicted Risk Score')
    plt.tight_layout()
    calib_path = REPORTS_DIR / 'calibration_plot.png'
    plt.savefig(calib_path)
    plt.close()
    
    # 5. SHAP Plots
    print("Generating SHAP plots...")
    explainer = shap.TreeExplainer(model)
    # Using a sample to speed up SHAP globally
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    shap_values = explainer(shap_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, show=False)
    shap_path = REPORTS_DIR / 'shap_summary.png'
    plt.savefig(shap_path, bbox_inches='tight')
    plt.close()
    
    # Force plots for TP and FP
    tp_idx = test_feat[(test_feat['is_fraud_ring'] == 1) & (test_feat['defense_action'] == 1)].index
    fp_idx = test_feat[(test_feat['is_fraud_ring'] == 0) & (test_feat['defense_action'] == 1)].index
    
    if len(tp_idx) > 0:
        tp_row = X_test.loc[tp_idx[0]:tp_idx[0]]
        shap_val_tp = explainer(tp_row)
        shap.plots.waterfall(shap_val_tp[0], show=False)
        plt.savefig(REPORTS_DIR / 'shap_tp.png', bbox_inches='tight')
        plt.close()
        
    if len(fp_idx) > 0:
        fp_row = X_test.loc[fp_idx[0]:fp_idx[0]]
        shap_val_fp = explainer(fp_row)
        shap.plots.waterfall(shap_val_fp[0], show=False)
        plt.savefig(REPORTS_DIR / 'shap_fp.png', bbox_inches='tight')
        plt.close()
        
    # Generate Markdown Report
    report_md = f"""# Model Evaluation Report

## Metrics
- **Precision:** {precision:.4f}
- **Recall:** {recall:.4f}
- **F1 Score:** {f1:.4f}
- **PR-AUC:** {pr_auc:.4f}
- **ROC-AUC:** {roc_auc:.4f}

## Confusion Matrix
| | Predicted Negative (Allow) | Predicted Positive (Block) |
|---|---|---|
| **Actual Negative (Legit/Power)** | {cm[0, 0]} | {cm[0, 1]} |
| **Actual Positive (Ring)** | {cm[1, 0]} | {cm[1, 1]} |

## Financial Cost Breakdown
Total Financial Cost on Test Set: **{total_cost} INR**

| Segment | Cost (INR) |
|---|---|
"""
    for pop, cost in cost_breakdown.items():
        report_md += f"| {pop} | {cost} |\n"
        
    report_md += f"\n*Cost defined as FP_COST = {FP_COST} INR, FN_COST = {FN_COST} INR*\n"
    
    report_path = REPORTS_DIR / 'evaluation_report.md'
    with open(report_path, 'w') as f:
        f.write(report_md)
        
    # Log to MLflow
    with mlflow.start_run(run_id=latest_run_id):
        mlflow.log_metrics({
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_pr_auc': pr_auc,
            'test_roc_auc': roc_auc,
            'test_total_cost': total_cost
        })
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(calib_path))
        mlflow.log_artifact(str(shap_path))
        
    print(f"Evaluation complete. Report saved to {report_path}")

if __name__ == '__main__':
    evaluate()
