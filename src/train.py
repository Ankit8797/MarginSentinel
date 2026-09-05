import os
import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, THRESHOLD_PATH,
    FP_COST, FN_COST, CATEGORICAL_FEATURES, RANDOM_SEED, REPORTS_DIR
)
from src.preprocessing import load_and_preprocess
from src.graph_features import GraphFeatureExtractor

def calculate_financial_cost(y_true, y_pred):
    """Calculate the financial cost of a confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fp * FP_COST) + (fn * FN_COST)

def find_optimal_threshold(y_true, y_proba):
    """Sweep thresholds from 0 to 1 to find the one that minimizes cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    
    best_threshold = 0.5
    min_cost = float('inf')
    
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        cost = calculate_financial_cost(y_true, y_pred)
        costs.append(cost)
        if cost < min_cost:
            min_cost = cost
            best_threshold = th
            
    return best_threshold, min_cost, thresholds, costs

def train():
    print("Loading datasets...")
    df_train, df_val, df_test = load_and_preprocess(force_resplit=False)
    
    print("Extracting graph features...")
    extractor = GraphFeatureExtractor()
    train_feat = extractor.fit_transform_sequential(df_train)
    val_feat = extractor.transform(df_val)
    
    # Define features to use
    features = [
        'address_quality_score', 'cart_value_inr', 'checkout_velocity_seconds',
        'is_discount_applied', 'customer_order_history_count',
        'connected_component_size', 'neighborhood_rto_rate', 'max_customers_per_device'
    ] + CATEGORICAL_FEATURES
    
    X_train = train_feat[features]
    y_train = train_feat['is_fraud_ring'] # Target is is_fraud_ring as per prompt
    
    X_val = val_feat[features]
    y_val = val_feat['is_fraud_ring']
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        print("Training CatBoostClassifier...")
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights='Balanced',
            cat_features=CATEGORICAL_FEATURES,
            random_seed=RANDOM_SEED,
            verbose=50,
            early_stopping_rounds=50
        )
        
        train_pool = Pool(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
        val_pool = Pool(X_val, y_val, cat_features=CATEGORICAL_FEATURES)
        
        model.fit(train_pool, eval_set=val_pool)
        
        print("Evaluating on Validation Set...")
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        pr_auc = average_precision_score(y_val, y_val_proba)
        
        print("Finding optimal threshold for financial cost...")
        opt_th, min_cost, thresholds, costs = find_optimal_threshold(y_val, y_val_proba)
        
        print(f"Optimal Threshold: {opt_th:.2f} (Cost: {min_cost} INR)")
        
        # Log to MLflow
        mlflow.log_params({
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'auto_class_weights': 'Balanced',
            'fp_cost_inr': FP_COST,
            'fn_cost_inr': FN_COST
        })
        
        mlflow.log_metrics({
            'val_pr_auc': pr_auc,
            'optimal_threshold': opt_th,
            'min_val_cost': min_cost
        })
        
        # Plot cost curve
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, costs)
        plt.axvline(opt_th, color='r', linestyle='--', label=f'Optimal: {opt_th:.2f}')
        plt.title('Financial Cost vs. Decision Threshold (Validation)')
        plt.xlabel('Threshold')
        plt.ylabel('Cost (INR)')
        plt.legend()
        cost_curve_path = REPORTS_DIR / 'cost_curve_val.png'
        plt.savefig(cost_curve_path)
        plt.close()
        
        mlflow.log_artifact(str(cost_curve_path))
        
        # Save threshold
        with open(THRESHOLD_PATH, 'w') as f:
            json.dump({'optimal_threshold': opt_th}, f)
            
        # Log Model
        mlflow.catboost.log_model(model, artifact_path="catboost_model")
        
        print(f"Run ID {run.info.run_id} completed. Model logged to MLflow.")
        
if __name__ == '__main__':
    train()
