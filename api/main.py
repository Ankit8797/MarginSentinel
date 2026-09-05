from fastapi import FastAPI, HTTPException
import mlflow
import json
import pandas as pd
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.schemas import TransactionRequest, PredictionResponse
from src.preprocessing import load_and_preprocess, clean_data
from src.graph_features import GraphFeatureExtractor
from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, THRESHOLD_PATH,
    DEFENSE_ACTIONS, CATEGORICAL_FEATURES
)

# Global State
model = None
extractor = None
threshold = 0.5
is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, extractor, threshold, is_ready
    
    print("Initializing API state...")
    
    try:
        # Load Threshold
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = json.load(f)['optimal_threshold']
            
        # Initialize and Fit Graph Extractor
        df_train, _, _ = load_and_preprocess(force_resplit=False)
        extractor = GraphFeatureExtractor()
        extractor.fit(df_train)
        
        # Load Model from MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            raise RuntimeError("MLflow experiment not found. Run training first.")
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"], max_results=1)
        if runs.empty:
            raise RuntimeError("No MLflow runs found.")
            
        latest_run_id = runs.iloc[0]['run_id']
        model_uri = f"runs:/{latest_run_id}/catboost_model"
        model = mlflow.catboost.load_model(model_uri)
        
        is_ready = True
        print("API successfully initialized.")
    except Exception as e:
        print(f"Failed to initialize API: {e}")
        
    yield
    print("Shutting down API...")

app = FastAPI(title="MarginSentinel Risk Scorer", lifespan=lifespan)

@app.get("/health")
def health_check():
    if is_ready:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Model or graph not loaded yet")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    try:
        # Convert request to DataFrame
        row_dict = request.model_dump()
        df = pd.DataFrame([row_dict])
        
        # Clean Data (handle missing VPA)
        df_clean = clean_data(df)
        
        # Extract Graph Features (this also adds the incoming transaction to the in-memory graph)
        features_df = extractor.transform(df_clean)
        
        # Select Features for Model
        model_features = [
            'address_quality_score', 'cart_value_inr', 'checkout_velocity_seconds',
            'is_discount_applied', 'customer_order_history_count',
            'connected_component_size', 'neighborhood_rto_rate', 'max_customers_per_device'
        ] + CATEGORICAL_FEATURES
        
        X = features_df[model_features]
        
        # Debug Logs
        print(f"--- DEBUG: Incoming Request for Txn {request.transaction_id} ---")
        print(f"X Columns & Dtypes:\n{X.dtypes}")
        print(f"Feature Values:\n{X.iloc[0].to_dict()}")
        
        # Predict Risk Score
        risk_score = float(model.predict_proba(X)[:, 1][0])
        print(f"Raw Predict Proba: {risk_score}")
        print(f"--------------------------------------------------")
        
        # Determine Defense Action
        # Simplistic logic mapping threshold to action
        if risk_score >= threshold:
            action = DEFENSE_ACTIONS["BLOCK_TRANSACTION"]
        elif risk_score >= threshold * 0.7: # Example buffer for UPI preauth
            action = DEFENSE_ACTIONS["REQUIRE_UPI_PREAUTH"]
        else:
            action = DEFENSE_ACTIONS["ALLOW_FREE_COD"]
            
        return PredictionResponse(risk_score=risk_score, defense_action=action)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
