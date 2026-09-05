import time
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.main import app
from src.preprocessing import load_and_preprocess

# We use TestClient as a fixture to trigger lifespan
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def sample_test_data():
    _, _, df_test = load_and_preprocess(force_resplit=False)
    return df_test

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_valid_payload(client):
    payload = {
        "transaction_id": "pay_test_001",
        "customer_id": "USR_test_001",
        "client_canvas_hash": "hash_xyz",
        "app_set_id": "app_xyz",
        "vpa_handle_hash": "vpa_xyz",
        "delivery_pincode": "110001",
        "address_quality_score": 4.0,
        "payment_method": "UPI",
        "cart_value_inr": 3000.0,
        "category_volatility": "Low (Groceries)",
        "checkout_velocity_seconds": 120.0,
        "is_discount_applied": 0,
        "customer_order_history_count": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data['risk_score'] <= 1.0
    assert data['defense_action'] in ["ALLOW_FREE_COD", "REQUIRE_UPI_PREAUTH", "BLOCK_TRANSACTION"]

def test_missing_field(client):
    payload = {
        "transaction_id": "pay_test_002",
        # missing customer_id
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_latency(client):
    payload = {
        "transaction_id": "pay_test_003",
        "customer_id": "USR_test_003",
        "client_canvas_hash": "novel_hash",
        "app_set_id": "novel_app",
        "vpa_handle_hash": None,
        "delivery_pincode": "560001",
        "address_quality_score": 4.5,
        "payment_method": "COD",
        "cart_value_inr": 1500.0,
        "category_volatility": "Medium (Electronics)",
        "checkout_velocity_seconds": 60.0,
        "is_discount_applied": 1,
        "customer_order_history_count": 0
    }
    
    start_time = time.perf_counter()
    response = client.post("/predict", json=payload)
    end_time = time.perf_counter()
    
    assert response.status_code == 200
    latency_ms = (end_time - start_time) * 1000
    assert latency_ms < 500, f"Latency {latency_ms}ms exceeded 500ms limit"

def test_golden_set_regression(sample_test_data, client):
    """
    Send one known obvious ring transaction and one known legit transaction,
    assert ring score > legit score.
    """
    df = sample_test_data
    
    # Find a legit row
    legit_row = df[(df['is_fraud_ring'] == 0) & (df['is_rto'] == 0)].iloc[0]
    # Find an obvious ring row
    # The generator defines obvious rings as those with ring_id starting with RING_ (but not RING_CAMO_)
    ring_row = df[(df['is_fraud_ring'] == 1) & (df['ring_id'].str.startswith('RING_', na=False)) & (~df['ring_id'].str.startswith('RING_CAMO_', na=False))].iloc[0]
    
    def row_to_payload(row):
        return {
            "transaction_id": str(row['transaction_id']),
            "customer_id": str(row['customer_id']),
            "client_canvas_hash": str(row['client_canvas_hash']),
            "app_set_id": str(row['app_set_id']),
            "vpa_handle_hash": None if pd.isna(row['vpa_handle_hash']) or row['vpa_handle_hash'] == 'NO_VPA' else str(row['vpa_handle_hash']),
            "delivery_pincode": str(row['delivery_pincode']),
            "address_quality_score": float(row['address_quality_score']),
            "payment_method": str(row['payment_method']),
            "cart_value_inr": float(row['cart_value_inr']),
            "category_volatility": str(row['category_volatility']),
            "checkout_velocity_seconds": float(row['checkout_velocity_seconds']),
            "is_discount_applied": int(row['is_discount_applied']),
            "customer_order_history_count": int(row['customer_order_history_count'])
        }
        
    legit_payload = row_to_payload(legit_row)
    ring_payload = row_to_payload(ring_row)
    
    resp_legit = client.post("/predict", json=legit_payload)
    resp_ring = client.post("/predict", json=ring_payload)
    
    assert resp_legit.status_code == 200
    assert resp_ring.status_code == 200
    
    legit_score = resp_legit.json()['risk_score']
    ring_score = resp_ring.json()['risk_score']
    
    # The obvious ring MUST be scored higher than the clean legit transaction
    assert ring_score > legit_score, f"Regression! Ring score ({ring_score}) <= Legit score ({legit_score})"
