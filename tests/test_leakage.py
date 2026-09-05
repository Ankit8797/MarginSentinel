import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.graph_features import GraphFeatureExtractor

def test_no_leakage_in_graph_features():
    """
    Ensure neighborhood RTO rate does not leak the target row's own label
    or future labels.
    """
    # Create mock training data
    df_train = pd.DataFrame([
        {'transaction_id': 'pay_1', 'customer_id': 'USR_1', 'client_canvas_hash': 'hash1', 'app_set_id': 'app1', 'vpa_handle_hash': 'NO_VPA', 'is_rto': 1},
        {'transaction_id': 'pay_2', 'customer_id': 'USR_2', 'client_canvas_hash': 'hash1', 'app_set_id': 'app1', 'vpa_handle_hash': 'NO_VPA', 'is_rto': 0},
    ])
    
    extractor = GraphFeatureExtractor()
    train_feat_df = extractor.fit_transform_sequential(df_train)
    
    # 1. Check leave-one-out on training row
    # In sequential transform, pay_1 is processed first: comp_size=1, nbr_rto_rate = global_mean (0.5)
    feat_train_1 = train_feat_df.iloc[0]
    assert feat_train_1['neighborhood_rto_rate'] == 0.5
    
    # pay_2 is processed second: comp_size=2. Neighbor is pay_1 (is_rto=1).
    feat_train_2 = train_feat_df.iloc[1]
    assert feat_train_2['neighborhood_rto_rate'] == 1.0
    
    # 2. Check strict non-leakage for a test row
    # A test row connects to the component, but its label is NOT in extractor's train_txn_labels
    df_test = pd.DataFrame([
        {'transaction_id': 'pay_3', 'customer_id': 'USR_3', 'client_canvas_hash': 'hash1', 'app_set_id': 'app1', 'vpa_handle_hash': 'NO_VPA', 'is_rto': 1}
    ])
    
    test_feat_df = extractor.transform(df_test)
    feat_test_3 = test_feat_df.iloc[0]
    
    # The component now has USR_1, USR_2, and USR_3.
    # Training transactions in component: pay_1 (1), pay_2 (0).
    # Since pay_3 is NOT a training transaction, its neighborhood RTO rate is (1+0)/2 = 0.5.
    # Its own label (1) is completely ignored.
    assert feat_test_3['neighborhood_rto_rate'] == 0.5
    assert feat_test_3['connected_component_size'] == 3
