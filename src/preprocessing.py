import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import sys
from pathlib import Path

# Add project root to sys.path so config can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATASET_PATH, SPLIT_INDICES_PATH, MODELS_DIR,
    RANDOM_SEED, TRAIN_RATIO, CATEGORICAL_FEATURES
)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and prepare categorical types."""
    df_clean = df.copy()
    
    # Handle missing vpa_handle_hash explicitly
    df_clean['vpa_handle_hash'] = df_clean['vpa_handle_hash'].fillna("NO_VPA")
    
    # Fill remaining object columns with 'UNKNOWN' just in case
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("UNKNOWN")
            df_clean[col] = df_clean[col].astype(str)
            
    return df_clean


def perform_split(df: pd.DataFrame) -> dict:
    """
    Perform 3-way GroupShuffleSplit ensuring rings stay together.
    Legit transactions get their own unique group via transaction_id.
    """
    # Create a grouping variable where each legit user is their own group
    groups = df['ring_id'].fillna(df['transaction_id'])
    
    # We want e.g. 70% Train, 15% Val, 15% Test
    val_test_ratio = 1.0 - TRAIN_RATIO
    
    # Split 1: Train vs (Val + Test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=RANDOM_SEED)
    train_idx, val_test_idx = next(gss1.split(df, groups=groups))
    
    # Split 2: Val vs Test (from the val_test_idx pool)
    # Split the remaining pool in half
    val_test_groups = groups.iloc[val_test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    val_sub_idx, test_sub_idx = next(gss2.split(df.iloc[val_test_idx], groups=val_test_groups))
    
    # Map back to original dataframe indices
    val_idx = val_test_idx[val_sub_idx]
    test_idx = val_test_idx[test_sub_idx]
    
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }


def load_and_preprocess(force_resplit: bool = False):
    """
    Loads raw data, cleans it, and returns train, val, test DataFrames.
    If split_indices.json exists (and force_resplit=False), loads it.
    Otherwise, computes the split and saves it.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run Phase 1.")
        
    df = pd.read_csv(DATASET_PATH)
    df_clean = clean_data(df)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if os.path.exists(SPLIT_INDICES_PATH) and not force_resplit:
        print(f"Loading existing split indices from {SPLIT_INDICES_PATH}")
        with open(SPLIT_INDICES_PATH, 'r') as f:
            indices = json.load(f)
    else:
        print("Performing new GroupShuffleSplit...")
        indices = perform_split(df_clean)
        with open(SPLIT_INDICES_PATH, 'w') as f:
            json.dump(indices, f)
        print(f"Saved split indices to {SPLIT_INDICES_PATH}")
        
    df_train = df_clean.iloc[indices['train']].copy()
    df_val = df_clean.iloc[indices['val']].copy()
    df_test = df_clean.iloc[indices['test']].copy()
    
    return df_train, df_val, df_test

if __name__ == '__main__':
    train, val, test = load_and_preprocess(force_resplit=True)
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")
