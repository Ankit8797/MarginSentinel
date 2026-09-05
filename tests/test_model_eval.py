import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import THRESHOLD_PATH

def test_model_threshold_validity():
    """Check that the chosen threshold is within bounds."""
    assert os.path.exists(THRESHOLD_PATH), "Threshold file not found."
    
    with open(THRESHOLD_PATH, 'r') as f:
        data = json.load(f)
        
    assert 'optimal_threshold' in data
    th = data['optimal_threshold']
    
    assert 0.0 < th < 1.0, f"Threshold {th} out of bounds"
