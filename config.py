import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
API_DIR = PROJECT_ROOT / "api"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TESTS_DIR = PROJECT_ROOT / "tests"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"

# File Paths
DATASET_PATH = DATA_DIR / "ecommerce_rto_dataset.csv"
SPLIT_INDICES_PATH = MODELS_DIR / "split_indices.json"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.json"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

# MLflow
import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
MLFLOW_TRACKING_URI = f"file:///{PROJECT_ROOT}/mlruns"
MLFLOW_EXPERIMENT_NAME = "ecommerce_rto_risk"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Data Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Financial Cost Weights (INR)
FP_COST = 4000  # Cost of blocking a legit customer (LTV loss)
FN_COST = 300   # Cost of shipping an RTO package

# Features
CATEGORICAL_FEATURES = [
    "client_canvas_hash",
    "app_set_id",
    "vpa_handle_hash",
    "delivery_pincode",
    "payment_method",
]

# API Defense Action Thresholds
# These will be dynamically updated by the model, but these are fallbacks/structures
DEFENSE_ACTIONS = {
    "ALLOW_FREE_COD": "ALLOW_FREE_COD",
    "REQUIRE_UPI_PREAUTH": "REQUIRE_UPI_PREAUTH",
    "BLOCK_TRANSACTION": "BLOCK_TRANSACTION"
}
