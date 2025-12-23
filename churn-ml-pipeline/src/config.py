"""
Configuration Module
Centralized configuration for the MLOps pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_FILE = DATA_RAW_DIR / "telco_churn.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "telco_churn_processed.csv"
FEATURES_FILE = DATA_PROCESSED_DIR / "features.csv"
TARGET_FILE = DATA_PROCESSED_DIR / "target.csv"

# Model configuration
MODEL_FILE = MODELS_DIR / "churn_model.joblib"
MODEL_NAME = "churn_prediction_model"

# Model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

# Validation thresholds
MIN_MODEL_ACCURACY = 0.75  # 75% minimum accuracy threshold

# Weights & Biases configuration
WANDB_PROJECT = "churn-ml-pipeline"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "personal")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# Target variable
TARGET_COLUMN = "Churn"

# Categorical columns to encode
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# Numerical columns to scale
NUMERICAL_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

# Feature names (for API)
FEATURE_NAMES = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
