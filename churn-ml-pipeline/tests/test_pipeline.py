"""
Unit tests for the churn ML pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.preprocess import DataPreprocessor
from src.ingest import check_url_reachable


class TestDataIngestion:
    """Test data ingestion module."""
    
    def test_url_reachability(self):
        """Test if dataset URL is reachable."""
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        assert check_url_reachable(url), "Dataset URL should be reachable"


class TestDataPreprocessing:
    """Test data preprocessing module."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataset for testing."""
        return pd.DataFrame({
            'customerID': ['001', '002', '003'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No'],
            'tenure': [1, 34, 2],
            'PhoneService': ['No', 'Yes', 'Yes'],
            'MultipleLines': ['No phone service', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service'],
            'OnlineBackup': ['No', 'Yes', 'No'],
            'DeviceProtection': ['Yes', 'No', 'No'],
            'TechSupport': ['No', 'Yes', 'No'],
            'StreamingTV': ['No', 'Yes', 'No'],
            'StreamingMovies': ['No', 'No', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card'],
            'MonthlyCharges': [29.85, 64.80, 20.05],
            'TotalCharges': [29.85, 2234.40, 40.10],
            'Churn': ['Yes', 'No', 'Yes']
        })
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.label_encoders == {}
        assert preprocessor.scaler is not None
    
    def test_handle_missing_values(self, sample_df):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        df = sample_df.copy()
        
        df_clean = preprocessor.handle_missing_values(df)
        
        # Check that TotalCharges was processed
        assert df_clean['TotalCharges'].dtype in [np.float64, np.float32]
        assert not df_clean.isnull().any().any(), "No missing values should remain"
    
    def test_drop_unnecessary_columns(self, sample_df):
        """Test dropping unnecessary columns."""
        preprocessor = DataPreprocessor()
        df = sample_df.copy()
        
        df_cleaned = preprocessor.drop_unnecessary_columns(df)
        
        assert 'customerID' not in df_cleaned.columns
        assert len(df_cleaned.columns) == len(df.columns) - 1


class TestModelValidation:
    """Test model validation logic."""
    
    def test_accuracy_threshold(self):
        """Test accuracy validation threshold."""
        from src.config import MIN_MODEL_ACCURACY
        
        assert MIN_MODEL_ACCURACY == 0.75, "Minimum accuracy should be 75%"


def test_feature_names_configured():
    """Test that feature names are configured."""
    from src.config import FEATURE_NAMES, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
    
    expected_features = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
    assert FEATURE_NAMES == expected_features


def test_config_paths_exist():
    """Test that configuration paths are properly set."""
    from src.config import PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR
    
    assert PROJECT_ROOT.exists(), "Project root should exist"
    assert PROJECT_ROOT.is_dir(), "Project root should be a directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
