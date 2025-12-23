"""
Data Preprocessing Module
Handles data cleaning, encoding, and scaling for the churn model.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Add src directory to path for absolute imports when running as script
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, DATA_PROCESSED_DIR,
    CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load raw data from CSV file."""
        if filepath is None:
            filepath = RAW_DATA_FILE
        
        logger.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # TotalCharges has empty strings instead of NaN
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
            logger.info("Converted TotalCharges empty strings to 0")
        
        # Check for other missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            df.fillna(df.mean(numeric_only=True), inplace=True)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        logger.info("Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataset")
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                else:
                    logger.warning(f"Encoder not found for {col}, skipping")
        
        return df_encoded
    
    def scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        numerical_cols = [col for col in NUMERICAL_COLUMNS if col in df.columns]
        
        if fit:
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_scaled
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns like customerID."""
        logger.info("Dropping unnecessary columns...")
        
        columns_to_drop = ['customerID']
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            logger.info(f"Dropped columns: {existing_cols}")
        
        return df
    
    def preprocess(self, df: pd.DataFrame = None, fit: bool = True) -> tuple:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame. If None, loads from RAW_DATA_FILE
            fit: If True, fits scalers and encoders. If False, uses existing ones.
            
        Returns:
            Tuple of (X, y) - features and target
        """
        if df is None:
            df = self.load_data()
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Drop unnecessary columns
        df = self.drop_unnecessary_columns(df)
        
        # Step 3: Separate target variable
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
        
        y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[TARGET_COLUMN])
        
        # Step 4: Encode categorical variables
        X = self.encode_categorical(X, fit=fit)
        
        # Step 5: Scale numerical features
        X = self.scale_numerical(X, fit=fit)
        
        self.feature_names = X.columns.tolist()
        logger.info(f"Feature names: {self.feature_names}")
        
        return X, y
    
    def save_preprocessing_artifacts(self):
        """Save scalers and encoders for production use."""
        logger.info("Saving preprocessing artifacts...")
        
        artifacts_dir = DATA_PROCESSED_DIR
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save encoders
        encoders_file = artifacts_dir / "label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_file)
        logger.info(f"Saved label encoders to {encoders_file}")
        
        # Save scaler
        scaler_file = artifacts_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_file)
        logger.info(f"Saved scaler to {scaler_file}")
        
        # Save feature names
        features_file = artifacts_dir / "feature_names.joblib"
        joblib.dump(self.feature_names, features_file)
        logger.info(f"Saved feature names to {features_file}")


def main():
    """Main preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    X, y = preprocessor.preprocess(fit=True)
    
    logger.info(f"Final dataset shape - Features: {X.shape}, Target: {y.shape}")
    logger.info(f"Churn distribution:\n{y.value_counts()}")
    
    # Save processed data
    X.to_csv(DATA_PROCESSED_DIR / "X_processed.csv", index=False)
    y.to_csv(DATA_PROCESSED_DIR / "y_processed.csv", index=False)
    logger.info(f"Saved processed data to {DATA_PROCESSED_DIR}")
    
    # Save preprocessing artifacts
    preprocessor.save_preprocessing_artifacts()
    
    print(f"\n✓ Preprocessing complete!")
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")


if __name__ == "__main__":
    main()
