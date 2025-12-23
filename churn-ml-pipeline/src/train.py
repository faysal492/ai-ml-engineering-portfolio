"""
Model Training Module
Trains XGBoost model and logs metrics to Weights & Biases.
"""

import sys
import logging
import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Add src directory to path for absolute imports when running as script
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError can occur with protobuf incompatibility on Python 3.14
    WANDB_AVAILABLE = False
    wandb = None

from config import (
    DATA_PROCESSED_DIR, MODELS_DIR, MODEL_FILE, MODEL_NAME,
    XGBOOST_PARAMS, WANDB_PROJECT, WANDB_API_KEY, MIN_MODEL_ACCURACY
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages the XGBoost churn prediction model."""
    
    def __init__(self, use_wandb: bool = True):
        self.model = None
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.run = None
        
        if self.use_wandb and WANDB_API_KEY:
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            logger.info("Weights & Biases API key configured")
    
    def initialize_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        if not self.use_wandb:
            logger.info("W&B integration disabled")
            return
        
        try:
            self.run = wandb.init(
                project=WANDB_PROJECT,
                name=f"{MODEL_NAME}-run",
                config=XGBOOST_PARAMS,
                notes="Churn prediction model training"
            )
            logger.info(f"W&B run initialized: {self.run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def load_data(self) -> tuple:
        """Load preprocessed data."""
        logger.info("Loading preprocessed data...")
        
        X = pd.read_csv(DATA_PROCESSED_DIR / "X_processed.csv")
        y = pd.read_csv(DATA_PROCESSED_DIR / "y_processed.csv").squeeze()
        
        logger.info(f"Data loaded - Shape: {X.shape}")
        
        return X, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """Split data into training and testing sets."""
        logger.info(f"Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        self.model = XGBClassifier(
            **XGBOOST_PARAMS,
            verbosity=1,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            verbose=False
        )
        
        logger.info("Model training completed")
        
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model on test set."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix, classification_report
        )
        
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Evaluation Results:\n{classification_report(y_test, y_pred)}")
        
        return metrics
    
    def log_metrics_to_wandb(self, metrics: dict):
        """Log metrics to Weights & Biases."""
        if not self.use_wandb or not self.run:
            return
        
        logger.info("Logging metrics to W&B...")
        
        wandb_metrics = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
        }
        
        # Log confusion matrix (with try-catch for API changes)
        try:
            cm = np.array(metrics['confusion_matrix'])
            wandb_metrics['confusion_matrix'] = wandb.plot.confusion_matrix(
                y_true=None,
                preds=None,
                class_names=['No Churn', 'Churn'],
                matrix=cm
            )
        except TypeError:
            # Fallback for newer W&B versions with different API
            logger.info("Skipping confusion matrix logging (API mismatch)")
        
        self.run.log(wandb_metrics)
        logger.info("Metrics logged to W&B")
    
    def save_model(self, model_path: str = None):
        """Save trained model to disk."""
        if model_path is None:
            model_path = MODEL_FILE
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        if self.use_wandb and self.run:
            self.run.log_model(str(model_path), name=MODEL_NAME)
            logger.info(f"Model logged to W&B as {MODEL_NAME}")
    
    def validate_model_quality(self, metrics: dict) -> bool:
        """
        Validate that model meets minimum quality threshold.
        
        Returns:
            True if model is good enough, False otherwise
        """
        accuracy = metrics['accuracy']
        
        if accuracy < MIN_MODEL_ACCURACY:
            logger.warning(
                f"Model accuracy ({accuracy:.2%}) is below minimum threshold ({MIN_MODEL_ACCURACY:.2%})"
            )
            return False
        
        logger.info(f"Model passed quality check (accuracy: {accuracy:.2%})")
        return True
    
    def finish_wandb_run(self):
        """Finish W&B run."""
        if self.use_wandb and self.run:
            self.run.finish()
            logger.info("W&B run finished")


def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = ModelTrainer(use_wandb=True)
    
    # Initialize W&B
    trainer.initialize_wandb()
    
    # Load data
    X, y = trainer.load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.train_test_split(X, y)
    
    # Train model
    trainer.train(X_train, y_train)
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Log metrics to W&B
    trainer.log_metrics_to_wandb(metrics)
    
    # Validate model quality
    if trainer.validate_model_quality(metrics):
        trainer.save_model()
        print(f"\n✓ Model training and validation complete!")
        print(f"Metrics: {metrics}")
    else:
        print(f"\n✗ Model did not pass quality validation")
        if trainer.use_wandb:
            trainer.finish_wandb_run()
        exit(1)
    
    # Finish W&B run
    if trainer.use_wandb:
        trainer.finish_wandb_run()


if __name__ == "__main__":
    main()
