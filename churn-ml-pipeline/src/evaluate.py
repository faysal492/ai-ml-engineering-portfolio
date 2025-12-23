"""
Model Evaluation Module
Comprehensive model evaluation and diagnostics.
"""

import sys
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

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
    DATA_PROCESSED_DIR, MODELS_DIR, MODEL_FILE, 
    MIN_MODEL_ACCURACY, WANDB_API_KEY
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model_path: str = None, use_wandb: bool = True):
        self.model = None
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if model_path is None:
            model_path = MODEL_FILE
        
        self.model_path = Path(model_path)
        self.load_model()
    
    def load_model(self):
        """Load trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully")
    
    def load_test_data(self):
        """Load test dataset."""
        logger.info("Loading test data...")
        
        X_test = pd.read_csv(DATA_PROCESSED_DIR / "X_processed.csv")
        y_test = pd.read_csv(DATA_PROCESSED_DIR / "y_processed.csv").squeeze()
        
        # Use last 20% for testing (similar to train.py split)
        test_size = int(len(X_test) * 0.2)
        X_test = X_test.iloc[-test_size:]
        y_test = y_test.iloc[-test_size:]
        
        logger.info(f"Test data loaded: {X_test.shape}")
        
        return X_test, y_test
    
    def compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray) -> dict:
        """Compute evaluation metrics."""
        logger.info("Computing evaluation metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
        }
        
        return metrics
    
    def generate_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """Generate classification report."""
        report = classification_report(
            y_true, y_pred,
            target_names=['No Churn', 'Churn'],
            digits=4
        )
        return report
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None) -> str:
        """Plot and save confusion matrix."""
        if save_path is None:
            save_path = MODELS_DIR / "confusion_matrix.png"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return str(save_path)
    
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      save_path: str = None) -> str:
        """Plot and save ROC curve."""
        if save_path is None:
            save_path = MODELS_DIR / "roc_curve.png"
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
        return str(save_path)
    
    def evaluate(self) -> dict:
        """Run full evaluation."""
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = self.compute_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate report
        report = self.generate_report(y_test, y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        
        # Create plots
        cm_path = self.plot_confusion_matrix(metrics['confusion_matrix'])
        roc_path = self.plot_roc_curve(y_test, y_pred_proba)
        
        # Compile results
        results = {
            'metrics': metrics,
            'report': report,
            'cm_path': cm_path,
            'roc_path': roc_path,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
        
        return results
    
    def log_to_wandb(self, results: dict):
        """Log evaluation results to W&B."""
        if not self.use_wandb or not WANDB_API_KEY:
            logger.info("W&B logging disabled")
            return
        
        try:
            import os
            os.environ["WANDB_API_KEY"] = WANDB_API_KEY
            
            with wandb.init(project="churn-ml-pipeline", name="model-evaluation") as run:
                metrics = results['metrics']
                
                # Log metrics
                wandb.log({
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc'],
                })
                
                # Log confusion matrix image
                cm_image = wandb.Image(results['cm_path'])
                wandb.log({'confusion_matrix': cm_image})
                
                # Log ROC curve image
                roc_image = wandb.Image(results['roc_path'])
                wandb.log({'roc_curve': roc_image})
                
                # Log text report
                wandb.log({'classification_report': results['report']})
                
                logger.info("Results logged to W&B")
        
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    
    def print_summary(self, results: dict):
        """Print evaluation summary."""
        metrics = results['metrics']
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1 Score:      {metrics['f1_score']:.4f}")
        print(f"ROC AUC:       {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        print("="*60)


def main():
    """Main evaluation pipeline."""
    evaluator = ModelEvaluator(use_wandb=True)
    
    results = evaluator.evaluate()
    
    evaluator.log_to_wandb(results)
    
    evaluator.print_summary(results)
    
    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
