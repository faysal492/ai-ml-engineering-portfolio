"""
FastAPI Inference API
Serves predictions from the trained churn model.
"""

import sys
import logging
import joblib
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add src directory to path for absolute imports when running as script
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import (
    MODEL_FILE, DATA_PROCESSED_DIR, CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS, FEATURE_NAMES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="ML model API for predicting customer churn",
    version="1.0.0"
)

# Global model and preprocessing artifacts
model = None
label_encoders = None
scaler = None
feature_names = None


# Request/Response Models
class CustomerData(BaseModel):
    """Schema for single customer prediction request."""
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: float = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="Yes")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="One year")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=65.5)
    TotalCharges: float = Field(..., example=786.0)


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    churn_prediction: str = Field(..., description="Binary prediction: 'Yes' or 'No'")
    confidence: float = Field(..., description="Confidence in the prediction")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse]
    total_requests: int


# Startup and Shutdown Events
@app.on_event("startup")
async def load_model_on_startup():
    """Load model and preprocessing artifacts on startup."""
    global model, label_encoders, scaler, feature_names
    
    logger.info("Loading model and preprocessing artifacts...")
    
    try:
        # Load model
        if not Path(MODEL_FILE).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        
        model = joblib.load(MODEL_FILE)
        logger.info(f"Model loaded from {MODEL_FILE}")
        
        # Load preprocessing artifacts
        encoders_path = DATA_PROCESSED_DIR / "label_encoders.joblib"
        scaler_path = DATA_PROCESSED_DIR / "scaler.joblib"
        features_path = DATA_PROCESSED_DIR / "feature_names.joblib"
        
        if encoders_path.exists():
            label_encoders = joblib.load(encoders_path)
            logger.info(f"Label encoders loaded from {encoders_path}")
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        if features_path.exists():
            feature_names = joblib.load(features_path)
            logger.info(f"Feature names loaded from {features_path}")
        
        logger.info("Model and artifacts loaded successfully")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server")


# Health and Info Endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/info", tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "churn_prediction_model",
        "model_type": "XGBClassifier",
        "features": feature_names,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS
    }


# Prediction Endpoints
def preprocess_customer_data(data: CustomerData) -> np.ndarray:
    """
    Preprocess customer data for model prediction.
    
    Args:
        data: CustomerData input
        
    Returns:
        Preprocessed feature array ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Encode categorical variables
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and label_encoders and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except Exception as e:
                logger.warning(f"Failed to encode {col}: {e}")
    
    # Scale numerical features
    numerical_cols = [col for col in NUMERICAL_COLUMNS if col in df.columns]
    if numerical_cols and scaler:
        try:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        except Exception as e:
            logger.warning(f"Failed to scale numerical features: {e}")
    
    # Ensure column order matches training data
    if feature_names:
        df = df[[col for col in feature_names if col in df.columns]]
    
    return df.values


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerData):
    """
    Predict whether a customer will churn.
    
    Args:
        customer: Customer data
        
    Returns:
        Prediction with probability and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        X = preprocess_customer_data(customer)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        # Calculate confidence (distance from 0.5 boundary)
        confidence = max(abs(probability - 0.5) * 2, 0.51)
        
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction="Yes" if prediction == 1 else "No",
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(customers: List[CustomerData]):
    """
    Make batch predictions for multiple customers.
    
    Args:
        customers: List of customer data
        
    Returns:
        Batch prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Empty customer list")
    
    try:
        predictions = []
        
        for customer in customers:
            # Preprocess data
            X = preprocess_customer_data(customer)
            
            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]
            confidence = max(abs(probability - 0.5) * 2, 0.51)
            
            predictions.append(
                PredictionResponse(
                    churn_probability=float(probability),
                    churn_prediction="Yes" if prediction == 1 else "No",
                    confidence=float(confidence)
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_requests=len(customers)
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )
