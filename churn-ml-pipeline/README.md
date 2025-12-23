# 🚀 Churn ML Pipeline - Production-Grade MLOps System

A complete, production-ready Machine Learning Operations (MLOps) pipeline for predicting customer churn. This project demonstrates clean code practices, automation, and enterprise-grade ML deployment patterns.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Configuration](#configuration)

---

## ✨ Features

✅ **Automated Data Ingestion** - Fetches Telco Churn dataset from GitHub  
✅ **Robust Data Preprocessing** - Handles missing values, encoding, and scaling  
✅ **Model Training** - XGBoost with configurable hyperparameters  
✅ **Experiment Tracking** - Weights & Biases integration for metrics and artifacts  
✅ **REST API** - FastAPI inference endpoint for real-time predictions  
✅ **CI/CD Pipeline** - GitHub Actions for automated weekly retraining  
✅ **Quality Validation** - Automatic accuracy threshold checks  
✅ **Model Versioning** - Artifacts tracked and versioned  

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **ML Model** | XGBoost, Scikit-learn |
| **Experiment Tracking** | Weights & Biases (W&B) |
| **API** | FastAPI, Uvicorn |
| **Automation** | GitHub Actions |
| **Deployment** | Render (Free tier) |
| **Testing** | Pytest |
| **Environment** | Python 3.11+, venv |

---

## 📁 Project Structure

```
churn-ml-pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration
│   ├── ingest.py              # Data fetching & ingestion
│   ├── preprocess.py          # Data cleaning & preprocessing
│   ├── train.py               # Model training with W&B logging
│   ├── evaluate.py            # Model evaluation & metrics
│   └── api.py                 # FastAPI inference server
├── data/
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Cleaned & preprocessed data
├── models/                    # Trained model artifacts
├── tests/                     # Unit tests
├── .github/
│   └── workflows/
│       └── train_deploy.yml   # CI/CD pipeline
├── pipeline.py                # Main orchestration script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- pip or conda
- Git
- (Optional) Weights & Biases account

### Step 1: Clone Repository

```bash
git clone https://github.com/faysal492/ai-ml-engineering-portfolio.git
cd churn-ml-pipeline
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)

```bash
cp .env.example .env
# Edit .env with your Weights & Biases API key
```

---

## ⚡ Quick Start

### Run Full Pipeline

Execute all steps (ingest → preprocess → train → evaluate):

```bash
python pipeline.py
```

### Run Individual Steps

```bash
# 1. Fetch data
python src/ingest.py

# 2. Preprocess data
python src/preprocess.py

# 3. Train model
python src/train.py

# 4. Evaluate model
python src/evaluate.py

# 5. Start API server
python -m uvicorn src.api:app --reload --port 8000
```

### Make Predictions

**Single Prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0
  }'
```

**Batch Predictions:**

```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"gender": "Male", "SeniorCitizen": 0, ...},
    {"gender": "Female", "SeniorCitizen": 1, ...}
  ]'
```

---

## 🔧 Pipeline Components

### 1. **Data Ingestion** (`src/ingest.py`)

Fetches Telco Customer Churn dataset from GitHub:

- ✓ URL reachability check
- ✓ Automated download
- ✓ Local storage in `data/raw/`
- ✓ Shape and column validation

**Key Features:**
- Error handling for network issues
- Logging at each step
- 7,043 records × 21 features

### 2. **Data Preprocessing** (`src/preprocess.py`)

Transforms raw data into model-ready format:

- ✓ Handle missing values (TotalCharges empty strings → 0)
- ✓ Categorical encoding (LabelEncoder)
- ✓ Numerical scaling (StandardScaler)
- ✓ Feature engineering
- ✓ Drop unnecessary columns (customerID)

**Output:**
- `X_processed.csv` - Feature matrix (7,043 × 19)
- `y_processed.csv` - Target variable
- `label_encoders.joblib` - Encoding artifacts
- `scaler.joblib` - Scaling artifacts

### 3. **Model Training** (`src/train.py`)

Trains XGBoost classifier with W&B integration:

- ✓ Configurable hyperparameters in `config.py`
- ✓ 80/20 train-test split (stratified)
- ✓ Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- ✓ Model versioning via W&B
- ✓ Quality validation (75% accuracy threshold)

**Default Hyperparameters:**
```python
n_estimators: 100
max_depth: 7
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
```

### 4. **Model Evaluation** (`src/evaluate.py`)

Comprehensive evaluation and visualization:

- ✓ Classification metrics (accuracy, precision, recall, F1)
- ✓ Confusion matrix visualization
- ✓ ROC curve plotting
- ✓ W&B logging with charts
- ✓ Performance diagnostics

### 5. **FastAPI Server** (`src/api.py`)

Production-grade REST API:

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch predictions |

**Response Format:**
```json
{
  "churn_probability": 0.78,
  "churn_prediction": "Yes",
  "confidence": 0.85
}
```

---

## 📊 API Documentation

### Interactive Swagger UI

Once API is running, visit: `http://localhost:8000/docs`

### Request/Response Models

**CustomerData (Input):**
```python
- gender: "Male" | "Female"
- SeniorCitizen: 0 | 1
- Partner: "Yes" | "No"
- Dependents: "Yes" | "No"
- tenure: float (months)
- PhoneService: "Yes" | "No"
- MultipleLines: "Yes" | "No" | "No phone service"
- InternetService: "DSL" | "Fiber optic" | "No"
- OnlineSecurity: "Yes" | "No" | "No internet service"
- ... (15 more fields)
```

---

## 🚀 Deployment

### Deploy to Render (Free Tier)

1. **Push to GitHub:**
```bash
git add .
git commit -m "feat: Add production MLOps pipeline"
git push origin main
```

2. **Create Render Service:**
   - Connect GitHub repository to Render
   - Select this project
   - Set Environment: Python 3.11
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`

3. **Set Environment Variables:**
   - Add `WANDB_API_KEY` in Render dashboard

4. **Deploy:** Click "Deploy"

### GitHub Actions CI/CD

Pipeline triggers on:
- ✓ Push to `main` branch (src or requirements changes)
- ✓ Weekly schedule (Sunday 2 AM UTC)

**Actions:**
1. Install dependencies
2. Fetch latest data
3. Preprocess
4. Train model
5. Evaluate
6. Upload artifacts
7. Notify on failure

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```bash
# Weights & Biases
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_username

# Model Configuration
MIN_MODEL_ACCURACY=0.75
```

### Hyperparameters

Edit `src/config.py`:

```python
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
```

---

## 📈 Monitoring & Tracking

### Weights & Biases Dashboard

- Real-time metrics
- Model artifacts versioning
- Confusion matrix & ROC curves
- Experiment comparison
- Automated alerts

**Features:**
- Track accuracy, precision, recall, F1, ROC-AUC
- Compare model versions
- Visualize training dynamics

---

## 🧪 Testing

Run tests:

```bash
pytest tests/ -v --cov=src
```

---

## 📝 Data Pipeline Flow

```
Raw Data (CSV)
     ↓
[Ingest] → Fetch from GitHub
     ↓
Raw data stored
     ↓
[Preprocess] → Clean, encode, scale
     ↓
Processed features & target
     ↓
[Train] → XGBoost classifier
     ↓
Model artifacts + metrics
     ↓
[Evaluate] → Validation & charts
     ↓
Performance dashboard (W&B)
     ↓
[API] → FastAPI server
     ↓
Real-time predictions
```

---

## 🔍 Key Metrics

**Model Performance:**
- Accuracy: ~80-85%
- Precision: ~65-70%
- Recall: ~50-60%
- F1-Score: ~57-65%
- ROC-AUC: ~85-90%

**Dataset:**
- Total Records: 7,043
- Features: 19 (after preprocessing)
- Target: Churn (binary)
- Class Distribution: ~73% No Churn, ~27% Churn

---

## 🎯 Next Steps

1. **Production Deployment:**
   - Deploy API to Render
   - Set up monitoring & alerts

2. **Model Improvements:**
   - Experiment with hyperparameters
   - Try ensemble methods
   - Feature engineering enhancements

3. **Advanced Features:**
   - Model explainability (SHAP)
   - Drift detection
   - A/B testing framework
   - Real-time feature store

4. **Scaling:**
   - Docker containerization
   - Kubernetes orchestration
   - Distributed training

---

## 📞 Support & Documentation

- **W&B Docs:** https://docs.wandb.ai
- **FastAPI:** https://fastapi.tiangolo.com
- **XGBoost:** https://xgboost.readthedocs.io
- **Sklearn:** https://scikit-learn.org

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙌 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ for production-grade ML systems**
