#!/usr/bin/env bash

"""
Main MLOps Pipeline Orchestration Script
Runs the complete pipeline: ingest -> preprocess -> train -> evaluate
"""

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_CMD="${PYTHON_CMD:-.venv/bin/python}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}🚀 Churn ML Pipeline Orchestration${NC}"
echo -e "${YELLOW}========================================${NC}"

# Step 1: Data Ingestion
echo -e "\n${YELLOW}Step 1: Data Ingestion${NC}"
echo "Fetching Telco Churn dataset from GitHub..."
$PYTHON_CMD src/ingest.py || { echo -e "${RED}✗ Data ingestion failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Data ingestion complete${NC}"

# Step 2: Data Preprocessing
echo -e "\n${YELLOW}Step 2: Data Preprocessing${NC}"
echo "Cleaning and transforming data..."
$PYTHON_CMD src/preprocess.py || { echo -e "${RED}✗ Preprocessing failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Data preprocessing complete${NC}"

# Step 3: Model Training
echo -e "\n${YELLOW}Step 3: Model Training${NC}"
echo "Training XGBoost model and logging to W&B..."
$PYTHON_CMD src/train.py || { echo -e "${RED}✗ Training failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Model training complete${NC}"

# Step 4: Model Evaluation
echo -e "\n${YELLOW}Step 4: Model Evaluation${NC}"
echo "Evaluating model performance..."
$PYTHON_CMD src/evaluate.py || { echo -e "${RED}✗ Evaluation failed${NC}"; exit 1; }
echo -e "${GREEN}✓ Model evaluation complete${NC}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Pipeline execution successful!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n📊 Next steps:"
echo "1. Check W&B dashboard for metrics and artifacts"
echo "2. Review model/churn_model.joblib for the trained model"
echo "3. Deploy API: $PYTHON_CMD -m uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo "4. Test predictions with: curl http://localhost:8000/docs"
