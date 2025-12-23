"""
Data Ingestion Module
Fetches the Telco Customer Churn dataset from GitHub and saves it locally.
"""

import os
import sys
import logging
import requests
import pandas as pd
from pathlib import Path

# Add src directory to path for absolute imports when running as script
_src_path = Path(__file__).parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Dataset URL
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def check_url_reachable(url: str, timeout: int = 5) -> bool:
    """
    Check if the dataset URL is reachable.
    
    Args:
        url: The URL to check
        timeout: Timeout in seconds
        
    Returns:
        True if reachable, False otherwise
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error(f"URL check failed: {e}")
        return False


def ingest_data(output_path: str = None) -> pd.DataFrame:
    """
    Fetch the Telco Churn dataset from GitHub and save it locally.
    
    Args:
        output_path: Path to save the raw CSV file. If None, uses default data/raw/ path.
        
    Returns:
        DataFrame containing the fetched data
        
    Raises:
        SystemExit: If URL is unreachable or data fetch fails
    """
    # Set default output path
    if output_path is None:
        output_dir = Path(__file__).parent.parent / "data" / "raw"
    else:
        output_dir = Path(output_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "telco_churn.csv"
    
    logger.info(f"Checking if URL is reachable: {DATASET_URL}")
    if not check_url_reachable(DATASET_URL):
        logger.error(f"Cannot reach URL: {DATASET_URL}")
        sys.exit(1)
    
    logger.info(f"Fetching dataset from: {DATASET_URL}")
    try:
        df = pd.read_csv(DATASET_URL)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Save locally
        df.to_csv(output_file, index=False)
        logger.info(f"Dataset saved to: {output_file}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to fetch dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    df = ingest_data()
    print(f"\n✓ Data ingestion complete! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
