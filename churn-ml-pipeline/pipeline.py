#!/usr/bin/env python
"""
Main Pipeline Orchestration Script
Runs the complete MLOps pipeline: ingest → preprocess → train → evaluate
"""

import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
SCRIPTS = {
    'ingest': 'src/ingest.py',
    'preprocess': 'src/preprocess.py',
    'train': 'src/train.py',
    'evaluate': 'src/evaluate.py'
}


def run_script(script_name: str, script_path: str, env_vars: dict = None) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_name: Name of the script (for logging)
        script_path: Relative path to the script
        env_vars: Optional environment variables
        
    Returns:
        True if successful, False otherwise
    """
    full_path = PROJECT_ROOT / script_path
    
    if not full_path.exists():
        logger.error(f"Script not found: {full_path}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting: {script_name.upper()}")
    logger.info(f"{'='*60}")
    
    try:
        import os
        env = os.environ.copy()
        # Add src directory to PYTHONPATH for script imports
        src_path = str(PROJECT_ROOT / "src")
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{src_path}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = src_path
        if env_vars:
            env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True
        )
        
        logger.info(f"✓ {script_name.upper()} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name.upper()} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {script_name.upper()} encountered an error: {e}")
        return False


def main():
    """Execute the full MLOps pipeline."""
    logger.info(f"\n{'#'*60}")
    logger.info("# CHURN ML PIPELINE - FULL EXECUTION")
    logger.info(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}\n")
    
    # Prepare environment variables
    import os
    env_vars = {}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    
    # Run pipeline steps
    pipeline_steps = ['ingest', 'preprocess', 'train', 'evaluate']
    results = {}
    
    for step in pipeline_steps:
        success = run_script(step, SCRIPTS[step], env_vars)
        results[step] = success
        
        if not success:
            logger.error(f"\nPipeline stopped at step: {step}")
            break
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    
    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{step.ljust(15)}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"{'='*60}")
    
    if all_passed:
        logger.info("✓ Pipeline execution completed successfully!")
        logger.info(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    else:
        logger.error("✗ Pipeline execution failed!")
        logger.error(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
