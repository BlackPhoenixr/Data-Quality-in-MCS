"""
This module provides the following utilities:
    - Directory and Path Management: Managing paths and directories for the project.
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Make Tiny Dataset: Function to create a small synthetic dataset for testing.
    - Smoke Test: Function to run a smoke test for TPOT and FLAML model training.
    - Script Entry Point: The conditional block that executes the main function.
"""

# ======= Directory and Path Management =======
import sys, pathlib  # For path manipulations and system path management

ROOT_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
)  # Get the root directory of the project
if (
    str(ROOT_DIR) not in sys.path
):  # Check if the root directory is already in the system path
    sys.path.append(
        str(ROOT_DIR)
    )  # Add the root directory to the system path if not present

# =========== Imports and Dependencies ===========
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import time  # For timing the smoke‑test runtime
import signal  # For catching SIGINT (Ctrl+C) and SIGTERM for graceful shutdown
from colorama import Fore  # Provides colored terminal text output

from modules.system_guard import (
    create_dask_client,
    logger,
    timer_decorator,
)  # Use centralized Dask client, logger, timer decorator, and cleanup
import modules.system_guard as system  # For system functions and global variables
import modules.tpot_functions as tpot_f  # For TPOT model training and prediction
import modules.flaml_functions as flaml_f  # For FLAML model training and prediction


# ======= Make Tiny Dataset =======
def _make_tiny_dataset(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """
    Function which creates a small synthetic dataset for testing purposes.
    """
    rng = np.random.default_rng(seed=42)  # Random number generator for reproducibility
    X = pd.DataFrame(
        {
            "Numeric1": rng.standard_normal(n),
            "Numeric2": rng.random(n) * 10,
            "Category": rng.choice(["A", "B", "C"], size=n),
        }
    )  # Create a DataFrame with two numeric features and one categorical feature
    y = (X["Numeric1"] + X["Numeric2"] > 5).astype(
        int
    )  # Create a binary target variable
    return X, y  # Return the features and target variable


# ======= Smoke Test =======
@timer_decorator
def test_smoke() -> None:
    """
    Function which runs a smoke test for TPOT and FLAML model training.
    """
    signal.signal(
        signal.SIGTERM, system.signal_handler
    )  # Catch SIGTERM for graceful shutdown
    signal.signal(
        signal.SIGINT, system.signal_handler
    )  # Catch SIGINT (Ctrl+C) for graceful shutdown

    script_start = time.time()  # Start the script execution timer
    # logger.debug("\n--- Starting script execution timer ---")

    logger.info(f"\nEscape the Matrix!\nEnter the Ducktrix!")  # Print a cool message

    try:  # Try-except block for catching exceptions and handling them gracefully
        try:  # Try to create a Dask client for parallel processing
            client = (
                create_dask_client()
            )  # Create a Dask client for parallel processing
            logger.info(
                f"Dask client initialized. Dashboard: {client.dashboard_link}"
            )  # Log the Dask dashboard link

        except Exception as e:  # Catch exceptions during Dask client creation
            client = None  # Set client to None if Dask client creation fails
            logger.warning(f"Dask client could not be started: {e}")  # Log the warning

        X, y = (
            _make_tiny_dataset()
        )  # Create a DataFrame with two numeric features and one categorical feature
        X = pd.get_dummies(
            X, columns=["Category"], drop_first=True
        )  # One-hot encode the categorical feature

        preds_tpot, _ = tpot_f.tpot_model_runner(
            X,
            y,
            file_name="smoke",
            generations=1,
            population_size=2,
        )  # Train the TPOT model and make predictions
        assert len(preds_tpot) == len(
            y
        ), "TPOT prediction length mismatch"  # Check if the length of predictions matches the target variable

        preds_flaml, _ = flaml_f.flaml_model_runner(
            X, y, file_name="smoke", budget=2
        )  # Train the FLAML model and make predictions
        assert len(preds_flaml) == len(
            y
        ), "FLAML prediction length mismatch"  # Check if the length of predictions matches the target variable

        logger.info("✅  Smoke test passed: TPOT & FLAML ran successfully.")

    finally:  # Ensure that the Dask client and cluster are closed
        try:
            if client:  # Check if the Dask client was created successfully
                client.close()  # Close the Dask client to free up resources

                system.cleanup_resources()  # Cleanup temporary resources

                script_end = time.time()  # End the script execution timer
                total_duration = (
                    script_end - script_start
                )  # Calculate the total script execution time
                logger.info(
                    f"\nPipeline completed in: {Fore.LIGHTYELLOW_EX}{total_duration:.2f}s"
                )  # Log the total script execution time

        except Exception as e:  # Catch exceptions during cleanup
            logger.error(f"\nCleanup failed: {e}")  # Log the error
            raise  # Raise the exception


# ======= Script Entry Point =======
if __name__ == "__main__":  # If the script is run directly
    test_smoke()  # Run the smoke test
