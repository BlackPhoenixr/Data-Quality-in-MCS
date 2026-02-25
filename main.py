"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Main Function: The main entry point for the data processing pipeline.
    - Script Entry Point: The conditional block that executes the main function.
"""

# =========== Imports and Dependencies ===========
import shutup  # Used to suppress warnings that may clutter console logs
import signal  # For catching SIGINT (Ctrl+C) and SIGTERM for graceful shutdown
import time  # Used to measure overall script execution time
from colorama import Fore  # Provides colored terminal text output

import modules.config as config  # Configuration module
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import (
    logger,
    create_dask_client,
)  # Logger for logging messages and errors, and function to create a Dask client for parallel processing
import modules.utils as utils  # File utilities for loading/exporting CSV files
import modules.original_reputation_exporter as exporter  # Exports original reputations
import modules.final_reputation_predictor as reputation  # Predicts final reputations using TPOT
import modules.outlier_processor as outliers  # Processes outliers in data chunks
import modules.data_processor as processor  # Maps column types, validates target column, etc.
import modules.misbehaving_data_splitter as splitter  # Splits misbehaving data into intentional/unintentional
import modules.feedback_loop as feedback  # Orchestrates feedback loop to unify data & update reputations
import modules.visualizer as vis  # Contains plotting functions for final insights

shutup.please()  # Suppresses warnings from libraries like NumPy, Pandas, etc.


# =========== Main ===========
def main() -> None:
    """
    Function which acts as the main entry point for the data processing pipeline.

    Workflow Steps
    ---------------------------------
    1) Clean up existing output directories for a fresh run.
    2) Register signal handlers for graceful shutdown.
    3) Start an execution timer for performance measurement.
    4) Create a Dask client for parallel/distributed computing.
    5) Load the primary dataset from a CSV file.
    6) Prepare data by mapping columns, classifying features, and validating the target column.
    7) Process outliers in chunks, separating normal and misbehaving data.
    8) Further split misbehaving data into intentional and unintentional subsets.
    9) Execute a feedback loop to unify data and update user reputations.
    10) Optionally adjust reputations using ML predictions or post-processing rules.
    11) Generate visualizations to summarize key insights.
    12) Perform a final cleanup of systems and log total execution time.

    Raises Exception
    ---------------------------------
        Any exception raised during the execution of the pipeline steps.
    """

    logger.info(f"\nScript has started")  # Log the start of the script

    # system.clean_project_outputs("outputs")  # Clean up old files from previous runs

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

        target_column_name = (
            config.TARGET_COLUMN_NAME
        )  # The target column for reputation predictions

        original = utils.load_csv(
            folder_name=config.DATA_FOLDER,
            file_name=config.DATASET_FILE_NAME,
            sep=",",
        )  # Load the dataset from the CSV file

        exporter.export_original_reputations(
            original
        )  # Export original reputations to CSV

        column_types = processor.column_mapping(
            original, target_column_name
        )  # Map column types

        reputation.predict_final_reputations_on_raw_data(
            original
        )  # Predict final reputations using TPOT

        normal_unsup, misbehaving_unsup = outliers.process_in_chunks(
            original,
            column_types,
        )  # Process outliers in chunks, separating normal and misbehaving data

        intentional_unsup, unintentional_unsup = splitter.split_misbehaving_data(
            misbehaving_unsup
        )  # Split misbehaving data into intentional and unintentional subsets

        feedback.feedback_task(
            normal_unsup,
            intentional_unsup,
            unintentional_unsup,
            run_type="unsupervised",
        )  # Orchestrate feedback loop for unsupervised data
        reputation.predict_final_reputations_on_processed_data(
            run_type="unsupervised"
        )  # Predict final reputations on processed data

        ground_truth = utils.load_csv(
            folder_name=config.GROUND_TRUTH_FOLDER,
            file_name="full_" + str(config.GROUND_TRUTH_FILE_NAME),
            sep=",",
        )  # Load ground truth labels from a CSV file

        normal_sup = ground_truth[
            ground_truth["True_Class"] == "Normal"
        ]  # Filter normal data for supervised learning
        intentional_sup = ground_truth[
            ground_truth["True_Class"] == "Intentional"
        ]  # Filter intentional data for supervised learning
        unintentional_sup = ground_truth[
            ground_truth["True_Class"] == "Unintentional"
        ]  # Filter unintentional data for supervised learning

        feedback.feedback_task(
            normal_sup, intentional_sup, unintentional_sup, run_type="supervised"
        )  # Orchestrate feedback loop for supervised data
        reputation.predict_final_reputations_on_processed_data(
            run_type="supervised"
        )  # Predict final reputations on processed data

        vis.plot_all_visualizations()  # Plot all visualizations

    except Exception as e:  # Catch exceptions and handle them gracefully
        logger.error(f"\nMain Execution: Process failed | {str(e)}")  # Log the error
        exit(1)  # Exit the script with an error code

    finally:  # Finally block for cleanup and resource deallocation
        try:  # Try-except block for catching exceptions during cleanup
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


# =========== Script Entry Point ===========
if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Execute the main function
