"""
This module provides the following utilities:
    - Directory and Path Management: Manages directories and paths for importing modules and creating output directories.
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Is Intentional: Checks if a row in the dataset contains intentional misbehavior based on predefined fields and values.
    - Is Unintentional: Checks if a row in the dataset contains unintentional misbehavior based on specific conditions.
    - Assign True Class: Assigns a true class label to each row based on intentional and unintentional misbehavior.
    - Main: The main function that orchestrates the ground truth labeling dataset creation process.
    - Script Entry Point: The entry point of the script that executes the main function.
"""

# ======= Directory and Path Management =======
import sys, pathlib  # For path manipulations and importing modules.

ROOT_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
)  # Get the root directory of the project
if str(ROOT_DIR) not in sys.path:  # Check if the root directory is already in sys.path
    sys.path.append(
        str(ROOT_DIR)
    )  # Add the root directory to sys.path if not already present

# ======= Imports and Dependencies =======
from pathlib import Path  # For file path manipulations.
import time  # Used for tracking execution time and implementing delays.
import signal  # Enables handling of system signals (e.g., SIGTERM, SIGINT) for graceful shutdown.
from colorama import (
    Fore,
)  # Provides ANSI escape sequences for colorizing terminal output.

import modules.config as config  # Import configuration settings.
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import (
    logger,
)  # Import logger for logging messages and errors.
import modules.utils as utils  # Import utility functions for file handling.


intentional_fields = [
    "User_Activity",
    "University_Area",
    "Road_Accident",
    "Road_Type",
    "Speed_Limit",
    "Violation_Type",
]  # Fields that indicate intentional misbehavior

intentional_values = {
    "User_Activity": {"Flying", "Swimming"},
    "University_Area": {"Center", "Spinari"},
    "Road_Accident": {6, 7, 8, 9},
    "Road_Type": {"5way", "6way"},
    "Speed_Limit": {140, 150, 160},
    "Violation_Type": {"Lane_Discards", "Harassing_Driver", "Illegal_Parking"},
}  # Values that indicate intentional misbehavior for each field


# ======= Is Intentional =======
def is_intentional(row):
    """
    Function which checks if a row contains intentional misbehavior based on predefined fields and values.
    Logic is a direct mirror of dataset_creation.py to ensure synchronization.
    """
    for field in intentional_fields:  # Iterate through each intentional field
        if (
            row[field] in intentional_values[field]
        ):  # Check if the value in the field matches any of the intentional values
            return True  # If any field matches, return True indicating intentional misbehavior
    return False  # If no fields match, return False indicating normal behavior


# ======= Is Unintentional =======
def is_unintentional(row):
    """
    Function which checks if a row contains unintentional misbehavior based on specific conditions.
    Logic is a direct mirror of dataset_creation.py to ensure synchronization.
    """
    if (
        row["Battery_Percentage"] > 100
    ):  # Check if battery percentage is greater than 100
        return True  # Return True indicating unintentional misbehavior
    if isinstance(
        row["Internet_Signal_Strength"], (int, float)
    ):  # Check if Internet_Signal_Strength is a number
        if (
            row["Internet_Signal_Strength"] < -120
            or row["Internet_Signal_Strength"] > -10
        ):  # Check if Internet_Signal_Strength is outside the valid range
            return True  # Return True indicating unintentional misbehavior
    if (
        row["Accelerometer_Range"] < -16 or row["Accelerometer_Range"] > 16
    ):  # Check if Accelerometer_Range is outside the valid range
        return True  # Return True indicating unintentional misbehavior
    if (
        isinstance(row["Accelerometer_Resolution"], float)
        and row["Accelerometer_Resolution"] > 0.02
    ):  # Check if Accelerometer_Resolution is a float and greater than 0.02
        return True  # Return True indicating unintentional misbehavior
    if (
        row["Latitude"] < 37 or row["Latitude"] > 43
    ):  # Check if Latitude is outside the valid range
        return True  # Return True indicating unintentional misbehavior
    if (
        row["Longitude"] < 17 or row["Longitude"] > 24
    ):  # Check if Longitude is outside the valid range
        return True  # Return True indicating unintentional misbehavior
    return False  # If none of the conditions match, return False indicating normal behavior


# ======= Assign True Class =======
def assign_true_class(row):
    """
    Function which assigns a true class label to each row.
    If a row is both intentional and unintentional, it is labeled as 'Intentional'.
    """
    intentional = is_intentional(
        row
    )  # Check if the row contains intentional misbehavior
    unintentional = is_unintentional(
        row
    )  # Check if the row contains unintentional misbehavior
    if intentional:  # If the row is intentional
        return "Intentional"  # Return 'Intentional' as the true class
    elif unintentional:  # If the row is unintentional
        return "Unintentional"  # Return 'Unintentional' as the true class
    else:  # If the row is neither intentional nor unintentional
        return "Normal"  # Return 'Normal' as the true class


# ======= Main =======
def main() -> None:
    """"""
    try:
        signal.signal(
            signal.SIGTERM, system.signal_handler
        )  # Catch SIGTERM for graceful shutdown
        signal.signal(
            signal.SIGINT, system.signal_handler
        )  # Catch SIGINT for graceful shutdown

        script_start = time.time()  # Start the script execution timer
        logger.debug(
            "\n--- Starting ground truth labeling dataset creation timer ---"
        )  # Log the start of the script

        OUT_DIR = Path(
            config.GROUND_TRUTH_FOLDER
        )  # Define the output directory for ground truth labels
        OUT_DIR.mkdir(
            parents=True, exist_ok=True
        )  # Create the directory if it doesn't exist

        df = utils.load_csv(
            config.DATA_FOLDER, config.DATASET_FILE_NAME
        )  # Load the dataset from the specified CSV file

        df["True_Class"] = df.apply(
            assign_true_class, axis=1
        )  # Apply the assign_true_class function to each row in the DataFrame

        df_with_labels = (
            df.copy()
        )  # Create a copy of the DataFrame with assigned true classes
        utils.export_csv(
            df_with_labels,
            folder_name=config.GROUND_TRUTH_FOLDER,
            file_name="Full_" + str(config.GROUND_TRUTH_FILE_NAME),
        )  # Export the DataFrame with true classes to a CSV file

        intentional_count = sum(
            df["True_Class"] == "Intentional"
        )  # Count the occurrences of each true class in the DataFrame
        unintentional_count = sum(
            df["True_Class"] == "Unintentional"
        )  # Count the occurrences of each true class in the DataFrame
        normal_count = sum(
            df["True_Class"] == "Normal"
        )  # Count the occurrences of each true class in the DataFrame

        logger.info(
            f"Labeled {len(df)} rows: Intentional={intentional_count}, "
            f"Unintentional={unintentional_count}, "
            f"Normal={normal_count}"
        )  # Log the counts of each true class in the DataFrame

        sample_size = min(
            config.GROUND_SAMPLE_SIZE, len(df_with_labels)
        )  # Define the sample size, ensuring it does not exceed the number of rows in the DataFrame
        sample = df_with_labels.sample(
            n=sample_size, random_state=42
        )  # Randomly sample rows from the DataFrame with a fixed random state for reproducibility
        utils.export_csv(
            sample,
            folder_name=config.GROUND_TRUTH_FOLDER,
            file_name="Sample_" + str(config.GROUND_TRUTH_FILE_NAME),
        )  # Export the sampled DataFrame to a CSV file

        sample_intentional_count = sum(
            sample["True_Class"] == "Intentional"
        )  # Count the occurrences of each true class in the sampled DataFrame
        sample_unintentional_count = sum(
            sample["True_Class"] == "Unintentional"
        )  # Count the occurrences of each true class in the sampled DataFrame
        sample_normal_count = sum(
            sample["True_Class"] == "Normal"
        )  # Count the occurrences of each true class in the sampled DataFrame

        logger.info(
            f"Sample labeled dataset: {len(sample)} rows | "
            f"Intentional={sample_intentional_count}, "
            f"Unintentional={sample_unintentional_count}, "
            f"Normal={sample_normal_count}"
        )  # Log the counts of each true class in the sampled DataFrame

        for cls in [
            "Normal",
            "Intentional",
            "Unintentional",
        ]:  # Iterate through each true class
            cls_df = df_with_labels[
                df_with_labels["True_Class"] == cls
            ]  # Filter the DataFrame for the current true class
            utils.export_csv(
                cls_df,
                folder_name=config.GROUND_TRUTH_FOLDER,
                file_name=f"{cls}_Data_supervised",
            )  # Export the filtered DataFrame to a CSV file named after the true class
            logger.info(
                f"Exported {len(cls_df)} rows to {cls}_Data_supervised.csv for supervised pipeline."
            )  # Log the number of rows exported for the current true class

    except Exception as e:  # Catch any exceptions that occur during execution
        logger.error(f"An error occurred: {e}")  # Log the error message
        raise  #

    finally:  # Finally block to ensure proper cleanup and logging
        try:  # Try-except block to catch and log any errors during cleanup
            system.cleanup_resources()  # Cleanup temporary resources

            script_end = time.time()  # End the script execution timer
            duration = script_end - script_start  # Calculate the total script execution
            logger.info(
                f"\nTotal script execution time: {Fore.LIGHTYELLOW_EX}{duration:.2f}s"
            )  # Log the total script execution time

        except Exception as e:  # Catch and log any errors during cleanup
            logger.error(f"\nCleanup Failed: {e}")  # Log the error
            raise  # Raise the exception


# =========== Script Entry Point ===========
if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Execute the main function
