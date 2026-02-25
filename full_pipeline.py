"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Set Scenario ID in Config: Function to update the scenario ID in the configuration file.
    - Run Experiment: Function to run the data processing pipeline for a given scenario ID.
    - Main Function: The main entry point for the data processing pipeline.
    - Script Entry Point: The conditional block that executes the main function.
"""

# =========== Imports and Dependencies ===========
import sys  #    Used to exit the script with an error code if needed
import subprocess  # Used to run external Python scripts as subprocesses
import time

from modules import (
    config as config_module,
)  # Configuration module containing paths and settings
from modules.system_guard import (
    logger,
)  # logger.info for logging messages and errors, and function to create a Dask client for parallel processing

CONFIG_PATH = config_module.__file__  # Path to the configuration file


# =========== Set Scenario ID in Config ===========
def set_scenario_id_in_config(scenario_id):
    """
    Function which updates the scenario ID in the configuration file.
    """
    with open(CONFIG_PATH, "r") as f:  # Open the config file for reading
        lines = f.readlines()  # Read all lines from the config file
    new_lines = []  # List to hold modified lines
    found = False  # Flag to check if RUN_SCENARIO_ID was found
    for line in lines:  # Iterate through each line in the config file
        if line.startswith(
            "RUN_SCENARIO_ID"
        ):  # Check if the line defines RUN_SCENARIO_ID
            new_lines.append(
                f"RUN_SCENARIO_ID = {scenario_id}\n"
            )  # Update the scenario ID
            found = True  #   Set the flag to True if RUN_SCENARIO_ID was found
        else:  # If the line does not define RUN_SCENARIO_ID, keep it unchanged
            new_lines.append(line)  # Append the line as is to new_lines
    if not found:  # If RUN_SCENARIO_ID was not found in the file, append it
        new_lines.append(
            f"RUN_SCENARIO_ID = {scenario_id}\n"
        )  # Append the new scenario ID
    with open(CONFIG_PATH, "w") as f:  # Open the config file for writing
        f.writelines(new_lines)  # Write the modified lines back to the config file


# =========== Run Experiment ===========
def run_experiment(scenario_id):
    """
    Function which runs the data processing pipeline for a given scenario ID.
    """
    scripts = [
        (
            "dataset_creation/dataset_creation.py",
            "Dataset creation",
        ),
        (
            "ground_truth_labeling/ground_truth_labeling.py",
            "Ground truth labeling",
        ),
        ("main.py", "Main pipeline"),
    ]  # List of scripts to run in the pipeline, with their descriptions
    for (
        script,
        description,
    ) in scripts:  # Iterate through each script and its description
        logger.info(f"Running {description}...")  # Log the start of the script
        logger.info(f"Script path: {script}")  # Log the path of the script being run
        try:  # Run the script as a subprocess
            subprocess.run(
                ["python", script, "--scenario", str(scenario_id)], check=True
            )  # Run the script as a subprocess with the scenario ID as an argument
            logger.info(
                f"{description} completed successfully.\n"
            )  # Log success message
        except (
            subprocess.CalledProcessError
        ) as e:  # Catch any errors that occur during script execution
            logger.error(
                f"Error: {description} failed with exit code {e.returncode}. Stopping the pipeline."
            )  # Log the error message and exit code
            sys.exit(1)  # Exit the script with an error code if any script fails


# =========== Main ===========
def main():
    """
    Function which serves as the main entry point for the data processing pipeline.
    """
    start_time = time.time()  # Start the timer to measure total runtime
    user_input = input(
        "Please enter the scenario ID(s) to run (1–3, or 4 for all): "
    )  # Prompt the user for scenario IDs
    scenario_ids = []  # Initialize an empty list to hold scenario IDs
    if user_input.strip() == "4":  # If the user inputs '5', run all scenarios
        scenario_ids = [1, 2, 3]  # List of all scenario IDs
    else:  # If the user inputs specific scenario IDs, split and process them
        try:  # Split the input by commas and convert to integers
            parts = [
                part.strip() for part in user_input.split(",")
            ]  # Split input by commas and strip whitespace
            scenario_ids = [
                int(part) for part in parts
            ]  # Convert each part to an integer
        except ValueError:  # If conversion fails, log an error and exit
            logger.error(
                "Invalid input! Please enter integers (1–3) separated by commas, or 4 for all."  # Log an error message
            )
            sys.exit(1)  # Exit the script with an error code if input is invalid
    for sid in scenario_ids:  # Check if each scenario ID is valid
        if sid not in [1, 2, 3, 4]:  # If the scenario ID is not in the valid range
            logger.warning(
                "Scenario IDs must be between 1 and 4"
            )  # Log a warning message
            sys.exit(
                1
            )  # Exit the script with an error code if an invalid scenario ID is provided
    for (
        scenario_id
    ) in scenario_ids:  # Iterate through each scenario ID provided by the user
        set_scenario_id_in_config(
            scenario_id
        )  # Update the scenario ID in the configuration file
        run_experiment(
            scenario_id
        )  # Run the data processing pipeline for the specified scenario ID
        logger.info(
            f"==== Finished scenario {scenario_id} ===="
        )  # Log completion of the scenario
        logger.info(
            "Scenario config summary:\n"
            f"RUN_SCENARIO_ID: {getattr(config_module, 'RUN_SCENARIO_ID', 'N/A')}\n"
            f"DATA_FOLDER: {getattr(config_module, 'DATA_FOLDER', 'N/A')}\n"
            f"GROUND_TRUTH_FOLDER: {getattr(config_module, 'GROUND_TRUTH_FOLDER', 'N/A')}\n"
            f"OUTPUT_FOLDER: {getattr(config_module, 'OUTPUT_FOLDER', 'N/A')}\n"
            f"INTENTIONAL_PERC: {getattr(config_module, 'INTENTIONAL_PERC', 'N/A')}\n"
            f"UNINTENTIONAL_PERC: {getattr(config_module, 'UNINTENTIONAL_PERC', 'N/A')}\n"
            f"MISSING_PERC: {getattr(config_module, 'MISSING_PERC', 'N/A')}\n"
            f"OVERLAP_PERC: {getattr(config_module, 'OVERLAP_PERC', 'N/A')}\n"
        )  # Log the scenario configuration summary
    total_time = (
        time.time() - start_time
    )  # Calculate the total runtime for all scenarios
    logger.info(
        f"\nTotal runtime for all selected scenarios: {total_time:.2f} seconds"
    )  # Log the total runtime for all scenarios


# =========== Script Entry Point ===========
if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Execute the main function
