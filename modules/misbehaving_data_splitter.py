"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Split Misbehaving Data: Splits misbehaving data into intentional and unintentional subsets.
"""

# =========== Imports and Dependencies ===========
import pandas as pd  # Used for creating and manipulating DataFrames.

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import logger  # Logger for logging messages and errors.
import modules.utils as utils  # File utilities for loading/exporting CSV files.
import modules.data_processor as processor  # Handles column mapping, validation, and encoding processes.


# =========== Split Misbehaving Data ===========
@system.timer_decorator
def split_misbehaving_data(
    misbehaving_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function which splits the misbehaving data into intentional and unintentional subsets.

    Parameters
    ----------
    misbehaving_data : pd.DataFrame
        DataFrame containing all misbehaving user records.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames: (intentional_data, unintentional_data)

    Raises Exceptions
    -----------------
    - Raises an exception if any required columns are missing in the DataFrame.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        features = config.FEATURES  # List of features for prediction

        if not all(
            col in misbehaving_data.columns for col in features
        ):  # Check if all required columns are present in the DataFrame
            missing = list(
                set(features) - set(misbehaving_data.columns)
            )  # Identify missing columns
            logger.error(
                f"Missing required columns in misbehaving data: {missing}"
            )  # Log the missing columns
            raise  # Raise an error if any required columns are missing

        misbehaving_data = misbehaving_data.reset_index(
            drop=True
        )  # Reset the index of the DataFrame
        X = misbehaving_data[
            features
        ].copy()  # Copy the feature columns from the misbehaving data.
        X = X.reset_index(drop=True)

        X = processor.encode_and_scale_features(
            X, fit=True, encoder_path=config.ENCODER_MODEL_PATH, scaler_type="minmax"
        )  # Encode categorical columns using the processor module.

        logger.info(
            "Applying outlier detection methods column-wise and aggregating outliers to identify intentional misbehaving users..."
        )  # Log the start of the outlier detection process

        columns_to_check = [
            c
            for c in getattr(config, "INTENTIONAL_OUTLIER_COLUMNS", [])
            if c in X.columns
        ]  # Get the columns to check for intentional outliers from the config.
        if (
            not columns_to_check
        ):  # If no specific columns are provided, fallback to all columns.
            columns_to_check = X.columns  # fallback to all if none specified or present
        logger.info(
            f"Using columns for intentional outlier detection: {list(columns_to_check)}"
        )  # Log the columns used for intentional outlier detection

        outlier_indices_dict = {}  # Dictionary to store outlier indices for each method

        for (
            method_name,
            method_func,
        ) in (
            config.OUTLIER_METHODS.items()
        ):  # Iterate through each outlier detection method
            try:  # Apply the method to each column and aggregate the outliers
                aggregated_outliers = set()  # Set to store unique outlier indices
                for col in columns_to_check:  # Iterate through each column
                    outlier_df = method_func(
                        X, col
                    )  # Apply the outlier detection method to the column
                    outlier_rows = (
                        outlier_df.index
                    )  # Get the indices of the outlier rows
                    aggregated_outliers.update(
                        outlier_rows.tolist()
                    )  # Update the set with the new outlier indices
                outlier_indices_dict[method_name] = (
                    aggregated_outliers  # Store the aggregated outliers for the method
                )
                logger.info(
                    f"Method {method_name} detected {len(aggregated_outliers)} unique outliers aggregated across all columns."
                )  # Log the number of unique outliers detected by the method
            except (
                Exception
            ) as e:  # Catch any exceptions that occur during the method application
                logger.warning(
                    f"Method {method_name} failed with error: {e}"
                )  # Log the error

        if (
            not outlier_indices_dict
        ):  # If no outlier detection methods succeeded, log an error and raise an exception
            logger.error("No outlier detection methods succeeded.")  # Log the error
            raise  # Raise an exception to indicate failure

        votes = {
            idx: 0 for idx in misbehaving_data.index
        }  # Initialize a dictionary to count votes for each row
        for (
            outlier_set
        ) in (
            outlier_indices_dict.values()
        ):  # Iterate through the outlier sets from each method
            for idx in outlier_set:  # For each outlier index in the set
                if idx in votes:  # If the index is in the votes dictionary
                    votes[idx] += 1  # Increment the vote count for the index

        vote_series = pd.Series(votes, name="vote_count").reindex(
            misbehaving_data.index, fill_value=0
        )  # Create a Series from the votes dictionary and reindex it to match the original DataFrame

        intentional_perc = getattr(
            config, "INTENTIONAL_PERC", 0.15
        )  # Get the intentional percentage from the config
        n_intentional = max(
            1, int(round(len(vote_series) * intentional_perc))
        )  # Calculate the number of intentional outliers based on the percentage

        best_outlier_indices = set(
            vote_series.sort_values(ascending=False).head(n_intentional).index
        )  # Get the indices of the top intentional outliers based on the vote count
        best_method_name = f"VOTE_TOP_{intentional_perc:.2f}"  # Name of the best method based on the vote count
        logger.info(
            f"Picked top {intentional_perc:.0%} of vote counts "
            f"→ {len(best_outlier_indices)} intentional rows."
        )  # Log the number of intentional rows picked based on the vote count

        total_rows = len(
            misbehaving_data
        )  # Total number of rows in the misbehaving data
        if len(best_outlier_indices) in (
            0,
            total_rows,
        ):  # If no intentional outliers or all rows are flagged
            logger.warning(
                "Best-method selection flagged 0 or ALL rows; treating as no intentional outliers."
            )  # Log a warning
            best_outlier_indices = (
                set()
            )  # Reset the best outlier indices to an empty set

        misbehaving_data = (
            misbehaving_data.copy()
        )  # Create a copy of the misbehaving data to avoid modifying the original DataFrame
        misbehaving_data["Is_Intentional"] = (
            0  # Initialize a new column to mark intentional outliers
        )
        if len(best_outlier_indices) > 0:  # If there are intentional outliers detected
            misbehaving_data.loc[list(best_outlier_indices), "Is_Intentional"] = (
                1  # Mark the detected intentional outliers in the DataFrame
            )
        else:  # If no intentional outliers detected, log a warning
            logger.warning(
                "No intentional outliers detected by the best method; no rows assigned as intentional."
            )  # Log the warning

        intentional_data = misbehaving_data[
            misbehaving_data["Is_Intentional"] == 1
        ].copy()  # Filter the misbehaving data to get intentional users.
        unintentional_data = misbehaving_data[
            misbehaving_data["Is_Intentional"] == 0
        ].copy()  # Filter the misbehaving data to get unintentional users.

        logger.info(
            f"Final split counts: {len(intentional_data)} intentional, {len(unintentional_data)} unintentional users."
        )  # Log the final counts of intentional and unintentional users

        utils.export_csv(
            intentional_data, config.SPLITTED_DATA, "Intentional_Data"
        )  # Export the intentional data to a CSV file.
        utils.export_csv(
            unintentional_data, config.SPLITTED_DATA, "Unintentional_Data"
        )  # Export the unintentional data to a CSV file.

        logger.info(
            f"Split and exported intentional and unintentional misbehaving users using method '{best_method_name}'."
        )  # Log the successful split and export of the data.

        return (
            intentional_data,
            unintentional_data,
        )  # Return the intentional and unintentional data as a tuple.

    except Exception as e:  # Catch unexpected exceptions
        logger.error(f"Failed to split misbehaving data: {e}")  # Log the error
        raise  # Raise the exception to ensure visibility and prevent silent failures
