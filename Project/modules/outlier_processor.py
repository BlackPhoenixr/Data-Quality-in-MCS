"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Dictionary: Outlier Detection Methods: A dictionary mapping outlier detection methods to their respective functions.
    - Process in Chunks: Process a DataFrame in chunks, apply outlier detection, and return cleaned subsets.
    - Process All Outliers: Apply outlier detection to each column in a chunk and return cleaned data.
    - Apply Outliers: Apply a single outlier detection method to a specific column and return the cleaned DataFrame.
"""

# =========== Imports and Dependencies ===========
from dask import (
    delayed,
    compute,
)  # Dask library for parallel computing and delayed execution
from typing import (
    Dict,
    Any,
)  # Type hints for function parameters and return values.
import gc  # Provides garbage collection functionality for optimizing memory usage.
import numpy as np  # Used for splitting a DataFrame into chunks (via numpy.array_split).
import pandas as pd  # Core library for DataFrame creation, manipulation, and analysis.
from colorama import (
    Fore,
)  # Enables colored console output for improved log readability.

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Contains global variables and system functions.
from modules.system_guard import logger  # For logging and system-related functions.
import modules.utils as utils  # File handling utilities for loading and saving data.
import modules.data_processor as processor  # Handles column mapping, validation, and encoding processes.


# =========== Process in Chunks ===========
@system.timer_decorator
def process_in_chunks(
    data: pd.DataFrame, column_types: Dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function which processes a DataFrame in chunks, applies outlier detection methods, and returns cleaned subsets.

    Features
        - Processes the data in chunks to optimize memory usage.
        - Applies outlier detection methods to each column in parallel.
        - Returns cleaned data subsets and detected outliers.

    Parameters
    ---------------------------------
        - data: pd.DataFrame
            The input DataFrame to process.
        - column_types: Dict[str, Any]
            A dictionary mapping column names to their respective types.
        - client: Any
            The Dask client for parallel processing.

    Returns
    ---------------------------------
        - normal_data: pd.DataFrame
            The cleaned data subset with outliers removed.
        - misbehaving_data: pd.DataFrame
            The detected outliers in the dataset.

    Raises Exception
    ---------------------------------
        If an error occurs during processing.
    """
    chunk_size = max(
        1, int(len(data) * config.CHUNK_PERCENT)
    )  # Calculate chunk size based on percentage of data.

    outlier_exclude_columns = [
        "User_Id",
        "Timestamp",
        "User_Reputation",
        "User_Experience",
    ]  # Columns to exclude from outlier processing

    all_chunk_methods_metrics = (
        []
    )  # List to store outlier detection metrics for each chunk.

    try:  # Attempt to process the data in chunks.
        columns_to_exclude = (
            outlier_exclude_columns if outlier_exclude_columns else []
        )  # Columns to exclude from outlier detection.

        normal_results = []  # List to store cleaned data subsets.
        misbehaving_results = []  # List to store detected outliers.
        all_outlier_indices = set()  # Set to store unique outlier indices.

        original_size = len(data)  # Original size of the dataset.

        n_chunks = max(1, len(data) // chunk_size)  # Calculate the number of chunks.
        logger.debug(
            f"Processing {n_chunks} chunks with chunk size ~{chunk_size}"
        )  # Log the number of chunks.

        chunks = np.array_split(data, n_chunks)  # Split the data into chunks.

        tasks = [
            delayed(process_all_outliers)(chunk, column_types, columns_to_exclude)
            for chunk in chunks
        ]  # Create delayed tasks for processing each chunk.
        outlier_results = compute(*tasks)  # Execute the delayed tasks in parallel.

        chunk_counter = 1  # Counter for tracking the current chunk being processed.

        for (
            result
        ) in outlier_results:  # Iterate over the results from the parallel tasks.

            normal_chunk, misbehaving_chunk, chunk_metrics = (
                result  # Unpack the results for the current chunk.
            )

            normal_results.append(normal_chunk)  # Append the cleaned data subset.

            misbehaving_results.append(
                misbehaving_chunk
            )  # Append the detected outliers.

            for (
                col,
                metrics,
            ) in (
                chunk_metrics.items()
            ):  # Iterate over the column metrics for the chunk.
                best_method = max(
                    metrics["methods"].items(),
                    key=lambda x: x[1]["outliers_removed"],
                )[
                    0
                ]  # Get the name of the best method based on outliers removed.
                best_indices = metrics["methods"][best_method].get(
                    "outlier_indices", set()
                )  # Get the outlier indices for the best method.
                all_outlier_indices.update(
                    best_indices
                )  # Update the set of unique outlier indices.

            for (
                col,
                col_data,
            ) in (
                chunk_metrics.items()
            ):  # Iterate over the column metrics for the chunk.
                if "all_methods" in col_data:  # Check if all methods data is available.
                    for method_name, info in col_data["all_methods"].items():  #
                        all_chunk_methods_metrics.append(
                            {
                                "column": col,
                                "method": method_name,
                                "outliers_removed": info["outliers_removed"],
                                "outlier_indices": list(
                                    info.get("outlier_indices", [])
                                ),
                                "chunk_id": chunk_counter,
                            }
                        )  # Append the metrics for each method.

            if all_chunk_methods_metrics:  # Check if there are any metrics to export.
                df_all_methods = pd.DataFrame(
                    all_chunk_methods_metrics
                )  # Create a DataFrame from the metrics.
                df_outlier_numbers = df_all_methods[
                    [
                        "method",
                        "column",
                        "outliers_removed",
                        "outlier_indices",
                        "chunk_id",
                    ]
                ]  # Select relevant columns for export.
                utils.export_csv(
                    df_outlier_numbers,
                    folder_name=config.OUTLIER_DATA,
                    file_name="outlier_numbers",
                    chunk_id=chunk_counter,
                )  # Export the metrics to a CSV file.

            chunk_counter += 1  # Increment the chunk counter.

        normal_data = pd.concat(
            normal_results, ignore_index=True
        )  # Concatenate the cleaned data subsets.

        misbehaving_data = pd.concat(
            misbehaving_results, ignore_index=True
        )  # Concatenate the detected outliers.

        misbehaving_data.sort_values(
            by="Timestamp", inplace=True
        )  # Sort the misbehaving data by timestamp.

        utils.export_csv(
            misbehaving_data,
            folder_name=config.SPLITTED_DATA,
            file_name="Misbehaving_Data",
        )  # Export the misbehaving data to a CSV file.

        utils.export_csv(
            normal_data,
            folder_name=config.SPLITTED_DATA,
            file_name="Normal_Data",
        )  # Export the cleaned data to a CSV file.

        unique_outliers_removed = len(
            all_outlier_indices
        )  # Calculate the number of unique outliers removed.

        cleaned_size = (
            original_size - unique_outliers_removed
        )  # Calculate the size of the cleaned dataset.

        removal_percentage = (
            (unique_outliers_removed / original_size * 100) if original_size > 0 else 0
        )  # Calculate the percentage of outliers removed.

        config.FINAL_OUTLIER_SUMMARY = {
            "original_size": original_size,
            "cleaned_size": cleaned_size,
            "unique_outliers_removed": unique_outliers_removed,
            "removal_percentage": removal_percentage,
        }  # Update the final outlier summary in the config.

        logger.info(
            f"\n{Fore.CYAN} ============ Final Unique Outlier Summary ============{Fore.RESET}\n"
            f"{Fore.BLUE}Overall Outlier Summary:{Fore.RESET}\n"
            f"- Original dataset rows:     {Fore.WHITE}{original_size:,}{Fore.RESET}\n"
            f"- Normal dataset rows:       {Fore.GREEN}{cleaned_size:,}{Fore.RESET}\n"
            f"- Misbehaving dataset rows:  {Fore.RED}{unique_outliers_removed:,}{Fore.RESET}\n"
            f"- Removal rate:              {Fore.YELLOW}{removal_percentage:.2f}%{Fore.RESET}\n"
        )  # Log the final outlier summary.

        return normal_data, misbehaving_data  # Return the cleaned and misbehaving data.

    except Exception as e:  # Catch any exceptions that occur during processing.
        logger.error(f"\nChunk processing failed: {e}")  # Log an error.
        raise  # Raise an exception.


# =========== Process All Outliers ===========
@system.timer_decorator
def process_all_outliers(
    data: pd.DataFrame, column_types: dict, columns_to_exclude=None
):
    """
    Function which applies outlier detection to each column in a chunk and returns cleaned data

    Features
    ---------------------------------
        - Applies outlier detection methods to each column in a chunk.
        - Returns cleaned data subsets and detected outliers.

    Parameters
    ---------------------------------
        - data: pd.DataFrame
            The input DataFrame to process.
        - column_types: dict
            A dictionary mapping column names to their respective types.
        - columns_to_exclude: list
            A list of columns to exclude from outlier detection.

    Returns
    ---------------------------------
        - normal_data: pd.DataFrame
            The cleaned data subset with outliers removed.
        - misbehaving_data: pd.DataFrame
            The detected outliers in the dataset.
        - column_metrics: dict
            A dictionary containing outlier detection metrics for each column.

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """
    exclude_cols = (
        columns_to_exclude if columns_to_exclude else []
    )  # Columns to exclude from outlier detection.

    try:  # Attempt to process the outliers in the data.
        column_metrics = (
            {}
        )  # Dictionary to store outlier detection metrics for each column.

        outlier_indices = set()  # Set to store the indices of detected outliers.

        original_data_copy = data.copy()  # Create a copy of the original data.

        processed_data = processor.encode_and_scale_features(
            data.copy(), fit=True, scaler_type="standard"
        )  # Encode and scale the features in the data.
        feature_columns = [
            col for col in processed_data.columns if col not in (exclude_cols or [])
        ]  # Get the feature columns excluding the specified columns.

        for column in feature_columns:  # Iterate over each feature column in the data.

            original_data = (
                processed_data.copy()
            )  # Create a copy of the data for processing.
            method_metrics_list = (
                []
            )  # List to store outlier detection metrics for each method.

            for (
                method_name,
                method_func,
            ) in (
                config.OUTLIER_METHODS.items()
            ):  # Iterate over the outlier detection methods.
                try:  # Apply the method to the column and get the cleaned data.
                    temp_data = apply_outliers(
                        method_func, processed_data.copy(), column, method_name
                    )  # Apply the outlier detection method to the column.
                    outliers_removed = len(original_data) - len(
                        temp_data
                    )  # Calculate the number of outliers removed.
                    outlier_indices_method = set(original_data.index) - set(
                        temp_data.index.tolist()
                    )  # Get the indices of the outliers removed.

                    logger.debug(
                        f"{method_name} removed {outliers_removed} rows from {column}"
                    )  # Log the number of outliers removed by the method.

                    method_metrics_list.append(
                        {
                            "method": method_name,
                            "column": column,
                            "outliers_removed": outliers_removed,
                            "original_size": len(original_data),
                            "cleaned_size": len(temp_data),
                            "outlier_indices": outlier_indices_method,
                        }
                    )  # Append the metrics for the method.

                except (
                    Exception
                ) as method_e:  # Catch any exceptions that occur during method application.
                    logger.error(
                        f"\nOutlier detection failed for method '{method_name}' on column '{column}': {method_e}"  # Log an error.
                    )
                    raise  # Raise an exception.

            if not method_metrics_list:  # Check if any methods were successful.
                logger.warning(
                    f"\nNo successful outlier detection methods for column '{column}'. Skipping."  # Log a warning.
                )
                continue  # Skip the column if no methods were successful.

            removal_counts = [
                mm["outliers_removed"] for mm in method_metrics_list
            ]  # Get the number of outliers removed by each method.
            median_count = np.median(
                removal_counts
            )  # Calculate the median number of outliers removed.
            best_method_metrics = min(
                method_metrics_list,
                key=lambda x: abs(x["outliers_removed"] - median_count),
            )  # Find the method closest to the median number of outliers removed.
            best_method_name = best_method_metrics[
                "method"
            ]  # Get the name of the best method.
            best_outlier_indices = best_method_metrics[
                "outlier_indices"
            ]  # Get the outlier indices for the best method.

            outlier_indices.update(
                best_outlier_indices
            )  # Update the set of outlier indices.
            processed_data = processed_data.drop(
                index=best_outlier_indices
            )  # Drop the outliers from the data.
            removal_percentage = (
                (
                    best_method_metrics["outliers_removed"]
                    / best_method_metrics["original_size"]
                    * 100
                )  # Calculate the percentage of outliers removed.
                if best_method_metrics["original_size"]
                > 0  # Check if the original size is greater than 0.
                else 0  # Set to 0 if original size is 0.
            )

            logger.info(
                f"""
                {Fore.GREEN}Column: {column}{Fore.RESET}
                {Fore.YELLOW}{'='*40}{Fore.RESET}
                {Fore.BLUE}Best Method: {best_method_name}{Fore.RESET}
                - Original rows:     {Fore.WHITE}{best_method_metrics['original_size']:,}{Fore.RESET}
                - Remaining rows:    {Fore.GREEN}{best_method_metrics['cleaned_size']:,}{Fore.RESET}
                - Outliers removed:  {Fore.RED}{best_method_metrics['outliers_removed']:,}{Fore.RESET}
                - Removal rate:      {Fore.YELLOW}{removal_percentage:.2f}%{Fore.RESET}
                - Outlier indices:   {Fore.CYAN}{sorted(list(best_outlier_indices))}{Fore.RESET}
                """
            )  # Log the metrics for the best method.

            column_metrics[column] = {
                "methods": {
                    best_method_name: {
                        "method": best_method_name,
                        "column": column,
                        "outliers_removed": best_method_metrics["outliers_removed"],
                        "original_size": best_method_metrics["original_size"],
                        "cleaned_size": best_method_metrics["cleaned_size"],
                        "outlier_indices": best_method_metrics["outlier_indices"],
                    }
                },
                "outliers_removed": best_method_metrics["outliers_removed"],
                "original_size": best_method_metrics["original_size"],
                "cleaned_size": best_method_metrics["cleaned_size"],
            }  # Store the overall metrics for the column.

            column_metrics[column]["all_methods"] = {
                mm["method"]: {
                    "outliers_removed": mm["outliers_removed"],
                    "outlier_indices": mm["outlier_indices"],
                }
                for mm in method_metrics_list
            }  # Store the metrics for all methods applied to the column.

            gc.collect()  # Perform garbage collection to free up memory.

        outlier_indices_list = list(
            outlier_indices
        )  # Convert the set of outlier indices to a list.
        normal_indices = list(
            set(original_data_copy.index) - set(outlier_indices_list)
        )  # Get the indices of normal data.

        misbehaving_data = original_data_copy.loc[
            outlier_indices_list
        ]  # Get the misbehaving data.

        normal_data = original_data_copy.loc[normal_indices]  # Get the normal data.

        return (
            normal_data,
            misbehaving_data,
            column_metrics,
        )  # Return the cleaned and misbehaving data, and the column metrics.

    except Exception as e:  # Catch any exceptions that occur during processing.
        logger.error(f"Outlier processing failed: {e}")  # Log an error.
        raise  # Raise an exception.


# =========== Apply Outliers ===========
@system.timer_decorator
def apply_outliers(method, data: pd.DataFrame, column_name: str, method_name: str):
    """
    Function which applies a single outlier detection method to a specific column and returns the cleaned DataFrame

    Features
    ---------------------------------
        - Applies a single outlier detection method to a specific column.
        - Returns the cleaned DataFrame with outliers removed.

    Parameters
    ---------------------------------
        - method: function
            The outlier detection method to apply.
        - data: pd.DataFrame
            The input DataFrame to process.
        - column_name: str
            The name of the column to process.
        - method_name: str
            The name of the outlier detection method.

    Returns
    ---------------------------------
        - cleaned_column: pd.DataFrame
            The cleaned DataFrame with outliers removed.

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """
    try:  # Attempt to apply the outlier detection
        column_data = data[[column_name]]  # Get the column data.

        cleaned_column = delayed(method)(
            column_data, column_name
        )  # Apply the outlier detection method as a Dask delayed task.

        if hasattr(
            cleaned_column, "compute"
        ):  # Check if the result is a Dask DataFrame.
            cleaned_column = cleaned_column.compute()  # Compute the Dask DataFrame.

        if isinstance(
            cleaned_column, pd.DataFrame
        ):  # Check if the result is a DataFrame.
            return cleaned_column  # Return the cleaned column.

        else:  # If the result is not a DataFrame.
            logger.error(
                f"\nType Check: Expected DataFrame from {method_name} for {column_name}, "
                f"but received {type(cleaned_column)} instead."
            )  # Log an error.
            raise  # Raise an exception.

    except Exception as e:  # Catch any exceptions that occur during processing.
        logger.error(
            f"\nOutlier Detection Failed: {column_name} | {method_name} | {str(e)}"
        )  # Log an error.
        raise  # Raise an exception.
