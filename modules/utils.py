"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Load CSV Files: A function to load CSV files into DataFrames with error handling and logging.
    - Export CSV Files: A function to export DataFrames to CSV files with error handling and logging.
"""

# =========== Imports and Dependencies ===========
from pathlib import Path  # Importing Path for file path handling
import pandas as pd  # Importing pandas for DataFrame manipulation
from colorama import Fore  # Importing Fore for colored terminal text

from modules.system_guard import (
    timer_decorator,
    logger,
)  # Importing timer_decorator and logger for logging and timing functions


# =========== Load CSV Files ===========
@timer_decorator
def load_csv(folder_name, file_name, sep=","):
    """
    Function which loads a CSV file into a DataFrame with robust error handling and logging.

    Features
    ---------------------
        - Constructs the full file path using the base directory, folder name, and file name.
        - Checks if the file exists at the specified path and logs an error if it does not.
        - Loads the CSV file into a DataFrame using `pandas.read_csv()` with the specified separator.
        - Missing values represented as empty strings or "None" are automatically interpreted as NaN and imputed:
            * Numeric columns have missing values filled with the median.
            * Categorical columns have missing values filled with the mode or "Unknown" if mode is empty.
        - Logs a success message if the DataFrame is loaded successfully.

    Parameters
    ---------------------
    - folder_name : str or Path
        The directory where the CSV file is stored.
    - file_name : str
        The name of the file **without** the `.csv` extension.
    - sep : str, optional
        The delimiter used in the CSV file (default is ',').

    Returns
    ---------------------
        dataframe :
            A DataFrame containing the contents of the loaded CSV file, with missing values imputed.

    Raises Exception
    ---------------------
        If the file cannot be read due to parsing errors or other issues.
    """
    try:  # Attempt to load the CSV file into a DataFrame.
        if not str(file_name).endswith(
            ".csv"
        ):  # If the file name does not end with ".csv".
            file_name = (
                f"{file_name}.csv"  # Append the ".csv" extension to the file name.
            )

        full_path = (
            Path(folder_name) / file_name
        ).resolve()  # Construct the full path to the CSV file.

        if not full_path.exists():  # If the file does not exist at the specified path.
            logger.error(
                f"\nCSV Loading: File not found at {full_path}"
            )  # Log an error message.
            raise  # Raise an exception to halt execution and propagate the error.

        dataframe = pd.read_csv(
            str(full_path), sep=sep, na_values=["", "None"]
        )  # Load the CSV file into a DataFrame, treating empty strings and "None" as NaN.

        if len(dataframe) == 0:  # If the DataFrame is empty after loading.
            logger.error(
                f"CSV Loading: Dataset at {full_path} is empty."
            )  # Log an error message.
            raise  # Raise an exception to halt execution and propagate the error.

        if not isinstance(
            dataframe, pd.DataFrame
        ):  # Ensure the loaded object is a valid DataFrame
            logger.error(
                f"CSV Loading: Loaded object is not a valid pandas DataFrame from {full_path}"
            )  # Log an error message.
            raise  # Raise an exception to halt execution and propagate the error.

        numeric_cols = dataframe.select_dtypes(
            include=["number"]
        ).columns  # Select numeric columns from the DataFrame.
        for col in numeric_cols:  # Iterate over each numeric column.
            median_val = dataframe[
                col
            ].median()  # Calculate the median value of the column.
            dataframe[col].fillna(
                median_val, inplace=True
            )  # Fill missing values in the column with the median value.

        categorical_cols = dataframe.select_dtypes(
            include=["object"]
        ).columns  # Select categorical columns from the DataFrame.
        for col in categorical_cols:  # Iterate over each categorical column.
            mode_val = dataframe[col].mode()  # Calculate the mode value of the column.
            if not mode_val.empty:  # If the mode value is not empty.
                fill_val = mode_val[
                    0
                ]  # Use the first mode value to fill missing values.
            else:  # If the mode value is empty (no mode found).
                fill_val = (
                    "Unknown"  # Use "Unknown" as the fill value for missing values.
                )
            dataframe[col].fillna(
                fill_val, inplace=True
            )  # Fill missing values in the column with the fill value.

        logger.info(
            f"\nCSV loaded {Fore.GREEN} successfully {Fore.RESET} from {Fore.YELLOW} {full_path}"
        )  # Log a success message indicating the DataFrame was loaded successfully.

        return dataframe  # Return the loaded DataFrame.

    except Exception as e:  # If an error occurs during the CSV loading operation.
        logger.error(
            f"CSV Loading: Operation failed | {str(e)}"
        )  # Log an error message if the CSV loading operation fails.
        raise  # Raise an exception to halt execution and propagate the error.


# =========== Export CSV Files ===========
@timer_decorator
def export_csv(
    dataframe, folder_name, file_name, chunk_id=None, sep=",", index=False, append=False
) -> Path:
    """
    Function which exports a DataFrame to a CSV file with robust error handling and logging.

    Features
    ---------------------
        - Constructs the full file path using the base directory, folder name, and file name.
        - Creates the folder if it does not exist and logs an error if it cannot be created.
        - Appends the `.csv` extension to the file name if it is not already present.
        - Sorts the DataFrame by the `Timestamp` column if it exists to ensure chronological order.
        - Logs an informational message indicating the file being exported.
        - Exports the DataFrame to a CSV file using `pandas.to_csv()` with the specified separator.
        - Logs a success message if the DataFrame is exported successfully.

    Parameters
    ---------------------
    - dataframe : pd.DataFrame
        The DataFrame to be exported to a CSV file.
    - folder_name : str or Path
        The directory where the CSV file should be stored.
    - file_name : str or Path
        The name of the file **without** the `.csv` extension.
    - chunk_id : int, optional
        The chunk ID to be appended to the file name (default is None).
    - sep : str, optional
        The delimiter to be used in the CSV file (default is ',').
    - index : bool, optional
        Whether to include the index in the exported CSV file (default is False).
    - append : bool, optional
        Whether to append the DataFrame to an existing file (default is False).

    Raises Exception
    ---------------------
        If the DataFrame cannot be exported due to parsing errors or other issues.
    """
    try:  # Attempt to export the DataFrame to a CSV file.
        folder_path = Path(folder_name)  # Construct the full path to the folder.
        folder_path.mkdir(
            parents=True, exist_ok=True
        )  # Create the folder if it does not exist.

        if not str(file_name).endswith(
            ".csv"
        ):  # If the file name does not end with ".csv".
            file_name = (
                f"{file_name}.csv"  # Append the ".csv" extension to the file name.
            )

        full_path = folder_path / file_name  # Construct the full path to the CSV file.

        write_mode = "w"  # Default to write mode (overwrite existing files).
        write_header = True  # Default to writing headers.

        if (
            append and full_path.exists()
        ):  # If the file exists and append mode is enabled.
            write_mode = "a"  # Append mode
            write_header = False  # Avoid writing headers again in append mode

        if (
            "Timestamp" in dataframe.columns
        ):  # If a Timestamp column is present in the DataFrame.
            dataframe.sort_values(
                by="Timestamp", inplace=True
            )  # Sort the DataFrame by the Timestamp column.
        else:  # If the Timestamp column is not present.
            logger.warning(
                f"\nTimestamp column not found; skipping sort."
            )  # Log a warning message.

        dataframe.to_csv(
            str(full_path),
            sep=sep,
            index=index,
            mode=write_mode,
            header=write_header,
        )  # Export the DataFrame to a CSV file.

        logger.info(
            f"\nDataFrame exported successfully to {full_path} ."
        )  # Log a success message indicating the DataFrame was exported successfully.

        return full_path  # Return the full path of the exported CSV file.

    except Exception as e:  # If an error occurs during the export operation.
        logger.error(
            f"CSV Export: Failed to write file in folder {folder_name} | {str(e)}"
        )  # Log an error message if the export operation fails.
        raise  # Raise an exception to halt execution and propagate the error.
