"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Column Mapping: Maps the column types of a DataFrame.
    - Validate Target Feature: Validates the target feature and feature columns.
    - Encode and Scale Features: Encodes categorical columns and scales numeric features.
"""

# =========== Imports and Dependencies ===========
import pandas as pd  # For DataFrame creation/manipulation
import joblib  # For saving/loading the fitted encoder
from pathlib import Path  # For file path handling
from colorama import Fore  # For colorized output in logs
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)  # For encoding and scaling features

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import logger  # For logging and system-related functions


# =========== Column Mapping ===========
@system.timer_decorator
def column_mapping(data: pd.DataFrame, target_column_name: str) -> dict:
    """
    Function which maps the column types of a DataFrame.

    Features
    ---------------------------------
        - Maps the column types of a DataFrame.
        - Logs the verification of the target variable.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The dataset to be mapped.
        - target_column_name : str
            The target column name.

    Returns
    ---------------------------------
        - column_types : dict
            A dictionary containing the column names as keys and their types as values.

    Raises Exception
    ---------------------------------
        - If the dataset is empty or invalid.
        - If the target column is missing.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        if data is None or data.empty:  # Check if the dataset is empty or invalid
            logger.error(
                "\nData Validation: Empty or invalid dataset provided."
            )  # Log an error
            raise  # Raise an exception

        if (
            target_column_name not in data.columns
        ):  # Check if the target column is missing
            logger.error(
                f"\nColumn Access: Target variable '{target_column_name}' not found. "
                f"Available columns: {data.columns.tolist()}"
            )  # Log an error
            raise  # Raise an exception

        if not pd.api.types.is_numeric_dtype(
            data[target_column_name]
        ):  # Check if the target column is numeric
            logger.error(
                f"\nTarget column '{target_column_name}' must be numeric. Found: {data[target_column_name].dtype}"
            )  # Log an error
            raise  # Raise an exception

        if (
            data[target_column_name].isnull().any()
        ):  # Check if the target column contains NaN values
            logger.error(
                f"\nTarget column '{target_column_name}' contains NaN values."
            )  # Log an error
            raise  # Raise an exception

        target_data = data[target_column_name]  # Extract the target column

        logger.info(
            f"\nTarget variable {Fore.YELLOW}{target_column_name} "
            f"({target_data.dtype})verified {Fore.GREEN}  successfully {Fore.RESET}"
        )  # Log the successful verification of the target variable

        print(f"\n{'-'*70}")  # Print a separator line

        column_types = {}  # Initialize an empty dictionary to store column types

        # logger.debug(f"Columns in DataFrame: {data.columns.tolist()}") # Log the columns in the DataFrame

        for col in data.columns:  # Iterate over each column in the DataFrame
            if data[col].dtype in [
                "int64",
                "float64",
            ]:  # Check if the column is numerical
                column_types[col] = "numerical"  # Mark the column as numerical.
            elif data[col].dtype == "object":  # Check if the column is categorical
                column_types[col] = "categorical"  # Mark the column as categorical.
            else:  # If the column type is unknown
                column_types[col] = "unknown"  # Mark the column as unknown.

        # logger.debug(f"Detected column types: {column_types}")

        if not isinstance(column_types, dict):  # Ensure the result is a dictionary
            logger.error(
                "\nColumn mapping result is not a valid dictionary."
            )  # Log the error
            raise  # Raise an exception

        if not all(
            isinstance(k, str) and isinstance(v, str) for k, v in column_types.items()
        ):  # Check if keys and values are strings
            logger.error(
                "\nColumn mapping dictionary contains invalid key-value types."
            )  # Log the error
            raise  # Raise an exception

        return column_types  # Return the dictionary containing column types

    except Exception as e:  # Catch unexpected exceptions
        logger.error(
            f"\nData preparation failed (unexpected): {str(e)}"
        )  # Log the error
        raise  # Raise the exception


# =========== Validate Target Feature ===========
@system.timer_decorator
def validate_target_feature(
    data: pd.DataFrame, target_column_name: str, feature_columns_name: list
) -> tuple:
    """
    Function which validates the target feature and feature columns.

    Features
    ---------------------------------
        - Validates the target feature and feature columns.
        - Encodes categorical features using Label Encoding.
        - Logs errors and exceptions gracefully.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The dataset to be validated.
        - target_column_name : str
            The target column name.
        - feature_columns_name : list
            The list of feature column names.

    Returns
    ---------------------------------
        - features : pd.DataFrame
            The DataFrame containing the encoded feature columns.
        - target_data : pd.Series
            The Series containing the target column.

    Raises Exception
    ---------------------------------
        - If the input is not a DataFrame.
        - If the target column is missing.
        - If the specified feature columns are missing.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        if not isinstance(data, pd.DataFrame):  # Check if the input is a DataFrame
            logger.error(
                f"\nInput must be a pandas DataFrame, got {type(data)} instead."
            )  # Log the error about the invalid input type
            raise  # Raise an exception

        if (
            target_column_name not in data.columns
        ):  # Check if the target column is missing
            logger.error(
                f"\nTarget column '{target_column_name}' not found in the DataFrame."
            )  # Log the error about the missing target column
            raise  # Raise an exception

        target_data = data[target_column_name].copy()  # Extract the target column

        if not pd.api.types.is_numeric_dtype(target_data):  # Ensure target is numeric
            logger.error(
                f"\nTarget column '{target_column_name}' must be numeric."
            )  # Log the error
            raise  # Raise an exception

        if feature_columns_name:  # Check if feature columns are specified
            missing_cols = [
                col for col in feature_columns_name if col not in data.columns
            ]  # Find columns that are missing in the DataFrame

            if missing_cols:  # If any columns are missing
                logger.error(
                    f"\nSpecified feature columns not found in data: {missing_cols}"
                )  # Log the error about missing feature columns
                raise  # Raise an exception

            features = data[
                feature_columns_name
            ].copy()  # Extract the specified feature columns

        else:
            features = data.drop(
                columns=[target_column_name]
            ).copy()  # Extract all columns except the target column

        for col in features.columns:  # Iterate over each column in the DataFrame
            if features[col].dtype == "object":  # Check if the column is categorical
                le = LabelEncoder()  # Create a LabelEncoder instance
                features[col] = le.fit_transform(
                    features[col].astype(str)
                )  # Encode the categorical column using Label Encoding

        if features.isnull().any().any():  # Check for NaNs in feature columns
            logger.error("\nFeature columns contain missing values.")  # Log the error
            raise  # Raise an exception

        logger.info(
            f"\nEncoded categorical features: {[col for col in features.columns if data[col].dtype == 'object']}"
        )

        return (
            features,
            target_data,
        )  # Return the encoded feature columns and target column

    except Exception as e:  # Catch unexpected exceptions
        logger.error(
            f"\nPreprocessing: Failed (unexpected) | {str(e)}"
        )  # Log the error
        raise  # Raise the exception


# =========== Encode and Scale Features ===========
@system.timer_decorator
def encode_and_scale_features(
    data: pd.DataFrame,
    fit: bool = True,
    encoder_path: Path = config.ENCODER_MODEL_PATH,
    scaler_type: str = None,
    scaler_path: Path = None,
) -> pd.DataFrame:
    """
    Encodes categorical columns in the DataFrame using one-hot encoding and ensures consistency across training and prediction phases.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing features.
    fit : bool, default=True
        If True, fits a new encoder and saves it. If False, loads and uses the existing encoder.
    encoder_path : Path
        The path to save/load the fitted encoder.
    scaler_type : str, default=None
        Type of scaler to apply to numeric features. Options: 'standard', 'minmax', or None (no scaling).
    scaler_path : Path, default=None
        Path to save/load the fitted scaler. Defaults to encoder_path with filename 'scaler_model.joblib'.

    Returns
    -------
    pd.DataFrame
        A one-hot encoded and (optionally) scaled DataFrame with consistent column alignment.

    Raises
    ------
    Exception
        If encoding fails or encoder cannot be saved/loaded.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        if not isinstance(data, pd.DataFrame):  # Check if the input is a DataFrame
            logger.error(
                "Input to encode_categorical_columns must be a DataFrame."
            )  # Log the error
            raise  # Raise an exception

        df = (
            data.copy()
        )  # Create a copy of the input DataFrame to avoid modifying the original
        for geo_col in ["Latitude", "Longitude"]:  # Convert geo columns to numeric
            if geo_col in df.columns:  # Check if geo columns exist
                df[geo_col] = pd.to_numeric(
                    df[geo_col], errors="coerce"
                )  # Convert to numeric, coerce errors to NaN

        categorical_cols = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()  # Select categorical columns

        one_hot_max_cardinality = 15  # Define maximum cardinality for one-hot encoding
        one_hot_cols = [
            col
            for col in categorical_cols
            if df[col].nunique() <= one_hot_max_cardinality
        ]  # Select columns suitable for one-hot encoding
        label_encode_cols = [
            col
            for col in categorical_cols
            if df[col].nunique() > one_hot_max_cardinality
        ]  # Select columns suitable for label encoding

        if one_hot_cols:  # Check if there are columns for one-hot encoding
            ohe = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )  # Create OneHotEncoder instance
            if fit:  # If fitting a new encoder
                ohe_arr = ohe.fit_transform(
                    df[one_hot_cols]
                )  # Fit and transform the data
                joblib.dump(ohe, encoder_path)  # Save the fitted encoder
            else:  # If loading an existing encoder
                ohe = joblib.load(encoder_path)  # Load the existing encoder
                ohe_arr = ohe.transform(df[one_hot_cols])  # Transform the data
            ohe_df = pd.DataFrame(
                ohe_arr, columns=ohe.get_feature_names_out(one_hot_cols), index=df.index
            )  # Create a DataFrame from the one-hot encoded array
        else:  # If no columns for one-hot encoding
            ohe_df = pd.DataFrame(index=df.index)  # Create an empty DataFrame

        le_df = pd.DataFrame(
            index=df.index
        )  # Create an empty DataFrame for label encoding
        for col in label_encode_cols:  # Iterate over label encoding columns
            le = LabelEncoder()  # Create a LabelEncoder instance
            le_vals = le.fit_transform(
                df[col].astype(str)
            )  # Fit and transform the data
            le_df[col] = le_vals  # Add the encoded values to the DataFrame

        numeric_cols = df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()  # Select numeric columns
        num_df = df[numeric_cols]  # Create a DataFrame for numeric columns

        final_df = pd.concat(
            [num_df, ohe_df, le_df], axis=1
        )  # Concatenate all DataFrames
        final_df = final_df.fillna(
            final_df.median(numeric_only=True)
        )  # Fill NaN values with median for numeric columns

        if scaler_type is not None:  # If a scaler type is specified
            if scaler_type not in [
                "standard",
                "minmax",
            ]:  # Check if the scaler type is valid
                raise ValueError(
                    "scaler_type must be either 'standard', 'minmax', or None."
                )  # Raise an error if the scaler type is invalid
            if scaler_path is None:  # If no scaler path is provided
                scaler_path = encoder_path.with_name(
                    "scaler_model.joblib"
                )  # Set default scaler path

            if scaler_type == "standard":  # If using standard scaling
                scaler = StandardScaler()  # Create a StandardScaler instance
            elif scaler_type == "minmax":  # If using min-max scaling
                scaler = MinMaxScaler()  # Create a MinMaxScaler instance

            numeric_cols_for_scaling = final_df.select_dtypes(
                include=["number", "bool"]
            ).columns.tolist()  # Select numeric columns for scaling
            if fit:  # If fitting a new scaler
                final_df[numeric_cols_for_scaling] = scaler.fit_transform(
                    final_df[numeric_cols_for_scaling]
                )  # Fit and transform the numeric columns
                joblib.dump(scaler, scaler_path)  # Save the fitted scaler
            else:  # If loading an existing scaler
                scaler = joblib.load(scaler_path)
                final_df[numeric_cols_for_scaling] = scaler.transform(
                    final_df[numeric_cols_for_scaling]
                )  # Transform the numeric columns using the loaded scaler

        return final_df  # Return the final DataFrame

    except Exception as e:  # Catch unexpected exceptions
        logger.error(f"Hybrid categorical encoding failed: {e}")  # Log the error
        raise  # Raise the exception
