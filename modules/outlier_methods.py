"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Clean Column: Cleans a column by filling NaN values with the mean of the column.
    - Histogram-Based Method: Detects outliers using the Histogram-Based Outlier Score (HBOS) method.
    - k-Nearest Neighbors Method: Detects outliers using the k-Nearest Neighbors (kNN) algorithm.
    - Isolation Forest Method: Detects outliers using the Isolation Forest algorithm.
    - Lightweight On-Line Detector Method: Detects outliers using the Lightweight On-Line Detector of Anomalies (LODA) method.
    - One-Class SVM Method: Detects outliers using the One-Class Support Vector Machine (OC-SVM) method.
    - Z-Score Method: Detects outliers using the Z-score method.
    - IQR Method: Detects outliers using the Interquartile Range (IQR) method.
    - MAD Method: Detects outliers using the Median Absolute Deviation (MAD) method.
"""

# =========== Imports and Dependencies ===========
import numpy as np  # Provides support for array operations and numerical transformations.
from pyod.models.hbos import (
    HBOS,
)  # Histogram-Based Outlier Score for efficient outlier detection.
from pyod.models.knn import (
    KNN,
)  # k-Nearest Neighbors for anomaly detection based on distance.
from pyod.models.iforest import (
    IForest,
)  # Isolation Forest for detecting global anomalies.
from pyod.models.loda import (
    LODA,
)  # Lightweight On-Line Detector of Anomalies, optimized for streaming data.
from pyod.models.ocsvm import (
    OCSVM,
)  # One-Class Support Vector Machine for novelty detection.

import modules.config as config  # Configuration file for global variables and constants.
from modules.system_guard import logger  # Logger for logging messages and errors.
import modules.system_guard as system  # Contains global variables and system functions


# ===========  Clean Column  ===========
def clean_column(X, col):
    """
    Function which cleans a column by filling NaN values with the mean of the column.

    Parameters
    ---------------------------------
        - X : np.ndarray
            The column data as a NumPy array.
        - col : str
            The name of the column being cleaned.

    Returns
    ---------------------------------
        - X : np.ndarray
            The cleaned column data with NaN values filled with the mean.
    """
    if np.isnan(X).any():  # Check for NaN values in the column
        logger.warning(
            f"\nNaNs detected in '{col}', filling with mean."
        )  # Log a warning message
        return np.nan_to_num(
            X, nan=np.nanmean(X)
        )  # Fill NaN values with the column mean
    return X  # Return the cleaned column


# =========== Histogram-Based Method ===========
@system.timer_decorator
def histogram_based_method(data, column):
    """
    Function which detects outliers in a specified column using the Histogram-Based Outlier Score (HBOS) method.

    Features
    ---------------------------------
        - The HBOS method creates histograms to estimate the probability of data points.
        - It assigns an anomaly score based on the log of the inverse of the probability.
        - Outliers have lower probabilities and higher anomaly scores.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the HBOS method.

    Returns
    ---------------------------------
        - _apply_hbos : delayed
            A delayed object representing the HBOS method applied to the specified column.

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the HBOS method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        clf = HBOS(contamination=config.CONTAMINATION_RATE)  # Create an HBOS model
        data["outlier"] = clf.fit_predict(X)  # Fit the model and predict outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nHBOS method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== k-Nearest Neighbors Method ===========
@system.timer_decorator
def k_nearest_neighbors_method(data, column):
    """
    Function which detects outliers in a specified column using the k-Nearest Neighbors (kNN) algorithm.

    Features
    ---------------------------------
        - The kNN method identifies outliers based on their distance to the k-nearest neighbors.
        - It calculates the distance between each point and its neighbors to detect anomalies.
        - Outliers have larger distances to their neighbors than normal data points.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the
            k-Nearest Neighbors (kNN) method.

    Returns
    ---------------------------------
        - _apply_knn : delayed
            A delayed object representing the kNN method applied to the specified column

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the kNN method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        clf = KNN(
            contamination=config.CONTAMINATION_RATE,
            n_neighbors=20,
            method="largest",
        )  # Create a kNN model
        data["outlier"] = clf.fit_predict(X)  # Fit the model and predict outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nkNN method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== Isolation Forest Method ===========
@system.timer_decorator
def isolation_forest_method(data, column):
    """
    Function which detects outliers in a specified column using the Isolation Forest algorithm.

    Features
    ---------------------------------
        - The Isolation Forest method isolates anomalies through recursive partitioning.
        - Outliers require fewer splits to be isolated, making them distinct from normal data.
        - The contamination rate controls the proportion of expected outliers.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the Isolation Forest method.

    Returns
    ---------------------------------
        - _apply_isolation_forest : delayed
            A delayed object representing the Isolation Forest method applied to the specified column.

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the Isolation Forest method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        clf = IForest(
            contamination=config.CONTAMINATION_RATE, random_state=42
        )  # Create an Isolation Forest model
        data["outlier"] = clf.fit_predict(X)  # Fit the model and predict outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nIsolation Forest method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== Lightweight On-Line Detector Method ===========
@system.timer_decorator
def lightweight_on_line_detector_method(data, column):
    """
    Function which detects outliers in a specified column using the Lightweight On-Line Detector of Anomalies (LODA) method.

    Features
    ---------------------------------
        - The LODA method is designed for anomaly detection in streaming data.
        - It combines multiple one-dimensional projections to identify outliers.
        - The contamination rate determines the expected proportion of outliers in the data.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the LODA method.

    Returns
    ---------------------------------
        - _apply_loda : delayed
            A delayed object representing the LODA method applied to the specified column

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the LODA method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        clf = LODA(contamination=config.CONTAMINATION_RATE)  # Create a LODA model
        data["outlier"] = clf.fit_predict(X)  # Fit the model and predict outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nLODA method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== One-Class SVM Method ===========
@system.timer_decorator
def one_class_svm_method(data, column):
    """
    Function which detects outliers in a specified column using the One-Class Support Vector Machine (OC-SVM) method.

    Features
    ---------------------------------
        - The OC-SVM method learns a boundary to separate normal data from anomalies.
        - It assumes normal data forms a compact cluster and anomalies deviate from it.
        - The contamination rate determines the fraction of expected outliers.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the OC-SVM method.

    Returns
    ---------------------------------
        - _apply_oc_svm : delayed
            A delayed object representing the OC-SVM method applied to the specified column

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the OC-SVM method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        clf = OCSVM(contamination=config.CONTAMINATION_RATE)  # Create an OC-SVM model
        data["outlier"] = clf.fit_predict(X)  # Fit the model and predict outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nOC-SVM method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== Z-Score Method ===========
@system.timer_decorator
def z_score_method(data, column):
    """
    Function which detects outliers in a specified column using the Z-score method.

    Parameters
    ---------------------------------
        - data : pd.DataFrame
            The DataFrame containing the column to be analyzed for outliers.
        - column : str
            The name of the column on which to apply the Z-score method.

    Returns
    ---------------------------------
        - _apply_z_score : delayed
            A delayed object representing the Z-score method applied to the specified column

    Raises Exception
    ---------------------------------
        If an error occurs during processing
    """

    try:  # Try-except block for catching exceptions and handling them gracefully
        # Apply the Z-score method to the specified column in the DataFrame.
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        mean_val = np.mean(X)  # Calculate the mean of the column
        std_val = (
            np.std(X) if np.std(X) != 0 else 1e-9
        )  # Calculate the standard deviation of the column
        data["outlier"] = (np.abs((X - mean_val) / std_val) > config.THRESHOLD).astype(
            int
        )  # Calculate Z-scores and identify outliers
        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the outliers
    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nZ-score method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception


# =========== IQR Method ===========
@system.timer_decorator
def iqr_method(data, column):
    """
    Function which detects outliers in a specified column using the Interquartile Range (IQR) method.

    Parameters
    ----------
    - data : pd.DataFrame
        The DataFrame containing the column to be analyzed.
    - column : str
        The name of the column on which to apply the IQR method.

    Returns
    -------
    pd.DataFrame
        DataFrame of detected outliers (rows where value is outside IQR limits).

    Raises Εxception
    ---------------------------------
        If an error occurs during processing.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        q1 = np.percentile(X, 25)  # Calculate the first quartile (25th percentile)
        q3 = np.percentile(X, 75)  # Calculate the third quartile (75th percentile)
        iqr = q3 - q1  # Calculate the interquartile range (IQR)
        lower_bound = (
            q1 - config.THRESHOLD * iqr
        )  # Calculate the lower bound for outliers
        upper_bound = (
            q3 + config.THRESHOLD * iqr
        )  # Calculate the upper bound for outliers
        data["outlier"] = ((X < lower_bound) | (X > upper_bound)).astype(
            int
        )  # Identify outliers based on the bounds

        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the DataFrame without outliers

    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nIQR method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception to halt execution and propagate the error


# =========== MAD Method ===========
@system.timer_decorator
def mad_method(data, column):
    """
    Function which detects outliers in a specified column using the Median Absolute Deviation (MAD) method.

    Parameters
    ----------
    - data : pd.DataFrame
        The DataFrame containing the column to be analyzed.
    - column : str
        The name of the column on which to apply the MAD method.

    Returns
    -------
    pd.DataFrame
        DataFrame of detected outliers (rows where value is outside MAD limits).

    Raises Exception
    ---------------------------------
        If an error occurs during processing.
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        X = (
            data[[column]].to_numpy().astype(float)
        )  # Convert the column to a NumPy array
        X = clean_column(X, column)  # Clean the column by filling NaN values
        median = np.median(X)  # Calculate the median of the column
        mad = np.median(
            np.abs(X - median)
        )  # Calculate the Median Absolute Deviation (MAD)
        if mad == 0:  # Check if MAD is zero to prevent division by zero
            mad = 1e-9  # Prevent division by zero
        modified_z_scores = (
            0.6745 * (X - median) / mad
        )  # Calculate modified Z-scores using MAD
        data["outlier"] = (np.abs(modified_z_scores) > config.THRESHOLD).astype(
            int
        )  # Identify outliers based on the threshold

        return data[data["outlier"] != 1].drop(
            columns=["outlier"]
        )  # Return the DataFrame without outliers

    except Exception as e:  # Catch any exceptions and store them in variable 'e'
        logger.error(
            f"\nMAD method failed on column '{column}': {e}"
        )  # Log an error message
        raise  # Raise an exception to halt execution and propagate the error
