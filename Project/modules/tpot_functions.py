"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Tpot Type: Function to select the appropriate TPOT model based on the target variable type.
    - Load and Predict: Function to load a saved TPOT model and predict values using provided features.
    - Tpot Model Runner: Function to train a TPOT model and export the best pipeline to a Python file.
"""

# =========== Imports and Dependencies ===========
from pathlib import Path  # Import Path for handling file paths.
import pandas as pd  # Import Pandas for handling DataFrame operations and data manipulation.
import joblib  # Import joblib for saving and loading models.
from tpot import (
    TPOTClassifier,
    TPOTRegressor,
)  # Import TPOTClassifier and TPOTRegressor for automated machine learning.
from joblib import parallel_backend  # Import parallel_backend for parallel processing.

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Import custom logging utilities, including @timer_decorator.
from modules.system_guard import (
    logger,
)  # Import logger for logging messages and errors.
import modules.utils as utils  # File handling utilities for loading and saving data.


# ========== Tpot Type ==========
@system.timer_decorator
def tpot_type(target_column, file_name=None):
    """
    Function which selects the appropriate TPOT model based on the target variable type.

    Features
    ---------------------------------
        - Determines whether to use a classifier or regressor based on the target variable.
        - Logs the selected model type (classifier or regressor) for reference.

    Parameters
    ---------------------------------
        - target_column : pd.Series
            The target variable (numeric for regression, categorical for classification).
        - file_name : str, optional
            A user-provided name for the exported pipeline file and logging references.

    Returns
    ---------------------------------
        - tpot_model : TPOTClassifier or TPOTRegressor
            The selected TPOT model based on the target variable type.

    Raises Exception
    ---------------------------------
        - If the target variable is empty or invalid.
        - If the selection of the TPOT model fails.
    """
    try:  # Try-except block to catch and handle exceptions
        if (
            target_column is None or len(target_column) == 0
        ):  # Check if the target variable is empty
            logger.error(
                "\nTarget Validation: Empty or invalid target variable"
            )  # Log an error
            raise  # Raise an exception

        if pd.api.types.is_numeric_dtype(
            target_column
        ):  # Check if the target variable is numeric
            unique_vals = (
                target_column.dropna().unique()
            )  # Get unique values in the target variable
            is_all_intlike = (
                pd.Series(unique_vals).apply(lambda x: float(x).is_integer()).all()
            )  # Check if all unique values are integer

            if (
                len(unique_vals) <= 10 and is_all_intlike
            ):  # Check if the target variable is categorical
                if set(unique_vals).issubset(
                    {0.0, 1.0}
                ):  # Check if the target variable is binary
                    target_column = target_column.astype(
                        int
                    )  # Ensure binary labels are integers
                tpot_model = TPOTClassifier()
                logger.info(
                    "\nUsing Classifier methods (auto‑detected)"
                )  # Log the model type
            else:  # If the target variable is numeric and not categorical
                tpot_model = TPOTRegressor()
                logger.info(
                    "\nUsing Regressor methods (auto‑detected)"
                )  # Log the model type
        else:  # If the target variable is not numeric
            tpot_model = TPOTClassifier()
            logger.info(
                "\nUsing Classifier methods (non-numeric target)"
            )  # Log the model type

        return tpot_model  # Return the selected TPOT model

    except Exception as e:  # Catch and handle exceptions
        logger.error(f"\nTPOT model selection failed: {str(e)}")  # Log an error message
        raise  # Raise the exception to propagate the error


# ========= Load and Predict ==========
@system.timer_decorator
def load_and_predict(features: pd.DataFrame, model_name: str) -> list:
    """
    Loads a saved TPOT model and predicts values using the provided features.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing the features to predict on.
    model_name : str
        Name of the saved TPOT model (without .pkl extension).

    Returns
    -------
    list
        List of predicted values from the loaded model.

    Raises
    ------
    Exception
        If loading or prediction fails.
    """
    try:  # Try-except block to catch and handle exceptions
        mdl = joblib.load(
            config.EXPORTED_MODELS / f"{model_name}_tpot.pkl"
        )  # Load the model
        logger.info(
            f"Successfully loaded and predicted with model: {mdl}"
        )  # Log success message
        return mdl.predict(features).tolist()  # Predict using the loaded model

    except Exception as e:  # Catch and handle exceptions
        logger.error(
            f"Failed to load and predict with TPOT model: {e}"
        )  # Log an error message
        raise  # Raise the exception to propagate the error


# ========== Tpot Model Runner ==========
@system.timer_decorator
def tpot_model_runner(
    features_columns,
    target_column,
    file_name,
    budget: int | None = None,
    generations: int | None = None,
    population_size: int | None = None,
    return_pipeline: bool = False,
):
    """
    Function which trains a TPOT model and exports the best pipeline to a Python file.

    Features
    ---------------------------------
        - Fits a TPOT model to the provided features and target columns.
        - Exports the best pipeline to a Python file for future use.
        - Logs the model score and export path for reference.

    Parameters
    ---------------------------------
        - features_columns : pd.DataFrame
            The input features for training the TPOT model.
        - target_column : pd.Series
            The target variable for training the TPOT model.
        - file_name : str
            The name of the dataset file for reference in the exported pipeline.
        - budget : int, optional
            Wall‑clock time in **seconds** for TPOT to search (mirrors FLAML).

    Returns
    ---------------------------------
        -tuple[np.ndarray, float] (predictions, model_score)
            The predicted values from the TPOT model and the model score.

    Raises Exception
    ---------------------------------
        - If the TPOT model creation fails.
    """
    try:  # Try-except block to catch and handle exceptions
        tpot = tpot_type(target_column, file_name)  # Select the appropriate TPOT model

        if budget is None:  # If budget is not provided, use the default budget
            budget = (
                config.AUTOML_BUDGET_DEFAULT
            )  # Set budget to default if not provided

        max_time_mins = max(1, round(budget / 60, 2))  # Convert budget to minutes

        if isinstance(tpot, TPOTRegressor):  # Check if the model is a regressor
            scoring = "r2"  # Use R-squared as the scoring metric
            y_fit = target_column.astype(
                float
            )  # Ensure target column is float for regression
        else:  # If the model is a classifier
            scoring = "accuracy"  # Use accuracy as the scoring metric
            y_fit = target_column.round().astype(
                int
            )  # Ensure target column is rounded and int for classification

        common_params = {
            # "generations": generations or 1000,
            # "population_size": population_size or 1000,
            "generations": generations or 100,  # Number of generations for TPOT
            "population_size": population_size or 100,  # Population size for TPOT
            "cv": 5,
            "verbosity": 4,
            "random_state": 420,
            "max_time_mins": max_time_mins,
            "scoring": scoring,
        }  # Define common parameters for TPOT models

        tpot.set_params(**common_params)  # Set the common parameters for TPOT models

        logger.info("\nFitting TPOT model")  # Log a message

        with parallel_backend("dask"):  # Use Dask for parallel processing
            tpot.fit(features_columns, y_fit)  # Fit the TPOT model to the data

        model_score = tpot.fitted_pipeline_.score(
            features_columns, y_fit
        )  # Evaluate the model

        results_dir = Path(config.EXPORTED_MODELS)  # Directory to save the model
        export_path = results_dir / f"{file_name}_tpot.py"  # Define the export path
        logger.info(f"Exporting TPOT pipeline to {export_path}")  # Log the export path

        pipeline_data = {
            "prefix": "TPOT",
            "pipeline_file": export_path.name,
            "score": f"{model_score:.4f}",
        }  # Create a dictionary with model details

        utils.export_csv(
            pd.DataFrame([pipeline_data]),
            folder_name=config.EXPORTED_MODELS,
            file_name=config.MODEL_RESULTS,
            append=True,
        )  # Export the pipeline data to a CSV file

        tpot.export(str(export_path))  # Export the best pipeline to a Python file
        tpot_model = tpot.predict(
            features_columns
        )  # Get the predicted values from the TPOT model

        model_path = results_dir / f"{file_name}_tpot.pkl"  # Define the model path
        joblib.dump(
            tpot.fitted_pipeline_, model_path
        )  # Save the fitted pipeline as a pickle file
        logger.info(f"Saved TPOT model to {model_path}")  # Log the model path

        if (
            return_pipeline
        ):  # If return_pipeline is True, return the fitted pipeline and model score
            return (
                tpot.fitted_pipeline_,
                model_score,
            )  # Return the fitted pipeline and score
        return tpot_model, model_score  # Return predictions and score

    except Exception as e:  # Catch and handle exceptions
        logger.error(f"\nTPOT model creation failed: {str(e)}")  # Log an error message
        raise  # Raise the exception to propagate the error
