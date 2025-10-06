"""
This module provides the following utilities:
    - Import and Dependency Management: Importing necessary libraries and modules.
    - FLAML Model Runner: Fits a FLAML model, saves artefacts, and appends results to a CSV.
    - Load and Predict: Loads a FLAML model and makes predictions.
"""

# ========== Imports and Dependencies ===========
from pathlib import Path  # For file path manipulations.
import pandas as pd  # For data manipulation and analysis.
import joblib  # For saving and loading models.
from joblib import parallel_backend  # to route FLAML jobs through Dask
from flaml import AutoML  # For automated machine learning.

import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Contains global variables and system functions.
from modules.system_guard import logger  # Logger for logging messages and errors.
import modules.utils as utils  # File utilities for loading/exporting CSV files.


# ========== FLAML Model Runner ==========
@system.timer_decorator
def flaml_model_runner(
    features: pd.DataFrame,
    target: pd.Series,
    file_name: str,
    budget: int | None = None,
    task: str | None = None,
    return_model: bool = False,
):
    """
    Function to fit a FLAML model, save artefacts, and append results to a CSV.

    Features
    ---------------------------------
        - Clean and preprocess data.
        - Fit a FLAML model to the data.
        - Save the model and log file.
        - Export results to a CSV file.
        - Return predictions.

    Parameters
    ---------------------------------
        - features (pd.DataFrame): The feature set for training the model.
        - target (pd.Series): The target variable for training the model.
        - file_name (str): The base name for saving the model and log files.
        - budget (int, optional): Time budget in seconds for FLAML (default from config).
        - task (str, optional): The type of task ('classification' or 'regression'). If None, it will be inferred from the target data type.
        - return_model (bool): If True, return the trained model instead of predictions.

    Returns
    ---------------------------------
        - tuple[list, float]: (predictions, best_score)

    Raises Exception
    ---------------------------------
        - Any exception raised during the model fitting or prediction process.
    """
    try:  # Check if the features and target are not empty
        budget = (
            budget or config.AUTOML_BUDGET_DEFAULT
        )  # Set budget to default if not provided

        automl = AutoML()  # Initialize FLAML's AutoML class
        logger.info(
            f"FLAML search | task={task}  budget={budget}s"
        )  # Log task and budget

        with parallel_backend("dask"):  # Use Dask for parallel processing
            settings = {
                "estimator_list": [
                    "rf",
                    "xgboost",
                    "extra_tree",
                    "xgb_limitdepth",
                    "sgd",
                ],
            }  # Settings for FLAML model
            automl.fit(
                X_train=features,
                y_train=target,
                task=task,
                time_budget=budget,
                **settings,
            )  # Fit the FLAML model

        best_score = automl.score(features, target)  # Evaluate the model
        results_dir = Path(config.EXPORTED_MODELS)  # Directory to save the model
        results_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create directory if it doesn't exist

        model_path = results_dir / f"{file_name}_flaml.pkl"  # Path to save the model
        joblib.dump(automl, model_path)  # Save the model
        logger.info(f"Saved FLAML model to {model_path}")  # Log the model save

        flaml_py = (
            results_dir / f"{file_name}_flaml.py"
        )  # Path to save the model summary
        with open(flaml_py, "w") as f:  # Open the file to write the model summary
            f.write("# Fallback FLAML model summary\n")  # Write header
            f.write(
                f"# Model type: {type(automl.model).__name__}\n"
            )  # Write model type
            params = automl.model.get_params()  # Get model parameters
            f.write(f"# Parameters: {params}\n")  # Write model parameters
        logger.info(f"Exporting TPOT pipeline to {flaml_py}")  # Log the model summary

        row = {
            "prefix": "FLAML",
            "pipeline_file": flaml_py.name,
            "score": best_score,
        }  # Create a row with model details

        utils.export_csv(
            pd.DataFrame([row]),
            folder_name=config.EXPORTED_MODELS,
            file_name=config.MODEL_RESULTS,
            append=True,
        )  # Export the results to a CSV file

        predictions = automl.predict(features).tolist()  # Make predictions

        if return_model:  # If return_model is True, return the model and score
            return automl.model, best_score  # Return model and score
        return predictions, best_score  # Return predictions *and* score

    except Exception as e:  # Log any exceptions that occur
        logger.error(f"FLAML runner failed: {e}")  # Log the error
        raise  # Raise the exception for further handling


# --------- Load and Predict -----------
@system.timer_decorator
def load_and_predict(features: pd.DataFrame, model_name: str) -> list:
    """
    Function to load a FLAML model and make predictions.

    Features
    ---------------------------------
        - Load a FLAML model from a file.
        - Make predictions using the loaded model.
        - Return predictions as a list.

    Parameters
    ---------------------------------
        - features (pd.DataFrame): The feature set for making predictions.
        - model_name (str): The name of the model file to load (without extension).

    Returns
    ---------------------------------
        - list: Predictions made by the loaded FLAML model.

    Raises Exception
    ---------------------------------
        - Any exception raised during the model loading or prediction process.
    """
    try:  # Check if the model file exists
        mdl = joblib.load(
            config.EXPORTED_MODELS / f"{model_name}_flaml.pkl"
        )  # Load the model
        logger.info(
            f"Successfully loaded and predicted with model: {mdl}"
        )  # Log success message
        return mdl.predict(features).tolist()  # Predict using the loaded model

    except Exception as e:  # Log any exceptions that occur
        logger.error(f"FLAML load/predict failed: {e}")  # Log the error
        raise  # Raise the exception for further handling
