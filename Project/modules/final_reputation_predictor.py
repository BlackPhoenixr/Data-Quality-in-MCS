"""
This module contains the function to predict final reputations using a TPOT model.
    - Import and Dependency Management: Importing necessary libraries and modules.
    - Predict Final Reputations on Raw Data: Train TPOT and FLAML models on the full original dataset to predict a single final reputation per user.
    - Predict Final Reputations on Processed Data: Train TPOT and FLAML models on a processed dataset to predict a single final reputation per user.
    - Batch Prediction: Function to predict the final reputation in batches.
"""

# =========== Imports and Dependencies ===========
import gc  # Garbage collection to free up memory
import pandas as pd  # Data manipulation and analysis library

import modules.config as config  # Configuration module
import modules.utils as utils  # File utilities for loading/exporting CSV files
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import logger  # Logger for logging messages and errors
import modules.tpot_functions as tpot_f  # TPOT model functions
import modules.flaml_functions as flaml_f  # FLAML model functions
import modules.data_processor as processor  # Maps column types, validates target column, etc.


# =========== Predict Final Reputations on Raw Data ===========
@system.timer_decorator
def predict_final_reputations_on_raw_data(original: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a TPOT model on the full original dataset to predict a single final reputation per user.

    Parameters
    ----------
    original : pd.DataFrame
        The full dataset including features and the 'User_Reputation' column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'User_Id' and the predicted 'Predicted_Reputation'.
    """
    try:  # Check if the original DataFrame is empty
        features = config.FEATURES  # List of features for prediction

        if not all(
            col in original.columns for col in features + ["User_Reputation"]
        ):  # Check if all required columns are present
            missing = list(
                set(features + ["User_Reputation"]) - set(original.columns)
            )  # Identify missing columns
            logger.error(f"Missing required columns: {missing}")  # Log the error
            raise  # Raise an exception

        user_ids = (
            original["User_Id"].astype(str).reset_index(drop=True)
        )  # Reset index for user IDs
        X = original[features].copy()  # Copy the features for processing
        X = processor.encode_and_scale_features(
            X, fit=True
        )  # Encode categorical columns and scale numerical features

        non_numeric_cols = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()  # Check for non-numeric columns
        if non_numeric_cols:  # If any non-numeric columns are found
            logger.error(
                f"Non-numeric columns detected in features before model training: {non_numeric_cols}"  # Log the error
            )
            raise  # Raise an exception

        y = original["User_Reputation"].reset_index(
            drop=True
        )  # Reset index for target variable

        X["User_Id"] = user_ids  # Add user IDs to the features DataFrame

        if len(X) > config.AUTOML_SAMPLE_SIZE:  # Downsample for AutoML if needed
            X_sampled = X.sample(
                n=config.AUTOML_SAMPLE_SIZE, random_state=42
            )  # Sample a subset of the data
            y_sampled = y.loc[X_sampled.index]  # Select the corresponding target values
        else:  # If the dataset is smaller than the sample size
            X_sampled = X  # Use the entire dataset
            y_sampled = y  # Use the entire target variable

        logger.info(
            "Training TPOT model on sampled data for pipeline search..."
        )  # Log the start of training
        gc.collect()  # Collect garbage to free up memory
        tpot_model, _ = tpot_f.tpot_model_runner(
            X_sampled.drop(columns=["User_Id"]),
            y_sampled,
            "final_raw",
            return_pipeline=True,
        )  # Train the TPOT model on the sampled data
        logger.info(
            "Fitting TPOT-discovered pipeline on sampled data..."
        )  # Log the fitting process
        tpot_model.fit(
            X_sampled.drop(columns=["User_Id"]), y_sampled
        )  # Fit the model on the sampled data
        predictions_tpot = batch_predict(
            tpot_model, X.drop(columns=["User_Id"])
        )  # Predict on the full dataset in batches

        logger.info(
            "Training FLAML model on sampled data for pipeline search..."
        )  # Log the start of training
        gc.collect()  # Collect garbage to free up memory
        flaml_model, _ = flaml_f.flaml_model_runner(
            X_sampled.drop(columns=["User_Id"]),
            y_sampled,
            "final_raw",
            task="regression",
            return_model=True,
        )  # Train the FLAML model on the sampled data
        # Insert logger statements to debug FLAML input
        logger.info(
            f"Fitting FLAML-discovered estimator on sampled data..."
        )  # Log the fitting process
        flaml_model.fit(
            X_sampled.drop(columns=["User_Id"]), y_sampled
        )  # Fit the model on the sampled data
        predictions_flaml = batch_predict(
            flaml_model, X.drop(columns=["User_Id"])
        )  # Predict on the full dataset in batches

        if len(predictions_tpot) != len(user_ids) or len(predictions_flaml) != len(
            user_ids
        ):  # Check if the lengths of predictions match user IDs
            logger.error("Prediction length mismatch with user IDs.")  # Log the error
            raise  # Raise an exception

        result_df = pd.DataFrame(
            {
                "User_Id": user_ids,
                "Predicted_Final_Reputation_TPOT": predictions_tpot,
                "Predicted_Final_Reputation_FLAML": predictions_flaml,
            }
        )  # Create a DataFrame with user IDs and predicted reputations

        result_df = (
            result_df.groupby("User_Id", as_index=False)
            .median()
            .sort_values("User_Id")
            .reset_index(drop=True)
        )  # Group by user ID and calculate median reputation

        tpot_df = result_df[["User_Id", "Predicted_Final_Reputation_TPOT"]].rename(
            columns={"Predicted_Final_Reputation_TPOT": "Predicted_Final_Reputation"}
        )  # Rename the column for consistency
        utils.export_csv(
            tpot_df,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name="predicted_reputations_raw_tpot",
        )  # Export the predictions to CSV files

        flaml_df = result_df[["User_Id", "Predicted_Final_Reputation_FLAML"]].rename(
            columns={"Predicted_Final_Reputation_FLAML": "Predicted_Final_Reputation"}
        )  # Rename the column for consistency
        utils.export_csv(
            flaml_df,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name="predicted_reputations_raw_flaml",
        )  # Export the predictions to CSV files

        logger.info(
            "Exported TPOT predictions to predicted_reputations_raw_tpot.csv "
            "and FLAML predictions to predicted_reputations_raw_flaml.csv"
        )  # Log the export paths

        return result_df  # Return the DataFrame with predicted reputations

    except Exception as e:  # Catch any exceptions that occur during the process
        logger.error(f"Failed to predict final reputations: {str(e)}")  # Log the error
        raise  # Raise the exception to be handled by the calling function


# =========== Predict Final Reputations on Processed Data ===========
@system.timer_decorator
def predict_final_reputations_on_processed_data(run_type) -> pd.DataFrame:
    """
    Trains TPOT and FLAML models on a processed dataset (with Dataset_Source)
    to predict a single final reputation per user. Uses Dataset_Source as a feature.

    Parameters
    ----------
    run_type : str
        The run type identifier (used to select the processed dataset and name output files).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'User_Id' and the predicted reputations from both models.
    """
    try:  # Check if the processed DataFrame is empty
        logger.info(
            f"Starting prediction on processed data for run_type={run_type}..."
        )  # Log the start of prediction

        processed = utils.load_csv(
            folder_name=config.SPLITTED_DATA,
            file_name=f"Feedback_Loop_{run_type}.csv",
            sep=",",
        )  # Load the dataset from the CSV file

        features = config.FEATURES  # List of features for prediction

        if (
            "Dataset_Source" not in processed.columns
        ):  # Check if 'Dataset_Source' column is present
            logger.error(
                f"Dataset_Source not found in columns for run_type={run_type}."
            )  # Log the error
            raise  # Raise an exception
        else:  # If 'Dataset_Source' column is found
            logger.info(
                f"Dataset_Source column found in processed data for run_type={run_type}."
            )  # Log the success

        processed = processed.copy()  # Create a copy of the processed DataFrame

        required_cols = features + [
            "User_Reputation",
            "Dataset_Source",
        ]  # List of required columns for prediction
        if not all(
            col in processed.columns for col in required_cols
        ):  # Check if all required columns are present
            missing = list(
                set(required_cols) - set(processed.columns)
            )  # Identify missing columns
            logger.error(f"Missing required columns: {missing}")  # Log the error
            raise  # Raise an exception

        user_ids = (
            processed["User_Id"].astype(str).reset_index(drop=True)
        )  # Reset index for user IDs
        X = processed[
            features + ["Dataset_Source"]
        ].copy()  # Copy the features for processing
        X = processor.encode_and_scale_features(
            X, fit=True
        )  # Encode categorical columns and scale numerical features

        non_numeric_cols = X.select_dtypes(
            exclude=["number"]
        ).columns.tolist()  # Check for non-numeric columns
        if non_numeric_cols:  # If any non-numeric columns are found
            logger.error(
                f"Non-numeric columns detected in features before model training: {non_numeric_cols}"
            )  # Log the error
            raise  # Raise an exception

        y = processed["User_Reputation"].reset_index(
            drop=True
        )  # Reset index for target variable
        X["User_Id"] = user_ids  # Add user IDs to the features DataFrame

        if len(X) > config.AUTOML_SAMPLE_SIZE:  # Downsample for AutoML if needed
            X_sampled = X.sample(
                n=config.AUTOML_SAMPLE_SIZE, random_state=42
            )  # Sample a subset of the data
            y_sampled = y.loc[X_sampled.index]  # Select the corresponding target values
        else:  # If the dataset is smaller than the sample size
            X_sampled = X  # Use the entire dataset
            y_sampled = y  # Use the entire target variable

        logger.info(
            f"Training TPOT model on sampled data for pipeline search (run_type={run_type})..."
        )  # Log the start of training
        gc.collect()  # Collect garbage to free up memory
        tpot_model, _ = tpot_f.tpot_model_runner(
            X_sampled.drop(columns=["User_Id"]),
            y_sampled,
            f"final_processed_{run_type}",
            return_pipeline=True,
        )  # Train the TPOT model on the sampled data
        logger.info(
            f"Fitting TPOT-discovered pipeline on sampled data (run_type={run_type})..."
        )  # Log the fitting process
        tpot_model.fit(
            X_sampled.drop(columns=["User_Id"]), y_sampled
        )  # Fit the model on the sampled data
        predictions_tpot = batch_predict(
            tpot_model, X.drop(columns=["User_Id"])
        )  # Predict on the full dataset in batches

        logger.info(
            f"Training FLAML model on sampled data for pipeline search (run_type={run_type})..."
        )  # Log the start of training
        gc.collect()  # Collect garbage to free up memory
        flaml_model, _ = flaml_f.flaml_model_runner(
            X_sampled.drop(columns=["User_Id"]),
            y_sampled,
            f"final_processed_{run_type}",
            task="regression",
            return_model=True,
        )  # Train the FLAML model on the sampled data
        logger.info(
            f"Fitting FLAML-discovered estimator on sampled data (run_type={run_type})..."
        )  # Log the fitting process
        flaml_model.fit(
            X_sampled.drop(columns=["User_Id"]), y_sampled
        )  # Fit the model on the sampled data
        predictions_flaml = batch_predict(
            flaml_model, X.drop(columns=["User_Id"])
        )  # Predict on the full dataset in batches

        if len(predictions_tpot) != len(user_ids) or len(predictions_flaml) != len(
            user_ids
        ):  # Check if the lengths of predictions match user IDs
            logger.error("Prediction length mismatch with user IDs.")  # Log the error
            raise  # Raise an exception

        result_df = pd.DataFrame(
            {
                "User_Id": user_ids,
                "Predicted_Final_Reputation_TPOT": predictions_tpot,
                "Predicted_Final_Reputation_FLAML": predictions_flaml,
            }
        )  # Create a DataFrame with user IDs and predicted reputations
        result_df = (
            result_df.groupby("User_Id", as_index=False)
            .median()
            .sort_values("User_Id")
            .reset_index(drop=True)
        )  # Group by user ID and calculate median reputation

        tpot_df = result_df[["User_Id", "Predicted_Final_Reputation_TPOT"]].rename(
            columns={"Predicted_Final_Reputation_TPOT": "Predicted_Final_Reputation"}
        )  # Rename the column for consistency
        utils.export_csv(
            tpot_df,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name=f"predicted_reputations_processed_tpot_{run_type}",
        )  # Export the predictions to CSV files

        flaml_df = result_df[["User_Id", "Predicted_Final_Reputation_FLAML"]].rename(
            columns={"Predicted_Final_Reputation_FLAML": "Predicted_Final_Reputation"}
        )  # Rename the column for consistency
        utils.export_csv(
            flaml_df,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name=f"predicted_reputations_processed_flaml_{run_type}",
        )  # Export the predictions to CSV files

        logger.info(
            f"Exported TPOT predictions to predicted_reputations_processed_tpot_{run_type}.csv "
            f"and FLAML predictions to predicted_reputations_processed_flaml_{run_type}.csv"
        )  # Log the export paths

        return result_df  # Return the DataFrame with predicted reputations

    except Exception as e:  # Catch any exceptions that occur during the process
        logger.error(
            f"Failed to predict final reputations on processed data for run_type={run_type}: {str(e)}"
        )  # Log the error
        raise  # Raise an exception to be handled by the calling function


# =========== Batch Prediction ===========
def batch_predict(model, X, batch_size=10):
    """
    Function which predicts the final reputation in batches.

    Parameters
    ----------
    model : object
        The trained model to use for predictions.
    X : pd.DataFrame
        The input features for prediction.
    batch_size : int, optional
        The size of each batch for prediction (default is 10).

    Returns
    -------
    list
        A list of predictions for each batch.

    Raises
    -------
    Exception
        If the model is invalid or if any error occurs during prediction.
    """
    try:  # Check if the model is valid
        preds = []  # Initialize an empty list to store predictions
        for i in range(0, len(X), batch_size):  # Iterate over the dataset in batches
            X_batch = X.iloc[i : i + batch_size]  # Select the current batch
            preds.extend(
                model.predict(X_batch)
            )  # Predict on the current batch and extend the preds list
        return preds  # Return the list of predictions

    except Exception as e:  # Catch any exceptions that occur during the process
        logger.error(f"Batch prediction failed: {str(e)}")  # Log the error
        raise  # Raise the exception to be handled by the calling function
