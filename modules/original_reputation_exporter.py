"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Export Original Reputations: Exports the 'original' user reputations from a dataframe to CSV.
"""

# =========== Imports and Dependencies ===========
import pandas as pd  # For data manipulation and analysis.
import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Contains global variables and system functions.
from modules.system_guard import logger  # Logger for logging messages and errors.
import modules.utils as utils  # File utilities for loading/exporting CSV files.


# =========== Export Original Reputations ===========
@system.timer_decorator
def export_original_reputations(
    df: pd.DataFrame, user_id_col="User_Id", rep_col="User_Reputation"
):
    """
    Function which exports the 'original' user reputations from a dataframe to CSV.

    Parameters
    ---------------------------------
        - df (pd.DataFrame): The dataframe containing user reputations.
        - user_id_col (str): The column name for user IDs (default is 'User_Id').
        - rep_col (str): The column name for user reputations (default is 'User_Reputation').

    Returns
    ---------------------------------
        - pd.DataFrame: A DataFrame containing user IDs and their median reputations, sorted by user ID.

    Raises Exception
    ---------------------------------
        - Any exception raised during the export process, logged and re-raised.
    """
    try:  # Check if the dataframe is empty
        user_ids = (
            df[user_id_col].astype(str).str.replace("User_", "").astype(int)
        )  # Convert User_Id to integer by removing 'User_' prefix
        reputations = pd.DataFrame(
            {"User_Id": user_ids, "User_Reputation": df[rep_col]}
        )  # Create a new DataFrame with User_Id and User_Reputation columns

        reputations = reputations.dropna(
            subset=["User_Reputation"]
        )  # Drop rows where User_Reputation is NaN
        reputations = (
            reputations.groupby("User_Id", as_index=False)["User_Reputation"]
            .median()
            .sort_values("User_Id")
            .reset_index(drop=True)
        )  # Group by User_Id, calculate median User_Reputation, sort by User_Id, and reset index

        utils.export_csv(
            reputations,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name="original_reputations",
        )  # Export the reputations DataFrame to CSV in the specified folder

        return reputations  # Return the reputations DataFrame

    except Exception as e:  # Catch any exceptions that occur during the export process
        logger.error(
            f"Error exporting original reputations: {e}"
        )  # Log the error message
        raise  # Re-raise the exception for further handling
