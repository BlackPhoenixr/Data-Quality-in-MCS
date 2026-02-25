"""
This module provides the following utilities:
    - Imports and Dependencies: Importing necessary libraries and modules.
    - Kafka Producer Configuration: Setting up the Kafka producer for streaming data.
    - Get Kafka Producer with Retry Logic: Function to create a Kafka producer with retry logic.
    - Feedback Task: Orchestrates the feedback loop to unify data and update user reputations.
"""

# =========== Imports and Dependencies ===========
import pandas as pd  # For DataFrame handling
from kafka import KafkaProducer  # For streaming reputation updates to a Kafka topic
import json  # For encoding data sent to Kafka
import time  # For throttling message rate to Kafka
import os  # For environment variable management
from kafka.errors import NoBrokersAvailable  # For handling Kafka connection errors


import modules.config as config  # Configuration settings for the project.
import modules.system_guard as system  # Import custom logging utilities, including @timer_decorator.
from modules.system_guard import (
    logger,
)  # Import logger for logging messages and errors.
import modules.utils as utils  # File handling utilities for loading and saving data.

# =========== Kafka Producer Configuration ===========
user_reputation_store = {}  # Tracks dynamically updated user reputations
original_user_reputation_store = {}  # Stores each user's baseline reputation


# KAFKA_BROKER = os.environ.get(
#     "KAFKA_BOOTSTRAP_SERVERS", "kafka:29092"
# )  # Kafka broker address, defaulting to "kafka:29092" if not set in environment variables.
KAFKA_BROKER = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
)  # Uncomment this line to use localhost for local testing
TOPIC = "user_reputation_updates"  # Kafka topic for streaming reputation updates


# ====== Get Kafka Producer with Retry Logic ======
@system.timer_decorator
def get_kafka_producer_with_retry(brokers, retries=12, delay=5):
    """
    Function which attempts to create a Kafka producer with retries.

    Parameters
    ---------------------------------
        - brokers : str
            The Kafka broker address to connect to.
        - retries : int
            The number of retry attempts to connect to Kafka.
        - delay : int
            The delay in seconds between retry attempts.

    Returns
    ---------------------------------
        - KafkaProducer
            A KafkaProducer instance connected to the specified brokers.

    Raises Exception
    ---------------------------------
        - RuntimeError
            If the Kafka producer cannot be created after the specified number of retries.
    """
    for attempt in range(retries):  # Loop through the number of retries
        try:  # Attempt to create a Kafka producer
            return KafkaProducer(
                bootstrap_servers=brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )  # Create a Kafka producer with JSON serialization for messages
        except (
            NoBrokersAvailable
        ):  # Catch the exception if Kafka brokers are not available
            logger.warning(
                f"Kafka not available yet, retrying in {delay}s (attempt {attempt+1}/{retries})..."
            )  # Log a warning message indicating the retry attempt
            time.sleep(delay)  # Wait for the specified delay before retrying
    raise RuntimeError(
        f"Failed to connect to Kafka at {brokers} after {retries} attempts."
    )  # Raise an error if all retries fail


producer = get_kafka_producer_with_retry(
    KAFKA_BROKER
)  # Create a Kafka producer with retry logic


# =========== Feedback Task ===========
@system.timer_decorator
def feedback_task(
    normal: pd.DataFrame,
    intentional: pd.DataFrame,
    unintentional: pd.DataFrame,
    run_type: str,
) -> dict:
    """
    Function which orchestrates the feedback loop to unify data and update user reputations.

    Features
    ---------------------------------
        - Annotates DataFrames by their source (normal, intentional, unintentional).
        - Concatenates all rows into a single DataFrame.
        - Converts timestamp column to datetime format and sorts the DataFrame.
        - Exports the combined DataFrame to CSV for further analysis.
        - Initializes baseline reputations for all users.
        - Processes rows and sends updates to a Kafka topic.
        - Returns the combined dataset and final reputation scores after processing.

    Parameters
    ---------------------------------
        - normal : pd.DataFrame
            The normal user behavior dataset.
        - intentional : pd.DataFrame
            The intentional misbehavior dataset.
        - unintentional : pd.DataFrame
            The unintentional misbehavior dataset.
        - run_type : str
            A string indicating the type or context of the current run, used for logging and file naming.

    Returns
    ---------------------------------
        - dict
            A dictionary containing the combined dataset and final reputation scores.

    Raises Exception
    ---------------------------------
        - If any step in the process fails
    """
    try:  # Try-except block for catching exceptions and handling them gracefully
        normal["Dataset_Source"] = "normal"  # Annotate the normal dataset
        intentional["Dataset_Source"] = (
            "intentional"  # Annotate the intentional dataset
        )
        unintentional["Dataset_Source"] = (
            "unintentional"  # Annotate the unintentional dataset
        )

        combined = pd.concat(
            [normal, intentional, unintentional], ignore_index=True
        )  # Concatenate all datasets

        if "Timestamp" in combined.columns:  # Check if the timestamp column exists
            combined["Timestamp"] = pd.to_datetime(
                combined["Timestamp"], errors="coerce"
            )  # Convert the timestamp column to datetime format

        combined = combined.sort_values(by="Timestamp").reset_index(
            drop=True
        )  # Sort the DataFrame by timestamp

        utils.export_csv(
            combined,
            folder_name=config.SPLITTED_DATA,
            file_name=f"Feedback_Loop_{run_type}.csv",
        )  # Export the combined DataFrame to CSV

        original_df = combined.drop_duplicates(subset=["User_Id"])[
            ["User_Id", "User_Reputation"]
        ].rename(
            columns={"User_Reputation": "Original_Reputation"}
        )  # Extract unique user IDs and original reputations

        for _, row in combined.iterrows():  # Iterate over each row
            user_id = row["User_Id"]  # Extract the user ID

            if (
                user_id not in user_reputation_store
            ):  # Check if user ID is not already in the store
                initial_rep = row.get(
                    "User_Reputation", 0.0
                )  # Get the initial reputation

                original_user_reputation_store[user_id] = (
                    initial_rep  # Store the original reputation
                )
                user_reputation_store[user_id] = (
                    initial_rep  # Initialize the user reputation store
                )

        for (
            _,
            row,
        ) in combined.iterrows():  # Iterate over each row in the combined DataFrame
            user_id = row["User_Id"]  # Extract the user ID

            dataset_source = row["Dataset_Source"]  # Extract the dataset source

            if dataset_source == "normal":  # Check the dataset source
                user_reputation_store[user_id] = min(
                    1.0, user_reputation_store[user_id] + config.NORMAL_INCREMENT
                )  # Update the user reputation based on the dataset source

            elif dataset_source == "unintentional":  # Check the dataset source
                user_reputation_store[user_id] = max(
                    0.0, user_reputation_store[user_id] - config.UNINTENTIONAL_PENALTY
                )  # Update the user reputation based on the dataset source

            elif dataset_source == "intentional":  # Check the dataset source
                user_reputation_store[user_id] = max(
                    0.0, user_reputation_store[user_id] - config.INTENTIONAL_PENALTY
                )  # Update the user reputation based on the dataset source

            kafka_message = {
                "user_id": user_id,  # Unique identifier for the user.
                "new_reputation": user_reputation_store[
                    user_id
                ],  # Updated reputation score.
                "dataset_source": dataset_source,  # The type of event that triggered the update.
                "timestamp": (
                    row[
                        "Timestamp"
                    ].isoformat()  # Convert timestamp to ISO format if it's valid.
                    if pd.notnull(row.get("Timestamp", None))
                    else None  # Set to None if timestamp is missing.
                ),
            }  # Create a message to send to Kafka

            time.sleep(0.000001)  # Throttle the message rate to Kafka

            producer.send(
                TOPIC, value=kafka_message
            )  # Send the message to the Kafka topic

            logger.info(
                f"Sent to Kafka ({run_type}): {kafka_message}"
            )  # Log the message sent to Kafka

        producer.flush()  # Flush the producer to ensure all messages are sent

        final_reputations_df = (
            pd.DataFrame(
                {
                    "User_Id": list(user_reputation_store.keys()),
                    "Final_Reputation": list(user_reputation_store.values()),
                }
            )
            .sort_values("User_Id")
            .reset_index(drop=True)
        )  # Create a DataFrame for final reputations

        utils.export_csv(
            final_reputations_df,
            folder_name=config.EXPORTED_REPUTATIONS,
            file_name=f"feedback_loop_reputations_{run_type}.csv",
        )  # Export the final reputations to CSV

        logger.info(
            f"Exported final reputations to: {config.EXPORTED_REPUTATIONS}/feedback_loop_reputations_{run_type}.csv"
        )  # Log the export of final reputations

        return {
            "combined_data": combined,
            "user_reputation_store": user_reputation_store,
        }  # Return the combined dataset and final reputation scores

    except Exception as e:  # Catch any exceptions that occur
        logger.error(f"Failed in feedback_task ({run_type}): {str(e)}")  # Log the error
        raise  # Raise an exception to halt execution and log the error
