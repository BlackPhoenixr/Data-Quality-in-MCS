"""
This module provides the following utilities:
    - Directory and Path Management: Manages directories and paths for importing modules and creating output directories.
    - Imports and Dependencies: Imports necessary libraries and modules for data generation.
    - Generate Combined Data: Generates synthetic user activity data based on predefined parameters.
    - Apply Experience: Applies incremental experience updates to user submissions.
    - Add Intentional and Unintentional Outliers: Injects outliers into the dataset to simulate anomalous behavior.
    - Get Random Location in Radius: Generates a random geographic location within a specified radius.
    - Convert to DataFrame: Converts a list of tuples into a structured Pandas DataFrame.
    - Main: Orchestrates the creation of a synthetic user activity dataset.
    - Script Entry Point: The conditional block that executes the main function.
"""

# ======= Directory and Path Management =======
import sys, pathlib  # For path manipulations and system path management

ROOT_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
)  # Get the root directory of the project
if (
    str(ROOT_DIR) not in sys.path
):  # Check if the root directory is already in the system path
    sys.path.append(
        str(ROOT_DIR)
    )  # Add the root directory to the system path if not present

# =========== Imports and Dependencies ===========
from typing import Any  # Used for type hints and annotations.
import random  # Provides functions for generating random numbers and making random selections.
import time  # Used for tracking execution time and implementing delays.
import signal  # Enables handling of system signals (e.g., SIGTERM, SIGINT) for graceful shutdown.
from datetime import (
    datetime,  # Provides functionality to create, format, and manipulate timestamps.
    timedelta,  # Enables arithmetic operations on dates (e.g., adding/subtracting days, hours).
)
from geopy.distance import (
    distance as geopy_distance,  # Used to compute geographical distances between two locations.
)
from geopy.point import (
    Point,
)  # Represents a geographical point with latitude and longitude.
from colorama import (
    Fore,
)  # Provides ANSI escape sequences for colorizing terminal output.
import pandas as pd
from pandas import (
    DataFrame,
)  # Provides powerful DataFrame structures for organizing and analyzing dataset contents.

import modules.config as config  # Contains configuration settings for the script.
import modules.system_guard as system  # Contains global variables and system functions
from modules.system_guard import (
    logger,
)  # Logger for capturing and reporting errors and events.
import modules.utils as utils  # Provides utility functions for file operations.


# =========== Generate Combined Data ===========
@system.timer_decorator
def generate_combined_data(
    user_ids,
    start_date,
    end_date,
    submissions,
    activities,
    internet_types,
    university_areas,
    original_locations,
    reputation,
    experience,
    accident_values,
    accident_probabilities,
    road_types,
    road_probabilities,
    speed_limits,
    speed_probabilities,
    violation_types,
) -> list[Any]:
    """
    Function which generates synthetic user activity data based on predefined parameters.

    Features
    ---------------------------------
        - Simulates user submissions over a specified date range.
        - Assigns random user activities, locations, and device metrics.
        - Determines road conditions, accidents, and traffic violations.
        - Generates outliers in approximately 30% of the dataset.

    Parameters
    ---------------------------------
        - user_ids : list
            A list of unique user identifiers participating in the dataset.
        - start_date : datetime
            The starting date for generating user submissions.
        - end_date : datetime
            The ending date for generating user submissions.
        - submissions : int
            The number of submissions per user per day.
        - activities : list
            A list of possible user activities (e.g., Walking, Driving).
        - internet_types : list
            A list of internet connection types (e.g., WiFi, 4G).
        - university_areas : list
            A list of university areas where users can be located.
        - original_locations : dict
            A dictionary mapping university areas to their geographic coordinates.
        - reputation : dict
            A dictionary mapping user IDs to their reputation scores.
        - experience : dict
            A dictionary mapping user IDs to their experience levels.
        - accident_values : list
            A list of possible road accident severity levels.
        - accident_probabilities : list
            A list of probabilities corresponding to each accident severity level.
        - road_types : list
            A list of possible road types (e.g., 1way, 2way).
        - road_probabilities : list
            A list of probabilities corresponding to each road type.
        - speed_limits : list
            A list of possible speed limits.
        - speed_probabilities : list
            A list of probabilities corresponding to each speed limit.
        - violation_types : list
            A list of possible traffic violation types.

    Returns
    ---------------------------------
        - combined_data : list
            A list of generated user submissions, each represented as a tuple.

    Raises Exception
    ---------------------------------
        - If an unexpected error occurs during data generation.
    """

    try:  # Try-except block to catch and log any errors that occur during data generation

        combined_data = []  # Initialize an empty list to store generated data

        for user_id in user_ids:  # Iterate through the user IDs
            current_date = start_date  # Initialize the starting date for the user

            while current_date <= end_date:  # Iterate through the date range
                for idx in range(submissions):  # Generate multiple submissions per day
                    rand_hour = random.randint(
                        0, 23
                    )  # Generate a random timestamp within the day
                    rand_minute = random.randint(
                        0, 59
                    )  # Generate a random timestamp within the day
                    rand_second = random.randint(
                        0, 59
                    )  # Generate a random timestamp within the day
                    timestamp = current_date + timedelta(
                        hours=rand_hour, minutes=rand_minute, seconds=rand_second
                    )  # Generate a random timestamp within the day

                    if user_id == "User_04":  # Fixed activity for User_44
                        activity = "Driving"  # Fixed activity for User_44
                    elif user_id == "User_35":  # Fixed activity for User_55
                        activity = "Walking"  # Fixed activity for User_55
                    else:  # Random activity for other users
                        activity = random.choice(activities)  # Random user activity

                    internet_type = random.choice(
                        internet_types
                    )  # Random internet connection type
                    if (
                        internet_type == "WiFi"
                    ):  # Assign signal strength based on connection type
                        internet_strength = random.randint(
                            -90, -50
                        )  # Random WiFi signal strength
                    else:  # For mobile connections
                        internet_strength = random.randint(
                            -120, -70
                        )  # Random mobile signal strength

                    university_area = random.choice(
                        university_areas
                    )  # Random university area
                    base_latitude = original_locations[university_area][
                        "latitude"
                    ]  # Base location
                    base_longtitude = original_locations[university_area][
                        "longitude"
                    ]  # Base location
                    latitude, longtitude = get_random_location_in_radius(
                        {"latitude": base_latitude, "longitude": base_longtitude}, 200
                    )  # Random location within 200 meters of the university area

                    battery_value = random.randint(0, 100)  # Random battery percentage
                    accelerometer_range = random.randint(
                        -16, 16
                    )  # Random accelerometer range
                    accelerometer_resolution = random.uniform(
                        0.001, 0.01
                    )  # Random accelerometer resolution

                    road_accident = random.choices(
                        accident_values, accident_probabilities, k=1
                    )[
                        0
                    ]  # Random road accident severity
                    road_type = random.choices(road_types, road_probabilities, k=1)[
                        0
                    ]  # Random road type
                    speed_limit = random.choices(
                        speed_limits, speed_probabilities, k=1
                    )[
                        0
                    ]  # Random speed limit
                    violation = random.choice(violation_types)  # Random violation type

                    combined_data.append(
                        (
                            user_id,  # User_Id
                            timestamp,  # Timestamp
                            reputation[user_id],  # User_Reputation
                            round(experience[user_id], 3),  # User_Experience
                            activity,  # User_Activity
                            university_area,  # University_Area
                            violation,  # Violation_Type
                            road_type,  # Road_Type
                            road_accident,  # Road_Accident
                            speed_limit,  # Speed_Limit
                            internet_type,  # Internet_Connection
                            internet_strength,  # Internet_Signal_Strength
                            battery_value,  # Battery_Percentage
                            accelerometer_range,  # Accelerometer_Range
                            accelerometer_resolution,  # Accelerometer_Resolution
                            latitude,  # Latitude
                            longtitude,  # Longitude
                        )
                    )

                current_date += timedelta(days=1)  # Move to the next day

        return combined_data  # Return the generated dataset

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(
            f"\nData Generation Failed (unexpected): {str(e)}"
        )  # Log the error
        raise  # Raise the exception


# =========== Apply Experience ===========
@system.timer_decorator
def apply_experience(combined_data, experience) -> Any:
    """
    Function which applies incremental experience updates to user submissions.

    Features
    ---------------------------------
        - Sorts the dataset chronologically based on timestamps.
        - Iterates through the sorted dataset to update experience values.
        - Increments experience values slightly if below 1.0.
        - Caps experience values at 1.0 to prevent exceeding the maximum.
        - Constructs new tuples with updated experience values.

    Parameters
    ---------------------------------
        - combined_data : list
            The dataset after experience updates, where each tuple represents a user submission.
        - experience : dict
            A dictionary mapping user IDs to their accumulated experience levels.

    Returns
    ---------------------------------
        - combined_data : list
            The updated dataset with modified experience values.
    """

    try:  # Try-except block to catch and log any errors that occur during experience application
        combined_data.sort(key=lambda x: x[1])  # Sort the dataset by timestamps

        for idx, row in enumerate(combined_data):  # Iterate through the sorted dataset
            user_id = row[0]  # Extract the user ID from the tuple
            current_exp = experience[user_id]  # Get the current experience for the user
            increment = 0.001  # Increment for each new submission
            if (
                current_exp < 1.0
            ):  #    Check if the current experience is below the maximum
                new_exp = min(
                    current_exp + increment, 1.0
                )  # Increment experience, capping at 1.0
            else:  # If the experience is already at the maximum, reset it to 1.0
                new_exp = 1.0  # Reset experience to 1.0 if it exceeds the maximum
            experience[user_id] = round(
                new_exp, 4
            )  # Update the experience dictionary with the new value
            updated_tuple = (
                row[0],  # User ID
                row[1],  # Timestamp
                row[2],  # User's reputation
                round(new_exp, 4),  # Updated experience value (rounded)
                *row[4:],  # The rest of the fields
            )
            combined_data[idx] = (
                updated_tuple  # Replace the old tuple with the updated one
            )

        return combined_data  # Return the updated dataset

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(
            f"\nExperience Application Failed (unexpected): {str(e)}"
        )  # Log the error
        raise  # Raise the exception


# =========== Add Intentional and Unintentional Outliers ===========
@system.timer_decorator
def add_intentional_and_unintentional_outliers(
    combined_data,
    intentional_percent,
    unintentional_percent,
    overlap_percent,
):
    """
    Function which injects intentional and unintentional outliers into the dataset, with a controlled overlap.

    Parameters
    ---------------------------------
        - combined_data : list
            The dataset after experience updates, where each tuple represents a user submission.
        - intentional_percent : float
            Desired fraction of intentional outliers (e.g. 0.15 for 15%).
        - unintentional_percent : float
            Desired fraction of unintentional outliers (e.g. 0.25 for 25%).
        - overlap_percent : float
            Fraction of total rows to be both intentional and unintentional (e.g. 0.05 for 5%).

    Returns
    ---------------------------------
        - combined_data : list
            The updated dataset with injected outliers.

    Raises Exception
    ---------------------------------
        - If an unexpected error occurs during outlier injection.
    """
    try:  # Try-except block to catch and log any errors that occur during outlier injection
        field_indices = {
            "User_Activity": 4,
            "University_Area": 5,
            "Road_Accident": 6,
            "Road_Type": 7,
            "Speed_Limit": 8,
            "Violation_Type": 9,
            "Battery_Percentage": 10,
            "Internet_Connection": 11,
            "Internet_Signal_Strength": 12,
            "Accelerometer_Range": 13,
            "Accelerometer_Resolution": 14,
            "Latitude": 15,
            "Longitude": 16,
        }  # Mapping of field names to their indices in the tuple
        intentional_fields = [
            "User_Activity",
            "University_Area",
            "Road_Accident",
            "Road_Type",
            "Speed_Limit",
            "Violation_Type",
        ]  # Fields that can have intentional outliers
        outlier_fields = {
            "User_Activity": lambda x: random.choice(["Flying", "Swimming"]),
            "University_Area": lambda x: random.choice(["Center", "Spinari"]),
            "Road_Accident": lambda x: random.choice([6, 7, 8, 9]),
            "Road_Type": lambda x: random.choice(["5way", "6way"]),
            "Speed_Limit": lambda x: random.choice([140, 150, 160]),
            "Violation_Type": lambda x: random.choice(
                ["Lane_Discards", "Harassing_Driver", "Illegal_Parking"]
            ),
            "Battery_Percentage": lambda x: random.randint(101, 120),
            "Internet_Connection": lambda x: random.choice(["Ethernet", "Satellite"]),
            "Internet_Signal_Strength": lambda x: random.choice(
                [random.randint(-160, -121), random.randint(-49, -10)]
            ),
            "Accelerometer_Range": lambda x: random.choice(
                [random.randint(-25, -17), random.randint(17, 25)]
            ),
            "Accelerometer_Resolution": lambda x: round(random.uniform(0.02, 0.05), 3),
            "Latitude": lambda x: random.choice(
                [
                    round(random.uniform(37.0, 38.9), 6),
                    round(random.uniform(41.1, 43.0), 6),
                ]
            ),
            "Longitude": lambda x: random.choice(
                [
                    round(random.uniform(17.0, 19.9), 6),
                    round(random.uniform(22.1, 24.0), 6),
                ]
            ),
        }  # Functions to generate outlier values for each field

        N = len(combined_data)  # Total number of entries in the dataset
        n_both = int(
            N * overlap_percent
        )  # Number of entries that will have both intentional and unintentional outliers
        n_intentional_only = (
            int(N * intentional_percent) - n_both
        )  # Number of entries that will have only intentional outliers
        n_unintentional_only = (
            int(N * unintentional_percent) - n_both
        )  # Number of entries that will have only unintentional outliers

        all_indices = list(range(N))  # Create a list of all indices in the dataset
        both_indices = set(
            random.sample(all_indices, n_both)
        )  # Randomly select indices for entries that will have both intentional and unintentional outliers

        remaining_indices = list(
            set(all_indices) - both_indices
        )  # Remaining indices after selecting both indices
        intentional_only_indices = set(
            random.sample(remaining_indices, n_intentional_only)
        )  # Randomly select indices for entries that will have only intentional outliers
        remaining_indices2 = list(
            set(remaining_indices) - intentional_only_indices
        )  # Remaining indices after selecting intentional only indices
        unintentional_only_indices = set(
            random.sample(remaining_indices2, n_unintentional_only)
        )  # Randomly select indices for entries that will have only unintentional outliers

        for idx in range(N):  # Iterate through all indices in the dataset
            row = list(
                combined_data[idx]
            )  # Convert the tuple at the current index to a list for modification
            if (
                idx in both_indices
            ):  # If the index is in both_indices, apply both intentional and unintentional outliers
                intentional_field = random.choice(
                    intentional_fields
                )  # Randomly select an intentional field
                row[field_indices[intentional_field]] = outlier_fields[
                    intentional_field
                ](
                    row[field_indices[intentional_field]]
                )  # Apply the intentional outlier function to the selected field
                non_intentional_fields = list(
                    set(outlier_fields.keys()) - set(intentional_fields)
                )  # Get the remaining fields that can have unintentional outliers
                unintentional_field = random.choice(
                    non_intentional_fields
                )  # Randomly select an unintentional field
                row[field_indices[unintentional_field]] = outlier_fields[
                    unintentional_field
                ](
                    row[field_indices[unintentional_field]]
                )  # Apply the unintentional outlier function to the selected field

            elif (
                idx in intentional_only_indices
            ):  # If the index is in intentional_only_indices, apply only intentional outliers
                intentional_field = random.choice(
                    intentional_fields
                )  # Randomly select an intentional field
                row[field_indices[intentional_field]] = outlier_fields[
                    intentional_field
                ](
                    row[field_indices[intentional_field]]
                )  # Apply the intentional outlier function to the selected field

            elif (
                idx in unintentional_only_indices
            ):  # If the index is in unintentional_only_indices, apply only unintentional outliers
                non_intentional_fields = list(
                    set(outlier_fields.keys()) - set(intentional_fields)
                )  # Get the remaining fields that can have unintentional outliers
                unintentional_field = random.choice(
                    non_intentional_fields
                )  # Randomly select an unintentional field
                row[field_indices[unintentional_field]] = outlier_fields[
                    unintentional_field
                ](
                    row[field_indices[unintentional_field]]
                )  # Apply the unintentional outlier function to the selected field
            combined_data[idx] = tuple(row)  # Convert the modified list back to a tuple

        return combined_data  # Return the updated dataset with injected outliers

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(
            f"\nOutlier Injection Failed (unexpected): {str(e)}"
        )  # Log the error
        raise  # Raise the exception to propagate it upwards for further handling or logging


# =========== Inject Missing Values ===========
@system.timer_decorator
def inject_missing_values(data, missing_rate, columns=None):
    """
    Function which injects missing values into the dataset.

    Parameters
    ---------------------------------
        - data : list
            The dataset where missing values will be injected.
        - missing_rate : float
            The rate at which missing values will be injected (e.g., 0.1 for 10%).
        - columns : list, optional
            Specific columns to inject missing values into. If None, all columns except the first two are used.

    Returns
    ---------------------------------
        - data_copy : list
            The dataset with injected missing values, represented as tuples.

    Raises Exception
    ---------------------------------
        - If an unexpected error occurs during missing value injection.
    """
    try:  # Try-except block to catch and log any errors that occur during missing value injection
        random.seed(42)  # Set a seed for reproducibility in random operations

        if (
            not columns
        ):  # If no specific columns are provided, use all columns except the first two
            columns = list(
                range(2, len(data[0]))
            )  # Exclude the first two columns (User_Id and Timestamp)

        n_rows = len(data)  # Get the number of rows in the dataset
        n_cols = len(columns)  # Get the number of columns to inject missing values into
        n_missing = int(
            n_rows * n_cols * missing_rate
        )  # Calculate the total number of missing values to inject

        data_copy = [
            list(row) for row in data
        ]  # Create a copy of the dataset to avoid modifying the original data
        for _ in range(
            n_missing
        ):  # Loop to inject the specified number of missing values
            i = random.randint(0, n_rows - 1)  # Randomly select a row index
            j = random.choice(
                columns
            )  # Randomly select a column index from the specified columns
            data_copy[i][
                j
            ] = "None"  # Inject a missing value (represented as "None") into the selected cell

        return [
            tuple(row) for row in data_copy
        ]  # Convert the modified rows back to tuples and return the updated dataset

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(f"\nMissing Value Injection Failed: {str(e)}")  # Log the error
        raise  # Raise the exception


# =========== Get Random Location in Radius ===========
@system.timer_decorator
def get_random_location_in_radius(
    original_location, max_distance_meters=200
) -> tuple[float | Any, float | Any]:
    """
    Function which generates a random geographic location within a specified radius.

    Features
    ---------------------------------
        - Computes a new location based on a random distance and bearing.

    Parameters
    ---------------------------------
        - original_location : dict
            The original location from which the new location will be generated.
        - max_distance_meters : int
            The maximum distance in meters from the original location.

    Returns
    ---------------------------------
        - latitude : float
            The latitude of the generated location.
        - longitude : float
            The longitude of the generated location.

    Raises Exception
    ---------------------------------
        - If an unexpected error occurs during location generation.
    """

    try:  # Try-except block to catch and log any errors that occur during location generation
        base_point = Point(
            original_location["latitude"], original_location["longitude"]
        )  # Create a base point from the original location

        random_dist = random.uniform(
            0, max_distance_meters
        )  # Generate a random distance

        random_bearing = random.uniform(0, 360)  # Generate a random bearing

        new_point = geopy_distance(meters=random_dist).destination(
            base_point, random_bearing
        )  # Compute the new location based on the distance and bearing

        return (
            new_point.latitude,
            new_point.longitude,
        )  # Return the latitude and longitude of the new location

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(f"\nRandom Location Generation Failed: {str(e)}")  # Log the error
        raise  # Raise the exception


# =========== Convert to DataFrame ===========
@system.timer_decorator
def convert_to_dataframe(data_as_list) -> DataFrame:
    """
    Function which converts a list of tuples into a structured Pandas DataFrame.

    Features
    ---------------------------------
        - Defines column names for the DataFrame.
        - Converts the list of tuples into a DataFrame.

    Parameters
    ---------------------------------
        - data_as_list : list
            A list of tuples representing user submissions.

    Returns
    ---------------------------------
        - pd : DataFrame
            A structured DataFrame containing the user submissions.

    Raises Exception
    ---------------------------------
        - If an unexpected error occurs during DataFrame conversion
    """

    try:  # Try-except block to catch and log any errors that occur during DataFrame conversion
        columns = [
            "User_Id",
            "Timestamp",
            "User_Reputation",
            "User_Experience",
            "User_Activity",
            "University_Area",
            "Violation_Type",
            "Road_Type",
            "Road_Accident",
            "Speed_Limit",
            "Internet_Connection",
            "Internet_Signal_Strength",
            "Battery_Percentage",
            "Accelerometer_Range",
            "Accelerometer_Resolution",
            "Latitude",
            "Longitude",
        ]  # Define the column names for the DataFrame

        return pd.DataFrame(
            data_as_list, columns=columns
        )  # Convert the list to a DataFrame

    except Exception as e:  # Catch and log any unexpected errors
        logger.error(f"\nDataFrame Conversion Failed: {str(e)}")  # Log the error
        raise  # Raise the exception


# =========== Main  ===========
def main() -> None:
    """
    Function which orchestrates the creation of a synthetic user activity dataset.

    Features
    ---------------------------------
        - Initializes a Dask client for parallel processing.
        - Generates user activity data based on predefined parameters.
        - Applies incremental experience updates to user submissions.
        - Injects outliers into the dataset to simulate anomalous behavior.
        - Converts the processed data into a structured Pandas DataFrame.
        - Exports the DataFrame to a CSV file for further analysis.

    Raises Exception
    ---------------------------------
        - If any step in the process fails.
    """

    try:  # Try-except block to catch and log any errors that occur during dataset creation
        signal.signal(
            signal.SIGTERM, system.signal_handler
        )  # Catch SIGTERM for graceful shutdown
        signal.signal(
            signal.SIGINT, system.signal_handler
        )  # Catch SIGINT for graceful shutdown

        script_start = time.time()  # Start the script execution timer
        logger.debug(
            "\n--- Starting dataset creation timer ---"
        )  # Log the start of the script

        users = 100  # Number of users participating in the dataset
        submissions = 2  # Number of daily submissions per user

        user_ids = [f"User_{i+1:02d}" for i in range(users)]  # Generate user IDs

        start_date = datetime(2025, 1, 1)  # Beginning date (inclusive)
        end_date = datetime(2025, 12, 31)  # End date (inclusive)

        reputation = {
            uid: round(random.uniform(0.25, 0.75), 3) for uid in user_ids
        }  # Generate user reputations
        experience = {
            uid: round(random.uniform(0.001, 0.4), 4) for uid in user_ids
        }  # Generate user experience levels

        activities = ["Walking", "Driving", "Running"]  # Possible user activities

        university_areas = [
            "Zep",
            "Koila",
            "Ligeris",
        ]  # University areas for user locations
        original_locations = {
            "Zep": {"latitude": 40.279555, "longitude": 21.753262},
            "Koila": {"latitude": 40.322088, "longitude": 21.791262},
            "Ligeris": {"latitude": 40.306227, "longitude": 21.807493},
        }  # Geographic coordinates for university areas

        violation_types = [
            "Speeding",
            "Driving_Under_Influence",
            "Reckless_Driving",
        ]  # Possible violation types

        road_types = ["1way", "2way", "4way"]  # Possible road types
        road_probabilities = [0.3, 0.5, 0.2]  # Probabilities for road types
        road_accident_values = [
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Possible road accident severity levels
        road_accident_probabilities = [
            0.5,
            0.3,
            0.1,
            0.05,
            0.04,
            0.01,
        ]  # Probabilities for accident severities
        speed_limits = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
            130,
        ]  # Possible speed limits
        speed_limits_probabilities = [
            0.1,
            0.12,
            0.15,
            0.15,
            0.12,
            0.12,
            0.1,
            0.1,
            0.05,
            0.005,
            0.005,
            0.004,
            0.003,
        ]  # Probabilities for speed limits

        internet_types = [
            "WiFi",
            "3G",
            "4G",
            "5G",
        ]  # Possible internet connection types

        combined_data_task = apply_experience(
            generate_combined_data(
                user_ids,
                start_date,
                end_date,
                submissions,
                activities,
                internet_types,
                university_areas,
                original_locations,
                reputation,
                experience,
                road_accident_values,
                road_accident_probabilities,
                road_types,
                road_probabilities,
                speed_limits,
                speed_limits_probabilities,
                violation_types,
            ),
            experience,
        )  # Generate and apply experience updates to the dataset
        outliers_task = add_intentional_and_unintentional_outliers(
            combined_data_task,
            config.INTENTIONAL_PERC,
            config.UNINTENTIONAL_PERC,
            config.OVERLAP_PERC,
        )  # Inject outliers into the dataset
        missing_data_task = inject_missing_values(
            outliers_task, config.MISSING_PERC
        )  # Inject missing values into the dataset
        final_df_task = convert_to_dataframe(
            missing_data_task
        )  # Convert the processed data with missing values to a DataFrame

        utils.export_csv(
            final_df_task, folder_name=config.DATA_FOLDER, file_name="Dataset.csv"
        )  # Export the DataFrame to a CSV file

    except (
        Exception
    ) as e:  # Catch and log any errors that occur during dataset creation
        logger.error(f"\nMain Execution Failed: {str(e)}")  # Log the error
        exit(1)  # Exit the script with an error code

    finally:  # Finally block to ensure proper cleanup and logging
        try:  # Try-except block to catch and log any errors during cleanup
            system.cleanup_resources()  # Cleanup temporary resources

            script_end = time.time()  # End the script execution timer
            duration = script_end - script_start  # Calculate the total script execution
            logger.info(
                f"\nTotal script execution time: {Fore.LIGHTYELLOW_EX}{duration:.2f}s"
            )  # Log the total script execution time

        except Exception as e:  # Catch and log any errors during cleanup
            logger.error(f"\nCleanup Failed: {e}")  # Log the error
            raise  # Raise the exception


# =========== Script Entry Point ===========
if __name__ == "__main__":  # Check if the script is being run directly
    main()  # Execute the main function
